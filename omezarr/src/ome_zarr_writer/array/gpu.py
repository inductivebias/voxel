# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false
"""GPU-accelerated Zarr v3 sharded array writer using Numba CUDA + NVIDIA nvCOMP.

Writes 100% standard Zarr v3 ``sharding_indexed`` shard files readable by ``TSArrayReader``,
``zarr-python``, ``zarrs``, and Neuroglancer:

1. Gathers inner chunks ``(Cz, Cy, Cx)`` from the 3D slab in GPU VRAM (with automatic zero-padding
   at partial array boundaries and fused per-block byte-shuffling for Blosc codecs).
2. Compresses all inner chunks / sub-blocks in parallel on the GPU using ``nvidia.nvcomp``
   (supporting ``blosc.lz4``, ``lz4``, ``blosc.zstd``, ``zstd``, ``gzip``, and ``none``).
3. Packs the compressed chunks (and Blosc headers/block-tables when applicable) into a single
   contiguous GPU buffer and transfers only the compressed shard bytes to a pinned host buffer
   in one DMA transfer.
4. Appends the Zarr v3 ``(offset, nbytes)`` shard index and ``crc32c`` trailer and writes the
   shard file to disk on a pipelined I/O thread while the GPU processes the next shard.
"""

import logging
import math
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google_crc32c
import numpy as np
from cloudpathlib import S3Path
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from ome_zarr_writer.dataset import (
    BloscCodec,
    GzipCodec,
    ShardingIndexedCodec,
    Zarr3ArrayMeta,
    ZstdCodec,
)
from ome_zarr_writer.storage import S3Store

from .base import ArrayWriter
from .ts import TSArrayWriter

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

log = logging.getLogger(__name__)

_TARGET_BLOSC_BLOCKSIZE = 262144  # 256 KiB sub-blocks saturate GPU SMs and fit CPU L2 on decode
_MAX_ENCODE_BATCH_BYTES = 256 * 1024 * 1024  # 256 MiB mini-batch cap so d_sub_blocks + d_comp_out <= 520 MiB
_EMPTY_ENTRY = (1 << 64) - 1


def _has_nvcomp() -> bool:
    try:
        import nvidia.nvcomp  # noqa: F401, PLC0415

        return bool(cuda.is_available() and len(cuda.gpus) > 0)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CUDA Kernels: Inner-Chunk Gather + Byte-Shuffle, and Contiguous Shard Packing
# ---------------------------------------------------------------------------


@cuda.jit
def _gather_inner_chunks_kernel(
    d_slab_u8: Any,  # (slab_z, vy, vx, typesize) uint8
    slab_z0: int,
    shard_z0: int,
    shard_y0: int,
    shard_x0: int,
    cz: int,
    cy: int,
    cx: int,
    d_active_coords: Any,  # (n_active, 3) int32
    elems_per_block: int,
    blocks_per_chunk: int,
    shuffle_flag: int,
    d_sub_blocks: Any,  # (n_active * blocks_per_chunk, blocksize) uint8
) -> None:
    """Gather ``n_active`` inner chunks from ``d_slab_u8``, zero-pad out-of-bounds voxels, and optionally
    byte-shuffle each sub-block of ``elems_per_block`` elements."""
    v_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # type: ignore[attr-defined]
    a_idx = cuda.blockIdx.y  # type: ignore[attr-defined]
    chunk_elems = cz * cy * cx
    if a_idx >= d_active_coords.shape[0] or v_idx >= chunk_elems:
        return

    iz = d_active_coords[a_idx, 0]
    iy = d_active_coords[a_idx, 1]
    ix = d_active_coords[a_idx, 2]

    cy_cx = cy * cx
    dz = v_idx // cy_cx
    rem = v_idx - dz * cy_cx
    dy = rem // cx
    dx = rem - dy * cx

    gz = shard_z0 + iz * cz + dz - slab_z0
    gy = shard_y0 + iy * cy + dy
    gx = shard_x0 + ix * cx + dx

    slab_z, vy, vx, typesize = d_slab_u8.shape
    in_bounds = (0 <= gz < slab_z) and (0 <= gy < vy) and (0 <= gx < vx)

    b = v_idx // elems_per_block
    elem_in_b = v_idx - b * elems_per_block
    sub_idx = a_idx * blocks_per_chunk + b

    if shuffle_flag != 0 and typesize > 1:
        for k in range(typesize):
            val = d_slab_u8[gz, gy, gx, k] if in_bounds else np.uint8(0)
            d_sub_blocks[sub_idx, k * elems_per_block + elem_in_b] = val
    else:
        base = elem_in_b * typesize
        for k in range(typesize):
            val = d_slab_u8[gz, gy, gx, k] if in_bounds else np.uint8(0)
            d_sub_blocks[sub_idx, base + k] = val


@cuda.jit
def _pack_blosc_shard_kernel(  # noqa: C901
    d_comp_blocks: Any,  # (n_chunks, blocks_per_chunk, max_comp_size) uint8
    d_block_sizes: Any,  # (n_chunks, blocks_per_chunk) int32
    d_sub_blocks: Any,  # (n_chunks * blocks_per_chunk, blocksize) uint8 (for incompressible fallback)
    d_chunk_offsets: Any,  # (n_chunks,) int64
    d_chunk_ctbytes: Any,  # (n_chunks,) int64
    flag_fmt: int,
    shuffle_flag: int,
    typesize: int,
    chunk_nbytes: int,
    blocksize: int,
    d_shard_out: Any,  # (max_shard_bytes,) uint8
) -> None:
    """Pack compressed sub-blocks into Blosc 1.x frames inside a single contiguous shard buffer."""
    c = cuda.blockIdx.y  # type: ignore[attr-defined]
    b = cuda.blockIdx.x  # type: ignore[attr-defined]
    tid = cuda.threadIdx.x  # type: ignore[attr-defined]
    stride = cuda.blockDim.x  # type: ignore[attr-defined]

    n_chunks = d_comp_blocks.shape[0]
    blocks_per_chunk = d_comp_blocks.shape[1]
    if c >= n_chunks or b >= blocks_per_chunk:
        return

    chunk_out_base = d_chunk_offsets[c]
    ctbytes = d_chunk_ctbytes[c]

    # Incompressible fallback: write 16-byte BLOSC_MEMCPYED (0x02) header + unshuffled raw bytes
    if ctbytes == 16 + chunk_nbytes:
        if b == 0 and tid == 0:
            ct32 = int(ctbytes)
            d_shard_out[chunk_out_base + 0] = np.uint8(2)
            d_shard_out[chunk_out_base + 1] = np.uint8(1)
            d_shard_out[chunk_out_base + 2] = np.uint8(flag_fmt | 0x02)
            d_shard_out[chunk_out_base + 3] = np.uint8(typesize)
            for k in range(4):
                d_shard_out[chunk_out_base + 4 + k] = np.uint8((chunk_nbytes >> (8 * k)) & 0xFF)
                d_shard_out[chunk_out_base + 8 + k] = np.uint8((blocksize >> (8 * k)) & 0xFF)
                d_shard_out[chunk_out_base + 12 + k] = np.uint8((ct32 >> (8 * k)) & 0xFF)
        sub_idx = c * blocks_per_chunk + b
        raw_base = chunk_out_base + 16 + b * blocksize
        if shuffle_flag != 0 and typesize > 1:
            elems_per_block = blocksize // typesize
            for e in range(tid, elems_per_block, stride):
                for k in range(typesize):
                    d_shard_out[raw_base + e * typesize + k] = d_sub_blocks[sub_idx, k * elems_per_block + e]
        else:
            for i in range(tid, blocksize, stride):
                d_shard_out[raw_base + i] = d_sub_blocks[sub_idx, i]
        return

    bstart = 16 + 4 * blocks_per_chunk
    for prev in range(b):
        bstart += 4 + d_block_sizes[c, prev]

    csize = d_block_sizes[c, b]
    if tid == 0:
        if b == 0:
            ct32 = int(ctbytes)
            d_shard_out[chunk_out_base + 0] = np.uint8(2)
            d_shard_out[chunk_out_base + 1] = np.uint8(1)
            d_shard_out[chunk_out_base + 2] = np.uint8(flag_fmt | 0x10 | shuffle_flag)
            d_shard_out[chunk_out_base + 3] = np.uint8(typesize)
            for k in range(4):
                d_shard_out[chunk_out_base + 4 + k] = np.uint8((chunk_nbytes >> (8 * k)) & 0xFF)
                d_shard_out[chunk_out_base + 8 + k] = np.uint8((blocksize >> (8 * k)) & 0xFF)
                d_shard_out[chunk_out_base + 12 + k] = np.uint8((ct32 >> (8 * k)) & 0xFF)
        bstart_pos = chunk_out_base + 16 + 4 * b
        csize_pos = chunk_out_base + bstart
        for k in range(4):
            d_shard_out[bstart_pos + k] = np.uint8((bstart >> (8 * k)) & 0xFF)
            d_shard_out[csize_pos + k] = np.uint8((csize >> (8 * k)) & 0xFF)

    payload_pos = chunk_out_base + bstart + 4
    for i in range(tid, csize, stride):
        d_shard_out[payload_pos + i] = d_comp_blocks[c, b, i]


@cuda.jit
def _pack_raw_shard_kernel(
    d_comp_chunks: Any,  # (n_chunks, max_comp_size) uint8
    d_chunk_sizes: Any,  # (n_chunks,) int32
    d_chunk_offsets: Any,  # (n_chunks,) int64
    d_shard_out: Any,  # (max_shard_bytes,) uint8
) -> None:
    """Pack raw compressed chunks (Zstd / Gzip) contiguously into ``d_shard_out``."""
    c = cuda.blockIdx.y  # type: ignore[attr-defined]
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # type: ignore[attr-defined]
    stride = cuda.gridDim.x * cuda.blockDim.x  # type: ignore[attr-defined]

    if c >= d_comp_chunks.shape[0]:
        return
    csize = d_chunk_sizes[c]
    out_base = d_chunk_offsets[c]
    for i in range(tid, csize, stride):
        d_shard_out[out_base + i] = d_comp_chunks[c, i]


# ---------------------------------------------------------------------------
# Parsed Codec Specification & Reusable Per-Level GPU Shard Arena
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParsedShardSpec:
    volume_shape: tuple[int, int, int]
    shard_shape: tuple[int, int, int]
    chunk_shape: tuple[int, int, int]
    typesize: int
    dtype: np.dtype
    codec_kind: str  # "none" | "zstd" | "gzip" | "blosc"
    nvcomp_alg: str  # "" | "zstd" | "gzip" | "lz4"
    blosc_flag_fmt: int  # 0x20 (lz4) | 0x80 (zstd)
    shuffle_flag: int  # 0x01 (byte-shuffle) | 0x00


def _parse_array_meta(meta: Zarr3ArrayMeta) -> _ParsedShardSpec:
    if len(meta.shape) != 4:
        raise ValueError(f"GPUArrayWriter requires 4D [C, Z, Y, X] arrays, got shape {meta.shape}")
    vz, vy, vx = int(meta.shape[1]), int(meta.shape[2]), int(meta.shape[3])
    sz, sy, sx = (int(x) for x in meta.chunk_grid.configuration.chunk_shape[1:])
    dtype = np.dtype(meta.data_type.dtype)
    typesize = int(dtype.itemsize)

    sharding = next((c for c in meta.codecs if isinstance(c, ShardingIndexedCodec)), None)
    if sharding is None:
        raise ValueError("GPUArrayWriter requires a sharding_indexed array")
    if sharding.configuration.index_location != "end":
        raise ValueError(f"Unsupported shard index_location: {sharding.configuration.index_location}")
    cz, cy, cx = (int(x) for x in sharding.configuration.chunk_shape[1:])

    codec_kind = "none"
    nvcomp_alg = ""
    blosc_flag_fmt = 0
    shuffle_flag = 0

    for inner in sharding.configuration.codecs:
        if isinstance(inner, ZstdCodec):
            codec_kind = "zstd"
            nvcomp_alg = "zstd"
        elif isinstance(inner, GzipCodec):
            codec_kind = "gzip"
            nvcomp_alg = "gzip"
        elif isinstance(inner, BloscCodec):
            codec_kind = "blosc"
            cname = inner.configuration.cname
            if cname in ("lz4", "lz4hc"):
                nvcomp_alg = "lz4"
                blosc_flag_fmt = 0x20
            elif cname == "zstd":
                nvcomp_alg = "zstd"
                blosc_flag_fmt = 0x80
            else:
                raise ValueError(f"Unsupported Blosc cname for GPUArrayWriter: {cname}")
            if typesize > 1 and inner.configuration.shuffle != "noshuffle":
                shuffle_flag = 0x01

    return _ParsedShardSpec(
        volume_shape=(vz, vy, vx),
        shard_shape=(sz, sy, sx),
        chunk_shape=(cz, cy, cx),
        typesize=typesize,
        dtype=dtype,
        codec_kind=codec_kind,
        nvcomp_alg=nvcomp_alg,
        blosc_flag_fmt=blosc_flag_fmt,
        shuffle_flag=shuffle_flag,
    )


class _CudaSlice1D:
    """Lightweight 1D ``__cuda_array_interface__`` view into a row of a 2D device array.

    Avoids constructing thousands of Numba ``DeviceNDArray`` slice objects per shard when building
    ``nvidia.nvcomp.as_arrays`` batches (~30x faster than ``[d_arr[i] for i in range(n)]``).
    """

    __slots__ = ("__cuda_array_interface__", "_owner")

    def __init__(self, ptr: int, size: int, owner: Any) -> None:
        self._owner = owner
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


# Process-wide reusable pair of pinned host buffers and shared device scratch buffers
# so L0..L7 encoders in a worker share a single ~520 MiB nvcomp staging arena.
_SHARED_PINNED_PAIR: list[Any] = []
_SHARED_D_SUB_FLAT: list[Any] = []
_SHARED_D_COMP_FLAT: list[Any] = []
_SHARED_D_SHARD_FLAT: list[Any] = []


def _get_shared_pinned_pair(min_bytes: int) -> list[Any]:
    if not _SHARED_PINNED_PAIR or int(_SHARED_PINNED_PAIR[0].nbytes) < min_bytes:
        _SHARED_PINNED_PAIR.clear()
        _SHARED_PINNED_PAIR.extend(
            [
                cuda.pinned_array(min_bytes, dtype=np.uint8),
                cuda.pinned_array(min_bytes, dtype=np.uint8),
            ]
        )
    return _SHARED_PINNED_PAIR


def _get_shared_device_flat(holder: list[Any], min_bytes: int) -> Any:
    if not holder or int(holder[0].nbytes) < min_bytes:
        holder.clear()
        holder.append(cuda.device_array(min_bytes, dtype=np.uint8))
    return holder[0]


def release_shared_encoder_shard_buffer() -> None:
    """Release the large contiguous shard packing buffer while keeping the small (~520 MiB) nvcomp batch
    buffers and ``as_arrays`` caches warm across batches."""
    _SHARED_D_SHARD_FLAT.clear()


class _GpuShardEncoder:
    """Lazily-allocated GPU compression buffers for one level's shard geometry."""

    def __init__(self, spec: _ParsedShardSpec) -> None:
        from nvidia import nvcomp  # noqa: PLC0415

        self._nvcomp = nvcomp
        self.spec = spec
        sz, sy, sx = spec.shard_shape
        cz, cy, cx = spec.chunk_shape
        self.nc_z = sz // cz
        self.nc_y = sy // cy
        self.nc_x = sx // cx
        self.max_chunks = self.nc_z * self.nc_y * self.nc_x
        self.chunk_elems = cz * cy * cx
        self.chunk_nbytes = self.chunk_elems * spec.typesize

        if spec.codec_kind == "blosc" and self.max_chunks < 256:
            # Split into 256 KiB sub-blocks when a shard has few inner chunks so the GPU stays saturated.
            bs = min(self.chunk_nbytes, _TARGET_BLOSC_BLOCKSIZE)
            while bs > spec.typesize and (self.chunk_nbytes % bs != 0 or (bs // spec.typesize) == 0):
                bs //= 2
            self.blocksize = max(spec.typesize, bs)
            self.blocks_per_chunk = self.chunk_nbytes // self.blocksize
        else:
            self.blocksize = self.chunk_nbytes
            self.blocks_per_chunk = 1

        self.elems_per_block = self.blocksize // spec.typesize
        self.max_batch_chunks = max(1, min(self.max_chunks, _MAX_ENCODE_BATCH_BYTES // max(1, self.chunk_nbytes)))
        self.max_subs = self.max_batch_chunks * self.blocks_per_chunk

        self.codec: Any = None
        if spec.codec_kind != "none":
            self.codec = nvcomp.Codec(algorithm=spec.nvcomp_alg, bitstream_kind=nvcomp.BitstreamKind.RAW)

        self.max_shard_payload = self.max_chunks * (self.chunk_nbytes + 64 + 8 * self.blocks_per_chunk)
        self.index_and_crc_bytes = self.max_chunks * 16 + 4
        self._min_pinned_bytes = self.max_shard_payload + self.index_and_crc_bytes
        self.h_pinned = _get_shared_pinned_pair(self._min_pinned_bytes)
        self._slot_idx = 0

        self.d_sub_blocks: Any = None
        self.d_active_coords: Any = None
        self.d_comp_out: Any = None
        self.d_shard_out: Any = None
        self._sub_flat_ptr: int = 0
        self._comp_flat_ptr: int = 0
        self.max_comp_size = self.blocksize
        self._in_slices: list[_CudaSlice1D] = []
        self._out_slices: list[_CudaSlice1D] = []
        self._in_views_cache: dict[int, Any] = {}
        self._out_views_cache: dict[int, Any] = {}
        self._comp_cfg_cache: dict[int, Any] = {}
        self.ensure_device_buffers()

    def ensure_device_buffers(self) -> None:
        self.h_pinned = _get_shared_pinned_pair(self._min_pinned_bytes)
        sub_bytes = self.max_subs * self.blocksize
        d_sub_flat = _get_shared_device_flat(_SHARED_D_SUB_FLAT, sub_bytes)
        in_ptr = int(d_sub_flat.__cuda_array_interface__["data"][0])

        if self.d_sub_blocks is None or in_ptr != self._sub_flat_ptr:
            self._sub_flat_ptr = in_ptr
            self.d_sub_blocks = d_sub_flat[:sub_bytes].reshape(self.max_subs, self.blocksize)
            self.d_active_coords = cuda.device_array((self.max_batch_chunks, 3), dtype=np.int32)
            self._in_slices = [
                _CudaSlice1D(in_ptr + i * self.blocksize, self.blocksize, d_sub_flat) for i in range(self.max_subs)
            ]
            self._in_views_cache.clear()
            self._comp_cfg_cache.clear()

        if self.spec.codec_kind != "none" and self.codec is not None:
            self.max_comp_size = int(self.codec.get_max_comp_buffer_size(self._nvcomp.as_array(self._in_slices[0])))
            comp_bytes = self.max_subs * self.max_comp_size
            d_comp_flat = _get_shared_device_flat(_SHARED_D_COMP_FLAT, comp_bytes)
            out_ptr = int(d_comp_flat.__cuda_array_interface__["data"][0])
            if self.d_comp_out is None or out_ptr != self._comp_flat_ptr:
                self._comp_flat_ptr = out_ptr
                self.d_comp_out = d_comp_flat[:comp_bytes].reshape(self.max_subs, self.max_comp_size)
                self._out_slices = [
                    _CudaSlice1D(out_ptr + i * self.max_comp_size, self.max_comp_size, d_comp_flat)
                    for i in range(self.max_subs)
                ]
                self._out_views_cache.clear()

        d_shard_flat = _get_shared_device_flat(_SHARED_D_SHARD_FLAT, self.max_shard_payload)
        self.d_shard_out = d_shard_flat[: self.max_shard_payload]

    def release_device_buffers(self) -> None:
        self.d_shard_out = None

    def warmup(self) -> None:
        """Pre-allocate pinned/shared buffers and JIT-compile the CUDA gather/pack kernels before capture."""
        self.ensure_device_buffers()
        cz, cy, cx = self.spec.chunk_shape
        d_dummy = cuda.device_array((cz, cy, cx, self.spec.typesize), dtype=np.uint8)
        coords = np.zeros((1, 3), dtype=np.int32)
        ic = np.zeros(1, dtype=np.int64)
        self.encode_shard_to_host(d_dummy, 0, 0, 0, 0, coords, ic)
        self._slot_idx = 0
        self.release_device_buffers()

    def _get_batch_views(self, n_subs: int) -> tuple[Any, Any, Any]:
        if n_subs not in self._in_views_cache:
            nvcomp = self._nvcomp
            self._in_views_cache[n_subs] = nvcomp.as_arrays(self._in_slices[:n_subs])
            self._out_views_cache[n_subs] = nvcomp.as_arrays(self._out_slices[:n_subs])
            self._comp_cfg_cache[n_subs] = self.codec.compression_config([self.blocksize] * n_subs)
        return self._in_views_cache[n_subs], self._out_views_cache[n_subs], self._comp_cfg_cache[n_subs]

    def encode_shard_to_host(
        self,
        d_slab_u8: Any,
        slab_z0: int,
        shard_z0: int,
        shard_y0: int,
        shard_x0: int,
        active_coords: np.ndarray,
        active_ic: np.ndarray,
    ) -> memoryview:
        """Encode one shard on GPU in <=256 MiB mini-batches, transfer to the next pinned host buffer,
        append index + CRC32C, and return a memoryview of the complete Zarr v3 shard file bytes."""
        self.ensure_device_buffers()
        n_active = int(active_coords.shape[0])
        cz, cy, cx = self.spec.chunk_shape
        threads = 256
        blocks_x = math.ceil(self.chunk_elems / threads)

        host_buf = self.h_pinned[self._slot_idx]
        self._slot_idx ^= 1

        if self.spec.codec_kind == "none":
            for c0 in range(0, n_active, self.max_batch_chunks):
                nb = min(self.max_batch_chunks, n_active - c0)
                nb_subs = nb * self.blocks_per_chunk
                self.d_active_coords[:nb].copy_to_device(active_coords[c0 : c0 + nb])
                _gather_inner_chunks_kernel[(blocks_x, nb), threads](  # type: ignore[index]
                    d_slab_u8,
                    slab_z0,
                    shard_z0,
                    shard_y0,
                    shard_x0,
                    cz,
                    cy,
                    cx,
                    self.d_active_coords[:nb],
                    self.elems_per_block,
                    self.blocks_per_chunk,
                    self.spec.shuffle_flag,
                    self.d_sub_blocks[:nb_subs],
                )
                b_start = c0 * self.chunk_nbytes
                b_len = nb * self.chunk_nbytes
                self.d_sub_blocks[:nb_subs].reshape(b_len).copy_to_host(host_buf[b_start : b_start + b_len])
            cuda.synchronize()
            total_payload = n_active * self.chunk_nbytes
            chunk_lengths = np.full(n_active, self.chunk_nbytes, dtype=np.uint64)
            offsets = np.arange(n_active, dtype=np.uint64) * np.uint64(self.chunk_nbytes)
        else:
            all_offsets = np.zeros(n_active, dtype=np.int64)
            all_ctbytes = np.zeros(n_active, dtype=np.int64)
            cur_offset = 0

            for c0 in range(0, n_active, self.max_batch_chunks):
                nb = min(self.max_batch_chunks, n_active - c0)
                nb_subs = nb * self.blocks_per_chunk
                self.d_active_coords[:nb].copy_to_device(active_coords[c0 : c0 + nb])
                _gather_inner_chunks_kernel[(blocks_x, nb), threads](  # type: ignore[index]
                    d_slab_u8,
                    slab_z0,
                    shard_z0,
                    shard_y0,
                    shard_x0,
                    cz,
                    cy,
                    cx,
                    self.d_active_coords[:nb],
                    self.elems_per_block,
                    self.blocks_per_chunk,
                    self.spec.shuffle_flag,
                    self.d_sub_blocks[:nb_subs],
                )
                in_views, out_views, comp_cfg = self._get_batch_views(nb_subs)
                res = self.codec.encode(in_views, out=out_views, compression_config=comp_cfg)

                if self.spec.codec_kind == "blosc":
                    sizes = np.asarray(res.buffer_sizes, dtype=np.int32).reshape(nb, self.blocks_per_chunk)
                    comp_ctbytes = 16 + 8 * self.blocks_per_chunk + sizes.sum(axis=1, dtype=np.int64)
                    memcpy_ctbytes = 16 + self.chunk_nbytes
                    incompressible = (sizes.max(axis=1) >= self.blocksize) | (comp_ctbytes >= memcpy_ctbytes)
                    ctbytes = np.where(incompressible, memcpy_ctbytes, comp_ctbytes).astype(np.int64)
                    offsets_i64 = np.full(nb, cur_offset, dtype=np.int64)
                    if nb > 1:
                        offsets_i64[1:] += np.cumsum(ctbytes[:-1])
                    cur_offset = int(offsets_i64[-1] + ctbytes[-1])

                    d_sizes = cuda.to_device(sizes)
                    d_offsets = cuda.to_device(offsets_i64)
                    d_ctbytes = cuda.to_device(ctbytes)
                    d_comp_3d = self.d_comp_out[:nb_subs].reshape(nb, self.blocks_per_chunk, self.max_comp_size)
                    _pack_blosc_shard_kernel[(self.blocks_per_chunk, nb), 256](  # type: ignore[index]
                        d_comp_3d,
                        d_sizes,
                        self.d_sub_blocks[:nb_subs],
                        d_offsets,
                        d_ctbytes,
                        self.spec.blosc_flag_fmt,
                        self.spec.shuffle_flag,
                        self.spec.typesize,
                        self.chunk_nbytes,
                        self.blocksize,
                        self.d_shard_out,
                    )
                else:
                    sizes_1d = np.asarray(res.buffer_sizes, dtype=np.int32)
                    ctbytes = sizes_1d.astype(np.int64)
                    offsets_i64 = np.full(nb, cur_offset, dtype=np.int64)
                    if nb > 1:
                        offsets_i64[1:] += np.cumsum(ctbytes[:-1])
                    cur_offset = int(offsets_i64[-1] + ctbytes[-1])

                    d_sizes = cuda.to_device(sizes_1d)
                    d_offsets = cuda.to_device(offsets_i64)
                    pack_blocks_x = min(64, max(1, math.ceil(int(sizes_1d.max()) / 256)))
                    _pack_raw_shard_kernel[(pack_blocks_x, nb), 256](  # type: ignore[index]
                        self.d_comp_out[:nb],
                        d_sizes,
                        d_offsets,
                        self.d_shard_out,
                    )

                all_offsets[c0 : c0 + nb] = offsets_i64
                all_ctbytes[c0 : c0 + nb] = ctbytes

            total_payload = cur_offset
            self.d_shard_out[:total_payload].copy_to_host(host_buf[:total_payload])
            cuda.synchronize()
            chunk_lengths = all_ctbytes.astype(np.uint64)
            offsets = all_offsets.astype(np.uint64)

        # Append the Zarr v3 shard index ((offset, nbytes) uint64 pairs) + 4-byte CRC32C checksum
        index_table = np.full((self.max_chunks, 2), _EMPTY_ENTRY, dtype="<u8")
        index_table[active_ic, 0] = offsets
        index_table[active_ic, 1] = chunk_lengths
        index_bytes = index_table.tobytes()
        crc_bytes = struct.pack("<I", google_crc32c.value(index_bytes))
        trailer_len = len(index_bytes) + 4
        host_buf[total_payload : total_payload + len(index_bytes)] = np.frombuffer(index_bytes, dtype=np.uint8)
        host_buf[total_payload + len(index_bytes) : total_payload + trailer_len] = np.frombuffer(
            crc_bytes, dtype=np.uint8
        )
        return memoryview(host_buf)[: total_payload + trailer_len]

    def close(self) -> None:
        self.release_device_buffers()


def _write_shard_file(path: Path, data: memoryview) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(data)


def _active_shard_chunks(
    shard_z0: int,
    shard_y0: int,
    shard_x0: int,
    z_end: int,
    vy: int,
    vx: int,
    cz: int,
    cy: int,
    cx: int,
    nc_z: int,
    nc_y: int,
    nc_x: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(active_coords, active_ic)`` for inner chunks inside array bounds, or ``None`` if empty."""
    nz = min(nc_z, max(0, math.ceil((z_end - shard_z0) / cz)))
    ny = min(nc_y, max(0, math.ceil((vy - shard_y0) / cy)))
    nx = min(nc_x, max(0, math.ceil((vx - shard_x0) / cx)))
    if nz == 0 or ny == 0 or nx == 0:
        return None
    iz, iy, ix = np.meshgrid(
        np.arange(nz, dtype=np.int32),
        np.arange(ny, dtype=np.int32),
        np.arange(nx, dtype=np.int32),
        indexing="ij",
    )
    active_coords = np.stack((iz.ravel(), iy.ravel(), ix.ravel()), axis=1)
    active_ic = ((iz.ravel().astype(np.int64) * nc_y + iy.ravel().astype(np.int64)) * nc_x) + ix.ravel().astype(
        np.int64
    )
    return active_coords, active_ic


class GPUArrayWriter(ArrayWriter):
    """GPU-accelerated Zarr v3 sharded array writer.

    Uses ``nvidia.nvcomp`` + Numba CUDA to gather, shuffle, compress, and assemble Zarr v3 shards on
    the GPU for local and staged-S3 targets, falling back transparently to :class:`TSArrayWriter` for
    direct S3 targets or unaligned partial-shard updates.
    """

    def __init__(self) -> None:
        self._target: Path | S3Path | None = None
        self._spec: _ParsedShardSpec | None = None
        self._encoder: _GpuShardEncoder | None = None
        self._fallback: TSArrayWriter | None = None
        self._io_pool: ThreadPoolExecutor | None = None

    def open(self, target: Path | S3Path, store: S3Store | None = None) -> None:
        self.close()
        self._target = target
        if isinstance(target, S3Path) or not _has_nvcomp():
            self._fallback = TSArrayWriter()
            self._fallback.open(target, store)
            return
        try:
            meta_path = target.expanduser().resolve() / "zarr.json"
            meta = Zarr3ArrayMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
            self._spec = _parse_array_meta(meta)
            self._target = target.expanduser().resolve()
            self._io_pool = ThreadPoolExecutor(max_workers=1)
        except Exception:
            log.warning("GPUArrayWriter falling back to TSArrayWriter for %s", target, exc_info=True)
            self._spec = None
            self._fallback = TSArrayWriter()
            self._fallback.open(target, store)

    def _ensure_fallback(self) -> TSArrayWriter:
        if self._fallback is None:
            if self._target is None:
                raise RuntimeError("write_slice called before open")
            self._fallback = TSArrayWriter()
            self._fallback.open(self._target, None)
        return self._fallback

    def write_slice(self, c: int, z_offset: int, arr: np.ndarray) -> int:
        if self._target is None:
            raise RuntimeError("write_slice called before open")
        if self._fallback is not None or self._spec is None or not isinstance(self._target, Path):
            return self._ensure_fallback().write_slice(c, z_offset, arr)

        contig = np.ascontiguousarray(arr)
        d_slab = cuda.to_device(contig)
        return self.write_slice_gpu(c, z_offset, d_slab, host_fallback=contig)

    def write_slice_gpu(  # noqa: C901
        self,
        c: int,
        z_offset: int,
        d_slab: Any,
        *,
        host_fallback: np.ndarray | None = None,
    ) -> int:
        """Write a GPU-resident 3D slab ``d_slab`` (shape ``(slab_z, vy, vx)``) at channel ``c`` and ``z_offset``."""
        if self._target is None:
            raise RuntimeError("write_slice_gpu called before open")
        if self._fallback is not None or self._spec is None or not isinstance(self._target, Path):
            host_arr = host_fallback if host_fallback is not None else d_slab.copy_to_host()
            return self._ensure_fallback().write_slice(c, z_offset, host_arr)

        spec = self._spec
        slab_z, vy, vx = (int(x) for x in d_slab.shape)
        nbytes = slab_z * vy * vx * spec.typesize
        if slab_z == 0 or vy == 0 or vx == 0:
            return 0

        sz, sy, sx = spec.shard_shape
        vz_total = spec.volume_shape[0]
        z_end = z_offset + slab_z
        # Verify the slab covers complete Z-shards (either aligned to Sz or reaching the end of the array)
        if z_offset % sz != 0 or (z_end % sz != 0 and z_end != vz_total):
            host_arr = host_fallback if host_fallback is not None else d_slab.copy_to_host()
            return self._ensure_fallback().write_slice(c, z_offset, host_arr)

        if self._encoder is None:
            self._encoder = _GpuShardEncoder(spec)
        encoder = self._encoder

        # View the 3D device array (slab_z, vy, vx) as 4D uint8 (slab_z, vy, vx, typesize)
        d_slab_u8 = d_slab.view(np.uint8).reshape(slab_z, vy, vx, spec.typesize)
        cz, cy, cx = spec.chunk_shape
        nc_z, nc_y, nc_x = encoder.nc_z, encoder.nc_y, encoder.nc_x
        n_shards_y = math.ceil(vy / sy)
        n_shards_x = math.ceil(vx / sx)

        pending_write: Future[None] | None = None
        try:
            for shard_z0 in range(z_offset, z_end, sz):
                sz_idx = shard_z0 // sz
                for sy_idx in range(n_shards_y):
                    shard_y0 = sy_idx * sy
                    for sx_idx in range(n_shards_x):
                        shard_x0 = sx_idx * sx
                        active = _active_shard_chunks(
                            shard_z0, shard_y0, shard_x0, z_end, vy, vx, cz, cy, cx, nc_z, nc_y, nc_x
                        )
                        if active is None:
                            continue
                        active_coords, active_ic = active
                        shard_view = encoder.encode_shard_to_host(
                            d_slab_u8,
                            z_offset,
                            shard_z0,
                            shard_y0,
                            shard_x0,
                            active_coords,
                            active_ic,
                        )
                        shard_path = self._target / "c" / str(c) / str(sz_idx) / str(sy_idx) / str(sx_idx)
                        if pending_write is not None:
                            pending_write.result()
                            pending_write = None
                        if self._io_pool is not None:
                            pending_write = self._io_pool.submit(_write_shard_file, shard_path, shard_view)
                        else:
                            _write_shard_file(shard_path, shard_view)
            if pending_write is not None:
                pending_write.result()
        except Exception:
            if pending_write is not None:
                pending_write.cancel()
            raise

        return nbytes

    def release_device_buffers(self) -> None:
        """Release transient GPU device buffers while keeping the writer and pinned staging buffers open."""
        if self._encoder is not None:
            self._encoder.release_device_buffers()

    def warmup(self) -> None:
        """Pre-allocate pinned/shared buffers and JIT-compile the CUDA gather/pack kernels before capture."""
        if self._fallback is not None or self._spec is None:
            return
        if self._encoder is None:
            self._encoder = _GpuShardEncoder(self._spec)
        self._encoder.warmup()

    def close(self) -> None:
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=True)
            self._io_pool = None
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
        self._spec = None
        self._target = None
