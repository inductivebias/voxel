# pyright: reportArgumentType=false, reportOperatorIssue=false, reportAttributeAccessIssue=false
"""GPU-accelerated Zarr v3 sharded array writer using Numba CUDA + NVIDIA nvCOMP."""

import logging
import math
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cloudpathlib import S3Path
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

from ome_zarr_writer.dataset import (
    BloscCodec,
    BytesCodec,
    CRC32CCodec,
    ShardingIndexedCodec,
    Zarr3ArrayMeta,
    ZstdCodec,
)
from ome_zarr_writer.pyramid_gpu import flush_gpu_deallocations
from ome_zarr_writer.storage import S3Store

from .base import ArrayWriter
from .ts import TSArrayWriter

log = logging.getLogger(__name__)

_TARGET_BLOSC_BLOCKSIZE = 262144
_MAX_ENCODE_BATCH_BYTES = 256 * 1024 * 1024
_MAX_ENCODE_BATCH_CHUNKS = 32768
_EMPTY_ENTRY = (1 << 64) - 1


def max_batch_chunks_for(chunk_nbytes: int, max_chunks: int) -> int:
    """Return the mini-batch chunk count capped by VRAM budget and CUDA's gridDim.y limit."""
    return max(1, min(max_chunks, _MAX_ENCODE_BATCH_CHUNKS, _MAX_ENCODE_BATCH_BYTES // max(1, chunk_nbytes)))


def _crc32c(data: bytes) -> int:
    try:
        import google_crc32c  # noqa: PLC0415

        return int(google_crc32c.value(data))
    except ImportError:
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ 0x82F63B78 if (crc & 1) else (crc >> 1)
        return crc ^ 0xFFFFFFFF


def _has_nvcomp() -> bool:
    try:
        import nvidia.nvcomp  # noqa: F401, PLC0415

        return bool(cuda.is_available() and len(cuda.gpus) > 0)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CUDA Kernels: Inner-Chunk Gather + Shuffle & Contiguous Shard Packing
# ---------------------------------------------------------------------------


@cuda.jit(device=True)
def _read_slab_byte(
    d_slab_u8: Any,
    slab_z0: int,
    shard_z0: int,
    shard_y0: int,
    shard_x0: int,
    iz: int,
    iy: int,
    ix: int,
    cz: int,
    cy: int,
    cx: int,
    v_idx: int,
    k: int,
) -> np.uint8:
    cy_cx = cy * cx
    dz = v_idx // cy_cx
    rem = v_idx - dz * cy_cx
    dy = rem // cx
    dx = rem - dy * cx
    gz = shard_z0 + iz * cz + dz - slab_z0
    gy = shard_y0 + iy * cy + dy
    gx = shard_x0 + ix * cx + dx
    slab_z, vy, vx, _ = d_slab_u8.shape
    if (0 <= gz < slab_z) and (0 <= gy < vy) and (0 <= gx < vx):
        return d_slab_u8[gz, gy, gx, k]
    return np.uint8(0)


@cuda.jit
def _find_nonempty_chunks_kernel(
    d_slab_u8: Any,
    slab_z0: int,
    shard_z0: int,
    shard_y0: int,
    shard_x0: int,
    cz: int,
    cy: int,
    cx: int,
    active_coords: Any,
    out_nonempty: Any,
) -> None:
    c = cuda.blockIdx.x  # type: ignore[attr-defined]
    if c >= active_coords.shape[0]:
        return
    iz, iy, ix = active_coords[c, 0], active_coords[c, 1], active_coords[c, 2]
    typesize = d_slab_u8.shape[3]
    chunk_nbytes = cz * cy * cx * typesize
    tid = cuda.threadIdx.x  # type: ignore[attr-defined]
    stride = cuda.blockDim.x  # type: ignore[attr-defined]
    for idx in range(tid, chunk_nbytes, stride):
        v_idx = idx // typesize
        k = idx - v_idx * typesize
        if _read_slab_byte(d_slab_u8, slab_z0, shard_z0, shard_y0, shard_x0, iz, iy, ix, cz, cy, cx, v_idx, k) != 0:
            out_nonempty[c] = np.uint8(1)
            return


@cuda.jit
def _gather_inner_chunks_kernel(
    d_slab_u8: Any,
    slab_z0: int,
    shard_z0: int,
    shard_y0: int,
    shard_x0: int,
    cz: int,
    cy: int,
    cx: int,
    d_active_coords: Any,
    elems_per_block: int,
    blocks_per_chunk: int,
    shuffle_flag: int,
    d_sub_blocks: Any,
) -> None:
    v_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # type: ignore[attr-defined]
    a_idx = cuda.blockIdx.y  # type: ignore[attr-defined]
    if a_idx >= d_active_coords.shape[0] or v_idx >= cz * cy * cx:
        return
    iz, iy, ix = d_active_coords[a_idx, 0], d_active_coords[a_idx, 1], d_active_coords[a_idx, 2]
    typesize = d_slab_u8.shape[3]
    b = v_idx // elems_per_block
    elem_in_b = v_idx - b * elems_per_block
    sub_idx = a_idx * blocks_per_chunk + b

    for k in range(typesize):
        val = _read_slab_byte(d_slab_u8, slab_z0, shard_z0, shard_y0, shard_x0, iz, iy, ix, cz, cy, cx, v_idx, k)
        dst = k * elems_per_block + elem_in_b if (shuffle_flag == 0x01 and typesize > 1) else elem_in_b * typesize + k
        d_sub_blocks[sub_idx, dst] = val


@cuda.jit(device=True)
def _write_blosc_header(
    d_shard_out: Any, base: int, flags: int, typesize: int, chunk_nbytes: int, blocksize: int, ct32: int
) -> None:
    d_shard_out[base + 0] = np.uint8(2)
    d_shard_out[base + 1] = np.uint8(1)
    d_shard_out[base + 2] = np.uint8(flags)
    d_shard_out[base + 3] = np.uint8(typesize)
    for k in range(4):
        d_shard_out[base + 4 + k] = np.uint8((chunk_nbytes >> (8 * k)) & 0xFF)
        d_shard_out[base + 8 + k] = np.uint8((blocksize >> (8 * k)) & 0xFF)
        d_shard_out[base + 12 + k] = np.uint8((ct32 >> (8 * k)) & 0xFF)


@cuda.jit(device=True)
def _unshuffle_sub_block(
    d_sub_blocks: Any,
    sub_idx: int,
    d_shard_out: Any,
    raw_base: int,
    blocksize: int,
    shuffle_flag: int,
    typesize: int,
    tid: int,
    stride: int,
) -> None:
    if shuffle_flag != 0 and typesize > 1:
        elems_per_block = blocksize // typesize
        for e in range(tid, elems_per_block, stride):
            for k in range(typesize):
                d_shard_out[raw_base + e * typesize + k] = d_sub_blocks[sub_idx, k * elems_per_block + e]
    else:
        for i in range(tid, blocksize, stride):
            d_shard_out[raw_base + i] = d_sub_blocks[sub_idx, i]


@cuda.jit
def _pack_blosc_shard_kernel(
    d_comp_blocks: Any,
    d_block_sizes: Any,
    d_sub_blocks: Any,
    d_chunk_offsets: Any,
    d_chunk_ctbytes: Any,
    flag_fmt: int,
    shuffle_flag: int,
    typesize: int,
    chunk_nbytes: int,
    blocksize: int,
    d_shard_out: Any,
) -> None:
    c = cuda.blockIdx.y  # type: ignore[attr-defined]
    b = cuda.blockIdx.x  # type: ignore[attr-defined]
    tid = cuda.threadIdx.x  # type: ignore[attr-defined]
    stride = cuda.blockDim.x  # type: ignore[attr-defined]
    n_chunks, blocks_per_chunk = d_comp_blocks.shape[0], d_comp_blocks.shape[1]
    if c >= n_chunks or b >= blocks_per_chunk:
        return
    base = d_chunk_offsets[c]
    ctbytes = d_chunk_ctbytes[c]

    if ctbytes == 16 + chunk_nbytes:
        if b == 0 and tid == 0:
            _write_blosc_header(d_shard_out, base, flag_fmt | 0x02, typesize, chunk_nbytes, blocksize, int(ctbytes))
        _unshuffle_sub_block(
            d_sub_blocks,
            c * blocks_per_chunk + b,
            d_shard_out,
            base + 16 + b * blocksize,
            blocksize,
            shuffle_flag,
            typesize,
            tid,
            stride,
        )
        return

    bstart = 16 + 4 * blocks_per_chunk
    for prev in range(b):
        bstart += 4 + d_block_sizes[c, prev]
    csize = d_block_sizes[c, b]
    if tid == 0:
        if b == 0:
            _write_blosc_header(
                d_shard_out, base, flag_fmt | 0x10 | shuffle_flag, typesize, chunk_nbytes, blocksize, int(ctbytes)
            )
        for k in range(4):
            d_shard_out[base + 16 + 4 * b + k] = np.uint8((bstart >> (8 * k)) & 0xFF)
            d_shard_out[base + bstart + k] = np.uint8((csize >> (8 * k)) & 0xFF)
    payload_pos = base + bstart + 4
    for i in range(tid, csize, stride):
        d_shard_out[payload_pos + i] = d_comp_blocks[c, b, i]


@cuda.jit
def _pack_raw_shard_kernel(d_comp_chunks: Any, d_chunk_sizes: Any, d_chunk_offsets: Any, d_shard_out: Any) -> None:
    c = cuda.blockIdx.y  # type: ignore[attr-defined]
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # type: ignore[attr-defined]
    stride = cuda.gridDim.x * cuda.blockDim.x  # type: ignore[attr-defined]
    if c >= d_comp_chunks.shape[0]:
        return
    csize, out_base = d_chunk_sizes[c], d_chunk_offsets[c]
    for i in range(tid, csize, stride):
        d_shard_out[out_base + i] = d_comp_chunks[c, i]


# ---------------------------------------------------------------------------
# Parsed Spec, Shared Arena, and Per-Level Shard Encoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ParsedShardSpec:
    volume_shape: tuple[int, int, int]
    shard_shape: tuple[int, int, int]
    chunk_shape: tuple[int, int, int]
    typesize: int
    dtype: np.dtype
    codec_kind: str
    nvcomp_alg: str
    blosc_flag_fmt: int
    shuffle_flag: int


def _parse_inner_codecs(codecs: list[Any], typesize: int) -> tuple[str, str, int, int]:
    codec_kind, nvcomp_alg, blosc_flag_fmt, shuffle_flag = "none", "", 0, 0
    for inner in codecs:
        if isinstance(inner, BytesCodec):
            if inner.configuration.endian not in ("little", None):
                raise ValueError(f"Unsupported BytesCodec endianness for GPUArrayWriter: {inner.configuration.endian}")
        elif isinstance(inner, ZstdCodec) and codec_kind == "none":
            codec_kind, nvcomp_alg = "zstd", "zstd"
        elif isinstance(inner, BloscCodec) and codec_kind == "none":
            codec_kind = "blosc"
            cname = inner.configuration.cname
            if cname in ("lz4", "lz4hc"):
                nvcomp_alg, blosc_flag_fmt = "lz4", 0x20
            elif cname == "zstd":
                nvcomp_alg, blosc_flag_fmt = "zstd", 0x80
            else:
                raise ValueError(f"Unsupported Blosc cname for GPUArrayWriter: {cname}")
            shuffle = inner.configuration.shuffle
            if shuffle not in ("noshuffle", "shuffle"):
                raise ValueError(f"Unsupported Blosc shuffle for GPUArrayWriter: {shuffle}")
            if typesize > 1 and shuffle == "shuffle":
                shuffle_flag = 0x01
        else:
            raise ValueError(f"Unsupported inner codec for GPUArrayWriter: {inner}")
    return codec_kind, nvcomp_alg, blosc_flag_fmt, shuffle_flag


def _parse_array_meta(meta: Zarr3ArrayMeta) -> _ParsedShardSpec:
    if len(meta.shape) != 4:
        raise ValueError(f"GPUArrayWriter requires 4D [C, Z, Y, X] arrays, got shape {meta.shape}")
    vz, vy, vx = (int(x) for x in meta.shape[1:])
    sz, sy, sx = (int(x) for x in meta.chunk_grid.configuration.chunk_shape[1:])
    dtype = np.dtype(meta.data_type.dtype)
    sharding = next((c for c in meta.codecs if isinstance(c, ShardingIndexedCodec)), None)
    if sharding is None or sharding.configuration.index_location != "end":
        raise ValueError("GPUArrayWriter requires a sharding_indexed array with index_location='end'")
    idx_codecs = sharding.configuration.index_codecs
    if (
        len(idx_codecs) != 2
        or not isinstance(idx_codecs[0], BytesCodec)
        or idx_codecs[0].configuration.endian not in ("little", None)
        or not isinstance(idx_codecs[1], CRC32CCodec)
    ):
        raise ValueError(f"Unsupported shard index_codecs for GPUArrayWriter: {idx_codecs}")
    codec_kind, nvcomp_alg, blosc_flag_fmt, shuffle_flag = _parse_inner_codecs(
        sharding.configuration.codecs, int(dtype.itemsize)
    )
    cz, cy, cx = (int(x) for x in sharding.configuration.chunk_shape[1:])
    return _ParsedShardSpec(
        (vz, vy, vx),
        (sz, sy, sx),
        (cz, cy, cx),
        int(dtype.itemsize),
        dtype,
        codec_kind,
        nvcomp_alg,
        blosc_flag_fmt,
        shuffle_flag,
    )


class _CudaSlice1D:
    __slots__ = ("__cuda_array_interface__", "_owner")

    def __init__(self, ptr: int, size: int, owner: Any) -> None:
        self._owner = owner
        self.__cuda_array_interface__ = {"shape": (size,), "typestr": "|u1", "data": (ptr, False), "version": 3}


@dataclass
class _SharedArena:
    """Process-wide reusable pinned-host & GPU scratch buffers shared across L0..L7 encoders."""

    pinned_pair: list[Any] | None = None
    d_sub: Any = None
    d_comp: Any = None
    d_shard: Any = None

    def get_pinned_pair(self, min_bytes: int) -> list[Any]:
        if self.pinned_pair is None or int(self.pinned_pair[0].nbytes) < min_bytes:
            self.pinned_pair = [cuda.pinned_array(min_bytes, dtype=np.uint8) for _ in range(2)]
        return self.pinned_pair

    def get_device(self, attr: str, min_bytes: int) -> Any:
        buf = getattr(self, attr)
        if buf is None or int(buf.nbytes) < min_bytes:
            buf = cuda.device_array(min_bytes, dtype=np.uint8)
            setattr(self, attr, buf)
        return buf

    def release_device(self) -> None:
        self.d_sub = self.d_comp = self.d_shard = None
        flush_gpu_deallocations()

    def release_all(self) -> None:
        self.release_device()
        self.pinned_pair = None


_ARENA = _SharedArena()


def release_shared_encoder_device_buffers() -> None:
    """Release all shared GPU device buffers used by the shard encoder."""
    _ARENA.release_device()


def release_shared_encoder_buffers() -> None:
    """Release all shared encoder device and pinned host buffers."""
    _ARENA.release_all()


def _prefix_offsets(ctbytes: np.ndarray, start_offset: int) -> tuple[np.ndarray, int]:
    offsets = np.full(len(ctbytes), start_offset, dtype=np.int64)
    if len(ctbytes) > 1:
        offsets[1:] += np.cumsum(ctbytes[:-1])
    return offsets, int(offsets[-1] + ctbytes[-1])


class _GpuShardEncoder:
    def __init__(self, spec: _ParsedShardSpec) -> None:
        from nvidia import nvcomp  # noqa: PLC0415

        self._nvcomp = nvcomp
        self.spec = spec
        sz, sy, sx = spec.shard_shape
        cz, cy, cx = spec.chunk_shape
        self.nc_z, self.nc_y, self.nc_x = sz // cz, sy // cy, sx // cx
        self.max_chunks = self.nc_z * self.nc_y * self.nc_x
        self.chunk_elems = cz * cy * cx
        self.chunk_nbytes = self.chunk_elems * spec.typesize

        if spec.codec_kind == "blosc" and self.max_chunks < 256:
            bs = min(self.chunk_nbytes, _TARGET_BLOSC_BLOCKSIZE)
            while bs > spec.typesize and (self.chunk_nbytes % bs != 0 or (bs // spec.typesize) == 0):
                bs //= 2
            self.blocksize = max(spec.typesize, bs)
            self.blocks_per_chunk = self.chunk_nbytes // self.blocksize
        else:
            self.blocksize, self.blocks_per_chunk = self.chunk_nbytes, 1

        self.elems_per_block = self.blocksize // spec.typesize
        self.max_batch_chunks = max_batch_chunks_for(self.chunk_nbytes, self.max_chunks)
        self.max_subs = self.max_batch_chunks * self.blocks_per_chunk
        self.codec: Any = (
            nvcomp.Codec(algorithm=spec.nvcomp_alg, bitstream_kind=nvcomp.BitstreamKind.RAW)
            if spec.codec_kind != "none"
            else None
        )
        self.d_sub_blocks: Any = None
        self.d_active_coords: Any = None
        self.d_comp_out: Any = None
        self.d_shard_out: Any = None
        self._sub_flat_ptr = self._comp_flat_ptr = 0
        self.max_comp_size = self.blocksize
        self._in_slices: list[_CudaSlice1D] = []
        self._out_slices: list[_CudaSlice1D] = []
        self._batch_cache: dict[int, tuple[Any, Any, Any]] = {}

        if self.codec is not None:
            self._ensure_sub_buffer()
            self.max_comp_size = int(self.codec.get_max_comp_buffer_size(nvcomp.as_array(self._in_slices[0])))
        per_chunk_max = max(self.chunk_nbytes + 64 + 8 * self.blocks_per_chunk, self.max_comp_size)
        self.max_shard_payload = self.max_chunks * per_chunk_max
        self._min_pinned_bytes = self.max_shard_payload + self.max_chunks * 16 + 4
        self._slot_idx = 0
        self.ensure_device_buffers()

    def _ensure_sub_buffer(self) -> None:
        sub_bytes = self.max_subs * self.blocksize
        d_sub_flat = _ARENA.get_device("d_sub", sub_bytes)
        in_ptr = int(d_sub_flat.__cuda_array_interface__["data"][0])
        if self.d_sub_blocks is None or in_ptr != self._sub_flat_ptr:
            self._sub_flat_ptr = in_ptr
            self.d_sub_blocks = d_sub_flat[:sub_bytes].reshape(self.max_subs, self.blocksize)
            self.d_active_coords = cuda.device_array((self.max_batch_chunks, 3), dtype=np.int32)
            self._in_slices = [
                _CudaSlice1D(in_ptr + i * self.blocksize, self.blocksize, d_sub_flat) for i in range(self.max_subs)
            ]
            self._batch_cache.clear()

    def ensure_device_buffers(self) -> None:
        self.h_pinned = _ARENA.get_pinned_pair(self._min_pinned_bytes)
        self._ensure_sub_buffer()
        if self.codec is not None:
            comp_bytes = self.max_subs * self.max_comp_size
            d_comp_flat = _ARENA.get_device("d_comp", comp_bytes)
            out_ptr = int(d_comp_flat.__cuda_array_interface__["data"][0])
            if self.d_comp_out is None or out_ptr != self._comp_flat_ptr:
                self._comp_flat_ptr = out_ptr
                self.d_comp_out = d_comp_flat[:comp_bytes].reshape(self.max_subs, self.max_comp_size)
                self._out_slices = [
                    _CudaSlice1D(out_ptr + i * self.max_comp_size, self.max_comp_size, d_comp_flat)
                    for i in range(self.max_subs)
                ]
                self._batch_cache.clear()
        self.d_shard_out = _ARENA.get_device("d_shard", self.max_shard_payload)[: self.max_shard_payload]

    def release_device_buffers(self) -> None:
        self.d_sub_blocks = self.d_active_coords = self.d_comp_out = self.d_shard_out = None
        self._sub_flat_ptr = self._comp_flat_ptr = 0
        self._in_slices.clear()
        self._out_slices.clear()
        self._batch_cache.clear()

    def warmup(self) -> None:
        self.ensure_device_buffers()
        cz, cy, cx = self.spec.chunk_shape
        d_dummy = cuda.device_array((cz, cy, cx, self.spec.typesize), dtype=np.uint8)
        self.encode_shard_to_host(d_dummy, 0, 0, 0, 0, np.zeros((1, 3), dtype=np.int32), np.zeros(1, dtype=np.int64))
        self._slot_idx = 0
        self.release_device_buffers()
        release_shared_encoder_device_buffers()

    def _get_batch_views(self, n_subs: int) -> tuple[Any, Any, Any]:
        if n_subs not in self._batch_cache:
            self._batch_cache[n_subs] = (
                self._nvcomp.as_arrays(self._in_slices[:n_subs]),
                self._nvcomp.as_arrays(self._out_slices[:n_subs]),
                self.codec.compression_config([self.blocksize] * n_subs),
            )
        return self._batch_cache[n_subs]

    def _launch_gather(
        self, d_slab_u8: Any, slab_z0: int, shard_z0: int, shard_y0: int, shard_x0: int, batch_coords: np.ndarray
    ) -> int:
        nb = int(batch_coords.shape[0])
        nb_subs = nb * self.blocks_per_chunk
        cz, cy, cx = self.spec.chunk_shape
        self.d_active_coords[:nb].copy_to_device(batch_coords)
        _gather_inner_chunks_kernel[(math.ceil(self.chunk_elems / 256), nb), 256](  # type: ignore[index]
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
        return nb_subs

    def _filter_nonempty_chunks(
        self,
        d_slab_u8: Any,
        slab_z0: int,
        shard_z0: int,
        shard_y0: int,
        shard_x0: int,
        active_coords: np.ndarray,
        active_ic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_active = int(active_coords.shape[0])
        if n_active == 0:
            return active_coords, active_ic
        cz, cy, cx = self.spec.chunk_shape
        nonempty = np.zeros(n_active, dtype=np.uint8)
        d_nonempty = cuda.device_array(min(n_active, self.max_batch_chunks), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            for c0 in range(0, n_active, self.max_batch_chunks):
                nb = min(self.max_batch_chunks, n_active - c0)
                self.d_active_coords[:nb].copy_to_device(active_coords[c0 : c0 + nb])
                d_nonempty[:nb].copy_to_device(nonempty[c0 : c0 + nb])
                _find_nonempty_chunks_kernel[nb, 256](  # type: ignore[index]
                    d_slab_u8,
                    slab_z0,
                    shard_z0,
                    shard_y0,
                    shard_x0,
                    cz,
                    cy,
                    cx,
                    self.d_active_coords[:nb],
                    d_nonempty[:nb],
                )
                d_nonempty[:nb].copy_to_host(nonempty[c0 : c0 + nb])
        cuda.synchronize()
        keep = nonempty != 0
        return active_coords[keep], active_ic[keep]

    def _encode_compressed(
        self,
        d_slab_u8: Any,
        slab_z0: int,
        shard_z0: int,
        shard_y0: int,
        shard_x0: int,
        active_coords: np.ndarray,
        host_buf: Any,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        n_active = int(active_coords.shape[0])
        all_offsets = np.zeros(n_active, dtype=np.int64)
        all_ctbytes = np.zeros(n_active, dtype=np.int64)
        cur_offset = 0
        for c0 in range(0, n_active, self.max_batch_chunks):
            nb = min(self.max_batch_chunks, n_active - c0)
            nb_subs = self._launch_gather(d_slab_u8, slab_z0, shard_z0, shard_y0, shard_x0, active_coords[c0 : c0 + nb])
            in_views, out_views, comp_cfg = self._get_batch_views(nb_subs)
            res = self.codec.encode(in_views, out=out_views, compression_config=comp_cfg)
            if self.spec.codec_kind == "blosc":
                sizes = np.asarray(res.buffer_sizes, dtype=np.int32).reshape(nb, self.blocks_per_chunk)
                comp_ct = 16 + 8 * self.blocks_per_chunk + sizes.sum(axis=1, dtype=np.int64)
                memcpy_ct = 16 + self.chunk_nbytes
                incomp = (sizes.max(axis=1) >= self.blocksize) | (comp_ct >= memcpy_ct)
                ctbytes = np.where(incomp, memcpy_ct, comp_ct).astype(np.int64)
                offsets_i64, cur_offset = _prefix_offsets(ctbytes, cur_offset)
                _pack_blosc_shard_kernel[(self.blocks_per_chunk, nb), 256](  # type: ignore[index]
                    self.d_comp_out[:nb_subs].reshape(nb, self.blocks_per_chunk, self.max_comp_size),
                    cuda.to_device(sizes),
                    self.d_sub_blocks[:nb_subs],
                    cuda.to_device(offsets_i64),
                    cuda.to_device(ctbytes),
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
                offsets_i64, cur_offset = _prefix_offsets(ctbytes, cur_offset)
                pack_bx = min(64, max(1, math.ceil(int(sizes_1d.max()) / 256)))
                _pack_raw_shard_kernel[(pack_bx, nb), 256](  # type: ignore[index]
                    self.d_comp_out[:nb], cuda.to_device(sizes_1d), cuda.to_device(offsets_i64), self.d_shard_out
                )
            all_offsets[c0 : c0 + nb], all_ctbytes[c0 : c0 + nb] = offsets_i64, ctbytes
        self.d_shard_out[:cur_offset].copy_to_host(host_buf[:cur_offset])
        cuda.synchronize()
        return cur_offset, all_offsets.astype(np.uint64), all_ctbytes.astype(np.uint64)

    def encode_shard_to_host(
        self,
        d_slab_u8: Any,
        slab_z0: int,
        shard_z0: int,
        shard_y0: int,
        shard_x0: int,
        active_coords: np.ndarray,
        active_ic: np.ndarray,
        *,
        skip_empty_chunks: bool = False,
    ) -> memoryview:
        self.ensure_device_buffers()
        if skip_empty_chunks:
            active_coords, active_ic = self._filter_nonempty_chunks(
                d_slab_u8, slab_z0, shard_z0, shard_y0, shard_x0, active_coords, active_ic
            )
        n_active = int(active_coords.shape[0])
        if n_active == 0:
            return memoryview(b"")
        host_buf = self.h_pinned[self._slot_idx]
        self._slot_idx ^= 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            if self.spec.codec_kind == "none":
                for c0 in range(0, n_active, self.max_batch_chunks):
                    nb = min(self.max_batch_chunks, n_active - c0)
                    nb_subs = self._launch_gather(
                        d_slab_u8, slab_z0, shard_z0, shard_y0, shard_x0, active_coords[c0 : c0 + nb]
                    )
                    b_start, b_len = c0 * self.chunk_nbytes, nb * self.chunk_nbytes
                    self.d_sub_blocks[:nb_subs].reshape(b_len).copy_to_host(host_buf[b_start : b_start + b_len])
                cuda.synchronize()
                total_payload = n_active * self.chunk_nbytes
                offsets = np.arange(n_active, dtype=np.uint64) * np.uint64(self.chunk_nbytes)
                lengths = np.full(n_active, self.chunk_nbytes, dtype=np.uint64)
            else:
                total_payload, offsets, lengths = self._encode_compressed(
                    d_slab_u8, slab_z0, shard_z0, shard_y0, shard_x0, active_coords, host_buf
                )
        index_table = np.full((self.max_chunks, 2), _EMPTY_ENTRY, dtype="<u8")
        index_table[active_ic, 0] = offsets
        index_table[active_ic, 1] = lengths
        index_bytes = index_table.tobytes()
        crc_bytes = struct.pack("<I", _crc32c(index_bytes))
        end_idx = total_payload + len(index_bytes)
        host_buf[total_payload:end_idx] = np.frombuffer(index_bytes, dtype=np.uint8)
        host_buf[end_idx : end_idx + 4] = np.frombuffer(crc_bytes, dtype=np.uint8)
        return memoryview(host_buf)[: end_idx + 4]

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
    nz = min(nc_z, max(0, math.ceil((z_end - shard_z0) / cz)))
    ny = min(nc_y, max(0, math.ceil((vy - shard_y0) / cy)))
    nx = min(nc_x, max(0, math.ceil((vx - shard_x0) / cx)))
    if nz == 0 or ny == 0 or nx == 0:
        return None
    iz, iy, ix = np.meshgrid(
        np.arange(nz, dtype=np.int32), np.arange(ny, dtype=np.int32), np.arange(nx, dtype=np.int32), indexing="ij"
    )
    coords = np.stack((iz.ravel(), iy.ravel(), ix.ravel()), axis=1)
    ic = ((iz.ravel().astype(np.int64) * nc_y + iy.ravel().astype(np.int64)) * nc_x) + ix.ravel().astype(np.int64)
    return coords, ic


class GPUArrayWriter(ArrayWriter):
    """GPU-accelerated Zarr v3 sharded array writer with transparent TSArrayWriter fallback."""

    def __init__(self) -> None:
        self._target: Path | S3Path | None = None
        self._spec: _ParsedShardSpec | None = None
        self._encoder: _GpuShardEncoder | None = None
        self._fallback: TSArrayWriter | None = None
        self._io_pool: ThreadPoolExecutor | None = None
        self.skip_empty_chunks: bool = False
        self.last_written_bytes: int = 0

    def open(self, target: Path | S3Path, store: S3Store | None = None) -> None:
        self.close()
        self._target = target
        if isinstance(target, S3Path) or not _has_nvcomp():
            self._open_fallback(target, store)
            return
        try:
            resolved = target.expanduser().resolve()
            meta = Zarr3ArrayMeta.model_validate_json((resolved / "zarr.json").read_text(encoding="utf-8"))
            self._spec = _parse_array_meta(meta)
            self._target = resolved
            self._io_pool = ThreadPoolExecutor(max_workers=1)
        except Exception:
            log.warning("GPUArrayWriter falling back to TSArrayWriter for %s", target, exc_info=True)
            self._spec = None
            self._open_fallback(target, store)

    def _open_fallback(self, target: Path | S3Path, store: S3Store | None) -> TSArrayWriter:
        fallback = TSArrayWriter()
        fallback.open(target, store)
        self._fallback = fallback
        return fallback

    def _ensure_fallback(self) -> TSArrayWriter:
        if self._fallback is not None:
            return self._fallback
        if self._target is None:
            raise RuntimeError("write_slice called before open")
        return self._open_fallback(self._target, None)

    def write_slice(self, c: int, z_offset: int, arr: np.ndarray) -> int:
        self.last_written_bytes = 0
        if self._target is None:
            raise RuntimeError("write_slice called before open")
        if self._fallback is not None or self._spec is None or not isinstance(self._target, Path):
            return self._ensure_fallback().write_slice(c, z_offset, arr)
        contig = np.ascontiguousarray(arr)
        return self.write_slice_gpu(c, z_offset, cuda.to_device(contig), host_fallback=contig)

    def _write_aligned_shards(self, c: int, z_offset: int, d_slab_u8: Any, vy: int, vx: int, z_end: int) -> None:
        if self._spec is None or not isinstance(self._target, Path):
            raise RuntimeError("GPUArrayWriter is not open on a local path")
        spec = self._spec
        if self._encoder is None:
            self._encoder = _GpuShardEncoder(spec)
        encoder = self._encoder
        sz, sy, sx = spec.shard_shape
        cz, cy, cx = spec.chunk_shape
        pending: Future[None] | None = None
        try:
            for shard_z0 in range(z_offset, z_end, sz):
                for sy_idx in range(math.ceil(vy / sy)):
                    for sx_idx in range(math.ceil(vx / sx)):
                        active = _active_shard_chunks(
                            shard_z0,
                            sy_idx * sy,
                            sx_idx * sx,
                            z_end,
                            vy,
                            vx,
                            cz,
                            cy,
                            cx,
                            encoder.nc_z,
                            encoder.nc_y,
                            encoder.nc_x,
                        )
                        if active is None:
                            continue
                        shard_view = encoder.encode_shard_to_host(
                            d_slab_u8,
                            z_offset,
                            shard_z0,
                            sy_idx * sy,
                            sx_idx * sx,
                            active[0],
                            active[1],
                            skip_empty_chunks=self.skip_empty_chunks,
                        )
                        if len(shard_view) == 0:
                            continue
                        self.last_written_bytes += len(shard_view)
                        path = self._target / "c" / str(c) / str(shard_z0 // sz) / str(sy_idx) / str(sx_idx)
                        if pending is not None:
                            pending.result()
                            pending = None
                        if self._io_pool is not None:
                            pending = self._io_pool.submit(_write_shard_file, path, shard_view)
                        else:
                            _write_shard_file(path, shard_view)
            if pending is not None:
                pending.result()
        except Exception:
            if pending is not None:
                pending.cancel()
            raise

    def write_slice_gpu(self, c: int, z_offset: int, d_slab: Any, *, host_fallback: np.ndarray | None = None) -> int:
        """Write a GPU-resident 3D slab ``(slab_z, vy, vx)`` at channel ``c`` and ``z_offset``."""
        self.last_written_bytes = 0
        if self._target is None:
            raise RuntimeError("write_slice_gpu called before open")
        if self._fallback is not None or self._spec is None or not isinstance(self._target, Path):
            return self._ensure_fallback().write_slice(
                c, z_offset, host_fallback if host_fallback is not None else d_slab.copy_to_host()
            )
        slab_z, vy, vx = (int(x) for x in d_slab.shape)
        if slab_z == 0 or vy == 0 or vx == 0:
            return 0
        sz = self._spec.shard_shape[0]
        z_end = z_offset + slab_z
        if z_offset % sz != 0 or (z_end % sz != 0 and z_end != self._spec.volume_shape[0]):
            return self._ensure_fallback().write_slice(
                c, z_offset, host_fallback if host_fallback is not None else d_slab.copy_to_host()
            )
        d_slab_u8 = d_slab.view(np.uint8).reshape(slab_z, vy, vx, self._spec.typesize)
        self._write_aligned_shards(c, z_offset, d_slab_u8, vy, vx, z_end)
        return slab_z * vy * vx * self._spec.typesize

    def release_device_buffers(self) -> None:
        if self._encoder is not None:
            self._encoder.release_device_buffers()

    def warmup(self) -> None:
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
