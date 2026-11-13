"""Comprehensive tests for GPU-accelerated 3D pyramid generation, nvcomp Zarr v3 shard writing,
RAM/VRAM sizing, and the separate GPU BatchSlot worker run function across 8-bit, 12-bit, and 16-bit data.
"""

import os
import threading
from pathlib import Path

import numpy as np
import ome_zarr_writer.slot as slot_mod
import pytest
from ome_zarr_writer import (
    Compression,
    DownscaleType,
    Dtype,
    Local,
    OMEZarrWriter,
    ScaleLevel,
    UIVec3D,
    UVec3D,
    WriterConfig,
    is_gpu_available,
    pyramids_3d_gpu,
)
from ome_zarr_writer.array import ArrayWriter, GPUArrayWriter
from ome_zarr_writer.array.gpu import _parse_array_meta, max_batch_chunks_for
from ome_zarr_writer.array.ts import TSArrayReader
from ome_zarr_writer.dataset import (
    BloscCodec,
    BytesCodec,
    CRC32CCodec,
    ShardingIndexedCodec,
    Zarr3ArrayMeta,
)
from ome_zarr_writer.pyramid import pyramids_3d_numba
from ome_zarr_writer.pyramid_gpu import clear_gpu_workspace_cache
from ome_zarr_writer.sizing import (
    gpu_workspace_bytes,
    per_slot_bytes,
    ring_shm_bytes,
    slots_for_budget,
)
from ome_zarr_writer.slot import BatchSlot, OutputSetup, _gpu_worker_slot

_requires_gpu = pytest.mark.skipif(not is_gpu_available(), reason="CUDA GPU not available")


def _make_bitdepth_volume(shape: tuple[int, int, int], bit_depth: int, seed: int = 42) -> tuple[np.ndarray, Dtype]:
    """Generate a synthetic microscopy volume for 8-bit (uint8), 12-bit (uint16 0..4095), or 16-bit (uint16)."""
    rng = np.random.default_rng(seed)
    z, y, x = shape
    zz = np.linspace(0, 1, z, dtype=np.float32)[:, None, None]
    yy = np.linspace(0, 1, y, dtype=np.float32)[None, :, None]
    xx = np.linspace(0, 1, x, dtype=np.float32)[None, None, :]
    base = 0.5 + 0.4 * np.sin(6.0 * zz + 8.0 * yy) * np.cos(8.0 * xx)

    if bit_depth == 8:
        noise = rng.integers(0, 20, size=shape, dtype=np.uint8)
        vol = np.clip(base * 200.0, 0, 235).astype(np.uint8) + noise
        return vol, Dtype.UINT8
    if bit_depth == 12:
        # 12-bit sensor data (MONO12) is stored in a uint16 container with values in [0, 4095]
        noise = rng.integers(0, 120, size=shape, dtype=np.uint16)
        vol = np.clip(base * 3600.0, 0, 3975).astype(np.uint16) + noise
        return vol, Dtype.UINT16
    if bit_depth == 16:
        noise = rng.integers(0, 1500, size=shape, dtype=np.uint16)
        vol = np.clip(base * 55000.0, 0, 64000).astype(np.uint16) + noise
        return vol, Dtype.UINT16
    raise ValueError(f"Unsupported bit_depth: {bit_depth}")


# ── 1. Pure-Python CPU CI Tests (Sizing, Validation, Batching, Slot Locking) ───


@pytest.mark.parametrize("dtype", [Dtype.UINT8, Dtype.UINT16])
def test_gpu_sizing_reduces_host_ram_per_slot(dtype: Dtype) -> None:
    shape = UIVec3D(z=128, y=2048, x=2048)
    level = ScaleLevel.L5

    cpu_bytes = per_slot_bytes(shape, level, dtype, use_gpu=False)
    gpu_bytes = per_slot_bytes(shape, level, dtype, use_gpu=True)
    shm_only = ring_shm_bytes(1, shape, level, dtype)

    assert gpu_bytes == shm_only
    assert gpu_bytes < cpu_bytes
    assert gpu_workspace_bytes(shape, level, dtype) > shm_only

    # A budget that holds exactly 2 GPU slots would fail for 2 CPU slots
    budget = 2 * gpu_bytes + 1024
    assert slots_for_budget(shape, level, dtype, budget_bytes=budget, use_gpu=True) == 2


@pytest.mark.parametrize("dtype", [Dtype.UINT8, Dtype.UINT16])
@pytest.mark.parametrize("max_level", [ScaleLevel.L1, ScaleLevel.L3])
def test_gpu_workspace_bytes_matches_workspace_formula(dtype: Dtype, max_level: ScaleLevel) -> None:
    shape = UIVec3D(z=64, y=128, x=128)
    d_levels = sum(
        max(1, s.z) * max(1, s.y) * max(1, s.x) * dtype.itemsize for lvl in max_level.levels for s in [lvl.scale(shape)]
    )
    d_f32_levels = sum(
        max(1, s.z) * max(1, s.y) * max(1, s.x) * 4
        for lvl in max_level.levels
        if lvl != ScaleLevel.L0
        for s in [lvl.scale(shape)]
    )
    d_xy_scratch = max(1, shape.z) * max(1, shape.y // 2) * max(1, shape.x // 2) * 4
    expected = d_levels + d_f32_levels + d_xy_scratch
    assert gpu_workspace_bytes(shape, max_level, dtype) == expected


def test_max_batch_chunks_for_caps_cuda_grid_dim_y() -> None:
    # 16x16x16 uint8 = 4096 B; 65536 chunks would exceed CUDA gridDim.y <= 65535 without capping
    capped = max_batch_chunks_for(chunk_nbytes=16 * 16 * 16, max_chunks=65536)
    assert 1 <= capped <= 32768
    assert max_batch_chunks_for(chunk_nbytes=4096, max_chunks=100) == 100
    assert max_batch_chunks_for(chunk_nbytes=512 * 1024 * 1024, max_chunks=10) == 1


@pytest.mark.parametrize(
    ("compression", "expected_kind"),
    [
        (Compression.NONE, "none"),
        (Compression.BLOSC_LZ4, "blosc"),
        (Compression.BLOSC_ZSTD, "blosc"),
        (Compression.ZSTD, "zstd"),
        (Compression.GZIP, "gzip"),
    ],
)
def test_parse_array_meta_accepts_valid_codecs(compression: Compression, expected_kind: str) -> None:
    meta = Zarr3ArrayMeta.sharded(
        shape=[1, 64, 128, 128],
        shard_shape=[1, 64, 128, 128],
        chunk_shape=[1, 32, 64, 64],
        dtype=Dtype.UINT16,
        compression=compression,
    )
    spec = _parse_array_meta(meta)
    assert spec is not None
    assert spec.codec_kind == expected_kind


def test_parse_array_meta_rejects_invalid_codecs() -> None:
    base = Zarr3ArrayMeta.sharded(
        shape=[1, 64, 128, 128],
        shard_shape=[1, 64, 128, 128],
        chunk_shape=[1, 32, 64, 64],
        dtype=Dtype.UINT16,
        compression=Compression.BLOSC_LZ4,
    )
    sharding = next(c for c in base.codecs if isinstance(c, ShardingIndexedCodec))

    # 1. Big-endian BytesCodec in inner codecs
    big_endian_bytes = BytesCodec.model_validate({"configuration": {"endian": "big"}})
    meta_big_endian = base.model_copy(
        update={
            "codecs": [
                sharding.model_copy(
                    update={"configuration": sharding.configuration.model_copy(update={"codecs": [big_endian_bytes]})}
                )
            ]
        }
    )
    with pytest.raises(ValueError, match=r".+"):
        _parse_array_meta(meta_big_endian)

    # 2. Unexpected inner CRC32CCodec
    meta_inner_crc = base.model_copy(
        update={
            "codecs": [
                sharding.model_copy(
                    update={
                        "configuration": sharding.configuration.model_copy(
                            update={
                                "codecs": [
                                    BytesCodec(),
                                    BloscCodec.model_validate({"configuration": {"cname": "lz4"}}),
                                    CRC32CCodec(),
                                ]
                            }
                        )
                    }
                )
            ]
        }
    )
    with pytest.raises(ValueError, match=r".+"):
        _parse_array_meta(meta_inner_crc)

    # 3. Invalid index_codecs (missing crc32c, big-endian bytes, or wrong order)
    for bad_index in (
        [BytesCodec()],
        [big_endian_bytes, CRC32CCodec()],
        [CRC32CCodec(), BytesCodec()],
    ):
        meta_bad_idx = base.model_copy(
            update={
                "codecs": [
                    sharding.model_copy(
                        update={"configuration": sharding.configuration.model_copy(update={"index_codecs": bad_index})}
                    )
                ]
            }
        )
        with pytest.raises(ValueError, match=r".+"):
            _parse_array_meta(meta_bad_idx)


def test_gpu_worker_slot_uses_per_user_temp_dir_and_no_hol_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fcntl")

    opened_paths: list[Path] = []
    real_open = os.open

    def tracking_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
        opened_paths.append(Path(path))
        return real_open(path, flags, mode)

    monkeypatch.setattr(slot_mod.os, "open", tracking_open)
    monkeypatch.setattr(slot_mod, "_GPU_CONCURRENT_WORKERS", 2)

    with _gpu_worker_slot():
        pass

    assert opened_paths
    for p in opened_paths:
        assert not str(p).startswith("/tmp/voxel_gpu_worker_")  # noqa: S108
        assert p.parent != Path("/tmp")  # noqa: S108
        if hasattr(os, "getuid"):
            assert p.parent.stat().st_uid == os.getuid()

    # Verify concurrent acquisition across 2 slots and no pid % N head-of-line blocking
    acquired = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    t2_acquired = threading.Event()
    t2_release = threading.Event()
    errors: list[BaseException] = []

    def hold_slot(idx: int) -> None:
        try:
            with _gpu_worker_slot():
                acquired[idx].set()
                release[idx].wait(timeout=5.0)
        except BaseException as exc:
            errors.append(exc)

    def wait_slot() -> None:
        try:
            with _gpu_worker_slot():
                t2_acquired.set()
                t2_release.wait(timeout=5.0)
        except BaseException as exc:
            errors.append(exc)

    t0 = threading.Thread(target=hold_slot, args=(0,))
    t1 = threading.Thread(target=hold_slot, args=(1,))
    t2 = threading.Thread(target=wait_slot)

    t0.start()
    assert acquired[0].wait(timeout=2.0)
    t1.start()
    assert acquired[1].wait(timeout=2.0)

    t2.start()
    assert not t2_acquired.wait(timeout=0.1)

    # Release the slot that is NOT (pid % 2) so the pid % 2 slot stays held
    hold_idx = os.getpid() % 2
    release_idx = 1 - hold_idx
    release[release_idx].set()

    try:
        assert t2_acquired.wait(timeout=2.0)
    finally:
        release[hold_idx].set()
        t2_release.set()
        t0.join(timeout=5.0)
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

    assert not errors


def test_omezarr_writer_use_gpu_cpu_fallback(tmp_path: Path) -> None:
    """OMEZarrWriter with use_gpu=True on a CPU-only host falls back cleanly to CPU and writes valid OME-Zarr."""
    z, y, x = 64, 64, 64
    vol, dtype = _make_bitdepth_volume((z, y, x), bit_depth=16)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=z, y=y, x=x),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L1,
        dtype=dtype,
        compression=Compression.BLOSC_LZ4,
        downscale_type=DownscaleType.MEAN,
        use_gpu=True,
    )
    writer = OMEZarrWriter(slots=2)
    try:
        writer.begin_stack(cfg, Local(target=tmp_path / "cpu_fallback"))
        for k in range(z):
            with writer.new_frame() as frame:
                frame[...] = vol[k]
        writer.end_stack()
    finally:
        writer.close()

    root = tmp_path / "cpu_fallback.ome.zarr"
    assert (root / "zarr.json").exists()
    l0 = TSArrayReader(root / "0").read_3d(z0=0, n=z)
    l1 = TSArrayReader(root / "1").read_3d(z0=0, n=z // 2)
    np.testing.assert_array_equal(l0, vol)
    assert l1.shape == (z // 2, y // 2, x // 2)
    assert l1.dtype == np.uint16


# ── 2. GPU 3D Pyramid Parity Across All Bit-Depths & Reductions ─────────────────


@_requires_gpu
@pytest.mark.parametrize("bit_depth", [8, 12, 16])
@pytest.mark.parametrize(
    "reduction",
    [DownscaleType.GAUSSIAN, DownscaleType.MEAN, DownscaleType.MAX, DownscaleType.MIN],
)
def test_pyramids_3d_gpu_parity_all_bit_depths(bit_depth: int, reduction: DownscaleType) -> None:
    vol, _ = _make_bitdepth_volume((32, 128, 128), bit_depth=bit_depth)
    max_level = ScaleLevel.L3

    cpu_pyr = pyramids_3d_numba(vol, max_level, reduction=reduction, parallel=True)
    gpu_pyr = pyramids_3d_gpu(vol, max_level, reduction=reduction)
    clear_gpu_workspace_cache()

    assert set(cpu_pyr.keys()) == set(gpu_pyr.keys())
    for level in cpu_pyr:
        c_arr = cpu_pyr[level]
        g_arr = gpu_pyr[level]
        assert c_arr.shape == g_arr.shape
        assert c_arr.dtype == np.float32
        assert g_arr.dtype == np.float32
        if reduction in (DownscaleType.MEAN, DownscaleType.MAX, DownscaleType.MIN):
            np.testing.assert_allclose(g_arr, c_arr, rtol=1e-6, atol=1e-5)
        else:
            np.testing.assert_allclose(g_arr, c_arr, rtol=1e-5, atol=1e-3)
            # Integer cast parity (within 1 LSB from float32 rounding boundary)
            c_cast = c_arr.astype(vol.dtype)
            g_cast = g_arr.astype(vol.dtype)
            max_int_diff = int(np.max(np.abs(c_cast.astype(np.int64) - g_cast.astype(np.int64))))
            assert max_int_diff <= 1


# ── 3. GPUArrayWriter Codec & Bit-Depth Matrix (Readback via TensorStore) ───────


@_requires_gpu
@pytest.mark.parametrize("bit_depth", [8, 12, 16])
@pytest.mark.parametrize(
    "compression",
    [
        Compression.NONE,
        Compression.BLOSC_LZ4,
        Compression.BLOSC_ZSTD,
        Compression.LZ4,
        Compression.ZSTD,
        Compression.GZIP,
    ],
)
def test_gpu_array_writer_all_codecs_and_bit_depths(
    tmp_path: Path,
    bit_depth: int,
    compression: Compression,
) -> None:
    vol, dtype = _make_bitdepth_volume((96, 160, 160), bit_depth=bit_depth)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=96, y=160, x=160),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L1,
        dtype=dtype,
        compression=compression,
        target_shard_gb=0.001,  # Small shards so multiple Y/X and Z shards + partial tail shard are exercised
    )
    root = tmp_path / f"codec_{bit_depth}_{compression.value}.ome.zarr"
    cfg.dataset.write_metadata(root)

    writer = GPUArrayWriter()
    try:
        writer.open(root / "0")
        # Write first full Z-shard (64 slices) + second partial tail Z-shard (32 slices)
        b1 = writer.write_slice(0, 0, vol[:64])
        b2 = writer.write_slice(0, 64, vol[64:])
        assert b1 + b2 == vol.nbytes
    finally:
        writer.close()

    read_back = TSArrayReader(root / "0").read_3d(z0=0, n=96)
    np.testing.assert_array_equal(read_back, vol)


@_requires_gpu
def test_gpu_array_writer_incompressible_random_noise_fallback(tmp_path: Path) -> None:
    """High-entropy random bytes that expand under LZ4/Zstd must trigger BLOSC_MEMCPYED and decode bit-exact."""
    rng = np.random.default_rng(999)
    vol = rng.integers(0, 65536, size=(64, 128, 128), dtype=np.uint16)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=64, y=128, x=128),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L0,
        dtype=Dtype.UINT16,
        compression=Compression.BLOSC_LZ4,
    )
    root = tmp_path / "incompressible.ome.zarr"
    cfg.dataset.write_metadata(root)

    writer = GPUArrayWriter()
    try:
        writer.open(root / "0")
        writer.write_slice(0, 0, vol)
    finally:
        writer.close()

    read_back = TSArrayReader(root / "0").read_3d(z0=0, n=64)
    np.testing.assert_array_equal(read_back, vol)


@_requires_gpu
@pytest.mark.parametrize("compression", [Compression.ZSTD, Compression.GZIP])
def test_gpu_array_writer_incompressible_random_noise_zstd_gzip(
    tmp_path: Path,
    compression: Compression,
) -> None:
    """High-entropy random bytes under ZSTD/GZIP must not write out-of-bounds and must round-trip bit-exact."""
    rng = np.random.default_rng(999)
    vol = rng.integers(0, 65536, size=(64, 128, 128), dtype=np.uint16)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=64, y=128, x=128),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L0,
        dtype=Dtype.UINT16,
        compression=compression,
    )
    root = tmp_path / f"incompressible_{compression.value}.ome.zarr"
    cfg.dataset.write_metadata(root)

    writer = GPUArrayWriter()
    try:
        writer.open(root / "0")
        written = writer.write_slice(0, 0, vol)
        assert written == vol.nbytes
    finally:
        writer.close()

    read_back = TSArrayReader(root / "0").read_3d(z0=0, n=64)
    np.testing.assert_array_equal(read_back, vol)


# ── 4. Separate Worker Run Function & End-to-End OMEZarrWriter Parity ───────────


@_requires_gpu
@pytest.mark.slow
@pytest.mark.parametrize("bit_depth", [8, 12, 16])
def test_omezarr_writer_cpu_vs_gpu_end_to_end(tmp_path: Path, bit_depth: int) -> None:
    """Write the same multi-batch dataset (2 full batches + 1 partial batch) via the CPU worker
    (`_worker_process_and_write`) and the GPU worker (`_worker_process_and_write_gpu` + `GPUArrayWriter`)
    across 8-bit, 12-bit, and 16-bit volumes, and verify all pyramid levels L0..L2 match."""
    z, y, x = 160, 128, 128  # batch_z=64 -> 2 full batches (64, 64) + 1 partial batch (32)
    vol, dtype = _make_bitdepth_volume((z, y, x), bit_depth=bit_depth)

    for use_gpu, label in ((False, "cpu"), (True, "gpu")):
        cfg = WriterConfig(
            volume_shape=UIVec3D(z=z, y=y, x=x),
            voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
            max_level=ScaleLevel.L2,
            dtype=dtype,
            compression=Compression.BLOSC_LZ4,
            downscale_type=DownscaleType.GAUSSIAN,
            use_gpu=use_gpu,
        )
        writer = OMEZarrWriter(slots=2)
        try:
            writer.begin_stack(cfg, Local(target=tmp_path / f"ds_{label}"))
            for k in range(z):
                with writer.new_frame() as frame:
                    frame[...] = vol[k]
            writer.end_stack()
        finally:
            writer.close()

    cpu_root = tmp_path / "ds_cpu.ome.zarr"
    gpu_root = tmp_path / "ds_gpu.ome.zarr"

    # L0 must be 100% bit-exact with input volume
    gpu_l0 = TSArrayReader(gpu_root / "0").read_3d(z0=0, n=z)
    cpu_l0 = TSArrayReader(cpu_root / "0").read_3d(z0=0, n=z)
    np.testing.assert_array_equal(gpu_l0, vol)
    np.testing.assert_array_equal(cpu_l0, vol)

    # Pyramid levels L1 and L2 must match CPU within <= 1 LSB
    for lvl in (ScaleLevel.L1, ScaleLevel.L2):
        zn = z // lvl.factor
        c_lvl = TSArrayReader(cpu_root / str(lvl.value)).read_3d(z0=0, n=zn)
        g_lvl = TSArrayReader(gpu_root / str(lvl.value)).read_3d(z0=0, n=zn)
        assert c_lvl.shape == g_lvl.shape
        max_diff = int(np.max(np.abs(c_lvl.astype(np.int64) - g_lvl.astype(np.int64))))
        assert max_diff <= 1, f"Level {lvl.name} differed by {max_diff} LSB"


@_requires_gpu
@pytest.mark.slow
def test_batchslot_flush_cpu_and_flush_gpu_toggle(tmp_path: Path) -> None:
    """Verify a single BatchSlot can switch cleanly between CPU `flush(use_gpu=False)` and GPU `flush_gpu()`."""
    z, y, x = 128, 64, 64
    vol, dtype = _make_bitdepth_volume((z, y, x), bit_depth=12)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=z, y=y, x=x),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L1,
        dtype=dtype,
    )
    batch_z = cfg.batch_z
    assert batch_z == 64

    root = tmp_path / "toggle.ome.zarr"
    cfg.dataset.write_metadata(root)
    setup = OutputSetup(
        backend=ArrayWriter.Backend.GPU,
        author_root=root,
        store=None,
        channel=0,
        levels=tuple(cfg.dataset.arrays),
        batch_z=batch_z,
        volume_z=z,
        reduction=DownscaleType.MEAN,
        use_gpu=False,
    )

    slot = BatchSlot(name=f"tog_{os.getpid()}", shape_l0=cfg.batch_shape, max_level=cfg.max_level, dtype=cfg.dtype)
    try:
        slot.bind_output(setup)
        # Batch 0: run via CPU worker function
        slot.assign_batch(0)
        for i in range(batch_z):
            slot.frame_array()[...] = vol[i]
            slot.commit_frame()
        res_cpu = slot.flush(use_gpu=False).result()
        assert res_cpu.flushed_bytes > 0

        # Batch 1: run via GPU worker function
        slot.assign_batch(1)
        for i in range(batch_z):
            slot.frame_array()[...] = vol[batch_z + i]
            slot.commit_frame()
        res_gpu = slot.flush_gpu().result()
        assert res_gpu.flushed_bytes > 0
    finally:
        slot.close()

    read_l0 = TSArrayReader(root / "0").read_3d(z0=0, n=z)
    np.testing.assert_array_equal(read_l0, vol)
