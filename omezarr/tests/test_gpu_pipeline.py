"""Unit and integration tests for GPU/CPU spatial preprocessing, sizing, and GPU OME-Zarr writer."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts
from ome_zarr_writer import (
    BinningMode,
    Compression,
    DownscaleType,
    Dtype,
    Local,
    OMEZarrWriter,
    PreFilterType,
    PreprocessConfig,
    ScaleLevel,
    UIVec3D,
    UVec3D,
    WriterConfig,
    filter_and_bin_2d,
    filter_and_bin_volume,
    vram_fits_batch,
)
from ome_zarr_writer.array import ArrayWriter
from ome_zarr_writer.array.gpu import GPUArrayWriter, _has_nvcomp, max_batch_chunks_for
from ome_zarr_writer.pyramid_gpu import is_gpu_available, pyramids_3d_gpu
from ome_zarr_writer.sizing import gpu_workspace_bytes, per_slot_bytes
from ome_zarr_writer.slot import _gpu_worker_slot
from ome_zarr_writer.viewer.zarr_server.base import neuroglancer_compatible_zarr_json


def _write_stack(tmp_path: Path, name: str, cfg: WriterConfig, vol: np.ndarray, *, backend=None) -> Path:
    writer = OMEZarrWriter(slots=2, backend=backend)
    try:
        ds = writer.begin_stack(cfg, Local(target=tmp_path / name))
        target = Path(ds.target)
        for z in range(vol.shape[0]):
            with writer.new_frame() as dst:
                dst[...] = vol[z]
        writer.end_stack()
        return target
    finally:
        writer.close()


def _read_level(target: Path, level: int = 0) -> np.ndarray:
    arr = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(target / str(level))}}).result()
    return np.asarray(arr[0].read().result())


# ---------------------------------------------------------------------------
# Sizing, batch chunk capping & worker lock
# ---------------------------------------------------------------------------


def test_sizing_gpu_vs_cpu_and_vram_fit():
    batch = UIVec3D(z=64, y=256, x=256)
    shard = UIVec3D(z=64, y=128, x=128)
    cpu_slot = per_slot_bytes(batch, ScaleLevel.L3, Dtype.UINT16, use_gpu=False)
    gpu_slot = per_slot_bytes(batch, ScaleLevel.L3, Dtype.UINT16, use_gpu=True, shard_shape=shard)
    assert gpu_slot > 0
    assert cpu_slot > 0

    needed = gpu_workspace_bytes(batch, ScaleLevel.L3, Dtype.UINT16, shard_shape=shard)
    assert needed > 0
    assert vram_fits_batch(batch, ScaleLevel.L3, Dtype.UINT16, free_vram_bytes=needed * 2, shard_shape=shard)
    assert not vram_fits_batch(batch, ScaleLevel.L3, Dtype.UINT16, free_vram_bytes=needed // 2, shard_shape=shard)
    assert not vram_fits_batch(batch, ScaleLevel.L3, Dtype.UINT16, free_vram_bytes=0, shard_shape=shard)
    assert max_batch_chunks_for(64 * 64 * 64 * 2, 100000) <= 32768


def test_gpu_worker_slot_bounds_concurrency():
    active = 0
    max_seen = 0
    lock = threading.Lock()

    def _hold():
        nonlocal active, max_seen
        with _gpu_worker_slot():
            with lock:
                active += 1
                max_seen = max(max_seen, active)
            time.sleep(0.03)
            with lock:
                active -= 1

    threads = [threading.Thread(target=_hold) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert 1 <= max_seen <= 2


# ---------------------------------------------------------------------------
# Spatial pre-filtering & rectangular binning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pre_filter",
    [PreFilterType.NONE, PreFilterType.MEAN, PreFilterType.GAUSSIAN, PreFilterType.MEDIAN, PreFilterType.HIGHPASS],
)
@pytest.mark.parametrize(
    "bin_mode",
    [BinningMode.MEAN, BinningMode.MEDIAN, BinningMode.SUM, BinningMode.MAX, BinningMode.NEAREST],
)
def test_filter_and_bin_volume_all_modes_and_2d_parity(pre_filter: PreFilterType, bin_mode: BinningMode):
    rng = np.random.default_rng(7)
    vol = rng.integers(50, 2000, size=(2, 12, 16), dtype=np.uint16)
    out = filter_and_bin_volume(vol, bin_y=3, bin_x=2, pre_filter=pre_filter, bin_mode=bin_mode, use_gpu=False)
    assert out.shape == (2, 4, 8)
    assert out.dtype == np.uint16
    out_2d = filter_and_bin_2d(vol[0], bin_y=3, bin_x=2, pre_filter=pre_filter, bin_mode=bin_mode, use_gpu=False)
    np.testing.assert_array_equal(out[0], out_2d)
    if is_gpu_available():
        gpu_out = filter_and_bin_volume(vol, bin_y=3, bin_x=2, pre_filter=pre_filter, bin_mode=bin_mode, use_gpu=True)
        np.testing.assert_allclose(gpu_out.astype(np.float32), out.astype(np.float32), atol=1.0)


def test_median_binning_supports_large_bin_factors_without_truncation():
    # Regression test: 10x10 = 100 pixels (> 64) must take the true median, not truncate at 64.
    vol = np.arange(100, dtype=np.uint16).reshape(1, 10, 10)
    out = filter_and_bin_volume(vol, bin_y=10, bin_x=10, bin_mode=BinningMode.MEDIAN, use_gpu=False)
    assert int(out[0, 0, 0]) == 50  # median of 0..99 is 49.5 -> rounds to 50


def test_preprocess_config_spatial_metadata():
    assert PreprocessConfig().to_metadata() is None
    pp = PreprocessConfig(bin_y=2, bin_x=1, pre_filter=PreFilterType.HIGHPASS, filter_size=5, highpass_add=50.0)
    meta = pp.to_metadata()
    assert meta == {
        "bin_y": 2,
        "bin_x": 1,
        "bin_mode": "mean",
        "pre_filter": "highpass",
        "filter_size": 5,
        "highpass_add": 50.0,
    }


# ---------------------------------------------------------------------------
# End-to-end writer: preprocessing, codec knobs, VRAM fallback & Neuroglancer
# ---------------------------------------------------------------------------


def test_writer_applies_preprocessing_codec_knobs_and_records_metadata(tmp_path: Path):
    rng = np.random.default_rng(123)
    vol = rng.integers(80, 1000, size=(64, 64, 64), dtype=np.uint16)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=64, y=32, x=64),
        voxel_size=UVec3D(z=1.0, y=1.0, x=0.5),
        max_level=ScaleLevel.L1,
        compression=Compression.BLOSC_ZSTD,
        compression_level=5,
        compression_shuffle="shuffle",
        bin_y=2,
        bin_x=1,
        bin_mode=BinningMode.MEAN,
        raw_frame_y=64,
        raw_frame_x=64,
        use_gpu=False,
    )
    target = _write_stack(tmp_path, "proc", cfg, vol, backend=ArrayWriter.Backend.TS)
    readback = _read_level(target, 0)
    assert readback.shape == (64, 32, 64)

    group_meta = json.loads((target / "zarr.json").read_text())
    assert group_meta["attributes"]["preprocessing"]["bin_y"] == 2
    assert group_meta["attributes"]["preprocessing"]["bin_mode"] == "mean"
    arr_meta = json.loads((target / "0" / "zarr.json").read_text())
    blosc_cfg = arr_meta["codecs"][0]["configuration"]["codecs"][1]["configuration"]
    assert blosc_cfg["cname"] == "zstd"
    assert blosc_cfg["clevel"] == 5
    assert blosc_cfg["shuffle"] == "shuffle"


def test_vram_fallback_downgrades_gpu_backend_and_ring_reuse_skips_sizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("ome_zarr_writer.pyramid_gpu.is_gpu_available", lambda: True)
    monkeypatch.setattr("ome_zarr_writer.pyramid_gpu.get_free_gpu_bytes", lambda: 1024)  # tiny VRAM forces fallback

    cfg = WriterConfig(
        volume_shape=UIVec3D(z=64, y=64, x=64),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L0,
        use_gpu=True,
    )
    sizer_calls = 0

    def _sizer(_cfg: WriterConfig) -> int:
        nonlocal sizer_calls
        sizer_calls += 1
        return 2

    writer = OMEZarrWriter(backend=ArrayWriter.Backend.GPU)
    try:
        for idx in range(2):
            writer.begin_stack(cfg, Local(target=tmp_path / f"vram_fb_{idx}"), sizer=_sizer)
            assert writer._ring_use_gpu is False
            for _ in range(64):
                with writer.new_frame() as dst:
                    dst.fill(idx + 1)
            writer.end_stack()
        assert sizer_calls == 1  # ring reused on second stack without re-calling sizer
    finally:
        writer.close()


def test_neuroglancer_compatible_zarr_json_preserves_all_multiscales(tmp_path: Path):
    zarr_json = tmp_path / "zarr.json"
    doc = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.6",
                "multiscales": [
                    {
                        "type": "gaussian",
                        "coordinateSystems": [{"name": "physical", "axes": [{"name": "z", "type": "space"}]}],
                        "datasets": [
                            {
                                "path": "0",
                                "coordinateTransformations": [
                                    {"type": "sequence", "transformations": [{"type": "scale", "scale": [1, 1, 1, 1]}]}
                                ],
                            }
                        ],
                    },
                    {
                        "type": "mean",
                        "coordinateSystems": [{"name": "physical", "axes": [{"name": "z", "type": "space"}]}],
                        "datasets": [],
                    },
                ],
            }
        },
    }
    zarr_json.write_text(json.dumps(doc))
    payload = neuroglancer_compatible_zarr_json(zarr_json)
    assert payload is not None
    out = json.loads(payload.decode())
    assert out["attributes"]["ome"]["version"] == "0.5"
    assert len(out["attributes"]["ome"]["multiscales"]) == 2
    assert [m["type"] for m in out["attributes"]["ome"]["multiscales"]] == ["gaussian", "mean"]


@pytest.mark.skipif(not _has_nvcomp(), reason="requires CUDA + nvidia.nvcomp")
@pytest.mark.parametrize(
    "compression",
    [Compression.BLOSC_LZ4, Compression.BLOSC_ZSTD, Compression.ZSTD, Compression.GZIP, Compression.NONE],
)
def test_gpu_array_writer_roundtrip_and_incompressible_noise(tmp_path: Path, compression: Compression):
    rng = np.random.default_rng(99)
    vol = rng.integers(0, 65535, size=(64, 64, 64), dtype=np.uint16)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=64, y=64, x=64),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L1,
        compression=compression,
        use_gpu=True,
    )
    target = _write_stack(tmp_path, f"gpu_{compression.value}", cfg, vol, backend=ArrayWriter.Backend.GPU)
    np.testing.assert_array_equal(_read_level(target, 0), vol)


@pytest.mark.skipif(not is_gpu_available(), reason="requires CUDA GPU")
@pytest.mark.parametrize(
    "reduction", [DownscaleType.GAUSSIAN, DownscaleType.MEAN, DownscaleType.MAX, DownscaleType.MIN]
)
def test_pyramids_3d_gpu_runs_all_reductions(reduction: DownscaleType):
    vol = np.random.default_rng(0).integers(0, 4096, size=(8, 16, 16), dtype=np.uint16)
    pyr = pyramids_3d_gpu(vol, ScaleLevel.L2, reduction=reduction)
    assert pyr[ScaleLevel.L1].shape == (4, 8, 8)
    assert pyr[ScaleLevel.L2].shape == (2, 4, 4)


def test_gpu_array_writer_falls_back_cleanly_without_nvcomp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("ome_zarr_writer.array.gpu._has_nvcomp", lambda: False)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=64, y=64, x=64),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L0,
    )
    target = tmp_path / "fallback.ome.zarr"
    cfg.dataset.write_metadata(target)
    vol = np.arange(64 * 64 * 64, dtype=np.uint16).reshape(64, 64, 64)
    w = GPUArrayWriter()
    w.open(target / "0")
    w.write_slice(0, 0, vol)
    w.close()
    np.testing.assert_array_equal(_read_level(target, 0), vol)
