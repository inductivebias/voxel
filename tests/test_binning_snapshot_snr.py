"""Tests for per-camera GPU rectangular binning + pre-filtering, single-frame snapshot (.tif + preview),
and stationary multi-frame SNR & photobleaching diagnostics + self-contained HTML report.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from vxlib.vector import IVec2D, Vec2D

from vxl._utils.tiff import write_tiff_2d
from vxl.devices.camera.base import (
    Camera,
    CameraController,
    IntRange,
    PixelFormat,
    ROIGrid,
    SensorROI,
    StreamInfo,
    TriggerMode,
    TriggerPolarity,
)
from vxl.instrument.snr import (
    SnrReportResult,
    compute_camera_snr_diagnostics,
    render_snr_html_report,
)


class _DummyCamera(Camera):
    """Minimal in-memory camera implementation for testing Camera GPU binning, snapshots, and SNR collection."""

    def __init__(self, uid: str = "cam_test", height: int = 128, width: int = 192) -> None:
        super().__init__(uid=uid)
        self._sensor = IVec2D(y=height, x=width)
        self._roi = SensorROI(x=0, y=0, w=width, h=height)
        self._binning = 1
        self._pixel_format: PixelFormat = "MONO16"
        self._exposure_ms = 20.0
        self._fps = 50.0
        self._frame_seq = 0
        self.started_frame_counts: list[int | None] = []
        self.stop_count = 0

    @property
    def sensor_size_px(self) -> IVec2D:
        return self._sensor

    @property
    def pixel_size_um(self) -> Vec2D:
        return Vec2D(y=6.5, x=6.5)

    @property
    def pixel_format(self) -> PixelFormat:
        return self._pixel_format

    @pixel_format.setter
    def pixel_format(self, fmt: PixelFormat) -> None:
        self._pixel_format = fmt

    @property
    def binning(self) -> int:
        return self._binning

    @binning.setter
    def binning(self, b: int) -> None:
        self._binning = int(b)

    @property
    def exposure_time_ms(self) -> float:
        return self._exposure_ms

    @exposure_time_ms.setter
    def exposure_time_ms(self, val: float) -> None:
        self._exposure_ms = float(val)

    @property
    def frame_rate_hz(self) -> float:
        return self._fps

    @frame_rate_hz.setter
    def frame_rate_hz(self, val: float) -> None:
        self._fps = float(val)

    def _get_roi(self) -> SensorROI:
        return self._roi

    def _set_roi(self, roi: SensorROI) -> None:
        self._roi = roi

    @property
    def roi_grid(self) -> ROIGrid:
        return ROIGrid(
            h=IntRange(min=16, max=self._sensor.x, step=4),
            v=IntRange(min=16, max=self._sensor.y, step=4),
        )

    @property
    def stream_info(self) -> StreamInfo | None:
        return None

    def _configure_trigger_mode(self, mode: TriggerMode) -> None:
        pass

    def _configure_trigger_polarity(self, polarity: TriggerPolarity) -> None:
        pass

    def _allocate_buffer(self) -> None:
        pass

    def _start(self, frame_count: int | None = None) -> None:
        self.started_frame_counts.append(frame_count)
        self._frame_seq = 0

    def _grab_frame(self, out: np.ndarray) -> None:
        h, w = out.shape
        yy, xx = np.ogrid[:h, :w]
        base = 100 + ((yy * 13 + xx * 7 + self._frame_seq * 3) % 2000)
        out[...] = base.astype(out.dtype)
        self._frame_seq += 1

    def stop(self) -> None:
        self.stop_count += 1


# ── 1. Pure-Python Baseline TIFF Writer ─────────────────────────────────────────


@pytest.mark.parametrize(
    "dtype",
    [np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)],
)
def test_write_tiff_2d_header_and_payload(tmp_path: Path, dtype: np.dtype) -> None:
    rng = np.random.default_rng(42)
    if dtype == np.float32:
        img = rng.normal(loc=500.0, scale=25.0, size=(32, 48)).astype(np.float32)
    else:
        img = rng.integers(0, 255, size=(32, 48), dtype=dtype)

    out_path = tmp_path / "snapshots" / f"snap_{dtype.name}.tif"
    saved = write_tiff_2d(
        out_path,
        img,
        pixel_size_um=(0.195, 0.293),
        metadata={"camera": "cam_1", "exposure_ms": 20.0},
    )
    assert saved == out_path
    assert saved.exists()
    raw = saved.read_bytes()

    # Little-endian TIFF magic (II, 42, IFD offset = 8)
    byte_order, magic, ifd_offset = struct.unpack("<2sHI", raw[:8])
    assert byte_order == b"II"
    assert magic == 42
    assert ifd_offset == 8

    # Pixel payload at end of file matches image bytes exactly
    expected_bytes = np.ascontiguousarray(img, dtype=dtype.newbyteorder("<")).tobytes()
    assert raw[-len(expected_bytes) :] == expected_bytes


# ── 2. Per-Camera GPU Binning Properties & Snapshot Controller ──────────────────


@pytest.mark.asyncio
async def test_camera_gpu_binning_properties_and_snapshot_flow(tmp_path: Path) -> None:
    cam = _DummyCamera(uid="cam_1", height=120, width=180)
    assert not cam.has_gpu_binning
    assert cam.output_frame_size_px == IVec2D(y=120, x=180)

    cam.gpu_bin_y = 2
    cam.gpu_bin_x = 3
    cam.gpu_pre_filter = "gaussian"
    cam.gpu_filter_size = 3
    assert cam.has_gpu_binning
    assert cam.output_frame_size_px == IVec2D(y=60, x=60)

    ctrl = CameraController(cam)
    published: list[tuple[str, bytes]] = []

    async def _capture_publish(topic: str, data: bytes) -> None:
        published.append((topic, data))

    ctrl.publish = _capture_publish  # type: ignore[method-assign]
    try:
        # Test pre-armed two-phase snapshot flow (used by Instrument.snapshot to minimize laser-on time)
        await ctrl.arm_capture(num_frames=1)
        assert cam.started_frame_counts == [1]

        count = await ctrl.capture_frames_raw(1, slot_key="snapshot")
        assert count == 1
        assert cam.stop_count == 1

        snap_tif = tmp_path / "snap_cam1.tif"
        info = await ctrl.process_snapshot(
            slot_key="snapshot",
            save_path=str(snap_tif),
            pixel_size_um=(0.39, 0.879),
            metadata={"profile": "daq_dual", "camera": "cam_1"},
        )
        assert info.camera_id == "cam_1"
        assert (info.height, info.width) == (60, 60)
        assert info.gpu_bin_y == 2
        assert info.gpu_bin_x == 3
        assert info.gpu_pre_filter == "gaussian"
        assert info.saved_path == str(snap_tif)
        assert snap_tif.exists()
        assert any(topic == "preview" for topic, _ in published)

        # Also test multi-frame SNR collection + GPU binning analysis via CameraController
        sig_count = await ctrl.collect_frames_to_buffer(8, slot_key="signal", emit_last_preview=True)
        assert sig_count == 8
        dark_count = await ctrl.collect_frames_to_buffer(4, slot_key="dark", emit_last_preview=False)
        assert dark_count == 4

        snr_analysis = await ctrl.analyze_snr_buffer(
            output_dir=str(tmp_path / "ctrl_snr"),
            channel_id="ch_488",
            profile_id="daq_dual",
            pixel_size_um=(0.195, 0.293),
            exposure_ms=20.0,
        )
        assert snr_analysis.camera_id == "cam_1"
        assert (snr_analysis.height, snr_analysis.width) == (60, 60)
        assert snr_analysis.num_signal_frames == 8
        assert snr_analysis.num_dark_frames == 4
        assert snr_analysis.gpu_bin_y == 2
        assert snr_analysis.gpu_bin_x == 3
    finally:
        await ctrl.close()


# ── 3. Stationary Multi-Frame SNR & Photobleaching Analysis + HTML Report ───────


def test_stationary_snr_and_photobleaching_diagnostics_and_html_report(tmp_path: Path) -> None:
    """Verify photobleaching decay recovery, bleaching-detrended noise vs raw std, and HTML report output."""
    rng = np.random.default_rng(2026)
    num_frames = 32
    dark_count = 8
    h, w = 64, 80
    offset = 100.0
    read_noise = 2.5

    # Dark stack: baseline + read noise
    dark_stack = rng.normal(loc=offset, scale=read_noise, size=(dark_count, h, w)).clip(0, 65535).astype(np.uint16)

    # Spatial phantom: bright structured foreground center + dark background border
    yy, xx = np.mgrid[:h, :w]
    fg_region = (yy >= 12) & (yy < 52) & (xx >= 16) & (xx < 64)
    fine_stripes = 1.0 + 0.25 * np.sin(xx * 0.6) * np.cos(yy * 0.6)
    signal_t0 = np.where(fg_region, 1600.0 * fine_stripes, 0.0)

    # Exponential photobleaching: 1.0% decay per frame (k ≈ 0.01005 / frame)
    true_loss_per_frame = 0.01
    t_idx = np.arange(num_frames, dtype=np.float64)[:, None, None]
    decay_curve = (1.0 - true_loss_per_frame) ** t_idx
    expected_signal = signal_t0[None, :, :] * decay_curve

    # Shot noise (gain = 0.5 e-/ADU -> variance in ADU = 2.0 * signal_ADU) + read noise
    shot_std = np.sqrt(np.maximum(expected_signal * 2.0, 0.0))
    light_stack = (
        (offset + expected_signal + rng.normal(0.0, np.sqrt(shot_std**2 + read_noise**2)))
        .clip(0, 65535)
        .astype(np.uint16)
    )

    out_dir = tmp_path / "snr_diag"
    analysis = compute_camera_snr_diagnostics(
        light_stack,
        dark_frames=dark_stack,
        camera_id="cam_1",
        channel_id="ch_488",
        profile_id="daq_dual",
        valid_bits=16,
        exposure_time_ms=25.0,
        pixel_size_um=(0.195, 0.195),
        hardware_binning=1,
        gpu_bin_y=1,
        gpu_bin_x=1,
        gpu_pre_filter="none",
        gpu_filter_size=3,
        output_dir=out_dir,
    )

    # 1. Dark floor & read noise recovery
    assert analysis.dark_offset_adu == pytest.approx(offset, abs=1.0)
    assert analysis.read_noise_adu == pytest.approx(read_noise, abs=0.4)

    # 2. Photobleaching recovery (~1.0% per frame, ~27% total loss over 32 frames)
    assert analysis.bleach_rate_per_frame_pct == pytest.approx(1.0, abs=0.12)
    assert analysis.bleach_total_pct == pytest.approx(
        100.0 * (1.0 - (1.0 - true_loss_per_frame) ** (num_frames - 1)),
        abs=2.5,
    )
    assert analysis.bleach_fit_r2 > 0.98
    assert analysis.bleach_half_life_frames is not None
    assert 55.0 < analysis.bleach_half_life_frames < 80.0

    # 3. Bleaching-detrended SNR is significantly higher than raw SNR (which is inflated by bleaching drift)
    assert analysis.snr_fg_median_corrected > analysis.snr_fg_median_raw * 1.15
    assert analysis.cnr > 5.0
    assert analysis.sbr > 10.0
    assert analysis.sharp_in_focus_area_pct > 5.0

    # 4. Saved TIFF projections exist
    assert len(analysis.saved_tiffs) == 4
    for tif_path in analysis.saved_tiffs.values():
        assert Path(tif_path).exists()

    # 5. Render HTML report & verify self-contained HTML + JSON files
    from datetime import datetime, timezone

    report_path = render_snr_html_report(
        report_id="snr_20261117_150000_daq_dual",
        timestamp=datetime(2026, 11, 17, 15, 0, 0, tzinfo=timezone.utc),
        instrument_name="mobi",
        profile_id="daq_dual",
        stage_xyz_um=(1200.0, -450.0, 310.5),
        laser_summary={"laser_488": {"wavelength": 488, "power_setpoint_mw": 50.0}},
        cameras={"cam_1": analysis},
        output_dir=out_dir,
    )
    assert report_path.exists()
    html_text = report_path.read_text(encoding="utf-8")

    assert "<!doctype html>" in html_text.lower()
    assert "Photobleaching" in html_text
    assert "data:image/png;base64," in html_text


# ── 4. Smart Downsampling & Noise-Bounded Compression Options ───────────────────


@pytest.mark.parametrize("bin_mode", ["mean", "median", "sum", "max", "nearest"])
@pytest.mark.parametrize("pre_filter", ["none", "mean", "gaussian", "median", "highpass"])
def test_smart_downsampling_bin_modes_and_highpass_cpu_gpu_parity(bin_mode: str, pre_filter: str) -> None:
    """Verify CPU/GPU parity across all BinningModes (mean, median, sum, max, nearest) and PreFilterTypes."""
    from ome_zarr_writer import (
        BinningMode,
        PreFilterType,
        cutoff_to_sigma_px,
        filter_and_bin_volume,
        is_gpu_available,
    )

    sigma = cutoff_to_sigma_px(2.0, voxel_size_um=0.5, units="um")
    assert sigma == pytest.approx(0.1873906 * 4.0, rel=1e-4)

    rng = np.random.default_rng(123)
    vol = rng.integers(100, 4000, size=(4, 24, 36), dtype=np.uint16)

    cpu_out = filter_and_bin_volume(
        vol,
        bin_y=2,
        bin_x=3,
        pre_filter=PreFilterType(pre_filter),
        filter_size=3,
        bin_mode=BinningMode(bin_mode),
        highpass_add=100.0,
        use_gpu=False,
    )
    assert cpu_out.shape == (4, 12, 12)
    assert cpu_out.dtype == np.uint16

    if is_gpu_available():
        gpu_out = filter_and_bin_volume(
            vol,
            bin_y=2,
            bin_x=3,
            pre_filter=PreFilterType(pre_filter),
            filter_size=3,
            bin_mode=BinningMode(bin_mode),
            highpass_add=100.0,
            use_gpu=True,
        )
        np.testing.assert_allclose(gpu_out.astype(np.int32), cpu_out.astype(np.int32), atol=1)


def test_anscombe_sqrt_and_linear_noise_bounded_quantization(tmp_path: Path) -> None:
    """Verify MAD noise estimation, sqrt (Anscombe) & linear lossy rounding, bin-center translation,
    and end-to-end GPU/CPU OMEZarrWriter with Blosc bitshuffle."""
    from ome_zarr_writer import (
        BinningMode,
        Compression,
        Dtype,
        Local,
        LossyDomain,
        OMEZarrWriter,
        PreFilterType,
        ScaleLevel,
        UIVec3D,
        UVec3D,
        WriterConfig,
        estimate_noise_sigma,
        is_gpu_available,
        quantize_lossy_volume,
    )

    rng = np.random.default_rng(99)
    # Linear domain synthetic volume with known Gaussian noise sigma = 12.0 ADU
    clean_lin = np.full((8, 64, 64), 1000.0, dtype=np.float32)
    noisy_lin = (clean_lin + rng.normal(0.0, 12.0, size=clean_lin.shape)).clip(0, 65535).astype(np.uint16)

    sigma_lin, off_lin = estimate_noise_sigma(noisy_lin, domain=LossyDomain.LINEAR)
    assert off_lin == 0.0
    assert sigma_lin == pytest.approx(12.0, rel=0.1)

    q_lin_cpu = quantize_lossy_volume(
        noisy_lin,
        lossy_step_noise=1.0,
        lossy_domain=LossyDomain.LINEAR,
        noise_sigma=sigma_lin,
        use_gpu=False,
    )
    rms_err_lin = float(np.sqrt(np.mean((q_lin_cpu.astype(np.float32) - noisy_lin.astype(np.float32)) ** 2)))
    # Uniform rounding to step S has theoretical RMS = S / sqrt(12) ≈ 0.2887 * S
    assert rms_err_lin < 0.45 * sigma_lin

    # Anscombe (sqrt) domain test
    sigma_sqrt, off_sqrt = estimate_noise_sigma(noisy_lin, domain=LossyDomain.SQRT)
    assert sigma_sqrt > 0.0
    q_sqrt_cpu = quantize_lossy_volume(
        noisy_lin,
        lossy_step_noise=1.0,
        lossy_domain=LossyDomain.SQRT,
        noise_sigma=sigma_sqrt,
        offset=off_sqrt,
        use_gpu=False,
    )
    if is_gpu_available():
        q_sqrt_gpu = quantize_lossy_volume(
            noisy_lin,
            lossy_step_noise=1.0,
            lossy_domain=LossyDomain.SQRT,
            noise_sigma=sigma_sqrt,
            offset=off_sqrt,
            use_gpu=True,
        )
        np.testing.assert_allclose(q_sqrt_gpu.astype(np.int32), q_sqrt_cpu.astype(np.int32), atol=1)

    # Verify bin-center translation & preprocessing metadata in WriterConfig.dataset
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=64, y=64, x=64),
        voxel_size=UVec3D(z=1.0, y=0.4, x=0.6),
        dtype=Dtype.UINT16,
        max_level=ScaleLevel.L1,
        compression=Compression.BLOSC_LZ4,
        compression_shuffle="bitshuffle",
        bin_y=2,
        bin_x=3,
        bin_mode=BinningMode.MEAN,
        pre_filter=PreFilterType.HIGHPASS,
        filter_size=3,
        lossy_step_noise=1.0,
        lossy_domain=LossyDomain.SQRT,
        use_gpu=is_gpu_available(),
    )
    ds = cfg.dataset
    # L0 bin-center shift: raw_voxel * (bin - 1) / 2 = (voxel_binned / bin) * (bin - 1) / 2
    # y: 0.4 * (2 - 1) / (2 * 2) = 0.1; x: 0.6 * (3 - 1) / (2 * 3) = 0.2
    l0_transforms = ds.group.attributes.ome.multiscales[0].datasets[0].coordinateTransformations
    assert len(l0_transforms) == 1
    assert len(l0_transforms[0].transformations) == 2
    assert l0_transforms[0].transformations[1].translation == pytest.approx([0.0, 0.0, 0.1, 0.2])
    assert ds.group.attributes.preprocessing is not None
    assert ds.group.attributes.preprocessing["bin"] == [1, 2, 3]
    assert ds.group.attributes.preprocessing["bin_mode"] == "mean"
    assert ds.group.attributes.preprocessing["pre_filter"] == "highpass"
    assert ds.group.attributes.preprocessing["lossy_step_noise"] == 1.0
    assert ds.group.attributes.preprocessing["lossy_domain"] == "sqrt"

    # End-to-end write & TensorStore readback
    writer = OMEZarrWriter(slots=2)
    try:
        writer.begin_stack(cfg, Local(target=tmp_path / "smart_vol"))
        for _ in range(64):
            with writer.new_frame() as frame:
                assert frame.shape == (128, 192)
                frame[...] = rng.integers(100, 500, size=(128, 192), dtype=np.uint16)
        writer.end_stack()
    finally:
        writer.close()

    zarr_root = tmp_path / "smart_vol.ome.zarr"
    meta = json.loads((zarr_root / "zarr.json").read_text())
    assert meta["attributes"]["preprocessing"]["bin"] == [1, 2, 3]
    assert (zarr_root / "0" / "c" / "0" / "0" / "0" / "0").exists()


# ── 5. Dark-Frame Subdirectory & Metadata in OME-Zarr ───────────────────────────


async def test_dark_frames_subdirectory_and_metadata_in_ome_zarr(tmp_path: Path) -> None:
    """Verify writing dark/before.ome.zarr and dark/after.ome.zarr inside a scan's .ome.zarr
    and persisting attributes.darkfield in zarr.json."""
    from pathlib import PurePosixPath
    from unittest.mock import MagicMock

    from ome_zarr_writer import (
        Compression,
        Dtype,
        Local,
        OMEZarrWriter,
        ScaleLevel,
        UIVec3D,
        UVec3D,
        WriterConfig,
    )
    from vxl_records import StorageSpec

    from vxl.devices.camera.base import CameraController

    rng = np.random.default_rng(77)
    scan_target = tmp_path / "run" / "volumes" / "000001" / "single_gfp" / "gfp"
    dark_before_target = Path(f"{scan_target}.ome.zarr") / "dark" / "before"
    dark_after_target = Path(f"{scan_target}.ome.zarr") / "dark" / "after"

    dark_cfg = WriterConfig(
        volume_shape=UIVec3D(z=64, y=32, x=32),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        dtype=Dtype.UINT16,
        max_level=ScaleLevel.L0,
        compression=Compression.BLOSC_ZSTD,
        dark_floor=0,
        skip_empty_chunks=False,
    )
    scan_cfg = WriterConfig(
        volume_shape=UIVec3D(z=128, y=32, x=32),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        dtype=Dtype.UINT16,
        max_level=ScaleLevel.L0,
        compression=Compression.BLOSC_ZSTD,
        darkfield={
            "enabled": True,
            "frames": 64,
            "lasers_off": True,
            "before": "dark/before.ome.zarr",
            "after": None,
        },
    )

    writer = OMEZarrWriter(slots=2)
    try:
        # 1. Pre-scan dark frames in <scan>.ome.zarr/dark/before.ome.zarr
        writer.begin_stack(dark_cfg, Local(target=dark_before_target))
        for _ in range(64):
            with writer.new_frame() as frame:
                frame[...] = rng.integers(95, 108, size=(32, 32), dtype=np.uint16)
        writer.end_stack()

        # 2. Main scan stack in <scan>.ome.zarr (reusing writer ring)
        writer.begin_stack(scan_cfg, Local(target=scan_target))
        for _ in range(128):
            with writer.new_frame() as frame:
                frame[...] = rng.integers(100, 1500, size=(32, 32), dtype=np.uint16)
        writer.end_stack()

        # 3. Post-scan dark frames in <scan>.ome.zarr/dark/after.ome.zarr
        writer.begin_stack(dark_cfg, Local(target=dark_after_target))
        for _ in range(64):
            with writer.new_frame() as frame:
                frame[...] = rng.integers(96, 109, size=(32, 32), dtype=np.uint16)
        writer.end_stack()
    finally:
        writer.close()

    zarr_root = Path(f"{scan_target}.ome.zarr")
    assert (zarr_root / "dark" / "before.ome.zarr" / "zarr.json").is_file()
    assert (zarr_root / "dark" / "after.ome.zarr" / "zarr.json").is_file()

    # Update root zarr.json attributes.darkfield via CameraController.update_dataset_darkfield
    ctrl = MagicMock(spec=CameraController)
    ctrl._system = MagicMock()
    ctrl._system.resolve_storage.return_value = Local(target=scan_target)

    async def _run_sync(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    ctrl._run_sync = _run_sync
    updated_meta = {
        "enabled": True,
        "frames": 64,
        "lasers_off": True,
        "before": "dark/before.ome.zarr",
        "after": "dark/after.ome.zarr",
        "task_scan_duration_s": 725.4,
    }
    await CameraController.update_dataset_darkfield(
        ctrl,
        storage=StorageSpec(path=PurePosixPath("run")),
        subpath=PurePosixPath("volumes/000001/single_gfp/gfp"),
        darkfield=updated_meta,
    )

    root_meta = json.loads((zarr_root / "zarr.json").read_text(encoding="utf-8"))
    assert root_meta["attributes"]["darkfield"] == updated_meta
