"""Tests for 2D ImageJ TIFF writing and stationary SNR / photobleaching diagnostics."""

import datetime
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from vxl._utils.tiff import write_tiff_2d
from vxl.instrument import IdleActivity, Instrument
from vxl.instrument.snr import compute_camera_snr_diagnostics, read_snr_report_html, render_snr_html_report


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_write_tiff_2d_imagej_roundtrip(tmp_path: Path, dtype: type[np.unsignedinteger] | type[np.floating]) -> None:
    rng = np.random.default_rng(7)
    if dtype is np.float32:
        img = rng.uniform(0.0, 100.0, size=(32, 48)).astype(np.float32)
    else:
        img = rng.integers(0, 200, size=(32, 48), dtype=dtype)

    out = write_tiff_2d(
        tmp_path / "nested" / f"snap_{dtype.__name__}.tif",
        img,
        pixel_size_um=(0.5, 0.25),
        metadata={"camera_id": "cam_0", "exposure_time_ms": 12.5},
    )
    assert out.is_file()
    with tifffile.TiffFile(out) as tf:
        np.testing.assert_array_equal(tf.asarray(), img)
        assert tf.is_imagej
        meta = tf.imagej_metadata
        assert meta is not None
        assert meta["unit"] == "um"
        info = json.loads(meta["Info"])
        assert info == {"camera_id": "cam_0", "exposure_time_ms": 12.5}
        x_res, y_res = tf.pages[0].resolution
        assert x_res == pytest.approx(4.0, rel=1e-4)  # 1 / 0.25 um
        assert y_res == pytest.approx(2.0, rel=1e-4)  # 1 / 0.50 um


def test_write_tiff_2d_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2D"):
        write_tiff_2d(tmp_path / "bad3d.tif", np.zeros((2, 4, 4), dtype=np.uint16))
    with pytest.raises(TypeError, match="Unsupported TIFF dtype"):
        write_tiff_2d(tmp_path / "bad_dtype.tif", np.zeros((4, 4), dtype=np.int64))


def test_stationary_snr_photobleaching_and_html_report(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n_sig, n_dark, h, w = 24, 6, 64, 64
    yy, xx = np.ogrid[:h, :w]
    fg_signal = np.where(((yy - 32) ** 2 + (xx - 32) ** 2) <= 18**2, 2000.0, 0.0)

    # Strong monotonic photobleaching (~30% decay across stack) + shot/read noise
    frames = np.empty((n_sig, h, w), dtype=np.uint16)
    for t in range(n_sig):
        decay = float(np.exp(-0.016 * t))
        mean_t = 100.0 + fg_signal * decay
        noisy = mean_t + rng.normal(0.0, np.sqrt(np.maximum(mean_t - 100.0, 0.0) * 0.5 + 9.0))
        frames[t] = np.clip(noisy, 0, 65535).astype(np.uint16)

    dark = np.clip(100.0 + rng.normal(0.0, 3.0, size=(n_dark, h, w)), 0, 65535).astype(np.uint16)

    analysis = compute_camera_snr_diagnostics(
        frames,
        dark_frames=dark,
        camera_id="cam_a",
        channel_id="gfp",
        profile_id="488",
        valid_bits=16,
        exposure_time_ms=20.0,
        pixel_size_um=(0.4, 0.4),
        gpu_bin_y=2,
        gpu_bin_x=1,
        output_dir=tmp_path,
    )

    assert analysis.num_signal_frames == n_sig
    assert analysis.num_dark_frames == n_dark
    assert analysis.dark_offset_adu == pytest.approx(100.0, abs=3.0)
    assert analysis.read_noise_adu == pytest.approx(3.0, abs=1.0)
    # Allan-deviation detrending removes bleaching drift -> corrected SNR exceeds raw SNR
    assert analysis.snr_fg_mean_corrected > analysis.snr_fg_mean_raw
    assert analysis.bleach_total_pct > 20.0
    assert analysis.bleach_half_life_frames is not None
    assert analysis.bleach_fit_r2 > 0.9
    assert len(analysis.saved_tiffs) == 4
    assert all(Path(p).is_file() for p in analysis.saved_tiffs.values())
    assert set(analysis.previews) == {"mean", "std_detrended", "snr_corrected", "bleach_map", "masks", "sharpness"}
    assert analysis.without_previews().previews == {}

    report_path = render_snr_html_report(
        report_id="snr_test",
        timestamp=datetime.datetime(2026, 11, 21, 12, 0, 0, tzinfo=datetime.UTC),
        instrument_name="sim",
        profile_id="488",
        stage_xyz_um=(10.0, 20.0, 30.0),
        laser_summary={"488": {"wavelength": 488, "power_setpoint_mw": 50}},
        cameras={"cam_a": analysis},
        output_dir=tmp_path / "snr_test",
    )
    assert report_path.is_file()
    html_text = read_snr_report_html(tmp_path, "snr_test")
    assert "Stationary SNR, Photobleaching" in html_text
    assert "data:image/png;base64," in html_text

    metrics = json.loads((tmp_path / "snr_test" / "snr_metrics.json").read_text(encoding="utf-8"))
    assert "previews" not in metrics["cameras"]["cam_a"]

    with pytest.raises(FileNotFoundError):
        read_snr_report_html(tmp_path, "../escape")


async def test_instrument_snapshot_and_snr_end_to_end(opened_instrument: Instrument) -> None:
    snap = await opened_instrument.snapshot(save_tiff=True)
    assert isinstance(opened_instrument.activity.value, IdleActivity)
    assert snap.frames
    for frame in snap.frames.values():
        assert frame.saved_path is not None
        assert Path(frame.saved_path).is_file()

    report = await opened_instrument.run_snr_diagnostics(
        num_frames=4,
        dark_frames=2,
        report_url_for=lambda rid: f"/api/reports/{rid}",
    )
    assert report.report_url == f"/api/reports/{report.report_id}"
    assert Path(report.report_path).is_file()
    assert Path(report.metrics_path).is_file()
    assert all(cam.previews == {} for cam in report.cameras.values())
    assert "Stationary SNR, Photobleaching" in opened_instrument.get_snr_report_html(report.report_id)
