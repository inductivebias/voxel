import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import tifffile
from ome_zarr_writer.writer import BatchMetrics
from pydantic import ValidationError
from vxl_records import StorageSpec

from vxl.devices.camera import SensorROI
from vxl.devices.camera.base import CameraController
from vxl.devices.camera.simulated.simulated import SimulatedCamera
from vxl.system import System


def test_sensor_roi_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SensorROI.model_validate({"x": 0, "y": 0, "w": 128, "h": 128, "width": 128})


def test_writer_telemetry_excludes_pending_batches_and_bounds_history() -> None:

    batches = BatchMetrics.create_many(12000, 1)
    controller = cast("CameraController", SimpleNamespace(_writer=SimpleNamespace(batches=batches)))
    getter = CameraController.writer_metrics.fget
    assert getter is not None
    assert getter(controller) == []
    for batch in batches[:100]:
        batch.collecting.begin()
    assert [batch.batch_idx for batch in getter(controller)] == list(range(36, 100))
    assert len(batches) > 12000  # live filtering must not truncate the persisted history


async def test_gpu_binning_snapshot_and_snr_pipeline(tmp_path: Path) -> None:
    cam = SimulatedCamera("cam_0", frame_source={"sensor_size_px": "64,64"}, start_delay_s=0.0, stop_delay_s=0.0)
    cam.gpu_bin_y = 2
    cam.gpu_bin_x = 4
    cam.gpu_pre_filter = "mean"
    cam.gpu_filter_size = 3
    cam.gpu_bin_mode = "mean"
    assert cam.has_gpu_binning
    assert (cam.output_frame_size_px.y, cam.output_frame_size_px.x) == (32, 16)

    controller = CameraController(cam, system=System(store=tmp_path))
    try:
        # 0-frame disarm safely stops/frees buffer without touching _diag_buffers
        await controller.arm_capture(1)
        assert await controller.capture_frames_raw(0, slot_key="disarmed") == 0
        assert "disarmed" not in controller._diag_buffers
        assert not cam._buffer_allocated

        # 2-phase snapshot: capture raw -> process (GPU bin + TIFF + preview)
        await controller.arm_capture(1)
        assert await controller.capture_frames_raw(1, slot_key="snapshot") == 1
        tif_path = tmp_path / "snap.tif"
        info = await controller.process_snapshot(
            slot_key="snapshot",
            save_path=str(tif_path),
            pixel_size_um=(0.5, 0.5),
            metadata={"profile_id": "488"},
        )
        assert (info.height, info.width) == (32, 16)
        assert (info.gpu_bin_y, info.gpu_bin_x) == (2, 4)
        assert np.all(np.isfinite([info.mean_adu, info.std_adu]))
        assert info.saved_path == str(tif_path)
        with tifffile.TiffFile(tif_path) as tf:
            assert tf.asarray().shape == (32, 16)
            x_res, y_res = tf.pages[0].resolution
            assert x_res == pytest.approx(1.0 / (0.5 * 4), rel=1e-4)
            assert y_res == pytest.approx(1.0 / (0.5 * 2), rel=1e-4)

        # 2-phase SNR: capture raw signal + dark -> analyze once with rectangular GPU binning
        await controller.arm_capture(6)
        assert await controller.capture_frames_raw(6, slot_key="signal") == 6
        await controller.arm_capture(3)
        assert await controller.capture_frames_raw(3, slot_key="dark") == 3
        analysis = await controller.analyze_snr_buffer(
            output_dir=str(tmp_path / "snr"),
            channel_id="ch0",
            profile_id="488",
            pixel_size_um=(0.5, 0.5),
            exposure_ms=10.0,
        )
        assert (analysis.height, analysis.width) == (32, 16)
        assert analysis.num_signal_frames == 6
        assert analysis.num_dark_frames == 3
        assert analysis.pixel_size_um == (1.0, 2.0)

        # update_dataset_darkfield patches root zarr.json attributes
        zarr_dir = tmp_path / "run" / "ch0.ome.zarr"
        zarr_dir.mkdir(parents=True)
        (zarr_dir / "zarr.json").write_text(json.dumps({"zarr_format": 3, "attributes": {}}), encoding="utf-8")
        await controller.update_dataset_darkfield(
            storage=StorageSpec(path=PurePosixPath("run")),
            subpath=PurePosixPath("ch0"),
            darkfield={"enabled": True, "frames": 8},
        )
        updated = json.loads((zarr_dir / "zarr.json").read_text(encoding="utf-8"))
        assert updated["attributes"]["darkfield"] == {"enabled": True, "frames": 8}
    finally:
        await controller.close()
        assert not controller._diag_buffers
