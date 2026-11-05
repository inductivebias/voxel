from types import SimpleNamespace
from typing import cast

import pytest
from ome_zarr_writer.writer import BatchMetrics
from pydantic import ValidationError

from vxl.devices.camera import SensorROI
from vxl.devices.camera.base import CameraController


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
