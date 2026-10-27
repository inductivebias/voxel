import asyncio
from pathlib import Path, PurePosixPath

import pytest
from ome_zarr_writer import ScaleLevel, WriterSettings
from vxl_records import StorageSpec

from vxl.devices.camera import CaptureState
from vxl.devices.camera.base import CameraController, TriggerMode
from vxl.devices.camera.simulated import SimulatedCamera
from vxl.system import System

pytestmark = pytest.mark.slow


async def test_close_stack_during_collection_drains_real_writer_and_camera_restarts(tmp_path: Path) -> None:
    camera = SimulatedCamera(
        "cancel-camera",
        frame_source={"pattern": "gradient_x", "sensor_size_px": "64,64"},
        start_delay_s=0,
        stop_delay_s=0,
    )
    camera.exposure_time_ms = 100
    controller = CameraController(camera, system=System(store=tmp_path, scratch=tmp_path / "scratch"))

    async def closed() -> None:
        async with asyncio.timeout(15):
            while await controller.capture_state() is not CaptureState.CLOSED:  # noqa: ASYNC110 - exercise the public polling API
                await asyncio.sleep(0.01)

    try:
        for name, frames in (("cancelled", 100), ("next", 1)):
            await controller.open_stack(
                num_frames=frames,
                z_step=1,
                magnification=1,
                storage=StorageSpec(path=PurePosixPath(name)),
                subpath=PurePosixPath("stack"),
                settings=WriterSettings(max_level=ScaleLevel.L0),
            )
            await controller.begin_batch(frames, trigger_mode=TriggerMode.OFF)
            await asyncio.sleep(0.03)
            await controller.close_stack()
            await controller.close_stack()
            await closed()
            assert camera.stream_info is None, "camera must stop before buffers are freed"
        await controller.release_writer()
    finally:
        await controller.close()
