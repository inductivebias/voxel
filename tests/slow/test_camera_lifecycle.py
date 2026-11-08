import asyncio
from pathlib import Path, PurePosixPath

import pytest
from ome_zarr_writer import (
    Local,
    OmeZarrDataset,
    ScaleLevel,
    Vec3D,
    WriterSettings,
    deskew_dataset,
    deskewed_width,
    shear_xz_from_angle,
)
from vxl_records import LocalLocation, StorageSpec

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


async def test_simulated_camera_stack_writes_v06_metadata_and_deskews(tmp_path: Path) -> None:
    camera = SimulatedCamera(
        "oblique-camera",
        frame_source={"pattern": "gradient_x", "sensor_size_px": "64,64"},
        start_delay_s=0,
        stop_delay_s=0,
    )
    camera.exposure_time_ms = 5
    controller = CameraController(camera, system=System(store=tmp_path, scratch=tmp_path / "scratch"))

    async def wait_state(target_state: CaptureState) -> None:
        async with asyncio.timeout(15):
            while await controller.capture_state() is not target_state:  # noqa: ASYNC110
                await asyncio.sleep(0.01)

    try:
        shear_xz = shear_xz_from_angle(28.0)
        location = await controller.open_stack(
            num_frames=8,
            z_step=0.5,
            magnification=1.0,
            storage=StorageSpec(path=PurePosixPath("oblique_run")),
            subpath=PurePosixPath("volumes/000001/gfp"),
            settings=WriterSettings(max_level=ScaleLevel.L1),
            origin=Vec3D(z=37000.0, y=100.0, x=200.0),
            shear_xz=shear_xz,
            channel_label="GFP",
            channel_color="00FF00",
        )
        assert isinstance(location, LocalLocation)
        await controller.begin_batch(8, trigger_mode=TriggerMode.OFF)
        await wait_state(CaptureState.DONE)
        await controller.close_stack()
        await wait_state(CaptureState.CLOSED)
        await controller.release_writer()
    finally:
        await controller.close()

    raw_path = Path(location.path)
    raw_meta = OmeZarrDataset.read_metadata(raw_path)
    assert raw_meta.group.attributes.ome.version == "0.6"
    assert raw_meta.shear_xz == pytest.approx(shear_xz)
    assert raw_meta.group.attributes.ome.omero is not None
    assert raw_meta.group.attributes.ome.omero.channels[0].label == "GFP"
    assert raw_meta.group.attributes.ome.omero.channels[0].color == "00FF00"

    deskewed_path = deskew_dataset(raw_path, Local(target=tmp_path / "deskewed_gfp"), slots=3)
    deskewed_meta = OmeZarrDataset.read_metadata(deskewed_path)
    expected_shear_factor = shear_xz * raw_meta.voxel_size.z / raw_meta.voxel_size.x
    expected_w = deskewed_width(64, 8, expected_shear_factor)
    assert deskewed_meta.group.attributes.ome.version == "0.6"
    assert deskewed_meta.shear_xz == 0.0
    assert deskewed_meta.volume_shape.x == expected_w
    assert deskewed_meta.volume_shape.y == 64
    assert deskewed_meta.volume_shape.z == 8
