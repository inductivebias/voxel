import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from typing import cast

import numpy as np
import pytest
from vxl_drivers.cameras.teledyne import Kinetix

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.axes.discrete.handle import DiscreteAxisHandle
from vxl.devices.camera.base import PIXEL_FMT_TO_VALID_BITS, PixelFormat, SensorROI, TriggerMode
from vxl.devices.camera.handle import CameraHandle
from vxl.preview import PreviewFrame

SERIAL_VAR = "VOXEL_KINETIX_SERIAL"
WHEEL_SERIAL_VAR = "VOXEL_FW212C_SERIAL"

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not os.getenv(SERIAL_VAR), reason=f"set {SERIAL_VAR} to select the camera"),
]


def _serial() -> str:
    return os.environ[SERIAL_VAR]


async def _wait_for(condition: Callable[[], Awaitable[bool]], *, timeout_s: float, poll_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await condition():
            return True
        await asyncio.sleep(poll_s)
    return False


def test_kinetix_hardware_communicates_identity_and_capabilities_then_closes() -> None:
    """Open, read identity and capability tables, close; no acquisition."""
    camera = Kinetix("kinetix_hardware_test", _serial())
    try:
        assert camera.serial == _serial()
        assert camera.sensor_size_px.x > 0
        assert camera.sensor_size_px.y > 0
        assert camera.pixel_size_um.x > 0
        assert camera.pixel_size_um.y > 0
        assert camera.stream_info is None
        modes = list(camera.sensor_mode.options)
        assert modes, "PVCAM reported no readout ports"
        grid = camera.roi_grid
        print(
            f"Kinetix identity={camera.identity!r}\n"
            f"  sensor={camera.sensor_size_px} pixel_size_um={camera.pixel_size_um}\n"
            f"  sensor_modes={modes} current={camera.sensor_mode!r} gains={list(camera.gain.options or ())}\n"
            f"  pixel_format={camera.pixel_format} binning_options={list(camera.binning.options or ())}\n"
            f"  exposure_ms=[{camera.exposure_time_ms.minimum}, {camera.exposure_time_ms.maximum}]\n"
            f"  expose_out_modes={list(camera.expose_out_mode.options)} current={camera.expose_out_mode!r}\n"
            f"  roi_grid={grid.model_dump()} temperature_c={camera.sensor_temperature_c}"
        )
    finally:
        camera.close()


@pytest.mark.skipif(
    os.getenv("VOXEL_KINETIX_ALLOW_ACQUISITION") != "YES",
    reason="set VOXEL_KINETIX_ALLOW_ACQUISITION=YES to acquire frames with the internal trigger",
)
def test_kinetix_hardware_acquires_internally_triggered_frames_and_restarts_on_setting_change() -> None:
    """Short internally triggered acquisition, then a live exposure and binning change, then an ROI."""
    camera = Kinetix("kinetix_acquisition_test", _serial())
    try:
        camera.exposure_time_ms = 5.0
        camera.start(frame_count=5, trigger_mode=TriggerMode.OFF)
        started = time.perf_counter()
        frames = [camera.grab_frame_owned() for _ in range(5)]
        elapsed_s = time.perf_counter() - started
        valid_bits = PIXEL_FMT_TO_VALID_BITS[cast("PixelFormat", str(camera.pixel_format))]
        expected_shape = (camera.frame_size_px.y, camera.frame_size_px.x)
        for frame in frames:
            assert frame.shape == expected_shape
            assert frame.dtype == np.dtype(camera.pixel_type.dtype)
            assert int(frame.max()) < (1 << valid_bits), f"samples exceed {valid_bits} valid bits"
        info = camera.stream_info
        assert info is not None
        assert info.frame_index == 5
        print(
            f"Kinetix acquired 5 frames of {expected_shape} {frames[0].dtype} in {elapsed_s:.3f}s "
            f"(reported {info.frame_rate_fps:.1f} fps, dropped {info.dropped_frames}, "
            f"mean {float(frames[-1].mean()):.1f} ADU)"
        )

        camera.exposure_time_ms = 2.0
        camera.binning = 2 if 2 in (camera.binning.options or ()) else 1
        frame = camera.grab_frame_owned()
        assert frame.shape == (camera.frame_size_px.y, camera.frame_size_px.x)
        print(f"After live exposure/binning change: frame {frame.shape} {frame.dtype}")
        camera.stop()

        roi = camera.update_roi(SensorROI(x=0, y=0, w=camera.roi_grid.h.min * 4, h=camera.roi_grid.v.min * 4))
        camera.binning = 1
        camera.start(frame_count=1, trigger_mode=TriggerMode.OFF)
        frame = camera.grab_frame_owned()
        assert frame.shape == (roi.h, roi.w), f"PVCAM applied ROI {frame.shape[::-1]} for requested {roi}"
        print(f"ROI {roi} delivered {frame.shape[::-1]} (x, y)")
        camera.stop()
    finally:
        camera.close()


@pytest.mark.skipif(
    os.getenv("VOXEL_KINETIX_ALLOW_RIG_PREVIEW") != "YES",
    reason="set VOXEL_KINETIX_ALLOW_RIG_PREVIEW=YES to stream a preview through the Voxel handle",
)
def test_kinetix_hardware_streams_preview_through_voxel_camera_handle() -> None:
    """The preview path the web UI uses, without a browser."""
    config = RigConfig(
        devices={
            "camera_1": DeviceConfig(
                target="vxl_drivers.cameras.teledyne.Kinetix",
                init={"serial": _serial(), "preview_trigger_mode": "off"},
            ),
        }
    )

    async def run() -> None:
        rig = Rig(config, name="kinetix-voxel-handle-test")
        await rig.open()
        try:
            assert not rig.build_errors, f"device build failed: {rig.build_errors}"
            camera = CameraHandle.wrap(rig.devices["camera_1"])
            await camera.props.set(exposure_time_ms=5.0)
            packets: list[bytes] = []

            async def receive(packet: bytes) -> None:
                packets.append(packet)

            unsubscribe = camera.subscribe("preview", receive)
            preview_started = False

            async def ten_frames_published() -> bool:
                # rigup serializes the StreamInfo model to a dict.
                info = await camera.props.get_value("stream_info")
                return info is not None and info["frame_index"] >= 10 and bool(packets)

            try:
                assert await camera.start_preview() == "preview"
                preview_started = True
                await _wait_for(ten_frames_published, timeout_s=20.0, poll_s=0.1)
                info = await camera.props.get_value("stream_info")
                assert info is not None, "no stream info while previewing"
                assert packets, "camera frames did not reach the preview topic"
                frame = PreviewFrame.from_packed(packets[-1])
                pixels = frame.decode()
                assert pixels.size > 0
                assert pixels.dtype == np.uint16
                print(f"Voxel preview streamed: {info}; overview={pixels.shape}")
            finally:
                unsubscribe()
                if preview_started:
                    await camera.stop_preview()
        finally:
            await rig.close()

    asyncio.run(run())


@pytest.mark.skipif(
    os.getenv("VOXEL_KINETIX_ALLOW_RIG_PREVIEW") != "YES" or not os.getenv(WHEEL_SERIAL_VAR),
    reason=(f"set VOXEL_KINETIX_ALLOW_RIG_PREVIEW=YES and {WHEEL_SERIAL_VAR} to preview while reading an FW212C"),
)
def test_kinetix_and_fw212c_hardware_stream_and_report_position_in_one_rig() -> None:
    slots = {slot: f"slot-{slot:02d}" for slot in range(12)}
    config = RigConfig(
        devices={
            "camera_1": DeviceConfig(
                target="vxl_drivers.cameras.teledyne.Kinetix",
                init={"serial": _serial(), "preview_trigger_mode": False},
            ),
            "fw_emission": DeviceConfig(
                target="vxl_drivers.axes.fw212c.FW212C",
                init={"serial": os.environ[WHEEL_SERIAL_VAR], "slots": slots},
            ),
        }
    )

    async def run() -> None:
        rig = Rig(config, name="kinetix-fw212c-hardware-test")
        await rig.open()
        try:
            assert not rig.build_errors, f"device build failed: {rig.build_errors}"
            camera = CameraHandle.wrap(rig.devices["camera_1"])
            wheel = DiscreteAxisHandle.wrap(rig.devices["fw_emission"])
            initial_position = (await wheel.state.get()).position
            assert initial_position is not None
            assert 0 <= initial_position < len(slots)

            packets: list[bytes] = []

            async def receive(packet: bytes) -> None:
                packets.append(packet)

            unsubscribe = camera.subscribe("preview", receive)
            preview_started = False
            try:
                assert await camera.start_preview() == "preview"
                preview_started = True

                async def published() -> bool:
                    return bool(packets)

                assert await _wait_for(published, timeout_s=2.0, poll_s=0.01), (
                    "combined rig published no camera preview frame"
                )

                pixels = PreviewFrame.from_packed(packets[-1]).decode()
                assert pixels.size > 0
                assert pixels.dtype == np.uint16
                assert (await wheel.state.get()).position == initial_position
            finally:
                unsubscribe()
                if preview_started:
                    await camera.stop_preview()
        finally:
            await rig.close()

    asyncio.run(asyncio.wait_for(run(), timeout=30.0))
