import asyncio
import math
import os

import pytest
from vxl_drivers.axes.zaber import ZaberHub, ZaberLinearAxis

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.axes.continuous.handle import ContinuousAxisHandle

PORT_VAR = "VOXEL_ZABER_PORT"
HOST_VAR = "VOXEL_ZABER_HOST"
DEVICE_SERIAL_VAR = "VOXEL_ZABER_DEVICE_SERIAL"

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        (not os.getenv(PORT_VAR) and not os.getenv(HOST_VAR)) or not os.getenv(DEVICE_SERIAL_VAR),
        reason=f"set {HOST_VAR} or {PORT_VAR}, plus {DEVICE_SERIAL_VAR}, to read the Zaber stage",
    ),
]


def _connection_args() -> dict[str, str]:
    if host := os.getenv(HOST_VAR):
        return {"host": host}
    return {"port": os.environ[PORT_VAR]}


def _open_hub(uid: str) -> ZaberHub:
    if host := os.getenv(HOST_VAR):
        return ZaberHub(uid, host=host)
    return ZaberHub(uid, port=os.environ[PORT_VAR])


def test_zaber_hardware_reads_all_axes_without_moving_or_changing_settings() -> None:
    hub = _open_hub("zaber_hardware_test")
    axes: list[ZaberLinearAxis] = []
    try:
        serial = int(os.environ[DEVICE_SERIAL_VAR])
        device = hub.device(serial)
        assert device.axis_count > 0

        for axis_number in range(1, device.axis_count + 1):
            axis = ZaberLinearAxis(
                f"zaber_axis_{axis_number}",
                hub=hub,
                device_serial=serial,
                axis_number=axis_number,
            )
            axes.append(axis)
            assert axis.lower_limit < axis.upper_limit
            assert math.isfinite(axis.device_position)
            assert not axis.is_moving
            assert axis.peripheral_name
            assert axis.peripheral_serial_number > 0
            print(
                f"Zaber device={axis.device_name!r} serial={axis.serial_number} "
                f"axis={axis.axis_number} peripheral={axis.peripheral_name!r} "
                f"peripheral_serial={axis.peripheral_serial_number} "
                f"device_position_um={axis.device_position} "
                f"limits_um=[{axis.lower_limit}, {axis.upper_limit}] homed={axis.is_homed} "
                f"within_limits={axis.position_within_limits}"
            )
    finally:
        # Mirror Rigup's current hub-before-dependent teardown order.
        hub.close()
        for axis in axes:
            axis.close()


def test_zaber_hardware_builds_every_axis_through_one_rig_hub_then_closes() -> None:
    serial = int(os.environ[DEVICE_SERIAL_VAR])
    probe = _open_hub("zaber_axis_count")
    try:
        axis_count = probe.device(serial).axis_count
    finally:
        probe.close()
    uids = [f"axis_{number}" for number in range(1, axis_count + 1)]
    devices = {"zaber_controller": DeviceConfig(target="vxl_drivers.axes.zaber.ZaberHub", init=_connection_args())}
    for number, uid in enumerate(uids, start=1):
        devices[uid] = DeviceConfig(
            target="vxl_drivers.axes.zaber.ZaberLinearAxis",
            init={"hub": "zaber_controller", "device_serial": serial, "axis_number": number, "units": "um"},
        )

    async def run() -> None:
        rig = Rig(RigConfig(devices=devices), name="zaber-hardware-test")
        await rig.open()
        try:
            assert not rig.build_errors, f"device build failed: {rig.build_errors}"
            for uid in uids:
                axis = ContinuousAxisHandle.wrap(rig.devices[uid])
                state = await axis.props.get(
                    "device_position",
                    "position_within_limits",
                    "is_homed",
                    "is_moving",
                    "peripheral_serial_number",
                )
                assert math.isfinite(state["device_position"].unwrap().value)
                assert state["is_moving"].unwrap().value is False
                assert state["peripheral_serial_number"].unwrap().value > 0
        finally:
            await rig.close()

    asyncio.run(asyncio.wait_for(run(), timeout=15.0))
