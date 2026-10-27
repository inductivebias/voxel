import asyncio
import os
import time

import pytest
from vxl_drivers.axes.fw212c import FW212C

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.axes.discrete.handle import DiscreteAxisHandle

SERIAL_VAR = "VOXEL_FW212C_SERIAL"
MOVE_VARS = (SERIAL_VAR, "VOXEL_FW212C_EXPECT_START", "VOXEL_FW212C_MOVE_TARGET")

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not os.getenv(SERIAL_VAR), reason=f"set {SERIAL_VAR} to select the wheel"),
]


def test_fw212c_hardware_connects_reads_identity_and_position_then_closes() -> None:
    """Read device state without moving the wheel or changing persistent settings."""
    serial = os.environ[SERIAL_VAR]
    wheel = FW212C("fw212c_hardware_test", serial, slots={})
    try:
        state = wheel.state
        position = state.position
        assert "FW212C" in wheel.identity.upper()
        assert position is not None
        assert 0 <= position < wheel.slot_count
        assert not state.is_moving
        if expected_position := os.getenv("VOXEL_FW212C_EXPECT_POSITION"):
            assert position == int(expected_position)
        print(f"FW212C identity={wheel.identity!r}, Voxel slot={position}, slot count={wheel.slot_count}")
    finally:
        wheel.close()


@pytest.mark.skipif(
    os.getenv("VOXEL_FW212C_ALLOW_MOVEMENT") != "YES" or not all(map(os.getenv, MOVE_VARS)),
    reason=f"set VOXEL_FW212C_ALLOW_MOVEMENT=YES and {', '.join(MOVE_VARS)} to move the wheel",
)
def test_fw212c_hardware_moves_one_adjacent_slot_and_returns() -> None:
    """Move to one explicitly selected adjacent slot, verify, and return."""
    serial = os.environ[SERIAL_VAR]
    expected_start = int(os.environ["VOXEL_FW212C_EXPECT_START"])
    target = int(os.environ["VOXEL_FW212C_MOVE_TARGET"])
    wheel = FW212C("fw212c_movement_test", serial, slots={})
    target_command_completed = False
    try:
        start = wheel.state.position
        assert start is not None, "the wheel reported no slot; no move sent"
        assert start == expected_start, f"expected Voxel slot {expected_start}, found {start}; no move sent"
        assert abs(target - start) == 1, f"target {target} is not adjacent to starting slot {start}; no move sent"

        started = time.perf_counter()
        wheel.move(target, wait=True, timeout=5.0)
        target_command_completed = True
        outward_duration_s = time.perf_counter() - started
        assert wheel.state.position == target

        started = time.perf_counter()
        wheel.move(start, wait=True, timeout=5.0)
        return_duration_s = time.perf_counter() - started
        assert wheel.state.position == start

        print(
            f"FW212C moved Voxel slot {start} -> {target} in {outward_duration_s:.3f}s "
            f"and returned in {return_duration_s:.3f}s"
        )
    finally:
        try:
            if target_command_completed and wheel.state.position != expected_start:
                wheel.move(expected_start, wait=True, timeout=5.0)
        finally:
            wheel.close()


@pytest.mark.skipif(
    os.getenv("VOXEL_FW212C_ALLOW_RIG_MOVEMENT") != "YES" or not all(map(os.getenv, MOVE_VARS)),
    reason=f"set VOXEL_FW212C_ALLOW_RIG_MOVEMENT=YES and {', '.join(MOVE_VARS)} to move via Voxel",
)
def test_fw212c_hardware_selects_labels_through_voxel_handle() -> None:
    """Exercise the property and label-selection path used by Voxel."""
    serial = os.environ[SERIAL_VAR]
    expected_start = int(os.environ["VOXEL_FW212C_EXPECT_START"])
    target = int(os.environ["VOXEL_FW212C_MOVE_TARGET"])
    slots = {slot: f"slot-{slot:02d}" for slot in range(12)}
    config = RigConfig(
        devices={
            "filter_wheel": DeviceConfig(
                target="vxl_drivers.axes.fw212c.FW212C",
                init={"serial": serial, "slots": slots},
            )
        }
    )

    async def run() -> None:
        rig = Rig(config, name="fw212c-voxel-handle-test")
        await rig.open()
        outward_completed = False
        wheel: DiscreteAxisHandle | None = None
        try:
            assert not rig.build_errors
            wheel = DiscreteAxisHandle.wrap(rig.devices["filter_wheel"])
            initial = await wheel.props.get("slot_count", "labels")
            assert initial["slot_count"].unwrap().value == 12
            assert initial["labels"].unwrap().value == slots
            state = await wheel.state.get()
            assert state.position == expected_start
            assert state.is_moving is False
            assert abs(target - expected_start) == 1

            await wheel.select(slots[target], wait=True, timeout_s=5.0)
            outward_completed = True
            assert (await wheel.state.get()).position == target

            await wheel.select(slots[expected_start], wait=True, timeout_s=5.0)
            assert (await wheel.state.get()).position == expected_start
            outward_completed = False
        finally:
            if outward_completed and wheel is not None:
                await wheel.select(slots[expected_start], wait=True, timeout_s=5.0)
            await rig.close()

    asyncio.run(run())
