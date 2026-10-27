import pytest

from vxl.devices.axes.continuous import (
    ContinuousAxisController,
    ContinuousScanInfo,
    ScanMode,
    StepMode,
    TTLStepperConfig,
)
from vxl.devices.axes.simulated import SimulatedContinuousAxis

RELATIVE = TTLStepperConfig(step_mode=StepMode.RELATIVE)


def scanning_axis() -> SimulatedContinuousAxis:
    return SimulatedContinuousAxis("x_axis", speed=100.0, scan_edge_spacing=20.0, scan_max_speed=50_000.0)


async def test_controller_continuous_scan_runs_prepare_start_finish_and_returns_to_idle() -> None:
    axis = scanning_axis()
    controller = ContinuousAxisController(axis)
    assert controller.continuous_scan_info == ContinuousScanInfo(
        edge_spacing=20.0, acceleration=20_000.0, max_speed=50_000.0
    )

    for command, args in (
        (controller.start_continuous_scan, (100.0,)),
        (controller.continuous_scan_moving, ()),
        (controller.finish_continuous_scan, (100.0,)),
    ):
        with pytest.raises(RuntimeError, match="not prepared"):
            await command(*args)

    state = await controller.prepare_continuous_scan(0.0, 50_000.0)
    assert state == {"position": 0.0, "origin": 0.0, "moving": False}
    assert controller.scan_mode == ScanMode.CONTINUOUS
    with pytest.raises(RuntimeError, match="while in CONTINUOUS mode"):
        await controller.prepare_continuous_scan(0.0, 50_000.0)

    await controller.start_continuous_scan(100.0)
    axis.await_movement(timeout_s=2.0)
    assert not await controller.continuous_scan_moving()
    assert await controller.finish_continuous_scan(100.0) == {"position": 100.0, "origin": 100.0, "moving": False}
    assert controller.scan_mode == ScanMode.IDLE
    assert axis.speed == 100.0, "finish restores the speed the scan replaced"


async def test_controller_failed_prepare_or_finish_keeps_a_consistent_mode() -> None:
    axis = scanning_axis()
    controller = ContinuousAxisController(axis)

    with pytest.raises(RuntimeError, match="not the scan start"):
        await controller.prepare_continuous_scan(10.0, 1_000.0)
    assert controller.scan_mode == ScanMode.IDLE, "a refused prepare must not claim the axis"
    assert axis.speed == 100.0

    await controller.prepare_continuous_scan(0.0, 1_000.0)
    with pytest.raises(RuntimeError, match="not the scan target"):
        await controller.finish_continuous_scan(100.0)
    assert controller.scan_mode == ScanMode.CONTINUOUS, "an unsettled finish leaves the scan for reset"
    await controller.reset_continuous_scan()
    assert controller.scan_mode == ScanMode.IDLE
    assert axis.speed == 100.0


async def test_controller_continuous_scan_and_ttl_stepping_exclude_each_other() -> None:
    controller = ContinuousAxisController(scanning_axis())

    await controller.configure_ttl_stepper(RELATIVE)
    with pytest.raises(RuntimeError, match="while in TTL_STEPPING mode"):
        await controller.prepare_continuous_scan(0.0, 1_000.0)
    with pytest.raises(RuntimeError, match="while TTL stepping"):
        await controller.reset_continuous_scan()
    with pytest.raises(RuntimeError, match="not prepared"):
        await controller.start_continuous_scan(100.0)
    assert controller.scan_mode == ScanMode.TTL_STEPPING
    await controller.reset_ttl_stepper()

    await controller.prepare_continuous_scan(0.0, 1_000.0)
    with pytest.raises(RuntimeError, match="while in CONTINUOUS mode"):
        await controller.configure_ttl_stepper(RELATIVE)
    with pytest.raises(RuntimeError, match="not configured"):
        await controller.queue_relative_move(1.0)
    with pytest.raises(RuntimeError, match="during a continuous scan"):
        await controller.reset_ttl_stepper()
    assert controller.scan_mode == ScanMode.CONTINUOUS


async def test_controller_reset_continuous_scan_is_idempotent_and_returns_to_idle_after_driver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis = scanning_axis()
    controller = ContinuousAxisController(axis)
    await controller.reset_continuous_scan()
    assert controller.scan_mode == ScanMode.IDLE

    await controller.prepare_continuous_scan(0.0, 1_000.0)
    await controller.reset_continuous_scan()
    await controller.reset_continuous_scan()
    assert controller.scan_mode == ScanMode.IDLE
    assert axis.speed == 100.0

    scanner = axis.get_continuous_scanner()
    assert scanner is not None

    def fail() -> None:
        raise RuntimeError("encoder output did not turn off")

    await controller.prepare_continuous_scan(0.0, 1_000.0)
    monkeypatch.setattr(scanner, "reset", fail)
    with pytest.raises(RuntimeError, match="encoder output did not turn off"):
        await controller.reset_continuous_scan()
    assert controller.scan_mode == ScanMode.IDLE


async def test_controller_continuous_scan_without_scanner_names_the_axis() -> None:
    controller = ContinuousAxisController(SimulatedContinuousAxis("plain_axis"))
    assert controller.continuous_scan_info is None

    for command, args in (
        (controller.prepare_continuous_scan, (0.0, 1_000.0)),
        (controller.start_continuous_scan, (100.0,)),
        (controller.continuous_scan_moving, ()),
        (controller.finish_continuous_scan, (100.0,)),
    ):
        with pytest.raises(NotImplementedError, match="plain_axis does not support continuous scanning"):
            await command(*args)
    await controller.reset_continuous_scan()
    assert controller.scan_mode == ScanMode.IDLE
