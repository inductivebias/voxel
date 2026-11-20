"""Handheld-controller stage jogging: speed scaling, watchdog, ownership, lockout, and fault isolation."""

import asyncio

import pytest
from vxlib.reactivity import Cell

from vxl.instrument import CaptureActivity, IdleActivity, Instrument
from vxl.instrument.jog import MAX_JOG_SPEED_UM_S, StageJog


class FakeAxis:
    """Records jog and halt calls the way ``ContinuousAxisHandle`` would receive them."""

    def __init__(self, units: str = "um", *, available: bool = True) -> None:
        self.units = units
        self.available = available
        self.calls: list[tuple[str, float | None]] = []
        self.fail = False
        self.fail_halt = False

    async def jog_available(self) -> bool:
        return self.available

    async def get_units(self) -> str:
        return self.units

    async def jog(self, velocity: float) -> None:
        if self.fail:
            raise RuntimeError("controller rejected the move")
        self.calls.append(("jog", velocity))

    async def halt(self) -> None:
        if self.fail_halt:
            raise RuntimeError("controller unreachable on halt")
        self.calls.append(("halt", None))


def make_jog(*, z_units: str = "um", watchdog_s: float = 0.05, available: bool = True):
    axes = {"x": FakeAxis(), "y": FakeAxis(), "z": FakeAxis(z_units, available=available)}
    busy = Cell(False)
    jog = StageJog(lambda: axes, busy=busy, watchdog_s=watchdog_s)  # type: ignore[arg-type]
    return jog, axes, busy


async def test_jog_scales_to_speed_cap_converts_units_and_stops_on_zero() -> None:
    jog, axes, _busy = make_jog(z_units="mm", watchdog_s=10)
    await jog.open()

    await jog.command("ui", (0.5, -2.0, 1.0))  # Out-of-range deflection is clamped to full speed.
    assert axes["x"].calls == [("jog", 0.5 * MAX_JOG_SPEED_UM_S)]
    assert axes["y"].calls == [("jog", -MAX_JOG_SPEED_UM_S)]
    assert axes["z"].calls == [("jog", MAX_JOG_SPEED_UM_S / 1000)]  # Converted to the axis's mm/s.

    await jog.command("ui", (0.0, 0.0, 0.0))  # Stick released: a zero-velocity jog stops each axis.
    assert [axis.calls[-1] for axis in axes.values()] == [("jog", 0.0)] * 3


async def test_small_changes_are_not_resent() -> None:
    jog, axes, _busy = make_jog(watchdog_s=10)
    await jog.open()

    await jog.command("ui", (0.5, 0, 0))
    await jog.command("ui", (0.505, 0, 0))  # Within the re-command tolerance.
    assert axes["x"].calls == [("jog", 0.5 * MAX_JOG_SPEED_UM_S)]
    await jog.stop()
    assert axes["x"].calls[-1] == ("halt", None)


async def test_watchdog_halts_when_updates_stop() -> None:
    jog, axes, _busy = make_jog(watchdog_s=0.05)
    await jog.open()
    await jog.command("ui", (0, 0, -0.25))
    await asyncio.sleep(0.15)
    assert axes["z"].calls == [("jog", -0.25 * MAX_JOG_SPEED_UM_S), ("halt", None)]


async def test_only_the_driving_client_controls_the_stage_and_its_disconnect_halts() -> None:
    jog, axes, _busy = make_jog(watchdog_s=10)
    await jog.open()
    await jog.command("a", (1, 0, 0))
    await jog.command("b", (-1, 0, 0))
    assert axes["x"].calls == [("jog", MAX_JOG_SPEED_UM_S)]

    await jog.release("b")  # A bystander closing its tab changes nothing.
    assert axes["x"].calls == [("jog", MAX_JOG_SPEED_UM_S)]
    await jog.release("a")
    assert axes["x"].calls[-1] == ("halt", None)


async def test_busy_refuses_jog_and_stop_halts_motion() -> None:
    jog, axes, busy = make_jog(watchdog_s=10)
    await jog.open()
    await jog.command("ui", (0, 1, 0))

    await busy.set(True)
    await jog.stop()
    assert axes["y"].calls[-1] == ("halt", None)

    await jog.command("ui", (0, 1, 0))
    assert axes["y"].calls[-1] == ("halt", None)


async def test_axis_error_halts_other_axes_even_if_failed_axis_cannot_halt() -> None:
    jog, axes, _busy = make_jog(watchdog_s=10)
    await jog.open()
    await jog.command("ui", (1, 0, 0))
    axes["y"].fail = True
    axes["y"].fail_halt = True

    await jog.command("ui", (1, 1, 0))
    assert axes["x"].calls[-1] == ("halt", None)


async def test_unsupported_stage_ignores_jog() -> None:
    jog, axes, _busy = make_jog(available=False)
    await jog.open()
    await jog.command("ui", (1, 0, 0))
    assert axes["x"].calls == []


async def test_simulated_instrument_jogs_and_acquisition_locks_out(opened_instrument: Instrument) -> None:
    instrument = opened_instrument
    start = await instrument._hal.stage.x.get_position()

    await instrument.jog_stage("ui", (1.0, 0.0, 0.0))
    await asyncio.sleep(0.2)
    await instrument.jog_stage("ui", (0.0, 0.0, 0.0))
    moved = await instrument._hal.stage.x.get_position()
    assert moved > start
    assert not await instrument._hal.stage.x.is_moving()

    await instrument._activity.set(CaptureActivity())
    try:
        await instrument._jog.stop()
        await instrument.jog_stage("ui", (1.0, 0.0, 0.0))
        assert await instrument._hal.stage.x.get_position() == pytest.approx(moved)
    finally:
        await instrument._activity.set(IdleActivity())
