"""Stage jogging: speed cap, unit conversion, deadband, watchdog, busy lockout, and halt isolation."""

import asyncio

import pytest
from vxlib.reactivity import Cell

from vxl.instrument import CaptureActivity, IdleActivity, Instrument
from vxl.instrument.jog import MAX_JOG_SPEED_UM_S, StageJog


class FakeAxis:
    def __init__(self, units: str = "um") -> None:
        self.units = units
        self.calls: list[tuple[str, float | None]] = []
        self.fail = False
        self.fail_halt = False

    async def get_units(self) -> str:
        return self.units

    async def jog(self, velocity: float) -> None:
        if self.fail:
            raise RuntimeError("axis error")
        self.calls.append(("jog", velocity))

    async def halt(self) -> None:
        if self.fail_halt:
            raise RuntimeError("halt error")
        self.calls.append(("halt", None))


def make_jog(*, z_units: str = "um", watchdog_s: float = 10.0):
    axes = {"x": FakeAxis(), "y": FakeAxis(), "z": FakeAxis(z_units)}
    busy = Cell(False)
    return StageJog(lambda: axes, busy=busy, watchdog_s=watchdog_s), axes, busy  # type: ignore[arg-type]


async def test_scaling_units_deadband_and_zero_stop() -> None:
    jog, axes, _busy = make_jog(z_units="mm")
    await jog.command((0.5, -2.0, 1.0))
    await jog.command((0.505, -2.0, 1.0))  # Within deadband; ignored.
    assert axes["x"].calls == [("jog", 0.5 * MAX_JOG_SPEED_UM_S)]
    assert axes["y"].calls == [("jog", -MAX_JOG_SPEED_UM_S)]
    assert axes["z"].calls == [("jog", MAX_JOG_SPEED_UM_S / 1000)]

    await jog.command((0.0, 0.0, 0.0))
    assert [axis.calls[-1] for axis in axes.values()] == [("jog", 0.0)] * 3


async def test_watchdog_halts_after_silence() -> None:
    jog, axes, _busy = make_jog(watchdog_s=0.05)
    await jog.command((0, 0, -0.25))
    await asyncio.sleep(0.15)
    assert axes["z"].calls == [("jog", -0.25 * MAX_JOG_SPEED_UM_S), ("halt", None)]


async def test_busy_and_axis_error_halt_other_axes() -> None:
    jog, axes, busy = make_jog()
    await jog.command((1, 0, 0))
    axes["y"].fail = axes["y"].fail_halt = True
    await jog.command((1, 1, 0))
    assert axes["x"].calls[-1] == ("halt", None)

    await busy.set(True)
    await jog.command((1, 0, 0))
    assert axes["x"].calls[-1] == ("halt", None)


async def test_simulated_instrument_jogs_and_locks_out_during_capture(opened_instrument: Instrument) -> None:
    instrument = opened_instrument
    start = await instrument._hal.stage.x.get_position()

    await instrument.jog_stage((1.0, 0.0, 0.0))
    await asyncio.sleep(0.2)
    await instrument.jog_stage((0.0, 0.0, 0.0))
    moved = await instrument._hal.stage.x.get_position()
    assert moved > start
    assert not await instrument._hal.stage.x.is_moving()

    await instrument._activity.set(CaptureActivity())
    try:
        await instrument._jog.stop()
        await instrument.jog_stage((1.0, 0.0, 0.0))
        assert await instrument._hal.stage.x.get_position() == pytest.approx(moved)
    finally:
        await instrument._activity.set(IdleActivity())
