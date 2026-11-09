"""Ordering and failure recovery for continuous acquisition window cleanup."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vxl.instrument.core import Instrument, _PassProgress


@pytest.mark.asyncio
async def test_return_to_start_resets_window_before_moving():
    events = []

    async def reset():
        events.append("reset")

    async def move(*_args, **_kwargs):
        events.append("move")

    axis = SimpleNamespace(reset_continuous_scan=reset, move_abs=move)
    inst = SimpleNamespace(_volume_scan_axis=lambda _v: axis, _hold_violations=AsyncMock(return_value=[]))
    progress = _PassProgress()
    await Instrument._position_volume(inst, SimpleNamespace(scan=object()), SimpleNamespace(start=10), progress)
    assert events == ["reset", "move"]
    assert progress.prepared
    assert not progress.moving


@pytest.mark.asyncio
@pytest.mark.parametrize("finished", [False, True])
@pytest.mark.parametrize("disarm_fails", [False, True])
async def test_teardown_closes_window_first_and_continues_after_failure(monkeypatch, finished, disarm_fails):
    events = []

    async def disarm():
        events.append("window")
        if disarm_fails:
            raise RuntimeError("DO2 readback stayed HIGH")

    async def disable(*_args, **_kwargs):
        events.append("laser/halt")
        return [], []

    async def stop():
        events.append("daq")

    async def abort(*_args):
        events.append("abort/retry")
        return []

    axis = SimpleNamespace(disarm_scan_window=disarm, reset_ttl_stepper=AsyncMock())
    inst = SimpleNamespace(
        _all_lasers=list, _stop_signal_generators=stop, _abort_pass=abort, _close_stacks=AsyncMock(return_value=[])
    )
    monkeypatch.setattr("vxl.instrument.core._disable_lasers_and_halt", disable)
    errors = await Instrument._teardown_capture(inst, [], axis, _PassProgress(prepared=True, finished=finished))
    assert events[:3] == ["window", "laser/halt", "daq"]
    assert ("abort/retry" in events) is not finished
    assert len(errors) == int(disarm_fails)
    inst._close_stacks.assert_awaited_once()
