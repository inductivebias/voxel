"""Velocity jogging of the stage from a handheld controller, guarded by an enable switch and a dead-man watchdog.

A client (the web UI reading a gamepad) streams stick deflections in ``[-1, 1]`` for the logical ``x``, ``y`` and
``z`` axes. ``StageJog`` scales them to at most ``max_speed_um_s`` and drives each axis with a velocity move.
Motion stops when the sticks are released, updates stop arriving (the watchdog), the driving client disconnects,
or jogging is disabled — by the operator, an axis error, an acquisition starting, or the instrument closing.
"""

import asyncio
import logging
from collections.abc import Callable, Mapping
from typing import Literal

from vxlib.reactivity import Cell, Readable
from vxlib.schema import FrozenModel

from vxl.devices.axes import ContinuousAxisHandle

from .errors import InstrumentBusyError, OperationRejectedError

logger = logging.getLogger(__name__)

type JogAxis = Literal["x", "y", "z"]
JOG_AXES: tuple[JogAxis, ...] = ("x", "y", "z")

#: Full-stick speed.
MAX_JOG_SPEED_UM_S = 2000.0
#: Halt when no update arrives for this long while any axis is jogging.
JOG_WATCHDOG_S = 0.3
#: Re-command a moving axis only when its velocity changes by more than this fraction of full speed.
_RECOMMAND_FRACTION = 0.02
_UM_PER_UNIT = {"nm": 1e-3, "um": 1.0, "mm": 1e3}


class JogState(FrozenModel):
    """Operator-visible jog state, published with the instrument view."""

    available: bool = False
    enabled: bool = False
    active: bool = False
    max_speed_um_s: float = MAX_JOG_SPEED_UM_S
    message: str | None = None


class StageJog:
    """Turns streamed stick deflections into guarded stage velocity moves."""

    def __init__(
        self,
        axes: Callable[[], Mapping[JogAxis, ContinuousAxisHandle]],
        *,
        busy: Readable[bool],
        max_speed_um_s: float = MAX_JOG_SPEED_UM_S,
        watchdog_s: float = JOG_WATCHDOG_S,
    ) -> None:
        self._axes = axes
        self._busy = busy
        self._max_speed = max_speed_um_s
        self._watchdog_s = watchdog_s
        self._state = Cell(JogState(max_speed_um_s=max_speed_um_s))
        self._lock = asyncio.Lock()
        self._um_per_unit: dict[JogAxis, float] = {}
        self._commanded: dict[JogAxis, float] = dict.fromkeys(JOG_AXES, 0.0)  # µm/s last sent
        self._owner: str | None = None
        self._seq: dict[str, int] = {}
        self._watchdog: asyncio.Task[None] | None = None

    @property
    def state(self) -> Readable[JogState]:
        return self._state

    async def open(self) -> None:
        """Discover whether every stage axis can jog, and the axes' units."""
        axes = self._axes()
        for name in JOG_AXES:
            self._um_per_unit[name] = _UM_PER_UNIT[await axes[name].get_units()]
        available = all(await asyncio.gather(*(axes[name].jog_available() for name in JOG_AXES)))
        await self._update(available=available, enabled=False, active=False, message=None)

    async def set_enabled(self, enabled: bool) -> None:
        """Arm or disarm jogging. Arming is refused while a capture owns the hardware."""
        async with self._lock:
            if enabled and not self._state.value.available:
                raise OperationRejectedError("This stage does not support jogging")
            if enabled and self._busy.value:
                raise InstrumentBusyError("Unable to enable jogging while a capture is running")
            if not enabled:
                await self._stop()
            await self._update(enabled=enabled, message=None)
            logger.info("Stage jogging %s", "enabled" if enabled else "disabled")

    async def lockout(self, reason: str) -> None:
        """Stop any jog and disable jogging, e.g. when an acquisition starts or the instrument closes."""
        async with self._lock:
            await self._disable(reason)

    async def release(self, client: str) -> None:
        """Forget a disconnected client; if it was driving, stop and disable jogging."""
        async with self._lock:
            self._seq.pop(client, None)
            if self._owner == client:
                await self._disable("the controlling client disconnected")

    async def command(self, client: str, seq: int, velocity: tuple[float, float, float]) -> None:
        """Apply one stick update: each component is a fraction of full jog speed, clamped to ``[-1, 1]``."""
        async with self._lock:
            if seq <= self._seq.get(client, -1):
                return  # Stale or duplicate update.
            self._seq[client] = seq
            if not self._state.value.enabled or self._busy.value:
                await self._stop()
                return
            if self._owner not in (None, client):
                return  # Another client is driving.

            targets: dict[JogAxis, float] = {
                name: max(-1.0, min(1.0, v)) * self._max_speed for name, v in zip(JOG_AXES, velocity, strict=True)
            }
            changed: list[JogAxis] = [
                name for name in JOG_AXES if self._needs_command(self._commanded[name], targets[name])
            ]
            self._commanded.update({name: targets[name] for name in changed})
            axes = self._axes()
            try:
                await asyncio.gather(*(axes[n].jog(targets[n] / self._um_per_unit[n]) for n in changed))
            except Exception as error:
                await self._disable(f"axis error: {error}")
                return

            moving = any(self._commanded.values())
            self._owner = client if moving else None
            self._reset_watchdog(moving)
            await self._update(active=moving)

    def _needs_command(self, current: float, target: float) -> bool:
        if target == 0 or current == 0:
            return target != current  # Always honour starts and stops.
        return abs(target - current) > _RECOMMAND_FRACTION * self._max_speed

    def _reset_watchdog(self, moving: bool) -> None:
        if self._watchdog is not None and self._watchdog is not asyncio.current_task():
            self._watchdog.cancel()
        self._watchdog = asyncio.create_task(self._expire(), name="stage-jog-watchdog") if moving else None

    async def _expire(self) -> None:
        await asyncio.sleep(self._watchdog_s)
        async with self._lock:
            if self._watchdog is asyncio.current_task():
                logger.warning("Stage jog watchdog expired after %.0f ms without an update", self._watchdog_s * 1e3)
                await self._stop()

    async def _disable(self, reason: str) -> None:
        if self._state.value.enabled:
            logger.warning("Stage jogging disabled: %s", reason)
            await self._update(enabled=False, message=f"Jogging disabled: {reason}")
        await self._stop()

    async def _stop(self) -> None:
        """Halt every axis this jog set moving."""
        self._reset_watchdog(False)
        moving: list[JogAxis] = [name for name in JOG_AXES if self._commanded[name]]
        self._commanded = dict.fromkeys(JOG_AXES, 0.0)
        self._owner = None
        await asyncio.gather(*(self._axes()[name].halt() for name in moving))
        await self._update(active=False)

    async def _update(self, **changes: object) -> None:
        await self._state.set(self._state.value.model_copy(update=changes))


__all__ = ["JOG_AXES", "JOG_WATCHDOG_S", "MAX_JOG_SPEED_UM_S", "JogState", "StageJog"]
