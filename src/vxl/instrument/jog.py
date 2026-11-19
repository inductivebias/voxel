"""Velocity jogging of the stage from a handheld controller, guarded by an enable switch and a dead-man watchdog.

A client (the web UI reading a gamepad) streams normalized stick deflections in ``[-1, 1]`` for the logical
``x``, ``y`` and ``z`` axes. ``StageJog`` scales them to at most ``max_speed_um_s`` and drives each axis with a
velocity move. Motion stops when:

- the client sends zeros (stick released) or stops sending (the watchdog expires),
- the client that owns the jog disconnects,
- jogging is disabled, by the operator or because an acquisition starts or the instrument closes.

Jogging is off until the operator enables it and is refused while a capture owns the hardware.
"""

import asyncio
import logging
import math
from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import Field
from vxlib.reactivity import Cell, Readable
from vxlib.schema import FrozenModel

from vxl.devices.axes import ContinuousAxisHandle

from .errors import InstrumentBusyError, OperationRejectedError

logger = logging.getLogger(__name__)

type JogAxis = Literal["x", "y", "z"]
JOG_AXES: tuple[JogAxis, ...] = ("x", "y", "z")

#: Hard ceiling on jog speed, whatever a caller asks for.
MAX_JOG_SPEED_UM_S = 2000.0
#: Halt when no update arrives for this long while any axis is jogging.
JOG_WATCHDOG_S = 0.3
#: Re-command an axis only when its velocity changes by more than this fraction of full speed.
_RECOMMAND_FRACTION = 0.02
_UM_PER_UNIT = {"nm": 1e-3, "um": 1.0, "mm": 1e3}


class JogState(FrozenModel):
    """Operator-visible jog state, published with the instrument view."""

    available: bool = False
    enabled: bool = False
    active: bool = False
    max_speed_um_s: float = Field(default=MAX_JOG_SPEED_UM_S, gt=0)
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
        if not (0 < max_speed_um_s <= MAX_JOG_SPEED_UM_S):
            raise ValueError(f"max_speed_um_s must be in (0, {MAX_JOG_SPEED_UM_S:g}], got {max_speed_um_s}")
        if watchdog_s <= 0:
            raise ValueError("watchdog_s must be positive")
        self._axes = axes
        self._busy = busy
        self._max_speed = max_speed_um_s
        self._watchdog_s = watchdog_s
        self._state = Cell(JogState(max_speed_um_s=max_speed_um_s))
        self._lock = asyncio.Lock()
        self._units: dict[JogAxis, float] = {}  # µm per axis unit
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
        try:
            available = all(await asyncio.gather(*(axes[name].jog_available() for name in JOG_AXES)))
            for name in JOG_AXES:
                units = await axes[name].get_units()
                self._units[name] = _UM_PER_UNIT[units]
        except Exception:
            logger.exception("Could not determine stage jog support; jogging is unavailable")
            available = False
        await self._update(available=available, enabled=False, active=False, message=None)

    async def set_enabled(self, enabled: bool) -> None:
        """Arm or disarm jogging. Arming is refused while a capture owns the hardware."""
        async with self._lock:
            if enabled:
                if not self._state.value.available:
                    raise OperationRejectedError("This stage does not support jogging")
                if self._busy.value:
                    raise InstrumentBusyError("Unable to enable jogging while a capture is running")
                await self._update(enabled=True, message=None)
                logger.info("Stage jogging enabled")
            else:
                await self._stop_unlocked()
                await self._update(enabled=False, message=None)
                logger.info("Stage jogging disabled")

    async def lockout(self, reason: str) -> None:
        """Stop any jog and disable jogging, e.g. when an acquisition starts or the instrument closes."""
        async with self._lock:
            was = self._state.value
            await self._stop_unlocked()
            if was.enabled or was.active:
                logger.warning("Stage jogging disabled: %s", reason)
                await self._update(enabled=False, message=f"Jogging disabled: {reason}")

    async def release(self, client: str) -> None:
        """Forget a disconnected client; if it owned the jog, stop and disable jogging."""
        async with self._lock:
            self._seq.pop(client, None)
            if self._owner != client:
                return
            await self._stop_unlocked()
            await self._update(enabled=False, message="Jogging disabled: the controlling client disconnected")
            logger.warning("Stage jogging disabled: controlling client disconnected")

    async def command(self, client: str, seq: int, velocity: tuple[float, float, float]) -> None:
        """Apply one stick update: each component in ``[-1, 1]`` is a fraction of full jog speed."""
        if len(velocity) != len(JOG_AXES) or not all(math.isfinite(v) for v in velocity):
            raise OperationRejectedError("jog velocity must be three finite numbers")
        async with self._lock:
            if seq <= self._seq.get(client, -1):
                return  # Stale or duplicate update.
            self._seq[client] = seq
            state = self._state.value
            if not state.enabled or self._busy.value:
                if any(self._commanded.values()):
                    await self._stop_unlocked()
                return
            if self._owner is not None and self._owner != client:
                return  # Another client is driving; ignore until it releases the sticks.

            targets = {
                name: max(-1.0, min(1.0, fraction)) * self._max_speed
                for name, fraction in zip(JOG_AXES, velocity, strict=True)
            }
            axes = self._axes()
            changed: list[JogAxis] = [
                name for name in JOG_AXES if self._needs_command(self._commanded[name], targets[name])
            ]
            results = await asyncio.gather(
                *(axes[name].jog(targets[name] / self._units.get(name, 1.0)) for name in changed),
                return_exceptions=True,
            )
            errors: list[str] = []
            for name, result in zip(changed, results, strict=True):
                if isinstance(result, BaseException):
                    if not isinstance(result, Exception):
                        raise result
                    errors.append(f"{name}: {result}")
                    logger.error("Jogging axis %s failed: %s", name, result)
                else:
                    self._commanded[name] = targets[name]
            if errors:
                await self._stop_unlocked()
                await self._update(enabled=False, message="Jogging stopped after an error: " + "; ".join(errors))
                return

            moving = any(self._commanded.values())
            self._owner = client if moving else None
            self._arm_watchdog(moving)
            if moving != state.active:
                await self._update(active=moving)

    def _needs_command(self, current: float, target: float) -> bool:
        if target == current:
            return False
        if target == 0 or current == 0:
            return True  # Always honour starts and stops.
        return abs(target - current) > _RECOMMAND_FRACTION * self._max_speed

    def _arm_watchdog(self, moving: bool) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        if moving:
            self._watchdog = asyncio.create_task(self._expire(), name="stage-jog-watchdog")

    async def _expire(self) -> None:
        await asyncio.sleep(self._watchdog_s)
        async with self._lock:
            if self._watchdog is not asyncio.current_task():
                return
            self._watchdog = None
            logger.warning("Stage jog watchdog expired after %.0f ms without an update", self._watchdog_s * 1e3)
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        """Halt every axis this jog set moving. Halt failures are logged; state is cleared regardless."""
        if self._watchdog is not None and self._watchdog is not asyncio.current_task():
            self._watchdog.cancel()
        self._watchdog = None
        moving: list[JogAxis] = [name for name in JOG_AXES if self._commanded[name] != 0]
        self._commanded = dict.fromkeys(JOG_AXES, 0.0)
        self._owner = None
        if moving:
            axes = self._axes()
            results = await asyncio.gather(*(axes[name].halt() for name in moving), return_exceptions=True)
            for name, result in zip(moving, results, strict=True):
                if isinstance(result, Exception):
                    logger.error("Halting jogged axis %s failed: %s", name, result)
                elif isinstance(result, BaseException):
                    raise result
        if self._state.value.active:
            await self._update(active=False)

    async def _update(self, **changes: object) -> None:
        await self._state.set(self._state.value.model_copy(update=changes))


__all__ = ["JOG_AXES", "JOG_WATCHDOG_S", "MAX_JOG_SPEED_UM_S", "JogState", "StageJog"]
