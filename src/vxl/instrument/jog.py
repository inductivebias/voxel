"""Guarded velocity jogging of the stage: hard speed cap, unit conversion, busy lockout, and dead-man watchdog."""

import asyncio
import logging
from collections.abc import Callable, Mapping

from vxlib.reactivity import Readable

from vxl.devices.axes import ContinuousAxisHandle

logger = logging.getLogger(__name__)

JOG_AXES = ("x", "y", "z")
MAX_JOG_SPEED_UM_S = 2000.0
JOG_WATCHDOG_S = 0.3
_RECOMMAND_FRACTION = 0.02
_UM_PER_UNIT = {"nm": 1e-3, "um": 1.0, "mm": 1e3}


class StageJog:
    """Turns streamed stick deflections in ``[-1, 1]`` into capped, watchdog-guarded stage velocity moves."""

    def __init__(
        self,
        axes: Callable[[], Mapping[str, ContinuousAxisHandle]],
        *,
        busy: Readable[bool],
        max_speed_um_s: float = MAX_JOG_SPEED_UM_S,
        watchdog_s: float = JOG_WATCHDOG_S,
    ) -> None:
        self._axes = axes
        self._busy = busy
        self._max_speed = max_speed_um_s
        self._watchdog_s = watchdog_s
        self._lock = asyncio.Lock()
        self._um_per_unit: dict[str, float] = {}
        self._commanded: dict[str, float] = dict.fromkeys(JOG_AXES, 0.0)
        self._watchdog: asyncio.Task[None] | None = None

    async def stop(self) -> None:
        """Halt any active jog (acquisition start or instrument close)."""
        async with self._lock:
            await self._stop()

    async def command(self, velocity: tuple[float, float, float]) -> None:
        """Apply one stick update: logical x, y, z as fractions of full jog speed, clamped to ``[-1, 1]``."""
        async with self._lock:
            if self._busy.value:
                await self._stop()
                return

            axes = self._axes()
            targets = {
                name: max(-1.0, min(1.0, v)) * self._max_speed for name, v in zip(JOG_AXES, velocity, strict=True)
            }
            changed = [name for name in JOG_AXES if self._needs_command(self._commanded[name], targets[name])]
            try:
                if not self._um_per_unit:
                    for name in JOG_AXES:
                        self._um_per_unit[name] = _UM_PER_UNIT[await axes[name].get_units()]
                self._commanded.update({name: targets[name] for name in changed})
                await asyncio.gather(*(axes[n].jog(targets[n] / self._um_per_unit[n]) for n in changed))
            except Exception as error:
                logger.warning("Stopping stage jog after axis error: %s", error)
                await self._stop()
                return

            self._reset_watchdog(any(self._commanded.values()))

    def _needs_command(self, current: float, target: float) -> bool:
        if target == 0 or current == 0:
            return target != current
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

    async def _stop(self) -> None:
        self._reset_watchdog(False)
        moving = [name for name in JOG_AXES if self._commanded[name]]
        self._commanded = dict.fromkeys(JOG_AXES, 0.0)
        if not moving:
            return
        axes = self._axes()
        results = await asyncio.gather(*(axes[name].halt() for name in moving), return_exceptions=True)
        for name, result in zip(moving, results, strict=True):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result
                logger.warning("Failed to halt %s axis while stopping stage jog: %s", name, result)


__all__ = ["JOG_AXES", "JOG_WATCHDOG_S", "MAX_JOG_SPEED_UM_S", "StageJog"]
