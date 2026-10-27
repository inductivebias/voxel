"""Simulated position recorder that synthesizes the ideal pass for each arm request."""

import math
from collections.abc import Collection, Mapping

import numpy as np

from .base import PassAssociator, PositionRecorder, RecordedPass, RecorderArm

_RATE_HZ = 10_000.0
# Stationary samples before the motion start, and samples recorded after the pass.
_LEAD_SAMPLES = 100
_TAIL_SAMPLES = 100
_BLOCK_SAMPLES = 10_000


class SimulatedPositionRecorder(PositionRecorder):
    """In-memory recorder whose pass is synthesized at ``finish`` and validated like hardware.

    The encoder rests at ``origin`` until the motion sample, then moves at ``speed``,
    quantized to one X4 count (``edge_spacing / 4``). Exposure ``k`` of every marker lands
    on the first sample at or past ``first_exposure[0] + edge_spacing / 2 + k * spacing``,
    so the first-exposure window must be at least half an edge wide to validate.
    ``missing_frames`` maps a marker to 0-based frames whose exposure never arrives.
    """

    def __init__(
        self,
        uid: str,
        *,
        markers: Mapping[str, str],
        encoder_a: str = "pfi1",
        filter_us: float = 0.0,
        missing_frames: Mapping[str, Collection[int]] | None = None,
    ) -> None:
        super().__init__(uid=uid, markers=markers, encoder_a=encoder_a, filter_s=filter_us * 1e-6)
        self._missing = {name: frozenset(frames) for name, frames in (missing_frames or {}).items()}
        if unknown := sorted(set(self._missing) - set(self._markers)):
            raise ValueError(f"{uid}: missing_frames names unknown markers {unknown}")
        self._request: RecorderArm | None = None
        self._associator: PassAssociator | None = None
        self._moving = False

    @property
    def is_recording(self) -> bool:
        return self._request is not None

    def arm(self, request: RecorderArm) -> None:
        if self._request is not None:
            raise RuntimeError(f"{self.uid}: a pass is already recording; finish or abort it first")
        self._associator = PassAssociator(
            request, list(self._markers), rate_hz=_RATE_HZ, filter_s=self.filter_s, filter_clock_s=0.0
        )
        self._request = request
        self._moving = False

    def mark_motion(self) -> None:
        if self._associator is None:
            raise RuntimeError(f"{self.uid}: mark_motion requires an armed pass")
        self._associator.mark_motion(_LEAD_SAMPLES)
        self._moving = True

    def finish(self, timeout_s: float) -> RecordedPass:
        del timeout_s  # the pass is synthesized instantly
        request, associator, moving = self._request, self._associator, self._moving
        if request is None or associator is None:
            raise RuntimeError(f"{self.uid}: finish requires an armed pass")
        self.abort()

        step = request.edge_spacing / 4
        first = request.first_exposure[0] + request.edge_spacing / 2
        targets = {
            name: np.array(
                [first + k * request.spacing for k in range(request.frames) if k not in self._missing.get(name, ())]
            )
            for name in self._markers
        }
        travel_s = (first + (request.frames - 1) * request.spacing + step - request.origin) / request.speed
        total = _LEAD_SAMPLES + _TAIL_SAMPLES + (max(math.ceil(travel_s * _RATE_HZ), 0) if moving else 0)
        for start in range(0, total, _BLOCK_SAMPLES):
            index = np.arange(start, min(start + _BLOCK_SAMPLES, total))
            travel = np.maximum(index - _LEAD_SAMPLES, 0) / _RATE_HZ * request.speed if moving else np.zeros(len(index))
            positions = request.origin + np.floor(travel / step) * step
            counts = {name: np.searchsorted(marks, positions, side="right") for name, marks in targets.items()}
            associator.add_block(positions, counts)
        return associator.result(
            {
                "sample_rate_hz": _RATE_HZ,
                "filter_s": self.filter_s,
                "filter_clock_s": 0.0,
                "encoder_a": self.encoder_a,
                "markers": self.markers,
            }
        )

    def abort(self) -> None:
        self._request = None
        self._associator = None
        self._moving = False


__all__ = ["SimulatedPositionRecorder"]
