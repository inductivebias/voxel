"""Vendor-neutral recording of the stage encoder position at each camera exposure.

A recorder samples the stage encoder and one marker per camera (its exposure-start
output, counted as edges) on a single shared sample clock through one monotonic pass,
then associates every exposure with the encoder position at the sample where its
marker count incremented.
"""

import math
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Self

import numpy as np
from pydantic import BaseModel, Field, TypeAdapter, model_validator

from rigup import Device, DeviceController, DeviceHandle, describe
from vxl.devices.base import DeviceType

RecorderState = Literal["idle", "recording"]
_MARKERS_ADAPTER = TypeAdapter(dict[str, str])
# Backward travel tolerated between the motion start and the last exposure. This only
# absorbs float noise: one encoder count is far larger.
_REVERSAL_LIMIT = 1e-6


class RecorderArm(BaseModel, frozen=True, allow_inf_nan=False):
    """The expected pass, in the stage axis's units.

    Attributes:
        origin: Encoder position at arm time, from the stage's own feedback.
        edge_spacing: Travel per encoder A rising edge; one X4 count is a quarter of it.
        frames: Exposures expected on every marker.
        spacing: Expected travel between consecutive exposures.
        first_exposure: Inclusive window for the first exposure's position.
        speed: Expected scan speed in axis units per second, for the tolerance.
    """

    origin: float
    edge_spacing: float = Field(gt=0)
    frames: int = Field(ge=1)
    spacing: float = Field(gt=0)
    first_exposure: tuple[float, float]
    speed: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered_window(self) -> Self:
        low, high = self.first_exposure
        if low > high:
            raise ValueError(f"first_exposure window ({low}, {high}) is not ordered")
        return self


class FramePosition(BaseModel, frozen=True):
    """One exposure: the sample where its marker counted and the encoder position there."""

    frame: int
    sample: int
    time_s: float
    position: float


class RecordedPass(BaseModel, frozen=True):
    """A validated pass: exactly ``RecorderArm.frames`` positions per marker.

    ``readback`` holds what the driver accepted (sample rate, input filter and its
    timebase, terminals, buffer size).
    """

    frames: dict[str, list[FramePosition]]
    samples: int
    motion_sample: int
    tolerance: float
    readback: dict[str, Any]


@dataclass
class _MarkerTrack:
    """Bounded per-marker state: running counts plus at most ``frames`` retained exposures."""

    first_count: int | None = None
    last_count: int = 0
    exposures: int = 0
    bad_samples: int = 0
    first_bad_sample: int | None = None
    samples: list[int] = field(default_factory=list)
    positions: list[float] = field(default_factory=list)
    # Largest reversal between the motion start and each retained exposure.
    reversals: list[float] = field(default_factory=list)


class PassAssociator:
    """Associates marker count increments with encoder positions sampled on one clock.

    Feed consecutive samples with :meth:`add_block` as they are read. Memory stays
    bounded however long the pass runs: each marker keeps running counts and its first
    ``frames`` exposures, and the encoder keeps only its running peak and largest
    reversal since the motion sample. Not thread-safe.

    The position tolerance is a quarter edge (one X4 count) plus the travel at ``speed``
    during the accepted input filter, its clock period, and one sample period.

    Raises:
        ValueError: If the input filter is not shorter than the time between encoder
            A/B transitions at ``speed``, or the clock gives fewer than two samples
            between exposures.
    """

    def __init__(
        self,
        request: RecorderArm,
        markers: Sequence[str],
        *,
        rate_hz: float,
        filter_s: float,
        filter_clock_s: float,
    ) -> None:
        phase_s = request.edge_spacing / (4 * request.speed)
        if filter_s + filter_clock_s >= phase_s:
            raise ValueError(
                f"Input filter {filter_s:g} s plus its {filter_clock_s:g} s clock is not shorter than the "
                f"{phase_s:g} s between encoder A/B transitions at speed {request.speed:g}"
            )
        if rate_hz * request.spacing / request.speed < 2:
            raise ValueError(f"A {rate_hz:g} Hz sample clock gives fewer than two samples between exposures")
        self._request = request
        self._rate_hz = rate_hz
        self.tolerance = request.edge_spacing / 4 + request.speed * (filter_s + filter_clock_s + 1 / rate_hz)
        self._tracks = {name: _MarkerTrack() for name in markers}
        self._samples = 0
        self._motion: int | None = None
        self._peak = -math.inf
        self._reversal = 0.0
        self._nonfinite = 0

    @property
    def samples(self) -> int:
        return self._samples

    def mark_motion(self, sample: int) -> None:
        """Mark ``sample`` as the motion start; it must not precede samples already added."""
        if self._motion is not None:
            raise RuntimeError(f"Motion start is already marked at sample {self._motion}")
        if sample < self._samples:
            raise RuntimeError(f"Motion sample {sample} precedes the {self._samples} samples already associated")
        self._motion = sample

    def add_block(self, positions: np.ndarray, counts: Mapping[str, np.ndarray]) -> None:
        """Add the next consecutive samples: encoder positions and each marker's edge count."""
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)
        if n == 0:
            return
        start = self._samples
        self._nonfinite += n - int(np.count_nonzero(np.isfinite(positions)))
        reversals = self._track_reversal(positions, start)
        for name, track in self._tracks.items():
            block = np.asarray(counts[name], dtype=np.int64)
            if len(block) != n:
                raise ValueError(f"Marker '{name}' block has {len(block)} samples; positions have {n}")
            if track.first_count is None:
                track.first_count = int(block[0])
                previous = track.first_count
            else:
                previous = track.last_count
            steps = np.diff(block, prepend=previous)
            if (bad := np.flatnonzero((steps != 0) & (steps != 1))).size:
                track.bad_samples += bad.size
                if track.first_bad_sample is None:
                    track.first_bad_sample = start + int(bad[0])
            edges = np.flatnonzero(steps > 0)
            track.exposures += int(steps[edges].sum())
            for index in edges[: self._request.frames - len(track.samples)]:
                track.samples.append(start + int(index))
                track.positions.append(float(positions[index]))
                track.reversals.append(float(reversals[index]))
            track.last_count = int(block[-1])
        self._samples += n

    def _track_reversal(self, positions: np.ndarray, start: int) -> np.ndarray:
        """Return, per sample, the largest fall below the running peak since the motion sample."""
        reversals = np.zeros(len(positions))
        if self._motion is None or (offset := max(self._motion - start, 0)) >= len(positions):
            return reversals
        moving = positions[offset:]
        # fmax skips non-finite samples, which are reported separately.
        peak = np.fmax.accumulate(np.fmax(moving, self._peak))
        reversal = np.fmax.accumulate(np.fmax(peak - moving, self._reversal))
        reversals[offset:] = reversal
        self._peak = float(peak[-1])
        self._reversal = float(reversal[-1])
        return reversals

    def result(self, readback: dict[str, Any]) -> RecordedPass:
        """Validate the whole pass and return each marker's frame positions.

        Raises:
            ValueError: Listing every violation found.
        """
        motion = self._motion
        violations: list[str] = []
        if self._samples < 2:
            violations.append(f"recorded {self._samples} samples; at least two are needed")
        if motion is None:
            violations.append("the motion start was never marked")
        elif motion >= self._samples:
            violations.append(f"motion sample {motion} is outside the {self._samples} recorded samples")
        if self._nonfinite:
            violations.append(f"{self._nonfinite} encoder samples are not finite")

        last_sample, last_reversal = -1, 0.0
        for name, track in self._tracks.items():
            violations.extend(self._marker_violations(name, track))
            if track.samples and track.samples[-1] > last_sample:
                last_sample, last_reversal = track.samples[-1], track.reversals[-1]
        if last_reversal > _REVERSAL_LIMIT:
            violations.append(
                f"the encoder reversed by {last_reversal:g} between the motion start and the last exposure"
            )

        if violations or motion is None:
            raise ValueError("Recorded pass is invalid:\n" + "\n".join(f"- {violation}" for violation in violations))
        return RecordedPass(
            frames={
                name: [
                    FramePosition(frame=frame, sample=sample, time_s=sample / self._rate_hz, position=position)
                    for frame, (sample, position) in enumerate(zip(track.samples, track.positions, strict=True))
                ]
                for name, track in self._tracks.items()
            },
            samples=self._samples,
            motion_sample=motion,
            tolerance=self.tolerance,
            readback=readback,
        )

    def _marker_violations(self, name: str, track: _MarkerTrack) -> list[str]:
        request, tolerance, motion = self._request, self.tolerance, self._motion
        violations: list[str] = []
        if track.first_count:
            violations.append(
                f"{name}: count was {track.first_count} at the first sample, so an exposure preceded recording"
            )
        if track.bad_samples:
            violations.append(
                f"{name}: {track.bad_samples} samples changed the count by other than 0 or 1 "
                f"(first at sample {track.first_bad_sample})"
            )
        if track.exposures != request.frames:
            violations.append(f"{name}: expected {request.frames} exposures, recorded {track.exposures}")
        if motion is not None and (early := sum(sample <= motion for sample in track.samples)):
            violations.append(f"{name}: {early} exposures at or before the motion sample {motion}")
        errors = np.abs(np.diff(track.positions) - request.spacing)
        if (wrong := int(np.count_nonzero(errors > tolerance))) > 0:
            violations.append(
                f"{name}: {wrong} exposure gaps differ from the spacing {request.spacing:g} by more than "
                f"{tolerance:g} (worst by {float(errors.max()):g})"
            )
        low, high = request.first_exposure
        if track.positions and not low - tolerance <= track.positions[0] <= high + tolerance:
            violations.append(
                f"{name}: first exposure at {track.positions[0]:g} is outside [{low:g}, {high:g}] ± {tolerance:g}"
            )
        return violations


class PositionRecorderController(DeviceController["PositionRecorder"]):
    def __init__(self, device: "PositionRecorder", stream_interval: float = 0.1) -> None:
        super().__init__(device, stream_interval=stream_interval)

    @property
    @describe(label="State", desc="Whether a pass is holding the recorder", stream=True)
    def state(self) -> RecorderState:
        return "recording" if self.device.is_recording else "idle"

    @describe(label="Arm", desc="Start recording encoder position and exposure markers for one pass")
    async def arm(self, request: RecorderArm) -> None:
        await self._run_sync(self.device.arm, request)

    @describe(label="Mark Motion", desc="Mark the current sample as the start of motion")
    async def mark_motion(self) -> None:
        await self._run_sync(self.device.mark_motion)

    @describe(label="Finish", desc="Stop recording and return the encoder position at every exposure")
    async def finish(self, timeout_s: float) -> RecordedPass:
        return await self._run_sync(self.device.finish, timeout_s)

    @describe(label="Abort", desc="Stop recording and discard the pass")
    async def abort(self) -> None:
        await self._run_sync(self.device.abort)


class PositionRecorder(Device):
    """Records the stage encoder and camera exposure markers on one shared sample clock.

    One pass at a time: ``arm`` starts recording, ``mark_motion`` marks the sample at
    which the stage was commanded to move, and ``finish`` returns the encoder position at
    every marker's exposures. Positions must increase through the pass.
    """

    __DEVICE_TYPE__: ClassVar[str] = DeviceType.POSITION_RECORDER
    __CONTROLLER_TYPE__: ClassVar[type] = PositionRecorderController

    def __init__(self, uid: str, *, markers: Mapping[str, str], encoder_a: str, filter_s: float) -> None:
        super().__init__(uid=uid)
        if not markers:
            raise ValueError(f"{uid}: a position recorder needs at least one marker")
        if not math.isfinite(filter_s) or filter_s < 0:
            raise ValueError(f"{uid}: input filter width must be finite and non-negative, got {filter_s}")
        self._markers: dict[str, str] = dict(markers)
        self._encoder_a = encoder_a
        self._filter_s = filter_s

    @property
    @describe(label="Markers", desc="Marker name (a camera uid) -> exposure input terminal")
    def markers(self) -> dict[str, str]:
        return dict(self._markers)

    @property
    @describe(label="Encoder A", desc="Encoder A input terminal")
    def encoder_a(self) -> str:
        return self._encoder_a

    @property
    @describe(label="Input Filter", units="s", desc="Requested encoder input filter width; 0 is unfiltered")
    def filter_s(self) -> float:
        return self._filter_s

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """True from a successful ``arm`` until ``finish`` or ``abort`` releases the hardware."""

    @abstractmethod
    def arm(self, request: RecorderArm) -> None:
        """Reserve the inputs and start recording at the encoder position ``request.origin``."""

    @abstractmethod
    def mark_motion(self) -> None:
        """Mark the current sample as the motion start; call right after commanding the move."""

    @abstractmethod
    def finish(self, timeout_s: float) -> RecordedPass:
        """Stop recording, release the hardware, and validate the pass.

        Args:
            timeout_s: Longest to wait for a read already in progress to finish.

        Raises:
            ValueError: Listing every way the recording disagrees with the arm request.
            RuntimeError: If recording itself failed, such as an input buffer overrun.
        """

    @abstractmethod
    def abort(self) -> None:
        """Stop recording, release the hardware, and discard the pass. Idempotent."""

    def close(self) -> None:
        self.abort()


class PositionRecorderHandle(DeviceHandle["PositionRecorder"]):
    async def arm(self, request: RecorderArm) -> None:
        await self.call("arm", request)

    async def mark_motion(self) -> None:
        await self.call("mark_motion")

    async def finish(self, timeout_s: float) -> RecordedPass:
        value = await self.call("finish", timeout_s)
        return value if isinstance(value, RecordedPass) else RecordedPass.model_validate(value)

    async def abort(self) -> None:
        await self.call("abort")

    async def get_state(self) -> RecorderState:
        return await self.props.get_value("state")

    async def get_markers(self) -> dict[str, str]:
        return _MARKERS_ADAPTER.validate_python(await self.props.get_value("markers"))

    async def get_encoder_a(self) -> str:
        return str(await self.props.get_value("encoder_a"))

    async def get_filter_s(self) -> float:
        return float(await self.props.get_value("filter_s"))


__all__ = [
    "FramePosition",
    "PassAssociator",
    "PositionRecorder",
    "PositionRecorderController",
    "PositionRecorderHandle",
    "RecordedPass",
    "RecorderArm",
    "RecorderState",
]
