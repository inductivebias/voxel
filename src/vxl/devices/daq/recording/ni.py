"""NI-DAQmx position recorder.

``NiPositionRecorder`` counts an X4 linear encoder on one counter and each marker's
rising edges on another. The hardware-timed DI engine's sample clock paces every
counter, so all inputs share sample indices; a worker thread drains aligned blocks
while the pass runs.
"""

import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from nidaqmx.constants import AcquisitionType, Edge, EncoderType, LengthUnits, LineGrouping, TaskMode
from nidaqmx.error_codes import DAQmxErrors
from nidaqmx.errors import DaqError
from nidaqmx.stream_readers import CounterReader, DigitalMultiChannelReader
from nidaqmx.task import Task as NiTask
from nidaqmx.task.channels import CIChannel

from vxl.devices.daq.hub_ni import NiDaqmx
from vxl.devices.daq.hub_ni.resources import NiTaskLease

from .base import PassAssociator, PositionRecorder, RecordedPass, RecorderArm

# The drain worker reads once a block of this duration is available on every input.
_BLOCK_S = 0.02
# Longest one aligned read may take; its samples are already acquired when it starts.
_READ_TIMEOUT_S = 1.0
# How long abort() waits for the drain worker to finish its current read.
_STOP_WAIT_S = 1.0
_ENCODER_FILTERS = ("ci_encoder_a_input_dig_fltr", "ci_encoder_b_input_dig_fltr")

type _Read = Callable[..., int]


def _apply_input_filter(channel: CIChannel, prefix: str, width_s: float) -> None:
    """Request one DAQmx digital input filter width; zero leaves the filter disabled."""
    setattr(channel, prefix + "_enable", False)
    if width_s > 0:
        setattr(channel, prefix + "_min_pulse_width", width_s)
        setattr(channel, prefix + "_enable", True)


def _read_encoder_filters(channel: CIChannel, *, enabled: bool) -> tuple[list[dict[str, Any]], float, float]:
    """Verify the committed encoder A/B filters match each other.

    Returns:
        Each enabled filter's readback, the accepted minimum pulse width, and the filter
        clock period, in seconds; both are zero when unfiltered.
    """
    readbacks: list[dict[str, Any]] = []
    for prefix in _ENCODER_FILTERS:
        if getattr(channel, prefix + "_enable") != enabled:
            raise RuntimeError(f"Encoder input filter {prefix} enable readback mismatch")
        if enabled:
            readbacks.append(
                {
                    "input": prefix,
                    "minimum_pulse_width_s": float(getattr(channel, prefix + "_min_pulse_width")),
                    "timebase_source": str(getattr(channel, prefix + "_timebase_src")),
                    "timebase_rate_hz": float(getattr(channel, prefix + "_timebase_rate")),
                }
            )
    if not readbacks:
        return [], 0.0, 0.0
    for key in ("minimum_pulse_width_s", "timebase_source", "timebase_rate_hz"):
        if any(readback[key] != readbacks[0][key] for readback in readbacks):
            raise RuntimeError(f"Encoder A and B input filters disagree on {key}")
    width_s, rate_hz = readbacks[0]["minimum_pulse_width_s"], readbacks[0]["timebase_rate_hz"]
    if not (width_s > 0 and rate_hz > 0):
        raise RuntimeError(f"Encoder input filter accepted width {width_s} s at a {rate_hz} Hz timebase")
    return readbacks, width_s, 1 / rate_hz


def _require(valid: bool, message: str) -> None:
    if not valid:
        raise RuntimeError(message)


class _Pass:
    """The NI tasks, lease, drain worker, and association of one armed pass.

    Tasks are kept in start order: the counters, then the DI task whose sample clock
    paces them. Inputs are never stopped or closed while a read is active, and a
    failed stop or close keeps its handle and the lease for a retry.
    """

    def __init__(self, uid: str, lease: NiTaskLease, *, block_samples: int, rate_hz: float) -> None:
        self._uid = uid
        self.lease = lease
        self._block_samples = block_samples
        self._poll_s = block_samples / rate_hz / 2
        self.tasks: dict[str, NiTask] = {}
        self.position: NiTask | None = None
        self.edges: dict[str, NiTask] = {}
        self.clock: NiTask | None = None
        self.associator: PassAssociator | None = None
        self.readback: dict[str, Any] = {}
        self._started: set[str] = set()
        self._position_read: _Read | None = None
        self._edge_reads: dict[str, _Read] = {}
        self._clock_read: _Read | None = None
        self._read_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._drain, name=f"{uid}-recording", daemon=True)
        self._failure: BaseException | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self.lease.released

    def add_task(self, role: str) -> NiTask:
        task = NiTask(f"{self._uid}_{role}")
        self.tasks[role] = task
        return task

    def start(self) -> None:
        """Start the counters, then the clock that paces them, then the drain worker."""
        if self.position is None or self.clock is None:
            raise RuntimeError("start() called before the recorder tasks were configured")
        self._position_read = CounterReader(self.position.in_stream).read_many_sample_double
        self._edge_reads = {
            name: CounterReader(task.in_stream).read_many_sample_uint32 for name, task in self.edges.items()
        }
        self._clock_read = DigitalMultiChannelReader(self.clock.in_stream).read_many_sample_port_byte
        for role, task in self.tasks.items():
            # A failed start may still have started the hardware, so it needs stopping too.
            self._started.add(role)
            task.start()
        self._worker.start()

    def mark_motion(self) -> None:
        with self._state_lock:
            self._associator().mark_motion(self._acquired())

    def finish(self, timeout_s: float) -> RecordedPass:
        """Stop the clock, drain the remaining samples, close everything, and associate."""
        if self._closed:
            raise RuntimeError(f"{self._uid}: an earlier cleanup failed; abort() retries it")
        self._stop.set()
        self._worker.join(timeout_s)
        if self._worker.is_alive():
            raise RuntimeError(f"{self._uid}: the recording worker is still reading; abort() retries the cleanup")
        failure: Exception | None = None
        try:
            self._stop_task("clock")  # every counter now holds its final sample
            self._drain_remainder()
        except Exception as error:
            failure = error
        try:
            self.close()
        except Exception as cleanup_error:
            if failure is None:
                raise
            failure.add_note(f"Cleanup also failed; abort() retries it: {cleanup_error}")
        if failure is not None:
            raise failure
        return self._associator().result(self.readback)

    def abort(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(_STOP_WAIT_S)
        self.close()

    def close(self) -> None:
        """Stop and close every task, clock first, then release the lease."""
        if not self._read_lock.acquire(blocking=False):
            raise RuntimeError(f"{self._uid}: an input read is still active; the NI tasks are kept for a retry")
        try:
            self._closed = True
            failures: list[Exception] = []
            for role in sorted(self.tasks, key=lambda name: name != "clock"):
                try:
                    self._stop_task(role)
                except Exception as error:
                    failures.append(error)
                try:
                    self.tasks[role].close()
                except Exception as error:
                    failures.append(error)
                else:
                    del self.tasks[role]
                    self._started.discard(role)
            if not self.tasks and not self.lease.released:
                self.lease.release()
            if failures:
                raise ExceptionGroup(f"{self._uid}: NI recorder cleanup failed", failures)
        finally:
            self._read_lock.release()

    def _stop_task(self, role: str) -> None:
        if role in self._started:
            self.tasks[role].stop()
            self._started.discard(role)

    def _associator(self) -> PassAssociator:
        if self.associator is None:
            raise RuntimeError("The recorder pass has no associator")
        return self.associator

    def _acquired(self) -> int:
        """Samples every counter has acquired; indices below it are complete on all of them."""
        return min(task.in_stream.total_samp_per_chan_acquired for role, task in self.tasks.items() if role != "clock")

    def _drain(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._read_available_block():
                    self._stop.wait(self._poll_s)
        except BaseException as error:
            if self._failure is None:
                self._failure = error

    def _read_available_block(self) -> bool:
        with self._read_lock:
            if self._closed or self._failure is not None:
                return False
            if min(task.in_stream.avail_samp_per_chan for task in self.tasks.values()) < self._block_samples:
                return False
            self._read_locked(self._block_samples, include_clock=True)
            return True

    def _drain_remainder(self) -> None:
        with self._read_lock:
            if self._failure is None and (remaining := self._acquired() - self._associator().samples):
                # A failed read is recorded in _failure and reported below with the worker's.
                with contextlib.suppress(Exception):
                    self._read_locked(remaining, include_clock=False)
            if self._failure is not None:
                raise RuntimeError(f"{self._uid}: {_describe_failure(self._failure)}") from self._failure

    def _read_locked(self, samples: int, *, include_clock: bool) -> None:
        """Read the same consecutive sample indices from every input and associate them.

        Any failure leaves the inputs on different sample indices, so it ends the pass.
        """
        try:
            positions, counts = self._read_aligned(samples, include_clock=include_clock)
            with self._state_lock:
                self._associator().add_block(positions, counts)
        except BaseException as error:
            self._failure = error
            raise

    def _read_aligned(self, samples: int, *, include_clock: bool) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self._position_read is None or self._clock_read is None:
            raise RuntimeError("read before the recorder tasks were started")
        deadline = time.monotonic() + _READ_TIMEOUT_S
        positions = np.empty(samples, dtype=np.float64)
        counts = {name: np.empty(samples, dtype=np.uint32) for name in self._edge_reads}
        reads = [(self._position_read, positions), *((read, counts[name]) for name, read in self._edge_reads.items())]
        if include_clock:
            # The clock lines' states are not recorded; reading them keeps their buffer from overflowing.
            reads.append((self._clock_read, np.empty((len(self.lease.di_paths), samples), dtype=np.uint8)))
        for read, data in reads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out reading {samples} aligned samples")
            if (count := read(data, number_of_samples_per_channel=samples, timeout=remaining)) != samples:
                raise RuntimeError(f"Short read of {count} of {samples} samples")
        return positions, counts


def _describe_failure(error: BaseException) -> str:
    if isinstance(error, DaqError) and error.error_code == DAQmxErrors.SAMPLES_NO_LONGER_AVAILABLE:
        return "the input buffer overran: samples were overwritten before the recorder read them"
    return f"a read failed ({type(error).__name__}: {error}), so the inputs are no longer aligned"


class NiPositionRecorder(PositionRecorder):
    """Record an NI X4 linear encoder and camera exposure edges on one hardware sample clock.

    Each pass reserves one counter for the encoder (``encoder_a``/``encoder_b``), one
    rising-edge counter per marker terminal, and the hardware-timed DI engine on
    ``clock_lines``; that DI task's sample clock paces every counter. Input buffers hold
    ``buffer_s`` of samples, and the pass fails if the drain falls further behind.

    ``filter_us`` enables NI's digital filter on both encoder inputs. NI rejects that
    filter together with input synchronization (-200770), so synchronization is enabled
    only when unfiltered. Marker inputs are unfiltered. Positions are the filtered,
    quantized encoder record in the units of ``RecorderArm`` (NI's meters scale is used
    as a plain linear scale).
    """

    def __init__(
        self,
        uid: str,
        *,
        hub: NiDaqmx,
        encoder_a: str,
        encoder_b: str,
        markers: Mapping[str, str],
        clock_lines: Sequence[str],
        sample_rate_hz: float = 100_000,
        filter_us: float = 0.0,
        buffer_s: float = 2.0,
    ) -> None:
        super().__init__(uid=uid, markers=markers, encoder_a=encoder_a, filter_s=filter_us * 1e-6)
        if not clock_lines:
            raise ValueError(f"{uid}: the sample clock needs at least one DI line")
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError(f"{uid}: sample_rate_hz must be finite and positive, got {sample_rate_hz}")
        if not math.isfinite(buffer_s) or buffer_s <= 0:
            raise ValueError(f"{uid}: buffer_s must be finite and positive, got {buffer_s}")
        self._hub = hub
        self._encoder_b = encoder_b
        self._clock_lines = tuple(clock_lines)
        self._rate_hz = float(sample_rate_hz)
        self._buffer_samples = max(2, math.ceil(sample_rate_hz * buffer_s))
        self._block_samples = max(1, min(math.ceil(sample_rate_hz * _BLOCK_S), self._buffer_samples // 2))
        self._log = logging.getLogger(f"{uid}.NiPositionRecorder")
        self._pass: _Pass | None = None

    @property
    def is_recording(self) -> bool:
        return self._pass is not None

    def arm(self, request: RecorderArm) -> None:
        if self._pass is not None:
            raise RuntimeError(f"{self.uid}: a pass is already recording; finish or abort it first")
        lease = self._hub.reserve_input_task(
            self.uid,
            counters=1 + len(self._markers),
            timed_di_lines=self._clock_lines,
            input_pfis=(self.encoder_a, self._encoder_b, *self._markers.values()),
        )
        recording = _Pass(self.uid, lease, block_samples=self._block_samples, rate_hz=self._rate_hz)
        try:
            self._configure(recording, request)
            recording.start()
        except BaseException as error:
            try:
                recording.close()
            except Exception as cleanup_error:
                self._log.warning("cleanup after a failed arm failed", exc_info=True)
                error.add_note(f"Cleanup also failed; abort() retries it: {cleanup_error}")
                self._pass = recording
            raise
        self._pass = recording

    def mark_motion(self) -> None:
        self._require_pass().mark_motion()

    def finish(self, timeout_s: float) -> RecordedPass:
        recording = self._require_pass()
        try:
            return recording.finish(timeout_s)
        finally:
            if recording.closed:
                self._pass = None

    def abort(self) -> None:
        if self._pass is None:
            return
        self._pass.abort()
        self._pass = None

    def _require_pass(self) -> _Pass:
        if self._pass is None:
            raise RuntimeError(f"{self.uid}: no pass is recording; call arm() first")
        return self._pass

    def _configure(self, recording: _Pass, request: RecorderArm) -> None:
        """Create and commit every task, verify the readbacks, and build the associator."""
        lease = recording.lease
        encoder_a, encoder_b, *marker_terminals = lease.input_pfi_paths
        clock_source = f"/{self._hub.device_name}/di/SampleClock"

        position = recording.add_task("position")
        recording.position = position
        encoder = position.ci_channels.add_ci_lin_encoder_chan(
            lease.counter_paths[0],
            decoding_type=EncoderType.X_4,
            zidx_enable=False,
            units=LengthUnits.METERS,
            dist_per_pulse=request.edge_spacing,
            initial_pos=request.origin,
        )
        encoder.ci_encoder_a_input_term = encoder_a
        encoder.ci_encoder_b_input_term = encoder_b
        for prefix in _ENCODER_FILTERS:
            _apply_input_filter(encoder, prefix, self.filter_s)
        synchronized = self.filter_s == 0
        encoder.ci_encoder_a_input_dig_sync_enable = synchronized
        encoder.ci_encoder_b_input_dig_sync_enable = synchronized

        edge_channels: dict[str, CIChannel] = {}
        for index, (name, counter, terminal) in enumerate(
            zip(self._markers, lease.counter_paths[1:], marker_terminals, strict=True)
        ):
            task = recording.add_task(f"marker{index}")
            recording.edges[name] = task
            channel = task.ci_channels.add_ci_count_edges_chan(counter, edge=Edge.RISING, initial_count=0)
            channel.ci_count_edges_term = terminal
            channel.ci_count_edges_dig_fltr_enable = False
            edge_channels[name] = channel

        clock = recording.add_task("clock")
        recording.clock = clock
        clock.di_channels.add_di_chan(",".join(lease.di_paths), line_grouping=LineGrouping.CHAN_PER_LINE)
        clock.timing.cfg_samp_clk_timing(
            self._rate_hz, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=self._buffer_samples
        )
        for task in (position, *recording.edges.values()):
            task.timing.cfg_samp_clk_timing(
                self._rate_hz,
                source=clock_source,
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self._buffer_samples,
            )
        for task in recording.tasks.values():
            task.in_stream.input_buf_size = self._buffer_samples
        # Committing every task together surfaces route and resource conflicts before any start.
        for task in recording.tasks.values():
            task.control(TaskMode.TASK_COMMIT)

        filters, filter_s, filter_clock_s = _read_encoder_filters(encoder, enabled=self.filter_s > 0)
        rate_hz = clock.timing.samp_clk_rate
        _require(rate_hz == self._rate_hz, f"Sample clock rate was coerced from {self._rate_hz} to {rate_hz} Hz")
        for actual, expected, name in (
            (encoder.ci_encoder_a_input_term, encoder_a, "encoder A terminal"),
            (encoder.ci_encoder_b_input_term, encoder_b, "encoder B terminal"),
            (encoder.ci_encoder_decoding_type, EncoderType.X_4, "encoder decoding"),
            (encoder.ci_encoder_a_input_dig_sync_enable, synchronized, "encoder A synchronization"),
            (encoder.ci_encoder_b_input_dig_sync_enable, synchronized, "encoder B synchronization"),
            *(
                (channel.ci_count_edges_term, terminal, f"marker '{name}' terminal")
                for (name, channel), terminal in zip(edge_channels.items(), marker_terminals, strict=True)
            ),
        ):
            _require(actual == expected, f"{name} readback {actual!r} does not match {expected!r}")
        edge_spacing = encoder.ci_lin_encoder_dist_per_pulse
        _require(
            math.isclose(edge_spacing, request.edge_spacing, rel_tol=1e-12),
            f"Encoder edge spacing readback {edge_spacing} does not match {request.edge_spacing}",
        )

        recording.associator = PassAssociator(
            request, list(self._markers), rate_hz=rate_hz, filter_s=filter_s, filter_clock_s=filter_clock_s
        )
        recording.readback = {
            "device": self._hub.device_name,
            "sample_rate_hz": rate_hz,
            "sample_clock": clock_source,
            "clock_lines": list(lease.di_paths),
            "encoder_counter": lease.counter_paths[0],
            "encoder_a": encoder_a,
            "encoder_b": encoder_b,
            "edge_spacing": edge_spacing,
            "markers": {
                name: {"counter": counter, "terminal": terminal}
                for name, counter, terminal in zip(
                    self._markers, lease.counter_paths[1:], marker_terminals, strict=True
                )
            },
            "filter_requested_s": self.filter_s,
            "filter_s": filter_s,
            "filter_clock_s": filter_clock_s,
            "filters": filters,
            "input_synchronization": synchronized,
            "buffer_samples": self._buffer_samples,
        }


__all__ = ["NiPositionRecorder"]
