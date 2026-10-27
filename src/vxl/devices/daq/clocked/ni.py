"""NI-DAQmx clocked analog signal generator.

``NiSignalGenerator`` (clocked) owns a single NI AO task. A ``CounterTrigger`` adds a
counter-output task whose pulses retrigger finite AO cycles, periodically or, for one
finite run, by dividing the edges of an input PFI; an ``InputTrigger`` restarts a
cycle for each edge received from a configured PFI.

Both take a ``NiDaqmx`` hub (``vxl.devices.daq.hub_ni``) at construction; the card and pins
live there.
"""

import logging
import time
from collections.abc import Callable, Mapping
from typing import Annotated, Literal

import numpy as np
from nidaqmx.constants import AcquisitionType as NiAcqType
from nidaqmx.constants import Edge, Level, RegenerationMode, TaskMode
from nidaqmx.task import Task as NiTask
from nidaqmx.task.channels import COChannel
from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter
from vxlib.quantity import VoltageRange

from vxl.devices.daq.hub_ni import NiDaqmx
from vxl.devices.daq.hub_ni.resources import NiTaskLease  # noqa: TC001 - annotations are evaluated at runtime

from .base import EdgeDivision, EdgePacing, SignalGenerator, Signals


class CounterTrigger(BaseModel):
    """Periodically retrigger AO cycles using an NI counter.

    The trigger frequency derives from ``Signals.duration + rest_time``.
    ``output`` optionally specifies the NI terminal to which the counter pulse is
    routed for downstream devices.

    ``edge_source`` optionally names an NI input whose rising edges the same counter
    can divide instead, for one finite run (``start_paced``). ``edge_filter_us`` is
    the minimum pulse width of that input's digital filter; 0 leaves it unfiltered.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["counter"] = "counter"
    output: str | None = None
    edge_source: str | None = None
    edge_filter_us: float = Field(default=0.0, ge=0)


class InputTrigger(BaseModel):
    """Retrigger one finite AO cycle for each rising edge from an NI input.

    ``source`` specifies the NI terminal that supplies the trigger edges.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["input"] = "input"
    source: str


Trigger = Annotated[CounterTrigger | InputTrigger, Discriminator("type")]

_TRIGGER_ADAPTER = TypeAdapter(Trigger)

# stop() and wait_until_done() wait at most one cycle period plus this margin for the
# cycle in progress to end.
_CYCLE_END_MARGIN_S = 0.1
_CYCLE_END_POLL_S = 0.001

# Each edge-divider pulse is high for this many source edges and low for the rest of
# the divider; AO cycles start on its rising edge.
_DIVIDER_HIGH_TICKS = 2


class NiSignalGenerator(SignalGenerator):
    """NI-DAQmx analog-output engine.

    ``trigger`` controls how each finite AO buffer cycle starts. ``CounterTrigger``
    generates periodic cycle triggers using an NI counter; ``InputTrigger`` starts
    a cycle for each rising edge received from its configured source. This does not
    select the AO sample clock, which runs at ``Signals.sample_rate``.

    Omitting ``trigger`` selects ``CounterTrigger``. ``InputTrigger`` currently
    supports unbounded external pacing only; ``start(repeat=N)`` is unsupported.
    A ``CounterTrigger`` with an ``edge_source`` supports ``start_paced``.

    Enables regeneration on the AO stream so each trigger can replay the same
    finite buffer.
    """

    def __init__(
        self,
        uid: str,
        *,
        hub: NiDaqmx,
        ports: Mapping[str, str],
        trigger: Trigger | Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(uid=uid, ports=ports)
        self._hub = hub
        self._trigger = CounterTrigger() if trigger is None else _TRIGGER_ADAPTER.validate_python(trigger)
        self._log = logging.getLogger(f"{uid}.NiSignalGenerator")

        # Hardware handles — populated on setup, cleared on teardown
        self._ao_task: NiTask | None = None
        self._co_task: NiTask | None = None
        self._lease: NiTaskLease | None = None
        self._ao_channel_order: list[str] = []  # port names in the order NI applied them
        self._cycle_samples = 0  # AO samples per channel in one cycle
        self._cycle_period_s = 0.0  # duration + rest_time
        self._counter_route: tuple[str, str] | None = None  # leased (counter, output PFI) paths

        self._finite_repeat: int | None = None  # cycles in the current finite run; None = continuous
        self._edge_paced = False  # _co_task is the start_paced edge divider

    # ---- introspection ----

    @property
    def voltage_range(self) -> VoltageRange:
        return self._hub.voltage_range

    @property
    def edge_pacing(self) -> EdgePacing | None:
        trigger = self._trigger
        if not isinstance(trigger, CounterTrigger) or trigger.edge_source is None:
            return None
        return EdgePacing(source=trigger.edge_source, filter_s=trigger.edge_filter_us * 1e-6)

    # ---- hardware primitives ----

    def setup(self, signals: Signals) -> None:
        if self._ao_task is not None or self._co_task is not None or self._lease is not None:
            raise RuntimeError("setup() called with live AO task; teardown() first")

        sample_rate = signals.sample_rate
        trigger = self._trigger
        duration = signals.duration
        rest_time = signals.rest_time

        num_samples = int(float(sample_rate) * float(duration))
        self._cycle_samples = num_samples
        self._cycle_period_s = float(duration) + float(rest_time)
        frame_freq = 1.0 / self._cycle_period_s

        # Input PFIs are validated but not exclusively claimed; an output PFI
        # driven by the counter is part of the lease.
        physical_output = trigger.output if isinstance(trigger, CounterTrigger) else None
        if isinstance(trigger, InputTrigger):
            physical_input_pins = (trigger.source,)
        elif trigger.edge_source is not None:
            physical_input_pins = (trigger.edge_source,)
        else:
            physical_input_pins = ()

        lease = self._hub.reserve_ao_task(
            self.uid,
            tuple(self._ports.values()),
            hardware_timed=True,
            needs_counter=isinstance(trigger, CounterTrigger),
            output_pfi=physical_output,
            input_pfis=physical_input_pins,
        )
        self._lease = lease

        counter_trigger_resources: tuple[str, str] | None = None
        if isinstance(trigger, CounterTrigger):
            if lease.counter_path is None or lease.output_pfi_path is None:
                lease.release()
                self._lease = None
                raise RuntimeError("Counter-trigger reservation did not provide a counter and output route")
            counter_trigger_resources = (lease.counter_path, lease.output_pfi_path)
        self._counter_route = counter_trigger_resources

        try:
            # Acquire the complete lease before creating any NI task. Keep task
            # handles on self immediately so cleanup can retry a failed close.
            self._ao_task = NiTask(f"{self.uid}_ao")
            for port_name, path in zip(self._ports, lease.ao_paths, strict=True):
                self._ao_task.ao_channels.add_ao_voltage_chan(path)
                self._ao_channel_order.append(port_name)

            # Each counter or input trigger plays one finite buffer cycle, then
            # the task arms for the next trigger.
            self._ao_task.timing.cfg_samp_clk_timing(
                rate=float(sample_rate),
                sample_mode=NiAcqType.FINITE,
                samps_per_chan=num_samples,
            )

            # Enable regeneration so retriggered finite playback re-reads the same
            # buffer on each trigger cycle.
            self._ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION

            if counter_trigger_resources is not None:
                self._create_counter_trigger(
                    counter_trigger_resources[0],
                    frame_freq,
                    counter_trigger_resources[1],
                )

            if isinstance(trigger, CounterTrigger) and lease.counter_name is not None:
                trig_src = f"/{self._hub.device_name}/{lease.counter_name.upper()}InternalOutput"
                self._ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source=trig_src,
                    trigger_edge=Edge.RISING,
                )
                self._ao_task.triggers.start_trigger.retriggerable = True
            elif isinstance(trigger, InputTrigger):
                pfi_path = lease.input_pfi_paths[0]
                self._ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source=pfi_path,
                    trigger_edge=Edge.RISING,
                )
                self._ao_task.triggers.start_trigger.retriggerable = True
        except Exception:
            self._close_tasks_and_release(suppress_close_errors=True)
            if self._lease is None:
                self._ao_channel_order = []
                self._counter_route = None
            raise

    def _create_counter_trigger(
        self,
        counter_path: str,
        frequency_hz: float,
        output_pfi_path: str | None = None,
    ) -> NiTask:
        """Create a CO task that emits a continuous pulse train at ``frequency_hz``.

        When ``output_pfi_path`` is provided, the CO pulse is routed to that physical
        terminal via ``co_pulse_term``, making the cycle trigger visible to downstream
        devices.
        """
        co_task = NiTask(f"{self.uid}_clock")
        self._co_task = co_task
        chan = co_task.co_channels.add_co_pulse_chan_freq(
            counter_path,
            freq=frequency_hz,
            duty_cycle=0.5,
        )
        if output_pfi_path is not None:
            chan.co_pulse_term = output_pfi_path
        co_task.timing.cfg_implicit_timing(sample_mode=NiAcqType.CONTINUOUS)
        return co_task

    def write(self, port_arrays: Mapping[str, np.ndarray]) -> None:
        if self._ao_task is None:
            raise RuntimeError("write() called before setup()")

        # Infer sample count from the first provided array
        n_samples = next(iter(port_arrays.values())).shape[-1] if port_arrays else 0

        # Assemble arrays in the NI channel order (same order pins were added).
        # Ports with no waveform get a zero-filled array (0 V output).
        ordered = [
            np.asarray(port_arrays[name], dtype=np.float64)
            if name in port_arrays
            else np.zeros(n_samples, dtype=np.float64)
            for name in self._ao_channel_order
        ]

        data = np.vstack(ordered) if len(ordered) > 1 else ordered[0]
        self._ao_task.write(data)

    def close(self) -> None:
        self.teardown()

    def teardown(self) -> None:
        self._close_tasks_and_release(suppress_close_errors=False)
        if self._lease is None:
            self._ao_channel_order = []
            self._counter_route = None
            self._finite_repeat = None
            self._edge_paced = False

    def _close_tasks_and_release(self, *, suppress_close_errors: bool) -> None:
        """Close every configured NI task before releasing the task-scoped lease.

        A handle whose ``close`` raises is retained and its lease stays claimed so
        a later teardown can retry without advertising hardware resources as free.
        """
        close_errors: list[Exception] = []
        for attribute in ("_co_task", "_ao_task"):
            task = getattr(self, attribute)
            if task is None:
                continue
            try:
                task.close()
            except Exception as error:
                close_errors.append(error)
                self._log.warning("failed to close NI task during cleanup", exc_info=True)
            else:
                setattr(self, attribute, None)

        if self._co_task is None and self._ao_task is None and self._lease is not None:
            self._lease.release()
            self._lease = None

        if close_errors and not suppress_close_errors:
            raise close_errors[0]

    def start(self, repeat: int | None = None) -> None:
        if self._ao_task is None:
            raise RuntimeError("start() called before setup()")

        if self._co_task is not None:
            # Counter trigger — reprogram CO timing just-in-time. CONTINUOUS pumps
            # forever; FINITE stops after exactly `repeat` edges, which bounds the
            # retriggered AO task to N cycles. NI allows cfg_implicit_timing on an
            # idle task, so we flip between modes across start() calls without
            # rebuilding the CO task.
            if repeat is None:
                self._co_task.timing.cfg_implicit_timing(sample_mode=NiAcqType.CONTINUOUS)
            else:
                self._co_task.timing.cfg_implicit_timing(
                    sample_mode=NiAcqType.FINITE,
                    samps_per_chan=repeat,
                )
        elif repeat is not None:
            # Input trigger — no owned counter to bound the trigger count.
            # Implementing this would require a counter-gate: reserve a second
            # counter in FINITE edge-count mode on the trigger PFI and route its
            # InternalOutput as the AO start trigger, so the AO only arms for the
            # first N external edges. Not yet implemented.
            raise NotImplementedError(
                f"{self.uid}: input-trigger repeat bounding is not supported yet. "
                "Use repeat=None and count cycles externally, or switch to a counter trigger."
            )

        # AO arms first (waits for a trigger), then the optional CO task starts.
        self._ao_task.start()
        if self._co_task is not None:
            self._co_task.start()
        self._finite_repeat = repeat

    def wait_until_done(self, timeout_s: float) -> None:
        """Wait up to ``timeout_s`` for the counter's finite run, then for its last AO cycle.

        Raises:
            RuntimeError: No finite run is active, or the AO did not generate exactly
                one cycle per counter pulse (a pulse arrived while a cycle was still
                playing, or the last cycle did not finish within one cycle period
                plus 0.1 s).
        """
        if self._finite_repeat is None:
            raise RuntimeError(
                f"{self.uid}: wait_until_done requires a finite acquisition "
                "(start was called with repeat=None or not at all)"
            )
        # The CO task is the finite one when repeat is set; retriggerable AO tasks
        # never report done on their own.
        if self._co_task is None or self._ao_task is None:
            # Defensive: should not reach here — InputTrigger + repeat raises in start().
            raise RuntimeError(f"{self.uid}: no CO task to wait on (input-trigger path)")
        self._co_task.wait_until_done(timeout=timeout_s)
        # The counter's last pulse has started the last AO cycle, which may still be playing.
        expected = self._finite_repeat * self._cycle_samples
        generated = self._wait_for_generated(self._ao_task, lambda count: count >= expected)
        if generated != expected:
            raise RuntimeError(
                f"{self.uid}: AO generated {generated} samples per channel, expected {expected} "
                f"({self._finite_repeat} cycles of {self._cycle_samples})"
            )

    def stop(self) -> None:
        """Stop the counter, let the AO cycle in progress finish, then stop the AO task.

        NI AO tasks hold the last generated sample on the DAC after stop, so a whole
        cycle leaves every line on its rest sample. Ending an edge-paced run rebuilds
        the periodic counter task, so ``start`` behaves as before ``start_paced``.

        Raises:
            RuntimeError: The cycle did not finish within one cycle period plus 0.1 s.
                The AO task is stopped anyway, so the lines may not be at rest.
        """
        counter_error: Exception | None = None
        if self._co_task is not None:
            try:
                self._co_task.stop()
            except Exception as error:
                counter_error = error
        ao_task = self._ao_task
        if ao_task is None:
            self._finite_repeat = None
            if counter_error is not None:
                raise counter_error
            return
        try:
            # A retriggered AO task that was never triggered has generated zero samples.
            generated = self._wait_for_generated(ao_task, lambda count: count % self._cycle_samples == 0)
        finally:
            ao_task.stop()
            self._finite_repeat = None
            if self._edge_paced:
                self._restore_counter_trigger()
        if counter_error is not None:
            raise counter_error
        if generated % self._cycle_samples:
            raise RuntimeError(
                f"{self.uid}: AO output stopped mid-cycle after waiting "
                f"{self._cycle_period_s + _CYCLE_END_MARGIN_S:.3f} s; the lines may not be at rest"
            )

    def _wait_for_generated(self, ao_task: NiTask, reached: Callable[[int], bool]) -> int:
        """Poll the AO's generated-sample count until ``reached`` accepts it or one cycle
        period plus a margin elapses; return the last count read."""
        deadline = time.monotonic() + self._cycle_period_s + _CYCLE_END_MARGIN_S
        count = ao_task.out_stream.total_samp_per_chan_generated
        while not reached(count) and time.monotonic() < deadline:
            time.sleep(_CYCLE_END_POLL_S)
            count = ao_task.out_stream.total_samp_per_chan_generated
        return count

    # ---- edge pacing ----

    def start_paced(self, division: EdgeDivision) -> dict[str, float | int | str]:
        """Play ``division.cycles`` AO cycles, each started by dividing the edge source.

        The leased counter's periodic task is replaced by a finite pulse train clocked
        by the edge source's rising edges: the first pulse after ``initial_delay``
        edges, then one every ``divider`` edges. Its pulses reach the output PFI like
        the periodic ones. On failure the generator is left stopped with its periodic
        counter task.

        Returns:
            Readbacks after commit: ``filter_s`` (accepted minimum pulse width, 0 when
            unfiltered), ``filter_clock_s`` (filter timebase period, 0 when unfiltered),
            ``high_ticks``, ``low_ticks``, ``initial_delay_ticks``, ``source`` (counter
            timebase terminal), and ``ao_trigger_source``.

        Raises:
            RuntimeError: Not set up, no ``edge_source`` configured, already edge-paced,
                or the committed divider differs from the request.
        """
        ao_task = self._ao_task
        lease = self._lease
        pacing = self.edge_pacing
        if ao_task is None or lease is None:
            raise RuntimeError("start_paced() called before setup()")
        if pacing is None:
            raise RuntimeError(f"{self.uid}: start_paced requires a counter trigger with an edge_source")
        if self._edge_paced:
            raise RuntimeError(f"{self.uid}: an edge-paced run is already active; stop() it first")

        source_path = lease.input_pfi_paths[0]
        self._close_counter_task()
        try:
            divider, channel = self._create_edge_divider(source_path, pacing.filter_s, division)
            readback = self._read_edge_divider(channel, source_path, pacing.filter_s, division)
            readback["ao_trigger_source"] = ao_task.triggers.start_trigger.dig_edge_src
            # AO arms first (waits for a trigger), then the divider starts counting edges.
            ao_task.start()
            divider.start()
        except Exception:
            try:
                ao_task.stop()
            finally:
                self._restore_counter_trigger()
            raise
        self._finite_repeat = division.cycles
        return readback

    def _create_edge_divider(
        self, source_path: str, filter_s: float, division: EdgeDivision
    ) -> tuple[NiTask, COChannel]:
        """Create and commit the finite edge-divider CO task on the leased counter."""
        if self._counter_route is None:
            raise RuntimeError("edge divider requires a leased counter")
        counter_path, output_pfi_path = self._counter_route
        divider = NiTask(f"{self.uid}_divider")
        self._co_task = divider
        self._edge_paced = True
        channel = divider.co_channels.add_co_pulse_chan_ticks(
            counter_path,
            source_path,
            idle_state=Level.LOW,
            initial_delay=division.initial_delay,
            low_ticks=division.divider - _DIVIDER_HIGH_TICKS,
            high_ticks=_DIVIDER_HIGH_TICKS,
        )
        channel.co_pulse_term = output_pfi_path
        channel.co_ctr_timebase_active_edge = Edge.RISING
        channel.co_ctr_timebase_dig_fltr_enable = False
        if filter_s > 0:
            channel.co_ctr_timebase_dig_fltr_min_pulse_width = filter_s
            channel.co_ctr_timebase_dig_fltr_enable = True
        # DAQmx rejects the synchronizer and the digital filter enabled together
        # (-200770); an enabled filter qualifies edges on its own clock, so only an
        # unfiltered source is synchronized.
        channel.co_ctr_timebase_dig_sync_enable = filter_s == 0
        divider.timing.cfg_implicit_timing(sample_mode=NiAcqType.FINITE, samps_per_chan=division.cycles)
        divider.control(TaskMode.TASK_COMMIT)
        return divider, channel

    def _read_edge_divider(
        self, channel: COChannel, source_path: str, filter_s: float, division: EdgeDivision
    ) -> dict[str, float | int | str]:
        """Check the committed divider against the request and return its readbacks."""
        ticks = (channel.co_pulse_high_ticks, channel.co_pulse_low_ticks, channel.co_pulse_ticks_initial_delay)
        expected_ticks = (_DIVIDER_HIGH_TICKS, division.divider - _DIVIDER_HIGH_TICKS, division.initial_delay)
        for name, actual, expected in (
            ("high/low/initial-delay ticks", ticks, expected_ticks),
            ("timebase source", channel.co_ctr_timebase_src, source_path),
            ("timebase edge", channel.co_ctr_timebase_active_edge, Edge.RISING),
            ("filter enable", channel.co_ctr_timebase_dig_fltr_enable, filter_s > 0),
        ):
            if actual != expected:
                raise RuntimeError(f"{self.uid}: edge divider {name} read back {actual!r}, expected {expected!r}")
        filtered = filter_s > 0
        return {
            "filter_s": channel.co_ctr_timebase_dig_fltr_min_pulse_width if filtered else 0.0,
            "filter_clock_s": 1.0 / channel.co_ctr_timebase_dig_fltr_timebase_rate if filtered else 0.0,
            "high_ticks": ticks[0],
            "low_ticks": ticks[1],
            "initial_delay_ticks": ticks[2],
            "source": channel.co_ctr_timebase_src,
        }

    def _close_counter_task(self) -> None:
        if self._co_task is not None:
            self._co_task.close()
            self._co_task = None

    def _restore_counter_trigger(self) -> None:
        """Replace the edge divider with the periodic counter task built by ``setup``."""
        if self._counter_route is None:
            raise RuntimeError("periodic counter trigger requires a leased counter")
        self._close_counter_task()
        counter_path, output_pfi_path = self._counter_route
        self._create_counter_trigger(counter_path, 1.0 / self._cycle_period_s, output_pfi_path)
        self._edge_paced = False

    def can_hotswap(self, old: Signals | None, new: Signals) -> bool:
        """Always False for this driver — every load takes the full rebuild path.

        NI-DAQmx rejects writes to a task whose buffer was auto-sized by a prior
        write and then never started (error -200547). Supporting hot-swap cleanly
        would require either explicitly arming the AO task on first write and
        tracking its state across stop/start cycles, or bounded buffer rewrites
        gated on whether the task is running. Neither is worth the complexity
        while mid-run seamless updates aren't a product requirement; rebuilding
        on every load is a few ms of extra work and causes a brief output gap if
        the task was running.
        """
        del old, new
        return False


__all__ = [
    "CounterTrigger",
    "InputTrigger",
    "NiSignalGenerator",
    "Trigger",
]
