"""Hardware-free checks for NI signal generator behavior."""

from collections.abc import Callable, Iterable
from unittest.mock import MagicMock

import pytest
from nidaqmx.constants import AcquisitionType, Edge, Level

from vxl.devices.daq.clocked import EdgeDivision, EdgePacing, ni
from vxl.devices.daq.clocked.ni import CounterTrigger, InputTrigger, NiSignalGenerator
from vxl.devices.daq.clocked.waveform import validate_waveform
from vxl.devices.daq.hub_ni import NiDaqModel, NiDaqmx
from vxl.devices.daq.hub_ni.resources import _NiResourceManager, capabilities_for_model

from .helpers import _signals, _triangle


class TestNiSignalGeneratorHelpers:
    """Pure-Python parts of ``NiSignalGenerator`` testable without NI hardware."""

    def test_can_hotswap_always_false(self):
        """Pinned to False: see can_hotswap docstring on NI-DAQmx error -200547."""
        generator = NiSignalGenerator.__new__(NiSignalGenerator)
        generator._ports = {"galvo": "ao0"}  # type: ignore[attr-defined]
        # Identical signals — would have been hotswappable before the buffer-error workaround.
        assert generator.can_hotswap(_signals(), _signals()) is False
        # Different waveforms — also False under the disabled-hotswap policy.
        assert generator.can_hotswap(_signals(), _signals(waveforms={"galvo": validate_waveform(_triangle())})) is False

    def test_can_hotswap_false_when_nothing_loaded(self):
        generator = NiSignalGenerator.__new__(NiSignalGenerator)
        generator._ports = {"galvo": "ao0"}  # type: ignore[attr-defined]
        assert generator.can_hotswap(None, _signals()) is False

    def test_start_input_trigger_with_repeat_raises(self):
        # InputTrigger + repeat requires counter-gate hardware support that
        # isn't implemented yet — start() must raise cleanly rather than silently
        # ignore the bound. No CO task = input-triggered operation.
        generator = NiSignalGenerator.__new__(NiSignalGenerator)
        generator.uid = "ao_test"  # type: ignore[attr-defined]
        generator._ao_task = object()  # type: ignore[attr-defined]  # truthy: passes "is not None" check
        generator._co_task = None  # type: ignore[attr-defined]
        with pytest.raises(NotImplementedError, match="input-trigger repeat"):
            generator.start(repeat=10)

    def test_wait_until_done_raises_without_finite_repeat(self):
        generator = NiSignalGenerator.__new__(NiSignalGenerator)
        generator.uid = "ao_test"  # type: ignore[attr-defined]
        generator._finite_repeat = None  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="finite acquisition"):
            generator.wait_until_done(timeout_s=1.0)


class _FakeOutStream:
    """Reports successive generated-sample counts, repeating the last one."""

    def __init__(self, counts: Iterable[int]) -> None:
        self._counts = list(counts)
        self.reads = 0

    @property
    def total_samp_per_chan_generated(self) -> int:
        count = self._counts[min(self.reads, len(self._counts) - 1)]
        self.reads += 1
        return count


class _FakeStoppableTask:
    def __init__(self, name: str, events: list[tuple[str, str]], counts: Iterable[int] = (0,)) -> None:
        self._name = name
        self._events = events
        self.out_stream = _FakeOutStream(counts)

    def stop(self) -> None:
        self._events.append(("stop", self._name))


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _running_generator(
    ao_counts: Iterable[int], events: list[tuple[str, str]]
) -> tuple[NiSignalGenerator, _FakeStoppableTask]:
    """A generator with 100-sample, 10 ms cycles whose AO reports ``ao_counts``."""
    generator = NiSignalGenerator.__new__(NiSignalGenerator)
    generator.uid = "ao_test"  # type: ignore[attr-defined]
    ao_task = _FakeStoppableTask("ao", events, ao_counts)
    generator._ao_task = ao_task  # type: ignore[attr-defined]
    generator._co_task = _FakeStoppableTask("co", events)  # type: ignore[attr-defined]
    generator._cycle_samples = 100  # type: ignore[attr-defined]
    generator._cycle_period_s = 0.01  # type: ignore[attr-defined]
    generator._finite_repeat = None  # type: ignore[attr-defined]
    generator._edge_paced = False  # type: ignore[attr-defined]
    return generator, ao_task


class TestNiSignalGeneratorStop:
    def test_stop_running_cycle_stops_counter_before_ao(self):
        events: list[tuple[str, str]] = []
        generator, _ = _running_generator([200], events)
        generator.stop()
        assert events == [("stop", "co"), ("stop", "ao")]

    def test_stop_mid_cycle_waits_until_count_is_whole_cycles(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ni, "time", _FakeClock())
        events: list[tuple[str, str]] = []
        generator, ao_task = _running_generator([130, 170, 199, 200], events)
        generator.stop()
        assert ao_task.out_stream.reads == 4, "AO must not stop before the fourth read reports a whole cycle"
        assert events[-1] == ("stop", "ao")

    def test_stop_never_triggered_ao_stops_at_once(self, monkeypatch: pytest.MonkeyPatch):
        clock = _FakeClock()
        monkeypatch.setattr(ni, "time", clock)
        events: list[tuple[str, str]] = []
        generator, ao_task = _running_generator([0], events)
        generator.stop()
        assert ao_task.out_stream.reads == 1
        assert clock.now == 0.0
        assert events == [("stop", "co"), ("stop", "ao")]

    def test_stop_cycle_never_ends_stops_ao_then_raises(self, monkeypatch: pytest.MonkeyPatch):
        clock = _FakeClock()
        monkeypatch.setattr(ni, "time", clock)
        events: list[tuple[str, str]] = []
        generator, _ = _running_generator([150], events)
        with pytest.raises(RuntimeError, match="may not be at rest"):
            generator.stop()
        assert events == [("stop", "co"), ("stop", "ao")]
        assert clock.now == pytest.approx(0.01 + 0.1, abs=0.002), "wait is bounded by one cycle period plus 0.1 s"


class _FakeNiTasks:
    """Stands in for ``nidaqmx.task.Task``: records task lifecycle and reads configuration back.

    ``commit_hook(channel)`` runs on each committed edge-divider channel, e.g. to emulate coercion.
    """

    def __init__(self, commit_hook: Callable[[MagicMock], None] | None = None) -> None:
        self.created: list[tuple[str, MagicMock]] = []
        self.events: list[tuple[str, str]] = []
        self._commit_hook = commit_hook

    def __call__(self, name: str) -> MagicMock:
        task = MagicMock(name=name)
        task.start.side_effect = lambda: self.events.append(("start", name))
        task.stop.side_effect = lambda: self.events.append(("stop", name))
        task.close.side_effect = lambda: self.events.append(("close", name))
        task.out_stream.total_samp_per_chan_generated = 0
        task.triggers.start_trigger.cfg_dig_edge_start_trig.side_effect = lambda trigger_source, **_: setattr(
            task.triggers.start_trigger, "dig_edge_src", trigger_source
        )

        def add_ticks(_counter: str, source: str, **kwargs: object) -> MagicMock:
            channel = task.co_channels.add_co_pulse_chan_ticks.return_value
            channel.co_pulse_high_ticks = kwargs["high_ticks"]
            channel.co_pulse_low_ticks = kwargs["low_ticks"]
            channel.co_pulse_ticks_initial_delay = kwargs["initial_delay"]
            channel.co_ctr_timebase_src = source
            channel.co_ctr_timebase_dig_fltr_timebase_rate = 100_000.0
            return channel

        def commit(_mode: object) -> None:
            channel = task.co_channels.add_co_pulse_chan_ticks.return_value
            # NI rejects the timebase digital filter and synchronizer enabled together (-200770).
            if channel.co_ctr_timebase_dig_fltr_enable is True and channel.co_ctr_timebase_dig_sync_enable is True:
                raise RuntimeError("-200770: digital filter and synchronization both enabled")
            if self._commit_hook is not None:
                self._commit_hook(channel)

        task.co_channels.add_co_pulse_chan_ticks.side_effect = add_ticks
        task.control.side_effect = commit
        self.created.append((name, task))
        return task

    def latest(self, name: str) -> MagicMock:
        return next(task for task_name, task in reversed(self.created) if task_name == name)

    def count(self, name: str) -> int:
        return sum(task_name == name for task_name, _ in self.created)


def _ni_generator(
    monkeypatch: pytest.MonkeyPatch,
    trigger: CounterTrigger | InputTrigger,
    commit_hook: Callable[[MagicMock], None] | None = None,
) -> tuple[NiSignalGenerator, _FakeNiTasks]:
    """A generator on a hardware-free PCIe-6738 allocator, not yet set up."""
    fake = _FakeNiTasks(commit_hook)
    monkeypatch.setattr(ni, "NiTask", fake)
    monkeypatch.setattr(ni, "time", _FakeClock())
    hub = NiDaqmx.__new__(NiDaqmx)
    hub._device_name = "Dev1"
    hub._resource_manager = _NiResourceManager(
        device_name="Dev1", capabilities=capabilities_for_model(NiDaqModel.NI6738)
    )
    generator = NiSignalGenerator("ao_test", hub=hub, ports={"galvo": "ao0", "etl": "ao1"}, trigger=trigger)
    return generator, fake


def _set_up_ni_generator(
    monkeypatch: pytest.MonkeyPatch,
    trigger: CounterTrigger | InputTrigger,
    commit_hook: Callable[[MagicMock], None] | None = None,
) -> tuple[NiSignalGenerator, _FakeNiTasks]:
    """A set-up generator with 100-sample, 10 ms cycles."""
    generator, fake = _ni_generator(monkeypatch, trigger, commit_hook)
    generator.setup(_signals())
    return generator, fake


_EDGE_TRIGGER = CounterTrigger(edge_source="pfi1", edge_filter_us=5.12)
_DIVISION = EdgeDivision(cycles=3, divider=5, initial_delay=10)


class TestNiSignalGeneratorEdgePacing:
    def test_edge_pacing_with_edge_source_reports_source_and_filter_seconds(self, monkeypatch: pytest.MonkeyPatch):
        generator, _ = _ni_generator(monkeypatch, _EDGE_TRIGGER)
        pacing = generator.edge_pacing
        assert isinstance(pacing, EdgePacing)
        assert pacing.source == "pfi1"
        assert pacing.filter_s == pytest.approx(5.12e-6)
        assert _ni_generator(monkeypatch, CounterTrigger())[0].edge_pacing is None
        assert _ni_generator(monkeypatch, InputTrigger(source="pfi1"))[0].edge_pacing is None

    @pytest.mark.parametrize(("edge_source", "message"), [("pfi99", "Unknown PFI"), ("ao3", "PFI terminal")])
    def test_setup_invalid_edge_source_raises_before_creating_tasks(
        self, monkeypatch: pytest.MonkeyPatch, edge_source: str, message: str
    ):
        generator, fake = _ni_generator(monkeypatch, CounterTrigger(edge_source=edge_source))
        with pytest.raises(ValueError, match=message):
            generator.setup(_signals())
        assert fake.created == []

    def test_start_paced_divides_edge_source_into_finite_cycles(self, monkeypatch: pytest.MonkeyPatch):
        generator, fake = _set_up_ni_generator(monkeypatch, _EDGE_TRIGGER)
        readback = generator.start_paced(_DIVISION)

        divider = fake.latest("ao_test_divider")
        add_ticks = divider.co_channels.add_co_pulse_chan_ticks
        assert add_ticks.call_args.args == ("/Dev1/ctr0", "/Dev1/PFI1")
        assert add_ticks.call_args.kwargs == {
            "idle_state": Level.LOW,
            "initial_delay": 10,
            "low_ticks": 3,
            "high_ticks": 2,
        }
        assert add_ticks.return_value.co_pulse_term == "/Dev1/PFI7", "divider pulses reach the leased output PFI"
        assert add_ticks.return_value.co_ctr_timebase_active_edge == Edge.RISING
        divider.timing.cfg_implicit_timing.assert_called_once_with(sample_mode=AcquisitionType.FINITE, samps_per_chan=3)
        assert fake.events == [
            ("close", "ao_test_clock"),
            ("start", "ao_test_ao"),
            ("start", "ao_test_divider"),
        ], "the periodic task is closed first and the AO armed before the divider counts edges"
        assert readback == {
            "filter_s": pytest.approx(5.12e-6),
            "filter_clock_s": pytest.approx(1e-5),
            "high_ticks": 2,
            "low_ticks": 3,
            "initial_delay_ticks": 10,
            "source": "/Dev1/PFI1",
            "ao_trigger_source": "/Dev1/CTR0InternalOutput",
        }

    def test_start_paced_unfiltered_source_enables_synchronization(self, monkeypatch: pytest.MonkeyPatch):
        generator, fake = _set_up_ni_generator(monkeypatch, CounterTrigger(edge_source="pfi1"))
        readback = generator.start_paced(_DIVISION)
        channel = fake.latest("ao_test_divider").co_channels.add_co_pulse_chan_ticks.return_value
        assert channel.co_ctr_timebase_dig_fltr_enable is False
        assert channel.co_ctr_timebase_dig_sync_enable is True
        assert (readback["filter_s"], readback["filter_clock_s"]) == (0.0, 0.0)

    def test_start_paced_filtered_source_disables_synchronization(self, monkeypatch: pytest.MonkeyPatch):
        generator, fake = _set_up_ni_generator(monkeypatch, _EDGE_TRIGGER)
        generator.start_paced(_DIVISION)
        channel = fake.latest("ao_test_divider").co_channels.add_co_pulse_chan_ticks.return_value
        assert channel.co_ctr_timebase_dig_fltr_enable is True
        assert channel.co_ctr_timebase_dig_fltr_min_pulse_width == pytest.approx(5.12e-6)
        assert channel.co_ctr_timebase_dig_sync_enable is False

    def test_start_paced_without_edge_source_raises(self, monkeypatch: pytest.MonkeyPatch):
        generator, fake = _set_up_ni_generator(monkeypatch, CounterTrigger())
        with pytest.raises(RuntimeError, match="edge_source"):
            generator.start_paced(_DIVISION)
        assert fake.events == []

    def test_start_paced_tick_readback_mismatch_restores_periodic_task(self, monkeypatch: pytest.MonkeyPatch):
        def coerce_low_ticks(channel: MagicMock) -> None:
            channel.co_pulse_low_ticks = 4

        generator, fake = _set_up_ni_generator(monkeypatch, _EDGE_TRIGGER, commit_hook=coerce_low_ticks)
        with pytest.raises(RuntimeError, match="ticks"):
            generator.start_paced(_DIVISION)

        assert not any(event == "start" for event, _ in fake.events), "nothing may start after a mismatch"
        assert ("close", "ao_test_divider") in fake.events
        assert fake.count("ao_test_clock") == 2, "the periodic counter task is rebuilt"
        periodic = fake.latest("ao_test_clock")
        periodic.co_channels.add_co_pulse_chan_freq.assert_called_once_with("/Dev1/ctr0", freq=100.0, duty_cycle=0.5)
        assert periodic.co_channels.add_co_pulse_chan_freq.return_value.co_pulse_term == "/Dev1/PFI7"
        generator.start()
        assert fake.events[-2:] == [("start", "ao_test_ao"), ("start", "ao_test_clock")]

    def test_stop_paced_run_restores_periodic_task(self, monkeypatch: pytest.MonkeyPatch):
        generator, fake = _set_up_ni_generator(monkeypatch, _EDGE_TRIGGER)
        generator.start_paced(_DIVISION)
        generator.stop()

        assert fake.events[-3:] == [
            ("stop", "ao_test_divider"),
            ("stop", "ao_test_ao"),
            ("close", "ao_test_divider"),
        ]
        periodic = fake.latest("ao_test_clock")
        assert fake.count("ao_test_clock") == 2
        periodic.timing.cfg_implicit_timing.assert_called_once_with(sample_mode=AcquisitionType.CONTINUOUS)
        generator.start(repeat=2)
        periodic.timing.cfg_implicit_timing.assert_called_with(sample_mode=AcquisitionType.FINITE, samps_per_chan=2)
        assert fake.events[-1] == ("start", "ao_test_clock")

    def test_wait_until_done_paced_run_all_cycles_generated_returns(self, monkeypatch: pytest.MonkeyPatch):
        generator, fake = _set_up_ni_generator(monkeypatch, _EDGE_TRIGGER)
        generator.start_paced(_DIVISION)
        fake.latest("ao_test_ao").out_stream.total_samp_per_chan_generated = 300
        generator.wait_until_done(timeout_s=2.0)
        fake.latest("ao_test_divider").wait_until_done.assert_called_once_with(timeout=2.0)

    def test_wait_until_done_paced_run_missing_cycle_raises(self, monkeypatch: pytest.MonkeyPatch):
        generator, fake = _set_up_ni_generator(monkeypatch, _EDGE_TRIGGER)
        generator.start_paced(_DIVISION)
        fake.latest("ao_test_ao").out_stream.total_samp_per_chan_generated = 200
        with pytest.raises(RuntimeError, match="generated 200 samples per channel, expected 300"):
            generator.wait_until_done(timeout_s=2.0)

    def test_wait_until_done_finite_start_missing_cycle_raises(self, monkeypatch: pytest.MonkeyPatch):
        generator, fake = _set_up_ni_generator(monkeypatch, CounterTrigger())
        generator.start(repeat=4)
        fake.latest("ao_test_ao").out_stream.total_samp_per_chan_generated = 300
        with pytest.raises(RuntimeError, match="expected 400"):
            generator.wait_until_done(timeout_s=2.0)
