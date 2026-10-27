"""Hardware-free checks for NI signal generator behavior."""

from collections.abc import Iterable

import pytest

from vxl.devices.daq.clocked import ni
from vxl.devices.daq.clocked.ni import NiSignalGenerator
from vxl.devices.daq.clocked.waveform import validate_waveform

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
