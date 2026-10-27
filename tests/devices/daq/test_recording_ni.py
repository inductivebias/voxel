"""NiPositionRecorder against fake NI tasks; no NI device is opened."""

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from nidaqmx.constants import AcquisitionType, Edge, EncoderType, LengthUnits, LineGrouping, TaskMode
from nidaqmx.error_codes import DAQmxErrors
from nidaqmx.errors import DaqError, DaqReadError

from vxl.devices.daq.hub_ni import NiDaqmx
from vxl.devices.daq.hub_ni import hub as hub_module
from vxl.devices.daq.recording import RecorderArm
from vxl.devices.daq.recording import ni as recording_ni
from vxl.devices.daq.recording.ni import NiPositionRecorder

RATE_HZ = 1000.0
MOTION = 50
SAMPLES = 600
REQUEST = RecorderArm(origin=10.0, edge_spacing=0.02, frames=4, spacing=0.1, first_exposure=(10.06, 10.1), speed=1.0)
MARKERS = {"camera_a": "pfi2", "camera_b": "pfi3"}


class FakeStream:
    def __init__(self, task: "FakeTask") -> None:
        self.task = task
        self.input_buf_size = 0

    @property
    def total_samp_per_chan_acquired(self) -> int:
        return self.task.ni.acquired

    @property
    def avail_samp_per_chan(self) -> int:
        return self.task.ni.acquired - self.task.read


class FakeTask:
    """One NI task: records configuration, enforces -200770 at commit, and serves scripted samples."""

    def __init__(self, ni: "FakeNi", name: str) -> None:
        self.ni = ni
        self.name = name
        self.channels: list[SimpleNamespace] = []
        self.ci_channels = SimpleNamespace(
            add_ci_lin_encoder_chan=self._add_encoder, add_ci_count_edges_chan=self._add_edges
        )
        self.di_channels = SimpleNamespace(add_di_chan=self._add_lines)
        self.timing = SimpleNamespace(cfg_samp_clk_timing=self._timing, samp_clk_rate=None, kwargs={})
        self.in_stream = FakeStream(self)
        self.read = 0
        self.running = False

    def _add_encoder(self, counter: str, **kwargs: object) -> SimpleNamespace:
        channel = SimpleNamespace(
            kind="encoder",
            counter=counter,
            kwargs=kwargs,
            ci_lin_encoder_dist_per_pulse=kwargs["dist_per_pulse"],
            ci_encoder_decoding_type=kwargs["decoding_type"],
        )
        for side in "ab":
            setattr(channel, f"ci_encoder_{side}_input_dig_fltr_timebase_src", "/Dev1/100kHzTimebase")
            setattr(channel, f"ci_encoder_{side}_input_dig_fltr_timebase_rate", 100_000.0)
        self.channels.append(channel)
        return channel

    def _add_edges(self, counter: str, **kwargs: object) -> SimpleNamespace:
        channel = SimpleNamespace(kind="edges", counter=counter, kwargs=kwargs)
        self.channels.append(channel)
        return channel

    def _add_lines(self, lines: str, **kwargs: object) -> SimpleNamespace:
        channel = SimpleNamespace(kind="lines", lines=lines, kwargs=kwargs)
        self.channels.append(channel)
        return channel

    def _timing(self, rate: float, **kwargs: object) -> None:
        self.timing.samp_clk_rate = rate
        self.timing.kwargs = kwargs

    def control(self, mode: TaskMode) -> None:
        assert mode == TaskMode.TASK_COMMIT
        for channel in self.channels:
            for side in "ab" if channel.kind == "encoder" else ():
                prefix = f"ci_encoder_{side}_input_dig"
                if getattr(channel, prefix + "_fltr_enable") and getattr(channel, prefix + "_sync_enable"):
                    raise DaqError(
                        "filter and synchronization both enabled", DAQmxErrors.DIG_FILTER_AND_SYNC_BOTH_ENABLED
                    )
        self.ni.events.append(("commit", self.name))
        if self.ni.on_commit is not None:
            self.ni.on_commit(self)

    def start(self) -> None:
        self.ni.events.append(("start", self.name))
        self.running = True

    def stop(self) -> None:
        self.ni.events.append(("stop", self.name))
        self.running = False

    def close(self) -> None:
        self.ni.events.append(("close", self.name))

    def serve(self, data: np.ndarray, samples: int) -> int:
        if self.ni.on_read is not None:
            self.ni.on_read(self)
        backlog = self.ni.acquired - self.read
        if backlog > self.in_stream.input_buf_size:
            raise DaqReadError("samples were overwritten", DAQmxErrors.SAMPLES_NO_LONGER_AVAILABLE, 0)
        if backlog < samples:
            raise DaqReadError("samples are not yet available", DAQmxErrors.SAMPLES_NOT_YET_AVAILABLE, 0)
        index = np.arange(self.read, self.read + samples)
        (channel,) = self.channels
        if channel.kind == "encoder":
            data[...] = self.ni.position(index)
        elif channel.kind == "edges":
            data[...] = np.searchsorted(self.ni.exposures.get(channel.ci_count_edges_term, ()), index, side="right")
        else:
            data[...] = 0
        self.read += samples
        self.ni.reads.append((self.name, samples))
        return samples


class FakeReader:
    def __init__(self, stream: FakeStream) -> None:
        self._task = stream.task

    def read_many_sample_double(self, data: np.ndarray, number_of_samples_per_channel: int, timeout: float) -> int:
        assert timeout > 0
        return self._task.serve(data, number_of_samples_per_channel)

    read_many_sample_uint32 = read_many_sample_double
    read_many_sample_port_byte = read_many_sample_double


class FakeNi:
    """The NI boundary: tasks, the shared sample clock the test advances, and the recorded signals."""

    def __init__(
        self,
        position: Callable[[np.ndarray], np.ndarray] | None = None,
        exposures: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.position = position or (lambda index: np.full(len(index), REQUEST.origin))
        self.exposures = dict(exposures or {})
        self.tasks: dict[str, FakeTask] = {}
        self.events: list[tuple[str, str]] = []
        self.reads: list[tuple[str, int]] = []
        self.acquired = 0
        self.on_commit: Callable[[FakeTask], None] | None = None
        self.on_read: Callable[[FakeTask], None] | None = None

    def task(self, name: str) -> FakeTask:
        task = FakeTask(self, name)
        self.tasks[name.removeprefix("recorder_")] = task
        return task

    def advance(self, samples: int) -> None:
        if self.tasks["clock"].running:
            self.acquired += samples


class FakeNiDevice:
    product_type = "PCIe-6738"

    def __init__(self, name: str) -> None:
        self.ao_physical_chans = SimpleNamespace(channel_names=[f"{name}/ao{i}" for i in range(32)])
        self.co_physical_chans = SimpleNamespace(channel_names=[f"{name}/ctr{i}" for i in range(4)])
        self.do_lines = SimpleNamespace(
            channel_names=[f"{name}/port0/line{i}" for i in range(2)] + [f"{name}/port1/line{i}" for i in range(8)]
        )

    def reset_device(self) -> None:
        pass


def _recorder(monkeypatch: pytest.MonkeyPatch, ni: FakeNi, **overrides: Any) -> tuple[NiDaqmx, NiPositionRecorder]:
    monkeypatch.setattr(hub_module, "NiDevice", FakeNiDevice)
    monkeypatch.setattr(hub_module, "NiSystem", SimpleNamespace(local=lambda: None))
    monkeypatch.setattr(recording_ni, "NiTask", ni.task)
    monkeypatch.setattr(recording_ni, "CounterReader", FakeReader)
    monkeypatch.setattr(recording_ni, "DigitalMultiChannelReader", FakeReader)
    hub = NiDaqmx("daq", device_name="Dev1")
    options: dict[str, Any] = {
        "hub": hub,
        "encoder_a": "pfi0",
        "encoder_b": "pfi1",
        "markers": MARKERS,
        "clock_lines": ["port0/line0", "port0/line1"],
        "sample_rate_hz": RATE_HZ,
    }
    return hub, NiPositionRecorder("recorder", **(options | overrides))


def _scan() -> FakeNi:
    """A constant-speed pass from the motion sample; camera_b exposes one sample after camera_a."""
    step = REQUEST.edge_spacing / 4

    def position(index: np.ndarray) -> np.ndarray:
        travel = np.maximum(index - MOTION, 0) / RATE_HZ * REQUEST.speed
        return REQUEST.origin + np.floor(travel / step) * step

    targets = REQUEST.first_exposure[0] + REQUEST.edge_spacing / 2 + np.arange(REQUEST.frames) * REQUEST.spacing
    first = np.searchsorted(position(np.arange(SAMPLES)), targets)
    return FakeNi(position, {"/Dev1/PFI2": first.tolist(), "/Dev1/PFI3": (first + 1).tolist()})


def _wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 2
    while not condition():
        assert time.monotonic() < deadline, "the recording worker did not get there in time"
        time.sleep(0.001)


def _closed_everything(hub: NiDaqmx, ni: FakeNi, recorder: NiPositionRecorder) -> bool:
    closed = {name for event, name in ni.events if event == "close"}
    return closed == {task.name for task in ni.tasks.values()} and not hub.assigned_pins and not recorder.is_recording


def test_arm_commits_every_route_then_starts_the_counters_before_their_clock(monkeypatch):
    ni = FakeNi()
    hub, recorder = _recorder(monkeypatch, ni)

    recorder.arm(REQUEST)

    (encoder,) = ni.tasks["position"].channels
    assert encoder.counter == "/Dev1/ctr0"
    assert encoder.kwargs == {
        "decoding_type": EncoderType.X_4,
        "zidx_enable": False,
        "units": LengthUnits.METERS,
        "dist_per_pulse": REQUEST.edge_spacing,
        "initial_pos": REQUEST.origin,
    }
    assert (encoder.ci_encoder_a_input_term, encoder.ci_encoder_b_input_term) == ("/Dev1/PFI0", "/Dev1/PFI1")
    for index, (counter, terminal) in enumerate((("/Dev1/ctr1", "/Dev1/PFI2"), ("/Dev1/ctr2", "/Dev1/PFI3"))):
        (edges,) = ni.tasks[f"marker{index}"].channels
        assert (edges.counter, edges.ci_count_edges_term) == (counter, terminal)
        assert edges.kwargs == {"edge": Edge.RISING, "initial_count": 0}
        assert edges.ci_count_edges_dig_fltr_enable is False
    (lines,) = ni.tasks["clock"].channels
    assert lines.lines == "/Dev1/port0/line0,/Dev1/port0/line1"
    assert lines.kwargs == {"line_grouping": LineGrouping.CHAN_PER_LINE}
    assert ni.tasks["clock"].timing.kwargs == {"sample_mode": AcquisitionType.CONTINUOUS, "samps_per_chan": 2000}
    for role in ("position", "marker0", "marker1"):
        assert ni.tasks[role].timing.kwargs["source"] == "/Dev1/di/SampleClock"
    assert {task.in_stream.input_buf_size for task in ni.tasks.values()} == {2000}
    assert ni.events == [
        *(("commit", f"recorder_{role}") for role in ("position", "marker0", "marker1", "clock")),
        *(("start", f"recorder_{role}") for role in ("position", "marker0", "marker1", "clock")),
    ]
    assert hub.assigned_pins == dict.fromkeys(["ctr0", "ctr1", "ctr2", "port0/line0", "port0/line1"], "recorder")
    assert recorder.is_recording
    with pytest.raises(RuntimeError, match="already recording"):
        recorder.arm(REQUEST)

    recorder.abort()
    assert next(name for event, name in ni.events if event == "stop") == "recorder_clock"
    assert _closed_everything(hub, ni, recorder)
    recorder.abort()


@pytest.mark.parametrize("filter_us", [0.0, 20.48])
def test_finish_associates_aligned_blocks_and_reports_the_accepted_filter(monkeypatch, filter_us):
    ni = _scan()

    def coerce_filter(task: FakeTask) -> None:
        for channel in task.channels:
            for side in "ab" if channel.kind == "encoder" and filter_us else ():
                setattr(channel, f"ci_encoder_{side}_input_dig_fltr_min_pulse_width", 20e-6)

    ni.on_commit = coerce_filter
    hub, recorder = _recorder(monkeypatch, ni, filter_us=filter_us)
    recorder.arm(REQUEST)
    ni.advance(MOTION)
    recorder.mark_motion()
    ni.advance(SAMPLES - MOTION)
    _wait_for(lambda: ni.tasks["clock"].read >= SAMPLES - 20)

    recorded = recorder.finish(timeout_s=1.0)

    encoder = ni.tasks["position"].channels[0]
    assert encoder.ci_encoder_a_input_dig_sync_enable is (filter_us == 0)
    assert encoder.ci_encoder_b_input_dig_sync_enable is (filter_us == 0)
    assert recorded.readback["filter_s"] == (20e-6 if filter_us else 0.0)
    assert recorded.readback["filter_clock_s"] == (1e-5 if filter_us else 0.0)
    assert recorded.readback["input_synchronization"] is (filter_us == 0)
    filter_budget_s = recorded.readback["filter_s"] + recorded.readback["filter_clock_s"] + 1 / RATE_HZ
    assert recorded.tolerance == pytest.approx(REQUEST.edge_spacing / 4 + REQUEST.speed * filter_budget_s)
    assert recorded.samples == SAMPLES
    assert recorded.motion_sample == MOTION
    for name, terminal in (("camera_a", "/Dev1/PFI2"), ("camera_b", "/Dev1/PFI3")):
        samples = np.array(ni.exposures[terminal])
        assert [frame.sample for frame in recorded.frames[name]] == samples.tolist()
        assert [frame.position for frame in recorded.frames[name]] == ni.position(samples).tolist()
    for role in ("position", "marker0", "marker1"):
        assert ni.tasks[role].read == SAMPLES, f"{role} read every sample"
    block_reads = [samples for name, samples in ni.reads if name == "recorder_clock"]
    assert block_reads, "the worker drained blocks while recording"
    assert set(block_reads) == {20}
    assert _closed_everything(hub, ni, recorder)


def test_partial_read_failure_ends_the_pass(monkeypatch):
    ni = _scan()

    def fail_marker_read(task: FakeTask) -> None:
        if task.name == "recorder_marker0":
            raise DaqError("read failed", DAQmxErrors.UNKNOWN)

    ni.on_read = fail_marker_read
    hub, recorder = _recorder(monkeypatch, ni)
    recorder.arm(REQUEST)
    ni.advance(100)
    _wait_for(lambda: ni.tasks["position"].read > 0)

    with pytest.raises(RuntimeError, match="the inputs are no longer aligned"):
        recorder.finish(timeout_s=1.0)

    assert ni.tasks["position"].read == 20
    assert ni.tasks["marker0"].read == 0
    assert _closed_everything(hub, ni, recorder)
    ni.on_read = None
    recorder.arm(REQUEST)
    recorder.abort()


def test_abort_never_closes_inputs_during_an_active_read(monkeypatch):
    ni = _scan()
    entered, release = threading.Event(), threading.Event()

    def block_first_read(_task: FakeTask) -> None:
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=2)

    ni.on_read = block_first_read
    hub, recorder = _recorder(monkeypatch, ni)
    recorder.arm(REQUEST)
    ni.advance(100)
    assert entered.wait(timeout=2)
    monkeypatch.setattr(recording_ni, "_STOP_WAIT_S", 0.01)

    try:
        with pytest.raises(RuntimeError, match="an input read is still active"):
            recorder.abort()
        assert not any(event in {"stop", "close"} for event, _ in ni.events)
        assert recorder.is_recording
        assert hub.assigned_pins
    finally:
        release.set()

    monkeypatch.setattr(recording_ni, "_STOP_WAIT_S", 2.0)
    recorder.abort()
    assert _closed_everything(hub, ni, recorder)


def test_buffer_overrun_fails_the_pass(monkeypatch):
    ni = _scan()
    hub, recorder = _recorder(monkeypatch, ni, buffer_s=0.05)
    recorder.arm(REQUEST)
    ni.advance(200)

    with pytest.raises(RuntimeError, match="the input buffer overran"):
        recorder.finish(timeout_s=1.0)

    assert _closed_everything(hub, ni, recorder)


def test_arm_coerced_clock_rate_closes_every_task_before_starting(monkeypatch):
    ni = FakeNi()

    def coerce_rate(task: FakeTask) -> None:
        if task.name == "recorder_clock":
            task.timing.samp_clk_rate = 999.0

    ni.on_commit = coerce_rate
    hub, recorder = _recorder(monkeypatch, ni)

    with pytest.raises(RuntimeError, match=r"coerced from 1000\.0 to 999\.0 Hz"):
        recorder.arm(REQUEST)

    assert not any(event == "start" for event, _ in ni.events)
    assert _closed_everything(hub, ni, recorder)
