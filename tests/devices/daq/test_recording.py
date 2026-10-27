"""Exposure association, the simulated recorder, and the recorder's rig interface."""

import numpy as np
import pytest
from pydantic import ValidationError

from rigup import DeviceConfig, Rig, RigConfig
from vxl.devices.daq.recording import (
    PassAssociator,
    PositionRecorderHandle,
    RecordedPass,
    RecorderArm,
    SimulatedPositionRecorder,
)

RATE_HZ = 1000.0
MOTION = 50
REQUEST = RecorderArm(origin=10.0, edge_spacing=0.02, frames=4, spacing=0.1, first_exposure=(10.06, 10.1), speed=1.0)


def _ideal_pass(request: RecorderArm = REQUEST, samples: int = 600):
    """A quantized constant-speed pass; camera_b exposes one sample after camera_a."""
    step = request.edge_spacing / 4
    index = np.arange(samples)
    travel = np.maximum(index - MOTION, 0) / RATE_HZ * request.speed
    positions = request.origin + np.floor(travel / step) * step
    targets = request.first_exposure[0] + request.edge_spacing / 2 + np.arange(request.frames) * request.spacing
    first = np.searchsorted(positions, targets)
    edges = {"camera_a": first, "camera_b": first + 1}
    counts = {name: np.searchsorted(marks, index, side="right") for name, marks in edges.items()}
    return positions, counts, edges


def _associate(positions, counts, *, block: int | None = None, motion: int | None = MOTION) -> RecordedPass:
    associator = PassAssociator(REQUEST, list(counts), rate_hz=RATE_HZ, filter_s=0.0, filter_clock_s=0.0)
    if motion is not None:
        associator.mark_motion(motion)
    block = block or len(positions)
    for start in range(0, len(positions), block):
        stop = start + block
        associator.add_block(positions[start:stop], {name: count[start:stop] for name, count in counts.items()})
    return associator.result({"sample_rate_hz": RATE_HZ})


@pytest.mark.parametrize("block", [None, 1, 7, 64])
def test_pass_associator_any_block_split_returns_each_exposure_position(block):
    positions, counts, edges = _ideal_pass()

    recorded = _associate(positions, counts, block=block)

    assert recorded.samples == len(positions)
    assert recorded.motion_sample == MOTION
    assert recorded.tolerance == pytest.approx(0.005 + 1.0 / RATE_HZ)
    assert recorded.readback == {"sample_rate_hz": RATE_HZ}
    for name, samples in edges.items():
        frames = recorded.frames[name]
        assert [frame.frame for frame in frames] == [0, 1, 2, 3]
        assert [frame.sample for frame in frames] == samples.tolist()
        assert [frame.position for frame in frames] == positions[samples].tolist()
        assert [frame.time_s for frame in frames] == pytest.approx(samples / RATE_HZ)


def _drop_last(_positions, counts, edges):
    counts["camera_a"][edges["camera_a"][-1] :] -= 1


def _add_extra(_positions, counts, _edges):
    counts["camera_b"][-1] += 1


def _double_bin(_positions, counts, edges):
    counts["camera_a"][edges["camera_a"][0] :] += 1


def _before_baseline(_positions, counts, _edges):
    counts["camera_a"] += 1


def _before_motion(_positions, counts, edges):
    counts["camera_a"][MOTION : edges["camera_a"][0]] = 1


def _reversal(positions, _counts, _edges):
    positions[MOTION + 32] -= 0.01


def _irregular_spacing(positions, _counts, edges):
    positions[edges["camera_a"][2] :] += 0.02


def _late_first_exposure(positions, _counts, _edges):
    positions[MOTION + 1 :] += 0.1


def _nonfinite(positions, _counts, _edges):
    positions[10] = np.nan


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_drop_last, "camera_a: expected 4 exposures, recorded 3"),
        (_add_extra, "camera_b: expected 4 exposures, recorded 5"),
        (_double_bin, "camera_a: 1 samples changed the count by other than 0 or 1"),
        (_before_baseline, "camera_a: count was 1 at the first sample"),
        (_before_motion, "camera_a: 1 exposures at or before the motion sample 50"),
        (_reversal, "the encoder reversed by 0.01 "),
        (_irregular_spacing, "camera_a: 1 exposure gaps differ from the spacing 0.1"),
        (_late_first_exposure, "first exposure at 10.17"),
        (_nonfinite, "1 encoder samples are not finite"),
    ],
)
def test_pass_associator_invalid_pass_raises_naming_the_violation(corrupt, message):
    positions, counts, edges = _ideal_pass()
    corrupt(positions, counts, edges)
    with pytest.raises(ValueError, match="Recorded pass is invalid") as raised:
        _associate(positions, counts, block=64)
    assert message in str(raised.value)


def test_pass_associator_lists_every_violation_at_once():
    positions, counts, edges = _ideal_pass()
    _drop_last(positions, counts, edges)
    _reversal(positions, counts, edges)

    with pytest.raises(ValueError, match="Recorded pass is invalid") as raised:
        _associate(positions, counts, motion=None)

    message = str(raised.value)
    assert "the motion start was never marked" in message
    assert "camera_a: expected 4 exposures, recorded 3" in message
    assert "reversed" not in message, "without a motion start there is no reversal window"


def test_pass_associator_settling_after_the_last_exposure_is_not_a_reversal():
    positions, counts, edges = _ideal_pass()
    positions[edges["camera_b"][-1] + 2 :] -= 0.005
    assert len(_associate(positions, counts).frames["camera_b"]) == REQUEST.frames


def test_pass_associator_retains_at_most_frames_exposures_per_marker():
    associator = PassAssociator(REQUEST, ["noisy"], rate_hz=RATE_HZ, filter_s=0.0, filter_clock_s=0.0)
    associator.mark_motion(0)
    blocks = 100
    for start in range(0, blocks * 1000, 1000):
        index = np.arange(start, start + 1000)
        associator.add_block(np.full(1000, REQUEST.origin), {"noisy": index // 2})

    assert len(associator._tracks["noisy"].samples) == REQUEST.frames, "retained exposures are bounded"
    with pytest.raises(ValueError, match=f"expected 4 exposures, recorded {blocks * 500 - 1}"):
        associator.result({})


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"filter_s": 0.004, "filter_clock_s": 0.001}, "not shorter than the 0.005 s between encoder A/B"),
        ({"rate_hz": 15.0}, "fewer than two samples between exposures"),
    ],
)
def test_pass_associator_unresolvable_timing_raises(options, message):
    kwargs = {"rate_hz": RATE_HZ, "filter_s": 0.0, "filter_clock_s": 0.0} | options
    with pytest.raises(ValueError, match=message):
        PassAssociator(REQUEST, ["camera_a"], **kwargs)


def test_pass_associator_mark_motion_before_associated_samples_raises():
    associator = PassAssociator(REQUEST, ["camera_a"], rate_hz=RATE_HZ, filter_s=0.0, filter_clock_s=0.0)
    associator.add_block(np.full(10, REQUEST.origin), {"camera_a": np.zeros(10)})
    with pytest.raises(RuntimeError, match="precedes the 10 samples"):
        associator.mark_motion(9)
    associator.mark_motion(10)
    with pytest.raises(RuntimeError, match="already marked"):
        associator.mark_motion(11)


@pytest.mark.parametrize(
    "change",
    [{"first_exposure": (10.1, 10.06)}, {"speed": float("inf")}, {"frames": 0}, {"edge_spacing": 0.0}],
)
def test_recorder_arm_invalid_request_raises(change):
    with pytest.raises(ValidationError):
        RecorderArm.model_validate(REQUEST.model_dump() | change)


def test_simulated_recorder_finish_returns_the_ideal_ladder():
    recorder = SimulatedPositionRecorder("recorder", markers={"camera_a": "pfi2", "camera_b": "pfi3"})
    recorder.arm(REQUEST)
    assert recorder.is_recording
    recorder.mark_motion()

    recorded = recorder.finish(timeout_s=1.0)

    assert not recorder.is_recording
    targets = REQUEST.first_exposure[0] + REQUEST.edge_spacing / 2 + np.arange(REQUEST.frames) * REQUEST.spacing
    for frames in recorded.frames.values():
        positions = np.array([frame.position for frame in frames])
        assert np.all(positions >= targets - 1e-9)
        assert np.all(positions - targets <= REQUEST.edge_spacing / 4 + 1e-9)


def test_simulated_recorder_missing_frame_fails_count_and_spacing():
    recorder = SimulatedPositionRecorder("recorder", markers={"camera_a": "pfi2"}, missing_frames={"camera_a": [2]})
    recorder.arm(REQUEST)
    recorder.mark_motion()
    with pytest.raises(ValueError, match="Recorded pass is invalid") as raised:
        recorder.finish(timeout_s=1.0)
    assert "camera_a: expected 4 exposures, recorded 3" in str(raised.value)
    assert "camera_a: 1 exposure gaps differ" in str(raised.value)
    assert not recorder.is_recording


def test_simulated_recorder_lifecycle_rules():
    recorder = SimulatedPositionRecorder("recorder", markers={"camera_a": "pfi2"})
    with pytest.raises(RuntimeError, match="requires an armed pass"):
        recorder.finish(timeout_s=1.0)
    recorder.arm(REQUEST)
    with pytest.raises(RuntimeError, match="already recording"):
        recorder.arm(REQUEST)
    recorder.abort()
    recorder.abort()
    assert not recorder.is_recording
    recorder.arm(REQUEST)
    with pytest.raises(ValueError, match="the motion start was never marked"):
        recorder.finish(timeout_s=1.0)
    with pytest.raises(ValueError, match="unknown markers"):
        SimulatedPositionRecorder("recorder", markers={"camera_a": "pfi2"}, missing_frames={"camera_z": [0]})


async def test_position_recorder_handle_round_trip_through_rig():
    target = "vxl.devices.daq.recording.SimulatedPositionRecorder"
    markers = {"camera_a": "pfi2", "camera_b": "pfi3"}
    rig = Rig(
        RigConfig(
            devices={
                "recorder": DeviceConfig(target=target, init={"markers": markers, "filter_us": 1.0}),
                "lossy": DeviceConfig(target=target, init={"markers": markers, "missing_frames": {"camera_b": [0]}}),
            }
        )
    )
    await rig.open()
    try:
        assert not rig.build_errors
        recorder = PositionRecorderHandle.wrap(rig.devices["recorder"])
        assert (await recorder.interface()).type == "position_recorder"
        assert await recorder.get_markers() == markers
        assert await recorder.get_encoder_a() == "pfi1"
        assert await recorder.get_filter_s() == pytest.approx(1e-6)
        assert await recorder.get_state() == "idle"

        await recorder.arm(REQUEST)
        assert await recorder.get_state() == "recording"
        await recorder.mark_motion()
        recorded = await recorder.finish(timeout_s=1.0)
        assert isinstance(recorded, RecordedPass)
        assert {name: len(frames) for name, frames in recorded.frames.items()} == {"camera_a": 4, "camera_b": 4}
        assert await recorder.get_state() == "idle"

        await recorder.arm(REQUEST)
        await recorder.abort()
        assert await recorder.get_state() == "idle"

        lossy = PositionRecorderHandle.wrap(rig.devices["lossy"])
        await lossy.arm(REQUEST)
        await lossy.mark_motion()
        with pytest.raises(RuntimeError, match="camera_b: expected 4 exposures, recorded 3"):
            await lossy.finish(timeout_s=1.0)
        assert await lossy.get_state() == "idle"
    finally:
        await rig.close()
