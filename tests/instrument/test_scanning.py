import pytest

from vxl.devices.axes.continuous import ContinuousScanInfo
from vxl.devices.daq.clocked import EdgeDivision, EdgePacing, Signals
from vxl.hal import ContinuousScan
from vxl.instrument.scanning import ContinuousPass, plan_continuous_pass

SCAN = ContinuousScan(generator="daq", recorder="recorder")
# A 20 um encoder edge and 20 mm/s^2, so 1 mm/s needs a 25 um ramp.
INFO = ContinuousScanInfo(edge_spacing=20.0, acceleration=20_000.0, max_speed=4_500_000.0)
PACING = EdgePacing(source="pfi1", filter_s=20.48e-6)


def make_signals(duration: str = "65 ms", rest_time: str = "15 ms") -> Signals:
    return Signals.model_validate(
        {"sample_rate": "100 kHz", "duration": duration, "rest_time": rest_time, "waveforms": {}}
    )


def plan(
    *,
    z_step: float = 80.0,
    signals: Signals | None = None,
    info: ContinuousScanInfo = INFO,
    pacing: EdgePacing = PACING,
    limits: tuple[float, float] = (0.0, 210_000.0),
) -> tuple[ContinuousPass | None, list[str]]:
    return plan_continuous_pass(
        z_start=26_060.0,
        z_step=z_step,
        frames=50,
        signals=signals or make_signals(),
        scan=SCAN,
        info=info,
        pacing=pacing,
        limits=limits,
    )


def test_plan_continuous_pass_matches_the_validated_standalone_scan() -> None:
    # The standalone encoder-paced scan validated 50 frames 80 um apart at 1 mm/s from 26000 to 30160 um.
    planned, violations = plan()

    assert violations == []
    assert planned is not None
    assert planned.speed == pytest.approx(1000.0)
    assert planned.division == EdgeDivision(cycles=50, divider=4, initial_delay=4)
    assert planned.start == pytest.approx(26_000.0), "three edges of lead-in put the first plane at z_start"
    assert planned.end == pytest.approx(30_160.0)
    assert planned.first_exposure == pytest.approx((26_060.0, 26_145.0))
    assert planned.timeout_s == pytest.approx(18.32)


def test_plan_continuous_pass_lengthens_the_lead_in_to_cover_the_acceleration_ramp() -> None:
    slow_ramp = INFO.model_copy(update={"acceleration": 1_000.0})  # 500 um to reach 1 mm/s

    planned, violations = plan(info=slow_ramp)

    assert violations == []
    assert planned is not None
    assert planned.division.initial_delay == 27
    assert planned.start == pytest.approx(26_060.0 - 26 * 20.0)


@pytest.mark.parametrize(
    ("z_step", "advice"),
    [(70.0, "use 80"), (90.0, "use 80 or 100"), (60.0, "use 80")],
    ids=["between-4-and-5-edges", "between-whole-edges", "below-4-edges"],
)
def test_plan_continuous_pass_refuses_spacing_that_is_not_whole_edges(z_step: float, advice: str) -> None:
    rest = f"{z_step - 65:g} ms" if z_step > 65 else "15 ms"
    planned, violations = plan(z_step=z_step, signals=make_signals(rest_time=rest))

    assert planned is None
    assert any("not a whole number" in violation and advice in violation for violation in violations), violations


def test_plan_continuous_pass_reports_every_violation_together() -> None:
    planned, violations = plan(
        z_step=70.0,
        signals=make_signals(duration="76 ms", rest_time="4 ms"),
        info=INFO.model_copy(update={"max_speed": 500.0}),
        pacing=PACING.model_copy(update={"filter_s": 10e-3}),
        limits=(26_000.0, 27_000.0),
    )

    assert planned is None
    reasons = ["not a whole number", "beyond the scan axis", "do not fit between planes", "encoder filter", "leaves"]
    for reason in reasons:
        assert any(reason in violation for violation in violations), f"missing {reason!r} in {violations}"


def test_plan_continuous_pass_refuses_a_cycle_that_cannot_rearm_at_the_speed_tolerance() -> None:
    # 80 um per 80 ms is 12.5 frames/s, but 5 % fast leaves 76.2 ms between triggers.
    fits, _ = plan(signals=make_signals(duration="75 ms", rest_time="5 ms"))
    planned, violations = plan(signals=make_signals(duration="76 ms", rest_time="4 ms"))

    assert fits is not None
    assert planned is None
    assert any("do not fit between planes" in violation for violation in violations)
