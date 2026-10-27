"""Plan one constant-speed, encoder-paced pass of the scan axis for a volume.

The pass starts ``lead_in`` encoder edges before the volume's first plane so the axis reaches full speed
before the first cycle, and ends past the last plane by the stopping distance plus one cycle. The signal
generator starts a cycle every ``z_step / edge_spacing`` edges, so the planes are a whole number of encoder
edges apart and nothing about the spacing is rounded.
"""

import math
from dataclasses import dataclass

from pydantic import ValidationError

from vxl.devices.axes.continuous import ContinuousScanInfo
from vxl.devices.daq.clocked import EdgeDivision, EdgePacing, Signals
from vxl.hal.topology import ContinuousScan

# Travel after the last cycle ends before the axis may start decelerating.
_LEAD_OUT_S = 0.01
# Edges past the last cycle's trigger: two for its high ticks and one for the unknown starting phase.
_TRAILING_EDGES = 3
_MIN_TIMEOUT_S = 10.0
_WHOLE_EDGES_TOLERANCE = 1e-9
# EdgeDivision's smallest divider: two high ticks and at least two low ticks per cycle.
_MIN_DIVIDER = 4


@dataclass(frozen=True)
class ContinuousPass:
    """Where one pass starts and ends, how fast it moves, and how the generator divides the encoder."""

    start: float
    end: float
    speed: float
    division: EdgeDivision
    first_exposure: tuple[float, float]
    """Bounds of the first exposure's position: after ``initial_delay`` edges from an unknown phase."""
    timeout_s: float


def plan_continuous_pass(
    *,
    z_start: float,
    z_step: float,
    frames: int,
    signals: Signals,
    scan: ContinuousScan,
    info: ContinuousScanInfo,
    pacing: EdgePacing,
    limits: tuple[float, float],
) -> tuple[ContinuousPass | None, list[str]]:
    """Plan the pass that images ``frames`` planes ``z_step`` apart from ``z_start``.

    Returns:
        The pass, or None with every reason it cannot run (all are reported, not just the first).
    """
    period = float(signals.duration) + float(signals.rest_time)
    cycle = float(signals.duration)
    speed = z_step / period
    edge = info.edge_spacing
    fastest = speed * (1 + scan.speed_tolerance)
    violations: list[str] = []

    ratio = z_step / edge
    divider = round(ratio)
    if abs(ratio - divider) > _WHOLE_EDGES_TOLERANCE * ratio or divider < _MIN_DIVIDER:
        nearest = sorted({max(_MIN_DIVIDER, n) * edge for n in (math.floor(ratio), math.ceil(ratio))})
        violations.append(
            f"z_step {z_step:g} is not a whole number of at least {_MIN_DIVIDER} encoder edges ({edge:g} each); "
            f"use {' or '.join(f'{step:g}' for step in nearest)}"
        )
    if fastest > info.max_speed:
        violations.append(
            f"z_step {z_step:g} per {period:g} s is {speed:g}/s, beyond the scan axis's {info.max_speed:g}/s "
            f"with the {scan.speed_tolerance:.0%} speed tolerance"
        )
    if cycle + scan.rearm_margin_s >= z_step / fastest:
        violations.append(
            f"the {cycle:g} s cycle and {scan.rearm_margin_s:g} s re-arm margin do not fit between planes "
            f"{z_step / fastest:g} s apart at {scan.speed_tolerance:.0%} above the planned speed"
        )
    if pacing.filter_s >= edge / (4 * fastest):
        violations.append(
            f"the {pacing.filter_s * 1e6:g} us encoder filter rejects the {edge / (4 * fastest) * 1e6:g} us "
            "encoder transitions at this speed"
        )

    ramp = speed**2 / (2 * info.acceleration)
    initial_delay = max(scan.lead_in_edges, math.ceil(ramp / edge) + 2)
    start = z_start - (initial_delay - 1) * edge
    end = (
        start + (initial_delay + (frames - 1) * divider + _TRAILING_EDGES) * edge + ramp + speed * (cycle + _LEAD_OUT_S)
    )
    lower, upper = limits
    if start < lower or end > upper:
        violations.append(
            f"the pass from {start:g} to {end:g}, including lead-in and stopping distance, "
            f"leaves the scan axis range [{lower:g}, {upper:g}]"
        )
    try:
        division = EdgeDivision(cycles=frames, divider=max(divider, _MIN_DIVIDER), initial_delay=initial_delay)
    except ValidationError:
        violations.append(f"{frames} planes of {divider} edges exceed the generator's 32-bit counters")
        division = None

    if violations or division is None:
        return None, violations
    return (
        ContinuousPass(
            start=start,
            end=end,
            speed=speed,
            division=division,
            first_exposure=(start + (initial_delay - 1) * edge, start + initial_delay * edge + speed * cycle),
            timeout_s=_MIN_TIMEOUT_S + 2 * (end - start) / speed,
        ),
        [],
    )


__all__ = ["ContinuousPass", "plan_continuous_pass"]
