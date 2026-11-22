#!/usr/bin/env python3
"""Zaber ServoTuner & scan velocity-ripple analysis/tuning script.

Supports two workflows:
  1. Analyze scan.json / scan_failure.json locally (--analyze)
  2. Direct 1 kHz Zaber oscilloscope + ServoTuner stiffness sweep (--zaber-host)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

PERSIST_SLOTS: frozenset[str] = frozenset(str(i) for i in range(1, 10))


@dataclass(frozen=True)
class RippleMetrics:
    file: str
    expected_frames: int
    recorded_frames: int
    dropped_frames: int
    extra_frames: int
    min_dt_ms: float
    p05_dt_ms: float
    median_dt_ms: float
    p95_dt_ms: float
    max_dt_ms: float
    dt_span_ms: float
    dt_std_ms: float
    steady_state_dt_std_ms: float
    max_spacing_error: float | None
    worst_reversal_distance: float
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    passed: bool

    def score(self) -> tuple[int, int, int, float, float, float]:
        """Lower is better: passed, zero frame-count errors, zero reversals, lowest steady-state jitter."""
        return (
            0 if self.passed else 1,
            self.dropped_frames + self.extra_frames,
            1 if self.worst_reversal_distance > 0.0 else 0,
            self.steady_state_dt_std_ms,
            self.p95_dt_ms - self.p05_dt_ms,
            self.dt_span_ms,
        )


@dataclass(frozen=True)
class OscilloscopeMetrics:
    samples: int
    sample_rate_hz: float
    following_error_rms: float
    following_error_max: float
    velocity_mean: float
    velocity_std: float
    velocity_ripple_pct: float
    worst_reversal_distance: float

    def score(self) -> tuple[int, float, float, float]:
        """Lower is better: zero reversals, lowest following-error RMS, velocity ripple, peak error."""
        return (
            1 if self.worst_reversal_distance > 0.0 else 0,
            self.following_error_rms,
            self.velocity_ripple_pct,
            self.following_error_max,
        )


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _trim(values: Sequence[float], frac: float = 0.05) -> Sequence[float]:
    n = int(len(values) * frac)
    return values[n : len(values) - n] if n > 0 and len(values) - 2 * n >= 2 else values


def _max_reversal(positions: Sequence[float]) -> float:
    if len(positions) < 2:
        return 0.0
    sign = 1.0 if positions[-1] >= positions[0] else -1.0
    peak, worst = positions[0] * sign, 0.0
    for pos in positions[1:]:
        p = pos * sign
        if p > peak:
            peak = p
        elif peak - p > worst:
            worst = peak - p
    return worst


def _expected_spacing(data: dict[str, Any], expected: int) -> float | None:
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else data
    if not isinstance(plan, dict):
        return None
    if isinstance(plan.get("z_step"), (int, float)) and float(plan["z_step"]) > 0:
        return float(plan["z_step"])
    window = plan.get("window")
    if isinstance(window, list) and len(window) == 2 and expected > 0:
        span = abs(float(window[1]) - float(window[0]))
        return span / expected if span > 0 else None
    return None


def analyze_scan_file(path: Path) -> RippleMetrics:
    data = json.loads(path.read_text())
    expected = int(data["plan"]["division"]["cycles"]) if "plan" in data else int(data["division"]["cycles"])
    recorded = data.get("recorded") or {}
    frames_map: dict[str, list[dict[str, Any]]] = recorded.get("frames") or {}
    if not frames_map:
        raise ValueError(f"No recorded frames found in {path}")

    spacing = _expected_spacing(data, expected)
    readback = recorded.get("readback") or {}
    marker_counts = readback.get("marker_counts") or {}
    warnings = tuple(str(w) for w in (recorded.get("warnings") or ()))
    warning_set = set(warnings)
    errors = tuple(str(v) for v in (recorded.get("violations") or ()) if str(v) not in warning_set)

    counts = [int(marker_counts.get(name, len(pts))) for name, pts in frames_map.items()]
    dropped = max(0, expected - min(counts))
    extra = max(0, max(counts) - expected)

    wr = readback.get("worst_reversal")
    worst_reversal = (
        max(0.0, float(wr["distance"]))
        if isinstance(wr, dict) and isinstance(wr.get("distance"), (int, float))
        else 0.0
    )

    per_marker: list[tuple[float, float, float, float, float, float, float, float, float | None]] = []
    for pts in frames_map.values():
        if len(pts) < 2:
            continue
        times = [float(p["time_s"]) for p in pts]
        dts = [(times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1)]
        s_dts = sorted(dts)
        steady = _trim(dts)
        pos_err: float | None = None
        if spacing is not None and all("position" in p for p in pts):
            positions = [float(p["position"]) for p in pts]
            pos_err = max(abs(abs(positions[i + 1] - positions[i]) - spacing) for i in range(len(positions) - 1))
            worst_reversal = max(worst_reversal, _max_reversal(positions))
        per_marker.append(
            (
                s_dts[0],
                _percentile(s_dts, 5.0),
                statistics.median(s_dts),
                _percentile(s_dts, 95.0),
                s_dts[-1],
                s_dts[-1] - s_dts[0],
                statistics.stdev(dts) if len(dts) > 1 else 0.0,
                statistics.stdev(steady) if len(steady) > 1 else 0.0,
                pos_err,
            )
        )
    if not per_marker:
        raise ValueError(f"Insufficient frame timestamps in {path}")

    spacing_errs = [m[8] for m in per_marker if m[8] is not None]
    return RippleMetrics(
        file=str(path),
        expected_frames=expected,
        recorded_frames=min(counts),
        dropped_frames=dropped,
        extra_frames=extra,
        min_dt_ms=min(m[0] for m in per_marker),
        p05_dt_ms=min(m[1] for m in per_marker),
        median_dt_ms=statistics.median(m[2] for m in per_marker),
        p95_dt_ms=max(m[3] for m in per_marker),
        max_dt_ms=max(m[4] for m in per_marker),
        dt_span_ms=max(m[5] for m in per_marker),
        dt_std_ms=max(m[6] for m in per_marker),
        steady_state_dt_std_ms=max(m[7] for m in per_marker),
        max_spacing_error=max(spacing_errs) if spacing_errs else None,
        worst_reversal_distance=worst_reversal,
        errors=errors,
        warnings=warnings,
        passed=path.name == "scan.json" and not errors and dropped == 0 and extra == 0,
    )


def analyze_oscilloscope_trace(
    commanded_pos: Sequence[float],
    encoder_pos: Sequence[float],
    sample_rate_hz: float,
    *,
    trim_fraction: float = 0.1,
) -> OscilloscopeMetrics:
    if len(commanded_pos) != len(encoder_pos) or len(commanded_pos) < 4:
        raise ValueError("commanded_pos and encoder_pos must have equal length >= 4")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"sample_rate_hz must be finite and positive, got {sample_rate_hz}")

    cmd = _trim(commanded_pos, trim_fraction)
    enc = _trim(encoder_pos, trim_fraction)
    ferr = [c - e for c, e in zip(cmd, enc, strict=True)]
    vel = [(enc[i + 1] - enc[i]) * sample_rate_hz for i in range(len(enc) - 1)]
    v_mean = statistics.fmean(vel)
    v_std = statistics.stdev(vel) if len(vel) > 1 else 0.0
    return OscilloscopeMetrics(
        samples=len(cmd),
        sample_rate_hz=sample_rate_hz,
        following_error_rms=math.sqrt(statistics.fmean(e * e for e in ferr)),
        following_error_max=max(abs(e) for e in ferr),
        velocity_mean=v_mean,
        velocity_std=v_std,
        velocity_ripple_pct=(v_std / abs(v_mean) * 100.0) if abs(v_mean) > 1e-12 else 0.0,
        worst_reversal_distance=_max_reversal(enc),
    )


def _validate_sweep(
    *,
    load_mass_kg: float,
    stiffness_values: Sequence[float],
    repeats: int,
    apply_best: bool,
    persist_paramset: str | None,
    scan_speed_mm_s: float,
) -> None:
    if not math.isfinite(load_mass_kg) or load_mass_kg < 0:
        raise ValueError(f"load_mass_kg must be finite and >= 0, got {load_mass_kg}")
    if not stiffness_values or any(not math.isfinite(s) or not 0.0 <= s <= 1.0 for s in stiffness_values):
        raise ValueError(f"stiffness_values must be non-empty values in [0.0, 1.0], got {stiffness_values}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if not math.isfinite(scan_speed_mm_s) or scan_speed_mm_s <= 0:
        raise ValueError(f"scan_speed_mm_s must be finite and > 0, got {scan_speed_mm_s}")
    if persist_paramset is not None:
        if not apply_best:
            raise ValueError("--persist-paramset requires --apply-best")
        if persist_paramset.strip() not in PERSIST_SLOTS:
            raise ValueError(f"persist_paramset must be in '1'..'9', got {persist_paramset!r}")


def _snapshot_live_tuning(tuner: Any, live: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        simple = tuner.get_simple_tuning(live)
        snap["simple"] = {
            "is_used": bool(simple.is_used),
            "load_mass_kg": float(simple.load_inertia),
            "enable_feed_forward": bool(simple.enable_feed_forward),
            "params": {p.name: float(p.value) for p in simple.tuning_params},
        }
    with contextlib.suppress(Exception):
        pid = tuner.get_pid_tuning(live)
        snap["pid"] = {"p": float(pid.p), "i": float(pid.i), "d": float(pid.d), "fc": float(pid.fc)}
    return snap


def _restore_live_tuning(tuner: Any, snap: dict[str, Any], units: Any, param_cls: Any, ps_cls: Any) -> None:
    try:
        simple = snap.get("simple")
        if isinstance(simple, dict) and simple.get("is_used") and simple.get("params"):
            tuner.set_simple_tuning(
                ps_cls.STAGING,
                [param_cls(name=k, value=float(v)) for k, v in simple["params"].items()],
                load_inertia=float(simple["load_mass_kg"]),
                load_inertia_units=units.INERTIA_KILOGRAMS,
                enable_feed_forward=bool(simple.get("enable_feed_forward", True)),
            )
            tuner.load_paramset(ps_cls.LIVE, ps_cls.STAGING)
            return
        pid = snap.get("pid")
        if isinstance(pid, dict) and all(pid.get(k) is not None for k in ("p", "i", "d", "fc")):
            tuner.set_pid_tuning(ps_cls.STAGING, pid["p"], pid["i"], pid["d"], pid["fc"])
            tuner.load_paramset(ps_cls.LIVE, ps_cls.STAGING)
            return
    except Exception:  # fall back to factory DEFAULT if custom restore fails
        pass
    tuner.load_paramset(ps_cls.LIVE, ps_cls.DEFAULT)


def run_oscilloscope_tune(
    *,
    host: str,
    axis_number: int,
    scan_start_mm: float,
    scan_end_mm: float,
    scan_speed_mm_s: float,
    load_mass_kg: float,
    stiffness_values: Sequence[float],
    enable_feed_forward: bool,
    repeats: int,
    apply_best: bool,
    persist_paramset: str | None,
    backup_path: Path | None = None,
    sample_rate_hz: float = 1000.0,
) -> tuple[float | None, OscilloscopeMetrics]:
    _validate_sweep(
        load_mass_kg=load_mass_kg,
        stiffness_values=stiffness_values,
        repeats=repeats,
        apply_best=apply_best,
        persist_paramset=persist_paramset,
        scan_speed_mm_s=scan_speed_mm_s,
    )
    from zaber_motion import Units  # noqa: PLC0415
    from zaber_motion.ascii import Connection, ServoTuner, ServoTuningParam, ServoTuningParamset  # noqa: PLC0415

    hostname, _, port_s = host.partition(":")
    with Connection.open_tcp(hostname, int(port_s) if port_s else 55551) as conn:
        device = conn.detect_devices()[0]
        axis = device.get_axis(axis_number)
        tuner = ServoTuner(axis)
        stiff_name = next(
            (d.name for d in tuner.get_simple_tuning_param_definitions() if d.name.lower() == "stiffness"),
            "Stiffness",
        )
        orig_speed = axis.settings.get("maxspeed", Units.VELOCITY_MILLIMETRES_PER_SECOND)
        original = _snapshot_live_tuning(tuner, ServoTuningParamset.LIVE)
        if backup_path is not None:
            backup_path.write_text(json.dumps(original, indent=2) + "\n")

        def apply_stiffness(val: float) -> None:
            tuner.set_simple_tuning(
                ServoTuningParamset.STAGING,
                [ServoTuningParam(name=stiff_name, value=float(val))],
                load_inertia=float(load_mass_kg),
                load_inertia_units=Units.INERTIA_KILOGRAMS,
                enable_feed_forward=enable_feed_forward,
            )
            tuner.load_paramset(ServoTuningParamset.LIVE, ServoTuningParamset.STAGING)

        def restore() -> None:
            _restore_live_tuning(tuner, original, Units, ServoTuningParam, ServoTuningParamset)

        def capture(label: str) -> OscilloscopeMetrics:
            results: list[OscilloscopeMetrics] = []
            for _ in range(repeats):
                axis.settings.set("maxspeed", orig_speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)
                axis.move_absolute(scan_start_mm, Units.LENGTH_MILLIMETRES)
                axis.settings.set("maxspeed", scan_speed_mm_s, Units.VELOCITY_MILLIMETRES_PER_SECOND)
                scope = device.oscilloscope
                scope.clear()
                scope.set_timebase(1000.0 / sample_rate_hz, Units.TIME_MILLISECONDS)
                scope.add_channel(axis_number, "pos")
                scope.add_channel(axis_number, "encoder.pos")
                scope.start()
                axis.move_absolute(scan_end_mm, Units.LENGTH_MILLIMETRES)
                scope.stop()
                ch = scope.read()
                results.append(
                    analyze_oscilloscope_trace(
                        ch[0].get_data(Units.LENGTH_MICROMETRES),
                        ch[1].get_data(Units.LENGTH_MICROMETRES),
                        sample_rate_hz,
                    )
                )
            worst = max(results, key=lambda m: m.score())
            print(
                f"  [{label}] ferr_rms={worst.following_error_rms:.4f}um "
                f"ferr_max={worst.following_error_max:.4f}um ripple={worst.velocity_ripple_pct:.3f}% "
                f"rev={worst.worst_reversal_distance:.4g}"
            )
            return worst

        try:
            best_stiffness: float | None = None
            best_metrics = capture("baseline")
            for stiffness in stiffness_values:
                apply_stiffness(stiffness)
                try:
                    metrics = capture(f"stiff_{stiffness:.2f}")
                except Exception as exc:
                    print(f"  [FAULT] stiffness={stiffness:.2f} failed ({exc}); restoring live tuning", file=sys.stderr)
                    restore()
                    continue
                if metrics.score() < best_metrics.score():
                    best_stiffness, best_metrics = stiffness, metrics
        except BaseException:
            restore()
            raise
        finally:
            axis.settings.set("maxspeed", orig_speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)

        if apply_best and best_stiffness is not None:
            apply_stiffness(best_stiffness)
            if persist_paramset is not None:
                slot = getattr(ServoTuningParamset, f"P_{persist_paramset.strip()}")
                tuner.load_paramset(slot, ServoTuningParamset.LIVE)
                tuner.set_startup_paramset(slot)
        else:
            restore()
        return best_stiffness, best_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyze", type=Path, help="Analyze scan.json / scan_failure.json and exit")
    parser.add_argument("--zaber-host", help="Direct Zaber TCP host[:port] for 1 kHz oscilloscope sweep")
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--scan-start-mm", type=float)
    parser.add_argument("--scan-end-mm", type=float)
    parser.add_argument("--scan-speed-mm-s", type=float)
    parser.add_argument("--load-mass-kg", type=float, help="Physical payload mass on the stage in kg")
    parser.add_argument("--stiffness-values", default="0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--no-feed-forward", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--backup", type=Path, help="Optional path to save pre-sweep LIVE tuning snapshot")
    parser.add_argument("--apply-best", action="store_true")
    parser.add_argument("--persist-paramset", help="Persist winning tuning to slot 1..9 (requires --apply-best)")
    args = parser.parse_args(argv)

    if args.analyze is not None:
        print(json.dumps(asdict(analyze_scan_file(args.analyze)), indent=2))
        return 0

    if not args.zaber_host:
        parser.error("Provide --analyze <scan.json> or --zaber-host <host[:port]>")
    if None in (args.load_mass_kg, args.scan_start_mm, args.scan_end_mm, args.scan_speed_mm_s):
        parser.error("--zaber-host requires --load-mass-kg, --scan-start-mm, --scan-end-mm, --scan-speed-mm-s")

    stiffness_vals = tuple(float(x.strip()) for x in args.stiffness_values.split(",") if x.strip())
    run_oscilloscope_tune(
        host=args.zaber_host,
        axis_number=args.axis,
        scan_start_mm=args.scan_start_mm,
        scan_end_mm=args.scan_end_mm,
        scan_speed_mm_s=args.scan_speed_mm_s,
        load_mass_kg=args.load_mass_kg,
        stiffness_values=stiffness_vals,
        enable_feed_forward=not args.no_feed_forward,
        repeats=args.repeats,
        apply_best=args.apply_best,
        persist_paramset=args.persist_paramset,
        backup_path=args.backup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
