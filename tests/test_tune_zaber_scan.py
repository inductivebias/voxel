"""Unit tests for scripts/tune_zaber_scan.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tune_zaber_scan as tzs  # type: ignore[import-not-found]


def _make_scan(
    path: Path,
    *,
    expected: int = 25,
    dt_s: float = 0.01,
    z_step: float = 1.0,
    marker_counts: dict[str, int] | None = None,
    violations: list[str] | None = None,
    warnings: list[str] | None = None,
    worst_reversal: float = 0.0,
    edge_spike_s: float = 0.0,
    extra_marker_dt_s: float | None = None,
) -> Path:
    def build_pts(step_s: float) -> list[dict[str, Any]]:
        t, pts = 0.0, [{"frame": 0, "sample": 0, "time_s": 0.0, "position": 0.0}]
        for i in range(1, expected):
            t += step_s + (edge_spike_s if i in (1, expected - 1) else 0.0)
            pts.append({"frame": i, "sample": i * 10, "time_s": t, "position": i * z_step})
        return pts

    frames: dict[str, list[dict[str, Any]]] = {"cam_a": build_pts(dt_s)}
    if extra_marker_dt_s is not None:
        frames["cam_b"] = build_pts(extra_marker_dt_s)
    payload = {
        "plan": {"z_step": z_step, "division": {"cycles": expected}},
        "recorded": {
            "frames": frames,
            "violations": violations or [],
            "warnings": warnings or [],
            "readback": {
                "marker_counts": marker_counts or {k: len(v) for k, v in frames.items()},
                "worst_reversal": {"distance": worst_reversal} if worst_reversal else None,
            },
        },
    }
    path.write_text(json.dumps(payload))
    return path


def test_analyze_scan_file_violations_extra_frames_reversals_and_edge_trim(tmp_path: Path) -> None:
    clean = tzs.analyze_scan_file(_make_scan(tmp_path / "scan.json", edge_spike_s=0.004))
    assert clean.passed
    assert clean.dropped_frames == clean.extra_frames == 0
    assert clean.steady_state_dt_std_ms < clean.dt_std_ms
    assert clean.max_spacing_error == pytest.approx(0.0)

    bad = tzs.analyze_scan_file(
        _make_scan(
            tmp_path / "scan.json",
            marker_counts={"cam_a": 27},
            violations=["spatial warn", "bad_samples"],
            warnings=["spatial warn"],
            worst_reversal=0.25,
        )
    )
    assert not bad.passed
    assert bad.extra_frames == 2
    assert bad.errors == ("bad_samples",)
    assert bad.warnings == ("spatial warn",)
    assert bad.worst_reversal_distance == pytest.approx(0.25)
    assert clean.score() < bad.score()

    multi = tzs.analyze_scan_file(_make_scan(tmp_path / "scan.json", dt_s=0.01, extra_marker_dt_s=0.015))
    assert multi.min_dt_ms == pytest.approx(10.0)
    assert multi.max_dt_ms == pytest.approx(15.0)
    assert not tzs.analyze_scan_file(_make_scan(tmp_path / "scan_failure.json")).passed


def test_oscilloscope_trace_validation_and_live_restore(tmp_path: Path) -> None:
    cmd = [float(i) for i in range(50)]
    m = tzs.analyze_oscilloscope_trace(cmd, [c - 0.1 for c in cmd], sample_rate_hz=1000.0)
    assert m.following_error_rms == pytest.approx(0.1)
    assert m.velocity_mean == pytest.approx(1000.0)
    assert m.worst_reversal_distance == pytest.approx(0.0)

    with pytest.raises(ValueError, match="stiffness"):
        tzs.run_oscilloscope_tune(
            host="127.0.0.1",
            axis_number=1,
            scan_start_mm=0.0,
            scan_end_mm=10.0,
            scan_speed_mm_s=1.0,
            load_mass_kg=1.0,
            stiffness_values=(1.5,),
            enable_feed_forward=True,
            repeats=1,
            apply_best=False,
            persist_paramset=None,
        )

    # Verify _snapshot_live_tuning + _restore_live_tuning restores original LIVE simple tuning instead of DEFAULT.
    calls: list[tuple[Any, ...]] = []
    ps_cls = SimpleNamespace(LIVE="live", STAGING="staging", DEFAULT="default")
    units = SimpleNamespace(INERTIA_KILOGRAMS="kg")
    param_cls = lambda *, name, value: (name, value)  # noqa: E731
    tuner = SimpleNamespace(
        get_simple_tuning=lambda _ps: SimpleNamespace(
            is_used=True,
            load_inertia=1.2,
            enable_feed_forward=False,
            tuning_params=[SimpleNamespace(name="Stiffness", value=0.65)],
        ),
        get_pid_tuning=lambda _ps: SimpleNamespace(p=10.0, i=1.0, d=0.1, fc=50.0),
        set_simple_tuning=lambda *a, **kw: calls.append(("simple", a, kw)),
        load_paramset=lambda to_ps, from_ps: calls.append(("load", to_ps, from_ps)),
    )
    snap = tzs._snapshot_live_tuning(tuner, ps_cls.LIVE)
    tzs._restore_live_tuning(tuner, snap, units, param_cls, ps_cls)
    assert calls[0][0] == "simple"
    assert calls[0][1] == ("staging", [("Stiffness", 0.65)])
    assert calls[0][2] == {"load_inertia": 1.2, "load_inertia_units": "kg", "enable_feed_forward": False}
    assert calls[1] == ("load", "live", "staging")
    assert tmp_path.exists()
