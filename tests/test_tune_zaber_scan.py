"""Unit tests for scripts/tune_zaber_scan.py."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any, Self

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tune_zaber_scan.py"
_SPEC = importlib.util.spec_from_file_location("tune_zaber_scan", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
tune = importlib.util.module_from_spec(_SPEC)
sys.modules["tune_zaber_scan"] = tune
_SPEC.loader.exec_module(tune)


def _make_frames(
    count: int,
    *,
    dt_s: float = 0.01,
    spacing: float = 2.0,
    start_time: float = 1.0,
    start_pos: float = 10.0,
    dt_overrides: dict[int, float] | None = None,
    pos_overrides: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    t = start_time
    p = start_pos
    for i in range(count):
        if i > 0:
            step_dt = dt_overrides.get(i - 1, dt_s) if dt_overrides else dt_s
            t += step_dt
            p += spacing
        pos = pos_overrides.get(i, p) if pos_overrides else p
        frames.append({"frame": i, "sample": i * 10, "time_s": t, "position": pos})
    return frames


def _write_scan(
    path: Path,
    *,
    expected: int = 40,
    frames: dict[str, list[dict[str, Any]]] | None = None,
    marker_counts: dict[str, int] | None = None,
    violations: list[str] | None = None,
    warnings: list[str] | None = None,
    worst_reversal: dict[str, Any] | None = None,
    spacing: float = 2.0,
    top_error: str | None = None,
) -> Path:
    if frames is None:
        frames = {"cam_a": _make_frames(expected, spacing=spacing)}
    readback: dict[str, Any] = {
        "marker_counts": marker_counts if marker_counts is not None else {k: len(v) for k, v in frames.items()},
        "worst_reversal": worst_reversal or {},
    }
    payload: dict[str, Any] = {
        "line_scan": {"spacing": spacing},
        "plan": {
            "z_step": spacing,
            "division": {"cycles": expected, "divider": 10, "initial_delay": 2},
        },
        "recorded": {
            "frames": frames,
            "samples": expected * 10,
            "motion_sample": 0,
            "tolerance": 0.25,
            "readback": readback,
            "violations": violations or [],
            "warnings": warnings or [],
        },
    }
    if top_error is not None:
        payload["error"] = top_error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class TestAnalyzeScanFile:
    def test_warnings_only_still_passes(self, tmp_path: Path) -> None:
        warn = "cam_a: 1 exposure gaps differ from the spacing 2 by more than 0.25"
        scan_path = _write_scan(tmp_path / "scan.json", violations=[warn], warnings=[warn])
        metrics = tune.analyze_scan_file(scan_path)
        assert metrics.passed is True
        assert metrics.errors == ()
        assert metrics.warnings == (warn,)
        assert metrics.dropped_frames == 0
        assert metrics.extra_frames == 0

    def test_fatal_violation_fails_even_without_errors_key(self, tmp_path: Path) -> None:
        fatal = "cam_a: count was 1 at the first sample"
        warn = "cam_a: first exposure outside window"
        scan_path = _write_scan(tmp_path / "scan.json", violations=[fatal, warn], warnings=[warn])
        metrics = tune.analyze_scan_file(scan_path)
        assert metrics.passed is False
        assert metrics.errors == (fatal,)
        assert metrics.warnings == (warn,)

    def test_extra_frames_detected_from_readback_marker_counts(self, tmp_path: Path) -> None:
        # _MarkerTrack caps retained frames at expected=40, while readback.marker_counts records 43.
        scan_path = _write_scan(
            tmp_path / "scan.json",
            expected=40,
            frames={"cam_a": _make_frames(40)},
            marker_counts={"cam_a": 43},
        )
        metrics = tune.analyze_scan_file(scan_path)
        assert metrics.passed is False
        assert metrics.dropped_frames == 0
        assert metrics.extra_frames == 3

    def test_reversals_and_spacing_error(self, tmp_path: Path) -> None:
        frames = _make_frames(20, spacing=2.0, pos_overrides={10: 27.5})  # nominal at i=10 is 30.0 -> reversal 2.5
        scan_path = _write_scan(
            tmp_path / "scan.json",
            expected=20,
            frames={"cam_a": frames},
            worst_reversal={"distance": 2.5},
            spacing=2.0,
        )
        metrics = tune.analyze_scan_file(scan_path)
        assert metrics.worst_reversal_distance == pytest.approx(2.5)
        assert metrics.max_spacing_error == pytest.approx(2.5)

    def test_steady_state_trim_ignores_edge_outliers(self, tmp_path: Path) -> None:
        # 41 frames -> 40 intervals; 5% trim drops 2 intervals on each end.
        frames = _make_frames(41, dt_s=0.010, dt_overrides={0: 0.025, 39: 0.002})
        scan_path = _write_scan(tmp_path / "scan.json", expected=41, frames={"cam_a": frames})
        metrics = tune.analyze_scan_file(scan_path)
        assert metrics.dt_span_ms == pytest.approx(23.0)
        assert metrics.dt_std_ms > 2.0
        assert metrics.steady_state_dt_std_ms == pytest.approx(0.0, abs=1e-9)
        assert metrics.p95_dt_ms - metrics.p05_dt_ms == pytest.approx(0.0, abs=1e-9)

    def test_multi_marker_not_blended_and_reports_worst_case(self, tmp_path: Path) -> None:
        # Marker A has constant 10ms dt; marker B has constant 20ms dt -> blending would report 10ms span!
        frames = {
            "cam_a": _make_frames(20, dt_s=0.010),
            "cam_b": _make_frames(20, dt_s=0.020),
        }
        scan_path = _write_scan(tmp_path / "scan.json", expected=20, frames=frames)
        metrics = tune.analyze_scan_file(scan_path)
        assert metrics.dt_span_ms == pytest.approx(0.0, abs=1e-9)
        assert metrics.steady_state_dt_std_ms == pytest.approx(0.0, abs=1e-9)
        assert metrics.min_dt_ms == pytest.approx(10.0)
        assert metrics.max_dt_ms == pytest.approx(20.0)

    def test_scan_failure_json_never_passes(self, tmp_path: Path) -> None:
        scan_path = _write_scan(
            tmp_path / "scan_failure.json",
            expected=20,
            frames={"cam_a": _make_frames(15)},
            top_error="camera timeout",
        )
        metrics = tune.analyze_scan_file(scan_path)
        assert metrics.passed is False
        assert metrics.dropped_frames == 5
        assert metrics.errors == ("camera timeout",)

    def test_score_prefers_steady_state_over_single_sample_span(self, tmp_path: Path) -> None:
        # Candidate A: one edge outlier (large dt_span_ms) but rock-solid steady-state.
        a_path = _write_scan(
            tmp_path / "a" / "scan.json",
            expected=41,
            frames={"cam_a": _make_frames(41, dt_s=0.010, dt_overrides={0: 0.020})},
        )
        # Candidate B: smaller dt_span_ms, but steady-state oscillates +-0.5ms.
        osc = {i: (0.0105 if i % 2 == 0 else 0.0095) for i in range(40)}
        b_path = _write_scan(
            tmp_path / "b" / "scan.json",
            expected=41,
            frames={"cam_a": _make_frames(41, dt_s=0.010, dt_overrides=osc)},
        )
        a = tune.analyze_scan_file(a_path)
        b = tune.analyze_scan_file(b_path)
        assert a.dt_span_ms > b.dt_span_ms
        assert a.score() < b.score()


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._buf = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class TestVoxelClient:
    def test_session_discovery_basic_auth_and_result_unwrap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[tuple[str, str, dict[str, str], Any]] = []

        def fake_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> _FakeResponse:
            _ = timeout
            body = json.loads(req.data.decode("utf-8")) if isinstance(req.data, bytes) else None
            seen.append((req.get_method(), req.full_url, dict(req.header_items()), body))
            if req.full_url.endswith("/snapshot"):
                return _FakeResponse({"status": "active", "session": {"info": {"id": "sess-123"}}})
            if req.full_url.endswith("/commands/get_setting"):
                return _FakeResponse({"ok": True, "value": 1.25})
            if req.full_url.endswith("/commands/set_simple_servo_tuning"):
                return _FakeResponse({"ok": False, "msg": "axis is busy"})
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        client = tune.VoxelClient("http://localhost:8000", "st-1", "x_axis", password="secret")  # noqa: S106
        assert client.get_setting("driver.current.servo") == pytest.approx(1.25)

        expected_auth = "Basic " + base64.b64encode(b"voxel:secret").decode("ascii")
        assert seen[0][1] == "http://localhost:8000/api/stations/st-1/snapshot"
        assert seen[0][2]["Authorization"] == expected_auth
        assert (
            seen[1][1]
            == "http://localhost:8000/api/stations/st-1/sessions/sess-123/instrument/devices/x_axis/commands/get_setting"
        )
        assert seen[1][3] == {"args": ["driver.current.servo"], "kwargs": {}}

        with pytest.raises(RuntimeError, match="axis is busy"):
            client.set_simple_servo_tuning(1.0, 0.7)

    def test_session_discovery_raises_when_idle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            urllib.request,
            "urlopen",
            lambda *_args, **_kwargs: _FakeResponse({"status": "idle", "session": None}),
        )
        client = tune.VoxelClient("http://localhost:8000", "st-1", "x_axis")
        with pytest.raises(RuntimeError, match="No active session"):
            client.resolve_session_id()

    def test_run_test_scan_interrupted_and_multi_volume(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        storage_path = "tune/run1"
        _write_scan(
            tmp_path / storage_path / "volumes" / "vol_0" / "scan.json",
            expected=20,
            frames={"cam_a": _make_frames(20, dt_s=0.010)},
        )
        _write_scan(
            tmp_path / storage_path / "volumes" / "vol_1" / "scan.json",
            expected=20,
            frames={"cam_a": _make_frames(20, dt_s=0.010, dt_overrides={5: 0.012})},
        )

        polls = iter(["running", "interrupted"])

        def fake_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> _FakeResponse:
            _ = timeout
            if req.full_url.endswith("/instrument/acquisition"):
                return _FakeResponse({"id": "acq-1", "status": "preparing"})
            if req.full_url.endswith("/api/stations/st-1/acquisitions/acq-1"):
                return _FakeResponse({"id": "acq-1", "status": next(polls)})
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        client = tune.VoxelClient("http://localhost:8000", "st-1", "x_axis", session_id="sess-9")
        metrics = client.run_test_scan(task_id="t1", storage_path=storage_path, data_root=tmp_path, poll_s=0.0)
        assert metrics.expected_frames == 40
        assert metrics.recorded_frames == 40
        assert metrics.dt_span_ms == pytest.approx(2.0)

    def test_run_test_scan_failed_without_scan_files_surfaces_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def fake_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> _FakeResponse:
            _ = timeout
            if req.full_url.endswith("/instrument/acquisition"):
                return _FakeResponse({"id": "acq-2", "status": "preparing"})
            if req.full_url.endswith("/api/stations/st-1/acquisitions/acq-2"):
                return _FakeResponse(
                    {"id": "acq-2", "status": "failed", "failure": {"kind": "HardwareError", "message": "limit hit"}}
                )
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        client = tune.VoxelClient("http://localhost:8000", "st-1", "x_axis", session_id="sess-9")
        with pytest.raises(RuntimeError, match="limit hit"):
            client.run_test_scan(task_id="t1", storage_path="tune/missing", data_root=tmp_path, poll_s=0.0)


class _FakeTuningClient:
    """In-memory stand-in for VoxelClient used to test simple_tune workflows."""

    def __init__(
        self,
        *,
        stiffness_std: dict[float | None, list[float]],
        fault_on: set[float] | None = None,
        interrupt_on: float | None = None,
        fail_restore: bool = False,
    ) -> None:
        self.stiffness_std = stiffness_std
        self.fault_on = fault_on or set()
        self.interrupt_on = interrupt_on
        self.fail_restore = fail_restore
        self.current_stiffness: float | None = None
        self.calls: list[tuple[str, Any]] = []
        self._repeat_idx: dict[float | None, int] = {}

    def snapshot_servo_tuning(self, paramset: str = "live") -> dict[str, Any]:
        self.calls.append(("snapshot", paramset))
        return {
            "paramset": "live",
            "startup_paramset": "default",
            "type": "ffpid",
            "version": 4,
            "params": {},
            "pid": {"p": 100.0, "i": 10.0, "d": 1.0, "fc": 50.0},
            "simple": {"is_used": True, "load_mass_kg": 1.2, "carriage_mass_kg": 0.5, "params": {"Stiffness": 0.5}},
            "telemetry": {"driver.current.servo": 0.4},
        }

    def set_simple_servo_tuning(
        self,
        load_mass_kg: float,
        stiffness: float,
        *,
        enable_feed_forward: bool = True,
        paramset: str = "live",
    ) -> dict[str, Any]:
        self.calls.append(("set_simple", (load_mass_kg, stiffness, enable_feed_forward, paramset)))
        self.current_stiffness = stiffness
        return {"paramset": paramset}

    def restore_servo_tuning(self, snapshot: dict[str, Any], *, paramset: str = "live") -> dict[str, Any]:
        self.calls.append(("restore", paramset))
        if self.fail_restore:
            raise RuntimeError("restore boom")
        self.current_stiffness = None
        return snapshot

    def load_servo_paramset(self, to_paramset: str, from_paramset: str) -> dict[str, Any]:
        self.calls.append(("load_paramset", (to_paramset, from_paramset)))
        return {"paramset": to_paramset}

    def set_servo_startup_paramset(self, paramset: str) -> str:
        self.calls.append(("set_startup", paramset))
        return paramset

    def read_telemetry(self) -> dict[str, float]:
        return {"driver.current.servo": 0.45, "cloop.servo.effort": 12.0}

    def run_test_scan(self, *, task_id: str, storage_path: str, data_root: Path) -> Any:
        _ = (task_id, data_root)
        stiff = self.current_stiffness
        if stiff is not None and stiff == self.interrupt_on:
            raise KeyboardInterrupt
        if stiff is not None and stiff in self.fault_on:
            raise RuntimeError(f"fault at stiffness={stiff}")
        idx = self._repeat_idx.get(stiff, 0)
        self._repeat_idx[stiff] = idx + 1
        vals = self.stiffness_std[stiff]
        std = vals[min(idx, len(vals) - 1)]
        return tune.RippleMetrics(
            file=storage_path,
            expected_frames=40,
            recorded_frames=40,
            dropped_frames=0,
            extra_frames=0,
            min_dt_ms=10.0 - std,
            p05_dt_ms=10.0 - std,
            median_dt_ms=10.0,
            p95_dt_ms=10.0 + std,
            max_dt_ms=10.0 + std,
            dt_span_ms=2.0 * std,
            dt_std_ms=std,
            steady_state_dt_std_ms=std,
            max_spacing_error=0.05,
            worst_reversal_distance=0.0,
            errors=(),
            warnings=(),
            passed=True,
        )


class TestSimpleTune:
    def test_repeats_best_selection_apply_and_persist(self, tmp_path: Path) -> None:
        # 0.7 has one lucky pass (0.05) and one bad pass (0.90) -> worst-case 0.90;
        # 0.8 is consistently 0.10 -> 0.8 wins!
        fake = _FakeTuningClient(
            stiffness_std={
                None: [0.40, 0.42],
                0.7: [0.05, 0.90],
                0.8: [0.10, 0.11],
                0.9: [0.25, 0.24],
            }
        )
        backup = tmp_path / "backup.json"
        summary = tune.simple_tune(
            fake,  # type: ignore[arg-type]
            task_id="task-1",
            storage_prefix="zaber-tune",
            data_root=tmp_path,
            load_mass_kg=None,  # inferred from snapshot (1.2 kg)
            stiffness_values=(0.7, 0.8, 0.9),
            repeats=2,
            apply_best=True,
            persist_paramset="3",
            backup_path=backup,
        )
        assert backup.exists()
        assert summary["load_mass_kg"] == pytest.approx(1.2)
        assert summary["best_stiffness"] == pytest.approx(0.8)
        assert summary["persisted_paramset"] == "3"
        assert ("set_simple", (1.2, 0.8, True, "live")) in fake.calls
        assert ("load_paramset", ("3", "live")) in fake.calls
        assert ("set_startup", "3") in fake.calls

    def test_fault_recovers_and_non_apply_restores_original(self, tmp_path: Path) -> None:
        fake = _FakeTuningClient(
            stiffness_std={None: [0.40], 0.6: [0.20], 0.9: [0.10]},
            fault_on={0.9},
        )
        summary = tune.simple_tune(
            fake,  # type: ignore[arg-type]
            task_id="task-1",
            storage_prefix="zaber-tune",
            data_root=tmp_path,
            load_mass_kg=1.5,
            stiffness_values=(0.6, 0.9),
            repeats=1,
            apply_best=False,
            backup_path=tmp_path / "backup.json",
        )
        assert summary["best_stiffness"] == pytest.approx(0.6)
        assert fake.calls[-1] == ("restore", "live")

    def test_keyboard_interrupt_rolls_back_to_default_when_restore_fails(self, tmp_path: Path) -> None:
        fake = _FakeTuningClient(
            stiffness_std={None: [0.40], 0.7: [0.20]},
            interrupt_on=0.7,
            fail_restore=True,
        )
        with pytest.raises(KeyboardInterrupt):
            tune.simple_tune(
                fake,  # type: ignore[arg-type]
                task_id="task-1",
                storage_prefix="zaber-tune",
                data_root=tmp_path,
                load_mass_kg=1.5,
                stiffness_values=(0.7,),
                repeats=1,
                backup_path=tmp_path / "backup.json",
            )
        assert ("load_paramset", ("live", "default")) in fake.calls

    def test_persist_without_apply_best_rejected(self, tmp_path: Path) -> None:
        fake = _FakeTuningClient(stiffness_std={None: [0.40]})
        with pytest.raises(ValueError, match="requires --apply-best"):
            tune.simple_tune(
                fake,  # type: ignore[arg-type]
                task_id="task-1",
                storage_prefix="zaber-tune",
                data_root=tmp_path,
                load_mass_kg=1.5,
                apply_best=False,
                persist_paramset="1",
                backup_path=tmp_path / "backup.json",
            )


class TestAnalyzeOscilloscopeTrace:
    def test_ideal_constant_velocity_trace(self) -> None:
        n = 100
        times = [i * 0.001 for i in range(n)]  # 1 kHz
        pos = [5.0 * t for t in times]  # 5 mm/s
        metrics = tune.analyze_oscilloscope_trace(times, pos, pos, target_speed_mm_s=5.0)
        assert metrics.sample_count == n
        assert metrics.sample_rate_hz == pytest.approx(1000.0)
        assert metrics.rms_following_error_um == pytest.approx(0.0, abs=1e-9)
        assert metrics.max_following_error_um == pytest.approx(0.0, abs=1e-9)
        assert metrics.mean_velocity_mm_s == pytest.approx(5.0)
        assert metrics.steady_state_velocity_ripple_pct == pytest.approx(0.0, abs=1e-9)
        assert metrics.worst_reversal_um == pytest.approx(0.0)

    def test_following_error_ripple_and_ramp_trimming(self) -> None:
        n = 100
        times = [i * 0.001 for i in range(n)]
        pos = [5.0 * t for t in times]
        # Constant 2.0 um lag + sinusoidal ripple in steady state, plus a big ramp artifact at sample 0..5.
        enc = [
            p - 0.002 + 0.0005 * math.sin(2.0 * math.pi * i / 10.0) - (0.05 if i < 5 else 0.0)
            for i, p in enumerate(pos)
        ]
        metrics = tune.analyze_oscilloscope_trace(times, pos, enc, target_speed_mm_s=5.0, trim_fraction=0.1)
        assert metrics.rms_following_error_um == pytest.approx(2.0, abs=0.2)
        assert metrics.steady_state_velocity_std_mm_s < metrics.velocity_std_mm_s
        assert metrics.steady_state_velocity_ripple_pct > 0.0

    def test_reversal_detected_and_penalized(self) -> None:
        times = [i * 0.001 for i in range(50)]
        pos = [5.0 * t for t in times]
        enc = list(pos)
        enc[25] -= 0.015  # 15 um backward dip
        bad = tune.analyze_oscilloscope_trace(times, pos, enc, target_speed_mm_s=5.0)
        good = tune.analyze_oscilloscope_trace(times, pos, pos, target_speed_mm_s=5.0)
        assert bad.worst_reversal_um == pytest.approx(10.0)
        assert good.score() < bad.score()

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            tune.analyze_oscilloscope_trace([0.0, 0.001], [0.0, 1.0], [0.0, 1.0], target_speed_mm_s=5.0)
        with pytest.raises(ValueError, match="strictly increasing"):
            tune.analyze_oscilloscope_trace(
                [0.0, 0.001, 0.001, 0.002], [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0], target_speed_mm_s=5.0
            )

    def test_oscilloscope_tune_validates_inputs_before_connecting(self) -> None:
        base_kwargs: dict[str, Any] = {
            "host": "127.0.0.1",
            "port": 55551,
            "axis_number": 1,
            "scan_start_mm": 0.0,
            "scan_end_mm": 10.0,
            "scan_speed_mm_s": 5.0,
            "load_mass_kg": 1.0,
        }
        with pytest.raises(ValueError, match="stiffness"):
            tune.run_oscilloscope_tune(**base_kwargs, stiffness_values=(1.5,))
        with pytest.raises(ValueError, match="load_mass_kg"):
            tune.run_oscilloscope_tune(**{**base_kwargs, "load_mass_kg": -0.1})
        with pytest.raises(ValueError, match="scan_speed_mm_s"):
            tune.run_oscilloscope_tune(**{**base_kwargs, "scan_speed_mm_s": 0.0})


def test_cli_prefers_voxel_web_password(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_auth: list[str | None] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> _FakeResponse:
        _ = timeout
        captured_auth.append(req.get_header("Authorization"))
        if req.full_url.endswith("/commands/get_servo_tuning"):
            return _FakeResponse({"ok": True, "value": {"paramset": "live"}})
        return _FakeResponse({"ok": False, "msg": "unsupported"})

    monkeypatch.setenv("VOXEL_WEB_PASSWORD", "web-secret")
    monkeypatch.setenv("VOXEL_PASSWORD", "fallback-secret")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    snap_path = tmp_path / "snap.json"
    rc = tune.main(["--station-id", "st-1", "--session-id", "sess-1", "--snapshot", str(snap_path)])
    assert rc == 0
    assert captured_auth[0] == "Basic " + base64.b64encode(b"voxel:web-secret").decode("ascii")
