"""Unit tests for scripts/tune_zaber_scan.py."""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path
from typing import Any, Self

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
        t = 0.0
        pts = [{"frame": 0, "sample": 0, "time_s": 0.0, "position": 0.0}]
        for i in range(1, expected):
            delta = step_s + (edge_spike_s if i in (1, expected - 1) else 0.0)
            t += delta
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
    assert clean.dropped_frames == 0
    assert clean.extra_frames == 0
    assert clean.steady_state_dt_std_ms < clean.dt_std_ms
    assert clean.max_spacing_error == pytest.approx(0.0)

    # Fatal violation (not in warnings) + extra frames via marker_counts must fail even on scan.json.
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

    # Multi-marker: worst-case marker is preserved rather than blended.
    multi = tzs.analyze_scan_file(_make_scan(tmp_path / "scan.json", dt_s=0.01, extra_marker_dt_s=0.015))
    assert multi.min_dt_ms == pytest.approx(10.0)
    assert multi.max_dt_ms == pytest.approx(15.0)

    # scan_failure.json never passes.
    fail = tzs.analyze_scan_file(_make_scan(tmp_path / "scan_failure.json"))
    assert not fail.passed


def test_voxel_client_session_discovery_result_unwrap_auth_and_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None, str | None]] = []

    class FakeResp:
        def __init__(self, payload: Any) -> None:
            self._raw = json.dumps(payload).encode()

        def read(self) -> bytes:
            return self._raw

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

    def fake_urlopen(req: Any, timeout: float = 30.0) -> FakeResp:
        del timeout
        body = json.loads(req.data.decode()) if req.data else None
        calls.append((req.get_method(), req.full_url, body, req.get_header("Authorization")))
        url = req.full_url
        if url.endswith("/api/stations/st-1/snapshot"):
            return FakeResp({"session": {"info": {"id": "sess-9"}}})
        if "/commands/get_servo_tuning" in url:
            return FakeResp({"ok": True, "value": {"paramset": "live", "simple": {"is_used": True}}})
        if "/commands/get_setting" in url:
            return FakeResp({"ok": False, "msg": "unknown setting"})
        if url.endswith("/instrument/acquisition"):
            return FakeResp({"id": "acq-1", "status": "preparing"})
        if url.endswith("/acquisitions/acq-1"):
            return FakeResp({"id": "acq-1", "status": "interrupted", "failure": {"message": "stage fault"}})
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(tzs.urllib.request, "urlopen", fake_urlopen)
    client = tzs.VoxelClient("http://127.0.0.1:8000", "st-1", "x_axis", password="secret")  # noqa: S106

    snap = client.get_servo_tuning("live")
    assert snap["paramset"] == "live"
    assert calls[0][1].endswith("/api/stations/st-1/snapshot")
    assert calls[1][1].endswith(
        "/api/stations/st-1/sessions/sess-9/instrument/devices/x_axis/commands/get_servo_tuning"
    )
    expected_auth = "Basic " + base64.b64encode(b"voxel:secret").decode("ascii")
    assert calls[1][3] == expected_auth

    with pytest.raises(RuntimeError, match="unknown setting"):
        client.get_setting("bad.setting")

    with pytest.raises(RuntimeError, match=r"interrupted.*stage fault"):
        client.run_test_scan(task_id="t1", storage_path="run1", data_root=tmp_path, poll_s=0.0)


def test_simple_tune_selects_best_persists_and_rolls_back_on_fault(tmp_path: Path) -> None:
    class FakeClient(tzs.VoxelClient):
        def __init__(self) -> None:
            super().__init__("http://127.0.0.1:8000", "st-1", "x_axis", session_id="s1")
            self.stiffness = 0.5
            self.actions: list[tuple[Any, ...]] = []
            self.fail_on: float | None = None

        def get_servo_tuning(self, paramset: str = "live") -> dict[str, Any]:
            return {
                "paramset": paramset,
                "simple": {"is_used": True, "load_mass_kg": 1.0, "params": {"Stiffness": 0.5}},
            }

        def set_simple_servo_tuning(
            self, load_mass_kg: float, stiffness: float, *, enable_feed_forward: bool = True, paramset: str = "live"
        ) -> dict[str, Any]:
            self.stiffness = stiffness
            self.actions.append(("simple", load_mass_kg, stiffness, enable_feed_forward, paramset))
            return {"paramset": paramset}

        def load_servo_paramset(self, to_paramset: str, from_paramset: str) -> dict[str, Any]:
            self.actions.append(("load", to_paramset, from_paramset))
            return {"paramset": to_paramset}

        def set_servo_startup_paramset(self, paramset: str) -> str:
            self.actions.append(("startup", paramset))
            return paramset

        def read_telemetry(self) -> dict[str, float]:
            return {"driver.current.servo": 0.4}

        def run_test_scan(self, *, task_id: str, storage_path: str, data_root: Path, **_: Any) -> tzs.RippleMetrics:
            del task_id, storage_path, data_root
            if self.fail_on is not None and math.isclose(self.stiffness, self.fail_on):
                raise RuntimeError("FF fault")
            std = {0.5: 0.20, 0.7: 0.05, 0.9: 0.12}[self.stiffness]
            return tzs.RippleMetrics(
                "s.json", 10, 10, 0, 0, 9.9, 9.95, 10.0, 10.05, 10.1, 0.2, std, std, 0.0, 0.0, (), (), True
            )

    fc = FakeClient()
    fc.fail_on = 0.9
    best, _ = tzs.simple_tune(
        fc,
        task_id="t1",
        storage_prefix="tune",
        data_root=tmp_path,
        load_mass_kg=1.0,
        stiffness_values=(0.7, 0.9),
        enable_feed_forward=True,
        repeats=2,
        apply_best=True,
        persist_paramset="2",
        backup_path=tmp_path / "backup.json",
    )
    assert best == 0.7
    assert ("load", "2", "live") in fc.actions
    assert ("startup", "2") in fc.actions
    assert (tmp_path / "backup.json").exists()


def test_oscilloscope_trace_and_validation() -> None:
    cmd = [float(i) for i in range(50)]
    enc = [c - 0.1 for c in cmd]
    m = tzs.analyze_oscilloscope_trace(cmd, enc, sample_rate_hz=1000.0)
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
