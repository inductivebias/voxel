#!/usr/bin/env python3
"""Zaber ServoTuner & scan velocity-ripple analysis/tuning script.

Supports three workflows:
  1. Analyze scan.json / scan_failure.json locally (--analyze)
  2. Direct 1 kHz Zaber oscilloscope + ServoTuner stiffness sweep (--zaber-host)
  3. Voxel REST ServoTuner snapshot/restore and scan validation sweep (--station-id)
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

TELEMETRY_SETTINGS: tuple[str, ...] = ("driver.current.servo", "cloop.servo.effort")
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled", "interrupted"})
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
    peak = positions[0] * sign
    worst = 0.0
    for pos in positions[1:]:
        p = pos * sign
        if p > peak:
            peak = p
        elif peak - p > worst:
            worst = peak - p
    return worst


def _expected_spacing(data: dict[str, Any], expected: int) -> float | None:
    plan = data.get("plan") if isinstance(data.get("plan"), dict) else data
    if isinstance(plan, dict) and isinstance(plan.get("z_step"), (int, float)) and float(plan["z_step"]) > 0:
        return float(plan["z_step"])
    window = plan.get("window") if isinstance(plan, dict) else None
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
    recorded_frames = min(counts)
    dropped = max(0, expected - min(counts))
    extra = max(0, max(counts) - expected)

    worst_reversal = 0.0
    wr = readback.get("worst_reversal")
    if isinstance(wr, dict) and isinstance(wr.get("distance"), (int, float)):
        worst_reversal = max(0.0, float(wr["distance"]))

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
    passed = path.name == "scan.json" and not errors and dropped == 0 and extra == 0
    return RippleMetrics(
        file=str(path),
        expected_frames=expected,
        recorded_frames=recorded_frames,
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
        passed=passed,
    )


def aggregate_metrics(items: Sequence[RippleMetrics], *, label: str | None = None) -> RippleMetrics:
    if not items:
        raise ValueError("Cannot aggregate empty metrics sequence")
    if len(items) == 1:
        return items[0]
    spacing_errs = [m.max_spacing_error for m in items if m.max_spacing_error is not None]
    return RippleMetrics(
        file=label or items[-1].file,
        expected_frames=sum(m.expected_frames for m in items),
        recorded_frames=sum(m.recorded_frames for m in items),
        dropped_frames=sum(m.dropped_frames for m in items),
        extra_frames=sum(m.extra_frames for m in items),
        min_dt_ms=min(m.min_dt_ms for m in items),
        p05_dt_ms=min(m.p05_dt_ms for m in items),
        median_dt_ms=statistics.median(m.median_dt_ms for m in items),
        p95_dt_ms=max(m.p95_dt_ms for m in items),
        max_dt_ms=max(m.max_dt_ms for m in items),
        dt_span_ms=max(m.dt_span_ms for m in items),
        dt_std_ms=max(m.dt_std_ms for m in items),
        steady_state_dt_std_ms=max(m.steady_state_dt_std_ms for m in items),
        max_spacing_error=max(spacing_errs) if spacing_errs else None,
        worst_reversal_distance=max(m.worst_reversal_distance for m in items),
        errors=tuple(dict.fromkeys(e for m in items for e in m.errors)),
        warnings=tuple(dict.fromkeys(w for m in items for w in m.warnings)),
        passed=all(m.passed for m in items),
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


class VoxelClient:
    def __init__(
        self,
        base_url: str,
        station_id: str,
        device_id: str,
        *,
        session_id: str | None = None,
        password: str | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must use http:// or https://, got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.station_id = station_id
        self.device_id = device_id
        self.session_id = session_id
        self._auth = "Basic " + base64.b64encode(f"voxel:{password}".encode()).decode("ascii") if password else None

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers: dict[str, str] = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self._auth is not None:
            headers["Authorization"] = self._auth
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc

    def resolve_session_id(self) -> str:
        if self.session_id:
            return self.session_id
        snap = self._request("GET", f"/api/stations/{self.station_id}/snapshot")
        sid = (((snap or {}).get("session") or {}).get("info") or {}).get("id")
        if not sid:
            raise RuntimeError(f"No active session on station {self.station_id}; pass --session-id")
        self.session_id = str(sid)
        return self.session_id

    def _instrument(self, suffix: str) -> str:
        return f"/api/stations/{self.station_id}/sessions/{self.resolve_session_id()}/instrument{suffix}"

    def device_command(self, command: str, *args: Any, **kwargs: Any) -> Any:
        raw = self._request(
            "POST",
            self._instrument(f"/devices/{self.device_id}/commands/{command}"),
            {"args": list(args), "kwargs": kwargs},
        )
        if not isinstance(raw, dict) or "ok" not in raw:
            raise RuntimeError(f"Unexpected Result envelope for {command}: {raw!r}")
        if not raw["ok"]:
            raise RuntimeError(f"Device command {command} failed: {raw.get('msg', 'unknown error')}")
        return raw.get("value")

    def get_setting(self, name: str) -> float:
        return float(self.device_command("get_setting", name))

    def get_servo_tuning(self, paramset: str = "live") -> dict[str, Any]:
        return dict(self.device_command("get_servo_tuning", paramset=paramset))

    def set_simple_servo_tuning(
        self,
        load_mass_kg: float,
        stiffness: float,
        *,
        enable_feed_forward: bool = True,
        paramset: str = "live",
    ) -> dict[str, Any]:
        return dict(
            self.device_command(
                "set_simple_servo_tuning",
                load_mass_kg=float(load_mass_kg),
                stiffness=float(stiffness),
                enable_feed_forward=enable_feed_forward,
                paramset=paramset,
            )
        )

    def set_pid_servo_tuning(
        self, p: float, i: float, d: float, fc: float, *, paramset: str = "live"
    ) -> dict[str, Any]:
        return dict(
            self.device_command(
                "set_pid_servo_tuning", p=float(p), i=float(i), d=float(d), fc=float(fc), paramset=paramset
            )
        )

    def load_servo_paramset(self, to_paramset: str, from_paramset: str) -> dict[str, Any]:
        return dict(self.device_command("load_servo_paramset", to_paramset=to_paramset, from_paramset=from_paramset))

    def set_servo_startup_paramset(self, paramset: str) -> str:
        return str(self.device_command("set_servo_startup_paramset", paramset=paramset))

    def read_telemetry(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in TELEMETRY_SETTINGS:
            with contextlib.suppress(RuntimeError):
                out[name] = self.get_setting(name)
        return out

    def restore_servo_tuning(self, snapshot: dict[str, Any], *, paramset: str = "live") -> dict[str, Any]:
        simple = snapshot.get("simple")
        if isinstance(simple, dict) and simple.get("is_used"):
            params = simple.get("params") or {}
            stiffness = next((float(v) for k, v in params.items() if k.lower() == "stiffness"), None)
            if stiffness is not None and simple.get("load_mass_kg") is not None:
                return self.set_simple_servo_tuning(float(simple["load_mass_kg"]), stiffness, paramset=paramset)
        pid = snapshot.get("pid")
        if isinstance(pid, dict) and all(pid.get(k) is not None for k in ("p", "i", "d", "fc")):
            return self.set_pid_servo_tuning(
                float(pid["p"]), float(pid["i"]), float(pid["d"]), float(pid["fc"]), paramset=paramset
            )
        return self.load_servo_paramset(paramset, "default")

    def run_test_scan(
        self,
        *,
        task_id: str,
        storage_path: str,
        data_root: Path,
        poll_s: float = 0.25,
        timeout_s: float = 120.0,
    ) -> RippleMetrics:
        manifest = self._request(
            "POST",
            self._instrument("/acquisition"),
            {"storage": {"path": storage_path}, "task_ids": [task_id]},
        )
        acq_id = manifest["id"]
        deadline = time.monotonic() + timeout_s
        status_obj = manifest
        while time.monotonic() < deadline:
            status_obj = self._request("GET", f"/api/stations/{self.station_id}/acquisitions/{acq_id}")
            if status_obj.get("status") in TERMINAL_STATUSES:
                break
            time.sleep(poll_s)
        else:
            self._request("POST", self._instrument("/acquisition/stop"), {})
            raise TimeoutError(f"Acquisition {acq_id} did not finish within {timeout_s}s")

        volumes_dir = data_root / storage_path / "volumes"
        candidates = (
            sorted(list(volumes_dir.rglob("scan.json")) + list(volumes_dir.rglob("scan_failure.json")))
            if volumes_dir.exists()
            else []
        )
        if not candidates:
            failure = status_obj.get("failure")
            raise RuntimeError(
                f"Acquisition {acq_id} ended with status={status_obj.get('status')!r}, failure={failure!r}"
            )
        return aggregate_metrics([analyze_scan_file(p) for p in candidates], label=str(data_root / storage_path))


def _validate_sweep(
    *,
    load_mass_kg: float,
    stiffness_values: Sequence[float],
    repeats: int,
    apply_best: bool,
    persist_paramset: str | None,
    scan_speed_mm_s: float | None = None,
) -> None:
    if not math.isfinite(load_mass_kg) or load_mass_kg < 0:
        raise ValueError(f"load_mass_kg must be finite and >= 0, got {load_mass_kg}")
    if not stiffness_values or any(not math.isfinite(s) or not 0.0 <= s <= 1.0 for s in stiffness_values):
        raise ValueError(f"stiffness_values must be non-empty values in [0.0, 1.0], got {stiffness_values}")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if persist_paramset is not None:
        if not apply_best:
            raise ValueError("--persist-paramset requires --apply-best")
        if persist_paramset.strip() not in PERSIST_SLOTS:
            raise ValueError(f"persist_paramset must be in '1'..'9', got {persist_paramset!r}")
    if scan_speed_mm_s is not None and (not math.isfinite(scan_speed_mm_s) or scan_speed_mm_s <= 0):
        raise ValueError(f"scan_speed_mm_s must be finite and > 0, got {scan_speed_mm_s}")


def _rollback(client: VoxelClient, original: dict[str, Any]) -> None:
    try:
        client.restore_servo_tuning(original, paramset="live")
    except Exception:
        client.load_servo_paramset("live", "default")


def simple_tune(
    client: VoxelClient,
    *,
    task_id: str,
    storage_prefix: str,
    data_root: Path,
    load_mass_kg: float,
    stiffness_values: Sequence[float],
    enable_feed_forward: bool,
    repeats: int,
    apply_best: bool,
    persist_paramset: str | None,
    backup_path: Path,
) -> tuple[float | None, RippleMetrics]:
    _validate_sweep(
        load_mass_kg=load_mass_kg,
        stiffness_values=stiffness_values,
        repeats=repeats,
        apply_best=apply_best,
        persist_paramset=persist_paramset,
    )
    original = client.get_servo_tuning("live")
    backup_path.write_text(json.dumps(original, indent=2) + "\n")
    run_idx = 0

    def evaluate(label: str) -> RippleMetrics:
        nonlocal run_idx
        passes: list[RippleMetrics] = []
        for rep in range(1, repeats + 1):
            run_idx += 1
            passes.append(
                client.run_test_scan(
                    task_id=task_id,
                    storage_path=f"{storage_prefix}/tune_{run_idx:03d}_{label}_r{rep}",
                    data_root=data_root,
                )
            )
        agg = aggregate_metrics(passes, label=label)
        telem = client.read_telemetry()
        print(
            f"  [{label}] passed={agg.passed} frames={agg.recorded_frames}/{agg.expected_frames} "
            f"steady_std={agg.steady_state_dt_std_ms:.4f}ms p05..p95=[{agg.p05_dt_ms:.3f}..{agg.p95_dt_ms:.3f}] "
            f"rev={agg.worst_reversal_distance:.4g} telem={telem}"
        )
        return agg

    try:
        best_stiffness: float | None = None
        best_metrics = evaluate("baseline")
        for stiffness in stiffness_values:
            client.set_simple_servo_tuning(
                load_mass_kg, stiffness, enable_feed_forward=enable_feed_forward, paramset="live"
            )
            try:
                metrics = evaluate(f"stiff_{stiffness:.2f}")
            except Exception as exc:
                print(f"  [FAULT] stiffness={stiffness:.2f} failed ({exc}); rolling back", file=sys.stderr)
                _rollback(client, original)
                continue
            if metrics.score() < best_metrics.score():
                best_stiffness = stiffness
                best_metrics = metrics
    except BaseException:
        _rollback(client, original)
        raise

    if apply_best and best_stiffness is not None:
        client.set_simple_servo_tuning(
            load_mass_kg, best_stiffness, enable_feed_forward=enable_feed_forward, paramset="live"
        )
        if persist_paramset is not None:
            slot = persist_paramset.strip()
            client.load_servo_paramset(slot, "live")
            client.set_servo_startup_paramset(slot)
    else:
        _rollback(client, original)
    return best_stiffness, best_metrics


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
    port = int(port_s) if port_s else 55551

    with Connection.open_tcp(hostname, port) as conn:
        device = conn.detect_devices()[0]
        axis = device.get_axis(axis_number)
        tuner = ServoTuner(axis)
        defs = tuner.get_simple_tuning_param_definitions()
        stiff_name = next((d.name for d in defs if d.name.lower() == "stiffness"), "Stiffness")
        orig_speed = axis.settings.get("maxspeed", Units.VELOCITY_MILLIMETRES_PER_SECOND)

        def apply_stiffness(val: float) -> None:
            tuner.set_simple_tuning(
                ServoTuningParamset.STAGING,
                [ServoTuningParam(name=stiff_name, value=float(val))],
                load_inertia=float(load_mass_kg),
                load_inertia_units=Units.INERTIA_KILOGRAMS,
                enable_feed_forward=enable_feed_forward,
            )
            tuner.load_paramset(ServoTuningParamset.LIVE, ServoTuningParamset.STAGING)

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
                cmd = ch[0].get_data(Units.LENGTH_MICROMETRES)
                enc = ch[1].get_data(Units.LENGTH_MICROMETRES)
                results.append(analyze_oscilloscope_trace(cmd, enc, sample_rate_hz))
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
                    print(f"  [FAULT] stiffness={stiffness:.2f} failed ({exc}); restoring DEFAULT", file=sys.stderr)
                    tuner.load_paramset(ServoTuningParamset.LIVE, ServoTuningParamset.DEFAULT)
                    continue
                if metrics.score() < best_metrics.score():
                    best_stiffness = stiffness
                    best_metrics = metrics
        except BaseException:
            tuner.load_paramset(ServoTuningParamset.LIVE, ServoTuningParamset.DEFAULT)
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
            tuner.load_paramset(ServoTuningParamset.LIVE, ServoTuningParamset.DEFAULT)
        return best_stiffness, best_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyze", type=Path, help="Analyze scan.json / scan_failure.json and exit")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--station-id", help="Station UUID")
    parser.add_argument("--session-id", help="Session UUID (auto-discovered if omitted)")
    parser.add_argument("--device-id", default="x_axis")
    parser.add_argument("--password", help="Control password (or VOXEL_WEB_PASSWORD / VOXEL_PASSWORD)")
    parser.add_argument("--snapshot", type=Path, help="Save live ServoTuner snapshot to JSON and exit")
    parser.add_argument("--restore", type=Path, help="Restore ServoTuner snapshot from JSON and exit")
    parser.add_argument("--task-id", help="Acquisition task ID for Voxel scan tuning")
    parser.add_argument("--data-root", type=Path, help="Filesystem root for storage.path")
    parser.add_argument("--storage-prefix", default="zaber-tune")
    parser.add_argument("--load-mass-kg", type=float, help="Physical payload mass on the stage in kg")
    parser.add_argument("--stiffness-values", default="0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--no-feed-forward", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--backup", type=Path, default=Path("zaber_tuning_backup.json"))
    parser.add_argument("--apply-best", action="store_true")
    parser.add_argument("--persist-paramset", help="Persist winning tuning to slot 1..9 (requires --apply-best)")
    parser.add_argument("--zaber-host", help="Direct Zaber TCP host[:port] for 1 kHz oscilloscope sweep")
    parser.add_argument("--axis", type=int, default=1)
    parser.add_argument("--scan-start-mm", type=float)
    parser.add_argument("--scan-end-mm", type=float)
    parser.add_argument("--scan-speed-mm-s", type=float)
    args = parser.parse_args(argv)

    if args.analyze is not None:
        print(json.dumps(asdict(analyze_scan_file(args.analyze)), indent=2))
        return 0

    stiffness_vals = tuple(float(x.strip()) for x in args.stiffness_values.split(",") if x.strip())

    if args.zaber_host:
        if None in (args.load_mass_kg, args.scan_start_mm, args.scan_end_mm, args.scan_speed_mm_s):
            parser.error("--zaber-host requires --load-mass-kg, --scan-start-mm, --scan-end-mm, --scan-speed-mm-s")
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
        )
        return 0

    if not args.station_id:
        parser.error("--station-id is required unless --analyze or --zaber-host is used")

    password = args.password or os.environ.get("VOXEL_WEB_PASSWORD") or os.environ.get("VOXEL_PASSWORD")
    client = VoxelClient(args.base_url, args.station_id, args.device_id, session_id=args.session_id, password=password)

    if args.snapshot is not None:
        snap = client.get_servo_tuning("live")
        args.snapshot.write_text(json.dumps(snap, indent=2) + "\n")
        print(json.dumps(snap, indent=2))
        return 0

    if args.restore is not None:
        print(json.dumps(client.restore_servo_tuning(json.loads(args.restore.read_text())), indent=2))
        return 0

    if not args.task_id or args.data_root is None or args.load_mass_kg is None:
        parser.error("--task-id, --data-root, and --load-mass-kg are required for Voxel tuning")

    simple_tune(
        client,
        task_id=args.task_id,
        storage_prefix=args.storage_prefix,
        data_root=args.data_root,
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
