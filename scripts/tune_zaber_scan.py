#!/usr/bin/env python3
"""Zaber ServoTuner + velocity-ripple tuning script for continuous scans.

Supports three workflows:
1. Local analysis of ``scan.json`` / ``scan_failure.json`` recordings (no server
   or ``zaber_motion`` required).
2. Voxel REST API Simple Tuning stiffness sweep using ``ZaberLinearAxis``
   ``ServoTuner`` commands (``get_servo_tuning``, ``set_simple_servo_tuning``,
   ``set_pid_servo_tuning``, ``load_servo_paramset``, ``set_servo_startup_paramset``)
   and read-only ``get_setting`` telemetry.
3. Direct Zaber Oscilloscope + ServoTuner mode (1 kHz ``pos`` vs ``encoder.pos``
   following error and velocity ripple directly on the controller, without
   cameras/DAQ/Zarr; ``zaber_motion`` is imported lazily).

Examples:
    # 1. Analyze an existing scan.json / scan_failure.json locally:
    python3 scripts/tune_zaber_scan.py --analyze /path/to/scan.json

    # 2. Snapshot / restore current Zaber live servo tuning via Voxel REST API:
    python3 scripts/tune_zaber_scan.py --station-id <UUID> --snapshot zaber_backup.json
    python3 scripts/tune_zaber_scan.py --station-id <UUID> --restore zaber_backup.json

    # 3. Run Simple Tuning stiffness sweep against a short Voxel acquisition task:
    python3 scripts/tune_zaber_scan.py \\
        --station-id <UUID> \\
        --task-id <TASK_ID> \\
        --data-root /data/users/cfxuser \\
        --load-mass-kg 1.5 \\
        --repeats 2 \\
        --apply-best \\
        --persist-paramset 1

    # 4. Run direct Zaber 1 kHz oscilloscope stiffness sweep (no Voxel server):
    python3 scripts/tune_zaber_scan.py \\
        --zaber-host 192.168.1.50 \\
        --axis 1 \\
        --scan-start-mm 5.0 \\
        --scan-end-mm 25.0 \\
        --scan-speed-mm-s 5.0 \\
        --load-mass-kg 1.5 \\
        --apply-best
"""

from __future__ import annotations

import argparse
import base64
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

DEFAULT_STIFFNESS_VALUES: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TELEMETRY_SETTINGS: tuple[str, ...] = (
    "driver.current.servo",
    "cloop.servo.effort",
    "system.voltage",
)
TERMINAL_ACQUISITION_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled", "interrupted"})
STORED_PARAMSET_SLOTS: tuple[str, ...] = tuple(str(i) for i in range(1, 10))
_REVERSAL_NOISE_FLOOR = 1e-6


@dataclass(frozen=True)
class RippleMetrics:
    """Velocity-ripple and integrity metrics extracted from a continuous-scan recording."""

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
    max_spacing_error: float
    worst_reversal_distance: float
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    passed: bool

    def score(self) -> tuple[bool, int, bool, float, float, float]:
        """Lower is better: passed first, zero frame-count error, no reversal, then steady-state ripple."""
        return (
            not self.passed,
            self.dropped_frames + self.extra_frames,
            self.worst_reversal_distance > 0.0,
            self.steady_state_dt_std_ms,
            self.p95_dt_ms - self.p05_dt_ms,
            self.dt_span_ms,
        )


@dataclass(frozen=True)
class _MarkerTiming:
    min_dt_ms: float
    p05_dt_ms: float
    median_dt_ms: float
    p95_dt_ms: float
    max_dt_ms: float
    dt_span_ms: float
    dt_std_ms: float
    steady_state_dt_std_ms: float
    max_spacing_error: float
    reversal_distance: float

    def rank_key(self) -> tuple[float, float, float]:
        return (self.steady_state_dt_std_ms, self.p95_dt_ms - self.p05_dt_ms, self.dt_span_ms)


@dataclass(frozen=True)
class OscilloscopeMetrics:
    """Following-error and velocity-ripple metrics from a direct Zaber oscilloscope trace."""

    sample_count: int
    sample_rate_hz: float
    target_speed_mm_s: float
    rms_following_error_um: float
    max_following_error_um: float
    mean_velocity_mm_s: float
    velocity_std_mm_s: float
    velocity_ripple_pct: float
    steady_state_velocity_std_mm_s: float
    steady_state_velocity_ripple_pct: float
    worst_reversal_um: float

    def score(self) -> tuple[bool, float, float, float]:
        """Lower is better: no reversal, then steady-state velocity ripple, then RMS following error."""
        return (
            self.worst_reversal_um > 0.0,
            self.steady_state_velocity_std_mm_s,
            self.rms_following_error_um,
            self.max_following_error_um,
        )


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile matching numpy's default method."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


def _trim_slice(values: Sequence[float], trim_fraction: float) -> Sequence[float]:
    """Trim ``trim_fraction`` from both ends when enough samples remain."""
    trim = int(len(values) * trim_fraction)
    if trim > 0 and len(values) - 2 * trim >= 2:
        return values[trim : len(values) - trim]
    return values


def _extract_expected_spacing(data: dict[str, Any], expected_frames: int) -> float | None:
    """Infer nominal frame-to-frame spatial spacing from scan metadata."""
    line_scan = data.get("line_scan")
    if isinstance(line_scan, dict) and line_scan.get("spacing") is not None:
        return float(line_scan["spacing"])

    plan = data.get("plan")
    if isinstance(plan, dict) and plan.get("z_step") is not None:
        return float(plan["z_step"])

    division = plan.get("division") if isinstance(plan, dict) else data.get("division")
    scan_info = data.get("scan_info")
    if isinstance(scan_info, dict) and isinstance(division, dict):
        edge_spacing = scan_info.get("edge_spacing")
        divider = division.get("divider")
        if edge_spacing is not None and divider is not None:
            return float(edge_spacing) * int(divider)

    window = plan.get("window") if isinstance(plan, dict) else data.get("window")
    if isinstance(window, (list, tuple)) and len(window) == 2 and expected_frames > 0:
        return abs(float(window[1]) - float(window[0])) / expected_frames

    return None


def _max_position_reversal(positions: Sequence[float]) -> float:
    """Largest backward excursion (assuming positive travel direction)."""
    if len(positions) < 2:
        return 0.0
    peak = positions[0]
    worst = 0.0
    for pos in positions[1:]:
        if pos > peak:
            peak = pos
        elif (drop := peak - pos) > worst:
            worst = drop
    return worst if worst > _REVERSAL_NOISE_FLOOR else 0.0


def _analyze_marker_frames(pts: list[dict[str, Any]], expected_spacing: float | None) -> _MarkerTiming:
    """Compute timing and spatial metrics for one camera marker's frame list."""
    times = [float(p["time_s"]) for p in pts]
    dt_ms = [(times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1)]
    steady_dt = _trim_slice(dt_ms, 0.05)

    positions = [float(p["position"]) for p in pts if "position" in p]
    max_spacing_err = 0.0
    reversal = 0.0
    if len(positions) >= 2:
        diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        spacing = expected_spacing if expected_spacing is not None else statistics.median(diffs)
        max_spacing_err = max(abs(d - spacing) for d in diffs)
        reversal = _max_position_reversal(positions)

    min_dt = min(dt_ms)
    max_dt = max(dt_ms)
    return _MarkerTiming(
        min_dt_ms=min_dt,
        p05_dt_ms=_percentile(dt_ms, 5.0),
        median_dt_ms=statistics.median(dt_ms),
        p95_dt_ms=_percentile(dt_ms, 95.0),
        max_dt_ms=max_dt,
        dt_span_ms=max_dt - min_dt,
        dt_std_ms=statistics.stdev(dt_ms) if len(dt_ms) > 1 else 0.0,
        steady_state_dt_std_ms=statistics.stdev(steady_dt) if len(steady_dt) > 1 else 0.0,
        max_spacing_error=max_spacing_err,
        reversal_distance=reversal,
    )


def analyze_scan_file(path: Path) -> RippleMetrics:
    """Analyze a Voxel ``scan.json`` or ``scan_failure.json`` file per marker (worst-case)."""
    with path.open() as fh:
        data: dict[str, Any] = json.load(fh)

    expected = int(data["plan"]["division"]["cycles"]) if "plan" in data else int(data["division"]["cycles"])
    expected_spacing = _extract_expected_spacing(data, expected)

    recorded: dict[str, Any] = data.get("recorded") or {}
    frames_map: dict[str, list[dict[str, Any]]] = recorded.get("frames") or {}
    if not frames_map:
        raise ValueError(f"No recorded frames found in {path}")

    readback: dict[str, Any] = recorded.get("readback") or {}
    raw_marker_counts = readback.get("marker_counts")
    marker_counts: dict[str, int] = {}
    if isinstance(raw_marker_counts, dict) and raw_marker_counts:
        for name, count in raw_marker_counts.items():
            marker_counts[str(name)] = int(count)
    else:
        for name, pts in frames_map.items():
            marker_counts[name] = len(pts)

    min_recorded = min(marker_counts.values())
    dropped_frames = max(max(0, expected - c) for c in marker_counts.values())
    extra_frames = max(max(0, c - expected) for c in marker_counts.values())

    per_marker = [_analyze_marker_frames(pts, expected_spacing) for pts in frames_map.values() if len(pts) >= 2]
    if not per_marker:
        raise ValueError(f"Insufficient frame timestamps in {path}")

    worst_marker = max(per_marker, key=lambda m: m.rank_key())

    # RecordedPass.errors is a @property not serialized into JSON; reconstruct from violations - warnings.
    raw_warnings = recorded.get("warnings") or []
    raw_violations = recorded.get("violations") or []
    warnings = tuple(str(w) for w in raw_warnings)
    warnings_set = set(warnings)
    errors = tuple(str(v) for v in raw_violations if str(v) not in warnings_set)

    # Also include scan_failure.json top-level error if present and not already listed.
    top_error = data.get("error")
    if top_error and not errors:
        errors = (str(top_error),)

    worst_reversal_info = readback.get("worst_reversal")
    readback_reversal = (
        float(worst_reversal_info["distance"])
        if isinstance(worst_reversal_info, dict) and "distance" in worst_reversal_info
        else 0.0
    )
    if readback_reversal <= _REVERSAL_NOISE_FLOOR:
        readback_reversal = 0.0
    worst_reversal_distance = max(readback_reversal, *(m.reversal_distance for m in per_marker))

    passed = path.name == "scan.json" and not errors and dropped_frames == 0 and extra_frames == 0
    return RippleMetrics(
        file=str(path),
        expected_frames=expected,
        recorded_frames=min_recorded,
        dropped_frames=dropped_frames,
        extra_frames=extra_frames,
        min_dt_ms=min(m.min_dt_ms for m in per_marker),
        p05_dt_ms=worst_marker.p05_dt_ms,
        median_dt_ms=worst_marker.median_dt_ms,
        p95_dt_ms=worst_marker.p95_dt_ms,
        max_dt_ms=max(m.max_dt_ms for m in per_marker),
        dt_span_ms=max(m.dt_span_ms for m in per_marker),
        dt_std_ms=max(m.dt_std_ms for m in per_marker),
        steady_state_dt_std_ms=max(m.steady_state_dt_std_ms for m in per_marker),
        max_spacing_error=max(m.max_spacing_error for m in per_marker),
        worst_reversal_distance=worst_reversal_distance,
        errors=errors,
        warnings=warnings,
        passed=passed,
    )


def aggregate_metrics(metrics_list: Sequence[RippleMetrics], *, label: str | None = None) -> RippleMetrics:
    """Aggregate multiple volume or repeat ``RippleMetrics`` conservatively (worst-case)."""
    if not metrics_list:
        raise ValueError("Cannot aggregate an empty sequence of RippleMetrics")
    if len(metrics_list) == 1 and label is None:
        return metrics_list[0]

    worst = max(metrics_list, key=lambda m: m.score())
    return RippleMetrics(
        file=label or worst.file,
        expected_frames=sum(m.expected_frames for m in metrics_list),
        recorded_frames=sum(m.recorded_frames for m in metrics_list),
        dropped_frames=sum(m.dropped_frames for m in metrics_list),
        extra_frames=sum(m.extra_frames for m in metrics_list),
        min_dt_ms=min(m.min_dt_ms for m in metrics_list),
        p05_dt_ms=worst.p05_dt_ms,
        median_dt_ms=worst.median_dt_ms,
        p95_dt_ms=worst.p95_dt_ms,
        max_dt_ms=max(m.max_dt_ms for m in metrics_list),
        dt_span_ms=max(m.dt_span_ms for m in metrics_list),
        dt_std_ms=max(m.dt_std_ms for m in metrics_list),
        steady_state_dt_std_ms=max(m.steady_state_dt_std_ms for m in metrics_list),
        max_spacing_error=max(m.max_spacing_error for m in metrics_list),
        worst_reversal_distance=max(m.worst_reversal_distance for m in metrics_list),
        errors=tuple(dict.fromkeys(e for m in metrics_list for e in m.errors)),
        warnings=tuple(dict.fromkeys(w for m in metrics_list for w in m.warnings)),
        passed=all(m.passed for m in metrics_list),
    )


def analyze_oscilloscope_trace(
    times_s: Sequence[float],
    pos_mm: Sequence[float],
    encoder_pos_mm: Sequence[float],
    *,
    target_speed_mm_s: float,
    trim_fraction: float = 0.1,
) -> OscilloscopeMetrics:
    """Analyze 1 kHz Zaber oscilloscope ``pos`` and ``encoder.pos`` traces (pure math)."""
    n = len(times_s)
    if n < 4 or len(pos_mm) != n or len(encoder_pos_mm) != n:
        raise ValueError(
            f"Expected equal-length traces with >= 4 samples; got times={n}, pos={len(pos_mm)}, "
            f"encoder={len(encoder_pos_mm)}"
        )
    if target_speed_mm_s <= 0 or not math.isfinite(target_speed_mm_s):
        raise ValueError(f"target_speed_mm_s must be finite and > 0, got {target_speed_mm_s}")

    dts = [times_s[i + 1] - times_s[i] for i in range(n - 1)]
    if any(dt <= 0 for dt in dts):
        raise ValueError("times_s must be strictly increasing")

    sample_rate_hz = 1.0 / statistics.median(dts)
    following_err_um = [(p - e) * 1000.0 for p, e in zip(pos_mm, encoder_pos_mm, strict=True)]
    velocities = [(encoder_pos_mm[i + 1] - encoder_pos_mm[i]) / dts[i] for i in range(n - 1)]

    direction = 1.0 if encoder_pos_mm[-1] >= encoder_pos_mm[0] else -1.0
    directed_pos = [direction * e for e in encoder_pos_mm]
    worst_reversal_um = _max_position_reversal(directed_pos) * 1000.0

    steady_err = _trim_slice(following_err_um, trim_fraction)
    steady_vel = _trim_slice(velocities, trim_fraction)

    rms_err = math.sqrt(sum(e * e for e in steady_err) / len(steady_err))
    max_err = max(abs(e) for e in steady_err)
    mean_vel = statistics.mean(steady_vel)
    vel_std = statistics.stdev(velocities) if len(velocities) > 1 else 0.0
    steady_vel_std = statistics.stdev(steady_vel) if len(steady_vel) > 1 else 0.0

    return OscilloscopeMetrics(
        sample_count=n,
        sample_rate_hz=sample_rate_hz,
        target_speed_mm_s=target_speed_mm_s,
        rms_following_error_um=rms_err,
        max_following_error_um=max_err,
        mean_velocity_mm_s=mean_vel,
        velocity_std_mm_s=vel_std,
        velocity_ripple_pct=(vel_std / target_speed_mm_s) * 100.0,
        steady_state_velocity_std_mm_s=steady_vel_std,
        steady_state_velocity_ripple_pct=(steady_vel_std / target_speed_mm_s) * 100.0,
        worst_reversal_um=worst_reversal_um,
    )


class VoxelClient:
    """Minimal stdlib REST client for ``vxl serve`` station & instrument endpoints."""

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
        self._session_id = session_id
        self._auth_header = (
            "Basic " + base64.b64encode(f"voxel:{password}".encode()).decode("ascii") if password else None
        )

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers: dict[str, str] = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self._auth_header is not None:
            headers["Authorization"] = self._auth_header
        req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {method} {path}: {detail}") from exc

    def _station(self, suffix: str) -> str:
        return f"/api/stations/{self.station_id}{suffix}"

    def resolve_session_id(self) -> str:
        """Return configured session_id or discover the active session from station snapshot."""
        if self._session_id:
            return self._session_id
        snapshot = self._request("GET", self._station("/snapshot"))
        session = snapshot.get("session") if isinstance(snapshot, dict) else None
        info = session.get("info") if isinstance(session, dict) else None
        sid = info.get("id") if isinstance(info, dict) else None
        if not sid:
            status = snapshot.get("status") if isinstance(snapshot, dict) else None
            raise RuntimeError(
                f"No active session on station {self.station_id} (status={status!r}); "
                "open a session or pass --session-id"
            )
        self._session_id = str(sid)
        return self._session_id

    def _instrument(self, suffix: str) -> str:
        return self._station(f"/sessions/{self.resolve_session_id()}/instrument{suffix}")

    def device_command(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke a device command and unwrap its ``rigup.Result`` envelope."""
        path = self._instrument(f"/devices/{self.device_id}/commands/{command}")
        resp = self._request("POST", path, {"args": list(args), "kwargs": kwargs})
        if isinstance(resp, dict) and "ok" in resp:
            if not resp["ok"]:
                msg = resp.get("msg", "unknown command error")
                raise RuntimeError(f"Device command {self.device_id}.{command} failed: {msg}")
            return resp.get("value")
        return resp

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
                enable_feed_forward=bool(enable_feed_forward),
                paramset=paramset,
            )
        )

    def set_pid_servo_tuning(
        self,
        p: float,
        i: float,
        d: float,
        fc: float,
        *,
        paramset: str = "live",
    ) -> dict[str, Any]:
        return dict(
            self.device_command(
                "set_pid_servo_tuning",
                p=float(p),
                i=float(i),
                d=float(d),
                fc=float(fc),
                paramset=paramset,
            )
        )

    def load_servo_paramset(self, to_paramset: str, from_paramset: str) -> dict[str, Any]:
        return dict(
            self.device_command(
                "load_servo_paramset",
                to_paramset=to_paramset,
                from_paramset=from_paramset,
            )
        )

    def set_servo_startup_paramset(self, paramset: str) -> str:
        return str(self.device_command("set_servo_startup_paramset", paramset=paramset))

    def read_telemetry(self, names: Sequence[str] = TELEMETRY_SETTINGS) -> dict[str, float]:
        """Best-effort read of read-only servo/power telemetry settings."""
        telemetry: dict[str, float] = {}
        for name in names:
            try:
                telemetry[name] = self.get_setting(name)
            except RuntimeError:
                continue
        return telemetry

    def snapshot_servo_tuning(self, paramset: str = "live") -> dict[str, Any]:
        """Capture current servo tuning plus read-only telemetry."""
        snapshot = self.get_servo_tuning(paramset=paramset)
        snapshot["telemetry"] = self.read_telemetry()
        return snapshot

    def restore_servo_tuning(self, snapshot: dict[str, Any], *, paramset: str = "live") -> dict[str, Any]:
        """Restore servo tuning from a ``snapshot_servo_tuning()`` payload."""
        simple = snapshot.get("simple")
        if isinstance(simple, dict) and simple.get("is_used"):
            params = simple.get("params") or {}
            stiffness = _extract_stiffness_param(params)
            if stiffness is not None and simple.get("load_mass_kg") is not None:
                return self.set_simple_servo_tuning(
                    float(simple["load_mass_kg"]),
                    stiffness,
                    paramset=paramset,
                )

        pid = snapshot.get("pid")
        if isinstance(pid, dict) and all(k in pid for k in ("p", "i", "d", "fc")):
            return self.set_pid_servo_tuning(
                float(pid["p"]),
                float(pid["i"]),
                float(pid["d"]),
                float(pid["fc"]),
                paramset=paramset,
            )

        print(
            "  [warn] Snapshot has neither usable simple nor PID tuning; falling back to DEFAULT paramset",
            file=sys.stderr,
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
        """Launch a Voxel acquisition task, wait for completion, and analyze all scan files."""
        manifest = self._request(
            "POST",
            self._instrument("/acquisition"),
            {"storage": {"path": storage_path}, "task_ids": [task_id]},
        )
        acq_id = manifest["id"]
        deadline = time.monotonic() + timeout_s
        status_obj: dict[str, Any] = manifest
        status = status_obj.get("status")
        while time.monotonic() < deadline:
            status_obj = self._request("GET", self._station(f"/acquisitions/{acq_id}"))
            status = status_obj.get("status")
            if status in TERMINAL_ACQUISITION_STATUSES:
                break
            time.sleep(poll_s)
        else:
            self._request("POST", self._instrument("/acquisition/stop"), {})
            raise TimeoutError(f"Acquisition {acq_id} did not finish within {timeout_s}s")

        run_dir = data_root / storage_path
        volumes_dir = run_dir / "volumes"
        candidates = (
            sorted(list(volumes_dir.rglob("scan.json")) + list(volumes_dir.rglob("scan_failure.json")))
            if volumes_dir.exists()
            else []
        )
        if not candidates:
            failure = status_obj.get("failure")
            if status == "failed" or failure:
                raise RuntimeError(f"Acquisition {acq_id} failed with no scan files under {run_dir}: {failure}")
            raise FileNotFoundError(f"No scan.json or scan_failure.json found under {run_dir} (status={status!r})")

        per_volume = [analyze_scan_file(path) for path in candidates]
        return aggregate_metrics(per_volume, label=str(run_dir) if len(per_volume) > 1 else None)


def _extract_stiffness_param(params: dict[str, Any]) -> float | None:
    for key, val in params.items():
        if key.lower() == "stiffness":
            return float(val)
    if params:
        return float(next(iter(params.values())))
    return None


def _rollback_tuning(client: VoxelClient, original: dict[str, Any]) -> None:
    """Restore original tuning, falling back to DEFAULT paramset if restore fails."""
    try:
        client.restore_servo_tuning(original, paramset="live")
    except Exception as exc:
        print(f"  [warn] Restore failed ({exc}); loading DEFAULT paramset into LIVE", file=sys.stderr)
        client.load_servo_paramset("live", "default")


def _format_metrics(label: str, metrics: RippleMetrics, telemetry: dict[str, float]) -> str:
    telem_str = " ".join(f"{k}={v:.3g}" for k, v in telemetry.items())
    suffix = f" {telem_str}" if telem_str else ""
    return (
        f"  [{label}] passed={metrics.passed} frames={metrics.recorded_frames}/{metrics.expected_frames} "
        f"(drop={metrics.dropped_frames}, extra={metrics.extra_frames}) "
        f"steady_std={metrics.steady_state_dt_std_ms:.4f}ms "
        f"p95-p05={metrics.p95_dt_ms - metrics.p05_dt_ms:.3f}ms "
        f"span={metrics.dt_span_ms:.3f}ms rev={metrics.worst_reversal_distance:.4g}{suffix}"
    )


def _validate_sweep_options(
    *,
    stiffness_values: Sequence[float],
    repeats: int,
    apply_best: bool,
    persist_paramset: str | None,
) -> None:
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if persist_paramset is not None:
        if persist_paramset not in STORED_PARAMSET_SLOTS:
            raise ValueError(f"persist_paramset must be one of 1..9, got {persist_paramset!r}")
        if not apply_best:
            raise ValueError("--persist-paramset requires --apply-best")
    for s in stiffness_values:
        if not math.isfinite(s) or not 0.0 <= s <= 1.0:
            raise ValueError(f"stiffness must be finite and within [0.0, 1.0], got {s}")


def simple_tune(
    client: VoxelClient,
    *,
    task_id: str,
    storage_prefix: str,
    data_root: Path,
    load_mass_kg: float | None,
    stiffness_values: tuple[float, ...] = DEFAULT_STIFFNESS_VALUES,
    enable_feed_forward: bool = True,
    repeats: int = 2,
    apply_best: bool = False,
    persist_paramset: str | None = None,
    backup_path: Path = Path("zaber_tuning_backup.json"),
) -> dict[str, Any]:
    """Sweep Zaber Simple Tuning stiffness over repeated Voxel test scans."""
    _validate_sweep_options(
        stiffness_values=stiffness_values,
        repeats=repeats,
        apply_best=apply_best,
        persist_paramset=persist_paramset,
    )

    print("Snapshotting current Zaber live servo tuning...")
    original = client.snapshot_servo_tuning("live")
    backup_path.write_text(json.dumps(original, indent=2) + "\n")
    print(f"Saved backup to {backup_path}")

    effective_mass = load_mass_kg
    if effective_mass is None:
        simple_info = original.get("simple")
        if isinstance(simple_info, dict) and simple_info.get("load_mass_kg") is not None:
            effective_mass = float(simple_info["load_mass_kg"])
        else:
            raise ValueError("--load-mass-kg is required when current tuning has no simple tuning load_mass_kg")
    if not math.isfinite(effective_mass) or effective_mass < 0:
        raise ValueError(f"load_mass_kg must be finite and >= 0, got {effective_mass}")

    run_idx = 0

    def eval_candidate(label: str) -> tuple[RippleMetrics, dict[str, float]]:
        nonlocal run_idx
        repeat_metrics: list[RippleMetrics] = []
        telemetry: dict[str, float] = {}
        for rep in range(1, repeats + 1):
            run_idx += 1
            rel_path = f"{storage_prefix}/tune_{run_idx:03d}_{label}_r{rep}"
            repeat_metrics.append(client.run_test_scan(task_id=task_id, storage_path=rel_path, data_root=data_root))
            telemetry = client.read_telemetry()
        agg = aggregate_metrics(repeat_metrics, label=label if repeats > 1 else None)
        print(_format_metrics(label, agg, telemetry))
        return agg, telemetry

    best_stiffness: float | None = None  # None indicates baseline won
    results: list[dict[str, Any]] = []

    try:
        best_metrics, baseline_telem = eval_candidate("baseline")
        results.append(
            {"label": "baseline", "stiffness": None, "metrics": asdict(best_metrics), "telemetry": baseline_telem}
        )

        for stiffness in stiffness_values:
            label = f"stiffness_{stiffness:.2f}"
            try:
                client.set_simple_servo_tuning(
                    effective_mass,
                    stiffness,
                    enable_feed_forward=enable_feed_forward,
                    paramset="live",
                )
                metrics, telem = eval_candidate(label)
            except Exception as exc:
                print(f"  [FAULT] {label} failed ({exc}); restoring baseline tuning", file=sys.stderr)
                _rollback_tuning(client, original)
                results.append({"label": label, "stiffness": stiffness, "error": str(exc)})
                continue

            results.append({"label": label, "stiffness": stiffness, "metrics": asdict(metrics), "telemetry": telem})
            if metrics.score() < best_metrics.score():
                print(f"  [IMPROVED] stiffness={stiffness:.2f} (steady_std={metrics.steady_state_dt_std_ms:.4f}ms)")
                best_metrics = metrics
                best_stiffness = stiffness

        print("\n=== Servo Tuning Summary ===")
        print(f"Load mass: {effective_mass} kg | Feed-forward: {enable_feed_forward}")
        print(f"Best candidate: {'baseline' if best_stiffness is None else f'stiffness={best_stiffness:.2f}'}")
        print(_format_metrics("best", best_metrics, {}))
    except BaseException:
        print("\nInterrupted or failed; rolling back Zaber servo tuning...", file=sys.stderr)
        _rollback_tuning(client, original)
        raise

    if apply_best:
        if best_stiffness is not None:
            client.set_simple_servo_tuning(
                effective_mass,
                best_stiffness,
                enable_feed_forward=enable_feed_forward,
                paramset="live",
            )
            print(f"Applied stiffness={best_stiffness:.2f} to LIVE paramset.")
        else:
            _rollback_tuning(client, original)
            print("Baseline remained best; restored original LIVE tuning.")

        if persist_paramset is not None:
            client.load_servo_paramset(persist_paramset, "live")
            startup = client.set_servo_startup_paramset(persist_paramset)
            print(f"Persisted LIVE tuning to slot {persist_paramset} and set startup paramset={startup}.")
    else:
        print("\nRestoring original baseline tuning (pass --apply-best to keep best tuning)...")
        _rollback_tuning(client, original)

    return {
        "load_mass_kg": effective_mass,
        "enable_feed_forward": enable_feed_forward,
        "best_stiffness": best_stiffness,
        "best_metrics": asdict(best_metrics),
        "applied": apply_best,
        "persisted_paramset": persist_paramset if apply_best else None,
        "candidates": results,
    }


def _parse_host_port(host_spec: str, default_port: int = 55551) -> tuple[str, int]:
    if ":" in host_spec:
        host, port_str = host_spec.rsplit(":", 1)
        return host, int(port_str)
    return host_spec, default_port


def run_oscilloscope_tune(
    *,
    host: str,
    port: int,
    axis_number: int,
    scan_start_mm: float,
    scan_end_mm: float,
    scan_speed_mm_s: float,
    load_mass_kg: float,
    stiffness_values: tuple[float, ...] = DEFAULT_STIFFNESS_VALUES,
    enable_feed_forward: bool = True,
    repeats: int = 2,
    apply_best: bool = False,
    persist_paramset: str | None = None,
    backup_path: Path = Path("zaber_tuning_backup.json"),
) -> dict[str, Any]:
    """Direct 1 kHz Zaber Oscilloscope + ServoTuner sweep (lazy ``zaber_motion`` import)."""
    _validate_sweep_options(
        stiffness_values=stiffness_values,
        repeats=repeats,
        apply_best=apply_best,
        persist_paramset=persist_paramset,
    )
    if not math.isfinite(load_mass_kg) or load_mass_kg < 0:
        raise ValueError(f"load_mass_kg must be finite and >= 0, got {load_mass_kg}")
    if not math.isfinite(scan_speed_mm_s) or scan_speed_mm_s <= 0:
        raise ValueError(f"scan_speed_mm_s must be finite and > 0, got {scan_speed_mm_s}")

    from zaber_motion import Units  # noqa: PLC0415
    from zaber_motion.ascii import Connection, ServoTuner, ServoTuningParam, ServoTuningParamset  # noqa: PLC0415

    slot_map = {str(i): getattr(ServoTuningParamset, f"P_{i}") for i in range(1, 10)}

    def apply_simple(tuner: Any, stiffness: float) -> None:
        defs = tuner.get_simple_tuning_param_definitions()
        param_name = next((d.name for d in defs if d.name.lower() == "stiffness"), "Stiffness")
        tuner.set_simple_tuning(
            ServoTuningParamset.STAGING,
            [ServoTuningParam(name=param_name, value=float(stiffness))],
            load_inertia=float(load_mass_kg),
            load_inertia_units=Units.INERTIA_KILOGRAMS,
            enable_feed_forward=enable_feed_forward,
        )
        tuner.load_paramset(ServoTuningParamset.LIVE, ServoTuningParamset.STAGING)

    def capture_pass(axis: Any, scope: Any) -> OscilloscopeMetrics:
        axis.move_absolute(scan_start_mm, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
        scope.clear()
        scope.set_timebase(1.0, Units.TIME_MILLISECONDS)
        scope.add_channel(axis_number, "pos")
        scope.add_channel(axis_number, "encoder.pos")
        scope.start()
        try:
            axis.move_absolute(
                scan_end_mm,
                Units.LENGTH_MILLIMETRES,
                velocity=scan_speed_mm_s,
                velocity_unit=Units.VELOCITY_MILLIMETRES_PER_SECOND,
                wait_until_idle=True,
            )
        finally:
            scope.stop()

        channels = {ch.setting: ch for ch in scope.read()}
        pos_ch = channels["pos"]
        enc_ch = channels["encoder.pos"]
        times_s = pos_ch.get_sample_times(Units.TIME_SECONDS)
        pos_mm = pos_ch.get_data(Units.LENGTH_MILLIMETRES)
        enc_mm = enc_ch.get_data(Units.LENGTH_MILLIMETRES)
        return analyze_oscilloscope_trace(times_s, pos_mm, enc_mm, target_speed_mm_s=scan_speed_mm_s)

    with Connection.open_tcp(host, port) as conn:
        devices = conn.detect_devices()
        if not devices:
            raise RuntimeError(f"No Zaber devices found at {host}:{port}")
        device = devices[0]
        axis = device.get_axis(axis_number)
        tuner = ServoTuner(axis)
        scope = device.oscilloscope

        # Snapshot original LIVE tuning to backup JSON and keep a copy in STAGING? No, keep original info.
        orig_info = tuner.get_tuning(ServoTuningParamset.LIVE)
        orig_params = [ServoTuningParam(name=p.name, value=p.value) for p in orig_info.params]
        backup_payload = {
            "type": orig_info.type,
            "version": orig_info.version,
            "params": {p.name: p.value for p in orig_info.params},
        }
        backup_path.write_text(json.dumps(backup_payload, indent=2) + "\n")
        print(f"Saved direct Zaber tuning backup to {backup_path}")

        def restore_original() -> None:
            try:
                tuner.set_tuning(ServoTuningParamset.STAGING, orig_params)
                tuner.load_paramset(ServoTuningParamset.LIVE, ServoTuningParamset.STAGING)
            except Exception as exc:
                print(f"  [warn] Direct restore failed ({exc}); loading DEFAULT into LIVE", file=sys.stderr)
                tuner.load_paramset(ServoTuningParamset.LIVE, ServoTuningParamset.DEFAULT)

        def eval_scope(label: str) -> OscilloscopeMetrics:
            runs = [capture_pass(axis, scope) for _ in range(repeats)]
            worst = max(runs, key=lambda m: m.score())
            print(
                f"  [{label}] samples={worst.sample_count}@{worst.sample_rate_hz:.0f}Hz "
                f"steady_vel_ripple={worst.steady_state_velocity_ripple_pct:.2f}% "
                f"rms_err={worst.rms_following_error_um:.2f}um "
                f"max_err={worst.max_following_error_um:.2f}um rev={worst.worst_reversal_um:.2f}um"
            )
            return worst

        best_stiffness: float | None = None
        results: list[dict[str, Any]] = []
        try:
            best_metrics = eval_scope("baseline")
            results.append({"label": "baseline", "stiffness": None, "metrics": asdict(best_metrics)})

            for stiffness in stiffness_values:
                label = f"stiffness_{stiffness:.2f}"
                try:
                    apply_simple(tuner, stiffness)
                    metrics = eval_scope(label)
                except Exception as exc:
                    print(f"  [FAULT] {label} failed ({exc}); restoring baseline tuning", file=sys.stderr)
                    restore_original()
                    results.append({"label": label, "stiffness": stiffness, "error": str(exc)})
                    continue

                results.append({"label": label, "stiffness": stiffness, "metrics": asdict(metrics)})
                if metrics.score() < best_metrics.score():
                    print(
                        f"  [IMPROVED] stiffness={stiffness:.2f} "
                        f"(ripple={metrics.steady_state_velocity_ripple_pct:.2f}%, "
                        f"rms_err={metrics.rms_following_error_um:.2f}um)"
                    )
                    best_metrics = metrics
                    best_stiffness = stiffness
        except BaseException:
            print("\nInterrupted or failed; restoring original Zaber tuning...", file=sys.stderr)
            restore_original()
            raise

        if apply_best:
            if best_stiffness is not None:
                apply_simple(tuner, best_stiffness)
                print(f"Applied stiffness={best_stiffness:.2f} to LIVE paramset.")
            else:
                restore_original()
                print("Baseline remained best; restored original LIVE tuning.")
            if persist_paramset is not None:
                target_slot = slot_map[persist_paramset]
                tuner.load_paramset(target_slot, ServoTuningParamset.LIVE)
                tuner.set_startup_paramset(target_slot)
                print(f"Persisted LIVE tuning to slot {persist_paramset} and set startup paramset.")
        else:
            print("\nRestoring original baseline tuning (pass --apply-best to keep best tuning)...")
            restore_original()

    return {
        "load_mass_kg": load_mass_kg,
        "enable_feed_forward": enable_feed_forward,
        "best_stiffness": best_stiffness,
        "best_metrics": asdict(best_metrics),
        "applied": apply_best,
        "persisted_paramset": persist_paramset if apply_best else None,
        "candidates": results,
    }


def _parse_stiffness_values(raw: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise ValueError("--stiffness-values must contain at least one float")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyze", type=Path, help="Analyze an existing scan.json or scan_failure.json and exit")

    voxel_group = parser.add_argument_group("Voxel REST mode")
    voxel_group.add_argument("--base-url", default="http://127.0.0.1:8000", help="Voxel server base URL")
    voxel_group.add_argument("--station-id", help="Station UUID")
    voxel_group.add_argument("--session-id", help="Active session UUID (auto-discovered if omitted)")
    voxel_group.add_argument("--device-id", default="x_axis", help="Zaber axis device UID (default: x_axis)")
    voxel_group.add_argument(
        "--password",
        help="Voxel control password (or set VOXEL_WEB_PASSWORD; falls back to VOXEL_PASSWORD)",
    )
    voxel_group.add_argument("--snapshot", type=Path, help="Save current Zaber servo tuning snapshot JSON and exit")
    voxel_group.add_argument("--restore", type=Path, help="Restore Zaber servo tuning from snapshot JSON and exit")
    voxel_group.add_argument("--task-id", help="Acquisition task ID to run for each tuning evaluation")
    voxel_group.add_argument("--data-root", type=Path, help="Filesystem root corresponding to storage.path")
    voxel_group.add_argument("--storage-prefix", default="zaber-tune", help="Storage subfolder prefix for test runs")

    tune_group = parser.add_argument_group("ServoTuner sweep options")
    tune_group.add_argument("--load-mass-kg", type=float, help="Stage payload mass in kg for Zaber Simple Tuning")
    tune_group.add_argument(
        "--stiffness-values",
        default="0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated stiffness values in [0.0, 1.0] (default: 0.5,0.6,0.7,0.8,0.9,1.0)",
    )
    tune_group.add_argument("--no-feed-forward", action="store_true", help="Disable Simple Tuning feed-forward")
    tune_group.add_argument("--repeats", type=int, default=2, help="Scan repeats per stiffness candidate (default: 2)")
    tune_group.add_argument("--backup", type=Path, default=Path("zaber_tuning_backup.json"), help="Backup JSON path")
    tune_group.add_argument("--apply-best", action="store_true", help="Keep best tuning in LIVE instead of restoring")
    tune_group.add_argument(
        "--persist-paramset",
        choices=STORED_PARAMSET_SLOTS,
        help="Copy winning LIVE tuning to stored slot 1..9 and set as startup paramset (requires --apply-best)",
    )

    scope_group = parser.add_argument_group("Direct Zaber Oscilloscope mode (no Voxel server)")
    scope_group.add_argument("--zaber-host", help="Zaber TCP host (or host:port) for direct oscilloscope tuning")
    scope_group.add_argument("--zaber-port", type=int, default=55551, help="Zaber TCP port (default: 55551)")
    scope_group.add_argument("--axis", type=int, default=1, help="Zaber axis number (default: 1)")
    scope_group.add_argument("--scan-start-mm", type=float, help="Direct scan start position in mm")
    scope_group.add_argument("--scan-end-mm", type=float, help="Direct scan end position in mm")
    scope_group.add_argument("--scan-speed-mm-s", type=float, help="Direct scan velocity in mm/s")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.analyze is not None:
        print(json.dumps(asdict(analyze_scan_file(args.analyze)), indent=2))
        return 0

    if args.persist_paramset is not None and not args.apply_best:
        parser.error("--persist-paramset requires --apply-best")

    if args.zaber_host is not None:
        if (
            args.scan_start_mm is None
            or args.scan_end_mm is None
            or args.scan_speed_mm_s is None
            or args.load_mass_kg is None
        ):
            parser.error("--zaber-host requires --scan-start-mm, --scan-end-mm, --scan-speed-mm-s, and --load-mass-kg")
        host, port = _parse_host_port(args.zaber_host, args.zaber_port)
        run_oscilloscope_tune(
            host=host,
            port=port,
            axis_number=args.axis,
            scan_start_mm=args.scan_start_mm,
            scan_end_mm=args.scan_end_mm,
            scan_speed_mm_s=args.scan_speed_mm_s,
            load_mass_kg=args.load_mass_kg,
            stiffness_values=_parse_stiffness_values(args.stiffness_values),
            enable_feed_forward=not args.no_feed_forward,
            repeats=args.repeats,
            apply_best=args.apply_best,
            persist_paramset=args.persist_paramset,
            backup_path=args.backup,
        )
        return 0

    if not args.station_id:
        parser.error("--station-id is required unless --analyze or --zaber-host is used")

    password = (
        args.password
        if args.password is not None
        else (os.environ.get("VOXEL_WEB_PASSWORD") or os.environ.get("VOXEL_PASSWORD"))
    )
    client = VoxelClient(
        args.base_url,
        args.station_id,
        args.device_id,
        session_id=args.session_id,
        password=password,
    )

    if args.snapshot is not None:
        snap = client.snapshot_servo_tuning("live")
        args.snapshot.write_text(json.dumps(snap, indent=2) + "\n")
        print(f"Saved snapshot to {args.snapshot}:\n{json.dumps(snap, indent=2)}")
        return 0

    if args.restore is not None:
        snap = json.loads(args.restore.read_text())
        restored = client.restore_servo_tuning(snap, paramset="live")
        print(f"Restored LIVE servo tuning:\n{json.dumps(restored, indent=2)}")
        return 0

    if not args.task_id or args.data_root is None:
        parser.error("--task-id and --data-root are required for Voxel tuning")

    simple_tune(
        client,
        task_id=args.task_id,
        storage_prefix=args.storage_prefix,
        data_root=args.data_root,
        load_mass_kg=args.load_mass_kg,
        stiffness_values=_parse_stiffness_values(args.stiffness_values),
        enable_feed_forward=not args.no_feed_forward,
        repeats=args.repeats,
        apply_best=args.apply_best,
        persist_paramset=args.persist_paramset,
        backup_path=args.backup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
