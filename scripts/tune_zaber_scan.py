#!/usr/bin/env python3
"""Bounded empirical Zaber servo/PID tuning script for continuous-scan velocity ripple.

Talks to a running ``vxl serve`` instance via the device command REST API
(``get_setting`` / ``set_setting`` on ``ZaberLinearAxis``) and evaluates each
candidate gain set by running a short acquisition task and measuring ``dt_ms``
ripple from the resulting ``scan.json`` / ``scan_failure.json``.

Examples:
    # 1. Analyze an existing scan.json / scan_failure.json locally:
    python scripts/tune_zaber_scan.py --analyze /path/to/scan.json

    # 2. Snapshot current Zaber tuning settings to a backup JSON:
    python scripts/tune_zaber_scan.py --station-id <UUID> --snapshot zaber_backup.json

    # 3. Restore settings from a backup JSON:
    python scripts/tune_zaber_scan.py --station-id <UUID> --restore zaber_backup.json

    # 4. Run bounded coordinate search around current gains:
    python scripts/tune_zaber_scan.py \\
        --station-id <UUID> \\
        --task-id <TASK_ID> \\
        --data-root /data/users/cfxuser \\
        --apply-best
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS: tuple[str, ...] = (
    "ictrl.pidff.kp",
    "ictrl.pidff.ki",
    "ictrl.pidff.kd",
    "ictrl.pidff.lp",
    "ictrl.ff.kp",
    "ictrl.ff.kd",
    "motion.accel.ramptime",
)

DEFAULT_TUNE_PARAMS: tuple[str, ...] = (
    "ictrl.pidff.kp",
    "ictrl.pidff.ki",
    "ictrl.pidff.kd",
    "ictrl.ff.kp",
    "ictrl.ff.kd",
)


@dataclass(frozen=True)
class RippleMetrics:
    file: str
    expected_frames: int
    recorded_frames: int
    dropped_frames: int
    min_dt_ms: float
    median_dt_ms: float
    max_dt_ms: float
    dt_span_ms: float
    dt_std_ms: float
    passed: bool

    def score(self) -> tuple[int, float, float]:
        """Lower is better: zero dropped frames first, then smallest dt span & std."""
        return (self.dropped_frames, self.dt_span_ms, self.dt_std_ms)


def analyze_scan_file(path: Path) -> RippleMetrics:
    with path.open() as fh:
        data = json.load(fh)

    expected = int(data["plan"]["division"]["cycles"]) if "plan" in data else int(data["division"]["cycles"])

    recorded = data.get("recorded") or {}
    frames_map: dict[str, list[dict[str, Any]]] = recorded.get("frames") or {}
    if not frames_map:
        raise ValueError(f"No recorded frames found in {path}")

    min_recorded = min(len(pts) for pts in frames_map.values())
    all_dt_ms: list[float] = []
    for pts in frames_map.values():
        if len(pts) < 2:
            continue
        times = [float(p["time_s"]) for p in pts]
        all_dt_ms.extend((times[i + 1] - times[i]) * 1000.0 for i in range(len(times) - 1))

    if not all_dt_ms:
        raise ValueError(f"Insufficient frame timestamps in {path}")

    min_dt = min(all_dt_ms)
    max_dt = max(all_dt_ms)
    passed = path.name == "scan.json" and not recorded.get("errors") and min_recorded == expected
    return RippleMetrics(
        file=str(path),
        expected_frames=expected,
        recorded_frames=min_recorded,
        dropped_frames=max(0, expected - min_recorded),
        min_dt_ms=min_dt,
        median_dt_ms=statistics.median(all_dt_ms),
        max_dt_ms=max_dt,
        dt_span_ms=max_dt - min_dt,
        dt_std_ms=statistics.stdev(all_dt_ms) if len(all_dt_ms) > 1 else 0.0,
        passed=passed,
    )


class VoxelClient:
    def __init__(self, base_url: str, station_id: str, device_id: str) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must use http:// or https://, got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.station_id = station_id
        self.device_id = device_id

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
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

    def device_command(self, command: str, *args: Any) -> Any:
        path = self._station(f"/instrument/devices/{self.device_id}/commands/{command}")
        return self._request("POST", path, {"args": list(args), "kwargs": {}})

    def get_setting(self, name: str) -> float:
        return float(self.device_command("get_setting", name))

    def set_setting(self, name: str, value: float) -> float:
        return float(self.device_command("set_setting", name, float(value)))

    def snapshot_settings(self, names: tuple[str, ...]) -> dict[str, float]:
        values: dict[str, float] = {}
        for name in names:
            try:
                values[name] = self.get_setting(name)
            except RuntimeError as exc:
                print(f"  [skip] {name}: {exc}", file=sys.stderr)
        return values

    def restore_settings(self, values: dict[str, float]) -> None:
        for name, val in values.items():
            readback = self.set_setting(name, val)
            print(f"  restored {name} = {readback}")

    def run_test_scan(
        self,
        *,
        task_id: str,
        storage_path: str,
        data_root: Path,
        poll_s: float = 1.0,
        timeout_s: float = 120.0,
    ) -> RippleMetrics:
        manifest = self._request(
            "POST",
            self._station("/instrument/acquisition"),
            {"storage": {"path": storage_path}, "task_ids": [task_id]},
        )
        acq_id = manifest["id"]
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status_obj = self._request("GET", self._station(f"/acquisitions/{acq_id}"))
            status = status_obj.get("status")
            if status in ("completed", "failed", "cancelled", "aborted"):
                break
            time.sleep(poll_s)
        else:
            self._request("POST", self._station("/instrument/acquisition/stop"), {})
            raise TimeoutError(f"Acquisition {acq_id} did not finish within {timeout_s}s")

        run_dir = data_root / storage_path
        volumes_dir = run_dir / "volumes"
        candidates = sorted(list(volumes_dir.rglob("scan.json")) + list(volumes_dir.rglob("scan_failure.json")))
        if not candidates:
            raise FileNotFoundError(f"No scan.json or scan_failure.json found under {run_dir}")
        return analyze_scan_file(candidates[-1])


def coordinate_tune(
    client: VoxelClient,
    *,
    task_id: str,
    storage_prefix: str,
    data_root: Path,
    params: tuple[str, ...],
    factors: tuple[float, ...],
    apply_best: bool,
    backup_path: Path,
) -> None:
    print("Snapshotting current Zaber settings...")
    originals = client.snapshot_settings(DEFAULT_SETTINGS)
    if not originals:
        raise RuntimeError("Could not read any Zaber settings; check station_id/device_id.")
    backup_path.write_text(json.dumps(originals, indent=2) + "\n")
    print(f"Saved backup to {backup_path}: {originals}")

    best_gains = dict(originals)
    run_idx = 0

    def eval_candidate(label: str) -> RippleMetrics:
        nonlocal run_idx
        run_idx += 1
        rel_path = f"{storage_prefix}/tune_{run_idx:03d}_{label}"
        metrics = client.run_test_scan(task_id=task_id, storage_path=rel_path, data_root=data_root)
        print(
            f"  [{label}] passed={metrics.passed} frames={metrics.recorded_frames}/{metrics.expected_frames} "
            f"dt_ms=[{metrics.min_dt_ms:.3f}..{metrics.max_dt_ms:.3f}] "
            f"span={metrics.dt_span_ms:.3f}ms std={metrics.dt_std_ms:.4f}ms"
        )
        return metrics

    try:
        best_metrics = eval_candidate("baseline")
        for param in params:
            if param not in best_gains:
                continue
            base_val = best_gains[param]
            if base_val == 0 or not math.isfinite(base_val):
                continue
            print(f"\nSweeping {param} (current={base_val})...")
            for factor in factors:
                if math.isclose(factor, 1.0):
                    continue
                candidate_val = round(base_val * factor, 6)
                client.set_setting(param, candidate_val)
                try:
                    metrics = eval_candidate(f"{param}_{factor:.2f}")
                except Exception as exc:
                    print(f"  [FAULT] {param}={candidate_val} failed ({exc}); reverting {param} to {best_gains[param]}")
                    client.set_setting(param, best_gains[param])
                    continue

                if metrics.score() < best_metrics.score():
                    print(f"  [IMPROVED] {param}: {best_gains[param]} -> {candidate_val}")
                    best_gains[param] = candidate_val
                    best_metrics = metrics
                else:
                    client.set_setting(param, best_gains[param])

        print("\n=== Tuning Summary ===")
        print(f"Best gains: {best_gains}")
        print(
            f"Best ripple: dt_ms=[{best_metrics.min_dt_ms:.3f}..{best_metrics.max_dt_ms:.3f}] "
            f"span={best_metrics.dt_span_ms:.3f}ms std={best_metrics.dt_std_ms:.4f}ms"
        )
    finally:
        if apply_best:
            print("\nApplying best gains...")
            client.restore_settings(best_gains)
        else:
            print("\nRestoring original baseline gains (pass --apply-best to keep tuned gains)...")
            client.restore_settings(originals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyze", type=Path, help="Analyze an existing scan.json or scan_failure.json and exit")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Voxel server base URL")
    parser.add_argument("--station-id", help="Station UUID")
    parser.add_argument("--device-id", default="x_axis", help="Zaber axis device UID (default: x_axis)")
    parser.add_argument("--snapshot", type=Path, help="Read current Zaber settings into JSON and exit")
    parser.add_argument("--restore", type=Path, help="Restore Zaber settings from backup JSON and exit")
    parser.add_argument("--task-id", help="Acquisition task ID to run for each tuning evaluation")
    parser.add_argument("--data-root", type=Path, help="Filesystem root corresponding to storage.path")
    parser.add_argument("--storage-prefix", default="zaber-tune", help="Storage subfolder prefix for test runs")
    parser.add_argument(
        "--factors",
        default="0.85,0.92,1.08,1.15",
        help="Comma-separated multiplicative factors to try per parameter (default: 0.85,0.92,1.08,1.15)",
    )
    parser.add_argument("--backup", type=Path, default=Path("zaber_tuning_backup.json"), help="Backup JSON path")
    parser.add_argument("--apply-best", action="store_true", help="Keep best gains instead of restoring originals")
    args = parser.parse_args()

    if args.analyze is not None:
        m = analyze_scan_file(args.analyze)
        print(json.dumps(m.__dict__, indent=2))
        return 0

    if not args.station_id:
        parser.error("--station-id is required unless --analyze is used")

    client = VoxelClient(args.base_url, args.station_id, args.device_id)

    if args.snapshot is not None:
        vals = client.snapshot_settings(DEFAULT_SETTINGS)
        args.snapshot.write_text(json.dumps(vals, indent=2) + "\n")
        print(f"Saved snapshot to {args.snapshot}:\n{json.dumps(vals, indent=2)}")
        return 0

    if args.restore is not None:
        vals = json.loads(args.restore.read_text())
        client.restore_settings(vals)
        return 0

    if not args.task_id or args.data_root is None:
        parser.error("--task-id and --data-root are required for tuning")

    factors = tuple(float(x.strip()) for x in args.factors.split(",") if x.strip())
    coordinate_tune(
        client,
        task_id=args.task_id,
        storage_prefix=args.storage_prefix,
        data_root=args.data_root,
        params=DEFAULT_TUNE_PARAMS,
        factors=factors,
        apply_best=args.apply_best,
        backup_path=args.backup,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
