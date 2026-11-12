"""Tests for cross-process stage gates (`HostStageGate`) and CPU-aware thread budgets."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
import types
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from ome_zarr_writer import Local, OMEZarrWriter, ScaleLevel, UIVec3D, UVec3D, WriterConfig
from ome_zarr_writer.array.ts import TSArrayReader
from ome_zarr_writer.concurrency import (
    HostStageGate,
    default_io_concurrency,
    default_numba_threads,
    downsample_gate,
    flush_gate,
    resolve_lock_dir,
    stage_limit,
)

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as MpEvent
    from multiprocessing.synchronize import Lock as MpLock


# ---------------------------------------------------------------------------
# Spawn-safe top-level helpers for child-process gate tests
# ---------------------------------------------------------------------------


def _gate_worker(
    stage: str,
    limit: int,
    hold_s: float,
    ready_q: mp.Queue,
    go_event: MpEvent,
    counter_lock: MpLock,
    active_val: object,
    peak_val: object,
    result_q: mp.Queue,
) -> None:
    ready_q.put(os.getpid())
    if not go_event.wait(timeout=15.0):
        raise TimeoutError("timed out waiting for go_event")
    with HostStageGate(stage, limit):
        enter_ns = time.monotonic_ns()
        with counter_lock:
            active_val.value += 1  # type: ignore[attr-defined]
            if active_val.value > peak_val.value:  # type: ignore[attr-defined]
                peak_val.value = active_val.value  # type: ignore[attr-defined]
        time.sleep(hold_s)
        with counter_lock:
            active_val.value -= 1  # type: ignore[attr-defined]
        exit_ns = time.monotonic_ns()
    result_q.put((enter_ns, exit_ns))


def _crashing_holder(stage: str, limit: int, acquired_q: mp.Queue, crash_event: MpEvent) -> None:
    gate = HostStageGate(stage, limit)
    slot = gate.acquire()
    acquired_q.put((os.getpid(), slot))
    crash_event.wait(timeout=15.0)
    os._exit(1)


def _recovering_waiter(stage: str, limit: int, contending_q: mp.Queue, recovered_q: mp.Queue) -> None:
    contending_q.put(os.getpid())
    with HostStageGate(stage, limit):
        recovered_q.put(time.monotonic_ns())


# ---------------------------------------------------------------------------
# Thread budget & configuration unit tests
# ---------------------------------------------------------------------------


def test_default_numba_threads_scales_with_cpu_and_honors_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXEL_NUMBA_THREADS", raising=False)

    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert default_numba_threads(16) == 8

    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert default_numba_threads(32) == 16

    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert default_numba_threads(64) == 16
    assert default_numba_threads(12) == 12

    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert default_numba_threads(1) == 1

    monkeypatch.setenv("VOXEL_NUMBA_THREADS", "12")
    assert default_numba_threads(16) == 12

    monkeypatch.setenv("VOXEL_NUMBA_THREADS", "64")
    assert default_numba_threads(16) == 16

    monkeypatch.setenv("VOXEL_NUMBA_THREADS", "0")
    assert default_numba_threads(16) == 1


def test_default_io_concurrency_scales_with_cpu_and_honors_override(monkeypatch: pytest.MonkeyPatch) -> None:
    env_var = "VOXEL_TS_CONCURRENCY"
    monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert default_io_concurrency(env_var) == 4

    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    assert default_io_concurrency(env_var) == 8

    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert default_io_concurrency(env_var) == 8

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert default_io_concurrency(env_var) == 1

    monkeypatch.setenv(env_var, "16")
    assert default_io_concurrency(env_var) == 16

    monkeypatch.setenv(env_var, "0")
    assert default_io_concurrency(env_var) == 1


def test_stage_gates_and_lock_dir_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VOXEL_LOCK_DIR", raising=False)
    default_dir = resolve_lock_dir()
    assert default_dir.parent == Path(tempfile.gettempdir())
    assert default_dir.name.startswith("voxel_ozw_locks_")

    isolated = tmp_path / "custom_locks"
    monkeypatch.setenv("VOXEL_LOCK_DIR", str(isolated))
    assert resolve_lock_dir() == isolated

    monkeypatch.delenv("VOXEL_MAX_CONCURRENT_DOWNSAMPLES", raising=False)
    monkeypatch.delenv("VOXEL_MAX_CONCURRENT_FLUSHES", raising=False)
    assert downsample_gate().limit == 1
    assert flush_gate().limit == 2

    monkeypatch.setenv("VOXEL_MAX_CONCURRENT_DOWNSAMPLES", "0")
    monkeypatch.setenv("VOXEL_MAX_CONCURRENT_FLUSHES", "0")
    unlimited = downsample_gate()
    assert unlimited.limit == 0
    assert stage_limit("VOXEL_MAX_CONCURRENT_FLUSHES", 2) == 0
    with unlimited:
        assert not isolated.exists()  # unlimited gate touches no lock files

    with pytest.raises(ValueError, match="invalid stage"):
        HostStageGate("bad/name", 1)
    with pytest.raises(ValueError, match="limit must be >= 0"):
        HostStageGate("valid", -1)

    gate = HostStageGate("unheld", 1, lock_dir=isolated)
    with pytest.raises(RuntimeError, match="not held"):
        gate.release()


def test_zarrs_chunk_concurrent_maximum_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXEL_ZARRS_SHARD_WORKERS", "6")
    expected_workers = default_io_concurrency("VOXEL_ZARRS_SHARD_WORKERS")
    assert expected_workers == 6

    captured: dict[str, Any] = {}
    fake_zarr = types.ModuleType("zarr")
    fake_zarr.config = types.SimpleNamespace(set=captured.update)  # type: ignore[attr-defined]
    fake_zarr.open_array = lambda **_kw: None  # type: ignore[attr-defined]
    fake_storage = types.ModuleType("zarr.storage")
    fake_storage.LocalStore = object  # type: ignore[attr-defined]
    fake_storage.ObjectStore = object  # type: ignore[attr-defined]
    fake_zarrs = types.ModuleType("zarrs")

    monkeypatch.setitem(sys.modules, "zarr", fake_zarr)
    monkeypatch.setitem(sys.modules, "zarr.storage", fake_storage)
    monkeypatch.setitem(sys.modules, "zarrs", fake_zarrs)

    zarrs_path = Path(__file__).resolve().parents[1] / "src" / "ome_zarr_writer" / "array" / "zarrs.py"
    spec = importlib.util.spec_from_file_location("ome_zarr_writer.array._test_zarrs_mod", zarrs_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._SHARD_WRITE_WORKERS == 6
    assert captured["codec_pipeline"]["chunk_concurrent_maximum"] == max(4, mod._SHARD_WRITE_WORKERS)


# ---------------------------------------------------------------------------
# Cross-process HostStageGate concurrency & crash-safety tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2])
def test_host_stage_gate_enforces_limit_across_spawned_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    lock_dir = tmp_path / f"locks_k{limit}"
    monkeypatch.setenv("VOXEL_LOCK_DIR", str(lock_dir))

    n_workers = 4
    hold_s = 0.08
    ctx = mp.get_context("spawn")
    ready_q = ctx.Queue()
    result_q = ctx.Queue()
    go_event = ctx.Event()
    counter_lock = ctx.Lock()
    active_val = ctx.Value("i", 0)
    peak_val = ctx.Value("i", 0)

    procs = [
        ctx.Process(
            target=_gate_worker,
            args=(f"stage_k{limit}", limit, hold_s, ready_q, go_event, counter_lock, active_val, peak_val, result_q),
        )
        for _ in range(n_workers)
    ]
    for proc in procs:
        proc.start()
    try:
        for _ in range(n_workers):
            ready_q.get(timeout=15.0)
        go_event.set()

        spans = [result_q.get(timeout=15.0) for _ in range(n_workers)]
        for proc in procs:
            proc.join(timeout=10.0)
            assert proc.exitcode == 0
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)

    assert peak_val.value == limit

    # Sweep-line check over recorded monotonic intervals to confirm global bound.
    events: list[tuple[int, int]] = []
    for enter_ns, exit_ns in spans:
        assert enter_ns < exit_ns
        events.append((enter_ns, +1))
        events.append((exit_ns, -1))
    events.sort(key=lambda item: (item[0], item[1]))  # releases (-1) ordered before simultaneous acquires (+1)
    running = 0
    max_running = 0
    for _, delta in events:
        running += delta
        max_running = max(max_running, running)
    assert max_running == limit

    if limit == 1:
        ordered = sorted(spans)
        for (_, prev_end), (next_start, _) in pairwise(ordered):
            assert prev_end <= next_start


@pytest.mark.parametrize("limit", [1, 2])
def test_host_stage_gate_releases_immediately_on_holder_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
) -> None:
    lock_dir = tmp_path / f"crash_locks_k{limit}"
    monkeypatch.setenv("VOXEL_LOCK_DIR", str(lock_dir))
    stage = f"crash_k{limit}"

    ctx = mp.get_context("spawn")
    acquired_q = ctx.Queue()
    contending_q = ctx.Queue()
    recovered_q = ctx.Queue()
    crash_event = ctx.Event()

    holders = [ctx.Process(target=_crashing_holder, args=(stage, limit, acquired_q, crash_event)) for _ in range(limit)]
    waiter = ctx.Process(target=_recovering_waiter, args=(stage, limit, contending_q, recovered_q))

    for holder in holders:
        holder.start()
    try:
        held_slots = {acquired_q.get(timeout=15.0)[1] for _ in range(limit)}
        assert held_slots == set(range(limit))

        waiter.start()
        contending_q.get(timeout=15.0)
        time.sleep(0.05)  # allow waiter to block/poll against the saturated gate
        assert recovered_q.empty()

        crash_ns = time.monotonic_ns()
        crash_event.set()
        for holder in holders:
            holder.join(timeout=10.0)
            assert holder.exitcode == 1

        recovered_ns = recovered_q.get(timeout=5.0)
        waiter.join(timeout=10.0)
        assert waiter.exitcode == 0
        assert (recovered_ns - crash_ns) < 2_000_000_000  # recovered well under 2s without hanging
    finally:
        for proc in [*holders, waiter]:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)


# ---------------------------------------------------------------------------
# End-to-end dual-OMEZarrWriter synchronized batch_z test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dual_writers_synchronized_batch_boundaries_stagger_downsamples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two OMEZarrWriter instances hitting batch_z on the exact same trigger pulse complete cleanly,
    write valid datasets + metrics.json, and never overlap their downsample windows when
    VOXEL_MAX_CONCURRENT_DOWNSAMPLES=1."""
    monkeypatch.setenv("VOXEL_LOCK_DIR", str(tmp_path / "dual_locks"))
    monkeypatch.setenv("VOXEL_MAX_CONCURRENT_DOWNSAMPLES", "1")
    monkeypatch.setenv("VOXEL_MAX_CONCURRENT_FLUSHES", "2")

    z, y, x = 128, 128, 128
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=z, y=y, x=x),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L6,
    )
    assert cfg.batch_z == 64  # two full batches per camera -> 4 synchronized batch flushes total

    writers = [OMEZarrWriter(slots=3), OMEZarrWriter(slots=3)]
    targets = [tmp_path / "cam_a", tmp_path / "cam_b"]
    bases = [1000, 5000]
    pulse_barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    for writer, target in zip(writers, targets, strict=True):
        writer.begin_stack(cfg, Local(target=target))

    def _drive_camera(writer: OMEZarrWriter, base_val: int) -> None:
        try:
            for frame_idx in range(z):
                if frame_idx in (cfg.batch_z - 1, z - 1):
                    pulse_barrier.wait(timeout=30.0)
                with writer.new_frame() as frame:
                    frame.fill(base_val + frame_idx)
            pulse_barrier.wait(timeout=30.0)
            writer.end_stack()
        except BaseException as exc:
            errors.append(exc)
            raise

    threads = [
        threading.Thread(target=_drive_camera, args=(writer, base_val))
        for writer, base_val in zip(writers, bases, strict=True)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60.0)
            assert not thread.is_alive(), "camera writer thread timed out"
        assert not errors, errors
    finally:
        for writer in writers:
            writer.close()

    downsample_spans: list[tuple[datetime, datetime, str, int]] = []
    for target, base_val in zip(targets, bases, strict=True):
        ds_root = Path(f"{target}.ome.zarr")
        assert (ds_root / "zarr.json").is_file()

        arr_l0 = TSArrayReader(ds_root / "0").read_3d(z0=0, n=z)
        assert arr_l0.shape == (z, y, x)
        for frame_idx in (0, cfg.batch_z - 1, cfg.batch_z, z - 1):
            assert int(arr_l0[frame_idx].min()) == base_val + frame_idx
            assert int(arr_l0[frame_idx].max()) == base_val + frame_idx

        arr_l6 = TSArrayReader(ds_root / "6").read_3d(z0=0, n=z // 64)
        assert arr_l6.shape == (z // 64, y // 64, x // 64)

        metrics_path = ds_root / "metrics.json"
        assert metrics_path.is_file()
        payload = json.loads(metrics_path.read_text())
        executed = [b for b in payload["batches"] if b["collected_frames"] > 0]
        assert len(executed) == 2

        for batch in executed:
            assert batch["flushed_bytes"] > 0
            p_start = datetime.fromisoformat(batch["processing"]["started"])
            p_end = datetime.fromisoformat(batch["processing"]["ended"])
            f_start = datetime.fromisoformat(batch["flushing"]["started"])
            f_end = datetime.fromisoformat(batch["flushing"]["ended"])
            assert p_start <= p_end <= f_start <= f_end
            downsample_spans.append((p_start, p_end, ds_root.name, batch["batch_idx"]))

    assert len(downsample_spans) == 4
    downsample_spans.sort(key=lambda item: (item[0], item[1]))
    for prev, curr in pairwise(downsample_spans):
        assert prev[1] <= curr[0], f"overlapping downsample windows: {prev} vs {curr}"
