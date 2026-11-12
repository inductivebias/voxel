import json
import multiprocessing as mp
import os
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event
from pathlib import Path

import psutil
import pytest
from pydantic import ValidationError

from vxl import system as system_module
from vxl._utils.files import load_yaml as real_load_yaml
from vxl.system import System

_GIB = 1024**3


@pytest.fixture(autouse=True)
def isolated_voxel_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``$HOME`` (and Windows ``$USERPROFILE``) at ``tmp_path`` so both parent and
    spawned child processes share an isolated ``~/.voxel`` registry per test."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for var in ("VOXEL_MAX_RAM_FRACTION", "VOXEL_RAM_REGISTRY"):
        monkeypatch.delenv(var, raising=False)
    System.Ram.reset()
    yield tmp_path / ".voxel"
    System.Ram.reset()


def _dual_camera_worker(
    consumer_id: str,
    ready_event: Event,
    all_ready_event: Event,
    release_event: Event,
    result_queue: mp.Queue,
) -> None:
    System.Ram.reserve(consumer_id, 1.0)
    ready_event.set()
    if not all_ready_event.wait(timeout=10.0):
        raise TimeoutError(f"{consumer_id} timed out waiting for sibling reservation")
    cap = System.Ram.max_bytes()
    share = System.Ram.reserved(consumer_id)
    result_queue.put((consumer_id, share, cap))
    if not release_event.wait(timeout=10.0):
        raise TimeoutError(f"{consumer_id} timed out waiting for release gate")
    System.Ram.release(consumer_id)


def _symmetric_commit_worker_a(
    shared_avail: Synchronized,
    a_reserved: Event,
    both_reserved: Event,
    b_checked_initial: Event,
    a_committed: Event,
    finish_gate: Event,
    result_queue: mp.Queue,
) -> None:
    System.Ram.total_bytes = classmethod(lambda _cls: 100 * _GIB)  # type: ignore[method-assign]
    System.Ram.available_bytes = classmethod(lambda _cls: int(shared_avail.value))  # type: ignore[method-assign]

    System.Ram.reserve("cam_a", 1.0)
    a_reserved.set()
    assert both_reserved.wait(timeout=10.0)
    initial_share = System.Ram.reserved("cam_a")
    assert b_checked_initial.wait(timeout=10.0)

    # Simulate prefaulting a 30 GiB ring buffer: OS available RAM drops by 30 GiB.
    shared_avail.value = 50 * _GIB
    System.Ram.record_committed("cam_a", 30 * _GIB)
    post_commit_share = System.Ram.reserved("cam_a")
    a_committed.set()

    assert finish_gate.wait(timeout=10.0)
    shared_avail.value = 80 * _GIB
    System.Ram.release("cam_a")
    result_queue.put(("cam_a", initial_share, post_commit_share))


def _symmetric_commit_worker_b(
    shared_avail: Synchronized,
    b_reserved: Event,
    both_reserved: Event,
    b_checked_initial: Event,
    a_committed: Event,
    b_checked_post_commit: Event,
    a_released: Event,
    result_queue: mp.Queue,
) -> None:
    System.Ram.total_bytes = classmethod(lambda _cls: 100 * _GIB)  # type: ignore[method-assign]
    System.Ram.available_bytes = classmethod(lambda _cls: int(shared_avail.value))  # type: ignore[method-assign]

    System.Ram.reserve("cam_b", 1.0)
    b_reserved.set()
    assert both_reserved.wait(timeout=10.0)
    initial_share = System.Ram.reserved("cam_b")
    b_checked_initial.set()

    assert a_committed.wait(timeout=10.0)
    share_after_a_prefault = System.Ram.reserved("cam_b")
    b_checked_post_commit.set()

    assert a_released.wait(timeout=10.0)
    share_after_a_release = System.Ram.reserved("cam_b")
    System.Ram.release("cam_b")
    result_queue.put(("cam_b", initial_share, share_after_a_prefault, share_after_a_release))


def _crashing_camera_worker(reserved_event: Event, crash_event: Event) -> None:
    System.Ram.reserve("crashed_cam", 1.0)
    System.Ram.record_committed("crashed_cam", 20 * _GIB)
    reserved_event.set()
    crash_event.wait(timeout=10.0)
    os._exit(1)


def _clean_exit_worker() -> None:
    System.Ram.reserve("ephemeral_cam", 1.0)


def test_max_bytes_and_reserved_cache_yaml_and_reflect_updates(
    monkeypatch: pytest.MonkeyPatch,
    isolated_voxel_home: Path,
) -> None:
    monkeypatch.setattr(System.Ram, "total_bytes", classmethod(lambda _cls: 100 * _GIB))
    monkeypatch.setattr(System.Ram, "available_bytes", classmethod(lambda _cls: 80 * _GIB))

    yaml_calls: list[Path] = []
    system_init_calls = 0
    orig_init = System.__init__

    def spy_load_yaml(path: Path):
        yaml_calls.append(path)
        return real_load_yaml(path)

    def spy_system_init(self: System, *args, **kwargs) -> None:
        nonlocal system_init_calls
        system_init_calls += 1
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(system_module, "load_yaml", spy_load_yaml)
    monkeypatch.setattr(System, "__init__", spy_system_init)

    # 1. Missing system.yaml / station.yaml -> default 0.75, zero YAML reads, zero System() inits.
    System.Ram.reserve("cam_1", 1.0)
    for _ in range(20):
        assert System.Ram.max_bytes() == int(80 * _GIB * 0.75)
        assert System.Ram.reserved("cam_1") == int(80 * _GIB * 0.75)
    assert yaml_calls == []
    assert system_init_calls == 0

    # 2. Creating system.yaml is picked up once, then cached across repeated ticks.
    isolated_voxel_home.mkdir(parents=True, exist_ok=True)
    system_yaml = isolated_voxel_home / "system.yaml"
    system_yaml.write_text("max_ram_fraction: 0.5\n", encoding="utf-8")

    for _ in range(20):
        assert System.Ram.max_bytes() == int(80 * _GIB * 0.5)
        assert System.Ram.reserved("cam_1") == int(80 * _GIB * 0.5)
    assert yaml_calls == [system_yaml]
    assert system_init_calls == 0

    # 3. Creating station.yaml overrides system.yaml and parses once.
    yaml_calls.clear()
    station_yaml = isolated_voxel_home / "station.yaml"
    station_yaml.write_text("max_ram_fraction: 0.4\n", encoding="utf-8")

    for _ in range(20):
        assert System.Ram.max_bytes() == int(80 * _GIB * 0.4)
        assert System.Ram.reserved("cam_1") == int(80 * _GIB * 0.4)
    assert yaml_calls == [station_yaml]
    assert system_init_calls == 0

    # 4. Updating station.yaml invalidates the mtime cache once, then stays cached.
    yaml_calls.clear()
    station_yaml.write_text("max_ram_fraction: 0.625\n", encoding="utf-8")

    for _ in range(20):
        assert System.Ram.max_bytes() == int(80 * _GIB * 0.625)
        assert System.Ram.reserved("cam_1") == int(80 * _GIB * 0.625)
    assert yaml_calls == [station_yaml]
    assert system_init_calls == 0

    # 5. VOXEL_MAX_RAM_FRACTION env var overrides station.yaml with zero YAML reads.
    yaml_calls.clear()
    monkeypatch.setenv("VOXEL_MAX_RAM_FRACTION", "0.25")
    for _ in range(10):
        assert System.Ram.max_bytes() == int(80 * _GIB * 0.25)
        assert System.Ram.reserved("cam_1") == int(80 * _GIB * 0.25)
    assert yaml_calls == []
    assert system_init_calls == 0

    # Removing env var restores cached station.yaml value without re-parsing disk YAML.
    monkeypatch.delenv("VOXEL_MAX_RAM_FRACTION")
    assert System.Ram.max_bytes() == int(80 * _GIB * 0.625)
    assert yaml_calls == []
    assert system_init_calls == 0


def test_steady_state_reserved_performs_zero_registry_file_reads(
    monkeypatch: pytest.MonkeyPatch,
    isolated_voxel_home: Path,
) -> None:
    monkeypatch.setattr(System.Ram, "total_bytes", classmethod(lambda _cls: 64 * _GIB))
    monkeypatch.setattr(System.Ram, "available_bytes", classmethod(lambda _cls: 48 * _GIB))

    System.Ram.reserve("cam_1", 1.0)
    reg_path = isolated_voxel_home / ".ram_consumers.json"
    assert reg_path.is_file()

    read_calls = 0
    orig_read_bytes = Path.read_bytes

    def spy_read_bytes(self: Path) -> bytes:
        nonlocal read_calls
        if self == reg_path:
            read_calls += 1
        return orig_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", spy_read_bytes)

    for _ in range(20):
        assert System.Ram.reserved("cam_1") == int(48 * _GIB * 0.75)
    assert read_calls == 0


def test_cross_process_dual_cameras_split_host_budget(isolated_voxel_home: Path) -> None:
    del isolated_voxel_home
    ctx = mp.get_context("spawn")
    ready_1 = ctx.Event()
    ready_2 = ctx.Event()
    all_ready = ctx.Event()
    release_gate = ctx.Event()
    results: mp.Queue = ctx.Queue()

    proc_1 = ctx.Process(target=_dual_camera_worker, args=("cam_1", ready_1, all_ready, release_gate, results))
    proc_2 = ctx.Process(target=_dual_camera_worker, args=("cam_2", ready_2, all_ready, release_gate, results))
    proc_1.start()
    proc_2.start()

    try:
        assert ready_1.wait(timeout=15.0)
        assert ready_2.wait(timeout=15.0)
        all_ready.set()

        out_1 = results.get(timeout=10.0)
        out_2 = results.get(timeout=10.0)
        by_id = {cid: (share, cap) for cid, share, cap in (out_1, out_2)}
        assert set(by_id) == {"cam_1", "cam_2"}

        for cid, (share, cap) in by_id.items():
            # Each camera gets ~50% of max_bytes() (37.5% of free RAM, 75% combined -- NOT 150%!).
            assert share == pytest.approx(0.5 * cap, rel=0.05), f"unexpected share for {cid}: {share} vs cap {cap}"
    finally:
        release_gate.set()
        proc_1.join(timeout=10.0)
        proc_2.join(timeout=10.0)

    assert proc_1.exitcode == 0
    assert proc_2.exitcode == 0


def test_record_committed_preserves_symmetric_shares_across_processes(isolated_voxel_home: Path) -> None:
    del isolated_voxel_home
    ctx = mp.get_context("spawn")
    shared_avail = ctx.Value("q", 80 * _GIB)
    a_reserved = ctx.Event()
    b_reserved = ctx.Event()
    both_reserved = ctx.Event()
    b_checked_initial = ctx.Event()
    a_committed = ctx.Event()
    b_checked_post_commit = ctx.Event()
    finish_gate = ctx.Event()
    a_released = ctx.Event()
    results: mp.Queue = ctx.Queue()

    proc_a = ctx.Process(
        target=_symmetric_commit_worker_a,
        args=(shared_avail, a_reserved, both_reserved, b_checked_initial, a_committed, finish_gate, results),
    )
    proc_b = ctx.Process(
        target=_symmetric_commit_worker_b,
        args=(
            shared_avail,
            b_reserved,
            both_reserved,
            b_checked_initial,
            a_committed,
            b_checked_post_commit,
            a_released,
            results,
        ),
    )
    proc_a.start()
    proc_b.start()

    try:
        assert a_reserved.wait(timeout=15.0)
        assert b_reserved.wait(timeout=15.0)
        both_reserved.set()

        assert b_checked_post_commit.wait(timeout=10.0)
        finish_gate.set()
        proc_a.join(timeout=10.0)
        assert proc_a.exitcode == 0
        a_released.set()
        proc_b.join(timeout=10.0)
        assert proc_b.exitcode == 0
    finally:
        finish_gate.set()
        a_released.set()
        if proc_a.is_alive():
            proc_a.terminate()
            proc_a.join(timeout=5.0)
        if proc_b.is_alive():
            proc_b.terminate()
            proc_b.join(timeout=5.0)

    collected = {item[0]: item[1:] for item in (results.get(timeout=5.0), results.get(timeout=5.0))}
    expected_half = int(80 * _GIB * 0.75 / 2)  # 30 GiB each
    expected_full = int(80 * _GIB * 0.75)  # 60 GiB once cam_a releases

    assert collected["cam_a"] == (expected_half, expected_half)
    assert collected["cam_b"] == (expected_half, expected_half, expected_full)


def test_dead_pid_pruned_after_crash_reclaims_full_budget(
    monkeypatch: pytest.MonkeyPatch,
    isolated_voxel_home: Path,
) -> None:
    monkeypatch.setattr(System.Ram, "total_bytes", classmethod(lambda _cls: 100 * _GIB))
    monkeypatch.setattr(System.Ram, "available_bytes", classmethod(lambda _cls: 80 * _GIB))

    full_budget = int(80 * _GIB * 0.75)
    assert System.Ram.reserve("survivor", 1.0) == full_budget

    ctx = mp.get_context("spawn")
    child_reserved = ctx.Event()
    crash_now = ctx.Event()
    child = ctx.Process(target=_crashing_camera_worker, args=(child_reserved, crash_now))
    child.start()

    assert child_reserved.wait(timeout=15.0)
    crashed_pid = child.pid
    assert crashed_pid is not None

    # While child is alive, survivor's share reflects both weights (plus child's committed 20 GiB capped by 100 GiB).
    assert System.Ram.reserved("survivor") == int(100 * _GIB * 0.75 / 2)

    # Terminate child abruptly via os._exit(1) without calling release().
    crash_now.set()
    child.join(timeout=10.0)
    assert child.exitcode == 1

    # Survivor immediately reclaims 100% of the un-committed host budget, and crashed_cam is gone.
    assert System.Ram.reserved("survivor") == full_budget
    with pytest.raises(KeyError):
        System.Ram.reserved("crashed_cam")

    # Next registry mutation prunes the dead PID entry from disk JSON as well.
    System.Ram.record_committed("survivor", 0)
    reg_data = json.loads((isolated_voxel_home / ".ram_consumers.json").read_text(encoding="utf-8"))
    assert str(crashed_pid) not in reg_data
    assert str(os.getpid()) in reg_data


def test_atexit_cleans_up_exiting_process_entry(isolated_voxel_home: Path) -> None:
    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_clean_exit_worker)
    child.start()
    child.join(timeout=15.0)
    assert child.exitcode == 0

    reg_path = isolated_voxel_home / ".ram_consumers.json"
    if reg_path.is_file():
        reg_data = json.loads(reg_path.read_text(encoding="utf-8"))
        assert str(child.pid) not in reg_data


def test_recycled_pid_with_mismatched_create_time_is_ignored_and_pruned(
    monkeypatch: pytest.MonkeyPatch,
    isolated_voxel_home: Path,
) -> None:
    monkeypatch.setattr(System.Ram, "total_bytes", classmethod(lambda _cls: 64 * _GIB))
    monkeypatch.setattr(System.Ram, "available_bytes", classmethod(lambda _cls: 40 * _GIB))

    isolated_voxel_home.mkdir(parents=True, exist_ok=True)
    reg_path = isolated_voxel_home / ".ram_consumers.json"

    # Pick a live PID distinct from os.getpid() and forge a stale create_time far outside the 0.1s tolerance.
    other_pid = os.getppid() if os.getppid() > 0 and os.getppid() != os.getpid() else os.getpid()
    real_ct = psutil.Process(other_pid).create_time()
    reg_path.write_text(
        json.dumps(
            {
                str(other_pid): {
                    "pid": other_pid,
                    "create_time": real_ct - 3600.0,
                    "consumers": {"stale_cam": {"weight": 3.0, "committed": 10 * _GIB}},
                }
            }
        ),
        encoding="utf-8",
    )

    expected = int(40 * _GIB * 0.75)
    assert System.Ram.reserve("live_cam", 1.0) == expected
    saved = json.loads(reg_path.read_text(encoding="utf-8"))
    assert str(other_pid) not in saved


def test_validation_and_fallback_when_registry_dir_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(System.Ram, "total_bytes", classmethod(lambda _cls: 64 * _GIB))
    monkeypatch.setattr(System.Ram, "available_bytes", classmethod(lambda _cls: 32 * _GIB))

    with pytest.raises(ValueError, match="weight must be > 0"):
        System.Ram.reserve("bad", 0.0)
    with pytest.raises(ValueError, match="weight must be > 0"):
        System.Ram.reserve("bad", -1.0)
    with pytest.raises(KeyError):
        System.Ram.reserved("unregistered")

    # Safe no-ops for unregistered consumers.
    System.Ram.release("unregistered")
    System.Ram.record_committed("unregistered", 10 * _GIB)

    # Invalid VOXEL_MAX_RAM_FRACTION still raises Pydantic ValidationError.
    monkeypatch.setenv("VOXEL_MAX_RAM_FRACTION", "1.5")
    with pytest.raises(ValidationError):
        System.Ram.max_bytes()
    monkeypatch.delenv("VOXEL_MAX_RAM_FRACTION")

    # Point _voxel_home() underneath a regular file so mkdir(parents=True) fails with OSError;
    # System.Ram must fall back cleanly to process-local mediation without raising.
    blocker_file = tmp_path / "not_a_dir"
    blocker_file.write_text("block", encoding="utf-8")
    monkeypatch.setattr(system_module, "_voxel_home", lambda: blocker_file / ".voxel")

    share_1 = System.Ram.reserve("local_a", 1.0)
    share_2 = System.Ram.reserve("local_b", 3.0)
    cap = int(32 * _GIB * 0.75)
    assert share_1 == cap
    assert share_2 == int(cap * 3.0 / 4.0)
    assert System.Ram.reserved("local_a") == int(cap * 1.0 / 4.0)
    assert System.Ram._consumers == {"local_a": 1.0, "local_b": 3.0}
    System.Ram.release("local_a")
    System.Ram.release("local_b")
    assert System.Ram._consumers == {}
