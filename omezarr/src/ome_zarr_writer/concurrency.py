"""Cross-process host stage gates and CPU-aware thread budgets for OME-Zarr acquisition.

With dual high-rate cameras sharing a hardware trigger train, multiple ``BatchSlot`` worker processes
across independent ``OMEZarrWriter`` instances hit ``batch_z`` on the exact same trigger pulse. Left
uncoordinated, every ready worker simultaneously launches a multi-threaded Numba pyramid downsample and
a multi-threaded TensorStore/zarrs shard flush, oversubscribing DRAM bandwidth and NVMe/S3 queues.

This module provides two lightweight, crash-safe controls:

1. :class:`HostStageGate` — a cross-process, host-local counting semaphore backed by OS advisory file
   locks (``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows) in a per-user directory under the
   system temp directory (overridable via ``VOXEL_LOCK_DIR``). Because the OS kernel owns the locks on
   open file descriptions, a killed or crashed worker releases its slot immediately with no stale state.
   - ``VOXEL_MAX_CONCURRENT_DOWNSAMPLES`` (default ``1``, ``0`` = unlimited): serializes DRAM-bandwidth-
     bound 3D Gaussian pyramid generation across cameras/slots, naturally staggering their downstream
     shard flushes so one camera writes to storage while the other builds its pyramid in DRAM.
   - ``VOXEL_MAX_CONCURRENT_FLUSHES`` (default ``2``, ``0`` = unlimited): bounds concurrent multi-level
     shard flushes across the host.

2. CPU-aware per-worker thread budgets (:func:`default_numba_threads`, :func:`default_io_concurrency`)
   that scale down on modest core-count hosts while honoring explicit environment overrides.
"""

import errno
import getpass
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Self

DEFAULT_MAX_CONCURRENT_DOWNSAMPLES = 1
DEFAULT_MAX_CONCURRENT_FLUSHES = 2


def _env_int(env_var: str) -> int | None:
    raw = os.environ.get(env_var)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def resolve_lock_dir() -> Path:
    """Return the host-local directory used for cross-process stage lock files.

    Honors ``VOXEL_LOCK_DIR`` when set (e.g. for hermetic tests); otherwise uses a per-user directory
    under :func:`tempfile.gettempdir` so distinct OS users on a shared workstation never collide on
    file permissions.
    """
    override = os.environ.get("VOXEL_LOCK_DIR")
    if override and override.strip():
        return Path(override)
    if hasattr(os, "getuid"):
        user_tag = str(os.getuid())
    else:
        try:
            raw_user = getpass.getuser()
        except Exception:
            raw_user = os.environ.get("USERNAME") or os.environ.get("USER") or "default"
        user_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_user) or "default"
    return Path(tempfile.gettempdir()) / f"voxel_ozw_locks_{user_tag}"


def stage_limit(env_var: str, default: int) -> int:
    """Read a non-negative concurrency limit from ``env_var`` (``0`` = unlimited), or return ``default``."""
    configured = _env_int(env_var)
    if configured is None:
        return max(0, default)
    return max(0, configured)


def default_numba_threads(pool_max: int) -> int:
    """Compute per-worker Numba thread count, honoring ``VOXEL_NUMBA_THREADS`` within ``1..pool_max``."""
    upper = max(1, pool_max)
    configured = _env_int("VOXEL_NUMBA_THREADS")
    if configured is not None:
        return max(1, min(configured, upper))
    return max(1, min(16, (os.cpu_count() or 1) // 2, upper))


def default_io_concurrency(env_var: str) -> int:
    """Compute per-worker I/O concurrency, honoring ``env_var`` (>= 1) or scaling by CPU count."""
    configured = _env_int(env_var)
    if configured is not None:
        return max(1, configured)
    return max(1, min(8, (os.cpu_count() or 1) // 4))


def _poll_sleep_seconds() -> float:
    """Short 1-3 ms jittered sleep when polling a multi-slot gate under contention."""
    jitter_us = (os.getpid() ^ time.perf_counter_ns()) % 2000
    return 0.001 + jitter_us * 1e-6


if sys.platform == "win32":
    import msvcrt

    def _try_lock_fd(fd: int) -> bool:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _lock_fd_blocking(fd: int) -> None:
        while not _try_lock_fd(fd):
            time.sleep(_poll_sleep_seconds())

    def _unlock_fd(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock_fd(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return False
            raise
        return True

    def _lock_fd_blocking(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class HostStageGate:
    """Crash-safe cross-process counting gate of capacity ``limit`` backed by OS file locks.

    When ``limit == 0``, the gate is disabled (unlimited concurrency) and acts as a zero-overhead no-op.
    When ``limit >= 1``, acquiring the gate claims exclusive ownership of one lock file in
    ``0 .. limit - 1`` starting from ``os.getpid() % limit``:
    - Uncontended: acquires on the first non-blocking syscall with zero sleep.
    - ``limit == 1`` under contention: blocks directly in the OS kernel (``flock(LOCK_EX)`` on POSIX) for
      immediate wakeup when the holder releases or terminates.
    - ``limit > 1`` under contention: cycles non-blocking lock attempts across all ``limit`` slots with a
      1-3 ms jittered backoff so whichever slot frees first is picked up promptly.
    """

    def __init__(self, stage: str, limit: int, *, lock_dir: Path | None = None) -> None:
        if not stage or "/" in stage or "\\" in stage or "\0" in stage:
            raise ValueError(f"invalid stage name: {stage!r}")
        if limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        self.stage = stage
        self.limit = limit
        self.lock_dir = Path(lock_dir) if lock_dir is not None else resolve_lock_dir()
        self._local = threading.local()
        self._guard = threading.Lock()
        self._all_held: list[int] = []

    def _slot_path(self, slot: int) -> Path:
        return self.lock_dir / f"{self.stage}.{slot}.lock"

    def _push_held(self, fd: int) -> None:
        stack: list[int] | None = getattr(self._local, "fds", None)
        if stack is None:
            stack = []
            self._local.fds = stack
        stack.append(fd)
        with self._guard:
            self._all_held.append(fd)

    def _pop_held(self) -> int:
        stack: list[int] | None = getattr(self._local, "fds", None)
        with self._guard:
            if stack:
                fd = stack.pop()
                self._all_held.remove(fd)
                return fd
            if self._all_held:
                return self._all_held.pop()
        raise RuntimeError(f"HostStageGate({self.stage!r}) release() called when not held")

    def acquire(self) -> int | None:
        """Acquire one slot in ``0 .. limit - 1`` (or ``None`` when ``limit == 0``)."""
        k = self.limit
        if k == 0:
            return None
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        start = os.getpid() % k
        fds: dict[int, int] = {}
        try:
            for offset in range(k):
                slot = (start + offset) % k
                fd = os.open(self._slot_path(slot), os.O_CREAT | os.O_RDWR, 0o666)
                fds[slot] = fd
                if _try_lock_fd(fd):
                    fds.pop(slot)
                    self._push_held(fd)
                    return slot

            if k == 1:
                slot = start
                _lock_fd_blocking(fds[slot])
                fd = fds.pop(slot)
                self._push_held(fd)
                return slot

            while True:
                time.sleep(_poll_sleep_seconds())
                for offset in range(k):
                    slot = (start + offset) % k
                    fd = fds[slot]
                    if _try_lock_fd(fd):
                        fds.pop(slot)
                        self._push_held(fd)
                        return slot
        finally:
            for unused_fd in fds.values():
                with suppress(OSError):
                    os.close(unused_fd)

    def release(self) -> None:
        """Release the most recently acquired slot."""
        if self.limit == 0:
            return
        fd = self._pop_held()
        try:
            _unlock_fd(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def downsample_gate(*, lock_dir: Path | None = None) -> HostStageGate:
    """Host-wide gate for pyramid downsampling (``VOXEL_MAX_CONCURRENT_DOWNSAMPLES``, default 1)."""
    limit = stage_limit("VOXEL_MAX_CONCURRENT_DOWNSAMPLES", DEFAULT_MAX_CONCURRENT_DOWNSAMPLES)
    return HostStageGate("downsample", limit, lock_dir=lock_dir)


def flush_gate(*, lock_dir: Path | None = None) -> HostStageGate:
    """Host-wide gate for multi-level shard flushes (``VOXEL_MAX_CONCURRENT_FLUSHES``, default 2)."""
    limit = stage_limit("VOXEL_MAX_CONCURRENT_FLUSHES", DEFAULT_MAX_CONCURRENT_FLUSHES)
    return HostStageGate("flush", limit, lock_dir=lock_dir)
