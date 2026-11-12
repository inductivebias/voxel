"""Machine-local facts and configuration for the box this code runs on.

``System`` exposes what is true of *this* machine — CPU count, platform, hostname,
the RAM budget (:class:`System.Ram`), and the local storage roots (``store``/``scratch``).
Machine-specific knobs are sourced from ``VOXEL_*`` env vars and one machine file. A
control station's ``station.yaml`` is a superset of ``system.yaml`` and takes its place
when present; nodes and other non-control processes can use ``system.yaml`` or defaults.
"""

import atexit
import contextlib
import json
import logging
import os
import socket
import stat
import sys
import threading
import time
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal, Self
from uuid import UUID, uuid4

import psutil
from cloudpathlib import S3Path
from dotenv import load_dotenv
from ome_zarr_writer import DirectS3, Local, StagedS3, Storage
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from vxl_records import (
    DatasetLocation,
    LocalLocation,
    LocationRole,
    LocationStatus,
    ObjectLocation,
    StorageSpec,
)
from vxlib.s3 import S3Store

from vxl._utils.files import load_yaml, save_yaml

log = logging.getLogger(__name__)

_StatKey = tuple[int, int, int]
_RemoteConsumers = dict[str, tuple[float, int]]
_RemoteRegistryMap = dict[int, tuple[float, _RemoteConsumers]]

_PID_CREATE_TIME_TOLERANCE_S = 0.1
_LIVENESS_RECHECK_INTERVAL_S = 2.0
_YAML_CACHE: dict[Path, tuple[_StatKey, dict[str, Any]]] = {}
_YAML_CACHE_LOCK = threading.Lock()


def _voxel_home() -> Path:
    """The fixed config/state home, ``~/.voxel`` -- holds machine config and ``instruments/``.
    Relocate it by setting ``$HOME``; move the data roots independently with
    ``VOXEL_STORE`` / ``VOXEL_SCRATCH``."""
    return Path.home() / ".voxel"


def _stat_key(path: Path) -> _StatKey | None:
    """Return ``(st_mtime_ns, st_size, st_ino)`` for a regular file, or ``None`` if absent."""
    try:
        st = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _load_yaml_cached(path: Path, stat_key: _StatKey | None = None) -> dict[str, Any]:
    """Read and cache a machine YAML dict keyed on file stat metadata."""
    key = _stat_key(path) if stat_key is None else stat_key
    if key is None:
        with _YAML_CACHE_LOCK:
            _YAML_CACHE.pop(path, None)
        return {}
    with _YAML_CACHE_LOCK:
        cached = _YAML_CACHE.get(path)
        if cached is not None and cached[0] == key:
            return dict(cached[1])
    loaded = load_yaml(path)
    data: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
    with _YAML_CACHE_LOCK:
        _YAML_CACHE[path] = (key, data)
    return dict(data)


def _yaml_source(settings_cls: type[BaseSettings], path: Path) -> InitSettingsSource:
    """Machine YAML as a settings source, read like every other Voxel YAML file; a missing file adds nothing."""
    return InitSettingsSource(settings_cls, _load_yaml_cached(path))


if sys.platform == "win32":
    import msvcrt

    def _acquire_lock_fd(fd: int) -> None:
        deadline = time.monotonic() + 10.0
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.005)

    def _release_lock_fd(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire_lock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _release_lock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Cross-platform exclusive file lock around host RAM registry mutations."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        _acquire_lock_fd(fd)
        try:
            yield
        finally:
            _release_lock_fd(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write ``data`` to a sibling temp file and atomically replace ``path``."""
    payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        if sys.platform == "win32":
            for attempt in range(5):
                try:
                    tmp_path.replace(path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.005)
        else:
            tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _is_pid_alive(pid: int, expected_create_time: float) -> bool:
    """Verify ``pid`` is alive, non-zombie, and matches ``expected_create_time`` within 0.1s."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        proc = psutil.Process(pid)
        if abs(proc.create_time() - expected_create_time) > _PID_CREATE_TIME_TOLERANCE_S:
            return False
        return proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, OSError, ValueError):
        return False


def _parse_live_remote_entries(
    raw: object,
    own_pid: int,
) -> tuple[dict[str, dict[str, Any]], _RemoteRegistryMap]:
    """Validate raw registry JSON and keep only live remote PIDs."""
    cleaned_json: dict[str, dict[str, Any]] = {}
    remote_map: _RemoteRegistryMap = {}
    if not isinstance(raw, dict):
        return cleaned_json, remote_map

    for raw_entry in raw.values():
        if not isinstance(raw_entry, dict):
            continue
        raw_pid = raw_entry.get("pid")
        raw_ct = raw_entry.get("create_time")
        raw_consumers = raw_entry.get("consumers")
        if (
            isinstance(raw_pid, bool)
            or not isinstance(raw_pid, int)
            or isinstance(raw_ct, bool)
            or not isinstance(raw_ct, int | float)
            or not isinstance(raw_consumers, dict)
        ):
            continue
        if raw_pid == own_pid:
            continue

        parsed_json_consumers: dict[str, dict[str, float | int]] = {}
        parsed_mem_consumers: _RemoteConsumers = {}
        for cid, spec in raw_consumers.items():
            if not isinstance(cid, str) or not isinstance(spec, dict):
                continue
            raw_w = spec.get("weight")
            raw_c = spec.get("committed", 0)
            if isinstance(raw_w, bool) or not isinstance(raw_w, int | float):
                continue
            if isinstance(raw_c, bool) or not isinstance(raw_c, int | float):
                continue
            weight = float(raw_w)
            if weight <= 0.0:
                continue
            committed = max(0, int(raw_c))
            parsed_json_consumers[cid] = {"weight": weight, "committed": committed}
            parsed_mem_consumers[cid] = (weight, committed)

        if not parsed_mem_consumers:
            continue
        create_time = float(raw_ct)
        if not _is_pid_alive(raw_pid, create_time):
            continue

        cleaned_json[str(raw_pid)] = {
            "pid": raw_pid,
            "create_time": create_time,
            "consumers": parsed_json_consumers,
        }
        remote_map[raw_pid] = (create_time, parsed_mem_consumers)

    return cleaned_json, remote_map


def load_voxel_env() -> None:
    """Load ``~/.voxel/.env`` into the process environment if present (a no-op when absent;
    existing vars win). A per-machine escape hatch for ambient settings the file layer can't
    reach: the ``VOXEL_*`` knobs *and* the AWS chain (``AWS_*``, ``S3_ENDPOINT_URL``) that
    s5cmd / boto3 / TensorStore read. Call once at startup, before constructing :class:`System`
    or any S3 client."""
    load_dotenv(_voxel_home() / ".env", override=False)


class Remote(BaseModel):
    """A configured object store in the registry: the :class:`~vxlib.S3Store` ``connection`` plus the
    ``roots`` (display label -> write root) an operator may select. A root is a bucket, optionally
    narrowed by a key prefix (``bucket`` or ``bucket/prefix``). Holds no secrets -- credentials come
    from the AWS chain.

    The connection is held, not inherited: ``roots`` is operator-facing catalog data, while a
    :class:`~vxlib.S3Store` is a frozen *value* that writers hash to key their S3 client per
    connection. Subclassing would make a ``Remote`` an unhashable ``S3Store`` -- accepted by every
    ``store:`` field, then failing inside a cache lookup layers away. Pass ``.connection``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    connection: S3Store = S3Store()
    roots: dict[str, str] = Field(default_factory=dict, description="display label -> bucket or bucket/prefix")


def remote_store_fingerprint(store: S3Store) -> str:
    """Return a non-secret identity for matching one object store across machines.

    Roots and credential strategies are deliberately excluded: machines may expose different selectable roots and
    use different credentials while addressing the same S3-compatible service.
    """
    identity = f"{store.endpoint or ''}\0{store.region or ''}".encode()
    return f"s3:{sha256(identity).hexdigest()}"


class System(BaseSettings):
    """This machine: ``VOXEL_*`` env knobs, machine YAML config, and introspected facts.

    Construct to read and validate the environment plus ``~/.voxel/station.yaml`` when it exists,
    otherwise ``~/.voxel/system.yaml`` (init > env > yaml). The selected file is not merged with
    the fallback. Exposes the local storage roots (``store``/``scratch``, overridable via
    ``VOXEL_STORE``/``VOXEL_SCRATCH``), the object-store registry (:attr:`remotes`), the fixed config
    home (:attr:`dir`), the pure machine
    facts (``cpu_count``/``platform``/``hostname``), and the shared RAM budget (:class:`Ram`). Paths
    are resolved only; whoever writes creates them.
    """

    model_config = SettingsConfigDict(env_prefix="VOXEL_", extra="ignore")

    store: Path = Field(default_factory=lambda: _voxel_home() / "store", description="VOXEL_STORE")
    scratch: Path = Field(default_factory=lambda: _voxel_home() / "scratch", description="VOXEL_SCRATCH")
    max_ram_fraction: float = Field(
        default=0.75,
        gt=0.0,
        le=1.0,
        description=(
            "VOXEL_MAX_RAM_FRACTION — fraction of *available* (not total) RAM that registered consumers "
            "divide between them by weight. Because the denominator excludes the OS and other processes, "
            "this is a share of what is free at the moment it is read, not a fixed provision."
        ),
    )
    remotes: dict[str, Remote] = Field(
        default_factory=dict, description="object-store name -> connection (from the selected machine config)"
    )

    @classmethod
    def config_path(cls) -> Path:
        """Return the selected machine config: station first, then system.

        Selection is whole-file precedence, not a merge. This keeps exactly one authority for
        structured machine settings while allowing nodes to run without a station identity.
        """
        station = _voxel_home() / "station.yaml"
        return station if station.is_file() else _voxel_home() / "system.yaml"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Layer the selected machine YAML under env (priority: init > env > yaml)."""
        del dotenv_settings, file_secret_settings
        return (init_settings, env_settings, _yaml_source(settings_cls, cls.config_path()))

    @property
    def dir(self) -> Path:
        """The fixed config/state home (``~/.voxel``); see :func:`_voxel_home`."""
        return _voxel_home()

    @field_validator("store", "scratch", mode="after")
    @classmethod
    def _expand(cls, p: Path) -> Path:
        return p.expanduser()

    @staticmethod
    def cpu_count() -> int:
        return psutil.cpu_count(logical=True) or 1

    @staticmethod
    def platform() -> str:
        return sys.platform

    @staticmethod
    def hostname() -> str:
        return socket.gethostname()

    @staticmethod
    def local_ip() -> str:
        """Return the best-effort outbound IPv4 address, falling back to loopback."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
                connection.settimeout(0.1)
                connection.connect(("8.8.8.8", 80))
                return connection.getsockname()[0]
        except OSError:
            return "127.0.0.1"

    def resolve_storage(self, spec: StorageSpec, subpath: PurePosixPath | None = None) -> Storage:
        """Resolve portable storage against this machine's configuration."""
        relpath = spec.path / (subpath or PurePosixPath())
        if spec.remote is None:
            return Local(target=self.store / relpath)

        if spec.remote.store not in self.remotes:
            raise KeyError(f"unknown remote store '{spec.remote.store}'; configured: {sorted(self.remotes)}")
        store = self.remotes[spec.remote.store].connection
        target = S3Path(f"s3://{spec.remote.root}") / relpath.as_posix()
        if spec.remote.stage:
            return StagedS3(scratch=self.scratch / relpath, target=target, store=store)
        return DirectS3(target=target, store=store)

    def describe_dataset_location(self, spec: StorageSpec, target: Path | S3Path) -> DatasetLocation:
        """Describe a concrete writer target as a durable acquisition-record location."""
        if spec.remote is None:
            if not isinstance(target, Path):
                raise TypeError(f"local storage resolved to a non-local target: {target}")
            return LocalLocation(
                role=LocationRole.DESTINATION,
                status=LocationStatus.WRITING,
                host=self.hostname(),
                path=str(target),
            )

        if not isinstance(target, S3Path):
            raise TypeError(f"object storage resolved to a non-object target: {target}")
        return ObjectLocation(
            role=LocationRole.DESTINATION,
            status=LocationStatus.WRITING,
            host=self.hostname(),
            store=spec.remote.store,
            bucket=target.bucket,
            key=target.key,
        )

    class Ram:
        """Weighted RAM-budget mediation for consumers sharing this machine's memory.

        Holds a policy fraction of available RAM and divides it among registered
        consumers across all live Voxel processes sharing ``_voxel_home()`` in
        proportion to their weights. Consumers register via ``reserve()`` /
        ``release()``, optionally report prefaulted allocations via
        ``record_committed()``, and read their allocation via ``reserved()``.
        """

        _consumers: ClassVar[dict[str, float]] = {}
        _committed: ClassVar[dict[str, int]] = {}
        _lock: ClassVar[threading.RLock] = threading.RLock()
        _active_registry_path: ClassVar[Path | None] = None
        _touched_registries: ClassVar[set[Path]] = set()
        _atexit_registered_pid: ClassVar[int | None] = None
        _own_create_time: ClassVar[tuple[int, float] | None] = None
        _fraction_cache: ClassVar[tuple[Path, _StatKey | None, float] | None] = None
        _registry_cache: ClassVar[tuple[Path, _StatKey | None, _RemoteRegistryMap, float] | None] = None

        @classmethod
        def _max_ram_fraction(cls) -> float:
            """Return ``max_ram_fraction`` without constructing ``System()`` or re-parsing unchanged YAML."""
            raw_env = os.environ.get("VOXEL_MAX_RAM_FRACTION")
            if raw_env is not None:
                try:
                    fraction = float(raw_env)
                except ValueError:
                    return System().max_ram_fraction
                if 0.0 < fraction <= 1.0:
                    return fraction
                return System().max_ram_fraction

            config_path = System.config_path()
            stat_key = _stat_key(config_path)
            with cls._lock:
                cached = cls._fraction_cache
                if cached is not None and cached[0] == config_path and cached[1] == stat_key:
                    return cached[2]

                if stat_key is None:
                    cls._fraction_cache = (config_path, None, 0.75)
                    return 0.75

                data = _load_yaml_cached(config_path, stat_key)
                if "max_ram_fraction" not in data:
                    fraction = 0.75
                else:
                    raw_val = data["max_ram_fraction"]
                    if isinstance(raw_val, bool) or not isinstance(raw_val, int | float | str):
                        return System().max_ram_fraction
                    try:
                        fraction = float(raw_val)
                    except ValueError:
                        return System().max_ram_fraction
                    if not (0.0 < fraction <= 1.0):
                        return System().max_ram_fraction

                cls._fraction_cache = (config_path, stat_key, fraction)
                return fraction

        @classmethod
        def _registry_paths(cls) -> tuple[Path, Path]:
            override = os.environ.get("VOXEL_RAM_REGISTRY")
            if override:
                reg_path = Path(override).expanduser()
                lock_path = reg_path.with_suffix(".lock") if reg_path.suffix else Path(f"{reg_path}.lock")
                return reg_path, lock_path
            home = _voxel_home()
            return home / ".ram_consumers.json", home / ".ram_consumers.lock"

        @classmethod
        def _sync_scope(cls, reg_path: Path) -> None:
            if cls._active_registry_path is not None and cls._active_registry_path != reg_path:
                cls._consumers.clear()
                cls._committed.clear()
            cls._active_registry_path = reg_path
            if cls._committed:
                for cid in list(cls._committed):
                    if cid not in cls._consumers:
                        del cls._committed[cid]

        @classmethod
        def _get_own_create_time(cls, pid: int) -> float:
            cached = cls._own_create_time
            if cached is not None and cached[0] == pid:
                return cached[1]
            try:
                ct = psutil.Process(pid).create_time()
            except (psutil.Error, OSError):
                ct = 0.0
            cls._own_create_time = (pid, ct)
            return ct

        @classmethod
        def _ensure_atexit_registered(cls, pid: int) -> None:
            if cls._atexit_registered_pid != pid:
                atexit.register(cls._cleanup_current_process)
                cls._atexit_registered_pid = pid

        @classmethod
        def _cleanup_current_process(cls) -> None:
            pid = os.getpid()
            pid_key = str(pid)
            with cls._lock:
                paths = set(cls._touched_registries)
                cls._touched_registries.clear()
                cls._consumers.clear()
                cls._committed.clear()
                cls._registry_cache = None
            for reg_path in paths:
                if not reg_path.exists():
                    continue
                lock_path = reg_path.with_suffix(".lock") if reg_path.suffix else Path(f"{reg_path}.lock")
                with contextlib.suppress(OSError, json.JSONDecodeError), _file_lock(lock_path):
                    raw = json.loads(reg_path.read_bytes()) if reg_path.is_file() else {}
                    if isinstance(raw, dict) and pid_key in raw:
                        cleaned, _ = _parse_live_remote_entries(raw, pid)
                        _atomic_write_json(reg_path, cleaned)

        @classmethod
        def _sync_own_to_registry(cls, reg_path: Path, lock_path: Path) -> None:
            pid = os.getpid()
            cls._ensure_atexit_registered(pid)
            if not cls._consumers and not reg_path.exists():
                cls._registry_cache = (reg_path, None, {}, time.monotonic())
                return
            try:
                with _file_lock(lock_path):
                    raw: object = {}
                    if reg_path.is_file():
                        with contextlib.suppress(OSError, json.JSONDecodeError):
                            raw = json.loads(reg_path.read_bytes())
                    cleaned_json, remote_map = _parse_live_remote_entries(raw, pid)
                    if cls._consumers:
                        cleaned_json[str(pid)] = {
                            "pid": pid,
                            "create_time": cls._get_own_create_time(pid),
                            "consumers": {
                                cid: {
                                    "weight": float(weight),
                                    "committed": max(0, int(cls._committed.get(cid, 0))),
                                }
                                for cid, weight in cls._consumers.items()
                            },
                        }
                    _atomic_write_json(reg_path, cleaned_json)
                    cls._touched_registries.add(reg_path)
                    cls._registry_cache = (reg_path, _stat_key(reg_path), remote_map, time.monotonic())
            except OSError:
                log.debug("RAM registry unavailable at %s; using process-local mediation", reg_path, exc_info=True)

        @classmethod
        def _read_remote_consumers(cls, reg_path: Path) -> _RemoteRegistryMap:
            cls._sync_scope(reg_path)
            stat_key = _stat_key(reg_path)
            now = time.monotonic()
            if stat_key is None:
                cls._registry_cache = (reg_path, None, {}, now)
                return {}

            cached = cls._registry_cache
            if cached is not None and cached[0] == reg_path and cached[1] == stat_key:
                remote_map = cached[2]
                last_check = cached[3]
                if remote_map:
                    if now - last_check >= _LIVENESS_RECHECK_INTERVAL_S:
                        remote_map = {
                            rpid: (ct, cons) for rpid, (ct, cons) in remote_map.items() if _is_pid_alive(rpid, ct)
                        }
                        cls._registry_cache = (reg_path, stat_key, remote_map, now)
                    else:
                        dead = [rpid for rpid in remote_map if not psutil.pid_exists(rpid)]
                        if dead:
                            remote_map = {rpid: val for rpid, val in remote_map.items() if rpid not in dead}
                            cls._registry_cache = (reg_path, stat_key, remote_map, last_check)
                return remote_map

            try:
                raw = json.loads(reg_path.read_bytes())
            except (OSError, json.JSONDecodeError):
                return {}
            _, remote_map = _parse_live_remote_entries(raw, os.getpid())
            cls._registry_cache = (reg_path, stat_key, remote_map, now)
            return remote_map

        @classmethod
        def total_bytes(cls) -> int:
            return psutil.virtual_memory().total

        @classmethod
        def available_bytes(cls) -> int:
            return psutil.virtual_memory().available

        @classmethod
        def max_bytes(cls) -> int:
            """Live policy cap: available system RAM * VOXEL_MAX_RAM_FRACTION."""
            return int(cls.available_bytes() * cls._max_ram_fraction())

        @classmethod
        def reserve(cls, consumer_id: str, weight: float = 1.0) -> int:
            """Register ``consumer_id`` with ``weight``; return its share in bytes."""
            if weight <= 0:
                raise ValueError(f"weight must be > 0, got {weight}")
            with cls._lock:
                reg_path, lock_path = cls._registry_paths()
                cls._sync_scope(reg_path)
                cls._consumers[consumer_id] = float(weight)
                cls._committed.setdefault(consumer_id, 0)
                cls._sync_own_to_registry(reg_path, lock_path)
            return cls.reserved(consumer_id)

        @classmethod
        def record_committed(cls, consumer_id: str, committed_bytes: int) -> None:
            """Record bytes already allocated/prefaulted by a registered consumer so sibling consumers
            sizing their rings afterward divide the same pre-allocation available pool."""
            with cls._lock:
                reg_path, lock_path = cls._registry_paths()
                cls._sync_scope(reg_path)
                if consumer_id not in cls._consumers:
                    return
                cls._committed[consumer_id] = max(0, int(committed_bytes))
                cls._sync_own_to_registry(reg_path, lock_path)

        @classmethod
        def release(cls, consumer_id: str) -> None:
            """Deregister ``consumer_id`` (no-op if not registered)."""
            with cls._lock:
                reg_path, lock_path = cls._registry_paths()
                cls._sync_scope(reg_path)
                removed_weight = cls._consumers.pop(consumer_id, None)
                cls._committed.pop(consumer_id, None)
                if removed_weight is not None:
                    cls._sync_own_to_registry(reg_path, lock_path)

        @classmethod
        def reset(cls) -> None:
            """Clear all RAM consumer reservations in this process and the active host registry."""
            with cls._lock:
                reg_path, lock_path = cls._registry_paths()
                cls._active_registry_path = reg_path
                cls._consumers.clear()
                cls._committed.clear()
                cls._registry_cache = None
                if not reg_path.exists() and not lock_path.exists():
                    return
                try:
                    with _file_lock(lock_path):
                        _atomic_write_json(reg_path, {})
                        cls._registry_cache = (reg_path, _stat_key(reg_path), {}, time.monotonic())
                except OSError:
                    log.debug("Failed to reset RAM registry at %s", reg_path, exc_info=True)

        @classmethod
        def reserved(cls, consumer_id: str) -> int:
            """Return the consumer's current weighted share in bytes.

            Raises KeyError if the consumer hasn't been registered via reserve().
            """
            with cls._lock:
                reg_path, _ = cls._registry_paths()
                remote_map = cls._read_remote_consumers(reg_path)

                my_weight: float | None = cls._consumers.get(consumer_id)
                if my_weight is None:
                    for _ct, remote_cons in remote_map.values():
                        if consumer_id in remote_cons:
                            my_weight = remote_cons[consumer_id][0]
                            break
                if my_weight is None:
                    raise KeyError(consumer_id)

                total_weight = sum(cls._consumers.values())
                total_committed = sum(cls._committed.get(cid, 0) for cid in cls._consumers)
                for _ct, remote_cons in remote_map.values():
                    for w, c in remote_cons.values():
                        total_weight += w
                        total_committed += c
                if total_weight <= 0.0:
                    total_weight = 1.0

            fraction = cls._max_ram_fraction()
            avail = cls.available_bytes()
            if total_committed > 0:
                total = cls.total_bytes()
                effective_available = min(total, avail + total_committed) if avail <= total else avail + total_committed
            else:
                effective_available = avail
            pool_bytes = int(effective_available * fraction)
            return int(pool_bytes * my_weight / total_weight)


class StationInfo(BaseModel):
    """Stable public identity of a Voxel control station."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    name: str


class _SystemFieldsEnvSource(EnvSettingsSource):
    """Restrict station env overrides to machine policy; identity remains file-backed."""

    def __call__(self) -> dict[str, object]:
        return {key: value for key, value in super().__call__().items() if key in System.model_fields}


class StationConfig(System):
    """Persisted control-station identity plus the complete local :class:`System` config.

    Unlike :class:`System`, this configuration never falls back to ``system.yaml``. Control
    entrypoints use :meth:`load` so a missing station is reported clearly; remote nodes continue
    to construct ``System`` and therefore need no station identity. The live station authority is
    intentionally a separate concept from this settings model.
    """

    schema_version: Literal[1] = 1
    id: UUID
    name: str = Field(min_length=1)

    @classmethod
    def config_path(cls) -> Path:
        """Return the required control-station configuration path."""
        return _voxel_home() / "station.yaml"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Read station YAML only; ``system.yaml`` is not a second configuration authority."""
        del env_settings, dotenv_settings, file_secret_settings
        return (
            init_settings,
            _SystemFieldsEnvSource(settings_cls),
            _yaml_source(settings_cls, cls.config_path()),
        )

    @classmethod
    def load(cls) -> Self:
        """Load the control station or explain how to initialize one."""
        # Import lazily because ``vxl.station`` models depend on ``StationInfo`` from this module.
        from vxl.station.errors import StationNotConfiguredError  # noqa: PLC0415

        path = cls.config_path()
        if not path.is_file():
            raise StationNotConfiguredError(
                f"No control station configured at {path}. Initialize one with `vxl station init --name <name>`."
            )
        return cls()  # pyright: ignore[reportCallIssue] -- required fields come from the checked YAML source

    @classmethod
    def initialize(cls, name: str, *, station_id: UUID | None = None) -> Self:
        """Create ``station.yaml`` from the effective System config, refusing to overwrite it."""
        path = cls.config_path()
        if path.exists():
            raise FileExistsError(f"A control station is already configured at {path}")

        system = System()
        station_config = cls(
            schema_version=1,
            id=station_id or uuid4(),
            name=name,
            store=system.store,
            scratch=system.scratch,
            max_ram_fraction=system.max_ram_fraction,
            remotes=system.remotes,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        save_yaml(path, station_config)
        return station_config

    @property
    def info(self) -> StationInfo:
        """Return the bounded station identity safe to expose to clients."""
        return StationInfo(id=self.id, name=self.name)


__all__ = [
    "Remote",
    "StationConfig",
    "StationInfo",
    "System",
    "load_voxel_env",
    "remote_store_fingerprint",
]
