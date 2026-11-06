import asyncio
import importlib
import logging
import traceback
from collections import defaultdict
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class BuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str  # Fully qualified class name, e.g., "rigup.objs.camera.Camera"
    init: dict[str, Any] = Field(default_factory=dict)  # Constructor arguments
    # Values set via setattr on the instance after construction. Goes through descriptor
    # setters (so validation/clamping applies there), but a setattr that raises aborts the
    # device's build with BuildError(error_type="defaults") rather than silently warning.
    defaults: dict[str, Any] | None = None
    fallback: "BuildConfig | None" = None

    def get_obj_class(self) -> type:
        """Dynamically import and return the obj class."""
        parts = self.target.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid target format: {self.target}")

        module_name, class_name = parts
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


class BuildError(BaseModel):
    """Error information for a failed obj build."""

    uid: str
    error_type: Literal["import", "instantiation", "defaults", "dependency", "circular", "internal"]
    message: str
    traceback: str | None = None


type BuildGroupSpec = Mapping[str, BuildConfig]


type BuildObjectsResult[T] = tuple[Mapping[str, T], Mapping[str, BuildError]]


def _check_dependency_status(
    uid: str,
    cfg: BuildConfig,
    cfgs: BuildGroupSpec,
    deps: set[str],
    errors: Mapping[str, BuildError],
    fell_back: set[str],
    *,
    single_failed_message: str | None = None,
) -> tuple[bool, BuildError | None]:
    """Return ``(force_fallback, error)`` given resolved dependency statuses for ``uid``."""
    failed_deps = deps & errors.keys()
    if cfg.fallback is not None and (failed_deps or (deps & fell_back)):
        fallback_deps = _extract_deps_from_init(cfgs, uid, cfg.fallback.init)
        if failed_fallback := sorted(fallback_deps & errors.keys()):
            return False, BuildError(
                uid=uid,
                error_type="dependency",
                message=f"Dependency failed: {', '.join(failed_fallback)}",
            )
        return True, None
    if failed_deps:
        msg = single_failed_message or f"Dependency failed: {', '.join(sorted(failed_deps))}"
        return False, BuildError(uid=uid, error_type="dependency", message=msg)
    return False, None


def _record_build_result[T](
    uid: str,
    obj: T | None,
    error: BuildError | None,
    used_fallback: bool,
    built: dict[str, T],
    errors: dict[str, BuildError],
    fell_back: set[str],
) -> T | BuildError:
    if used_fallback:
        fell_back.add(uid)
    if obj is not None:
        built[uid] = obj
    if error is not None:
        errors[uid] = error
        return error
    return built[uid]


def _filter_layer_groups(
    layer: list[list[str]],
    cfgs: BuildGroupSpec,
    errors: dict[str, BuildError],
    fell_back: set[str],
) -> tuple[list[list[str]], set[str]]:
    valid_groups: list[list[str]] = []
    force_fallback_uids: set[str] = set()
    for group in layer:
        valid_uids: list[str] = []
        for uid in group:
            force_fallback, dep_error = _check_dependency_status(
                uid,
                cfgs[uid],
                cfgs,
                _extract_deps(cfgs, uid),
                errors,
                fell_back,
            )
            if dep_error is not None:
                errors[uid] = dep_error
                continue
            if force_fallback:
                force_fallback_uids.add(uid)
            valid_uids.append(uid)
        if valid_uids:
            valid_groups.append(valid_uids)
    return valid_groups, force_fallback_uids


def build_objects[T](cfgs: BuildGroupSpec, base_cls: type[T] = object) -> BuildObjectsResult[T]:
    """Build objects from configuration with error accumulation and dependency resolution.

    Args:
        cfgs: Node configuration containing obj specifications
        base_cls: Base class for type checking (default: object)

    Returns:
        Tuple of (owned_objects, build_errors). A failed object whose close also fails is
        retained in both mappings so the owner can retry cleanup; it cannot be used as a dependency.
    """
    built: dict[str, T] = {}
    fell_back: set[str] = set()
    errors = {
        uid: BuildError(
            uid=uid,
            error_type="circular",
            message=f"Circular dependency detected for obj '{uid}'",
        )
        for uid in _find_cyclic_uids(cfgs)
    }
    building: set[str] = set()

    def _resolve_deps(obj_cfg: BuildConfig, deps: set[str]) -> str | None:
        for dep_uid in deps:
            if dep_uid in errors:
                if obj_cfg.fallback is None:
                    return f"Dependency '{dep_uid}' failed to build"
                continue
            dep_result = _build_one(dep_uid)
            if isinstance(dep_result, BuildError) and obj_cfg.fallback is None:
                return f"Dependency '{dep_uid}' failed: {dep_result.message}"
        return None

    def _build_one(uid: str) -> T | BuildError:
        """Build a single obj, resolving dependencies first."""
        if uid in errors:
            return errors[uid]
        if uid in built:
            return built[uid]
        if uid in building:
            errors[uid] = BuildError(
                uid=uid,
                error_type="circular",
                message=f"Circular dependency detected for obj '{uid}'",
            )
            return errors[uid]

        building.add(uid)
        try:
            obj_cfg = cfgs[uid]
            deps = _extract_deps(cfgs, uid)
            first_failure = _resolve_deps(obj_cfg, deps)
            force_fallback, dep_error = _check_dependency_status(
                uid,
                obj_cfg,
                cfgs,
                deps,
                errors,
                fell_back,
                single_failed_message=first_failure,
            )
            if dep_error is not None:
                errors[uid] = dep_error
                return dep_error

            obj, error, used_fallback = _construct_with_fallback(
                uid, obj_cfg, built, base_cls, force_fallback=force_fallback
            )
            return _record_build_result(uid, obj, error, used_fallback, built, errors, fell_back)
        finally:
            building.discard(uid)

    for uid in cfgs:
        if uid not in built and uid not in errors:
            _build_one(uid)

    return built, errors


def _extract_deps_from_init(cfgs: BuildGroupSpec, uid: str, init: Mapping[str, Any]) -> set[str]:
    """Extract device UIDs referenced in an init kwargs mapping."""
    deps: set[str] = set()

    def scan(value: Any) -> None:
        if isinstance(value, str) and value in cfgs and value != uid:
            deps.add(value)
        elif isinstance(value, list):
            for item in value:
                scan(item)
        elif isinstance(value, dict):
            for v in value.values():
                scan(v)

    for v in init.values():
        scan(v)
    return deps


def _extract_deps(cfgs: BuildGroupSpec, uid: str) -> set[str]:
    """Extract device UIDs referenced in a device's init kwargs."""
    return _extract_deps_from_init(cfgs, uid, cfgs[uid].init)


def _find_cyclic_uids(cfgs: BuildGroupSpec) -> set[str]:
    """Return exactly the UIDs participating in dependency cycles.

    Devices merely downstream of a cycle are excluded; the builders report
    those as dependency failures instead.
    """
    state: dict[str, Literal["visiting", "visited"]] = {}
    stack: list[str] = []
    stack_index: dict[str, int] = {}
    cyclic: set[str] = set()

    def visit(uid: str) -> None:
        state[uid] = "visiting"
        stack_index[uid] = len(stack)
        stack.append(uid)
        for dependency in _extract_deps(cfgs, uid):
            match state.get(dependency):
                case None:
                    visit(dependency)
                case "visiting":
                    cyclic.update(stack[stack_index[dependency] :])
                case "visited":
                    pass
        stack.pop()
        stack_index.pop(uid)
        state[uid] = "visited"

    for uid in cfgs:
        if uid not in state:
            visit(uid)
    return cyclic


def _topological_layers(cfgs: BuildGroupSpec) -> list[list[list[str]]]:
    """Sort devices into dependency layers, grouped by target within each layer.

    Returns a list of layers. Each layer is a list of groups. Each group is a
    list of UIDs sharing the same target class (built sequentially for SDK safety).
    Groups within a layer can be built concurrently.
    """
    # Build adjacency and in-degree
    deps_map: dict[str, set[str]] = {uid: _extract_deps(cfgs, uid) for uid in cfgs}
    in_degree: dict[str, int] = dict.fromkeys(cfgs, 0)
    for uid, deps in deps_map.items():
        in_degree[uid] = len(deps)

    remaining = set(cfgs.keys())
    layers: list[list[list[str]]] = []

    while remaining:
        # Find all devices with no unresolved dependencies
        ready = [uid for uid in cfgs if uid in remaining and in_degree[uid] == 0]
        if not ready:
            # Circular dependency — dump remaining as a single layer
            logger.warning("Circular dependency detected among: %s", remaining)
            layers.append([list(remaining)])
            break

        # Group ready devices by target
        by_target: dict[str, list[str]] = defaultdict(list)
        for uid in ready:
            by_target[cfgs[uid].target].append(uid)

        layers.append(list(by_target.values()))

        # Remove built devices and update in-degrees
        for uid in ready:
            remaining.discard(uid)
        for uid in remaining:
            in_degree[uid] = len(deps_map[uid] & remaining)

    return layers


def _resolve_references(value: Any, built: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value in built:
        return built[value]
    if isinstance(value, list):
        return [_resolve_references(item, built) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_references(item, built) for key, item in value.items()}
    return value


def _construct[T](
    uid: str, cfg: BuildConfig, built: Mapping[str, T], base_cls: type[T]
) -> tuple[T | None, BuildError | None]:
    """Construct and apply defaults, retaining any object whose cleanup fails."""
    obj: T | None = None
    phase: Literal["import", "instantiation", "defaults"] = "import"
    try:
        cls = cfg.get_obj_class()
        if not issubclass(cls, base_cls):
            logger.warning(
                "object %s uses class %s which is not a subclass of %s", uid, cls.__name__, base_cls.__name__
            )
        phase = "instantiation"
        init = _resolve_references(cfg.init, built)
        init.setdefault("uid", uid)
        obj = cls(**init)
        phase = "defaults"
        for name, value in (cfg.defaults or {}).items():
            setattr(obj, name, value)
        return obj, None
    except Exception as error:
        message = f"Failed {phase} for '{uid}' ({cfg.target}): {error}"
        trace = traceback.format_exc()
        if obj is not None:
            try:
                if close := getattr(obj, "close", None):
                    close()
            except Exception as cleanup_error:
                message += f"; cleanup failed and is retained for retry: {cleanup_error}"
            else:
                obj = None
        return obj, BuildError(uid=uid, error_type=phase, message=message, traceback=trace)


def _construct_with_fallback[T](
    uid: str,
    cfg: BuildConfig,
    built: Mapping[str, T],
    base_cls: type[T],
    *,
    force_fallback: bool = False,
) -> tuple[T | None, BuildError | None, bool]:
    """Construct ``cfg`` or its ``fallback``, returning ``(obj, error, used_fallback)``."""
    if force_fallback and cfg.fallback is not None:
        logger.warning(
            "Device '%s' (%s) using fallback %s because its hardware dependency is offline",
            uid,
            cfg.target,
            cfg.fallback.target,
        )
        obj, error = _construct(uid, cfg.fallback, built, base_cls)
        return obj, error, error is None

    obj, error = _construct(uid, cfg, built, base_cls)
    if error is None or cfg.fallback is None or obj is not None:
        return obj, error, False

    logger.warning(
        "Device '%s' (%s) unavailable (%s); falling back to %s",
        uid,
        cfg.target,
        error.message,
        cfg.fallback.target,
    )
    fallback_obj, fallback_error = _construct(uid, cfg.fallback, built, base_cls)
    return fallback_obj, fallback_error, fallback_error is None


def _build_group_sync[T](
    group: list[str],
    cfgs: BuildGroupSpec,
    built: dict[str, T],
    base_cls: type[T],
    force_fallback_uids: set[str] | None = None,
) -> list[tuple[str, T | None, BuildError | None, bool]]:
    # Driver groups are serialized; prior-layer objects are read-only in worker threads.
    forced = force_fallback_uids or set()
    return [
        (uid, *_construct_with_fallback(uid, cfgs[uid], built, base_cls, force_fallback=uid in forced)) for uid in group
    ]


def _prewarm_parent_modules(cfgs: BuildGroupSpec) -> None:
    """Import parent modules of all targets before worker threads launch.

    Worker threads import device classes concurrently via ``to_thread``.
    If two threads race on importing the same parent package whose
    ``__init__`` has a deep import chain (common in application packages
    like ``vxl``), one thread can see a partially-initialized module in
    ``sys.modules``. Pre-importing parent modules here — on the main
    thread, sequentially — guarantees every ``__init__`` is fully executed
    before any worker touches it.
    """
    parents = {cfg.target.rsplit(".", 1)[0] for cfg in cfgs.values()}
    for cfg in cfgs.values():
        if cfg.fallback is not None:
            parents.add(cfg.fallback.target.rsplit(".", 1)[0])
    for mod_name in sorted(parents):
        with suppress(Exception):
            importlib.import_module(mod_name)


async def build_objects_async[T](cfgs: BuildGroupSpec, base_cls: type[T] = object) -> BuildObjectsResult[T]:
    """Build devices asynchronously with dependency resolution.

    Devices are sorted into dependency layers. Within each layer, devices
    sharing the same target class are built sequentially (SDK thread safety),
    while groups with different targets are built concurrently.
    """
    if not cfgs:
        return {}, {}

    cyclic = _find_cyclic_uids(cfgs)
    errors = {
        uid: BuildError(
            uid=uid,
            error_type="circular",
            message=f"Circular dependency detected for obj '{uid}'",
        )
        for uid in cyclic
    }
    buildable = {uid: cfg for uid, cfg in cfgs.items() if uid not in cyclic}
    layers = _topological_layers(buildable)
    built: dict[str, T] = {}
    fell_back: set[str] = set()

    for layer in layers:
        valid_groups, force_fallback_uids = _filter_layer_groups(layer, cfgs, errors, fell_back)
        if not valid_groups:
            continue

        valid_configs = {uid: cfgs[uid] for group in valid_groups for uid in group}
        _prewarm_parent_modules(valid_configs)

        # Build groups concurrently, each group sequential in its own thread
        group_results = await asyncio.gather(
            *(
                asyncio.to_thread(_build_group_sync, group, cfgs, built, base_cls, force_fallback_uids)
                for group in valid_groups
            ),
            return_exceptions=True,
        )

        for group, group_result in zip(valid_groups, group_results, strict=True):
            if isinstance(group_result, BaseException):
                logger.error("Unexpected error during device build: %s", group_result)
                formatted_traceback = "".join(
                    traceback.format_exception(type(group_result), group_result, group_result.__traceback__)
                )
                for uid in group:
                    errors[uid] = BuildError(
                        uid=uid,
                        error_type="internal",
                        message=f"Unexpected error during device build: {group_result}",
                        traceback=formatted_traceback,
                    )
                continue
            for item in group_result:
                uid, obj, error = item[0], item[1], item[2]
                used_fallback = item[3] if len(item) > 3 else False
                _record_build_result(uid, obj, error, used_fallback, built, errors, fell_back)

    return built, errors


async def build_for_owner[T](
    cfgs: BuildGroupSpec, base_cls: type[T]
) -> tuple[Mapping[str, T], Mapping[str, BuildError], asyncio.CancelledError | None]:
    """Settle construction before returning resources and any pending cancellation to their owner."""
    building = asyncio.create_task(build_objects_async(cfgs, base_cls))
    cancelled: asyncio.CancelledError | None = None
    while not building.done():
        try:
            await asyncio.shield(building)
        except asyncio.CancelledError as error:
            # SDK constructor threads cannot be stopped. Their results must be adopted before cleanup.
            cancelled = error
    built, errors = building.result()
    return built, errors, cancelled
