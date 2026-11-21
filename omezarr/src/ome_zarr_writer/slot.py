"""Process-based buffer slot that downsamples AND writes — the whole batch pipeline in one worker."""

import logging
import math
import multiprocessing as mp
import os
import signal
import tempfile
import threading
import time
from collections.abc import Generator
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numba
import numba.core.config as numba_config
import numpy as np
from cloudpathlib import S3Path
from numba.np.ufunc.parallel import threading_layer
from pydantic import BaseModel, ConfigDict, Field
from vxlib.vector import UIVec3D

from ome_zarr_writer.array import ArrayWriter
from ome_zarr_writer.dataset import DownscaleType, Dtype, LossyDomain, ScaleLevel
from ome_zarr_writer.preprocess import PreprocessConfig, estimate_noise_sigma, filter_and_bin_volume, preprocess_volume
from ome_zarr_writer.storage import S3Store

from .pyramid import pyramids_3d_numba

log = logging.getLogger(__name__)

_PREFAULT_WORKERS = min(32, os.cpu_count() or 1)
_PREFAULT_MIN_BYTES = 1 << 28  # 256 MiB
_NUMBA_POOL_MAX = int(getattr(numba_config, "NUMBA_NUM_THREADS", os.cpu_count() or 1))
_NUMBA_THREADS = max(1, min(int(os.environ.get("VOXEL_NUMBA_THREADS", "16")), _NUMBA_POOL_MAX))
_GPU_CONCURRENT_WORKERS = max(1, int(os.environ.get("VOXEL_GPU_CONCURRENT_WORKERS", "2")))


@contextmanager
def _gpu_worker_slot() -> Generator[None]:
    """Cross-process flock semaphore capping concurrent GPU VRAM workers across camera nodes."""
    try:
        import fcntl  # noqa: PLC0415
    except ImportError:
        yield
        return
    lock_dir = Path(tempfile.gettempdir()) / f"voxel_ozw_locks_{getattr(os, 'getuid', lambda: 0)()}"
    lock_dir.mkdir(parents=True, exist_ok=True)
    start_idx = os.getpid() % _GPU_CONCURRENT_WORKERS
    fd: int | None = None
    try:
        while fd is None:
            for offset in range(_GPU_CONCURRENT_WORKERS):
                idx = (start_idx + offset) % _GPU_CONCURRENT_WORKERS
                cand = os.open(lock_dir / f"gpu_worker_{idx}.lock", os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(cand, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fd = cand
                    break
                except BlockingIOError:
                    os.close(cand)
            if fd is None:
                time.sleep(0.01)
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _prefault_zero(arr: np.ndarray) -> None:
    """Zero (commit) a newly allocated SharedMemory array in parallel off the capture path."""
    n = int(arr.shape[0])
    if arr.nbytes < _PREFAULT_MIN_BYTES or n <= 1 or _PREFAULT_WORKERS <= 1:
        arr.fill(0)
        return
    workers = min(_PREFAULT_WORKERS, n)
    step = math.ceil(n / workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda z: arr[z : z + step].fill(0), range(0, n, step)))


class SlotStage(IntEnum):
    ERROR = -1
    IDLE = 0
    COLLECTING = 1
    PROCESSING = 2


class OutputSetup(BaseModel):
    """Picklable description of where and how a slot's worker writes its pyramid."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    backend: ArrayWriter.Backend
    author_root: Path | S3Path
    store: S3Store | None
    channel: int
    levels: tuple[ScaleLevel, ...]
    batch_z: int
    volume_z: int
    reduction: DownscaleType = DownscaleType.MEAN
    use_gpu: bool = False
    skip_empty_chunks: bool = False
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)


@dataclass(frozen=True)
class BatchResult:
    process_started: datetime
    process_ended: datetime
    flush_started: datetime
    flush_ended: datetime
    flushed_bytes: int
    written_bytes: int | None = None


# ---------------------------------------------------------------------------
# Worker-side state & helpers
# ---------------------------------------------------------------------------

_RAW_SHM_KEY = -1
_WORKER_SHMS: list[SharedMemory] = []
_WORKER_ARRAYS: dict[ScaleLevel, np.ndarray] = {}
_WORKER_RAW: dict[str, np.ndarray] = {}
_WORKER_WRITERS: dict[ScaleLevel, ArrayWriter] = {}
_WORKER_STATE: dict[str, OutputSetup] = {}
_LAYER_WARNED: list[bool] = []
_WORKER_GPU_WS: dict[str, Any] = {}
_WORKER_PINNED_L0: list[np.ndarray] = []
_GPU_FALLBACK_WARNED: list[bool] = []
_WORKER_LOCKED_NOISE: dict[str, float] = {}


def _worker_init(shm_layout: list[tuple[int, str, tuple[int, int, int]]], dtype_str: str) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    numba.set_num_threads(_NUMBA_THREADS)
    for level_value, shm_name, shape in shm_layout:
        shm = SharedMemory(name=shm_name, create=False, track=False)
        arr = np.ndarray(shape, dtype=dtype_str, buffer=shm.buf)
        _WORKER_SHMS.append(shm)
        if level_value == _RAW_SHM_KEY:
            _WORKER_RAW["raw"] = arr
        else:
            _WORKER_ARRAYS[ScaleLevel(level_value)] = arr


def _is_gpu_writer(writer: ArrayWriter) -> bool:
    return hasattr(writer, "write_slice_gpu")


def _worker_bind_output(setup: OutputSetup) -> None:
    for writer in _WORKER_WRITERS.values():
        writer.close()
    _WORKER_WRITERS.clear()
    _WORKER_LOCKED_NOISE.clear()
    for level in setup.levels:
        writer = setup.backend()
        writer.open(setup.author_root / str(level.value), setup.store)
        if _is_gpu_writer(writer):
            writer.skip_empty_chunks = setup.skip_empty_chunks  # type: ignore[attr-defined]
        _WORKER_WRITERS[level] = writer
    _WORKER_STATE["setup"] = setup

    if setup.use_gpu or setup.backend == ArrayWriter.Backend.GPU:
        from ome_zarr_writer.array.gpu import release_shared_encoder_device_buffers  # noqa: PLC0415
        from ome_zarr_writer.pyramid_gpu import (  # noqa: PLC0415
            flush_gpu_deallocations,
            is_gpu_available,
            register_pinned_host_memory,
            warmup_gpu_pyramid,
        )

        if is_gpu_available():
            src_full = _WORKER_RAW.get("raw", _WORKER_ARRAYS.get(ScaleLevel.L0))
            if src_full is not None and not _WORKER_PINNED_L0 and register_pinned_host_memory(src_full):
                _WORKER_PINNED_L0.append(src_full)
            with _gpu_worker_slot():
                if src_full is not None:
                    warmup_gpu_pyramid(src_full.dtype, setup.reduction)
                for writer in _WORKER_WRITERS.values():
                    if _is_gpu_writer(writer):
                        writer.warmup()  # type: ignore[attr-defined]
                release_shared_encoder_device_buffers()
                flush_gpu_deallocations()


def _worker_close_output() -> None:
    for writer in _WORKER_WRITERS.values():
        writer.close()
    _WORKER_WRITERS.clear()
    _WORKER_LOCKED_NOISE.clear()
    if _WORKER_GPU_WS or _WORKER_PINNED_L0:
        from ome_zarr_writer.array.gpu import release_shared_encoder_buffers  # noqa: PLC0415
        from ome_zarr_writer.pyramid_gpu import (  # noqa: PLC0415
            flush_gpu_deallocations,
            unregister_pinned_host_memory,
        )

        for ws in _WORKER_GPU_WS.values():
            ws.close()
        _WORKER_GPU_WS.clear()
        release_shared_encoder_buffers()
        for pinned_arr in _WORKER_PINNED_L0:
            unregister_pinned_host_memory(pinned_arr)
        _WORKER_PINNED_L0.clear()
        flush_gpu_deallocations()


def _warn_if_slow_threading_layer() -> None:
    if _LAYER_WARNED:
        return
    _LAYER_WARNED.append(True)
    try:
        layer = threading_layer()
    except ValueError:
        return
    if layer == "workqueue":
        log.warning(
            "numba threading layer fell back to 'workqueue' (slowest); install a loadable 'tbb' (Linux) or "
            "ensure an OpenMP runtime is present for ~2x faster pyramid downsampling"
        )
    else:
        log.debug("numba threading layer: %s", layer)


def _write_host_levels(setup: OutputSetup, batch_idx: int, l0_block: np.ndarray | None = None) -> int:
    z_start = batch_idx * setup.batch_z
    z_end = min(z_start + setup.batch_z, setup.volume_z)
    flushed = 0
    for level in setup.levels:
        z0, z1 = z_start // level.factor, z_end // level.factor
        use_l0 = level == ScaleLevel.L0 and l0_block is not None
        src = l0_block[: z1 - z0] if use_l0 and l0_block is not None else _WORKER_ARRAYS[level][: z1 - z0]
        flushed += _WORKER_WRITERS[level].write_slice(setup.channel, z0, src)
    return flushed


def _worker_process_and_write(max_level_value: int, filled_l0: int, batch_idx: int) -> BatchResult:
    """CPU worker: preprocess, downsample into shared memory, and write every level."""
    setup = _WORKER_STATE.get("setup")
    if setup is None:
        raise RuntimeError("worker has no OutputSetup; call bind_output before flush")
    max_level = ScaleLevel(max_level_value)
    pp = setup.preprocess
    process_started = datetime.now(UTC)
    raw_full = _WORKER_RAW.get("raw")
    block = raw_full[:filled_l0] if raw_full is not None else _WORKER_ARRAYS[ScaleLevel.L0][:filled_l0]

    if pp.has_spatial_op:
        block = filter_and_bin_volume(
            block,
            bin_y=pp.bin_y,
            bin_x=pp.bin_x,
            pre_filter=pp.pre_filter,
            filter_size=pp.filter_size,
            bin_mode=pp.bin_mode,
            highpass_add=pp.highpass_add,
            use_gpu=False,
        )
        _WORKER_ARRAYS[ScaleLevel.L0][:filled_l0] = block

    if pp.has_uint_op or pp.has_lossy:
        sigma_val, offset_val = pp.noise_sigma, None
        if pp.has_lossy and (sigma_val is None or sigma_val <= 0.0 or pp.lossy_domain == LossyDomain.SQRT):
            if "sigma" not in _WORKER_LOCKED_NOISE:
                est_sigma, est_off = estimate_noise_sigma(block, domain=pp.lossy_domain)
                locked_sigma = est_sigma if (sigma_val is None or sigma_val <= 0.0) else float(sigma_val)
                _WORKER_LOCKED_NOISE["sigma"] = locked_sigma
                _WORKER_LOCKED_NOISE["offset"] = est_off
            sigma_val = _WORKER_LOCKED_NOISE["sigma"]
            offset_val = _WORKER_LOCKED_NOISE.get("offset", 0.0)
        block = preprocess_volume(
            block,
            keep_bits=pp.keep_bits,
            dark_floor=pp.dark_floor,
            lossy_step_noise=pp.lossy_step_noise,
            lossy_domain=pp.lossy_domain,
            noise_sigma=sigma_val,
            sqrt_offset=offset_val,
        )
        _WORKER_ARRAYS[ScaleLevel.L0][:filled_l0] = block

    pyramid = pyramids_3d_numba(block, max_level, reduction=setup.reduction, parallel=True)
    _warn_if_slow_threading_layer()
    for level, vol in pyramid.items():
        if level == ScaleLevel.L0:
            continue
        arr = _WORKER_ARRAYS[level]
        z, y, x = min(arr.shape[0], vol.shape[0]), min(arr.shape[1], vol.shape[1]), min(arr.shape[2], vol.shape[2])
        arr[:z, :y, :x] = vol[:z, :y, :x].astype(arr.dtype)

    process_ended = datetime.now(UTC)
    flushed = _write_host_levels(setup, batch_idx, l0_block=block)
    return BatchResult(process_started, process_ended, process_ended, datetime.now(UTC), flushed)


def _get_worker_gpu_workspace(
    l0_full: np.ndarray, raw_full: np.ndarray | None, max_level: ScaleLevel, needs_spatial: bool
) -> Any:
    from ome_zarr_writer.pyramid_gpu import GpuPyramidWorkspace  # noqa: PLC0415

    shape_l0 = UIVec3D(z=l0_full.shape[0], y=l0_full.shape[1], x=l0_full.shape[2])
    raw_shape_l0 = (
        UIVec3D(z=raw_full.shape[0], y=raw_full.shape[1], x=raw_full.shape[2])
        if raw_full is not None
        else (shape_l0 if needs_spatial else None)
    )
    ws = _WORKER_GPU_WS.get("ws")
    if (
        ws is None
        or ws.shape_l0 != shape_l0
        or ws.raw_shape_l0 != raw_shape_l0
        or ws.max_level != max_level
        or ws.dtype != l0_full.dtype
    ):
        if ws is not None:
            ws.close()
        ws = GpuPyramidWorkspace(shape_l0, max_level, l0_full.dtype, raw_shape_l0=raw_shape_l0)
        _WORKER_GPU_WS["ws"] = ws
    return ws


def _worker_process_and_write_gpu(max_level_value: int, filled_l0: int, batch_idx: int) -> BatchResult:
    """GPU worker: compute pyramid on GPU and write directly via GPUArrayWriter or host fallback."""
    setup = _WORKER_STATE.get("setup")
    if setup is None:
        raise RuntimeError("worker has no OutputSetup; call bind_output before flush")
    from ome_zarr_writer.array.gpu import release_shared_encoder_device_buffers  # noqa: PLC0415
    from ome_zarr_writer.pyramid_gpu import (  # noqa: PLC0415
        flush_gpu_deallocations,
        is_gpu_available,
        register_pinned_host_memory,
    )

    if not is_gpu_available():
        if not _GPU_FALLBACK_WARNED:
            _GPU_FALLBACK_WARNED.append(True)
            log.warning("GPU requested for BatchSlot worker, but CUDA is unavailable; falling back to CPU worker")
        return _worker_process_and_write(max_level_value, filled_l0, batch_idx)

    max_level = ScaleLevel(max_level_value)
    l0_full = _WORKER_ARRAYS[ScaleLevel.L0]
    raw_full = _WORKER_RAW.get("raw")
    src_full = raw_full if raw_full is not None else l0_full
    if not _WORKER_PINNED_L0 and register_pinned_host_memory(src_full):
        _WORKER_PINNED_L0.append(src_full)

    all_gpu_writers = all(_is_gpu_writer(_WORKER_WRITERS[lvl]) for lvl in setup.levels)
    z_start = batch_idx * setup.batch_z
    z_end = min(z_start + setup.batch_z, setup.volume_z)
    flushed = 0
    written_bytes: int | None = None

    with _gpu_worker_slot():
        process_started = datetime.now(UTC)
        ws: Any = None
        try:
            ws = _get_worker_gpu_workspace(l0_full, raw_full, max_level, setup.preprocess.has_spatial_op)
            d_levels: dict[ScaleLevel, Any] = {}
            try:
                d_levels = ws.compute_from_host(
                    src_full[:filled_l0],
                    max_level,
                    reduction=setup.reduction,
                    preprocess=setup.preprocess,
                    copy_levels_to=None if all_gpu_writers else _WORKER_ARRAYS,
                    copy_l0_to=(l0_full if (not all_gpu_writers and setup.preprocess.has_any) else None),
                    return_host_f32=False,
                )
                process_ended = datetime.now(UTC)
                if all_gpu_writers:
                    written_bytes = 0
                    for level in setup.levels:
                        z0, z1 = z_start // level.factor, z_end // level.factor
                        gpu_writer: Any = _WORKER_WRITERS[level]
                        flushed += gpu_writer.write_slice_gpu(setup.channel, z0, d_levels[level][: z1 - z0])
                        written_bytes += gpu_writer.last_written_bytes
                        gpu_writer.release_device_buffers()
            finally:
                d_levels.clear()
                ws.release_device_buffers()
                release_shared_encoder_device_buffers()
                flush_gpu_deallocations()
        except Exception as exc:
            if "OUT_OF_MEMORY" in str(exc) or "OutOfMemory" in type(exc).__name__:
                log.warning("GPU VRAM exhausted during batch processing (%s); falling back to CPU worker", exc)
                if ws is not None:
                    ws.release_device_buffers()
                release_shared_encoder_device_buffers()
                flush_gpu_deallocations()
                if not all_gpu_writers:
                    return _worker_process_and_write(max_level_value, filled_l0, batch_idx)
            raise

    if not all_gpu_writers:
        flushed = _write_host_levels(setup, batch_idx)
    return BatchResult(process_started, process_ended, process_ended, datetime.now(UTC), flushed, written_bytes)


# ---------------------------------------------------------------------------
# Main-process slot
# ---------------------------------------------------------------------------


class BatchSlot:
    """A ring slot backed by SharedMemory whose worker downsamples and writes each batch to the store."""

    def __init__(
        self,
        name: str,
        shape_l0: UIVec3D,
        max_level: ScaleLevel,
        dtype: Dtype,
        raw_shape_l0: UIVec3D | None = None,
    ) -> None:
        self.name = name
        self.shape_l0 = shape_l0
        self.raw_shape_l0 = raw_shape_l0 if (raw_shape_l0 is not None and raw_shape_l0 != shape_l0) else None
        self.max_level = max_level
        self._dtype = dtype.dtype
        self._dtype_str = np.dtype(self._dtype).str
        self.filled_l0 = 0
        self.batch_idx: int | None = None
        self._use_gpu = False
        self._condition = threading.Condition()
        self._active_readers = 0
        self._stage = SlotStage.IDLE
        self._future: Future[BatchResult] | None = None
        self._shms: dict[int, SharedMemory] = {}
        self._arrays: dict[ScaleLevel, np.ndarray] = {}
        self._raw_array: np.ndarray | None = None
        shm_layout: list[tuple[int, str, tuple[int, int, int]]] = []

        def _alloc(key: int, suffix: str, shape: tuple[int, int, int]) -> np.ndarray:
            nbytes = int(np.prod(shape)) * self._dtype.itemsize
            shm_name = f"{self.name}_{suffix}"
            shm = SharedMemory(name=shm_name, create=True, size=nbytes, track=False)
            arr = np.ndarray(shape, dtype=self._dtype, buffer=shm.buf)
            _prefault_zero(arr)
            self._shms[key] = shm
            shm_layout.append((key, shm_name, shape))
            return arr

        try:
            if self.raw_shape_l0 is not None:
                r = self.raw_shape_l0
                self._raw_array = _alloc(_RAW_SHM_KEY, "raw", (r.z, r.y, r.x))
            for level in self.max_level.levels:
                s = level.scale(self.shape_l0)
                self._arrays[level] = _alloc(level.value, str(level.value), (s.z, s.y, s.x))
        except Exception:
            self._release_shms()
            raise

        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(shm_layout, self._dtype_str),
        )

    @property
    def stage(self) -> SlotStage:
        fut = self._future
        if fut is None:
            return self._stage
        if not fut.done():
            return SlotStage.PROCESSING
        return SlotStage.ERROR if fut.exception() is not None else SlotStage.IDLE

    @property
    def active_readers(self) -> int:
        with self._condition:
            return self._active_readers

    @property
    def reusable(self) -> bool:
        return self.stage == SlotStage.IDLE and self.active_readers == 0

    def bind_output(self, setup: OutputSetup) -> None:
        self._use_gpu = setup.use_gpu
        self._executor.submit(_worker_bind_output, setup).result()

    def assign_batch(self, batch_idx: int) -> None:
        with self._condition:
            if not self.reusable:
                raise ValueError(
                    f"assign_batch requires a reusable slot, got stage={self.stage.name}, "
                    f"active_readers={self._active_readers}"
                )
            self._future = None
            self._stage = SlotStage.COLLECTING
            self.batch_idx = batch_idx
            self.filled_l0 = 0

    def frame_array(self) -> np.ndarray:
        with self._condition:
            if self.stage != SlotStage.COLLECTING:
                raise RuntimeError(f"frame_array requires COLLECTING, got {self.stage.name}")
            if self.filled_l0 >= self.shape_l0.z:
                raise IndexError(f"slot is full at L0 depth {self.shape_l0.z}")
            buf = self._raw_array if self._raw_array is not None else self._arrays[ScaleLevel.L0]
            return buf[self.filled_l0]

    def commit_frame(self) -> None:
        with self._condition:
            if self.stage != SlotStage.COLLECTING:
                raise RuntimeError(f"commit_frame requires COLLECTING, got {self.stage.name}")
            if self.filled_l0 >= self.shape_l0.z:
                raise IndexError(f"slot is full at L0 depth {self.shape_l0.z}")
            self.filled_l0 += 1

    def retain_last_frame(self) -> np.ndarray:
        with self._condition:
            if self.filled_l0 <= 0:
                raise RuntimeError("cannot retain a slot with no committed frames")
            self._active_readers += 1
            buf = self._raw_array if self._raw_array is not None else self._arrays[ScaleLevel.L0]
            frame = buf[self.filled_l0 - 1].view()
            frame.flags.writeable = False
            return frame

    def abort_empty_batch(self) -> None:
        with self._condition:
            if self.stage != SlotStage.COLLECTING or self.filled_l0 != 0:
                raise RuntimeError("only an empty collecting slot can be aborted")
            self._stage = SlotStage.IDLE
            self.batch_idx = None

    def release_reader(self) -> None:
        with self._condition:
            if self._active_readers <= 0:
                raise RuntimeError("release_reader called with no active readers")
            self._active_readers -= 1
            self._condition.notify_all()

    def wait_for_reader_change(self, timeout: float) -> None:
        with self._condition:
            self._condition.wait(timeout=timeout)

    def flush(self, *, use_gpu: bool | None = None) -> Future[BatchResult]:
        if self.batch_idx is None:
            raise RuntimeError("cannot flush a slot with no batch assigned")
        run_gpu = self._use_gpu if use_gpu is None else use_gpu
        worker_fn = _worker_process_and_write_gpu if run_gpu else _worker_process_and_write
        self._future = self._executor.submit(worker_fn, self.max_level.value, self.filled_l0, self.batch_idx)
        return self._future

    def close(self) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._active_readers == 0)
        try:
            self._executor.submit(_worker_close_output).result()
        except Exception:
            log.debug("Error closing output writers for %s", self.name, exc_info=True)
        finally:
            self._executor.shutdown(wait=True)
            self._release_shms()

    def _release_shms(self) -> None:
        self._raw_array = None
        self._arrays.clear()
        for shm in self._shms.values():
            try:
                shm.close()
            except Exception:
                log.debug("Error closing shared memory %s", shm.name, exc_info=True)
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                log.debug("Error unlinking shared memory %s", shm.name, exc_info=True)
        self._shms.clear()
