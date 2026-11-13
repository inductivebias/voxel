# pyright: reportArgumentType=false
"""GPU-accelerated 3D multi-scale pyramid generation using Numba CUDA.

Provides:
- :func:`pyramids_3d_gpu`: drop-in GPU counterpart to :func:`~ome_zarr_writer.pyramid.pyramids_3d_numba`,
  supporting all four :class:`~ome_zarr_writer.dataset.DownscaleType` modes (``GAUSSIAN``, ``MEAN``,
  ``MAX``, ``MIN``) and all bit-depths (8-bit ``uint8``, 12/16-bit ``uint16``, ``uint32``, ``float32``, etc.).
- :class:`GpuPyramidWorkspace`: pre-allocated, reusable device buffer arena for zero-allocation per-batch
  pyramid generation inside :class:`~ome_zarr_writer.slot.BatchSlot` workers.
- :func:`register_pinned_host_memory` / :func:`unregister_pinned_host_memory`: page-locks host
  ``SharedMemory`` segments via ``cudaHostRegister`` so H2D DMA transfers run at full PCIe bandwidth.
"""

import ctypes
import logging
import math
import warnings
from typing import Any

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from vxlib.vector import UIVec3D

from ome_zarr_writer.dataset import DownscaleType, Dtype, ScaleLevel

log = logging.getLogger(__name__)

_SUPPORTED_REDUCTIONS = frozenset({DownscaleType.MEAN, DownscaleType.MAX, DownscaleType.MIN, DownscaleType.GAUSSIAN})
_THREADS_3D = (32, 8, 1)


def is_gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is available to Numba CUDA."""
    try:
        return bool(cuda.is_available() and len(cuda.gpus) > 0)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Host memory page-locking (cudaHostRegister / cudaHostUnregister)
# ---------------------------------------------------------------------------

_CUDART: ctypes.CDLL | None = None
_CUDART_LOADED = False


def _get_cudart() -> ctypes.CDLL | None:
    global _CUDART, _CUDART_LOADED  # noqa: PLW0603
    if _CUDART_LOADED:
        return _CUDART
    _CUDART_LOADED = True
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            lib.cudaHostRegister.restype = ctypes.c_int
            lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            lib.cudaHostUnregister.restype = ctypes.c_int
            _CUDART = lib
            return _CUDART
        except OSError:
            continue
    return None


def register_pinned_host_memory(arr: np.ndarray) -> bool:
    """Page-lock a contiguous host numpy array (e.g. ``SharedMemory``) with ``cudaHostRegister``.

    Returns True if registration succeeded, False otherwise (best-effort; unpinned DMA still works).
    """
    if arr.nbytes <= 0 or not arr.flags.c_contiguous or not is_gpu_available():
        return False
    lib = _get_cudart()
    if lib is None:
        return False
    try:
        # Ensure the current thread's CUDA context is initialized before registering host pages.
        cuda.current_context()
        rc = int(lib.cudaHostRegister(ctypes.c_void_p(arr.ctypes.data), ctypes.c_size_t(arr.nbytes), ctypes.c_uint(0)))
    except Exception:
        return False
    return rc == 0


def unregister_pinned_host_memory(arr: np.ndarray) -> None:
    """Unpin a previously page-locked host numpy array before closing its ``SharedMemory``."""
    lib = _get_cudart()
    if lib is None or arr.nbytes <= 0:
        return
    try:
        lib.cudaHostUnregister(ctypes.c_void_p(arr.ctypes.data))
    except Exception:
        log.debug("cudaHostUnregister failed", exc_info=True)


# ---------------------------------------------------------------------------
# CUDA Kernels (MEAN, MAX, MIN, and 2-pass separable [1, 3, 3, 1] GAUSSIAN)
# ---------------------------------------------------------------------------


@cuda.jit(fastmath=True)
def _cuda_kernel_mean_cast(vol: Any, out_f32: Any, out_cast: Any) -> None:
    xo, yo, zo = cuda.grid(3)  # type: ignore[misc]
    z_out, y_out, x_out = out_f32.shape
    if zo >= z_out or yo >= y_out or xo >= x_out:
        return
    t2 = zo * 2
    i2 = yo * 2
    j2 = xo * 2
    s = np.float32(0.0)
    for dt in range(2):
        for di in range(2):
            for dj in range(2):
                s += np.float32(vol[t2 + dt, i2 + di, j2 + dj])
    val = s * np.float32(0.125)
    out_f32[zo, yo, xo] = val
    out_cast[zo, yo, xo] = val


@cuda.jit(fastmath=True)
def _cuda_kernel_max_cast(vol: Any, out_f32: Any, out_cast: Any) -> None:
    xo, yo, zo = cuda.grid(3)  # type: ignore[misc]
    z_out, y_out, x_out = out_f32.shape
    if zo >= z_out or yo >= y_out or xo >= x_out:
        return
    t2 = zo * 2
    i2 = yo * 2
    j2 = xo * 2
    m = np.float32(vol[t2, i2, j2])
    for dt in range(2):
        for di in range(2):
            for dj in range(2):
                v = np.float32(vol[t2 + dt, i2 + di, j2 + dj])
                if v > m:  # noqa: PLR1730 - Python max() unsupported in Numba @cuda.jit
                    m = v
    out_f32[zo, yo, xo] = m
    out_cast[zo, yo, xo] = m


@cuda.jit(fastmath=True)
def _cuda_kernel_min_cast(vol: Any, out_f32: Any, out_cast: Any) -> None:
    xo, yo, zo = cuda.grid(3)  # type: ignore[misc]
    z_out, y_out, x_out = out_f32.shape
    if zo >= z_out or yo >= y_out or xo >= x_out:
        return
    t2 = zo * 2
    i2 = yo * 2
    j2 = xo * 2
    m = np.float32(vol[t2, i2, j2])
    for dt in range(2):
        for di in range(2):
            for dj in range(2):
                v = np.float32(vol[t2 + dt, i2 + di, j2 + dj])
                if v < m:  # noqa: PLR1730 - Python min() unsupported in Numba @cuda.jit
                    m = v
    out_f32[zo, yo, xo] = m
    out_cast[zo, yo, xo] = m


@cuda.jit(fastmath=True)
def _cuda_gauss_xy(vol: Any, xy_out: Any) -> None:
    """Fused 2D separable [1, 3, 3, 1] X+Y pass per Z-plane (un-normalised sum in float32)."""
    xo, yo, z = cuda.grid(3)  # type: ignore[misc]
    _, y_in, x_in = vol.shape
    z_out, y_out, x_out = xy_out.shape
    if z >= z_out or yo >= y_out or xo >= x_out:
        return

    cx = xo * 2
    x0 = 0 if cx - 1 < 0 else cx - 1  # noqa: FURB136 - Python max() unsupported in Numba @cuda.jit
    x2 = x_in - 1 if cx + 1 >= x_in else cx + 1
    x3 = x_in - 1 if cx + 2 >= x_in else cx + 2

    cy = yo * 2
    y0 = 0 if cy - 1 < 0 else cy - 1  # noqa: FURB136 - Python max() unsupported in Numba @cuda.jit
    y2 = y_in - 1 if cy + 1 >= y_in else cy + 1
    y3 = y_in - 1 if cy + 2 >= y_in else cy + 2

    three = np.float32(3.0)
    r0 = (
        np.float32(vol[z, y0, x0])
        + three * np.float32(vol[z, y0, cx])
        + three * np.float32(vol[z, y0, x2])
        + np.float32(vol[z, y0, x3])
    )
    r1 = (
        np.float32(vol[z, cy, x0])
        + three * np.float32(vol[z, cy, cx])
        + three * np.float32(vol[z, cy, x2])
        + np.float32(vol[z, cy, x3])
    )
    r2 = (
        np.float32(vol[z, y2, x0])
        + three * np.float32(vol[z, y2, cx])
        + three * np.float32(vol[z, y2, x2])
        + np.float32(vol[z, y2, x3])
    )
    r3 = (
        np.float32(vol[z, y3, x0])
        + three * np.float32(vol[z, y3, cx])
        + three * np.float32(vol[z, y3, x2])
        + np.float32(vol[z, y3, x3])
    )

    xy_out[z, yo, xo] = r0 + three * r1 + three * r2 + r3


@cuda.jit(fastmath=True)
def _cuda_gauss_z_cast(xy_in: Any, out_f32: Any, out_cast: Any) -> None:
    """Z pass: combine four X/Y-reduced planes with weights [1, 3, 3, 1] * (1/512) into float32 and target dtype."""
    xo, yo, zo = cuda.grid(3)  # type: ignore[misc]
    z_in = xy_in.shape[0]
    z_out, y_out, x_out = out_f32.shape
    if zo >= z_out or yo >= y_out or xo >= x_out:
        return

    cz = zo * 2
    z0 = 0 if cz - 1 < 0 else cz - 1  # noqa: FURB136 - Python max() unsupported in Numba @cuda.jit
    z2 = z_in - 1 if cz + 1 >= z_in else cz + 1
    z3 = z_in - 1 if cz + 2 >= z_in else cz + 2

    three = np.float32(3.0)
    val = (xy_in[z0, yo, xo] + three * xy_in[cz, yo, xo] + three * xy_in[z2, yo, xo] + xy_in[z3, yo, xo]) * np.float32(
        1.0 / 512.0
    )
    out_f32[zo, yo, xo] = val
    out_cast[zo, yo, xo] = val


def _grid_3d(z: int, y: int, x: int) -> tuple[int, int, int]:
    tx, ty, tz = _THREADS_3D
    return (math.ceil(x / tx), math.ceil(y / ty), math.ceil(z / tz))


def _run_step_gpu(
    d_vol: Any,
    d_xy_scratch: Any,
    d_out_f32: Any,
    d_out_cast: Any,
    reduction: DownscaleType,
) -> None:
    """Execute one 2x 3D reduction step on the GPU."""
    zo, yo, xo = d_out_f32.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        if reduction == DownscaleType.GAUSSIAN:
            z_in = d_vol.shape[0]
            d_xy_view = d_xy_scratch[:z_in, :yo, :xo]
            _cuda_gauss_xy[_grid_3d(z_in, yo, xo), _THREADS_3D](d_vol, d_xy_view)  # type: ignore[index]
            _cuda_gauss_z_cast[_grid_3d(zo, yo, xo), _THREADS_3D](d_xy_view, d_out_f32, d_out_cast)  # type: ignore[index]
        elif reduction == DownscaleType.MEAN:
            _cuda_kernel_mean_cast[_grid_3d(zo, yo, xo), _THREADS_3D](d_vol, d_out_f32, d_out_cast)  # type: ignore[index]
        elif reduction == DownscaleType.MAX:
            _cuda_kernel_max_cast[_grid_3d(zo, yo, xo), _THREADS_3D](d_vol, d_out_f32, d_out_cast)  # type: ignore[index]
        elif reduction == DownscaleType.MIN:
            _cuda_kernel_min_cast[_grid_3d(zo, yo, xo), _THREADS_3D](d_vol, d_out_f32, d_out_cast)  # type: ignore[index]
        else:
            raise ValueError(f"Unsupported downscale type: {reduction}")


# ---------------------------------------------------------------------------
# Reusable GPU Pyramid Workspace
# ---------------------------------------------------------------------------


class GpuPyramidWorkspace:
    """Pre-allocated GPU device arena for computing 3D pyramids of a fixed maximum L0 shape and dtype.

    Holds:
    - ``d_levels[level]``: device arrays at ``dtype`` (e.g. ``uint8`` or ``uint16``) for ``L0..max_level``
    - ``d_f32_levels[level]``: ``float32`` device arrays for ``L1..max_level`` (preserves full float32
      precision across multi-step reductions and supports :func:`pyramids_3d_gpu`)
    - ``d_xy_scratch``: reusable ``float32`` buffer of shape ``(Z0, Y0 // 2, X0 // 2)`` for the separable
      Gaussian X+Y pass.
    """

    def __init__(self, shape_l0: UIVec3D, max_level: ScaleLevel, dtype: Dtype | np.dtype) -> None:
        self.shape_l0 = shape_l0
        self.max_level = max_level
        self.dtype = np.dtype(dtype.dtype if isinstance(dtype, Dtype) else dtype)

        self.d_levels: dict[ScaleLevel, Any] = {}
        self.d_f32_levels: dict[ScaleLevel, Any] = {}
        self.d_xy_scratch: Any = None
        self.ensure_device_buffers()

    def ensure_device_buffers(self) -> None:
        """Allocate device arrays if not currently resident in VRAM."""
        if self.d_levels:
            return
        for level in self.max_level.levels:
            shp = level.scale(self.shape_l0)
            shape = (max(1, shp.z), max(1, shp.y), max(1, shp.x))
            self.d_levels[level] = cuda.device_array(shape, dtype=self.dtype)
            if level != ScaleLevel.L0:
                self.d_f32_levels[level] = cuda.device_array(shape, dtype=np.float32)

        y_half = max(1, self.shape_l0.y // 2)
        x_half = max(1, self.shape_l0.x // 2)
        self.d_xy_scratch = cuda.device_array((max(1, self.shape_l0.z), y_half, x_half), dtype=np.float32)

    def release_device_buffers(self) -> None:
        """Release device array references so VRAM can be reclaimed between worker batches."""
        self.d_levels.clear()
        self.d_f32_levels.clear()
        self.d_xy_scratch = None

    def compute_from_host(
        self,
        block: np.ndarray,
        max_level: ScaleLevel,
        reduction: DownscaleType = DownscaleType.MEAN,
        *,
        copy_levels_to: dict[ScaleLevel, np.ndarray] | None = None,
        return_host_f32: bool = False,
    ) -> dict[ScaleLevel, Any]:
        """Transfer ``block`` (L0) to GPU, compute all pyramid levels up to ``max_level``, and optionally
        copy reduced levels back to host arrays.

        Args:
            block: Host L0 volume with shape ``(Z, Y, X)`` (where ``Z <= shape_l0.z``).
            max_level: Highest pyramid level to compute.
            reduction: Downscale method (``GAUSSIAN``, ``MEAN``, ``MAX``, ``MIN``).
            copy_levels_to: Optional mapping of ``ScaleLevel -> host np.ndarray`` (such as worker
                ``SharedMemory`` arrays) into which reduced ``dtype`` levels ``L1..max_level`` are copied.
            return_host_f32: If True, returns host ``float32`` arrays for ``L1..max_level`` (matching
                :func:`pyramids_3d_numba`). If False, returns device arrays at ``self.dtype`` for
                ``L0..max_level`` (for zero-copy consumption by :class:`GPUArrayWriter`).
        """
        if reduction not in _SUPPORTED_REDUCTIONS:
            raise ValueError(f"Unsupported downscale type: {reduction}")

        self.ensure_device_buffers()
        z0, y0, x0 = block.shape
        d_l0 = self.d_levels[ScaleLevel.L0][:z0, :y0, :x0]
        d_l0.copy_to_device(np.ascontiguousarray(block))

        levels = [lvl for lvl in max_level.levels if lvl != ScaleLevel.L0]
        sorted_levels = sorted(levels, key=lambda lvl: lvl.factor)

        current_factor = 1
        current_vol: Any = d_l0
        host_f32_results: dict[ScaleLevel, np.ndarray] = {}
        device_cast_results: dict[ScaleLevel, Any] = {ScaleLevel.L0: d_l0}

        for target_level in sorted_levels:
            target_factor = target_level.factor
            steps_needed = int(math.log2(target_factor)) - int(math.log2(current_factor))

            for _ in range(steps_needed):
                z, y, x = current_vol.shape
                z2, y2, x2 = (z // 2) * 2, (y // 2) * 2, (x // 2) * 2
                if min(z2, y2, x2) == 0:
                    break
                step_level = ScaleLevel(int(math.log2(current_factor * 2)))
                zo, yo, xo = z2 // 2, y2 // 2, x2 // 2
                d_out_f32 = self.d_f32_levels[step_level][:zo, :yo, :xo]
                d_out_cast = self.d_levels[step_level][:zo, :yo, :xo]
                _run_step_gpu(
                    current_vol[:z2, :y2, :x2],
                    self.d_xy_scratch,
                    d_out_f32,
                    d_out_cast,
                    reduction,
                )
                current_vol = d_out_f32
                current_factor *= 2

            z_out = z0 // target_factor
            y_out = y0 // target_factor
            x_out = x0 // target_factor
            d_lvl_cast = self.d_levels[target_level][:z_out, :y_out, :x_out]
            device_cast_results[target_level] = d_lvl_cast

            if copy_levels_to is not None and target_level in copy_levels_to:
                dst = copy_levels_to[target_level]
                zc = min(dst.shape[0], z_out)
                yc = min(dst.shape[1], y_out)
                xc = min(dst.shape[2], x_out)
                if zc > 0 and yc > 0 and xc > 0:
                    d_lvl_cast[:zc, :yc, :xc].copy_to_host(dst[:zc, :yc, :xc])

            if return_host_f32:
                d_lvl_f32 = current_vol[:z_out, :y_out, :x_out]
                host_f32_results[target_level] = d_lvl_f32.copy_to_host()

        cuda.synchronize()
        return host_f32_results if return_host_f32 else device_cast_results

    def close(self) -> None:
        """Release references to device buffers."""
        self.release_device_buffers()


def get_free_gpu_bytes() -> int:
    """Return currently free GPU VRAM in bytes, or 0 if CUDA is unavailable."""
    if not is_gpu_available():
        return 0
    try:
        free_b, _ = cuda.current_context().get_memory_info()
        return int(free_b)
    except Exception:
        return 0


def flush_gpu_deallocations() -> None:
    """Flush Numba's pending CUDA deallocations back to the driver so other worker processes can use the VRAM."""
    if not is_gpu_available():
        return
    try:
        ctx = cuda.current_context()
        ctx.memory_manager.deallocations.clear()
    except Exception:
        log.debug("Failed to flush CUDA deallocations", exc_info=True)


def warmup_gpu_pyramid(dtype: Dtype | np.dtype, reduction: DownscaleType = DownscaleType.GAUSSIAN) -> None:
    """Compile Numba CUDA pyramid kernels for ``dtype`` and ``reduction`` on a tiny 4x8x8 volume."""
    if not is_gpu_available():
        return
    dt = np.dtype(dtype.dtype if isinstance(dtype, Dtype) else dtype)
    d_in = cuda.device_array((4, 8, 8), dtype=dt)
    d_xy = cuda.device_array((4, 4, 4), dtype=np.float32)
    d_f32 = cuda.device_array((2, 4, 4), dtype=np.float32)
    d_cast = cuda.device_array((2, 4, 4), dtype=dt)
    _run_step_gpu(d_in, d_xy, d_f32, d_cast, reduction)
    if len({dt, np.dtype(np.float32)}) > 1:
        # Multi-level pyramids reduce L0 (dtype) -> L1 (float32) -> L2 (float32)
        d_f32_2 = cuda.device_array((1, 2, 2), dtype=np.float32)
        d_cast_2 = cuda.device_array((1, 2, 2), dtype=dt)
        _run_step_gpu(d_f32, d_xy[:2, :2, :2], d_f32_2, d_cast_2, reduction)
    cuda.synchronize()


_WORKSPACE_CACHE: dict[tuple[tuple[int, int, int], str, int], GpuPyramidWorkspace] = {}


def clear_gpu_workspace_cache() -> None:
    """Release any cached in-process GPU pyramid workspaces."""
    for ws in _WORKSPACE_CACHE.values():
        ws.close()
    _WORKSPACE_CACHE.clear()
    flush_gpu_deallocations()


def pyramids_3d_gpu(
    block: np.ndarray,
    max_level: ScaleLevel,
    reduction: DownscaleType = DownscaleType.MEAN,
) -> dict[ScaleLevel, np.ndarray]:
    """Compute a 3D multi-scale pyramid on the GPU via chained 2x2x2 reductions.

    Drop-in GPU counterpart to :func:`~ome_zarr_writer.pyramid.pyramids_3d_numba`: accepts any numeric
    dtype (``uint8``, ``uint16``, ``float32``, etc.) and returns a mapping of each non-L0 level to its
    reduced ``float32`` volume on the host.
    """
    if reduction not in _SUPPORTED_REDUCTIONS:
        raise ValueError(f"Unsupported downscale type: {reduction}")
    z, y, x = block.shape
    key = ((z, y, x), np.dtype(block.dtype).str, max_level.value)
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        # Keep at most 2 cached workspaces per process to bound VRAM in benchmarks/tests.
        if len(_WORKSPACE_CACHE) >= 2:
            clear_gpu_workspace_cache()
        ws = GpuPyramidWorkspace(UIVec3D(z=z, y=y, x=x), max_level, block.dtype)
        _WORKSPACE_CACHE[key] = ws
    return ws.compute_from_host(block, max_level, reduction=reduction, return_host_f32=True)
