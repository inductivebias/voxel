# pyright: reportArgumentType=false
"""GPU-accelerated 3D multi-scale pyramid generation and CUDA runtime helpers using Numba CUDA."""

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
from ome_zarr_writer.preprocess import (
    _BINMODE_TO_CODE,
    _PREFILTER_TO_CODE,
    _THREADS_3D,
    PreprocessConfig,
    _cuda_prefilter_and_bin_kernel,
    _dtype_int_bounds,
    _grid_3d,
)

log = logging.getLogger(__name__)

_BLOCK_MODES: dict[DownscaleType, int] = {DownscaleType.MEAN: 0, DownscaleType.MAX: 1, DownscaleType.MIN: 2}
_SUPPORTED_REDUCTIONS = frozenset({*list(_BLOCK_MODES), DownscaleType.GAUSSIAN})
_CUDART_STATE: list[ctypes.CDLL | None] = []  # one-element cache avoids global rebind


def is_gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is available to Numba CUDA."""
    try:
        return bool(cuda.is_available() and len(cuda.gpus) > 0)
    except Exception:
        return False


def _get_cudart() -> ctypes.CDLL | None:
    if _CUDART_STATE:
        return _CUDART_STATE[0]
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            lib = ctypes.CDLL(name)
            lib.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            lib.cudaHostRegister.restype = ctypes.c_int
            lib.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            lib.cudaHostUnregister.restype = ctypes.c_int
            _CUDART_STATE.append(lib)
            return lib
        except OSError:
            continue
    _CUDART_STATE.append(None)
    return None


def register_pinned_host_memory(arr: np.ndarray) -> bool:
    """Page-lock a contiguous host array via ``cudaHostRegister`` (best-effort)."""
    if arr.nbytes <= 0 or not arr.flags.c_contiguous or not is_gpu_available():
        return False
    lib = _get_cudart()
    if lib is None:
        return False
    try:
        cuda.current_context()
        return int(lib.cudaHostRegister(ctypes.c_void_p(arr.ctypes.data), ctypes.c_size_t(arr.nbytes), 0)) == 0
    except Exception:
        return False


def unregister_pinned_host_memory(arr: np.ndarray) -> None:
    """Unpin a previously page-locked host array before closing its ``SharedMemory``."""
    lib = _get_cudart()
    if lib is None or arr.nbytes <= 0:
        return
    try:
        lib.cudaHostUnregister(ctypes.c_void_p(arr.ctypes.data))
    except Exception:
        log.debug("cudaHostUnregister failed", exc_info=True)


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
    """Flush Numba's pending CUDA deallocations back to the driver."""
    if not is_gpu_available():
        return
    try:
        cuda.current_context().memory_manager.deallocations.clear()
    except Exception:
        log.debug("Failed to flush CUDA deallocations", exc_info=True)


# ---------------------------------------------------------------------------
# CUDA Pyramid Kernels (unified MEAN/MAX/MIN + separable [1, 3, 3, 1] GAUSSIAN)
# ---------------------------------------------------------------------------


@cuda.jit(fastmath=True)
def _cuda_kernel_block_cast(vol: Any, out_f32: Any, out_cast: Any, mode: int) -> None:
    xo, yo, zo = cuda.grid(3)  # type: ignore[misc]
    z_out, y_out, x_out = out_f32.shape
    if zo >= z_out or yo >= y_out or xo >= x_out:
        return
    t2, i2, j2 = zo * 2, yo * 2, xo * 2
    acc = np.float32(0.0) if mode == 0 else np.float32(vol[t2, i2, j2])
    for dt in range(2):
        for di in range(2):
            for dj in range(2):
                v = np.float32(vol[t2 + dt, i2 + di, j2 + dj])
                if mode == 0:
                    acc += v
                elif mode == 1:
                    acc = v if v > acc else acc  # noqa: FURB136
                else:
                    acc = v if v < acc else acc  # noqa: FURB136
    val = acc * np.float32(0.125) if mode == 0 else acc
    out_f32[zo, yo, xo] = val
    out_cast[zo, yo, xo] = val


@cuda.jit(device=True, fastmath=True)
def _clamp_idx(idx: int, hi: int) -> int:
    if idx < 0:
        return 0
    return hi - 1 if idx >= hi else idx


@cuda.jit(fastmath=True)
def _cuda_gauss_xy(vol: Any, xy_out: Any) -> None:
    """Fused 2D separable [1, 3, 3, 1] X+Y pass per Z-plane (un-normalized float32 sum)."""
    xo, yo, z = cuda.grid(3)  # type: ignore[misc]
    _, y_in, x_in = vol.shape
    z_out, y_out, x_out = xy_out.shape
    if z >= z_out or yo >= y_out or xo >= x_out:
        return
    cx, cy = xo * 2, yo * 2
    three = np.float32(3.0)
    total = np.float32(0.0)
    for ky in range(4):
        wy = three if ky in (1, 2) else np.float32(1.0)
        sy = _clamp_idx(cy + ky - 1, y_in)
        row = np.float32(0.0)
        for kx in range(4):
            wx = three if kx in (1, 2) else np.float32(1.0)
            row += wx * np.float32(vol[z, sy, _clamp_idx(cx + kx - 1, x_in)])
        total += wy * row
    xy_out[z, yo, xo] = total


@cuda.jit(fastmath=True)
def _cuda_gauss_z_cast(xy_in: Any, out_f32: Any, out_cast: Any) -> None:
    """Z pass: combine four X/Y-reduced planes with [1, 3, 3, 1] * (1/512)."""
    xo, yo, zo = cuda.grid(3)  # type: ignore[misc]
    z_in = xy_in.shape[0]
    z_out, y_out, x_out = out_f32.shape
    if zo >= z_out or yo >= y_out or xo >= x_out:
        return
    cz = zo * 2
    three = np.float32(3.0)
    z0 = _clamp_idx(cz - 1, z_in)
    z2 = _clamp_idx(cz + 1, z_in)
    z3 = _clamp_idx(cz + 2, z_in)
    val = (xy_in[z0, yo, xo] + three * xy_in[cz, yo, xo] + three * xy_in[z2, yo, xo] + xy_in[z3, yo, xo]) * np.float32(
        1.0 / 512.0
    )
    out_f32[zo, yo, xo] = val
    out_cast[zo, yo, xo] = val


def _run_step_gpu(d_vol: Any, d_xy_scratch: Any, d_out_f32: Any, d_out_cast: Any, reduction: DownscaleType) -> None:
    zo, yo, xo = d_out_f32.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        if reduction == DownscaleType.GAUSSIAN:
            z_in = d_vol.shape[0]
            d_xy_view = d_xy_scratch[:z_in, :yo, :xo]
            _cuda_gauss_xy[_grid_3d(z_in, yo, xo), _THREADS_3D](d_vol, d_xy_view)  # type: ignore[index]
            _cuda_gauss_z_cast[_grid_3d(zo, yo, xo), _THREADS_3D](d_xy_view, d_out_f32, d_out_cast)  # type: ignore[index]
        elif reduction in _BLOCK_MODES:
            _cuda_kernel_block_cast[_grid_3d(zo, yo, xo), _THREADS_3D](  # type: ignore[index]
                d_vol, d_out_f32, d_out_cast, _BLOCK_MODES[reduction]
            )
        else:
            raise ValueError(f"Unsupported downscale type: {reduction}")


# ---------------------------------------------------------------------------
# Reusable GPU Pyramid Workspace
# ---------------------------------------------------------------------------


class GpuPyramidWorkspace:
    """Pre-allocated GPU device arena for computing 3D pyramids of a fixed maximum L0 shape and dtype."""

    def __init__(
        self,
        shape_l0: UIVec3D,
        max_level: ScaleLevel,
        dtype: Dtype | np.dtype,
        raw_shape_l0: UIVec3D | None = None,
    ) -> None:
        self.shape_l0 = shape_l0
        self.raw_shape_l0 = raw_shape_l0
        self.max_level = max_level
        self.dtype = np.dtype(dtype.dtype if isinstance(dtype, Dtype) else dtype)
        self.d_raw: Any = None
        self.d_levels: dict[ScaleLevel, Any] = {}
        self.d_f32_levels: dict[ScaleLevel, Any] = {}
        self.d_xy_scratch: Any = None
        self.ensure_device_buffers()

    def ensure_device_buffers(self) -> None:
        if self.d_levels:
            return
        if self.raw_shape_l0 is not None and self.raw_shape_l0 != self.shape_l0:
            r = self.raw_shape_l0
            self.d_raw = cuda.device_array((max(1, r.z), max(1, r.y), max(1, r.x)), dtype=self.dtype)
        for level in self.max_level.levels:
            s = level.scale(self.shape_l0)
            shape = (max(1, s.z), max(1, s.y), max(1, s.x))
            self.d_levels[level] = cuda.device_array(shape, dtype=self.dtype)
            if level != ScaleLevel.L0:
                self.d_f32_levels[level] = cuda.device_array(shape, dtype=np.float32)
        self.d_xy_scratch = cuda.device_array(
            (max(1, self.shape_l0.z), max(1, self.shape_l0.y // 2), max(1, self.shape_l0.x // 2)), dtype=np.float32
        )

    def release_device_buffers(self) -> None:
        self.d_raw = None
        self.d_levels.clear()
        self.d_f32_levels.clear()
        self.d_xy_scratch = None

    def _upload_and_preprocess_l0(self, block: np.ndarray, pp: PreprocessConfig) -> Any:
        if pp.has_spatial_op:
            is_int, max_val_f = _dtype_int_bounds(self.dtype)
            z_raw, y_raw, x_raw = block.shape
            z0, y0, x0 = z_raw, y_raw // pp.bin_y, x_raw // pp.bin_x
            if self.d_raw is None or any(self.d_raw.shape[i] < d for i, d in enumerate((z_raw, y_raw, x_raw))):
                self.d_raw = cuda.device_array((z_raw, y_raw, x_raw), dtype=self.dtype)
            d_raw_view = self.d_raw[:z_raw, :y_raw, :x_raw]
            d_raw_view.copy_to_device(np.ascontiguousarray(block))
            d_l0 = self.d_levels[ScaleLevel.L0][:z0, :y0, :x0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NumbaPerformanceWarning)
                _cuda_prefilter_and_bin_kernel[_grid_3d(z0, y0, x0), _THREADS_3D](  # type: ignore[index]
                    d_raw_view,
                    d_l0,
                    pp.bin_y,
                    pp.bin_x,
                    _PREFILTER_TO_CODE[pp.pre_filter],
                    pp.filter_size,
                    _BINMODE_TO_CODE[pp.bin_mode],
                    float(pp.highpass_add),
                    is_int,
                    max_val_f,
                )
        else:
            z0, y0, x0 = block.shape
            d_l0 = self.d_levels[ScaleLevel.L0][:z0, :y0, :x0]
            d_l0.copy_to_device(np.ascontiguousarray(block))
        return d_l0

    def compute_from_host(
        self,
        block: np.ndarray,
        max_level: ScaleLevel,
        reduction: DownscaleType = DownscaleType.MEAN,
        *,
        preprocess: PreprocessConfig | None = None,
        copy_levels_to: dict[ScaleLevel, np.ndarray] | None = None,
        copy_l0_to: np.ndarray | None = None,
        return_host_f32: bool = False,
    ) -> dict[ScaleLevel, Any]:
        """Upload ``block``, apply optional preprocessing, build pyramid levels, and optionally copy to host."""
        if reduction not in _SUPPORTED_REDUCTIONS:
            raise ValueError(f"Unsupported downscale type: {reduction}")
        self.ensure_device_buffers()
        d_l0 = self._upload_and_preprocess_l0(block, preprocess or PreprocessConfig())
        z0, y0, x0 = d_l0.shape

        if copy_l0_to is not None:
            zc, yc, xc = min(copy_l0_to.shape[0], z0), min(copy_l0_to.shape[1], y0), min(copy_l0_to.shape[2], x0)
            if zc > 0 and yc > 0 and xc > 0:
                d_l0[:zc, :yc, :xc].copy_to_host(copy_l0_to[:zc, :yc, :xc])

        current_factor = 1
        current_vol: Any = d_l0
        host_f32: dict[ScaleLevel, np.ndarray] = {}
        device_cast: dict[ScaleLevel, Any] = {ScaleLevel.L0: d_l0}
        sorted_levels = sorted((lvl for lvl in max_level.levels if lvl != ScaleLevel.L0), key=lambda lvl: lvl.factor)
        for target_level in sorted_levels:
            steps = int(math.log2(target_level.factor)) - int(math.log2(current_factor))
            for _ in range(steps):
                z, y, x = current_vol.shape
                z2, y2, x2 = (z // 2) * 2, (y // 2) * 2, (x // 2) * 2
                if min(z2, y2, x2) == 0:
                    break
                step_level = ScaleLevel(int(math.log2(current_factor * 2)))
                zo, yo, xo = z2 // 2, y2 // 2, x2 // 2
                d_out_f32 = self.d_f32_levels[step_level][:zo, :yo, :xo]
                _run_step_gpu(
                    current_vol[:z2, :y2, :x2],
                    self.d_xy_scratch,
                    d_out_f32,
                    self.d_levels[step_level][:zo, :yo, :xo],
                    reduction,
                )
                current_vol = d_out_f32
                current_factor *= 2

            zo, yo, xo = z0 // target_level.factor, y0 // target_level.factor, x0 // target_level.factor
            d_lvl_cast = self.d_levels[target_level][:zo, :yo, :xo]
            device_cast[target_level] = d_lvl_cast
            if copy_levels_to is not None and target_level in copy_levels_to:
                dst = copy_levels_to[target_level]
                zc, yc, xc = min(dst.shape[0], zo), min(dst.shape[1], yo), min(dst.shape[2], xo)
                if zc > 0 and yc > 0 and xc > 0:
                    d_lvl_cast[:zc, :yc, :xc].copy_to_host(dst[:zc, :yc, :xc])
            if return_host_f32:
                host_f32[target_level] = current_vol[:zo, :yo, :xo].copy_to_host()

        cuda.synchronize()
        return host_f32 if return_host_f32 else device_cast

    def close(self) -> None:
        self.release_device_buffers()


def warmup_gpu_pyramid(dtype: Dtype | np.dtype, reduction: DownscaleType = DownscaleType.GAUSSIAN) -> None:
    """Compile Numba CUDA pyramid & preprocessing kernels on a tiny 4x8x8 volume."""
    if not is_gpu_available():
        return
    dt = np.dtype(dtype.dtype if isinstance(dtype, Dtype) else dtype)
    d_in = cuda.device_array((4, 8, 8), dtype=dt)
    d_xy = cuda.device_array((4, 4, 4), dtype=np.float32)
    d_f32 = cuda.device_array((2, 4, 4), dtype=np.float32)
    d_cast = cuda.device_array((2, 4, 4), dtype=dt)
    _run_step_gpu(d_in, d_xy, d_f32, d_cast, reduction)
    is_int, max_val_f = _dtype_int_bounds(dt)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        _cuda_prefilter_and_bin_kernel[_grid_3d(4, 4, 4), _THREADS_3D](  # type: ignore[index]
            d_in, d_in[:, :4, :4], 2, 2, 0, 3, 0, 100.0, is_int, max_val_f
        )
    if dt != np.dtype(np.float32):
        _run_step_gpu(
            d_f32,
            d_xy[:2, :2, :2],
            cuda.device_array((1, 2, 2), dtype=np.float32),
            cuda.device_array((1, 2, 2), dtype=dt),
            reduction,
        )
    cuda.synchronize()


_WORKSPACE_CACHE: dict[tuple[tuple[int, int, int], str, int], GpuPyramidWorkspace] = {}


def clear_gpu_workspace_cache() -> None:
    """Release cached in-process GPU pyramid workspaces."""
    for ws in _WORKSPACE_CACHE.values():
        ws.close()
    _WORKSPACE_CACHE.clear()
    flush_gpu_deallocations()


def pyramids_3d_gpu(
    block: np.ndarray,
    max_level: ScaleLevel,
    reduction: DownscaleType = DownscaleType.MEAN,
) -> dict[ScaleLevel, np.ndarray]:
    """Compute a 3D multi-scale pyramid on the GPU; drop-in counterpart to ``pyramids_3d_numba``."""
    if reduction not in _SUPPORTED_REDUCTIONS:
        raise ValueError(f"Unsupported downscale type: {reduction}")
    z, y, x = block.shape
    key = ((z, y, x), np.dtype(block.dtype).str, max_level.value)
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        if len(_WORKSPACE_CACHE) >= 2:
            clear_gpu_workspace_cache()
        ws = GpuPyramidWorkspace(UIVec3D(z=z, y=y, x=x), max_level, block.dtype)
        _WORKSPACE_CACHE[key] = ws
    return ws.compute_from_host(
        block,
        max_level,
        reduction=reduction,
        return_host_f32=True,
    )
