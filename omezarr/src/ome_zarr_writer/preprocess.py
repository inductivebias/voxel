# pyright: reportArgumentType=false
"""2D spatial pre-filtering and rectangular binning (CPU & CUDA)."""

import math
import warnings
from typing import Any

import numba
import numpy as np
from numba import cuda, prange
from numba.core.errors import NumbaPerformanceWarning
from pydantic import ConfigDict, Field
from vxlib.schema import SparseModel

from ome_zarr_writer.dataset import BinningMode, PreFilterType

_PREFILTER_TO_CODE: dict[PreFilterType, int] = {
    PreFilterType.NONE: 0,
    PreFilterType.MEAN: 1,
    PreFilterType.GAUSSIAN: 2,
    PreFilterType.MEDIAN: 3,
    PreFilterType.HIGHPASS: 4,
}
_BINMODE_TO_CODE: dict[BinningMode, int] = {
    BinningMode.MEAN: 0,
    BinningMode.MEDIAN: 1,
    BinningMode.SUM: 2,
    BinningMode.MAX: 3,
    BinningMode.NEAREST: 4,
}
_THREADS_3D = (32, 8, 1)


class PreprocessConfig(SparseModel):
    """Frozen preprocessing & binning parameters shared across writer, slot, and GPU workspace."""

    model_config = ConfigDict(frozen=True)

    bin_y: int = Field(default=1, ge=1, le=16)
    bin_x: int = Field(default=1, ge=1, le=16)
    pre_filter: PreFilterType = PreFilterType.NONE
    filter_size: int = Field(default=3, ge=3, le=5)
    bin_mode: BinningMode = BinningMode.MEAN
    highpass_add: float = 100.0

    @property
    def has_spatial_op(self) -> bool:
        return self.bin_y > 1 or self.bin_x > 1 or self.pre_filter != PreFilterType.NONE

    def to_metadata(self) -> dict[str, Any] | None:
        if not self.has_spatial_op:
            return None
        meta: dict[str, Any] = {}
        if self.bin_y > 1 or self.bin_x > 1:
            meta.update(bin_y=self.bin_y, bin_x=self.bin_x, bin_mode=self.bin_mode.value)
        if self.pre_filter != PreFilterType.NONE:
            meta.update(pre_filter=self.pre_filter.value, filter_size=self.filter_size)
            if self.pre_filter == PreFilterType.HIGHPASS:
                meta["highpass_add"] = self.highpass_add
        return meta or None


def _dtype_int_bounds(dt: np.dtype) -> tuple[int, float]:
    if dt.kind in ("u", "i"):
        return 1, float(np.iinfo(dt).max)
    return 0, float("inf")


def _grid_3d(z: int, y: int, x: int) -> tuple[int, int, int]:
    tx, ty, tz = _THREADS_3D
    return (math.ceil(x / tx), math.ceil(y / ty), math.ceil(z / tz))


def _cuda_ready(use_gpu: bool | None) -> bool:
    if use_gpu is False:
        return False
    try:
        return bool(cuda.is_available() and len(cuda.gpus) > 0)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Scalar & device helpers shared by CPU and CUDA kernels
# ---------------------------------------------------------------------------


def _make_base_ops(jit_dec: Any) -> tuple[Any, Any, Any]:
    @jit_dec
    def _clamp(idx: int, hi: int) -> int:
        if idx < 0:
            return 0
        return hi - 1 if idx >= hi else idx

    @jit_dec
    def _sort_f32(buf: Any, count: int) -> None:
        for i in range(1, count):
            key = buf[i]
            j = i - 1
            while j >= 0 and buf[j] > key:
                buf[j + 1] = buf[j]
                j -= 1
            buf[j + 1] = key

    @jit_dec
    def _gauss_w(offset: int, filter_size: int) -> float:
        a = -offset if offset < 0 else offset
        if filter_size == 3:
            return 2.0 if a == 0 else 1.0
        return 6.0 if a == 0 else (4.0 if a == 1 else 1.0)

    return _clamp, _sort_f32, _gauss_w


def _make_pixel_eval(jit_dec: Any, clamp: Any, sort_f32: Any, gauss_w: Any) -> Any:
    @jit_dec
    def _eval_pixel(
        vol_in: Any,
        z: int,
        py: int,
        px: int,
        filter_mode: int,
        filter_size: int,
        highpass_add: float,
        med_buf: Any,
    ) -> float:
        if filter_mode == 0:
            return float(vol_in[z, py, px])
        _, y_in, x_in = vol_in.shape
        radius = filter_size // 2
        if filter_mode == 1:
            acc = 0.0
            for ky in range(-radius, radius + 1):
                sy = clamp(py + ky, y_in)
                for kx in range(-radius, radius + 1):
                    acc += float(vol_in[z, sy, clamp(px + kx, x_in)])
            return acc / float(filter_size * filter_size)
        if filter_mode in (2, 4):
            w1d = 4.0 if filter_size == 3 else 16.0
            acc = 0.0
            for ky in range(-radius, radius + 1):
                sy = clamp(py + ky, y_in)
                wy = gauss_w(ky, filter_size)
                for kx in range(-radius, radius + 1):
                    acc += float(vol_in[z, sy, clamp(px + kx, x_in)]) * wy * gauss_w(kx, filter_size)
            blur = acc / (w1d * w1d)
            return blur if filter_mode == 2 else (float(vol_in[z, py, px]) - blur + highpass_add)
        idx = 0
        for ky in range(-radius, radius + 1):
            sy = clamp(py + ky, y_in)
            for kx in range(-radius, radius + 1):
                med_buf[idx] = np.float32(vol_in[z, sy, clamp(px + kx, x_in)])
                idx += 1
        sort_f32(med_buf, idx)
        return float(med_buf[idx // 2])

    return _eval_pixel


def _make_block_reducer(jit_dec: Any) -> Any:
    clamp, sort_f32, gauss_w = _make_base_ops(jit_dec)
    eval_pixel = _make_pixel_eval(jit_dec, clamp, sort_f32, gauss_w)

    @jit_dec
    def _reduce_block(
        vol_in: Any,
        z: int,
        yo: int,
        xo: int,
        bin_y: int,
        bin_x: int,
        filter_mode: int,
        filter_size: int,
        bin_mode: int,
        highpass_add: float,
        is_int: int,
        max_val: float,
        med_buf: Any,
        bin_buf: Any,
    ) -> float:
        dy0 = (bin_y - 1) // 2 if bin_mode == 4 else 0
        dy1 = dy0 + 1 if bin_mode == 4 else bin_y
        dx0 = (bin_x - 1) // 2 if bin_mode == 4 else 0
        dx1 = dx0 + 1 if bin_mode == 4 else bin_x
        block_sum = 0.0
        block_max = -1e30
        b_count = 0
        for dy in range(dy0, dy1):
            for dx in range(dx0, dx1):
                val = eval_pixel(
                    vol_in, z, yo * bin_y + dy, xo * bin_x + dx, filter_mode, filter_size, highpass_add, med_buf
                )
                block_sum += val
                if val > block_max:  # noqa: PLR1730
                    block_max = val
                if bin_mode == 1 and b_count < bin_buf.shape[0]:
                    bin_buf[b_count] = np.float32(val)
                    b_count += 1
        if bin_mode == 0:
            res = block_sum / float(bin_y * bin_x)
        elif bin_mode == 1:
            sort_f32(bin_buf, b_count)
            mid = b_count // 2
            res = float(bin_buf[mid]) if b_count % 2 == 1 else 0.5 * (float(bin_buf[mid - 1]) + float(bin_buf[mid]))
        elif bin_mode == 3:
            res = block_max
        else:
            res = block_sum
        if is_int == 1:
            r = math.floor(res + 0.5)
            return 0.0 if r < 0.0 else (max_val if r > max_val else r)  # noqa: FURB136
        return res

    return _reduce_block


_cpu_reduce_block = _make_block_reducer(numba.njit)
_cuda_reduce_block = _make_block_reducer(cuda.jit(device=True, fastmath=True))


@cuda.jit(fastmath=True)
def _cuda_prefilter_and_bin_kernel(
    vol_in: Any,
    vol_out: Any,
    bin_y: int,
    bin_x: int,
    filter_mode: int,
    filter_size: int,
    bin_mode: int,
    highpass_add: float,
    is_int: int,
    max_val: float,
) -> None:
    xo, yo, z = cuda.grid(3)  # type: ignore[misc]
    z_out, y_out, x_out = vol_out.shape
    if z >= z_out or yo >= y_out or xo >= x_out:
        return
    med_buf = cuda.local.array(25, numba.float32)  # type: ignore[misc]
    bin_buf = cuda.local.array(256, numba.float32)  # type: ignore[misc]  # supports up to 16x16 binning
    vol_out[z, yo, xo] = _cuda_reduce_block(
        vol_in,
        z,
        yo,
        xo,
        bin_y,
        bin_x,
        filter_mode,
        filter_size,
        bin_mode,
        highpass_add,
        is_int,
        max_val,
        med_buf,
        bin_buf,
    )


@numba.njit(parallel=True, cache=True)
def _cpu_prefilter_and_bin_kernel(
    vol_in: np.ndarray,
    vol_out: np.ndarray,
    bin_y: int,
    bin_x: int,
    filter_mode: int,
    filter_size: int,
    bin_mode: int,
    highpass_add: float,
    is_int: int,
    max_val: float,
) -> None:
    z_out, y_out, x_out = vol_out.shape
    for z in prange(z_out):
        med_buf = np.empty(25, dtype=np.float32)
        bin_buf = np.empty(max(64, bin_y * bin_x), dtype=np.float32)
        for yo in range(y_out):
            for xo in range(x_out):
                vol_out[z, yo, xo] = _cpu_reduce_block(
                    vol_in,
                    z,
                    yo,
                    xo,
                    bin_y,
                    bin_x,
                    filter_mode,
                    filter_size,
                    bin_mode,
                    highpass_add,
                    is_int,
                    max_val,
                    med_buf,
                    bin_buf,
                )


# ---------------------------------------------------------------------------
# Public host & device-dispatch entry points
# ---------------------------------------------------------------------------


def filter_and_bin_volume(
    vol: np.ndarray,
    *,
    bin_y: int = 1,
    bin_x: int = 1,
    pre_filter: PreFilterType | str = PreFilterType.NONE,
    filter_size: int = 3,
    bin_mode: BinningMode | str = BinningMode.MEAN,
    highpass_add: float = 100.0,
    use_gpu: bool | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a 2D spatial pre-filter and rectangular ``(bin_y, bin_x)`` reduction to a 3D volume ``(Z, Y, X)``."""
    pf = PreFilterType(pre_filter) if not isinstance(pre_filter, PreFilterType) else pre_filter
    bm = BinningMode(bin_mode) if not isinstance(bin_mode, BinningMode) else bin_mode
    if bin_y < 1 or bin_x < 1:
        raise ValueError(f"bin_y and bin_x must be >= 1, got ({bin_y}, {bin_x})")
    if filter_size not in (3, 5):
        raise ValueError(f"filter_size must be 3 or 5, got {filter_size}")
    if bin_y == 1 and bin_x == 1 and pf == PreFilterType.NONE:
        if out is not None:
            np.copyto(out, vol)
            return out
        return vol
    z, y_in, x_in = vol.shape
    y_out, x_out = y_in // bin_y, x_in // bin_x
    if y_out < 1 or x_out < 1:
        raise ValueError(f"Input shape {(y_in, x_in)} is smaller than bin factors {(bin_y, bin_x)}")
    if out is None:
        out = np.empty((z, y_out, x_out), dtype=vol.dtype)
    elif out.shape != (z, y_out, x_out):
        raise ValueError(f"out shape {out.shape} does not match expected {(z, y_out, x_out)}")
    args = (
        int(bin_y),
        int(bin_x),
        _PREFILTER_TO_CODE[pf],
        int(filter_size),
        _BINMODE_TO_CODE[bm],
        float(highpass_add),
        *_dtype_int_bounds(vol.dtype),
    )
    if _cuda_ready(use_gpu):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            d_in = cuda.to_device(np.ascontiguousarray(vol))
            d_out = cuda.device_array((z, y_out, x_out), dtype=vol.dtype)
            _cuda_prefilter_and_bin_kernel[_grid_3d(z, y_out, x_out), _THREADS_3D](d_in, d_out, *args)  # type: ignore[index]
            d_out.copy_to_host(out)
        return out
    _cpu_prefilter_and_bin_kernel(np.ascontiguousarray(vol), out, *args)
    return out


def filter_and_bin_2d(
    frame: np.ndarray,
    *,
    bin_y: int = 1,
    bin_x: int = 1,
    pre_filter: PreFilterType | str = PreFilterType.NONE,
    filter_size: int = 3,
    bin_mode: BinningMode | str = BinningMode.MEAN,
    highpass_add: float = 100.0,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Apply a 2D spatial pre-filter and rectangular ``(bin_y, bin_x)`` reduction to a 2D frame ``(Y, X)``."""
    pf = PreFilterType(pre_filter) if not isinstance(pre_filter, PreFilterType) else pre_filter
    bm = BinningMode(bin_mode) if not isinstance(bin_mode, BinningMode) else bin_mode
    if bin_y == 1 and bin_x == 1 and pf == PreFilterType.NONE:
        return frame
    return filter_and_bin_volume(
        frame[np.newaxis, ...],
        bin_y=bin_y,
        bin_x=bin_x,
        pre_filter=pf,
        filter_size=filter_size,
        bin_mode=bm,
        highpass_add=highpass_add,
        use_gpu=use_gpu,
    )[0]
