# pyright: reportArgumentType=false
"""2D spatial pre-filtering, rectangular binning, and noise-bounded lossy quantization (CPU & CUDA)."""

import math
import warnings
from typing import Any

import numba
import numpy as np
from numba import cuda, prange
from numba.core.errors import NumbaPerformanceWarning
from pydantic import ConfigDict, Field
from vxlib.schema import SparseModel

from ome_zarr_writer.dataset import BinningMode, LossyDomain, PreFilterType

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

    keep_bits: int | None = Field(default=None, ge=1, le=16)
    dark_floor: int = Field(default=0, ge=0)
    bin_y: int = Field(default=1, ge=1, le=16)
    bin_x: int = Field(default=1, ge=1, le=16)
    pre_filter: PreFilterType = PreFilterType.NONE
    filter_size: int = Field(default=3, ge=3, le=5)
    bin_mode: BinningMode = BinningMode.MEAN
    highpass_add: float = 100.0
    lossy_step_noise: float | None = Field(default=None, ge=0.0, le=10.0)
    lossy_domain: LossyDomain = LossyDomain.SQRT
    noise_sigma: float | None = Field(default=None, ge=0.0)

    @property
    def has_spatial_op(self) -> bool:
        return self.bin_y > 1 or self.bin_x > 1 or self.pre_filter != PreFilterType.NONE

    @property
    def has_uint_op(self) -> bool:
        return self.dark_floor > 0 or self.keep_bits is not None

    @property
    def has_lossy(self) -> bool:
        return (
            self.lossy_step_noise is not None
            and self.lossy_step_noise > 0.0
            and self.lossy_domain != LossyDomain.NONE
        )

    @property
    def has_any(self) -> bool:
        return self.has_spatial_op or self.has_uint_op or self.has_lossy

    def to_metadata(self) -> dict[str, Any] | None:
        if not self.has_any:
            return None
        meta: dict[str, Any] = {}
        if self.bin_y > 1 or self.bin_x > 1:
            meta.update(bin_y=self.bin_y, bin_x=self.bin_x, bin_mode=self.bin_mode.value)
        if self.pre_filter != PreFilterType.NONE:
            meta.update(pre_filter=self.pre_filter.value, filter_size=self.filter_size)
            if self.pre_filter == PreFilterType.HIGHPASS:
                meta["highpass_add"] = self.highpass_add
        if self.dark_floor > 0:
            meta["dark_floor"] = self.dark_floor
        if self.keep_bits is not None:
            meta["keep_bits"] = self.keep_bits
        if self.has_lossy:
            meta.update(lossy_step_noise=self.lossy_step_noise, lossy_domain=self.lossy_domain.value)
            if self.noise_sigma is not None:
                meta["noise_sigma"] = self.noise_sigma
        return meta or None


def cutoff_to_sigma_px(cutoff: float, voxel_size_um: float = 1.0, units: str = "um") -> float:
    """Convert spatial cut-off ``cutoff`` (um or voxels) to Gaussian std-dev in voxels."""
    if cutoff <= 0.0:
        return 0.0
    c_vox = (cutoff / max(voxel_size_um, 1e-9)) if units == "um" else float(cutoff)
    return float(math.sqrt(math.log(2.0) / 2.0) / math.pi * c_vox)


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


def estimate_noise_sigma(
    arr: np.ndarray,
    *,
    domain: LossyDomain | str = LossyDomain.SQRT,
    offset: float | None = None,
    max_sample_slices: int = 8,
) -> tuple[float, float]:
    """Estimate noise std-dev from MAD of first differences along X; returns ``(sigma, offset)``."""
    dom = LossyDomain(domain) if not isinstance(domain, LossyDomain) else domain
    if arr.size < 2 or dom == LossyDomain.NONE:
        return 0.0, 0.0
    sample = arr
    if sample.ndim >= 3 and sample.shape[0] > max_sample_slices:
        sample = sample[:: max(1, sample.shape[0] // max_sample_slices)]
    if sample.ndim >= 2 and sample.shape[-2] > 512:
        sample = sample[..., ::2, :]
    sample_f32 = sample.astype(np.float32, copy=False)
    if dom == LossyDomain.SQRT:
        off = float(np.percentile(sample_f32, 0.1)) if offset is None else float(offset)
        transformed = 2.0 * np.sqrt(np.maximum(0.0, sample_f32 - off) + 0.375)
    else:
        off = 0.0 if offset is None else float(offset)
        transformed = sample_f32
    if transformed.shape[-1] < 2:
        return 0.0, off
    diffs = np.diff(transformed, axis=-1)
    med = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - med)))
    return float(1.4826 * mad / math.sqrt(2.0)), off


# ---------------------------------------------------------------------------
# Scalar & device helpers shared by CPU and CUDA kernels
# ---------------------------------------------------------------------------


def _make_quantize_scalar(jit_dec: Any) -> Any:
    @jit_dec
    def _quant(val: float, domain_code: int, step: float, offset: float, is_int: int, max_val: float) -> float:
        v = np.float32(val)
        step_f = np.float32(step)
        off_f = np.float32(offset)
        if domain_code == 1:
            shifted = v - off_f
            if shifted < np.float32(0.0):  # noqa: PLR1730
                shifted = np.float32(0.0)
            t = np.float32(2.0) * math.sqrt(shifted + np.float32(0.375))
            tq = math.floor(t / step_f + np.float32(0.5)) * step_f
            rec = np.float32(0.25) * tq * tq - np.float32(0.375) + off_f
        else:
            rec = math.floor(v / step_f + np.float32(0.5)) * step_f
        if is_int == 1:
            r = math.floor(rec + np.float32(0.5))
            if r < 0.0:
                return 0.0
            if r > max_val:
                return max_val
            return r
        return float(rec)

    return _quant


_cpu_quant = _make_quantize_scalar(numba.njit)
_cuda_quant = _make_quantize_scalar(cuda.jit(device=True, fastmath=True))


@cuda.jit(fastmath=True)
def _cuda_lossy_quantize_kernel(
    arr: Any, domain_code: int, step: float, offset: float, is_int: int, max_val: float
) -> None:
    x, y, z = cuda.grid(3)  # type: ignore[misc]
    z_dim, y_dim, x_dim = arr.shape
    if z < z_dim and y < y_dim and x < x_dim:
        arr[z, y, x] = _cuda_quant(arr[z, y, x], domain_code, step, offset, is_int, max_val)


@numba.njit(parallel=True, cache=True)
def _cpu_lossy_quantize_kernel(
    arr: np.ndarray, domain_code: int, step: float, offset: float, is_int: int, max_val: float
) -> None:
    z_dim, y_dim, x_dim = arr.shape
    for z in prange(z_dim):
        for y in range(y_dim):
            for x in range(x_dim):
                arr[z, y, x] = _cpu_quant(arr[z, y, x], domain_code, step, offset, is_int, max_val)


@cuda.jit
def _cuda_preprocess_uint_kernel(arr: Any, dark_floor: int, shift: int, half: int, mask: int, max_val: int) -> None:
    x, y, z = cuda.grid(3)  # type: ignore[misc]
    z_dim, y_dim, x_dim = arr.shape
    if z >= z_dim or y >= y_dim or x >= x_dim:
        return
    val = np.int32(arr[z, y, x])
    if dark_floor > 0 and val <= dark_floor:
        arr[z, y, x] = 0
        return
    if shift > 0:
        rounded = val + half
        arr[z, y, x] = (max_val if rounded > max_val else rounded) & mask  # noqa: FURB136


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
        vol_in, z, yo, xo, bin_y, bin_x, filter_mode, filter_size, bin_mode, highpass_add, is_int, max_val,
        med_buf, bin_buf,
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
                    vol_in, z, yo, xo, bin_y, bin_x, filter_mode, filter_size, bin_mode, highpass_add, is_int, max_val,
                    med_buf, bin_buf,
                )


# ---------------------------------------------------------------------------
# Public host & device-dispatch entry points
# ---------------------------------------------------------------------------


def quantize_lossy_volume(
    arr: np.ndarray,
    *,
    lossy_step_noise: float | None = None,
    lossy_domain: LossyDomain | str = LossyDomain.SQRT,
    noise_sigma: float | None = None,
    offset: float | None = None,
    use_gpu: bool | None = None,
    in_place: bool = False,
) -> np.ndarray:
    """Quantize ``arr`` in ``sqrt`` (Anscombe) or ``linear`` domain to multiples of ``lossy_step_noise * sigma``."""
    dom = LossyDomain(lossy_domain) if not isinstance(lossy_domain, LossyDomain) else lossy_domain
    if lossy_step_noise is None or lossy_step_noise <= 0.0 or dom == LossyDomain.NONE:
        return arr
    if noise_sigma is None or noise_sigma <= 0.0 or (dom == LossyDomain.SQRT and offset is None):
        est_sigma, est_off = estimate_noise_sigma(arr, domain=dom, offset=offset)
        sigma = est_sigma if (noise_sigma is None or noise_sigma <= 0.0) else float(noise_sigma)
        off = est_off if offset is None else float(offset)
    else:
        sigma = float(noise_sigma)
        off = 0.0 if offset is None else float(offset)
    step = float(lossy_step_noise) * sigma
    if step <= 0.0:
        return arr

    out = arr if (in_place and arr.flags.c_contiguous) else np.ascontiguousarray(arr).copy()
    orig_shape = out.shape
    if out.ndim == 2:
        out = out[np.newaxis, ...]
    domain_code = 1 if dom == LossyDomain.SQRT else 2
    is_int, max_val = _dtype_int_bounds(out.dtype)
    if _cuda_ready(use_gpu):
        z, y, x = out.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            d_arr = cuda.to_device(out)
            _cuda_lossy_quantize_kernel[_grid_3d(z, y, x), _THREADS_3D](  # type: ignore[index]
                d_arr, domain_code, step, off, is_int, max_val
            )
            d_arr.copy_to_host(out)
    else:
        _cpu_lossy_quantize_kernel(out, domain_code, step, off, is_int, max_val)
    return out.reshape(orig_shape)


def preprocess_volume(
    arr: np.ndarray,
    *,
    keep_bits: int | None = None,
    dark_floor: int = 0,
    lossy_step_noise: float | None = None,
    lossy_domain: LossyDomain | str = LossyDomain.SQRT,
    noise_sigma: float | None = None,
    sqrt_offset: float | None = None,
) -> np.ndarray:
    """Apply dark-floor clamping, bit-rounding, and/or noise-bounded lossy rounding on the host."""
    res = arr
    if (dark_floor > 0 or keep_bits is not None) and res.dtype.kind == "u" and res.dtype.itemsize in (1, 2):
        bits = res.dtype.itemsize * 8
        max_val = (1 << bits) - 1
        shift = min(bits, max(0, bits - keep_bits)) if keep_bits is not None else 0
        if shift > 0:
            half = 1 << (shift - 1)
            mask = max_val ^ ((1 << shift) - 1)
            rounded = np.minimum(max_val, res.astype(np.uint32) + half) & mask
            if dark_floor > 0:
                rounded = np.where(res <= dark_floor, np.uint32(0), rounded)
            res = rounded.astype(res.dtype)
        elif dark_floor > 0:
            res = np.where(res <= dark_floor, res.dtype.type(0), res)
    if lossy_step_noise is not None and lossy_step_noise > 0.0:
        res = quantize_lossy_volume(
            res,
            lossy_step_noise=lossy_step_noise,
            lossy_domain=lossy_domain,
            noise_sigma=noise_sigma,
            offset=sqrt_offset,
            use_gpu=False,
            in_place=(res is not arr),
        )
    return res


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
