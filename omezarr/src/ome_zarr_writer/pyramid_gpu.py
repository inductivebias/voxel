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

import numba
import numpy as np
from numba import cuda, prange
from numba.core.errors import NumbaPerformanceWarning
from vxlib.vector import UIVec3D

from ome_zarr_writer.dataset import BinningMode, DownscaleType, Dtype, LossyDomain, PreFilterType, ScaleLevel

log = logging.getLogger(__name__)

_SUPPORTED_REDUCTIONS = frozenset({DownscaleType.MEAN, DownscaleType.MAX, DownscaleType.MIN, DownscaleType.GAUSSIAN})
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


def cutoff_to_sigma_px(cutoff: float, voxel_size_um: float = 1.0, units: str = "um") -> float:
    """Convert a spatial structure cut-off ``c`` (in µm or voxels) to a Gaussian standard deviation ``σ`` in voxels.

    At spatial period ``c``, the Gaussian filter passes half the signal amplitude:
    ``σ = sqrt(ln(2) / 2) / π * c_voxels ≈ 0.1873906 * c_voxels``.
    """
    if cutoff <= 0.0:
        return 0.0
    c_vox = (cutoff / max(voxel_size_um, 1e-9)) if units == "um" else float(cutoff)
    return float(math.sqrt(math.log(2.0) / 2.0) / math.pi * c_vox)


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


def estimate_noise_sigma(
    arr: np.ndarray,
    *,
    domain: LossyDomain | str = LossyDomain.SQRT,
    offset: float | None = None,
    max_sample_slices: int = 8,
) -> tuple[float, float]:
    """Estimate random noise ``σ`` from the MAD of first differences along the finest axis (X).

    Returns ``(sigma, offset)``, where ``offset`` is the 0.1st-percentile dark floor used by the
    Anscombe transform when ``domain == LossyDomain.SQRT`` (or ``0.0`` for ``LINEAR``).
    Locking ``(sigma, offset)`` on the first batch avoids batch-to-batch quantisation drift.
    """
    dom = LossyDomain(domain) if not isinstance(domain, LossyDomain) else domain
    if arr.size < 2 or dom == LossyDomain.NONE:
        return 0.0, 0.0

    sample = arr
    if sample.ndim >= 3 and sample.shape[0] > max_sample_slices:
        step = max(1, sample.shape[0] // max_sample_slices)
        sample = sample[::step]
    # Subsample Y if huge to keep percentile/median under a couple milliseconds
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
    sigma = float(1.4826 * mad / math.sqrt(2.0))
    return sigma, off


@cuda.jit(fastmath=True)
def _cuda_lossy_quantize_kernel(
    arr: Any,
    domain_code: int,
    step: float,
    offset: float,
    is_int: int,
    max_val: float,
) -> None:
    x, y, z = cuda.grid(3)  # type: ignore[misc]
    z_dim, y_dim, x_dim = arr.shape
    if z >= z_dim or y >= y_dim or x >= x_dim:
        return
    val = np.float32(arr[z, y, x])
    if domain_code == 1:  # SQRT (Anscombe)
        shifted = val - np.float32(offset)
        if shifted < np.float32(0.0):
            shifted = np.float32(0.0)
        t = np.float32(2.0) * math.sqrt(shifted + np.float32(0.375))
        tq = math.floor(t / np.float32(step) + np.float32(0.5)) * np.float32(step)
        rec = np.float32(0.25) * tq * tq - np.float32(0.375) + np.float32(offset)
    else:  # LINEAR
        rec = math.floor(val / np.float32(step) + np.float32(0.5)) * np.float32(step)

    if is_int == 1:
        r = math.floor(rec + np.float32(0.5))
        if r < 0.0:
            r = 0.0
        elif r > max_val:
            r = max_val
        arr[z, y, x] = r
    else:
        arr[z, y, x] = rec


@numba.njit(parallel=True, cache=True)
def _cpu_lossy_quantize_kernel(
    arr: np.ndarray,
    domain_code: int,
    step: float,
    offset: float,
    is_int: int,
    max_val: float,
) -> None:
    z_dim, y_dim, x_dim = arr.shape
    step_f = np.float32(step)
    off_f = np.float32(offset)
    for z in prange(z_dim):
        for y in range(y_dim):
            for x in range(x_dim):
                val = np.float32(arr[z, y, x])
                if domain_code == 1:
                    shifted = val - off_f
                    if shifted < np.float32(0.0):
                        shifted = np.float32(0.0)
                    t = np.float32(2.0) * math.sqrt(shifted + np.float32(0.375))
                    tq = math.floor(t / step_f + np.float32(0.5)) * step_f
                    rec = np.float32(0.25) * tq * tq - np.float32(0.375) + off_f
                else:
                    rec = math.floor(val / step_f + np.float32(0.5)) * step_f

                if is_int == 1:
                    r = math.floor(rec + np.float32(0.5))
                    if r < 0.0:
                        r = 0.0
                    elif r > max_val:
                        r = max_val
                    arr[z, y, x] = r
                else:
                    arr[z, y, x] = rec


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
    """Quantise ``arr`` in ``sqrt`` (Anscombe) or ``linear`` domain to multiples of ``lossy_step_noise * σ``."""
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
    run_gpu = is_gpu_available() if use_gpu is None else (use_gpu and is_gpu_available())

    if run_gpu:
        z, y, x = out.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            d_arr = cuda.to_device(out)
            _cuda_lossy_quantize_kernel[_grid_3d(z, y, x), _THREADS_3D](  # type: ignore[index]
                d_arr,
                int(domain_code),
                float(step),
                float(off),
                int(is_int),
                float(max_val),
            )
            d_arr.copy_to_host(out)
    else:
        _cpu_lossy_quantize_kernel(out, int(domain_code), float(step), float(off), int(is_int), float(max_val))

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


# ---------------------------------------------------------------------------
# CUDA Kernels (Preprocess, MEAN, MAX, MIN, and 2-pass separable [1, 3, 3, 1] GAUSSIAN)
# ---------------------------------------------------------------------------


@cuda.jit
def _cuda_preprocess_uint_kernel(
    arr: Any,
    dark_floor: int,
    shift: int,
    half: int,
    mask: int,
    max_val: int,
) -> None:
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
        if rounded > max_val:  # noqa: PLR1730 - Python min() unsupported in Numba @cuda.jit
            rounded = max_val
        arr[z, y, x] = rounded & mask


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
    """Fused 2D spatial pre-filter (none, mean, gaussian, median, highpass) + rectangular (bin_y x bin_x) reduction."""
    xo, yo, z = cuda.grid(3)  # type: ignore[misc]
    z_out, y_out, x_out = vol_out.shape
    if z >= z_out or yo >= y_out or xo >= x_out:
        return
    _, y_in, x_in = vol_in.shape

    y0 = yo * bin_y
    x0 = xo * bin_x
    radius = filter_size // 2
    k_total = filter_size * filter_size
    med_buf = cuda.local.array(25, numba.float32)  # type: ignore[misc]
    bin_buf = cuda.local.array(64, numba.float32)  # type: ignore[misc]

    w1d = 4 if filter_size == 3 else 16
    gauss_norm = np.float64(w1d * w1d)
    mean_norm = np.float64(k_total)

    block_sum = np.float64(0.0)
    block_max = np.float64(-1e30)
    b_count = 0

    dy_start = (bin_y - 1) // 2 if bin_mode == 4 else 0
    dy_end = dy_start + 1 if bin_mode == 4 else bin_y
    dx_start = (bin_x - 1) // 2 if bin_mode == 4 else 0
    dx_end = dx_start + 1 if bin_mode == 4 else bin_x

    for dy in range(dy_start, dy_end):
        py = y0 + dy
        for dx in range(dx_start, dx_end):
            px = x0 + dx
            if filter_mode == 0:
                val = np.float64(vol_in[z, py, px])
            elif filter_mode == 1:
                acc = np.float64(0.0)
                for ky in range(-radius, radius + 1):
                    sy = 0 if py + ky < 0 else (y_in - 1 if py + ky >= y_in else py + ky)  # noqa: FURB136
                    for kx in range(-radius, radius + 1):
                        sx = 0 if px + kx < 0 else (x_in - 1 if px + kx >= x_in else px + kx)  # noqa: FURB136
                        acc += np.float64(vol_in[z, sy, sx])
                val = acc / mean_norm
            elif filter_mode in (2, 4):
                acc = np.float64(0.0)
                for ky in range(-radius, radius + 1):
                    sy = 0 if py + ky < 0 else (y_in - 1 if py + ky >= y_in else py + ky)  # noqa: FURB136
                    if filter_size == 3:
                        wy = np.float64(2.0 if ky == 0 else 1.0)
                    else:
                        ay = -ky if ky < 0 else ky
                        wy = np.float64(6.0 if ay == 0 else (4.0 if ay == 1 else 1.0))
                    for kx in range(-radius, radius + 1):
                        sx = 0 if px + kx < 0 else (x_in - 1 if px + kx >= x_in else px + kx)  # noqa: FURB136
                        if filter_size == 3:
                            wx = np.float64(2.0 if kx == 0 else 1.0)
                        else:
                            ax = -kx if kx < 0 else kx
                            wx = np.float64(6.0 if ax == 0 else (4.0 if ax == 1 else 1.0))
                        w = wy * wx
                        acc += np.float64(vol_in[z, sy, sx]) * w
                blur = acc / gauss_norm
                val = blur if filter_mode == 2 else (np.float64(vol_in[z, py, px]) - blur + np.float64(highpass_add))
            else:
                idx = 0
                for ky in range(-radius, radius + 1):
                    sy = 0 if py + ky < 0 else (y_in - 1 if py + ky >= y_in else py + ky)  # noqa: FURB136
                    for kx in range(-radius, radius + 1):
                        sx = 0 if px + kx < 0 else (x_in - 1 if px + kx >= x_in else px + kx)  # noqa: FURB136
                        med_buf[idx] = np.float32(vol_in[z, sy, sx])
                        idx += 1
                for i in range(1, k_total):
                    key = med_buf[i]
                    j = i - 1
                    while j >= 0 and med_buf[j] > key:
                        med_buf[j + 1] = med_buf[j]
                        j -= 1
                    med_buf[j + 1] = key
                val = np.float64(med_buf[k_total // 2])

            block_sum += val
            if val > block_max:  # noqa: PLR1730
                block_max = val
            if bin_mode == 1 and b_count < 64:
                bin_buf[b_count] = np.float32(val)
                b_count += 1

    if bin_mode == 0:  # MEAN
        res = block_sum / np.float64(bin_y * bin_x)
    elif bin_mode == 1:  # MEDIAN
        for i in range(1, b_count):
            key = bin_buf[i]
            j = i - 1
            while j >= 0 and bin_buf[j] > key:
                bin_buf[j + 1] = bin_buf[j]
                j -= 1
            bin_buf[j + 1] = key
        mid = b_count // 2
        if b_count % 2 == 1:
            res = np.float64(bin_buf[mid])
        else:
            res = np.float64(0.5) * (np.float64(bin_buf[mid - 1]) + np.float64(bin_buf[mid]))
    elif bin_mode == 2:  # SUM
        res = block_sum
    elif bin_mode == 3:  # MAX
        res = block_max
    else:  # NEAREST
        res = block_sum

    if is_int == 1:
        r = math.floor(res + np.float64(0.5))
        if r < 0.0:
            r = 0.0
        elif r > max_val:
            r = max_val
        vol_out[z, yo, xo] = r
    else:
        vol_out[z, yo, xo] = res


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
    """CPU Numba parallel counterpart to :func:`_cuda_prefilter_and_bin_kernel`."""
    z_out, y_out, x_out = vol_out.shape
    _, y_in, x_in = vol_in.shape
    radius = filter_size // 2
    k_total = filter_size * filter_size
    w1d = 4 if filter_size == 3 else 16
    gauss_norm = np.float64(w1d * w1d)
    mean_norm = np.float64(k_total)

    dy_start = (bin_y - 1) // 2 if bin_mode == 4 else 0
    dy_end = dy_start + 1 if bin_mode == 4 else bin_y
    dx_start = (bin_x - 1) // 2 if bin_mode == 4 else 0
    dx_end = dx_start + 1 if bin_mode == 4 else bin_x

    for z in prange(z_out):
        med_buf = np.empty(25, dtype=np.float32)
        bin_buf = np.empty(max(64, bin_y * bin_x), dtype=np.float32)
        for yo in range(y_out):
            y0 = yo * bin_y
            for xo in range(x_out):
                x0 = xo * bin_x
                block_sum = np.float64(0.0)
                block_max = np.float64(-1e30)
                b_count = 0
                for dy in range(dy_start, dy_end):
                    py = y0 + dy
                    for dx in range(dx_start, dx_end):
                        px = x0 + dx
                        if filter_mode == 0:
                            val = np.float64(vol_in[z, py, px])
                        elif filter_mode == 1:
                            acc = np.float64(0.0)
                            for ky in range(-radius, radius + 1):
                                sy = 0 if py + ky < 0 else (y_in - 1 if py + ky >= y_in else py + ky)  # noqa: FURB136
                                for kx in range(-radius, radius + 1):
                                    sx = 0 if px + kx < 0 else (x_in - 1 if px + kx >= x_in else px + kx)  # noqa: FURB136
                                    acc += np.float64(vol_in[z, sy, sx])
                            val = acc / mean_norm
                        elif filter_mode in (2, 4):
                            acc = np.float64(0.0)
                            for ky in range(-radius, radius + 1):
                                sy = 0 if py + ky < 0 else (y_in - 1 if py + ky >= y_in else py + ky)  # noqa: FURB136
                                if filter_size == 3:
                                    wy = np.float64(2.0 if ky == 0 else 1.0)
                                else:
                                    ay = -ky if ky < 0 else ky
                                    wy = np.float64(6.0 if ay == 0 else (4.0 if ay == 1 else 1.0))
                                for kx in range(-radius, radius + 1):
                                    sx = 0 if px + kx < 0 else (x_in - 1 if px + kx >= x_in else px + kx)  # noqa: FURB136
                                    if filter_size == 3:
                                        wx = np.float64(2.0 if kx == 0 else 1.0)
                                    else:
                                        ax = -kx if kx < 0 else kx
                                        wx = np.float64(6.0 if ax == 0 else (4.0 if ax == 1 else 1.0))
                                    w = wy * wx
                                    acc += np.float64(vol_in[z, sy, sx]) * w
                            blur = acc / gauss_norm
                            val = (
                                blur
                                if filter_mode == 2
                                else (np.float64(vol_in[z, py, px]) - blur + np.float64(highpass_add))
                            )
                        else:
                            idx = 0
                            for ky in range(-radius, radius + 1):
                                sy = 0 if py + ky < 0 else (y_in - 1 if py + ky >= y_in else py + ky)  # noqa: FURB136
                                for kx in range(-radius, radius + 1):
                                    sx = 0 if px + kx < 0 else (x_in - 1 if px + kx >= x_in else px + kx)  # noqa: FURB136
                                    med_buf[idx] = np.float32(vol_in[z, sy, sx])
                                    idx += 1
                            for i in range(1, k_total):
                                key = med_buf[i]
                                j = i - 1
                                while j >= 0 and med_buf[j] > key:
                                    med_buf[j + 1] = med_buf[j]
                                    j -= 1
                                med_buf[j + 1] = key
                            val = np.float64(med_buf[k_total // 2])

                        block_sum += val
                        if val > block_max:  # noqa: PLR1730
                            block_max = val
                        if bin_mode == 1:
                            bin_buf[b_count] = np.float32(val)
                            b_count += 1

                if bin_mode == 0:
                    res = block_sum / np.float64(bin_y * bin_x)
                elif bin_mode == 1:
                    for i in range(1, b_count):
                        key = bin_buf[i]
                        j = i - 1
                        while j >= 0 and bin_buf[j] > key:
                            bin_buf[j + 1] = bin_buf[j]
                            j -= 1
                        bin_buf[j + 1] = key
                    mid = b_count // 2
                    if b_count % 2 == 1:
                        res = np.float64(bin_buf[mid])
                    else:
                        res = np.float64(0.5) * (np.float64(bin_buf[mid - 1]) + np.float64(bin_buf[mid]))
                elif bin_mode == 2:
                    res = block_sum
                elif bin_mode == 3:
                    res = block_max
                else:
                    res = block_sum

                if is_int == 1:
                    r = math.floor(res + np.float64(0.5))
                    if r < 0.0:
                        r = 0.0
                    elif r > max_val:
                        r = max_val
                    vol_out[z, yo, xo] = r
                else:
                    vol_out[z, yo, xo] = res


def _dtype_int_bounds(dt: np.dtype) -> tuple[int, float]:
    if dt.kind in ("u", "i"):
        info = np.iinfo(dt)
        return 1, float(info.max)
    return 0, float("inf")


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
    """Apply a 2D spatial pre-filter and rectangular ``(bin_y, bin_x)`` reduction to a 3D volume ``(Z, Y, X)``.

    Uses Numba CUDA when ``use_gpu`` is True (or None and a GPU is available) and falls back to parallel Numba CPU.
    """
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
    y_out = y_in // bin_y
    x_out = x_in // bin_x
    if y_out < 1 or x_out < 1:
        raise ValueError(f"Input shape {(y_in, x_in)} is smaller than bin factors {(bin_y, bin_x)}")

    if out is None:
        out = np.empty((z, y_out, x_out), dtype=vol.dtype)
    elif out.shape != (z, y_out, x_out):
        raise ValueError(f"out shape {out.shape} does not match expected {(z, y_out, x_out)}")

    filter_mode = _PREFILTER_TO_CODE[pf]
    bin_mode_code = _BINMODE_TO_CODE[bm]
    is_int, max_val = _dtype_int_bounds(vol.dtype)
    run_gpu = is_gpu_available() if use_gpu is None else (use_gpu and is_gpu_available())

    if run_gpu:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            d_in = cuda.to_device(np.ascontiguousarray(vol))
            d_out = cuda.device_array((z, y_out, x_out), dtype=vol.dtype)
            _cuda_prefilter_and_bin_kernel[_grid_3d(z, y_out, x_out), _THREADS_3D](  # type: ignore[index]
                d_in,
                d_out,
                int(bin_y),
                int(bin_x),
                int(filter_mode),
                int(filter_size),
                int(bin_mode_code),
                float(highpass_add),
                int(is_int),
                float(max_val),
            )
            d_out.copy_to_host(out)
        return out

    _cpu_prefilter_and_bin_kernel(
        np.ascontiguousarray(vol),
        out,
        int(bin_y),
        int(bin_x),
        int(filter_mode),
        int(filter_size),
        int(bin_mode_code),
        float(highpass_add),
        int(is_int),
        float(max_val),
    )
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
    vol_out = filter_and_bin_volume(
        frame[np.newaxis, ...],
        bin_y=bin_y,
        bin_x=bin_x,
        pre_filter=pf,
        filter_size=filter_size,
        bin_mode=bm,
        highpass_add=highpass_add,
        use_gpu=use_gpu,
    )
    return vol_out[0]


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
        self.locked_noise_sigma: float | None = None
        self.locked_sqrt_offset: float | None = None
        self.ensure_device_buffers()

    def ensure_device_buffers(self) -> None:
        """Allocate device arrays if not currently resident in VRAM."""
        if self.d_levels:
            return
        if self.raw_shape_l0 is not None and self.raw_shape_l0 != self.shape_l0:
            raw_shape = (max(1, self.raw_shape_l0.z), max(1, self.raw_shape_l0.y), max(1, self.raw_shape_l0.x))
            self.d_raw = cuda.device_array(raw_shape, dtype=self.dtype)
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
        self.d_raw = None
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
        copy_l0_to: np.ndarray | None = None,
        return_host_f32: bool = False,
        keep_bits: int | None = None,
        dark_floor: int = 0,
        bin_y: int = 1,
        bin_x: int = 1,
        pre_filter: PreFilterType = PreFilterType.NONE,
        filter_size: int = 3,
        bin_mode: BinningMode = BinningMode.MEAN,
        highpass_add: float = 100.0,
        lossy_step_noise: float | None = None,
        lossy_domain: LossyDomain = LossyDomain.SQRT,
        noise_sigma: float | None = None,
        sqrt_offset: float | None = None,
    ) -> dict[ScaleLevel, Any]:
        """Transfer ``block`` (raw or L0) to GPU, optionally apply 2D pre-filter + rectangular
        ``(bin_y x bin_x)`` binning to produce L0, compute all pyramid levels up to ``max_level``,
        and optionally copy reduced levels back to host arrays.
        """
        if reduction not in _SUPPORTED_REDUCTIONS:
            raise ValueError(f"Unsupported downscale type: {reduction}")

        self.ensure_device_buffers()
        is_int, max_val_f = _dtype_int_bounds(self.dtype)
        needs_bin_or_filter = bin_y > 1 or bin_x > 1 or pre_filter != PreFilterType.NONE
        if needs_bin_or_filter:
            z_raw, y_raw, x_raw = block.shape
            z0 = z_raw
            y0 = y_raw // bin_y
            x0 = x_raw // bin_x
            if self.d_raw is None or (
                self.d_raw.shape[0] < z_raw or self.d_raw.shape[1] < y_raw or self.d_raw.shape[2] < x_raw
            ):
                self.d_raw = cuda.device_array((z_raw, y_raw, x_raw), dtype=self.dtype)
            d_raw_view = self.d_raw[:z_raw, :y_raw, :x_raw]
            d_raw_view.copy_to_device(np.ascontiguousarray(block))
            d_l0 = self.d_levels[ScaleLevel.L0][:z0, :y0, :x0]
            filter_code = _PREFILTER_TO_CODE[pre_filter]
            bin_mode_code = _BINMODE_TO_CODE[bin_mode]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NumbaPerformanceWarning)
                _cuda_prefilter_and_bin_kernel[_grid_3d(z0, y0, x0), _THREADS_3D](  # type: ignore[index]
                    d_raw_view,
                    d_l0,
                    int(bin_y),
                    int(bin_x),
                    int(filter_code),
                    int(filter_size),
                    int(bin_mode_code),
                    float(highpass_add),
                    is_int,
                    max_val_f,
                )
        else:
            z0, y0, x0 = block.shape
            d_l0 = self.d_levels[ScaleLevel.L0][:z0, :y0, :x0]
            d_l0.copy_to_device(np.ascontiguousarray(block))

        if (dark_floor > 0 or keep_bits is not None) and self.dtype.kind == "u" and self.dtype.itemsize in (1, 2):
            bits = self.dtype.itemsize * 8
            max_val = (1 << bits) - 1
            shift = min(bits, max(0, bits - keep_bits)) if keep_bits is not None else 0
            if dark_floor > 0 or shift > 0:
                half = (1 << (shift - 1)) if shift > 0 else 0
                mask = max_val ^ ((1 << shift) - 1) if shift > 0 else max_val
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NumbaPerformanceWarning)
                    _cuda_preprocess_uint_kernel[_grid_3d(z0, y0, x0), _THREADS_3D](  # type: ignore[index]
                        d_l0,
                        dark_floor,
                        shift,
                        half,
                        mask,
                        max_val,
                    )

        if lossy_step_noise is not None and lossy_step_noise > 0.0 and lossy_domain != LossyDomain.NONE:
            sigma_val = noise_sigma if (noise_sigma is not None and noise_sigma > 0.0) else self.locked_noise_sigma
            offset_val = sqrt_offset if sqrt_offset is not None else self.locked_sqrt_offset
            if sigma_val is None or sigma_val <= 0.0 or (lossy_domain == LossyDomain.SQRT and offset_val is None):
                n_sub = min(4, z0)
                z_step = max(1, z0 // max(1, n_sub))
                sample_host = d_l0[::z_step].copy_to_host() if needs_bin_or_filter else block[::z_step]
                est_sigma, est_off = estimate_noise_sigma(sample_host, domain=lossy_domain, offset=offset_val)
                if sigma_val is None or sigma_val <= 0.0:
                    sigma_val = est_sigma
                    self.locked_noise_sigma = est_sigma
                if offset_val is None:
                    offset_val = est_off
                    self.locked_sqrt_offset = est_off
            step_val = float(lossy_step_noise) * float(sigma_val)
            if step_val > 0.0:
                domain_code = 1 if lossy_domain == LossyDomain.SQRT else 2
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NumbaPerformanceWarning)
                    _cuda_lossy_quantize_kernel[_grid_3d(z0, y0, x0), _THREADS_3D](  # type: ignore[index]
                        d_l0,
                        int(domain_code),
                        float(step_val),
                        float(0.0 if offset_val is None else offset_val),
                        is_int,
                        max_val_f,
                    )

        if copy_l0_to is not None:
            zc = min(copy_l0_to.shape[0], z0)
            yc = min(copy_l0_to.shape[1], y0)
            xc = min(copy_l0_to.shape[2], x0)
            if zc > 0 and yc > 0 and xc > 0:
                d_l0[:zc, :yc, :xc].copy_to_host(copy_l0_to[:zc, :yc, :xc])

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
    is_int = 1 if dt.kind in ("u", "i") else 0
    max_val_f = float(np.iinfo(dt).max) if is_int else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NumbaPerformanceWarning)
        _cuda_prefilter_and_bin_kernel[_grid_3d(4, 4, 4), _THREADS_3D](  # type: ignore[index]
            d_in,
            d_in[:, :4, :4],
            2,
            2,
            0,
            3,
            0,
            100.0,
            is_int,
            max_val_f,
        )
        _cuda_lossy_quantize_kernel[_grid_3d(4, 4, 4), _THREADS_3D](  # type: ignore[index]
            d_in[:, :4, :4],
            1,
            1.0,
            0.0,
            is_int,
            max_val_f,
        )
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
    *,
    keep_bits: int | None = None,
    dark_floor: int = 0,
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
    return ws.compute_from_host(
        block,
        max_level,
        reduction=reduction,
        return_host_f32=True,
        keep_bits=keep_bits,
        dark_floor=dark_floor,
    )
