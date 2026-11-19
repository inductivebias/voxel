"""CPU preprocessing for streaming OME-Zarr writes: Squish filters, per-axis binning, and LUT quantization.

Pipeline order (all steps optional; default config is a no-op):
1. Median filter over a small odd box ``median_zyx`` (``mz * my * mx <= 9`` on CPU), mirrored edges.
2. Low-pass separable Gaussian blur from per-axis structure cut-offs ``lowpass_cutoff_zyx`` (voxels).
3. High-pass background subtraction ``x - GaussianBlur(x, sigma_high) + highpass_add``.
4. Per-axis binning ``bin_zyx`` (``mean``, ``median``, ``max``, ``sum``, ``nearest``).
5. Point-wise ``uint16`` LUT quantization (Anscombe ``sqrt`` noise-bounded rounding, linear ``keep_bits``,
   and/or ``dark_floor`` clamping) while keeping values in original grey-level units so TensorStore and
   Neuroglancer read the resulting OME-Zarr v3 arrays natively.

Spatial filters run in bounded-RAM Z-strips with exact boundary-mirroring over the extended
``[halo_lead | core | halo_trail]`` slab without copying the full batch, so batched writes with a Z-halo
match whole-volume filtering. Input ``core`` buffers are never mutated (preserving ``FrameLease`` safety).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numba import njit, prange
from pydantic import BaseModel, ConfigDict, Field, model_validator
from vxlib.vector import UIVec3D

BinMode = Literal["mean", "median", "max", "sum", "nearest"]

_BIN_MODE_CODE: dict[BinMode, int] = {"mean": 0, "median": 1, "max": 2, "sum": 3, "nearest": 4}
_EMPTY_LUT = np.empty(0, dtype=np.uint16)

# Max CPU median box volume: 9 elements (e.g. 3x3x1, 3x1x3, 1x3x3, or 1D up to 9) runs at ~10 GB/s
# via a fixed comparator network; a 27-element 3x3x3 box drops ~85x and would stall live capture.
MAX_CPU_MEDIAN_ELEMENTS = 9

# Z-strip depth for streaming float32 spatial filters (rounded up to a multiple of bin_z).
_DEFAULT_STRIP_Z = 16


def cutoff_to_sigma(cutoff: float) -> float:
    """Convert a half-power structure cut-off (in voxels) to Gaussian sigma: ``sqrt(ln(2)/2)/pi * c``."""
    return 0.0 if cutoff <= 0.0 else math.sqrt(math.log(2.0) / 2.0) / math.pi * cutoff


def gaussian_radius(sigma: float, truncate: float = 3.0) -> int:
    """Discrete 1-D Gaussian kernel half-width for ``sigma`` truncated at ``truncate * sigma``."""
    return 0 if sigma <= 0.0 else max(1, math.ceil(truncate * sigma))


def gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Normalized 1-D float32 Gaussian kernel of length ``2 * radius + 1``."""
    rad = gaussian_radius(sigma, truncate=truncate)
    x = np.arange(-rad, rad + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k.astype(np.float32)


def _kernels_for(cutoffs: tuple[float, float, float]) -> tuple[np.ndarray | None, ...]:
    return tuple(gaussian_kernel_1d(cutoff_to_sigma(c)) if c > 0.0 else None for c in cutoffs)


class PreprocessSpec(BaseModel):
    """Picklable, frozen specification for worker-side preprocessing before pyramid + shard encoding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Quantization (uint16 LUT)
    dark_floor: int = Field(default=0, ge=0, le=65535)
    keep_bits: int | None = Field(default=None, ge=1, le=16)
    lossy_sigma: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    noise_offset: float | None = Field(default=None, allow_inf_nan=False)
    noise_sigma_y: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    # Per-axis binning
    bin_zyx: tuple[int, int, int] = (1, 1, 1)
    bin_mode: BinMode = "mean"

    # Spatial Squish filters
    median_zyx: tuple[int, int, int] = (1, 1, 1)
    lowpass_cutoff_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highpass_cutoff_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highpass_add: float = Field(default=100.0, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_axes(self) -> PreprocessSpec:
        if any(f < 1 for f in self.bin_zyx):
            raise ValueError(f"bin_zyx factors must all be >= 1, got {self.bin_zyx}")
        if any(m < 1 or m % 2 == 0 for m in self.median_zyx):
            raise ValueError(f"median_zyx sizes must be odd integers >= 1, got {self.median_zyx}")
        mz, my, mx = self.median_zyx
        if mz * my * mx > MAX_CPU_MEDIAN_ELEMENTS:
            raise ValueError(
                f"median_zyx={self.median_zyx} has {mz * my * mx} elements; CPU median is capped at "
                f"{MAX_CPU_MEDIAN_ELEMENTS} elements (e.g. 3x3x1 or 3x1x3) to avoid stalling capture"
            )
        for name, cutoffs in (
            ("lowpass_cutoff_zyx", self.lowpass_cutoff_zyx),
            ("highpass_cutoff_zyx", self.highpass_cutoff_zyx),
        ):
            if any(not math.isfinite(c) or c < 0.0 for c in cutoffs):
                raise ValueError(f"{name} values must be finite and >= 0, got {cutoffs}")
        if (self.noise_offset is None) != (self.noise_sigma_y is None):
            raise ValueError("noise_offset and noise_sigma_y must both be set or both be None")
        return self

    @property
    def has_quantize(self) -> bool:
        return self.dark_floor > 0 or self.keep_bits is not None or self.lossy_sigma is not None

    @property
    def has_binning(self) -> bool:
        return self.bin_zyx != (1, 1, 1)

    @property
    def has_median(self) -> bool:
        return self.median_zyx != (1, 1, 1)

    @property
    def has_lowpass(self) -> bool:
        return any(c > 0.0 for c in self.lowpass_cutoff_zyx)

    @property
    def has_highpass(self) -> bool:
        return any(c > 0.0 for c in self.highpass_cutoff_zyx)

    @property
    def has_filters(self) -> bool:
        return self.has_median or self.has_lowpass or self.has_highpass

    @property
    def is_active(self) -> bool:
        return self.has_quantize or self.has_binning or self.has_filters

    @property
    def needs_auto_sqrt(self) -> bool:
        return self.lossy_sigma is not None and (self.noise_offset is None or self.noise_sigma_y is None)

    @property
    def z_radii(self) -> tuple[int, int, int]:
        """Per-stage Z radii ``(r_median, r_lowpass, r_highpass)`` in raw voxel planes."""
        return (
            self.median_zyx[0] // 2,
            gaussian_radius(cutoff_to_sigma(self.lowpass_cutoff_zyx[0])),
            gaussian_radius(cutoff_to_sigma(self.highpass_cutoff_zyx[0])),
        )

    @property
    def halo_z(self) -> int:
        """Total Z-halo planes required on each side of a batch for exact boundary-free filtering."""
        return sum(self.z_radii)

    def strip_z(self, batch_z: int) -> int:
        """Z-strip depth for streaming filters: a multiple of ``bin_zyx[0]`` capped at ``batch_z``."""
        fz = self.bin_zyx[0]
        return min(batch_z, max(fz, ((_DEFAULT_STRIP_Z + fz - 1) // fz) * fz))

    def build_lut(self, *, offset: float | None = None, sigma_y: float | None = None) -> np.ndarray:
        """Build the configured ``uint16`` quantization LUT using optional resolved Anscombe parameters."""
        off = self.noise_offset if offset is None else offset
        sig = self.noise_sigma_y if sigma_y is None else sigma_y
        return build_uint16_lut(
            dark_floor=self.dark_floor,
            keep_bits=self.keep_bits,
            lossy_sigma=self.lossy_sigma,
            offset=0.0 if off is None else off,
            sigma_y=1.0 if sig is None else sig,
        )

    def metadata_attributes(
        self,
        *,
        resolved_offset: float | None = None,
        resolved_sigma_y: float | None = None,
    ) -> dict[str, Any]:
        """Provenance dictionary stored in Zarr array ``attributes`` when preprocessing is active."""
        if not self.is_active:
            return {}
        info: dict[str, Any] = {}
        if self.has_median:
            info["median_zyx"] = list(self.median_zyx)
        if self.has_lowpass:
            info["lowpass_cutoff_zyx"] = list(self.lowpass_cutoff_zyx)
        if self.has_highpass:
            info["highpass_cutoff_zyx"] = list(self.highpass_cutoff_zyx)
            info["highpass_add"] = self.highpass_add
        if self.has_binning:
            info["bin_zyx"] = list(self.bin_zyx)
            info["bin_mode"] = self.bin_mode
        if self.dark_floor > 0:
            info["dark_floor"] = self.dark_floor
        if self.keep_bits is not None:
            info["keep_bits"] = self.keep_bits
        if self.lossy_sigma is not None:
            info["lossy_sigma"] = self.lossy_sigma
            offset = self.noise_offset if resolved_offset is None else resolved_offset
            sigma_y = self.noise_sigma_y if resolved_sigma_y is None else resolved_sigma_y
            if offset is not None:
                info["noise_offset"] = round(float(offset), 4)
            if sigma_y is not None:
                info["noise_sigma_y"] = round(float(sigma_y), 6)
        return {"voxel_preprocessing": info}


def binned_shape(shape: UIVec3D, bin_zyx: tuple[int, int, int]) -> UIVec3D:
    """Output voxel extent after per-axis binning (far-edge partial bins are kept)."""
    fz, fy, fx = bin_zyx
    return UIVec3D(
        z=max(1, math.ceil(shape.z / fz)),
        y=max(1, math.ceil(shape.y / fy)),
        x=max(1, math.ceil(shape.x / fx)),
    )


def preprocess_scratch_bytes(raw_batch_shape: UIVec3D, spec: PreprocessSpec, itemsize: int) -> int:
    """Peak additional worker-heap RAM (bytes) one worker needs for preprocessing beyond shared memory."""
    if not spec.is_active:
        return 0
    plane_voxels = raw_batch_shape.y * raw_batch_shape.x
    out_shp = binned_shape(raw_batch_shape, spec.bin_zyx)
    out_l0_bytes = out_shp.z * out_shp.y * out_shp.x * itemsize

    filter_bytes = 0
    if spec.has_filters:
        strip_planes = min(raw_batch_shape.z + 2 * spec.halo_z, spec.strip_z(raw_batch_shape.z) + 2 * spec.halo_z)
        strip_in_bytes = strip_planes * plane_voxels * itemsize if spec.halo_z > 0 else 0
        n_f32_bufs = 3 if spec.has_highpass else 2
        filter_bytes = strip_in_bytes + n_f32_bufs * strip_planes * plane_voxels * 4

    return out_l0_bytes + filter_bytes


# ---------------------------------------------------------------------------
# Quantization LUT & Anscombe sqrt noise estimator
# ---------------------------------------------------------------------------


def estimate_sqrt_params(vol: np.ndarray, *, finest_axis: int = 0) -> tuple[float, float]:
    """Estimate ``(offset, sigma_y)`` in the Anscombe sqrt domain ``y = 2*sqrt(max(0, x - offset) + 3/8)``.

    Uses the 0.1st percentile for ``offset`` and the MAD of adjacent differences along ``finest_axis``
    for ``sigma_y``, subsampling large volumes to keep runtime to a few milliseconds.
    """
    axis = finest_axis if vol.shape[finest_axis] >= 2 else int(np.argmax(vol.shape))
    slices = [slice(None), slice(None), slice(None)]
    if vol.size > 4_000_000:
        for ax in range(3):
            if ax != axis and vol.shape[ax] >= 4:
                slices[ax] = slice(None, None, 2)
    sub = vol[tuple(slices)].astype(np.float32)
    offset = float(np.percentile(sub, 0.1))
    y = 2.0 * np.sqrt(np.maximum(np.float32(0.0), sub - np.float32(offset)) + np.float32(0.375))
    if y.shape[axis] < 2:
        return offset, 1.0
    d = np.diff(y, axis=axis).ravel()
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    sigma_y = float(1.4826 * mad / math.sqrt(2.0))
    return offset, max(sigma_y, 1e-3)


def build_uint16_lut(
    *,
    dark_floor: int = 0,
    keep_bits: int | None = None,
    lossy_sigma: float | None = None,
    offset: float = 0.0,
    sigma_y: float = 1.0,
) -> np.ndarray:
    """Build a 65,536-entry ``uint16`` quantization lookup table preserving original grey-level units.

    Order when multiple options are combined:
    1. Anscombe ``sqrt`` noise-bounded rounding (if ``lossy_sigma`` is set)
    2. Uniform ``keep_bits`` bit-rounding (if ``keep_bits`` is set and ``< 16``)
    3. ``dark_floor`` zero clamp (values originally ``<= dark_floor`` map to ``0``)
    """
    raw = np.arange(65536, dtype=np.uint32)
    q = raw.copy()

    if lossy_sigma is not None and lossy_sigma > 0.0:
        step = max(1e-6, lossy_sigma * sigma_y)
        x = raw.astype(np.float64)
        y = 2.0 * np.sqrt(np.maximum(0.0, x - offset) + 0.375)
        yq = np.rint(y / step) * step
        xq = np.square(yq * 0.5) - 0.375 + offset
        q = np.clip(np.rint(xq), 0, 65535).astype(np.uint32)

    if keep_bits is not None and keep_bits < 16:
        shift = max(0, 16 - keep_bits)
        if shift > 0:
            half = 1 << (shift - 1)
            mask = 0xFFFF ^ ((1 << shift) - 1)
            q = np.minimum(65535, q + half) & mask

    if dark_floor > 0:
        q = np.where(raw <= dark_floor, np.uint32(0), q)

    return q.astype(np.uint16)


# ---------------------------------------------------------------------------
# Numba kernels (compiled lazily on first use; respect numba.set_num_threads)
# ---------------------------------------------------------------------------


@njit(inline="always")
def _mirror(i: Any, n: int) -> int:
    """Reflect ``i`` into ``[0, n)`` using whole-sample symmetric mirroring (matching ``scipy`` mirror)."""
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * n - 2 - i
    return i


@njit(inline="always")
def _clamp_u16(val: float) -> int:
    iv = math.floor(val + 0.5) if val >= 0.0 else 0
    if iv < 0:
        return 0
    return min(iv, 65535)


@njit(parallel=True)
def apply_lut_u16(src: np.ndarray, lut: np.ndarray, dst: np.ndarray) -> None:
    """Apply a 65,536-entry ``uint16`` LUT from ``src`` into ``dst`` (may alias for in-place)."""
    nz, ny, nx = src.shape
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                dst[z, y, x] = lut[src[z, y, x]]


@njit(parallel=True)
def cast_and_lut_f32_to_u16(src: np.ndarray, lut: np.ndarray, dst: np.ndarray) -> None:
    """Truncate-cast ``float32`` pyramid volume to ``uint16`` and apply ``lut`` in one zero-alloc pass."""
    nz, ny, nx = dst.shape
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                dst[z, y, x] = lut[np.uint16(src[z, y, x])]


@njit(inline="always")
def _cswap(a: np.float32, b: np.float32) -> tuple[np.float32, np.float32]:
    return (b, a) if a > b else (a, b)


@njit(inline="always")
def _median9(
    a0: np.float32,
    a1: np.float32,
    a2: np.float32,
    a3: np.float32,
    a4: np.float32,
    a5: np.float32,
    a6: np.float32,
    a7: np.float32,
    a8: np.float32,
) -> np.float32:
    """19-comparator Bose-Nelson sorting-network exact median of 9 float32 values."""
    a1, a2 = _cswap(a1, a2)
    a4, a5 = _cswap(a4, a5)
    a7, a8 = _cswap(a7, a8)
    a0, a1 = _cswap(a0, a1)
    a3, a4 = _cswap(a3, a4)
    a6, a7 = _cswap(a6, a7)
    a1, a2 = _cswap(a1, a2)
    a4, a5 = _cswap(a4, a5)
    a7, a8 = _cswap(a7, a8)
    a0, a3 = _cswap(a0, a3)
    a5, a8 = _cswap(a5, a8)
    a4, a7 = _cswap(a4, a7)
    a3, a6 = _cswap(a3, a6)
    a1, a4 = _cswap(a1, a4)
    a2, a5 = _cswap(a2, a5)
    a4, a7 = _cswap(a4, a7)
    a4, a2 = _cswap(a4, a2)
    a6, a4 = _cswap(a6, a4)
    a4, a2 = _cswap(a4, a2)
    return a4


@njit(inline="always")
def _median9_plane_xy(
    src: np.ndarray,
    z0: Any,
    z1: Any,
    z2: Any,
    y0: Any,
    y1: Any,
    y2: Any,
    x0: Any,
    x1: Any,
    x2: Any,
) -> np.float32:
    return _median9(
        np.float32(src[z0, y0, x0]),
        np.float32(src[z0, y0, x1]),
        np.float32(src[z0, y0, x2]),
        np.float32(src[z1, y1, x0]),
        np.float32(src[z1, y1, x1]),
        np.float32(src[z1, y1, x2]),
        np.float32(src[z2, y2, x0]),
        np.float32(src[z2, y2, x1]),
        np.float32(src[z2, y2, x2]),
    )


@njit(parallel=True)
def _median_9_strip(
    src: np.ndarray,
    src_z0: int,
    dst: np.ndarray,
    dst_z0: int,
    n_ext: int,
    mz: int,
    my: int,
    mx: int,
) -> None:
    """Fast 9-element Bose-Nelson median for 3x3x1, 3x1x3, or 1x3x3 boxes with branchless interior X loop."""
    out_nz, ny, nx = dst.shape
    rz, ry, rx = mz // 2, my // 2, mx // 2
    for oz in prange(out_nz):
        gz = np.int64(dst_z0) + np.int64(oz)
        zm = _mirror(gz - rz, n_ext) - src_z0 if rz else gz - src_z0
        zc = gz - src_z0
        zp = _mirror(gz + rz, n_ext) - src_z0 if rz else gz - src_z0
        for y in range(ny):
            yi = np.int64(y)
            ym = _mirror(yi - ry, ny) if ry else yi
            yp = _mirror(yi + ry, ny) if ry else yi
            if rx == 0:  # 3x3x1
                for x in range(nx):
                    dst[oz, y, x] = _median9(
                        np.float32(src[zm, ym, x]),
                        np.float32(src[zm, yi, x]),
                        np.float32(src[zm, yp, x]),
                        np.float32(src[zc, ym, x]),
                        np.float32(src[zc, yi, x]),
                        np.float32(src[zc, yp, x]),
                        np.float32(src[zp, ym, x]),
                        np.float32(src[zp, yi, x]),
                        np.float32(src[zp, yp, x]),
                    )
            else:
                z0, z1, z2 = (zm, zc, zp) if ry == 0 else (zc, zc, zc)
                y0, y1, y2 = (yi, yi, yi) if ry == 0 else (ym, yi, yp)
                if nx > 0:
                    dst[oz, y, 0] = _median9_plane_xy(src, z0, z1, z2, y0, y1, y2, _mirror(-1, nx), 0, _mirror(1, nx))
                for x in range(1, nx - 1):
                    dst[oz, y, x] = _median9_plane_xy(src, z0, z1, z2, y0, y1, y2, x - 1, x, x + 1)
                if nx > 1:
                    e = nx - 1
                    dst[oz, y, e] = _median9_plane_xy(src, z0, z1, z2, y0, y1, y2, e - 1, e, _mirror(e + 1, nx))


@njit(parallel=True)
def _median_small_strip(
    src: np.ndarray,
    src_z0: int,
    dst: np.ndarray,
    dst_z0: int,
    n_ext: int,
    mz: int,
    my: int,
    mx: int,
) -> None:
    """General small-box median (for 1-D or non-3x3 boxes with ``mz * my * mx <= 9``)."""
    out_nz, ny, nx = dst.shape
    rz, ry, rx = mz // 2, my // 2, mx // 2
    n_elem = mz * my * mx
    mid = n_elem // 2
    for oz in prange(out_nz):
        gz = np.int64(dst_z0) + np.int64(oz)
        buf = np.empty(n_elem, dtype=np.float32)
        for y in range(ny):
            yi = np.int64(y)
            for x in range(nx):
                xi = np.int64(x)
                k = 0
                for dz in range(-rz, rz + 1):
                    zz = _mirror(gz + dz, n_ext) - src_z0
                    for dy in range(-ry, ry + 1):
                        yy = _mirror(yi + dy, ny)
                        for dx in range(-rx, rx + 1):
                            buf[k] = np.float32(src[zz, yy, _mirror(xi + dx, nx)])
                            k += 1
                for i in range(1, n_elem):
                    v = buf[i]
                    j = i - 1
                    while j >= 0 and buf[j] > v:
                        buf[j + 1] = buf[j]
                        j -= 1
                    buf[j + 1] = v
                dst[oz, y, x] = buf[mid]


@njit(parallel=True)
def _conv_z_strip(
    src: np.ndarray,
    dst: np.ndarray,
    src_z0: int,
    dst_z0: int,
    n_ext: int,
    k: np.ndarray,
) -> None:
    """1-D Z convolution with contiguous unit-stride X inner loops (L1-resident output row)."""
    out_nz, ny, nx = dst.shape
    k_len = len(k)
    rad = k_len // 2
    for oz in prange(out_nz):
        gz = np.int64(dst_z0) + np.int64(oz)
        sz0 = _mirror(gz - rad, n_ext) - src_z0
        w0 = k[0]
        for y in range(ny):
            for x in range(nx):
                dst[oz, y, x] = w0 * np.float32(src[sz0, y, x])
            for t in range(1, k_len):
                sz = _mirror(gz + t - rad, n_ext) - src_z0
                w = k[t]
                for x in range(nx):
                    dst[oz, y, x] += w * np.float32(src[sz, y, x])


@njit(parallel=True)
def _conv_y_strip(src: np.ndarray, dst: np.ndarray, k: np.ndarray) -> None:
    """1-D Y convolution with contiguous unit-stride X inner loops (L1-resident output row)."""
    nz, ny, nx = dst.shape
    k_len = len(k)
    rad = k_len // 2
    for z in prange(nz):
        for y in range(ny):
            yi = np.int64(y)
            yy0 = _mirror(yi - rad, ny)
            w0 = k[0]
            for x in range(nx):
                dst[z, y, x] = w0 * np.float32(src[z, yy0, x])
            for t in range(1, k_len):
                yy = _mirror(yi + t - rad, ny)
                w = k[t]
                for x in range(nx):
                    dst[z, y, x] += w * np.float32(src[z, yy, x])


@njit(parallel=True)
def _conv_x_strip(src: np.ndarray, dst: np.ndarray, k: np.ndarray) -> None:
    """1-D X convolution with branchless interior loop and mirrored borders."""
    nz, ny, nx = dst.shape
    k_len = len(k)
    rad = k_len // 2
    left_end = min(rad, nx)
    right_start = max(left_end, nx - rad)
    for z in prange(nz):
        for y in range(ny):
            for x in range(left_end):
                xi = np.int64(x)
                s = np.float32(0.0)
                for t in range(k_len):
                    s += k[t] * np.float32(src[z, y, _mirror(xi + t - rad, nx)])
                dst[z, y, x] = s
            for x in range(left_end, right_start):
                s = np.float32(0.0)
                base = x - rad
                for t in range(k_len):
                    s += k[t] * np.float32(src[z, y, base + t])
                dst[z, y, x] = s
            for x in range(right_start, nx):
                xi = np.int64(x)
                s = np.float32(0.0)
                for t in range(k_len):
                    s += k[t] * np.float32(src[z, y, _mirror(xi + t - rad, nx)])
                dst[z, y, x] = s


@njit(parallel=True)
def _highpass_subtract(orig: np.ndarray, orig_z0: int, blurred: np.ndarray, dst: np.ndarray, add: np.float32) -> None:
    """Compute ``orig[orig_z0 + z] - blurred[z] + add`` into ``dst``."""
    nz, ny, nx = dst.shape
    for z in prange(nz):
        sz = orig_z0 + z
        for y in range(ny):
            for x in range(nx):
                dst[z, y, x] = orig[sz, y, x] - blurred[z, y, x] + add


@njit(inline="always")
def _median_block(src: np.ndarray, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int, buf: np.ndarray) -> float:
    k = 0
    for zz in range(z0, z1):
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                buf[k] = np.float32(src[zz, yy, xx])
                k += 1
    for i in range(1, k):
        v = buf[i]
        j = i - 1
        while j >= 0 and buf[j] > v:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = v
    return float(0.5 * (buf[k // 2 - 1] + buf[k // 2])) if k % 2 == 0 else float(buf[k // 2])


@njit(inline="always")
def _reduce_block(
    src: np.ndarray,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    mode_code: int,
    buf: np.ndarray,
) -> float:
    if mode_code == 4:  # nearest (block center)
        return float(src[(z0 + z1 - 1) // 2, (y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2])
    if mode_code == 1:  # median
        return _median_block(src, z0, z1, y0, y1, x0, x1, buf)
    if mode_code == 2:  # max
        best = np.float32(-1e30)
        for zz in range(z0, z1):
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    best = max(best, np.float32(src[zz, yy, xx]))
        return float(best)
    total = 0.0
    cnt = 0
    for zz in range(z0, z1):
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                total += float(src[zz, yy, xx])
                cnt += 1
    return total if mode_code == 3 else (total / cnt)


@njit(parallel=True)
def _bin_strip_into_u16(
    src: np.ndarray,
    src_z0: int,
    src_nz: int,
    dst: np.ndarray,
    dst_z0: int,
    fz: int,
    fy: int,
    fx: int,
    mode_code: int,
    lut: np.ndarray,
) -> None:
    """Bin ``src[src_z0 : src_z0 + src_nz]`` by ``(fz, fy, fx)`` into ``dst[dst_z0:]`` with optional fused LUT."""
    _, ny, nx = src.shape
    out_nz = (src_nz + fz - 1) // fz
    out_ny = dst.shape[1]
    out_nx = dst.shape[2]
    max_blk = fz * fy * fx
    use_lut = lut.size == 65536

    for oz in prange(out_nz):
        z_start = src_z0 + oz * fz
        z_end = min(z_start + fz, src_z0 + src_nz)
        buf = np.empty(max_blk, dtype=np.float32) if mode_code == 1 else np.empty(1, dtype=np.float32)
        for oy in range(out_ny):
            y_start = oy * fy
            y_end = min(y_start + fy, ny)
            for ox in range(out_nx):
                x_start = ox * fx
                x_end = min(x_start + fx, nx)
                u = _clamp_u16(_reduce_block(src, z_start, z_end, y_start, y_end, x_start, x_end, mode_code, buf))
                dst[dst_z0 + oz, oy, ox] = lut[u] if use_lut else u


@njit(parallel=True)
def _copy_f32_to_u16_strip(src: np.ndarray, dst: np.ndarray, dst_z0: int, lut: np.ndarray) -> None:
    """Round and clip a float32 strip into ``uint16`` ``dst[dst_z0:]`` with optional fused LUT."""
    nz, ny, nx = src.shape
    use_lut = lut.size == 65536
    for z in prange(nz):
        for y in range(ny):
            for x in range(nx):
                u = _clamp_u16(float(src[z, y, x]))
                dst[dst_z0 + z, y, x] = lut[u] if use_lut else u


def bin_3d_numba(
    src: np.ndarray,
    bin_zyx: tuple[int, int, int],
    mode: BinMode = "mean",
    *,
    lut: np.ndarray | None = None,
) -> np.ndarray:
    """Bin a 3-D volume by ``bin_zyx`` into a new ``uint16`` array, optionally fusing a ``uint16`` LUT."""
    fz, fy, fx = bin_zyx
    nz, ny, nx = src.shape
    out = np.empty((math.ceil(nz / fz), math.ceil(ny / fy), math.ceil(nx / fx)), dtype=np.uint16)
    _bin_strip_into_u16(src, 0, nz, out, 0, fz, fy, fx, _BIN_MODE_CODE[mode], _EMPTY_LUT if lut is None else lut)
    return out


# ---------------------------------------------------------------------------
# High-level streaming batch processor
# ---------------------------------------------------------------------------


def _apply_gaussian_separable(
    src: np.ndarray,
    src_z0: int,
    dst_z0: int,
    dst_nz: int,
    n_ext: int,
    kernels: tuple[np.ndarray | None, ...],
    buf_first: np.ndarray,
    buf_second: np.ndarray,
) -> np.ndarray:
    """Run 3-D separable Gaussian blur using precomputed ``(kz, ky, kx)``; returns the view holding output."""
    kz, ky, kx = kernels
    view_a = buf_first[:dst_nz]
    view_b = buf_second[:dst_nz]
    if kz is not None:
        _conv_z_strip(src, view_a, src_z0, dst_z0, n_ext, kz)
        cur, scratch = view_a, view_b
    else:
        rel = dst_z0 - src_z0
        cur, scratch = src[rel : rel + dst_nz], view_a

    if ky is not None:
        _conv_y_strip(cur, scratch, ky)
        cur = scratch
        scratch = view_b if scratch is view_a else view_a

    if kx is not None:
        _conv_x_strip(cur, scratch, kx)
        cur = scratch

    if cur.dtype != np.float32:
        np.copyto(view_a, cur, casting="unsafe")
        return view_a
    return cur


def _slice_extended(
    core: np.ndarray,
    halo_lead: np.ndarray | None,
    halo_trail: np.ndarray | None,
    z0: int,
    z1: int,
    lead: int,
    buf: np.ndarray | None,
) -> np.ndarray:
    """Return ``[halo_lead | core | halo_trail][z0:z1]`` zero-copy when inside ``core``, else into ``buf``."""
    core_end = lead + core.shape[0]
    if z0 >= lead and z1 <= core_end:
        return core[z0 - lead : z1 - lead]
    if buf is None:  # pragma: no cover - allocated whenever lead > 0 or trail > 0
        raise RuntimeError("halo strip buffer not allocated")
    out = buf[: z1 - z0]
    pos = 0
    if halo_lead is not None and z0 < lead:
        hi = min(z1, lead)
        n = hi - z0
        out[pos : pos + n] = halo_lead[z0:hi]
        pos += n
    c_lo = max(z0, lead)
    c_hi = min(z1, core_end)
    if c_hi > c_lo:
        n = c_hi - c_lo
        out[pos : pos + n] = core[c_lo - lead : c_hi - lead]
        pos += n
    if halo_trail is not None and z1 > core_end:
        lo = max(z0, core_end)
        n = z1 - lo
        out[pos : pos + n] = halo_trail[lo - core_end : z1 - core_end]
    return out


def _run_median_strip(
    src: np.ndarray,
    src_z0: int,
    dst: np.ndarray,
    dst_z0: int,
    n_ext: int,
    median_zyx: tuple[int, int, int],
) -> None:
    mz, my, mx = median_zyx
    if mz * my * mx == 9 and (mz == 3 or my == 3 or mx == 3) and min(mz, my, mx) == 1:
        _median_9_strip(src, src_z0, dst, dst_z0, n_ext, mz, my, mx)
    else:
        _median_small_strip(src, src_z0, dst, dst_z0, n_ext, mz, my, mx)


def process_batch_l0(
    core: np.ndarray,
    spec: PreprocessSpec,
    *,
    halo_lead: np.ndarray | None = None,
    halo_trail: np.ndarray | None = None,
    lut: np.ndarray | None = None,
    finest_axis: int = 0,
) -> tuple[np.ndarray, float | None, float | None]:
    """Run configured Squish filters, per-axis binning, and LUT quantization on one L0 batch.

    Never mutates ``core`` (safe while ``FrameLease`` holds a view of ``core``).
    Returns ``(processed_u16, resolved_offset, resolved_sigma_y)``.
    """
    if not spec.is_active:
        return core, None, None

    # Pre-resolve LUT when it doesn't depend on the filtered/binned batch output (enables fused single-pass write).
    fused_lut: np.ndarray | None = None
    if spec.has_quantize and core.dtype == np.uint16:
        if lut is not None:
            fused_lut = lut
        elif not spec.needs_auto_sqrt:
            fused_lut = spec.build_lut()

    # Fast path 1: quantization LUT only
    if not spec.has_filters and not spec.has_binning:
        out_u16 = np.empty_like(core)
        resolved_offset, resolved_sigma_y, active_lut = _resolve_lut(core, spec, fused_lut, finest_axis)
        if active_lut is not None and core.dtype == np.uint16:
            apply_lut_u16(core, active_lut, out_u16)
        else:
            np.copyto(out_u16, core)
        return out_u16, resolved_offset, resolved_sigma_y

    # Fast path 2: binning only (fuses LUT directly when already known)
    if not spec.has_filters:
        out_u16 = bin_3d_numba(core, spec.bin_zyx, spec.bin_mode, lut=fused_lut)
    else:
        out_u16 = _run_streaming_filters(
            core,
            spec,
            halo_lead=halo_lead,
            halo_trail=halo_trail,
            mode_code=_BIN_MODE_CODE[spec.bin_mode],
            fused_lut=fused_lut,
        )

    # Deferred LUT pass only when auto-sqrt had to inspect the filtered+binned batch 0 first.
    if fused_lut is not None or not spec.has_quantize:
        return out_u16, spec.noise_offset, spec.noise_sigma_y
    resolved_offset, resolved_sigma_y, active_lut = _resolve_lut(out_u16, spec, None, finest_axis)
    if active_lut is not None and out_u16.dtype == np.uint16:
        apply_lut_u16(out_u16, active_lut, out_u16)
    return out_u16, resolved_offset, resolved_sigma_y


def _run_streaming_filters(
    core: np.ndarray,
    spec: PreprocessSpec,
    *,
    halo_lead: np.ndarray | None,
    halo_trail: np.ndarray | None,
    mode_code: int,
    fused_lut: np.ndarray | None,
) -> np.ndarray:
    """Execute median -> low-pass -> high-pass -> bin (+ optional fused LUT) in bounded Z-strips."""
    nz, ny, nx = core.shape
    fz, fy, fx = spec.bin_zyx
    lead = 0 if halo_lead is None else int(halo_lead.shape[0])
    trail = 0 if halo_trail is None else int(halo_trail.shape[0])
    n_ext = lead + nz + trail
    r_med_z, r_lp_z, r_hp_z = spec.z_radii

    lp_kernels = _kernels_for(spec.lowpass_cutoff_zyx) if spec.has_lowpass else (None, None, None)
    hp_kernels = _kernels_for(spec.highpass_cutoff_zyx) if spec.has_highpass else (None, None, None)
    lut_arr = _EMPTY_LUT if fused_lut is None else fused_lut

    out_u16 = np.empty((math.ceil(nz / fz), math.ceil(ny / fy), math.ceil(nx / fx)), dtype=np.uint16)
    strip_z = spec.strip_z(nz)
    max_strip_planes = min(n_ext, strip_z + 2 * (r_lp_z + r_hp_z))

    buf_in = (
        np.empty((min(n_ext, strip_z + 2 * (r_med_z + r_lp_z + r_hp_z)), ny, nx), dtype=core.dtype)
        if (lead > 0 or trail > 0)
        else None
    )
    buf_a = np.empty((max_strip_planes, ny, nx), dtype=np.float32)
    buf_b = np.empty((max_strip_planes, ny, nx), dtype=np.float32)
    buf_hp_src = np.empty((min(n_ext, strip_z + 2 * r_hp_z), ny, nx), dtype=np.float32) if spec.has_highpass else None

    for rel_c0 in range(0, nz, strip_z):
        rel_c1 = min(rel_c0 + strip_z, nz)
        c0, c1 = lead + rel_c0, lead + rel_c1
        core_len = c1 - c0

        # Stage 1: Median (valid over [m0, m1) needed by LP + HP)
        m0 = max(0, c0 - r_lp_z - r_hp_z)
        m1 = min(n_ext, c1 + r_lp_z + r_hp_z)
        in_z0 = 0 if n_ext <= r_med_z + 1 else max(0, m0 - r_med_z)
        in_z1 = n_ext if n_ext <= r_med_z + 1 else min(n_ext, m1 + r_med_z)
        raw_slice = _slice_extended(core, halo_lead, halo_trail, in_z0, in_z1, lead, buf_in)

        if spec.has_median:
            cur = buf_a[: m1 - m0]
            _run_median_strip(raw_slice, in_z0, cur, m0, n_ext, spec.median_zyx)
            cur_z0 = m0
        else:
            cur, cur_z0 = raw_slice, in_z0

        # Stage 2: Low-pass Gaussian (valid over [l0, l1) needed by HP)
        l0 = max(0, c0 - r_hp_z)
        l1 = min(n_ext, c1 + r_hp_z)
        l_len = l1 - l0
        if spec.has_lowpass:
            first, second = (buf_b, buf_a) if np.shares_memory(cur, buf_a) else (buf_a, buf_b)
            cur = _apply_gaussian_separable(cur, cur_z0, l0, l_len, n_ext, lp_kernels, first, second)
            cur_z0 = l0

        # Stage 3: High-pass Gaussian (valid over [c0, c1))
        if spec.has_highpass:
            if buf_hp_src is None:  # pragma: no cover - allocated whenever spec.has_highpass
                raise RuntimeError("high-pass scratch buffer not allocated")
            hp_src = buf_hp_src[:l_len]
            rel_l = l0 - cur_z0
            np.copyto(hp_src, cur[rel_l : rel_l + l_len], casting="unsafe")
            blurred = _apply_gaussian_separable(hp_src, l0, c0, core_len, n_ext, hp_kernels, buf_a, buf_b)
            dst_hp = buf_b[:core_len] if np.shares_memory(blurred, buf_a) else buf_a[:core_len]
            _highpass_subtract(hp_src, c0 - l0, blurred, dst_hp, np.float32(spec.highpass_add))
            cur, cur_z0 = dst_hp, c0

        # Stage 4: Bin (or cast) core [c0, c1) directly into out_u16 (with fused LUT when known)
        rel_out = c0 - cur_z0
        dst_z0 = rel_c0 // fz
        if spec.has_binning:
            _bin_strip_into_u16(cur, rel_out, core_len, out_u16, dst_z0, fz, fy, fx, mode_code, lut_arr)
        elif cur.dtype == np.float32:
            _copy_f32_to_u16_strip(cur[rel_out : rel_out + core_len], out_u16, dst_z0, lut_arr)
        elif fused_lut is not None and cur.dtype == np.uint16:
            apply_lut_u16(cur[rel_out : rel_out + core_len], fused_lut, out_u16[dst_z0 : dst_z0 + core_len])
        else:
            out_u16[dst_z0 : dst_z0 + core_len] = cur[rel_out : rel_out + core_len]

    return out_u16


def _resolve_lut(
    vol: np.ndarray,
    spec: PreprocessSpec,
    provided_lut: np.ndarray | None,
    finest_axis: int,
) -> tuple[float | None, float | None, np.ndarray | None]:
    if not spec.has_quantize:
        return None, None, None
    if provided_lut is not None:
        return spec.noise_offset, spec.noise_sigma_y, provided_lut

    offset, sigma_y = spec.noise_offset, spec.noise_sigma_y
    if spec.needs_auto_sqrt:
        est_off, est_sig = estimate_sqrt_params(vol, finest_axis=finest_axis)
        offset = est_off if offset is None else offset
        sigma_y = est_sig if sigma_y is None else sigma_y

    return offset, sigma_y, spec.build_lut(offset=offset, sigma_y=sigma_y)
