"""Unit tests for CPU Squish preprocessing, per-axis binning, and uint16 LUT quantization."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ome_zarr_writer import Compression, ScaleLevel, WriterConfig, WriterSettings
from ome_zarr_writer.preprocess import (
    PreprocessSpec,
    apply_lut_u16,
    bin_3d_numba,
    binned_shape,
    build_uint16_lut,
    cutoff_to_sigma,
    estimate_sqrt_params,
    gaussian_kernel_1d,
    process_batch_l0,
)
from pydantic import ValidationError
from vxlib.dtype import Dtype
from vxlib.vector import UIVec3D, UVec3D


def _mirror_idx(i: int, n: int) -> int:
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        if i >= n:
            i = 2 * n - 2 - i
    return i


def _ref_convolve_1d_mirror(vol: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Reference separable 1D convolution with whole-sample symmetric mirroring."""
    r = (kernel.size - 1) // 2
    n = vol.shape[axis]
    out = np.zeros_like(vol, dtype=np.float32)
    moved_in = np.moveaxis(vol.astype(np.float32), axis, 0)
    moved_out = np.zeros_like(moved_in)
    for i in range(n):
        acc = np.zeros_like(moved_in[0], dtype=np.float32)
        for k in range(-r, r + 1):
            acc += kernel[k + r] * moved_in[_mirror_idx(i + k, n)]
        moved_out[i] = acc
    out[...] = np.moveaxis(moved_out, 0, axis)
    return out


def _ref_median_3d(vol: np.ndarray, kzyx: tuple[int, int, int]) -> np.ndarray:
    mz, my, mx = kzyx
    rz, ry, rx = mz // 2, my // 2, mx // 2
    nz, ny, nx = vol.shape
    out = np.empty_like(vol)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                vals = [
                    vol[_mirror_idx(z + dz, nz), _mirror_idx(y + dy, ny), _mirror_idx(x + dx, nx)]
                    for dz in range(-rz, rz + 1)
                    for dy in range(-ry, ry + 1)
                    for dx in range(-rx, rx + 1)
                ]
                vals.sort()
                out[z, y, x] = vals[len(vals) // 2]
    return out


def test_lut_dark_floor_and_keep_bits() -> None:
    lut = build_uint16_lut(dark_floor=115, keep_bits=12)
    assert lut.dtype == np.uint16
    assert lut.shape == (65536,)
    assert np.all(lut[:116] == 0)
    # 12-bit rounding preserves grey-level scale (steps of 16)
    high = lut[116:]
    assert np.all((high & 0xF) == 0)
    assert int(lut[1000]) == 1008 or int(lut[1000]) == 992
    # Maximum quantization error up to top representable 12-bit code (65520) is <= 8 (half of 16)
    mid = lut[116:65521]
    assert int(np.max(np.abs(mid.astype(np.int32) - np.arange(116, 65521, dtype=np.int32)))) <= 8


def test_lut_sqrt_noise_bounded_rmse_and_auto_estimation() -> None:
    rng = np.random.default_rng(42)
    # Synthetic microscopy data: smooth fluorescence field + camera offset 100 + Poisson gain 2.0 + readout noise 2.5
    offset_true = 100.0
    gain = 2.0
    zz, yy, xx = np.meshgrid(
        np.linspace(0.0, 2.0 * np.pi, 32),
        np.linspace(0.0, 2.0 * np.pi, 64),
        np.linspace(0.0, 2.0 * np.pi, 64),
        indexing="ij",
    )
    signal_e = 10.0 + 1500.0 * (0.5 + 0.5 * np.sin(xx) * np.cos(yy) * np.cos(zz))
    noisy = offset_true + gain * rng.poisson(signal_e) + rng.normal(0.0, 2.5, size=signal_e.shape)
    vol = np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)

    est_offset, est_sigma_y = estimate_sqrt_params(vol, finest_axis=2)
    # Under Anscombe transform y = 2*sqrt(max(0, x-offset) + 3/8), for gain=2 ADU/e-, sigma_y ~= sqrt(gain) = 1.41
    assert 80.0 <= est_offset <= 160.0
    assert 0.8 <= est_sigma_y <= 2.2

    lut = build_uint16_lut(lossy_sigma=1.0, offset=est_offset, sigma_y=est_sigma_y)
    dst = np.empty_like(vol)
    apply_lut_u16(vol, lut, dst)

    # Verify error in Anscombe space is bounded by step/2 and RMSE ~= step / sqrt(12)
    step = 1.0 * est_sigma_y
    y_raw = 2.0 * np.sqrt(np.maximum(0.0, vol.astype(np.float64) - est_offset) + 0.375)
    y_q = 2.0 * np.sqrt(np.maximum(0.0, dst.astype(np.float64) - est_offset) + 0.375)
    rmse_y = float(np.sqrt(np.mean((y_q - y_raw) ** 2)))
    assert rmse_y < 0.45 * step
    # Mean intensity is preserved without bias
    assert abs(float(dst.mean()) - float(vol.mean())) < 2.0


def test_sqrt_lut_preserves_values_at_and_below_offset() -> None:
    lut = build_uint16_lut(lossy_sigma=1.0, offset=100.0, sigma_y=1.0)
    np.testing.assert_array_equal(lut[:101], np.arange(101, dtype=np.uint16))


def test_preprocessing_rejects_non_uint16() -> None:
    with pytest.raises(ValueError, match="requires uint16"):
        process_batch_l0(np.ones((1, 2, 2), dtype=np.uint8), PreprocessSpec(dark_floor=2))
    with pytest.raises(ValueError, match="requires uint16"):
        process_batch_l0(np.ones((1, 2, 2), dtype=np.float32), PreprocessSpec(bin_zyx=(1, 2, 2)))
    with pytest.raises(ValidationError, match="requires uint16"):
        WriterConfig(
            volume_shape=UIVec3D(z=1, y=2, x=2), voxel_size=UVec3D(z=1, y=1, x=1), dtype=Dtype.UINT8, dark_floor=2
        )
    WriterConfig(volume_shape=UIVec3D(z=1, y=2, x=2), voxel_size=UVec3D(z=1, y=1, x=1), dtype=Dtype.UINT8)


@pytest.mark.parametrize("mode", ["mean", "median", "max", "sum", "nearest"])
def test_bin_3d_numba_modes_and_partial_edges(mode: str) -> None:
    rng = np.random.default_rng(7)
    # Deliberately non-divisible shape (7, 9, 11) with bin (2, 3, 2) -> (4, 3, 6)
    vol = rng.integers(0, 20000, size=(7, 9, 11), dtype=np.uint16)
    got = bin_3d_numba(vol, (2, 3, 2), mode)  # type: ignore[arg-type]
    assert got.shape == (4, 3, 6)
    assert got.dtype == np.uint16

    for bz in range(4):
        z0, z1 = bz * 2, min((bz + 1) * 2, 7)
        for by in range(3):
            y0, y1 = by * 3, min((by + 1) * 3, 9)
            for bx in range(6):
                x0, x1 = bx * 2, min((bx + 1) * 2, 11)
                block = vol[z0:z1, y0:y1, x0:x1]
                if mode == "mean":
                    exp = math.floor(float(block.astype(np.float64).mean()) + 0.5)
                elif mode == "median":
                    exp = math.floor(float(np.median(block.astype(np.float64))) + 0.5)
                elif mode == "max":
                    exp = int(block.max())
                elif mode == "sum":
                    exp = min(65535, int(block.astype(np.uint64).sum()))
                else:
                    cz, cy, cx = (z0 + z1 - 1) // 2, (y0 + y1 - 1) // 2, (x0 + x1 - 1) // 2
                    exp = int(vol[cz, cy, cx])
                assert int(got[bz, by, bx]) == exp


def test_binned_shape_ceil() -> None:
    assert binned_shape(UIVec3D(z=7, y=9, x=11), (2, 3, 2)) == UIVec3D(z=4, y=3, x=6)
    assert binned_shape(UIVec3D(z=64, y=64, x=64), (1, 1, 1)) == UIVec3D(z=64, y=64, x=64)


def test_squish_filters_match_reference_and_stream_with_halo() -> None:
    rng = np.random.default_rng(123)
    vol = rng.integers(100, 3000, size=(24, 20, 18), dtype=np.uint16)

    # 1. Median (1x3x3) and (3x1x3) match exact reference
    for kzyx in ((1, 3, 3), (3, 1, 3), (3, 1, 1)):
        spec = PreprocessSpec(median_zyx=kzyx)
        got, _, _ = process_batch_l0(vol, spec)
        ref = _ref_median_3d(vol, kzyx)
        np.testing.assert_array_equal(got, ref)

    # 2. Separable Gaussian lowpass matches reference convolution (voxel-wavelength cutoffs)
    lp = (4.0, 5.0, 3.0)
    spec_lp = PreprocessSpec(lowpass_cutoff_zyx=lp)
    got_lp, _, _ = process_batch_l0(vol, spec_lp)
    ref_f = vol.astype(np.float32)
    for ax, c in enumerate(lp):
        ref_f = _ref_convolve_1d_mirror(ref_f, gaussian_kernel_1d(cutoff_to_sigma(c)), ax)
    ref_u16 = np.clip(np.rint(ref_f), 0, 65535).astype(np.uint16)
    assert int(np.max(np.abs(got_lp.astype(np.int32) - ref_u16.astype(np.int32)))) <= 1

    # 3. Full pipeline (median + lowpass + highpass + bin + LUT) split into batches with halo matches whole volume
    spec_full = PreprocessSpec(
        median_zyx=(3, 1, 3),
        lowpass_cutoff_zyx=(4.0, 4.0, 4.0),
        highpass_cutoff_zyx=(12.0, 12.0, 12.0),
        highpass_add=500.0,
        bin_zyx=(2, 2, 2),
        bin_mode="mean",
        lossy_sigma=1.0,
        noise_offset=100.0,
        noise_sigma_y=1.5,
    )
    lut = build_uint16_lut(lossy_sigma=1.0, offset=100.0, sigma_y=1.5)
    whole, _, _ = process_batch_l0(vol, spec_full, lut=lut)

    # Split into 2 batches (16 + 8) with halo_z = spec_full.halo_z
    hz = spec_full.halo_z
    assert hz >= 8
    b0, _, _ = process_batch_l0(vol[:16], spec_full, halo_lead=None, halo_trail=vol[16 : 16 + hz], lut=lut)
    b1, _, _ = process_batch_l0(vol[16:], spec_full, halo_lead=vol[16 - hz : 16], halo_trail=None, lut=lut)
    stitched = np.concatenate([b0, b1], axis=0)
    np.testing.assert_array_equal(stitched, whole)


def test_preprocess_and_writer_settings_validation() -> None:
    # 3x3x3 median rejected on CPU (> 9 elements)
    with pytest.raises(ValidationError, match="CPU median"):
        PreprocessSpec(median_zyx=(3, 3, 3))
    with pytest.raises(ValidationError, match="odd"):
        PreprocessSpec(median_zyx=(2, 1, 1))
    with pytest.raises(ValidationError, match="lowpass_cutoff_zyx"):
        PreprocessSpec(lowpass_cutoff_zyx=(-1.0, 0.0, 0.0))
    with pytest.raises(ValidationError, match="both be set"):
        PreprocessSpec(noise_offset=100.0)
    with pytest.raises(ValidationError, match="compression_level must be <= 9"):
        WriterSettings(compression=Compression.BLOSC_ZSTD, compression_level=15)
    for codec in (Compression.GZIP, Compression.LZ4, Compression.BLOSC_LZ4, Compression.BLOSC_ZSTD):
        WriterSettings(compression=codec, compression_level=9)
        with pytest.raises(ValidationError, match="compression_level must be <= 9"):
            WriterSettings(compression=codec, compression_level=10)
    WriterSettings(compression=Compression.ZSTD, compression_level=22)
    # Very large Z wavelength produces halo_z > batch_z (64 for L0) -> rejected cleanly
    with pytest.raises(ValidationError, match="Filter Z halo"):
        WriterSettings(max_level=ScaleLevel.L0, highpass_cutoff_zyx=(200.0, 0.0, 0.0))
    # Z-binning scales chunk_shape.z down while keeping >= max_level.factor
    ws = WriterSettings(max_level=ScaleLevel.L3, bin_zyx=(4, 2, 2))
    assert ws.chunk_shape.z == 16
    assert ws.binned_batch_z == 16
    assert ws.batch_z == 64
    assert ws.batch_z % ws.bin_zyx[0] == 0
    # Unbinned stays identical to legacy behavior
    ws_unbinned = WriterSettings(max_level=ScaleLevel.L3)
    assert ws_unbinned.chunk_shape == UIVec3D(z=64, y=64, x=64)
    assert ws_unbinned.batch_z == 64
    assert math.isfinite(ws_unbinned.io_timeout_s)
