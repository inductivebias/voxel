"""End-to-end tests for lossy LUT quantization, binning, Squish Z-halo, codec defaults, and live stats."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from ome_zarr_writer import (
    Compression,
    Local,
    OMEZarrWriter,
    ScaleLevel,
    WriterConfig,
)
from ome_zarr_writer.array.ts import TSArrayReader
from ome_zarr_writer.dataset import Zarr3ArrayMeta
from ome_zarr_writer.preprocess import PreprocessSpec, build_uint16_lut, process_batch_l0
from ome_zarr_writer.sizing import RingBudgetExceededError, per_slot_bytes, ring_shm_bytes, slots_for_budget
from vxlib.dtype import Dtype
from vxlib.vector import UIVec3D, UVec3D, Vec3D

if TYPE_CHECKING:
    from pathlib import Path


def _read_l0(dataset_dir: Path) -> np.ndarray:
    reader = TSArrayReader(dataset_dir / "0")
    return reader.read_3d(z0=0, z1=reader.shape[1])


def test_codec_defaults_and_overrides() -> None:
    meta_zstd = Zarr3ArrayMeta.sharded(
        shape=[1, 64, 64, 64],
        shard_shape=[1, 64, 64, 64],
        chunk_shape=[1, 64, 64, 64],
        dtype=Dtype.UINT16,
        compression=Compression.BLOSC_ZSTD,
    )
    blosc_zstd = meta_zstd.codecs[0].configuration.codecs[1]
    assert blosc_zstd.configuration.cname == "zstd"
    assert blosc_zstd.configuration.clevel == 1
    assert blosc_zstd.configuration.shuffle == "shuffle"

    meta_lz4 = Zarr3ArrayMeta.sharded(
        shape=[1, 64, 64, 64],
        shard_shape=[1, 64, 64, 64],
        chunk_shape=[1, 64, 64, 64],
        dtype=Dtype.UINT16,
        compression=Compression.BLOSC_LZ4,
    )
    blosc_lz4 = meta_lz4.codecs[0].configuration.codecs[1]
    assert blosc_lz4.configuration.cname == "lz4"
    assert blosc_lz4.configuration.clevel == 3
    assert blosc_lz4.configuration.shuffle == "bitshuffle"

    meta_custom = Zarr3ArrayMeta.sharded(
        shape=[1, 64, 64, 64],
        shard_shape=[1, 64, 64, 64],
        chunk_shape=[1, 64, 64, 64],
        dtype=Dtype.UINT16,
        compression=Compression.BLOSC_ZSTD,
        compression_level=5,
        compression_shuffle="bitshuffle",
    )
    custom = meta_custom.codecs[0].configuration.codecs[1]
    assert custom.configuration.clevel == 5
    assert custom.configuration.shuffle == "bitshuffle"


@pytest.mark.slow
def test_frame_lease_unmodified_by_lossy_quantization(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    vol = rng.integers(100, 5000, size=(16, 64, 64), dtype=np.uint16)
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=16, y=64, x=64),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L1,
        compression=Compression.BLOSC_ZSTD,
        dark_floor=500,
        keep_bits=10,
    )
    writer = OMEZarrWriter(slots=2)
    try:
        writer.begin_stack(cfg, Local(target=tmp_path / "lease_safe"))
        for z in range(16):
            with writer.new_frame() as dst:
                np.copyto(dst, vol[z])
        # Hold FrameLease across close? Release before close, verify frame equals raw vol[-1]
        with writer.latest_frame() as lease:
            np.testing.assert_array_equal(lease.array, vol[-1])
        writer.end_stack()
    finally:
        writer.close()

    written = _read_l0(tmp_path / "lease_safe.ome.zarr")
    lut = build_uint16_lut(dark_floor=500, keep_bits=10)
    np.testing.assert_array_equal(written, lut[vol])
    assert not np.array_equal(written, vol)


@pytest.mark.slow
def test_streaming_squish_z_halo_matches_whole_volume_on_two_slot_ring(tmp_path: Path) -> None:
    """Two-slot ring with Z-halo + short final batch (< halo_z) must not deadlock and must match whole-volume Squish."""
    rng = np.random.default_rng(99)
    # L1 => chunk_shape.z=64 for unbinned; with bin_zyx=(2, 2, 2), chunk_shape.z=32, raw batch_z=64.
    # Choose total_z = 64 + 64 + 3 = 131 so:
    # - batch 0 (64 frames) waits on trailing halo from batch 1
    # - batch 1 (64 frames) waits on trailing halo from batch 2
    # - batch 2 has ONLY 3 frames (< halo_z!), exercising short-tail mirror equivalence in close().
    nz, ny, nx = 131, 64, 64
    vol = rng.integers(100, 4000, size=(nz, ny, nx), dtype=np.uint16)

    cfg = WriterConfig(
        volume_shape=UIVec3D(z=nz, y=ny, x=nx),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L1,
        compression=Compression.BLOSC_ZSTD,
        bin_zyx=(2, 2, 2),
        bin_mode="mean",
        median_zyx=(3, 1, 3),
        lowpass_cutoff_zyx=(4.0, 4.0, 4.0),
        highpass_cutoff_zyx=(12.0, 12.0, 12.0),
        highpass_add=500.0,
        lossy_sigma=1.0,
        noise_offset=100.0,
        noise_sigma_y=1.5,
    )
    assert cfg.batch_z == 64
    assert cfg.preprocess.halo_z > 3

    writer = OMEZarrWriter(slots=2)
    try:
        writer.begin_stack(cfg, Local(target=tmp_path / "squish_halo"))
        for z in range(nz):
            with writer.new_frame() as dst:
                np.copyto(dst, vol[z])
        writer.end_stack()
        stats = writer.live_compression_stats()
    finally:
        writer.close()

    assert stats is not None
    assert stats.raw_bytes == vol.nbytes
    assert stats.written_bytes > 0
    assert stats.compression_ratio > 1.0

    written = _read_l0(tmp_path / "squish_halo.ome.zarr")
    lut = build_uint16_lut(lossy_sigma=1.0, offset=100.0, sigma_y=1.5)
    expected, _, _ = process_batch_l0(vol, cfg.preprocess, lut=lut)
    assert written.shape == expected.shape
    np.testing.assert_array_equal(written, expected)


@pytest.mark.slow
def test_auto_sqrt_batch0_consistency_and_compression_gain(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    # 2 batches of 64 frames each with realistic shot+readout noise
    signal_e = rng.uniform(20.0, 800.0, size=(128, 64, 64))
    noisy = 100.0 + 2.0 * rng.poisson(signal_e) + rng.normal(0.0, 2.5, size=signal_e.shape)
    vol = np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)

    cfg_lossless = WriterConfig(
        volume_shape=UIVec3D(z=128, y=64, x=64),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L1,
        compression=Compression.BLOSC_ZSTD,
    )
    cfg_lossy = WriterConfig(
        volume_shape=UIVec3D(z=128, y=64, x=64),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        max_level=ScaleLevel.L1,
        compression=Compression.BLOSC_ZSTD,
        lossy_sigma=1.0,
    )

    writer = OMEZarrWriter(slots=2)
    try:
        writer.begin_stack(cfg_lossless, Local(target=tmp_path / "lossless"))
        for z in range(128):
            with writer.new_frame() as dst:
                np.copyto(dst, vol[z])
        writer.end_stack()
        stats_lossless = writer.live_compression_stats()

        writer.begin_stack(cfg_lossy, Local(target=tmp_path / "lossy"))
        for z in range(128):
            with writer.new_frame() as dst:
                np.copyto(dst, vol[z])
        writer.end_stack()
        stats_lossy = writer.live_compression_stats()
    finally:
        writer.close()

    assert stats_lossless is not None
    assert stats_lossy is not None
    assert stats_lossy.written_bytes < stats_lossless.written_bytes
    assert stats_lossy.raw_bytes == vol.nbytes

    # Check resolved noise_offset & noise_sigma_y were recorded in zarr.json and used identically across both batches
    l0_meta = json.loads((tmp_path / "lossy.ome.zarr" / "0" / "zarr.json").read_text())
    pp_meta = l0_meta["attributes"]["voxel_preprocessing"]
    assert "noise_offset" in pp_meta
    assert "noise_sigma_y" in pp_meta
    lut = build_uint16_lut(
        lossy_sigma=1.0,
        offset=pp_meta["noise_offset"],
        sigma_y=pp_meta["noise_sigma_y"],
    )
    written = _read_l0(tmp_path / "lossy.ome.zarr")
    np.testing.assert_array_equal(written, lut[vol])


def test_binned_ome_metadata_and_oblique_shear_coordinates() -> None:
    cfg = WriterConfig(
        volume_shape=UIVec3D(z=128, y=128, x=128),
        voxel_size=UVec3D(z=1.0, y=0.5, x=0.5),
        origin=Vec3D(z=10.0, y=20.0, x=30.0),
        shear_xz=0.5,
        shear_yz=0.0,
        max_level=ScaleLevel.L2,
        bin_zyx=(2, 4, 2),
    )
    assert cfg.binned_volume_shape == UIVec3D(z=64, y=32, x=64)
    assert cfg.binned_voxel_size == UVec3D(z=2.0, y=2.0, x=1.0)
    assert cfg.binned_origin == Vec3D(z=10.5, y=20.75, x=30.25)

    # Verify L0 binned voxel (kz, ky, kx) maps through scale -> translation -> shear to the exact
    # mean physical coordinate of its constituent raw voxels.
    ds = cfg.dataset
    l0_transforms = ds.group.attributes.ome.multiscales[0].datasets[0].coordinateTransformations[0].transformations
    scale = np.array(l0_transforms[0].scale[1:])
    trans = np.array(l0_transforms[1].translation[1:])
    affine = np.array(l0_transforms[2].affine)

    def map_binned(kz: int, ky: int, kx: int) -> np.ndarray:
        p = np.array([kz, ky, kx], dtype=np.float64) * scale + trans
        v = np.array([0.0, p[0], p[1], p[2], 1.0])
        return (affine @ v)[1:4]

    def map_raw(z: int, y: int, x: int) -> np.ndarray:
        p = np.array([z * 1.0 + 10.0, y * 0.5 + 20.0, x * 0.5 + 30.0], dtype=np.float64)
        v = np.array([0.0, p[0], p[1], p[2], 1.0])
        return (affine @ v)[1:4]

    kz, ky, kx = 3, 2, 5
    raw_centers = [
        map_raw(z, y, x)
        for z in range(kz * 2, kz * 2 + 2)
        for y in range(ky * 4, ky * 4 + 4)
        for x in range(kx * 2, kx * 2 + 2)
    ]
    np.testing.assert_allclose(map_binned(kz, ky, kx), np.mean(raw_centers, axis=0))


def test_sizing_accounts_for_preprocess_and_halos() -> None:
    shape = UIVec3D(z=64, y=512, x=512)
    base_slot = per_slot_bytes(shape, ScaleLevel.L3, Dtype.UINT16)
    base_shm = ring_shm_bytes(2, shape, ScaleLevel.L3, Dtype.UINT16)

    spec_filt = PreprocessSpec(lowpass_cutoff_zyx=(0.25, 0.25, 0.25))
    filt_slot = per_slot_bytes(shape, ScaleLevel.L3, Dtype.UINT16, spec_filt)
    filt_shm = ring_shm_bytes(2, shape, ScaleLevel.L3, Dtype.UINT16, spec_filt)
    assert filt_slot > base_slot
    assert filt_shm > base_shm  # halo buffers accounted for

    # Tight budget that fits 2 lossless slots should fail when filter scratch pushes past budget
    with pytest.raises(RingBudgetExceededError):
        slots_for_budget(
            shape,
            ScaleLevel.L3,
            Dtype.UINT16,
            budget_bytes=2 * base_slot,
            preprocess=spec_filt,
        )
