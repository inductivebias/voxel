"""Tests for WriterPatch validation and ActiveAcquisition live compression stats on Instrument."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from ome_zarr_writer import Compression, CompressionStats, ScaleLevel

from vxl.instrument import AcquisitionPhase, ActiveAcquisition, CaptureActivity, Instrument
from vxl.instrument.config import WriterPatch
from vxl.instrument.errors import OperationRejectedError


async def test_update_output_accepts_and_clears_lossy_and_squish_fields(instrument: Instrument) -> None:
    await instrument.open()
    await instrument.update_output(
        WriterPatch(
            compression=Compression.BLOSC_ZSTD,
            compression_level=3,
            compression_shuffle="shuffle",
            lossy_sigma=1.0,
            dark_floor=115,
            bin_zyx=(2, 2, 2),
            bin_mode="mean",
            median_zyx=(1, 3, 3),
            lowpass_cutoff_zyx=(0.25, 0.3, 0.3),
            highpass_cutoff_zyx=(0.05, 0.05, 0.05),
            highpass_add=200.0,
        )
    )
    out = instrument.state.value.output
    assert out.compression is Compression.BLOSC_ZSTD
    assert out.compression_level == 3
    assert out.compression_shuffle == "shuffle"
    assert out.lossy_sigma == 1.0
    assert out.dark_floor == 115
    assert out.bin_zyx == (2, 2, 2)
    assert out.median_zyx == (1, 3, 3)

    # Nullable fields can be explicitly cleared back to None
    await instrument.update_output(
        WriterPatch(
            compression_level=None,
            compression_shuffle=None,
            lossy_sigma=None,
            keep_bits=None,
        )
    )
    out = instrument.state.value.output
    assert out.compression_level is None
    assert out.compression_shuffle is None
    assert out.lossy_sigma is None


async def test_update_output_rejects_invalid_preprocess_and_codec_settings(instrument: Instrument) -> None:
    await instrument.open()

    # 3x3x3 median exceeds CPU 9-element cap
    with pytest.raises(OperationRejectedError, match="CPU median"):
        await instrument.update_output(WriterPatch(median_zyx=(3, 3, 3)))

    # Blosc compression level > 9
    with pytest.raises(OperationRejectedError, match="Blosc compression_level"):
        await instrument.update_output(WriterPatch(compression=Compression.BLOSC_ZSTD, compression_level=15))

    # Manual noise_offset without noise_sigma_y
    with pytest.raises(OperationRejectedError, match="both be set"):
        await instrument.update_output(WriterPatch(lossy_sigma=1.0, noise_offset=100.0))

    # Very large Z wavelength whose halo exceeds batch_z
    with pytest.raises(OperationRejectedError, match="Filter Z halo"):
        await instrument.update_output(
            WriterPatch(
                max_level=ScaleLevel.L0,
                shard_z_chunks=1,
                batch_z_shards=1,
                highpass_cutoff_zyx=(200.0, 0.0, 0.0),
            )
        )


async def test_writer_compression_stats_aggregation_and_active_acquisition(instrument: Instrument) -> None:
    await instrument.open()

    class _FakeCam:
        def __init__(self, stats: CompressionStats | None) -> None:
            self._stats = stats

        async def writer_compression_stats(self) -> CompressionStats | None:
            return self._stats

    channels = {
        "ch0": SimpleNamespace(
            camera=_FakeCam(
                CompressionStats(raw_bytes=1_000_000, written_bytes=250_000, raw_gb_s=1.2, compressed_gb_s=0.3)
            )
        ),
        "ch1": SimpleNamespace(
            camera=_FakeCam(
                CompressionStats(raw_bytes=1_000_000, written_bytes=200_000, raw_gb_s=1.2, compressed_gb_s=0.24)
            )
        ),
    }

    agg = await instrument._writer_compression_stats(channels)  # type: ignore[arg-type]
    assert agg is not None
    assert agg.raw_bytes == 2_000_000
    assert agg.written_bytes == 450_000
    assert agg.raw_gb_s == pytest.approx(2.4)
    assert agg.compressed_gb_s == pytest.approx(0.54)

    acq_id = uuid.uuid4()
    manifest = SimpleNamespace(id=acq_id, revision=1)
    await instrument._activity.set(
        CaptureActivity(
            acquisition=ActiveAcquisition(
                id=acq_id,
                manifest_revision=1,
                volume_index=0,
                phase=AcquisitionPhase.CAPTURING,
                frames_captured=10,
            )
        )
    )
    updated = await instrument._update_acquisition(manifest, frames_captured=64, compression=agg)  # type: ignore[arg-type]
    assert updated.raw_bytes == 2_000_000
    assert updated.written_bytes == 450_000
    assert updated.raw_gb_s == pytest.approx(2.4)
    assert updated.compressed_gb_s == pytest.approx(0.54)

    reset = await instrument._update_acquisition(manifest, reset_compression=True)  # type: ignore[arg-type]
    assert reset.raw_bytes == 0
    assert reset.written_bytes == 0
    assert reset.raw_gb_s == 0.0
    assert reset.compressed_gb_s == 0.0
