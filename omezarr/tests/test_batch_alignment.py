import pytest
from ome_zarr_writer import WriterConfig


@pytest.mark.parametrize("level", range(8))
@pytest.mark.parametrize("shard_chunks", [1, 2, 32])
@pytest.mark.parametrize("batch_shards", [1, 2])
def test_batches_own_complete_shards(level: int, shard_chunks: int, batch_shards: int) -> None:
    cfg = WriterConfig(
        volume_shape=(12000, 800, 800),
        voxel_size=(0.5, 3.25, 3.25),
        max_level=level,
        shard_z_chunks=shard_chunks,
        batch_z_shards=batch_shards,
    )
    assert cfg.batch_z == batch_shards * shard_chunks * cfg.chunk_shape.z
    assert cfg.batch_z % cfg.shard_shape.z == 0
