"""Reading legacy acquisition metadata must preserve coordinates and source files."""

import json
from pathlib import Path

import pytest
from ome_zarr_writer import OmeZarrDataset, ScaleLevel, UIVec3D, UVec3D, WriterConfig
from pydantic import ValidationError


def legacy_dataset(path: Path) -> dict:
    config = WriterConfig(
        volume_shape=UIVec3D(z=8, y=16, x=16),
        voxel_size=UVec3D(z=0.5, y=0.4, x=0.4),
        max_level=ScaleLevel.L1,
    )
    config.dataset.write_metadata(path)
    root = path / "zarr.json"
    raw = json.loads(root.read_text())
    ome = raw["attributes"]["ome"]
    ome["version"] = "0.5"
    ms = ome["multiscales"][0]
    ms["axes"] = ms.pop("coordinateSystems")[0]["axes"]
    for index, level in enumerate(ms["datasets"]):
        level["coordinateTransformations"] = [
            {"type": "scale", "scale": [1, 0.5 * 2**index, 0.4 * 2**index, 0.4 * 2**index]},
            {"type": "translation", "translation": [0, 100 + index * 0.25, -20 + index * 0.2, 30 + index * 0.2]},
        ]
    root.write_text(json.dumps(raw))
    return raw


def test_read_05_normalizes_metadata_without_rewriting_source(tmp_path: Path) -> None:
    original = legacy_dataset(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("zarr.json")}
    dataset = OmeZarrDataset.read_metadata(tmp_path)
    assert dataset.group.attributes.ome.version == "0.6"
    assert dataset.volume_shape == UIVec3D(z=8, y=16, x=16)
    assert dataset.voxel_size == UVec3D(z=0.5, y=0.4, x=0.4)
    ms = dataset.group.attributes.ome.multiscales
    assert ms is not None
    legacy_levels = original["attributes"]["ome"]["multiscales"][0]["datasets"]
    for actual, legacy in zip(ms[0].datasets, legacy_levels, strict=True):
        assert actual.scale.scale == legacy["coordinateTransformations"][0]["scale"]
        assert actual.translation is not None
        assert actual.translation.translation == legacy["coordinateTransformations"][1]["translation"]
        assert actual.sequence.input is not None
        assert actual.sequence.input.path == actual.path
        assert actual.sequence.output is not None
        assert actual.sequence.output.name == "physical"
    assert {p: p.read_bytes() for p in before} == before


def test_read_metadata_rejects_unknown_version(tmp_path: Path) -> None:
    raw = legacy_dataset(tmp_path)
    raw["attributes"]["ome"]["version"] = "0.7"
    (tmp_path / "zarr.json").write_text(json.dumps(raw))
    with pytest.raises(ValidationError, match="version"):
        OmeZarrDataset.read_metadata(tmp_path)
