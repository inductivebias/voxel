"""Tests for oblique stage-scan deskewing (`ome_zarr_writer.deskew`) and OME-Zarr v0.6 affine metadata."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from ome_zarr_writer import (
    DownscaleType,
    Dtype,
    Local,
    OmeZarrDataset,
    OMEZarrWriter,
    ScaleLevel,
    UIVec3D,
    UVec3D,
    Vec3D,
    WriterConfig,
    deskew_dataset,
    deskew_slice,
    deskew_volume,
    deskewed_width,
    shear_factor_from_angle,
    shear_xz_from_angle,
)
from ome_zarr_writer.array.ts import TSArrayReader


def test_shear_factor_from_angle_computes_expected_shift() -> None:
    assert shear_xz_from_angle(45.0) == pytest.approx(math.sqrt(0.5))
    assert shear_factor_from_angle(0.5, 0.40625, 45.0) == pytest.approx(0.5 * math.sqrt(0.5) / 0.40625)
    with pytest.raises(ValueError, match="angle_deg"):
        shear_xz_from_angle(0.0)
    with pytest.raises(ValueError, match="positive"):
        shear_factor_from_angle(0.0, 0.5, 45.0)


def test_deskew_volume_integer_and_fractional_shifts_preserve_anchor_and_dtype() -> None:
    rng = np.random.default_rng(seed=424242)
    vol_u16 = rng.integers(100, 4000, size=(5, 6, 8), dtype=np.uint16)

    # Positive integer shear (+2 px/slice): slice z shifts right by 2*z, padded with cval=7.
    out_pos = deskew_volume(vol_u16, shear_factor=2.0, cval=7.0)
    assert out_pos.dtype == np.uint16
    assert out_pos.shape == (5, 6, 8 + 8)
    for z in range(5):
        np.testing.assert_array_equal(out_pos[z, :, 2 * z : 2 * z + 8], vol_u16[z])
        assert np.all(out_pos[z, :, : 2 * z] == 7)
        assert np.all(out_pos[z, :, 2 * z + 8 :] == 7)

    # Negative fractional shear (-0.25 px/slice over 5 slices -> min_shift = -1.0):
    # last slice z=4 has shift 0.0 (anchor plane), so column 0 is preserved without cval bleed.
    vol_f32 = np.ones((5, 4, 6), dtype=np.float32) * 100.0
    out_neg = deskew_volume(vol_f32, shear_factor=-0.25, cval=0.0)
    assert out_neg.dtype == np.float32
    assert out_neg.shape == (5, 4, 7)
    np.testing.assert_allclose(out_neg[4, :, :6], 100.0)
    np.testing.assert_allclose(out_neg[4, :, 6], 0.0)
    # Slice z=3 has shift s = -0.75 - (-1.0) = +0.25 -> col 0 is 0.25*0 + 0.75*100 = 75, interior is 100, col 6 is 25.
    np.testing.assert_allclose(out_neg[3, :, 0], 75.0)
    np.testing.assert_allclose(out_neg[3, :, 1:6], 100.0)
    np.testing.assert_allclose(out_neg[3, :, 6], 25.0)


def test_deskew_slice_writes_directly_into_preallocated_buffer() -> None:
    plane = np.array([[10, 20, 30, 40]], dtype=np.uint16)
    target = np.zeros((1, 6), dtype=np.uint16)
    ret = deskew_slice(plane, 1.5, cval=0.0, out=target)
    assert ret is target
    # k = 1, frac = 0.5: out[0]=0, out[1]=0.5*0+0.5*10=5, out[2]=15, out[3]=25, out[4]=35, out[5]=0.5*40=20
    np.testing.assert_array_equal(target, [[0, 5, 15, 25, 35, 20]])


@pytest.mark.slow
@pytest.mark.parametrize("shear_xz", [0.8, -0.8])
@pytest.mark.parametrize("legacy", [False, True])
def test_deskew_dataset_streams_v06_pyramid_from_affine_metadata(tmp_path: Path, shear_xz: float, legacy: bool) -> None:
    z, y, x = 8, 16, 20
    voxel_size = UVec3D(z=0.5, y=0.4, x=0.4)
    shear_factor = shear_xz * voxel_size.z / voxel_size.x
    raw_cfg = WriterConfig(
        volume_shape=UIVec3D(z=z, y=y, x=x),
        voxel_size=voxel_size,
        dtype=Dtype.UINT16,
        max_level=ScaleLevel.L1,
        downscale_type=DownscaleType.MEAN,
        origin=Vec3D(z=37000.0, y=100.0, x=200.0),
        shear_xz=0.0 if legacy else shear_xz,
        channel_label="Bench 00",
        channel_color="00FFFF",
    )
    rng = np.random.default_rng(seed=20261108)
    raw_vol = rng.integers(50, 2000, size=(z, y, x), dtype=np.uint16)

    writer = OMEZarrWriter(slots=3)
    try:
        writer.begin_stack(raw_cfg, Local(target=tmp_path / "raw_stack"))
        for frame_src in raw_vol:
            with writer.new_frame() as frame:
                np.copyto(frame, frame_src)
        raw_path = writer.target
        writer.end_stack()
    finally:
        writer.close()

    assert raw_path is not None
    if legacy:
        # Real 0.5 metadata has flat axes/scale/translation and no affine shear.
        # An explicit shear_factor supplies the geometry when deskewing that legacy dataset.
        path = raw_path / "zarr.json"
        payload = json.loads(path.read_text())
        ome = payload["attributes"]["ome"]
        ome["version"] = "0.5"
        multiscale = ome["multiscales"][0]
        multiscale["axes"] = multiscale.pop("coordinateSystems")[0]["axes"]
        for dataset in multiscale["datasets"]:
            dataset["coordinateTransformations"] = dataset["coordinateTransformations"][0]["transformations"]
        path.write_text(json.dumps(payload))
    raw_meta = OmeZarrDataset.read_metadata(raw_path)
    assert raw_meta.group.attributes.ome.version == "0.6"
    assert raw_meta.shear_xz == pytest.approx(0.0 if legacy else shear_xz)
    assert raw_meta.voxel_size == voxel_size

    deskewed_path = deskew_dataset(
        raw_path,
        Local(target=tmp_path / "deskewed_stack"),
        shear_factor=shear_factor if legacy else None,
        cval=0.0,
        slots=3,
    )
    deskewed_meta = OmeZarrDataset.read_metadata(deskewed_path)
    expected_w = deskewed_width(x, z, shear_factor=shear_factor)
    assert deskewed_meta.group.attributes.ome.version == "0.6"
    assert deskewed_meta.shear_xz == 0.0
    assert deskewed_meta.volume_shape == UIVec3D(z=z, y=y, x=expected_w)
    assert deskewed_meta.group.attributes.ome.omero is not None
    assert deskewed_meta.group.attributes.ome.omero.channels[0].label == "Bench 00"
    assert deskewed_meta.group.attributes.ome.omero.channels[0].color == "00FFFF"

    l0_disk = TSArrayReader(deskewed_path / "0").read_3d(0, z)
    expected_l0 = deskew_volume(raw_vol, shear_factor=shear_factor, cval=0.0)
    np.testing.assert_array_equal(l0_disk, expected_l0)

    l1_disk = TSArrayReader(deskewed_path / "1").read_3d(0, z // 2)
    assert l1_disk.shape == (z // 2, y // 2, expected_w // 2)

    def world(metadata, point, level=0):
        result = np.asarray(point, dtype=float)
        for transform in metadata.group.attributes.ome.multiscales[0].datasets[level].sequence.transformations:
            spec = transform.model_dump()
            if spec["type"] == "scale":
                result *= spec["scale"]
            elif spec["type"] == "translation":
                result += spec["translation"]
            elif spec["type"] == "affine":
                result = np.asarray(spec["affine"]) @ np.append(result, 1.0)
        return result

    # Corresponding landmarks must retain world coordinates, not just pixel appearance.
    minimum = min(0.0, (z - 1) * shear_factor)
    for zi in (0, z - 1):
        point = [0.0, zi, 3.0, 4.0]
        raw_world = world(raw_meta, point)
        if legacy:
            raw_world[3] += zi * shear_factor * voxel_size.x
        deskew_world = world(deskewed_meta, [0.0, zi, 3.0, 4.0 + zi * shear_factor - minimum])
        np.testing.assert_allclose(deskew_world, raw_world)
    # L1 voxel centers coincide with the L0 grid's (2*i + 0.5) centers.
    np.testing.assert_allclose(
        world(deskewed_meta, [0.0, 1.0, 2.0, 3.0], 1), world(deskewed_meta, [0.0, 2.5, 4.5, 6.5])
    )


def test_deskew_rejects_yz_shear_before_creating_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = WriterConfig(
        volume_shape=UIVec3D(z=8, y=16, x=20),
        voxel_size=UVec3D(z=0.5, y=0.4, x=0.4),
        max_level=ScaleLevel.L0,
        shear_xz=0.8,
        shear_yz=0.2,
    )
    monkeypatch.setattr(OmeZarrDataset, "read_metadata", lambda _target: config.dataset)
    output = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="shear_yz"):
        deskew_dataset(tmp_path / "source", Local(target=output))
    assert not output.exists()
