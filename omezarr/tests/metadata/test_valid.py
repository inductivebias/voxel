"""Tests for valid OME-Zarr metadata structures.

Note: Pydantic automatically validates and converts dicts to model instances.
Type checkers may warn about dict inputs, but this is expected behavior.
For maximum type safety in production code, construct models explicitly.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false

from ome_zarr_writer.dataset import (
    AffineTransform,
    ChannelAxis,
    CoordinateSystem,
    Dataset,
    DownscaleType,
    Multiscale,
    OmeMeta,
    ScaleLevel,
    ScaleTransform,
    SequenceTransform,
    SpaceUnit,
    TimeAxis,
    TranslationTransform,
    Zarr3GroupMeta,
)
from vxlib.vector import UVec3D, Vec3D


def test_minimal_2d():
    """Test minimal valid 2D image (YX)."""
    meta = OmeMeta.model_validate(
        {
            "multiscales": [
                {
                    "axes": [
                        {"type": "space", "name": "y", "unit": "micrometer"},
                        {"type": "space", "name": "x", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0, 1.0]},
                            ],
                        },
                    ],
                }
            ]
        }
    )
    assert meta.version == "0.6"
    assert len(meta.multiscales[0].axes) == 2
    assert meta.multiscales[0].coordinateSystems[0].name == "physical"
    assert len(meta.multiscales[0].datasets) == 1


def test_3d_zyx():
    """Test 3D image (ZYX)."""
    meta = OmeMeta(
        multiscales=[
            Multiscale(
                axes=[
                    {"type": "space", "name": "z", "unit": "micrometer"},
                    {"type": "space", "name": "y", "unit": "micrometer"},
                    {"type": "space", "name": "x", "unit": "micrometer"},
                ],
                datasets=[
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [0.5, 0.5, 0.5]},
                        ],
                    },
                ],
            )
        ]
    )
    assert len(meta.multiscales[0].axes) == 3


def test_2d_with_channel():
    """Test 2D image with channel axis (CYX)."""
    meta = OmeMeta(
        multiscales=[
            Multiscale(
                axes=[
                    {"type": "channel", "name": "c"},
                    {"type": "space", "name": "y", "unit": "micrometer"},
                    {"type": "space", "name": "x", "unit": "micrometer"},
                ],
                datasets=[
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                        ],
                    },
                ],
            )
        ]
    )
    assert meta.multiscales is not None
    assert len(meta.multiscales[0].axes) == 3
    assert isinstance(meta.multiscales[0].axes[0], ChannelAxis)


def test_2d_with_time():
    """Test 2D time series (TYX)."""
    meta = OmeMeta(
        multiscales=[
            Multiscale(
                axes=[
                    {"type": "time", "name": "t", "unit": "millisecond"},
                    {"type": "space", "name": "y", "unit": "micrometer"},
                    {"type": "space", "name": "x", "unit": "micrometer"},
                ],
                datasets=[
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                        ],
                    },
                ],
            )
        ]
    )
    assert len(meta.multiscales[0].axes) == 3
    assert isinstance(meta.multiscales[0].axes[0], TimeAxis)


def test_5d_tczyx():
    """Test full 5D image (TCZYX)."""
    meta = OmeMeta(
        multiscales=[
            Multiscale(
                name="example",
                type=DownscaleType.MEAN,
                axes=[
                    {"type": "time", "name": "t", "unit": "millisecond"},
                    {"type": "channel", "name": "c"},
                    {"type": "space", "name": "z", "unit": "micrometer"},
                    {"type": "space", "name": "y", "unit": "micrometer"},
                    {"type": "space", "name": "x", "unit": "micrometer"},
                ],
                datasets=[
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.5, 0.5]},
                        ],
                    },
                ],
            )
        ]
    )
    assert len(meta.multiscales[0].axes) == 5
    assert meta.multiscales[0].type == DownscaleType.MEAN


def test_multiscale_pyramid():
    """Test multiscale pyramid with multiple resolution levels."""
    meta = OmeMeta(
        multiscales=[
            Multiscale(
                axes=[
                    {"type": "space", "name": "y", "unit": "micrometer"},
                    {"type": "space", "name": "x", "unit": "micrometer"},
                ],
                datasets=[
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [0.5, 0.5]},
                        ],
                    },
                    {
                        "path": "1",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]},
                        ],
                    },
                    {
                        "path": "2",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [2.0, 2.0]},
                        ],
                    },
                ],
            )
        ]
    )
    assert len(meta.multiscales[0].datasets) == 3
    # Check resolution ordering
    assert meta.multiscales[0].datasets[0].resolution < meta.multiscales[0].datasets[1].resolution
    assert meta.multiscales[0].datasets[1].resolution < meta.multiscales[0].datasets[2].resolution


def test_with_translation():
    """Test dataset with both scale and translation."""
    meta = OmeMeta(
        multiscales=[
            Multiscale(
                axes=[
                    {"type": "space", "name": "y", "unit": "micrometer"},
                    {"type": "space", "name": "x", "unit": "micrometer"},
                ],
                datasets=[
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]},
                            {"type": "translation", "translation": [10.0, 20.0]},
                        ],
                    },
                ],
            )
        ]
    )
    dataset = meta.multiscales[0].datasets[0]
    assert isinstance(dataset.sequence, SequenceTransform)
    assert dataset.sequence.input is not None
    assert dataset.sequence.output is not None
    assert dataset.sequence.input.path == "0"
    assert dataset.sequence.output.name == "physical"
    assert len(dataset.sequence.transformations) == 2
    scale, translation = dataset.sequence.transformations[0], dataset.sequence.transformations[1]
    assert isinstance(scale, ScaleTransform)
    assert isinstance(translation, TranslationTransform)
    assert translation.translation == [10.0, 20.0]


def test_json_serialization():
    """Test that metadata can be serialized to JSON."""
    meta = OmeMeta(
        multiscales=[
            Multiscale(
                name="test",
                axes=[
                    {"type": "space", "name": "y", "unit": "micrometer"},
                    {"type": "space", "name": "x", "unit": "micrometer"},
                ],
                datasets=[
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0]},
                        ],
                    },
                ],
            )
        ]
    )
    json_str = meta.model_dump_json()
    assert '"version":"0.6"' in json_str
    assert '"coordinateSystems"' in json_str
    assert '"type":"sequence"' in json_str
    assert "test" in json_str
    assert "micrometer" in json_str
    round_trip = OmeMeta.model_validate_json(json_str)
    assert round_trip == meta


def test_multiscale_group_v06_with_origin_and_oblique_shear():
    """Test Zarr3GroupMeta.multiscale builds v0.6 sequence transforms, half-voxel pyramid shifts, and affine shear."""
    group = Zarr3GroupMeta.multiscale(
        voxel_size=UVec3D(z=0.5, y=0.4, x=0.4),
        voxel_unit=SpaceUnit.MICROMETER,
        max_level=ScaleLevel.L1,
        downscale_type=DownscaleType.GAUSSIAN,
        origin=Vec3D(z=100.0, y=200.0, x=300.0),
        shear_xz=0.70710678,
        channel_label="GFP",
        channel_color="00FF00",
    )
    ome = group.attributes.ome
    assert ome.version == "0.6"
    assert ome.omero is not None
    assert ome.omero.channels[0].label == "GFP"
    assert ome.omero.channels[0].color == "00FF00"
    ms = ome.multiscales[0]
    assert isinstance(ms.coordinateSystems[0], CoordinateSystem)
    ds0, ds1 = ms.datasets
    assert ds0.scale.scale == [1.0, 0.5, 0.4, 0.4]
    assert ds0.translation is not None
    assert ds0.translation.translation == [0.0, 100.0, 200.0, 300.0]
    assert isinstance(ds0.affine, AffineTransform)
    assert ds0.affine.affine[3][1] == 0.70710678
    assert ds1.scale.scale == [1.0, 1.0, 0.8, 0.8]
    assert ds1.translation is not None
    assert ds1.translation.translation == [0.0, 100.25, 200.2, 300.2]


def test_dataset_resolution():
    """Test Dataset resolution property."""
    ds_high_res = Dataset(
        path="0",
        coordinateTransformations=[
            {"type": "scale", "scale": [0.5, 0.5]},
        ],
    )
    ds_low_res = Dataset(
        path="1",
        coordinateTransformations=[
            {"type": "scale", "scale": [1.0, 1.0]},
        ],
    )

    # Smaller scale values = higher resolution
    assert ds_high_res.resolution == (0.5, 0.5)
    assert ds_low_res.resolution == (1.0, 1.0)
    assert ds_high_res.resolution < ds_low_res.resolution  # Compare tuples directly
