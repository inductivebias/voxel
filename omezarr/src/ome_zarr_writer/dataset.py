"""Dataset description: Zarr v3 metadata + OME-NGFF v0.6 spec + `OmeZarrDataset`.

Contains:
- Zarr v3 array metadata: codecs, chunk grid, `Zarr3ArrayMeta` — spec-faithful
  Pydantic models for `{root}/{level}/zarr.json`.
- OME-NGFF v0.6 spec types: coordinate systems, axes, transforms, multiscales, `Omero`, `OmeMeta`.
- `Zarr3GroupMeta`: Zarr v3 group metadata carrying OME-NGFF attributes
  (mirrors `{root}/zarr.json`).
- `OmeZarrDataset`: the top-level model mirroring the full on-disk shape of
  an OME-Zarr v0.6 dataset. Carries derived properties for the layout fields
  the writer cares about (`volume_shape`, `shard_shape`, `dtype`, `max_level`)
  and a single iterator for shard enumeration.
"""

import math
from collections.abc import Iterator
from enum import IntEnum, StrEnum
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
from cloudpathlib import S3Path
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    PlainSerializer,
    model_validator,
)
from vxlib.dtype import Dtype
from vxlib.schema import SparseModel
from vxlib.vector import UIVec3D, UVec3D, Vec3D


class ScaleLevel(IntEnum):
    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5
    L6 = 6
    L7 = 7

    @property
    def factor(self) -> int:
        return 1 << self.value

    @property
    def chunk_shape(self) -> UIVec3D:
        size = self.factor
        return UIVec3D(z=size, y=size, x=size)

    def scale(self, shape: UIVec3D) -> UIVec3D:
        f = self.factor
        return UIVec3D(z=shape.z // f, y=shape.y // f, x=shape.x // f)

    @cached_property
    def levels(self) -> tuple["ScaleLevel", ...]:
        return tuple(ScaleLevel(i) for i in range(self.value + 1))

    def get_path(self, root_path: str) -> Path:
        return Path(root_path).expanduser().resolve() / f"{self.value}"

    def __repr__(self) -> str:
        return f"{self.name}(factor={self.factor})"


class Compression(StrEnum):
    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"
    LZ4 = "lz4"
    BLOSC_LZ4 = "blosc.lz4"
    BLOSC_ZSTD = "blosc.zstd"


class _ChunkConfig(SparseModel):
    chunk_shape: list[int]


class ChunkGrid(SparseModel):
    name: Literal["regular"] = "regular"
    configuration: _ChunkConfig


class _ChunkKeyConfig(SparseModel):
    separator: Literal["/", "."] = "/"


class ChunkKeyEncoding(SparseModel):
    name: Literal["default", "v2"] = "default"
    configuration: _ChunkKeyConfig = Field(default_factory=_ChunkKeyConfig)


class _BytesConfig(SparseModel):
    endian: Literal["little", "big"] = "little"


class BytesCodec(SparseModel):
    name: Literal["bytes"] = "bytes"
    configuration: _BytesConfig = Field(default_factory=_BytesConfig)


class _BloscConfig(SparseModel):
    cname: Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
    clevel: int = 5
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "shuffle"
    typesize: int | None = None
    blocksize: int = 0


class BloscCodec(SparseModel):
    name: Literal["blosc"] = "blosc"
    configuration: _BloscConfig


class _GzipConfig(SparseModel):
    level: int = 5


class GzipCodec(SparseModel):
    name: Literal["gzip"] = "gzip"
    configuration: _GzipConfig = Field(default_factory=_GzipConfig)


class _ZstdConfig(SparseModel):
    level: int = 0
    checksum: bool = False


class ZstdCodec(SparseModel):
    name: Literal["zstd"] = "zstd"
    configuration: _ZstdConfig = Field(default_factory=_ZstdConfig)


class CRC32CCodec(SparseModel):
    name: Literal["crc32c"] = "crc32c"


InnerCodec = Annotated[
    BytesCodec | BloscCodec | GzipCodec | ZstdCodec | CRC32CCodec,
    Field(discriminator="name"),
]


class _ShardingConfig(SparseModel):
    chunk_shape: list[int]
    codecs: list[InnerCodec]
    index_codecs: list[InnerCodec] = Field(default_factory=lambda: [BytesCodec(), CRC32CCodec()])
    index_location: Literal["start", "end"] = "end"


class ShardingIndexedCodec(SparseModel):
    name: Literal["sharding_indexed"] = "sharding_indexed"
    configuration: _ShardingConfig


Codec = Annotated[
    BytesCodec | BloscCodec | GzipCodec | ZstdCodec | CRC32CCodec | ShardingIndexedCodec,
    Field(discriminator="name"),
]


class Zarr3ArrayMeta(SparseModel):
    """Mirrors `{root}/{level}/zarr.json` — Zarr v3 array metadata."""

    zarr_format: Literal[3] = 3
    node_type: Literal["array"] = "array"
    shape: list[int]
    data_type: Dtype
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding = Field(default_factory=ChunkKeyEncoding)
    fill_value: int | float = 0
    codecs: list[Codec]
    dimension_names: list[str | None] | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def sharded(
        cls,
        *,
        shape: list[int],
        shard_shape: list[int],
        chunk_shape: list[int],
        dtype: Dtype,
        compression: Compression,
        dimension_names: list[str | None] | None = None,
    ) -> "Zarr3ArrayMeta":
        """A sharded array meta — outer (shard) chunk grid wrapping an inner
        chunk shape inside a `sharding_indexed` codec.

        `shape`, `shard_shape`, `chunk_shape` all include the leading channel
        dimension. `compression` selects the inner codec chain (applied to each
        inner chunk inside a shard).
        """
        return cls(
            shape=shape,
            data_type=dtype,
            chunk_grid=ChunkGrid(configuration=_ChunkConfig(chunk_shape=shard_shape)),
            codecs=[
                ShardingIndexedCodec(
                    configuration=_ShardingConfig(
                        chunk_shape=chunk_shape,
                        codecs=inner_codecs_for(compression, dtype.itemsize),
                    )
                )
            ],
            dimension_names=dimension_names,
        )


def inner_codecs_for(compression: Compression, typesize: int) -> list[InnerCodec]:
    """Inner codec chain (applied to each inner chunk inside a shard) for one of
    our supported `Compression` presets. Encodes our level/shuffle policy."""
    codecs: list[InnerCodec] = [BytesCodec()]
    if compression == Compression.NONE:
        return codecs
    if compression == Compression.GZIP:
        codecs.append(GzipCodec(configuration=_GzipConfig(level=3)))
    elif compression == Compression.ZSTD:
        codecs.append(ZstdCodec(configuration=_ZstdConfig(level=3)))
    elif compression in (Compression.LZ4, Compression.BLOSC_LZ4):
        codecs.append(
            BloscCodec(configuration=_BloscConfig(cname="lz4", clevel=3, shuffle="bitshuffle", typesize=typesize))
        )
    elif compression == Compression.BLOSC_ZSTD:
        codecs.append(
            BloscCodec(configuration=_BloscConfig(cname="zstd", clevel=3, shuffle="bitshuffle", typesize=typesize))
        )
    else:
        raise ValueError(f"Unsupported compression: {compression}")
    return codecs


class Shard(NamedTuple):
    """Address of one shard file within an OME-Zarr v3 sharded dataset.

    Lightweight, hashable, and self-resolving — knows how to produce its own
    path under any storage root via `at(root)`.
    """

    level: ScaleLevel
    c: int
    z: int
    y: int
    x: int

    @property
    def relpath(self) -> PurePosixPath:
        return PurePosixPath(
            str(self.level.value),
            "c",
            str(self.c),
            str(self.z),
            str(self.y),
            str(self.x),
        )

    def at[R: (Path, S3Path)](self, root: R) -> R:
        return root / str(self.relpath)


class AxisName(StrEnum):
    X = "x"
    Y = "y"
    Z = "z"
    C = "c"
    T = "t"


class AxisType(StrEnum):
    SPACE = "space"
    TIME = "time"
    CHANNEL = "channel"


class TimeUnit(StrEnum):
    SECOND = "second"
    MILLISECOND = "millisecond"
    MICROSECOND = "microsecond"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class SpaceUnit(StrEnum):
    METER = "meter"
    MICROMETER = "micrometer"
    NANOMETER = "nanometer"
    MILLIMETER = "millimeter"
    INCH = "inch"
    FOOT = "foot"


class TransformType(StrEnum):
    IDENTITY = "identity"
    TRANSLATION = "translation"
    SCALE = "scale"
    AFFINE = "affine"
    SEQUENCE = "sequence"


class DownscaleType(StrEnum):
    GAUSSIAN = "gaussian"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"


class SpaceAxis(SparseModel):
    name: Literal[AxisName.X, AxisName.Y, AxisName.Z]
    type: Literal[AxisType.SPACE] = AxisType.SPACE
    unit: SpaceUnit

    @classmethod
    def axes(cls, unit: SpaceUnit) -> list["SpaceAxis"]:
        return [
            cls(name=AxisName.Z, unit=unit),
            cls(name=AxisName.Y, unit=unit),
            cls(name=AxisName.X, unit=unit),
        ]


class TimeAxis(SparseModel):
    name: Literal[AxisName.T] = AxisName.T
    type: Literal[AxisType.TIME] = AxisType.TIME
    unit: TimeUnit


class ChannelAxis(SparseModel):
    name: Literal[AxisName.C] = AxisName.C
    type: Literal[AxisType.CHANNEL] = AxisType.CHANNEL


Axis = Annotated[SpaceAxis | TimeAxis | ChannelAxis, Field(discriminator="type")]


def _validate_multiscale_axes(value: list) -> list[Axis]:
    """OME-NGFF axis ordering and count constraints.

    2-5 axes total, 2-3 space axes, ≤1 time axis, ≤1 channel axis. Order is
    time (if present), channel (if present), then space.
    """
    if not isinstance(value, list):
        raise ValueError(f"axes must be a list, got {type(value)}")
    if not (2 <= len(value) <= 5):
        raise ValueError(f"Must have 2-5 axes, found {len(value)}")

    space_axes = [ax for ax in value if isinstance(ax, SpaceAxis)]
    time_axes = [ax for ax in value if isinstance(ax, TimeAxis)]
    channel_axes = [ax for ax in value if isinstance(ax, ChannelAxis)]

    if not (2 <= len(space_axes) <= 3):
        raise ValueError(f"Must have 2-3 space axes, found {len(space_axes)}")
    if len(time_axes) > 1:
        raise ValueError(f"May have at most 1 time axis, found {len(time_axes)}")
    if len(channel_axes) > 1:
        raise ValueError(f"May have at most 1 channel axis, found {len(channel_axes)}")

    time_idx = [i for i, ax in enumerate(value) if isinstance(ax, TimeAxis)]
    channel_idx = [i for i, ax in enumerate(value) if isinstance(ax, ChannelAxis)]
    space_idx = [i for i, ax in enumerate(value) if isinstance(ax, SpaceAxis)]

    if time_idx and channel_idx and time_idx[0] > channel_idx[0]:
        raise ValueError("Time axis must come before channel axis")
    if time_idx and space_idx and time_idx[0] > space_idx[0]:
        raise ValueError("Time axis must come before space axes")
    if channel_idx and space_idx and channel_idx[0] > space_idx[0]:
        raise ValueError("Channel axis must come before space axes")

    return value


MultiscaleAxes = Annotated[
    list[Axis],
    AfterValidator(_validate_multiscale_axes),
    Field(min_length=2, max_length=5),
]


class CoordinateSystem(SparseModel):
    """OME-NGFF v0.6 named coordinate system."""

    name: str = "physical"
    axes: MultiscaleAxes


class TransformEndpoint(SparseModel):
    """OME-NGFF v0.6 transformation input or output reference (`{"path": ...}` or `{"name": ...}`)."""

    path: str | None = None
    name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value


class IdentityTransform(SparseModel):
    type: Literal[TransformType.IDENTITY] = TransformType.IDENTITY


class TranslationTransform(SparseModel):
    type: Literal[TransformType.TRANSLATION] = TransformType.TRANSLATION
    translation: list[float] | None = None
    path: str | None = None


class ScaleTransform(SparseModel):
    type: Literal[TransformType.SCALE] = TransformType.SCALE
    scale: list[float] | None = None
    path: str | None = None


class AffineTransform(SparseModel):
    """OME-NGFF v0.6 affine coordinate transformation (`N x (N + 1)` homogeneous matrix)."""

    type: Literal[TransformType.AFFINE] = TransformType.AFFINE
    affine: list[list[float]]

    @model_validator(mode="after")
    def _validate_matrix(self) -> "AffineTransform":
        n = len(self.affine)
        if not (2 <= n <= 5):
            raise ValueError(f"Affine matrix must have 2-5 rows, got {n}")
        for row in self.affine:
            if len(row) != n + 1 or not all(math.isfinite(v) for v in row):
                raise ValueError(f"Each affine row must have {n + 1} finite values")
        return self

    @classmethod
    def from_shear(
        cls,
        *,
        shear_xz: float = 0.0,
        shear_yz: float = 0.0,
        ndim: int = 4,
    ) -> "AffineTransform":
        """Build an `N x (N + 1)` affine matrix for oblique stage-scan shear (`z` shearing `y`/`x`).

        For `CZYX` (`ndim=4`) or `ZYX` (`ndim=3`), the spatial axes `(z, y, x)` occupy the trailing
        three indices `(ndim - 3, ndim - 2, ndim - 1)`:
            y_out = y_in + shear_yz * z_in
            x_out = x_in + shear_xz * z_in
        """
        if not (3 <= ndim <= 5):
            raise ValueError(f"ndim must be 3-5, got {ndim}")
        matrix = [[1.0 if i == j else 0.0 for j in range(ndim + 1)] for i in range(ndim)]
        z_idx = ndim - 3
        matrix[ndim - 2][z_idx] = float(shear_yz)
        matrix[ndim - 1][z_idx] = float(shear_xz)
        return cls(affine=matrix)


def _normalize_sequence_steps(value: list | tuple) -> tuple:
    """Accept `[scale]`, `[scale, translation]`, `[scale, affine]`, or `[scale, translation, affine]`."""
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"Expected list or tuple, got {type(value)}")
    if not (1 <= len(value) <= 3):
        raise ValueError(f"Must have 1-3 transforms, got {len(value)}")
    return tuple(value)


SequenceSteps = Annotated[
    tuple[ScaleTransform]
    | tuple[ScaleTransform, TranslationTransform]
    | tuple[ScaleTransform, AffineTransform]
    | tuple[ScaleTransform, TranslationTransform, AffineTransform],
    BeforeValidator(_normalize_sequence_steps),
    PlainSerializer(list, return_type=list),
]


class SequenceTransform(SparseModel):
    """OME-NGFF v0.6 sequence transformation mapping a dataset path to a named coordinate system."""

    type: Literal[TransformType.SEQUENCE] = TransformType.SEQUENCE
    input: TransformEndpoint | None = None
    output: TransformEndpoint | None = None
    name: str | None = "transformSequence"
    transformations: SequenceSteps

    @property
    def scale(self) -> ScaleTransform:
        return self.transformations[0]

    @property
    def translation(self) -> TranslationTransform | None:
        return next((t for t in self.transformations if isinstance(t, TranslationTransform)), None)

    @property
    def affine(self) -> AffineTransform | None:
        return next((t for t in self.transformations if isinstance(t, AffineTransform)), None)


def _normalize_coord_transforms(value: list | tuple) -> tuple[SequenceTransform]:
    """Normalize either `[SequenceTransform]` or flat `[ScaleTransform, ...]` into `(SequenceTransform,)`."""
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"Expected list or tuple, got {type(value)}")
    if not value:
        raise ValueError("Must have at least 1 transform")
    first = value[0]
    is_seq = isinstance(first, SequenceTransform) or (
        isinstance(first, dict) and first.get("type") == TransformType.SEQUENCE
    )
    if is_seq:
        if len(value) != 1:
            raise ValueError(f"Must have 1 sequence transform, got {len(value)}")
        return (first,)  # type: ignore[return-value]
    return (SequenceTransform(transformations=value),)  # type: ignore[arg-type]


CoordinateTransformations = Annotated[
    tuple[SequenceTransform],
    BeforeValidator(_normalize_coord_transforms),
    PlainSerializer(list, return_type=list),
]


class Dataset(SparseModel):
    """One resolution level inside a Multiscale.

    Per OME-NGFF v0.6: a `sequence` transformation mapping `{"path": path}` to
    `{"name": "physical"}` containing a `scale` transform and optional
    `translation` and `affine` transforms. Datasets compare by resolution —
    lower scale values are higher resolution.
    """

    path: str
    coordinateTransformations: CoordinateTransformations  # noqa: N815 - OME-NGFF schema field

    @model_validator(mode="after")
    def _bind_default_endpoints(self) -> "Dataset":
        seq = self.coordinateTransformations[0]
        if seq.input is None or seq.output is None:
            bound = SequenceTransform(
                input=seq.input or TransformEndpoint(path=self.path),
                output=seq.output or TransformEndpoint(name="physical"),
                name=seq.name,
                transformations=seq.transformations,
            )
            object.__setattr__(self, "coordinateTransformations", (bound,))
        return self

    @property
    def sequence(self) -> SequenceTransform:
        return self.coordinateTransformations[0]

    @property
    def scale(self) -> ScaleTransform:
        return self.sequence.scale

    @property
    def translation(self) -> TranslationTransform | None:
        return self.sequence.translation

    @property
    def affine(self) -> AffineTransform | None:
        return self.sequence.affine

    @property
    def resolution(self) -> tuple[float, ...]:
        scale = self.scale.scale
        if not scale:
            raise ValueError(f"Dataset {self.path} has no scale values")
        return tuple(scale)


class Multiscale(SparseModel):
    """OME-NGFF v0.6 multiscale: ordered datasets sharing a coordinate system.

    Datasets must be ordered highest-to-lowest resolution.
    """

    coordinateSystems: list[CoordinateSystem] = Field(..., min_length=1)  # noqa: N815 - OME-NGFF schema field
    datasets: list[Dataset] = Field(..., min_length=1)
    name: str | None = None
    type: DownscaleType | None = None
    metadata: dict | None = None
    coordinateTransformations: CoordinateTransformations | None = None  # noqa: N815 - OME-NGFF schema field

    def __init__(
        self,
        *,
        datasets: list[Any],
        coordinateSystems: list[Any] | None = None,  # noqa: N803
        axes: list[Any] | None = None,
        name: str | None = None,
        type: DownscaleType | None = None,  # noqa: A002 - OME-NGFF schema field
        metadata: dict | None = None,
        coordinateTransformations: Any = None,  # noqa: N803
        **extra: Any,
    ) -> None:
        kwargs: dict[str, Any] = {
            "datasets": datasets,
            "name": name,
            "type": type,
            "metadata": metadata,
            "coordinateTransformations": coordinateTransformations,
            **extra,
        }
        if coordinateSystems is not None:
            kwargs["coordinateSystems"] = coordinateSystems
        if axes is not None:
            kwargs["axes"] = axes
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _coerce_axes_to_coordinate_systems(cls, data: Any) -> Any:
        if isinstance(data, dict) and "axes" in data and "coordinateSystems" not in data:
            copied = dict(data)
            axes = copied.pop("axes")
            copied["coordinateSystems"] = [{"name": "physical", "axes": axes}]
            return copied
        return data

    @property
    def axes(self) -> list[Axis]:
        return self.coordinateSystems[0].axes

    @model_validator(mode="after")
    def _validate_dataset_ordering(self) -> "Multiscale":
        for i in range(len(self.datasets) - 1):
            curr = self.datasets[i].resolution
            nxt = self.datasets[i + 1].resolution
            if curr >= nxt:
                raise ValueError(
                    f"Datasets must be ordered highest-to-lowest resolution. "
                    f"Index {i} (path={self.datasets[i].path}, scale={curr}) is not "
                    f"strictly higher resolution than index {i + 1} "
                    f"(path={self.datasets[i + 1].path}, scale={nxt})"
                )
        return self


class OmeroWindow(SparseModel):
    min: float = 0.0
    max: float = 65535.0
    start: float = 0.0
    end: float = 65535.0


class OmeroChannel(SparseModel):
    label: str = "channel 0"
    color: str = "FFFFFF"
    active: bool = True
    window: OmeroWindow = Field(default_factory=OmeroWindow)


class Omero(SparseModel):
    channels: list[OmeroChannel] = Field(..., min_length=1)

    @classmethod
    def default(
        cls,
        dtype: Dtype = Dtype.UINT16,
        *,
        label: str = "channel 0",
        color: str = "FFFFFF",
    ) -> "Omero":
        top = float(np.iinfo(dtype.dtype).max) if np.issubdtype(dtype.dtype, np.integer) else 1.0
        return cls(
            channels=[
                OmeroChannel(
                    label=label,
                    color=color,
                    active=True,
                    window=OmeroWindow(min=0.0, max=top, start=0.0, end=top),
                )
            ]
        )


class OmeMeta(SparseModel):
    """Top-level OME-Zarr v0.6 group attributes."""

    version: Literal["0.6"] = "0.6"
    multiscales: list[Multiscale] | None = None
    omero: Omero | None = None


class ZarrGroupAttributes(SparseModel):
    ome: OmeMeta


class Zarr3GroupMeta(SparseModel):
    """Zarr v3 group metadata — mirrors `{root}/zarr.json`."""

    zarr_format: Literal[3] = 3
    node_type: Literal["group"] = "group"
    attributes: ZarrGroupAttributes

    @classmethod
    def multiscale(
        cls,
        *,
        voxel_size: UVec3D,
        voxel_unit: SpaceUnit,
        max_level: ScaleLevel,
        downscale_type: DownscaleType,
        dtype: Dtype = Dtype.UINT16,
        origin: Vec3D | None = None,
        shear_xz: float = 0.0,
        shear_yz: float = 0.0,
        channel_label: str = "channel 0",
        channel_color: str = "FFFFFF",
    ) -> "Zarr3GroupMeta":
        """A Zarr v3 group with OME-NGFF v0.6 multiscale attributes for a CZYX
        dataset. Builds the `physical` coordinate system, per-level sequence
        transforms (scale, half-voxel center translation, and optional oblique
        shear affine), and default `omero` channel display metadata."""
        oz, oy, ox = (origin.z, origin.y, origin.x) if origin is not None else (0.0, 0.0, 0.0)
        has_origin = oz != 0.0 or oy != 0.0 or ox != 0.0
        has_shear = shear_xz != 0.0 or shear_yz != 0.0
        affine = AffineTransform.from_shear(shear_xz=shear_xz, shear_yz=shear_yz, ndim=4) if has_shear else None

        datasets: list[Dataset] = []
        for level in max_level.levels:
            f = level.factor
            steps: list[ScaleTransform | TranslationTransform | AffineTransform] = [
                ScaleTransform(
                    scale=[
                        1.0,
                        voxel_size.z * f,
                        voxel_size.y * f,
                        voxel_size.x * f,
                    ]
                )
            ]
            if f > 1 or has_origin:
                steps.append(
                    TranslationTransform(
                        translation=[
                            0.0,
                            oz + voxel_size.z * (f - 1) / 2.0,
                            oy + voxel_size.y * (f - 1) / 2.0,
                            ox + voxel_size.x * (f - 1) / 2.0,
                        ]
                    )
                )
            if affine is not None:
                steps.append(affine)
            datasets.append(
                Dataset(
                    path=str(level.value),
                    coordinateTransformations=(
                        SequenceTransform(
                            input=TransformEndpoint(path=str(level.value)),
                            output=TransformEndpoint(name="physical"),
                            transformations=tuple(steps),  # type: ignore[arg-type]
                        ),
                    ),
                )
            )

        return cls(
            attributes=ZarrGroupAttributes(
                ome=OmeMeta(
                    multiscales=[
                        Multiscale(
                            coordinateSystems=[
                                CoordinateSystem(
                                    name="physical",
                                    axes=[ChannelAxis(), *SpaceAxis.axes(unit=voxel_unit)],
                                )
                            ],
                            datasets=datasets,
                            type=downscale_type,
                        )
                    ],
                    omero=Omero.default(dtype=dtype, label=channel_label, color=channel_color),
                )
            )
        )


class OmeZarrDataset(SparseModel):
    """Full on-disk shape of an OME-Zarr v0.6 / Zarr v3 dataset.

    Bundles the OME-NGFF group metadata (`group`) with the per-array metadata
    (`arrays`). Reading and writing are symmetric: an `OmeZarrDataset` parses
    from disk and serializes back to the same JSON shape.

    Derived properties (`volume_shape`, `shard_shape`, `dtype`, …) read from the
    nested spec structure on demand. They are plain `@property` (not pydantic
    `@computed_field`) so they're excluded from this model's own serialization.
    """

    group: Zarr3GroupMeta
    arrays: dict[ScaleLevel, Zarr3ArrayMeta]

    @property
    def volume_shape(self) -> UIVec3D:
        z, y, x = self.arrays[ScaleLevel.L0].shape[1:]
        return UIVec3D(z=z, y=y, x=x)

    @property
    def shard_shape(self) -> UIVec3D:
        z, y, x = self.arrays[ScaleLevel.L0].chunk_grid.configuration.chunk_shape[1:]
        return UIVec3D(z=z, y=y, x=x)

    @property
    def max_level(self) -> ScaleLevel:
        return ScaleLevel(max(self.arrays.keys()))

    @property
    def dtype(self) -> Dtype:
        return self.arrays[ScaleLevel.L0].data_type

    @property
    def voxel_size(self) -> UVec3D:
        multiscales = self.group.attributes.ome.multiscales
        if not multiscales:
            raise ValueError("Dataset group has no multiscales metadata")
        scale = multiscales[0].datasets[0].resolution
        z, y, x = scale[-3:]
        return UVec3D(z=z, y=y, x=x)

    @property
    def voxel_unit(self) -> SpaceUnit:
        multiscales = self.group.attributes.ome.multiscales
        if not multiscales:
            raise ValueError("Dataset group has no multiscales metadata")
        for axis in multiscales[0].axes:
            if isinstance(axis, SpaceAxis):
                return axis.unit
        return SpaceUnit.MICROMETER

    @property
    def shear_xz(self) -> float:
        multiscales = self.group.attributes.ome.multiscales
        if not multiscales:
            return 0.0
        affine = multiscales[0].datasets[0].affine
        if affine is None:
            return 0.0
        ndim = len(affine.affine)
        return float(affine.affine[ndim - 1][ndim - 3])

    def shards(self, channels: list[int], z_range: range | None = None) -> Iterator[Shard]:
        """Yield shards at every pyramid level for the given channels.

        `z_range` selects a window of z-shard indices in L0 coordinates (the
        same range applies at every level since shard sizes halve in lockstep
        with volume sizes). If `None`, iterates every shard in the dataset.

        Higher levels may have fewer total z-shards (tail) — out-of-range z
        values are skipped per level.

        Iteration order: level → channel → z → y → x.
        """
        if z_range is None:
            z_range = range(math.ceil(self.volume_shape.z / self.shard_shape.z))
        for level in self.max_level.levels:
            scaled_shard = level.scale(self.shard_shape)
            scaled_volume = level.scale(self.volume_shape)
            n_y = math.ceil(scaled_volume.y / scaled_shard.y)
            n_x = math.ceil(scaled_volume.x / scaled_shard.x)
            scaled_volume_z = level.scale(self.volume_shape).z
            scaled_shard_z = level.scale(self.shard_shape).z
            max_z = math.ceil(scaled_volume_z / scaled_shard_z)
            for c in channels:
                for z in z_range:
                    if z >= max_z:
                        continue
                    for y in range(n_y):
                        for x in range(n_x):
                            yield Shard(level=level, c=c, z=z, y=y, x=x)

    def write_metadata(self, target: Path | S3Path) -> None:
        """Write Zarr v3 group + per-level array `zarr.json` files under `target` — a local path
        or an S3Path (bind it to a client beforehand for a non-default endpoint/profile).

        One `zarr.json` at the group root plus one per pyramid level; existing files/objects are
        overwritten. Uniform across local and S3 paths.
        """
        for relpath, meta in (
            ("zarr.json", self.group),
            *((f"{level.value}/zarr.json", array_meta) for level, array_meta in self.arrays.items()),
        ):
            dest = target / relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(meta.model_dump_json(), encoding="utf-8")

    @classmethod
    def read_metadata(cls, target: Path | S3Path) -> "OmeZarrDataset":
        """Read Zarr v3 group + per-level array `zarr.json` files under `target`."""
        group = Zarr3GroupMeta.model_validate_json((target / "zarr.json").read_text(encoding="utf-8"))
        multiscales = group.attributes.ome.multiscales or []
        arrays: dict[ScaleLevel, Zarr3ArrayMeta] = {}
        if multiscales:
            for ds in multiscales[0].datasets:
                level = ScaleLevel(int(ds.path))
                arrays[level] = Zarr3ArrayMeta.model_validate_json(
                    (target / ds.path / "zarr.json").read_text(encoding="utf-8")
                )
        return cls(group=group, arrays=arrays)
