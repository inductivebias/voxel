"""Oblique stage-scan deskewing for 2D frames, 3D volumes, and OME-Zarr v0.6 datasets.

In an oblique light-sheet stage scan (such as `lm-1`'s X-scan), each frame step `z` advances along the
stage scan axis while the light sheet is tilted at `angle_deg` to the scan axis, shearing successive
frames laterally along `x` by `shear_factor` pixels per slice:

    shear_factor = (z_step * cos(angle_deg)) / pixel_size_x

Because each plane `z` only translates horizontally by `s(z) = z * shear_factor - min_shift` without
mixing adjacent `z` planes, deskewing reduces to an exact 1D 2-tap linear interpolation per row:

- `deskew_slice` deskews a single 2D `(Y, X)` frame directly into a preallocated `(Y, X_out)` destination
  (such as `OMEZarrWriter.new_frame()`), avoiding any intermediate frame allocation.
- `deskew_volume` deskews an in-memory 3D `(Z, Y, X)` array in parallel across slices.
- `deskew_dataset` streams an existing OME-Zarr v0.6 dataset through `OMEZarrWriter` to produce a
  sharded, multiscale deskewed OME-Zarr v0.6 dataset without materializing the full volume in RAM.
"""

import math
from pathlib import Path

import numpy as np
from cloudpathlib import S3Path
from numba import jit, prange
from vxlib.vector import UIVec3D, Vec3D

from ome_zarr_writer.array import ArrayWriter
from ome_zarr_writer.array.ts import TSArrayReader
from ome_zarr_writer.dataset import (
    AffineTransform,
    DownscaleType,
    OmeZarrDataset,
    ScaleLevel,
    ScaleTransform,
    TranslationTransform,
)
from ome_zarr_writer.sizing import MIN_SLOTS
from ome_zarr_writer.storage import S3Store, Storage
from ome_zarr_writer.writer import OMEZarrWriter, WriterConfig

_READ_SLAB_Z = 32


def shear_xz_from_angle(angle_deg: float) -> float:
    """Physical `dx / dz` shear ratio (`cos(angle_deg)`) for an oblique sheet at `angle_deg` ( in `(0, 90)`)."""
    if not (0.0 < angle_deg < 90.0):
        raise ValueError(f"angle_deg must be in (0, 90), got {angle_deg}")
    return float(math.cos(math.radians(angle_deg)))


def shear_factor_from_angle(z_step: float, pixel_size_x: float, angle_deg: float) -> float:
    """Voxel-space horizontal shift (`x` pixels per `z` slice) for an oblique sheet at `angle_deg`."""
    if z_step <= 0.0 or pixel_size_x <= 0.0:
        raise ValueError(f"z_step ({z_step}) and pixel_size_x ({pixel_size_x}) must be positive")
    return (z_step * shear_xz_from_angle(angle_deg)) / pixel_size_x


def deskewed_width(width: int, depth: int, shear_factor: float) -> int:
    """Expanded `x` width required to hold `depth` slices sheared by `shear_factor` pixels/slice."""
    if width <= 0 or depth <= 0:
        raise ValueError(f"width ({width}) and depth ({depth}) must be positive")
    if not math.isfinite(shear_factor):
        raise ValueError(f"shear_factor must be finite, got {shear_factor}")
    max_abs_shift = abs((depth - 1) * float(shear_factor))
    return width + math.ceil(max_abs_shift)


@jit(nopython=True, parallel=True, cache=True)
def _deskew_slice_float(plane: np.ndarray, out: np.ndarray, k: int, frac: float, cval: float) -> None:
    height, width = plane.shape
    out_w = out.shape[1]
    w1 = 1.0 - frac
    for y in prange(height):
        if frac == 0.0:
            for x in range(k):
                out[y, x] = cval
            for x in range(width):
                out[y, k + x] = plane[y, x]
            for x in range(k + width, out_w):
                out[y, x] = cval
        else:
            for x in range(k):
                out[y, x] = cval
            out[y, k] = frac * cval + w1 * plane[y, 0]
            for x in range(width - 1):
                out[y, k + 1 + x] = frac * plane[y, x] + w1 * plane[y, x + 1]
            out[y, k + width] = frac * plane[y, width - 1] + w1 * cval
            for x in range(k + width + 1, out_w):
                out[y, x] = cval


@jit(nopython=True, parallel=True, cache=True)
def _deskew_slice_int(plane: np.ndarray, out: np.ndarray, k: int, frac: float, cval: float) -> None:
    height, width = plane.shape
    out_w = out.shape[1]
    w1 = 1.0 - frac
    cval_int = np.rint(cval)
    for y in prange(height):
        if frac == 0.0:
            for x in range(k):
                out[y, x] = cval_int
            for x in range(width):
                out[y, k + x] = plane[y, x]
            for x in range(k + width, out_w):
                out[y, x] = cval_int
        else:
            for x in range(k):
                out[y, x] = cval_int
            out[y, k] = np.rint(frac * cval + w1 * plane[y, 0])
            for x in range(width - 1):
                out[y, k + 1 + x] = np.rint(frac * plane[y, x] + w1 * plane[y, x + 1])
            out[y, k + width] = np.rint(frac * plane[y, width - 1] + w1 * cval)
            for x in range(k + width + 1, out_w):
                out[y, x] = cval_int


def deskew_slice(
    plane: np.ndarray,
    shift_x: float,
    *,
    cval: float = 0.0,
    out: np.ndarray | None = None,
    out_width: int | None = None,
) -> np.ndarray:
    """Shift a 2D `(Y, X)` plane horizontally by `shift_x >= 0` pixels using exact 2-tap linear interpolation."""
    if plane.ndim != 2:
        raise ValueError(f"Expected 2D plane (Y, X), got {plane.ndim}D with shape {plane.shape}")
    height, width = plane.shape
    k = math.floor(shift_x)
    frac = float(shift_x - k)
    min_width = k + width + (1 if frac > 0.0 else 0)
    if k < 0:
        raise ValueError(f"shift_x must be non-negative, got {shift_x}")

    if out is None:
        target_w = out_width if out_width is not None else min_width
        if target_w < min_width:
            raise ValueError(f"out_width ({target_w}) is smaller than shifted plane extent ({min_width})")
        out = np.empty((height, target_w), dtype=plane.dtype)
    elif out.ndim != 2 or out.shape[0] != height or out.shape[1] < min_width:
        raise ValueError(f"out shape {out.shape} incompatible with shifted plane ({height}, >= {min_width})")

    if np.issubdtype(out.dtype, np.integer):
        _deskew_slice_int(plane, out, k, frac, float(cval))
    else:
        _deskew_slice_float(plane, out, k, frac, float(cval))
    return out


def deskew_volume(
    volume: np.ndarray,
    shear_factor: float,
    *,
    cval: float = 0.0,
) -> np.ndarray:
    """Deskew a 3D `(Z, Y, X)` volume along `X` by `shear_factor` pixels per `Z` slice."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z, Y, X), got {volume.ndim}D with shape {volume.shape}")
    depth, height, width = volume.shape
    out_w = deskewed_width(width, depth, shear_factor)
    min_shift = min(0.0, (depth - 1) * float(shear_factor))
    out = np.empty((depth, height, out_w), dtype=volume.dtype)
    for z in range(depth):
        shift_x = z * float(shear_factor) - min_shift
        deskew_slice(volume[z], shift_x, cval=cval, out=out[z])
    return out


def _physical_anchor(meta: OmeZarrDataset) -> Vec3D:
    """Compose the L0 CZYX transform; reject geometry the X-only resampler cannot preserve."""
    multiscales = meta.group.attributes.ome.multiscales
    if not multiscales:
        raise ValueError("Deskewing requires multiscales geometry")
    shape = meta.arrays[ScaleLevel.L0].shape
    if len(shape) != 4 or shape[0] != 1 or [str(axis.name) for axis in multiscales[0].axes] != ["c", "z", "y", "x"]:
        raise ValueError("Deskewing currently requires a single-channel CZYX dataset")
    if multiscales[0].datasets[0].path != "0":
        raise ValueError("Deskewing requires L0 (path 0) as the first resolution level")
    matrix = np.eye(5)
    for transform in multiscales[0].datasets[0].sequence.transformations:
        step = np.eye(5)
        if isinstance(transform, ScaleTransform):
            if transform.scale is None or len(transform.scale) != 4:
                raise ValueError("Deskewing requires an inline CZYX scale")
            step[np.arange(4), np.arange(4)] = transform.scale
        elif isinstance(transform, TranslationTransform):
            if transform.translation is None or len(transform.translation) != 4:
                raise ValueError("Deskewing requires an inline CZYX translation")
            step[:4, 4] = transform.translation
        elif isinstance(transform, AffineTransform):
            affine = np.asarray(transform.affine)
            if affine.shape != (4, 5):
                raise ValueError("Deskewing requires a CZYX affine transform")
            step[:4, :] = affine
        matrix = step @ matrix
    voxel = meta.voxel_size
    supported = np.diag([1.0, voxel.z, voxel.y, voxel.x])
    supported[3, 1] = matrix[3, 1]
    if not np.allclose(matrix[:4, :4], supported, rtol=1e-10, atol=1e-12):
        raise ValueError("X-only deskewing does not support shear_yz, rotations, or other affine components")
    return Vec3D(z=float(matrix[1, 4]), y=float(matrix[2, 4]), x=float(matrix[3, 4]))


def deskew_dataset(
    source: Path | S3Path,
    storage: Storage,
    *,
    shear_factor: float | None = None,
    angle_deg: float | None = None,
    cval: float = 0.0,
    slots: int = MIN_SLOTS,
    backend: ArrayWriter.Backend = ArrayWriter.Backend.TS,
    source_store: S3Store | None = None,
) -> Path | S3Path:
    """Stream an OME-Zarr v0.6 dataset from `source`, deskew L0 slice-by-slice, and write a new
    multiscale OME-Zarr v0.6 pyramid to `storage`.

    The horizontal `shear_factor` (in L0 `x` pixels per `z` slice) is resolved in priority order from:
    1. `shear_factor` if explicitly passed,
    2. `angle_deg` (`z_step * cos(angle_deg) / pixel_size_x`), or
    3. the source dataset's `AffineTransform` (`shear_xz * z_step / pixel_size_x`) in `zarr.json`.
    Physical placement is retained, including the offset introduced by negative-shear padding.
    Only single-channel CZYX data with XZ shear is supported; other affine components are rejected.
    """
    meta = OmeZarrDataset.read_metadata(source)
    v_shape = meta.volume_shape
    v_size = meta.voxel_size
    anchor = _physical_anchor(meta)

    if shear_factor is None:
        if angle_deg is not None:
            shear_factor = shear_factor_from_angle(v_size.z, v_size.x, angle_deg)
        elif meta.shear_xz != 0.0:
            shear_factor = (meta.shear_xz * v_size.z) / v_size.x
        else:
            raise ValueError("No shear_factor or angle_deg provided and source dataset has no non-zero affine shear_xz")

    out_w = deskewed_width(v_shape.x, v_shape.z, shear_factor)
    min_shift = min(0.0, (v_shape.z - 1) * float(shear_factor))

    multiscales = meta.group.attributes.ome.multiscales
    downscale_type = multiscales[0].type if multiscales and multiscales[0].type is not None else DownscaleType.GAUSSIAN
    omero = meta.group.attributes.ome.omero
    channel_label = omero.channels[0].label if omero and omero.channels else "channel 0"
    channel_color = omero.channels[0].color if omero and omero.channels else "FFFFFF"

    config = WriterConfig(
        volume_shape=UIVec3D(z=v_shape.z, y=v_shape.y, x=out_w),
        voxel_size=v_size,
        voxel_unit=meta.voxel_unit,
        dtype=meta.dtype,
        max_level=meta.max_level,
        downscale_type=downscale_type,
        # Pixel padding for negative shear moves the output grid origin left.
        origin=Vec3D(z=anchor.z, y=anchor.y, x=anchor.x + min_shift * v_size.x),
        shear_xz=0.0,
        channel_label=channel_label,
        channel_color=channel_color,
    )

    reader = TSArrayReader(source / "0", store=source_store)
    writer = OMEZarrWriter(backend=backend, slots=slots)
    try:
        writer.begin_stack(config, storage)
        for z0 in range(0, v_shape.z, _READ_SLAB_Z):
            z1 = min(z0 + _READ_SLAB_Z, v_shape.z)
            slab = reader.read_3d(z0=z0, z1=z1)
            for dz in range(z1 - z0):
                z = z0 + dz
                shift_x = z * float(shear_factor) - min_shift
                with writer.new_frame() as frame:
                    deskew_slice(slab[dz], shift_x, cval=cval, out=frame)
        target = writer.target
        writer.end_stack()
    finally:
        writer.close()
    if target is None:
        raise RuntimeError("OMEZarrWriter did not expose a target path")
    return target
