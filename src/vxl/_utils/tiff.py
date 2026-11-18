"""Pure-Python Baseline / ImageJ-compatible 2D TIFF writer for uint8, uint16, and float32 frames."""

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

# TIFF data types
_TIFF_BYTE = 1
_TIFF_ASCII = 2
_TIFF_SHORT = 3
_TIFF_LONG = 4
_TIFF_RATIONAL = 5

# TIFF tag IDs
_TAG_IMAGE_WIDTH = 256
_TAG_IMAGE_LENGTH = 257
_TAG_BITS_PER_SAMPLE = 258
_TAG_COMPRESSION = 259
_TAG_PHOTOMETRIC = 262
_TAG_IMAGE_DESCRIPTION = 270
_TAG_STRIP_OFFSETS = 273
_TAG_SAMPLES_PER_PIXEL = 277
_TAG_ROWS_PER_STRIP = 278
_TAG_STRIP_BYTE_COUNTS = 279
_TAG_X_RESOLUTION = 282
_TAG_Y_RESOLUTION = 283
_TAG_RESOLUTION_UNIT = 296
_TAG_SOFTWARE = 305
_TAG_SAMPLE_FORMAT = 339


def _rational_from_um(pixel_size_um: float) -> tuple[int, int]:
    """Convert pixel size in µm to TIFF resolution rational (pixels per unit = 1 / pixel_size_um)."""
    if pixel_size_um <= 0 or not np.isfinite(pixel_size_um):
        return (1, 1)
    denom = 1_000_000
    numer = max(1, round(denom / float(pixel_size_um)))
    return (numer, denom)


def write_tiff_2d(
    path: Path | str,
    image: np.ndarray,
    *,
    pixel_size_um: tuple[float, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a 2D grayscale numpy array (``uint8``, ``uint16``, or ``float32``) to an uncompressed
    ImageJ-compatible Baseline TIFF file with physical pixel size and metadata.

    Args:
        path: Output file path (parent directories are created automatically).
        image: 2D array ``(H, W)``. Supported dtypes: ``uint8``, ``uint16``, ``float32`` (other numeric
            dtypes are cast to ``float32`` or ``uint16``).
        pixel_size_um: Optional ``(py_um, px_um)`` physical pixel pitch in micrometers.
        metadata: Optional dictionary of acquisition metadata embedded in ``ImageDescription``.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"write_tiff_2d expects a 2D array, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        arr = np.ascontiguousarray(arr, dtype="<u1")
        bits_per_sample = 8
        sample_format = 1  # unsigned integer
    elif arr.dtype == np.uint16:
        arr = np.ascontiguousarray(arr, dtype="<u2")
        bits_per_sample = 16
        sample_format = 1  # unsigned integer
    else:
        arr = np.ascontiguousarray(arr, dtype="<f4")
        bits_per_sample = 32
        sample_format = 3  # IEEE floating point

    height, width = int(arr.shape[0]), int(arr.shape[1])
    raw_bytes = arr.tobytes()
    byte_count = len(raw_bytes)

    desc_lines = ["ImageJ=1.54f", "images=1", "slices=1", "unit=um"]
    if metadata:
        desc_lines.append(f"voxel_metadata={json.dumps(metadata, default=str)}")
    desc_bytes = ("\n".join(desc_lines) + "\x00").encode("utf-8")
    software_bytes = b"Voxel Microscope\x00"

    py_um, px_um = pixel_size_um if pixel_size_um is not None else (1.0, 1.0)
    x_res = _rational_from_um(px_um)
    y_res = _rational_from_um(py_um)

    # Layout:
    # [0..8): 8-byte little-endian TIFF header ("II", 42, offset_to_ifd=8)
    # [8..8 + 2 + 12*N + 4): IFD entries
    # Followed by extra values (XResolution, YResolution, Software, ImageDescription)
    # Followed by pixel data (2-byte aligned)
    num_entries = 14
    ifd_offset = 8
    ifd_size = 2 + num_entries * 12 + 4
    extra_offset = ifd_offset + ifd_size

    x_res_offset = extra_offset
    y_res_offset = x_res_offset + 8
    software_offset = y_res_offset + 8
    desc_offset = software_offset + len(software_bytes)
    data_offset = desc_offset + len(desc_bytes)
    if data_offset % 2 == 1:
        data_offset += 1

    entries: list[tuple[int, int, int, int]] = [
        (_TAG_IMAGE_WIDTH, _TIFF_LONG, 1, width),
        (_TAG_IMAGE_LENGTH, _TIFF_LONG, 1, height),
        (_TAG_BITS_PER_SAMPLE, _TIFF_SHORT, 1, bits_per_sample),
        (_TAG_COMPRESSION, _TIFF_SHORT, 1, 1),  # 1 = Uncompressed
        (_TAG_PHOTOMETRIC, _TIFF_SHORT, 1, 1),  # 1 = BlackIsZero
        (_TAG_IMAGE_DESCRIPTION, _TIFF_ASCII, len(desc_bytes), desc_offset),
        (_TAG_STRIP_OFFSETS, _TIFF_LONG, 1, data_offset),
        (_TAG_SAMPLES_PER_PIXEL, _TIFF_SHORT, 1, 1),
        (_TAG_ROWS_PER_STRIP, _TIFF_LONG, 1, height),
        (_TAG_STRIP_BYTE_COUNTS, _TIFF_LONG, 1, byte_count),
        (_TAG_X_RESOLUTION, _TIFF_RATIONAL, 1, x_res_offset),
        (_TAG_Y_RESOLUTION, _TIFF_RATIONAL, 1, y_res_offset),
        (_TAG_RESOLUTION_UNIT, _TIFF_SHORT, 1, 1),  # 1 = No absolute unit (ImageJ reads unit=um)
        (_TAG_SOFTWARE, _TIFF_ASCII, len(software_bytes), software_offset),
        (_TAG_SAMPLE_FORMAT, _TIFF_SHORT, 1, sample_format),
    ]
    entries.sort(key=lambda item: item[0])
    num_entries = len(entries)
    ifd_size = 2 + num_entries * 12 + 4
    extra_offset = ifd_offset + ifd_size
    x_res_offset = extra_offset
    y_res_offset = x_res_offset + 8
    software_offset = y_res_offset + 8
    desc_offset = software_offset + len(software_bytes)
    data_offset = desc_offset + len(desc_bytes)
    pad_bytes = b"\x00" if (data_offset % 2 == 1) else b""
    data_offset += len(pad_bytes)

    # Rebind offsets with exact num_entries
    final_entries: list[bytes] = []
    for tag, dtype_code, count, val_or_offset in entries:
        if tag == _TAG_IMAGE_DESCRIPTION:
            val_or_offset = desc_offset
        elif tag == _TAG_STRIP_OFFSETS:
            val_or_offset = data_offset
        elif tag == _TAG_X_RESOLUTION:
            val_or_offset = x_res_offset
        elif tag == _TAG_Y_RESOLUTION:
            val_or_offset = y_res_offset
        elif tag == _TAG_SOFTWARE:
            val_or_offset = software_offset
        final_entries.append(struct.pack("<HHII", tag, dtype_code, count, val_or_offset))

    header = struct.pack("<2sHI", b"II", 42, ifd_offset)
    ifd_block = struct.pack("<H", num_entries) + b"".join(final_entries) + struct.pack("<I", 0)
    extra_block = (
        struct.pack("<II", x_res[0], x_res[1])
        + struct.pack("<II", y_res[0], y_res[1])
        + software_bytes
        + desc_bytes
        + pad_bytes
    )

    with out_path.open("wb") as f:
        f.write(header)
        f.write(ifd_block)
        f.write(extra_block)
        f.write(raw_bytes)

    return out_path
