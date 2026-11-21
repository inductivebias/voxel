"""Minimal 2D ImageJ-compatible TIFF writer backed by ``tifffile``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

_SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32)


def write_tiff_2d(
    path: Path | str,
    image: np.ndarray,
    *,
    pixel_size_um: tuple[float, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a 2D ``uint8``, ``uint16``, or ``float32`` array as an ImageJ-compatible TIFF.

    Args:
        path: Output ``.tif`` file path (parent directories are created automatically).
        image: 2D numpy array.
        pixel_size_um: Optional ``(py_um, px_um)`` physical pixel pitch in micrometers.
        metadata: Optional JSON-serializable metadata stored in ImageJ's ``Info`` tag.
    """
    arr = np.ascontiguousarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image for TIFF write, got shape {arr.shape}")
    if arr.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"Unsupported TIFF dtype {arr.dtype}; expected uint8, uint16, or float32")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    resolution: tuple[float, float] | None = None
    ij_meta: dict[str, Any] = {}
    if pixel_size_um is not None and pixel_size_um[0] > 0 and pixel_size_um[1] > 0:
        resolution = (1.0 / float(pixel_size_um[1]), 1.0 / float(pixel_size_um[0]))
        ij_meta["unit"] = "um"
    if metadata:
        ij_meta["Info"] = json.dumps(metadata, sort_keys=True, default=str)

    tifffile.imwrite(
        out,
        arr,
        imagej=True,
        resolution=resolution,
        resolutionunit="NONE" if resolution is not None else None,
        metadata=ij_meta or None,
    )
    return out
