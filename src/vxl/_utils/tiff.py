"""Minimal 2D ImageJ-compatible TIFF writer backed by ``tifffile``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

_DTYPES = (np.uint8, np.uint16, np.float32)


def write_tiff_2d(
    path: Path | str,
    image: np.ndarray,
    *,
    pixel_size_um: tuple[float, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a 2D ``uint8``, ``uint16``, or ``float32`` array as an ImageJ-compatible TIFF."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image for TIFF write, got shape {arr.shape}")
    if arr.dtype not in _DTYPES:
        raise TypeError(f"Unsupported TIFF dtype {arr.dtype}; expected uint8, uint16, or float32")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    px = pixel_size_um
    res = (1.0 / float(px[1]), 1.0 / float(px[0])) if px is not None and px[0] > 0 and px[1] > 0 else None
    meta: dict[str, Any] = {"unit": "um"} if res is not None else {}
    if metadata:
        meta["Info"] = json.dumps(metadata, sort_keys=True, default=str)
    kw: dict[str, Any] = {"imagej": True, "resolution": res, "metadata": meta or None}
    if res is not None:
        kw["resolutionunit"] = "NONE"
    tifffile.imwrite(out, np.ascontiguousarray(arr), **kw)
    return out
