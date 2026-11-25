"""Stationary multi-frame SNR, photobleaching, contrast, and sharpness diagnostics + HTML report."""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from pydantic import Field
from vxlib.schema import FrozenModel

from vxl._utils.tiff import write_tiff_2d

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

_REPORT_TEMPLATE = Template(Path(__file__).with_name("_snr_report.html").read_text(encoding="utf-8"))


class PtcPoint(FrozenModel):
    """Single bin on the Photon Transfer Curve (variance vs. dark-subtracted mean intensity)."""

    mean_adu: float
    var_detrended: float
    var_raw: float
    snr_detrended: float
    pixel_count: int


class CameraSnrAnalysis(FrozenModel):
    """Stationary SNR & photobleaching diagnostic metrics for one camera."""

    camera_id: str
    channel_id: str
    profile_id: str
    width: int
    height: int
    dtype: str
    valid_bits: int
    num_signal_frames: int
    num_dark_frames: int
    exposure_time_ms: float
    pixel_size_um: tuple[float, float]
    hardware_binning: int = 1
    gpu_bin_y: int = 1
    gpu_bin_x: int = 1
    gpu_pre_filter: str = "none"
    gpu_filter_size: int = 3
    dark_offset_adu: float
    read_noise_adu: float
    background_mean_adu: float
    background_std_spatial_adu: float
    background_noise_temporal_adu: float
    signal_mean_fg_adu: float
    signal_median_fg_adu: float
    signal_p95_fg_adu: float
    signal_p99_fg_adu: float
    signal_max_adu: float
    otsu_threshold_adu: float
    foreground_area_pct: float
    saturated_pixel_pct: float
    well_fill_p99_pct: float
    dynamic_range_db: float
    effective_gain_adu_per_e: float | None = None
    snr_fg_mean_corrected: float
    snr_fg_median_corrected: float
    snr_fg_p95_corrected: float
    snr_fg_mean_corrected_db: float
    snr_fg_mean_raw: float
    snr_fg_median_raw: float
    snr_fg_p95_raw: float
    snr_fg_mean_raw_db: float
    bleach_initial_fg_adu: float
    bleach_final_fg_adu: float
    bleach_total_pct: float
    bleach_rate_per_frame_pct: float
    bleach_half_life_frames: float | None = None
    bleach_half_life_exposure_s: float | None = None
    bleach_fit_r2: float
    bleach_frame_indices: list[int] = Field(default_factory=list)
    bleach_fg_mean_trace: list[float] = Field(default_factory=list)
    bleach_fg_p95_trace: list[float] = Field(default_factory=list)
    bleach_bg_mean_trace: list[float] = Field(default_factory=list)
    bleach_fit_trace: list[float] = Field(default_factory=list)
    cnr: float
    sbr: float
    weber_contrast: float
    michelson_contrast: float
    rms_contrast: float
    tenengrad_mean_fg: float
    tenengrad_p95_fg: float
    laplacian_variance_fg: float
    brenner_score_fg: float
    sharp_in_focus_area_pct: float
    hist_bin_centers_adu: list[float] = Field(default_factory=list)
    hist_counts: list[int] = Field(default_factory=list)
    snr_hist_bin_centers: list[float] = Field(default_factory=list)
    snr_hist_counts: list[int] = Field(default_factory=list)
    ptc_points: list[PtcPoint] = Field(default_factory=list)
    saved_tiffs: dict[str, str] = Field(default_factory=dict)
    previews: dict[str, str] = Field(default_factory=dict)

    def without_previews(self) -> CameraSnrAnalysis:
        """Return a copy omitting large base64 PNG data URLs (for JSON API / metrics dumps)."""
        return self.model_copy(update={"previews": {}})


class SnrReportResult(FrozenModel):
    """Result returned by ``Instrument.run_snr_diagnostics``."""

    report_id: str
    timestamp: str
    profile_id: str
    num_frames: int
    dark_frames: int
    stage_xyz_um: tuple[float, float, float] | None = None
    output_dir: str
    report_path: str
    metrics_path: str
    report_url: str
    cameras: dict[str, CameraSnrAnalysis]


def _stack_temporal_stats(
    stack: np.ndarray, *, quartiles: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pixel mean, raw std, detrended std, and optional quartile means in one streaming pass."""
    n, (h, w) = int(stack.shape[0]), stack.shape[1:]
    mean = np.zeros((h, w), dtype=np.float32)
    m2, diff_sq = np.zeros_like(mean), np.zeros_like(mean)
    q = max(1, n // 4)
    fq = np.zeros_like(mean) if quartiles else np.empty(0, dtype=np.float32)
    lq = np.zeros_like(mean) if quartiles else np.empty(0, dtype=np.float32)
    delta, tmp, prev = np.empty_like(mean), np.empty_like(mean), None

    for t in range(n):
        frame = np.asarray(stack[t], dtype=np.float32)
        if quartiles and t < q:
            fq += frame
        if quartiles and t >= n - q:
            lq += frame
        np.subtract(frame, mean, out=delta)
        np.divide(delta, np.float32(t + 1), out=tmp)
        mean += tmp
        np.subtract(frame, mean, out=tmp)
        tmp *= delta
        m2 += tmp
        if prev is not None:
            np.subtract(frame, prev, out=tmp)
            tmp *= tmp
            diff_sq += tmp
        prev = frame

    if n >= 2:
        m2 /= np.float32(n - 1)
        np.sqrt(m2, out=m2)
        diff_sq *= np.float32(0.5 / (n - 1))
        np.sqrt(diff_sq, out=diff_sq)
    if quartiles:
        fq /= np.float32(q)
        lq /= np.float32(q)
    return mean, m2, diff_sq, fq, lq


def _downsample_2d(arr: np.ndarray, *, max_width: int = 720, nearest: bool = False) -> np.ndarray:
    h, w = arr.shape[:2]
    if w <= max_width or max_width <= 0:
        return arr
    dsize = (max_width, max(1, round(h * (max_width / w))))
    if arr.dtype == np.bool_:
        return cv2.resize(arr.astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST).astype(bool)
    return cv2.resize(
        arr.astype(np.float32, copy=False), dsize, interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    )


def _encode_png(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}" if ok else ""


def _normalize_u8(
    arr: np.ndarray, *, vmin: float | None = None, vmax: float | None = None, scale: float = 255.0, offset: float = 0.0
) -> np.ndarray:
    clean = np.nan_to_num(_downsample_2d(arr).astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    qs = ([1.0] if vmin is None else []) + ([99.5] if vmax is None else [])
    pcts = [float(v) for v in np.percentile(clean, qs)] if qs else []
    lo = pcts[0] if vmin is None else float(vmin)
    hi = pcts[-1] if vmax is None else float(vmax)
    hi = hi if hi > lo + 1e-9 else lo + 1.0
    return (np.clip((clean - lo) / (hi - lo), 0.0, 1.0) * scale + offset).astype(np.uint8)


def _render_masks_overlay_png(
    mean_img: np.ndarray, fg_mask: np.ndarray, bg_mask: np.ndarray, sharp_mask: np.ndarray
) -> str:
    fg_s, bg_s, sharp_s = (_downsample_2d(m, nearest=True) for m in (fg_mask, bg_mask, sharp_mask))
    bgr = cv2.cvtColor(_normalize_u8(mean_img, scale=200.0), cv2.COLOR_GRAY2BGR)
    for mask, scale, tint in (
        (bg_s, 0.6, (75.0, 45.0, 30.0)),
        (fg_s & ~sharp_s, 0.65, (180.0, 140.0, 20.0)),
        (sharp_s, 0.6, (90.0, 220.0, 40.0)),
    ):
        bgr[mask] = np.clip(bgr[mask] * scale + np.array(tint, dtype=np.float32), 0, 255).astype(np.uint8)
    return _encode_png(bgr)


def _dark_stats(
    mean_proj: np.ndarray, std_detrended: np.ndarray, dark_frames: np.ndarray | None, p1_all: float, p5_all: float
) -> tuple[int, float, float, np.ndarray]:
    if dark_frames is not None and (d_arr := np.asarray(dark_frames)).ndim == 3 and d_arr.shape[0] >= 1:
        dark_mean, _, read_map, _, _ = _stack_temporal_stats(d_arr)
        return (
            int(d_arr.shape[0]),
            float(np.median(dark_mean)),
            float(np.median(read_map)),
            np.maximum(mean_proj - dark_mean, np.float32(0.0)),
        )
    bg_floor = mean_proj <= p5_all
    read_noise = float(np.median(std_detrended[bg_floor])) if np.any(bg_floor) else 1.0
    return 0, p1_all, read_noise, np.maximum(mean_proj - np.float32(p1_all), np.float32(0.0))


def _segment_masks(
    mean_proj: np.ndarray, dark_offset: float, read_noise: float, p05: float, p25: float, p50: float, p998: float
) -> tuple[float, np.ndarray, np.ndarray]:
    span = max(1.0, p998 - p05)
    norm_u8 = np.clip((mean_proj - p05) * (255.0 / span), 0.0, 255.0).astype(np.uint8)
    otsu_u8 = (
        128.0
        if norm_u8.min() == norm_u8.max()
        else cv2.threshold(norm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU, dst=norm_u8)[0]
    )
    otsu_thresh = max(p05 + (otsu_u8 / 255.0) * span, dark_offset + 4.0 * read_noise)
    fg_mask = mean_proj > otsu_thresh
    if not np.any(fg_mask):
        fg_mask = mean_proj >= p50
    bg_mask = (mean_proj <= min(otsu_thresh, p25)) & ~fg_mask
    if not np.any(bg_mask):
        bg_mask = ~fg_mask if np.any(~fg_mask) else mean_proj <= p50
    return otsu_thresh, fg_mask, bg_mask


def _compute_bleaching(
    stack: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    dark_offset: float,
    exposure_ms: float,
    first_q: np.ndarray,
    last_q: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    n_frames, off = int(stack.shape[0]), np.float32(dark_offset)
    fg_mean, fg_p95, bg_mean = [], [], []
    for t in range(n_frames):
        frame = np.asarray(stack[t], dtype=np.float32)
        fg_vals = np.maximum(frame[fg_mask] - off, np.float32(0.0))
        fg_mean.append(float(np.mean(fg_vals)))
        fg_p95.append(float(np.percentile(fg_vals, 95.0)))
        bg_mean.append(float(np.mean(frame[bg_mask])))

    y_arr = np.maximum(np.asarray(fg_mean, dtype=np.float64), 1e-3)
    slope, intercept = np.polyfit(np.arange(n_frames, dtype=np.float64), np.log(y_arr), 1)
    k_decay, i0_fit = max(0.0, float(-slope)), float(math.exp(intercept))
    fit_trace = [float(i0_fit * math.exp(-k_decay * t)) for t in range(n_frames)]
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    fit_r2 = (
        float(max(0.0, 1.0 - float(np.sum((y_arr - np.asarray(fit_trace)) ** 2)) / ss_tot)) if ss_tot > 1e-9 else 0.0
    )
    initial_fg, final_fg = fg_mean[0], fg_mean[-1]
    total_pct = (initial_fg - final_fg) / max(initial_fg, 1e-3) * 100.0 if initial_fg > 0 else 0.0
    half_frames = round(math.log(2.0) / k_decay, 2) if k_decay > 1e-6 else None
    fq, lq = first_q - off, last_q - off
    bleach_map = np.where(
        fg_mask & (fq > 1.0),
        np.clip((fq - lq) / np.maximum(fq, np.float32(1.0)) * np.float32(100.0), -25.0, 100.0),
        0.0,
    ).astype(np.float32)
    fields: dict[str, Any] = {
        "bleach_initial_fg_adu": round(initial_fg, 2),
        "bleach_final_fg_adu": round(final_fg, 2),
        "bleach_total_pct": round(total_pct, 3),
        "bleach_rate_per_frame_pct": round((1.0 - math.exp(-k_decay)) * 100.0, 4),
        "bleach_half_life_frames": half_frames,
        "bleach_half_life_exposure_s": round(half_frames * (exposure_ms / 1000.0), 3)
        if half_frames is not None
        else None,
        "bleach_fit_r2": round(fit_r2, 4),
        "bleach_frame_indices": list(range(1, n_frames + 1)),
        "bleach_fg_mean_trace": [round(v, 2) for v in fg_mean],
        "bleach_fg_p95_trace": [round(v, 2) for v in fg_p95],
        "bleach_bg_mean_trace": [round(v, 2) for v in bg_mean],
        "bleach_fit_trace": [round(v, 2) for v in fit_trace],
    }
    return fields, bleach_map


def _compute_sharpness(
    mean_proj: np.ndarray, sig_mean: np.ndarray, fg_mask: np.ndarray, bg_noise: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    grad_mag = cv2.magnitude(
        cv2.Sobel(mean_proj, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(mean_proj, cv2.CV_32F, 0, 1, ksize=3)
    )
    lap = cv2.Laplacian(mean_proj, cv2.CV_32F, ksize=3)
    h, w = mean_proj.shape
    brenner_sum = 0.0
    if w > 2:
        dx = mean_proj[:, 2:][fg_mask[:, :-2]] - mean_proj[:, :-2][fg_mask[:, :-2]]
        brenner_sum += float(np.dot(dx, dx))
    if h > 2:
        dy = mean_proj[2:, :][fg_mask[:-2, :]] - mean_proj[:-2, :][fg_mask[:-2, :]]
        brenner_sum += float(np.dot(dy, dy))
    sharpness_map = cv2.blur(grad_mag, (15, 15)) / np.maximum(
        cv2.blur(sig_mean, (15, 15)), np.float32(max(10.0, bg_noise * 3.0))
    )
    fg_sharp, fg_grad = sharpness_map[fg_mask], grad_mag[fg_mask]
    sharp_mask = fg_mask & (sharpness_map >= (float(np.percentile(fg_sharp, 55.0)) if fg_sharp.size > 0 else 0.0))
    fields: dict[str, Any] = {
        "tenengrad_mean_fg": round(float(np.mean(fg_grad)), 2),
        "tenengrad_p95_fg": round(float(np.percentile(fg_grad, 95.0)), 2),
        "laplacian_variance_fg": round(float(np.var(lap[fg_mask])), 2),
        "brenner_score_fg": round(brenner_sum / max(1, int(np.count_nonzero(fg_mask))), 2),
        "sharp_in_focus_area_pct": round(float(np.mean(sharp_mask) * 100.0), 2),
    }
    return fields, sharpness_map, sharp_mask


def _compute_ptc(
    mean_proj: np.ndarray, sig_mean: np.ndarray, std_detrended: np.ndarray, std_raw: np.ndarray, max_code: float
) -> tuple[list[PtcPoint], float | None]:
    valid_px = (sig_mean > 1.0) & (mean_proj < 0.98 * max_code)
    if np.count_nonzero(valid_px) < 128:
        return [], None
    s_vals, sd_v, sr_v = sig_mean[valid_px], std_detrended[valid_px], std_raw[valid_px]
    order = np.argsort(s_vals)
    s_sorted, vd_sorted, vr_sorted = s_vals[order], (sd_v * sd_v)[order], (sr_v * sr_v)[order]
    edges = np.unique(np.percentile(s_sorted, np.linspace(2.0, 98.0, 21)))
    if len(edges) < 2:
        return [], None
    starts, ends = np.searchsorted(s_sorted, edges[:-1], side="left"), np.searchsorted(s_sorted, edges[1:], side="left")
    ends[-1] = int(np.searchsorted(s_sorted, edges[-1], side="right"))

    points: list[PtcPoint] = []
    for lo_i, hi_i in zip(starts.tolist(), ends.tolist(), strict=True):
        if (cnt := hi_i - lo_i) < 16:
            continue
        mid = lo_i + cnt // 2
        m_adu = float(s_sorted[mid]) if cnt % 2 else float(0.5 * (s_sorted[mid - 1] + s_sorted[mid]))
        v_det, v_raw = float(np.median(vd_sorted[lo_i:hi_i])), float(np.median(vr_sorted[lo_i:hi_i]))
        points.append(
            PtcPoint(
                mean_adu=round(m_adu, 2),
                var_detrended=round(v_det, 3),
                var_raw=round(v_raw, 3),
                snr_detrended=round(m_adu / math.sqrt(max(v_det, 1e-6)), 2),
                pixel_count=cnt,
            )
        )
    gain: float | None = None
    if len(points) >= 4:
        slope, _ = np.polyfit([p.mean_adu for p in points], [p.var_detrended for p in points], 1)
        if 0.001 < slope < 50.0:
            gain = round(float(slope), 4)
    return points, gain


def _histogram(
    data: np.ndarray, bins: int, value_range: tuple[float, float], decimals: int
) -> tuple[list[float], list[int]]:
    counts, edges = np.histogram(data, bins=bins, range=value_range)
    return [round(float(v), decimals) for v in (0.5 * (edges[:-1] + edges[1:]))], [int(c) for c in counts]


def compute_camera_snr_diagnostics(
    frames: np.ndarray,
    *,
    dark_frames: np.ndarray | None = None,
    camera_id: str,
    channel_id: str = "default",
    profile_id: str = "default",
    valid_bits: int = 16,
    exposure_time_ms: float = 10.0,
    pixel_size_um: tuple[float, float] = (1.0, 1.0),
    hardware_binning: int = 1,
    gpu_bin_y: int = 1,
    gpu_bin_x: int = 1,
    gpu_pre_filter: str = "none",
    gpu_filter_size: int = 3,
    output_dir: Path | str | None = None,
) -> CameraSnrAnalysis:
    """Compute stationary multi-frame SNR, photobleaching, contrast, and sharpness diagnostics."""
    stack = np.asarray(frames)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(f"Expected 3D frame stack with N >= 2 frames, got shape {stack.shape}")

    n_frames, height, width = int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2])
    mean_proj, std_raw, std_detrended, first_q, last_q = _stack_temporal_stats(stack, quartiles=True)
    p05, p1_all, p5_all, p25_all, p50_all, p99_all, p998 = (
        float(v) for v in np.percentile(mean_proj, [0.5, 1.0, 5.0, 25.0, 50.0, 99.0, 99.8])
    )
    n_dark, dark_offset, read_noise, sig_mean = _dark_stats(mean_proj, std_detrended, dark_frames, p1_all, p5_all)
    otsu_thresh, fg_mask, bg_mask = _segment_masks(mean_proj, dark_offset, read_noise, p05, p25_all, p50_all, p998)

    snr_corr_map = sig_mean / np.maximum(std_detrended, np.float32(1e-3))
    snr_raw_map = sig_mean / np.maximum(std_raw, np.float32(1e-3))
    fg_snr_corr, fg_snr_raw = snr_corr_map[fg_mask], snr_raw_map[fg_mask]
    snr_mean_c, snr_mean_r = float(np.mean(fg_snr_corr)), float(np.mean(fg_snr_raw))
    snr_med_c, snr_p95_c, snr_p995_c = (float(v) for v in np.percentile(fg_snr_corr, [50.0, 95.0, 99.5]))
    snr_med_r, snr_p95_r = (float(v) for v in np.percentile(fg_snr_raw, [50.0, 95.0]))

    bleach_fields, bleach_map = _compute_bleaching(
        stack, fg_mask, bg_mask, dark_offset, exposure_time_ms, first_q, last_q
    )
    fg_px, bg_px = mean_proj[fg_mask], mean_proj[bg_mask]
    sig_mean_fg, bg_mean = float(np.mean(fg_px)), float(np.mean(bg_px))
    sig_med_fg, sig_p95_fg, sig_p99_fg = (float(v) for v in np.percentile(fg_px, [50.0, 95.0, 99.0]))
    bg_temp_noise = float(np.median(std_detrended[bg_mask]))
    max_code = float((1 << valid_bits) - 1)
    cnr = max(0.0, sig_mean_fg - bg_mean) / math.sqrt(
        max(float(np.median(std_detrended[fg_mask]) ** 2) + bg_temp_noise**2, 1e-6)
    )

    sharp_fields, sharpness_map, sharp_mask = _compute_sharpness(mean_proj, sig_mean, fg_mask, bg_temp_noise)
    ptc_points, effective_gain = _compute_ptc(mean_proj, sig_mean, std_detrended, std_raw, max_code)
    mean_min, mean_max = float(np.min(mean_proj)), float(np.max(mean_proj))
    hist_centers, hist_counts = _histogram(mean_proj, 48, (mean_min, mean_max + 1.0), 1)
    snr_hist_centers, snr_hist_counts = _histogram(fg_snr_corr, 40, (0.0, max(1.0, snr_p995_c)), 2)

    header = {
        "camera_id": camera_id,
        "channel_id": channel_id,
        "profile_id": profile_id,
        "num_signal_frames": n_frames,
        "num_dark_frames": n_dark,
        "exposure_time_ms": exposure_time_ms,
        "hardware_binning": hardware_binning,
        "gpu_bin_y": gpu_bin_y,
        "gpu_bin_x": gpu_bin_x,
        "gpu_pre_filter": gpu_pre_filter,
    }
    saved_tiffs: dict[str, str] = {}
    if output_dir is not None:
        out_root = Path(output_dir)
        mean_out = mean_proj.astype(stack.dtype if stack.dtype in (np.uint8, np.uint16) else np.float32)
        for key, suffix, arr, proj_name in (
            ("mean", "mean", mean_out, "mean"),
            ("std_detrended", "std_detrended", std_detrended, "std_bleaching_detrended"),
            ("snr_corrected", "snr_bleach_corrected", snr_corr_map, "snr_bleaching_corrected"),
            ("bleach_pct_map", "bleach_pct_map", bleach_map, "photobleaching_pct_map"),
        ):
            saved_tiffs[key] = str(
                write_tiff_2d(
                    out_root / f"{camera_id}_{suffix}.tif",
                    arr,
                    pixel_size_um=pixel_size_um,
                    metadata={**header, "projection": proj_name},
                )
            )

    bleach_vmax = max(5.0, float(np.percentile(bleach_map[fg_mask], 98.0)) if np.any(fg_mask) else 10.0)
    previews = {
        k: _encode_png(cv2.applyColorMap(_normalize_u8(arr, vmin=lo, vmax=hi, offset=0.5), cmap))
        for k, arr, lo, hi, cmap in (
            ("mean", mean_proj, None, None, cv2.COLORMAP_VIRIDIS),
            ("std_detrended", std_detrended, None, None, cv2.COLORMAP_MAGMA),
            ("snr_corrected", snr_corr_map, 0.0, None, cv2.COLORMAP_INFERNO),
            ("bleach_map", bleach_map, 0.0, bleach_vmax, cv2.COLORMAP_PLASMA),
            ("sharpness", sharpness_map, 0.0, None, cv2.COLORMAP_TURBO),
        )
    }
    previews["masks"] = _render_masks_overlay_png(mean_proj, fg_mask, bg_mask, sharp_mask)
    fg_sig = sig_mean[fg_mask]

    return CameraSnrAnalysis.model_validate(
        {
            **header,
            **bleach_fields,
            **sharp_fields,
            "width": width,
            "height": height,
            "dtype": str(stack.dtype),
            "valid_bits": valid_bits,
            "pixel_size_um": pixel_size_um,
            "gpu_filter_size": gpu_filter_size,
            "dark_offset_adu": round(dark_offset, 2),
            "read_noise_adu": round(read_noise, 3),
            "background_mean_adu": round(bg_mean, 2),
            "background_std_spatial_adu": round(float(np.std(bg_px)), 3),
            "background_noise_temporal_adu": round(bg_temp_noise, 3),
            "signal_mean_fg_adu": round(sig_mean_fg, 2),
            "signal_median_fg_adu": round(sig_med_fg, 2),
            "signal_p95_fg_adu": round(sig_p95_fg, 2),
            "signal_p99_fg_adu": round(sig_p99_fg, 2),
            "signal_max_adu": round(mean_max, 2),
            "otsu_threshold_adu": round(otsu_thresh, 2),
            "foreground_area_pct": round(float(np.mean(fg_mask) * 100.0), 2),
            "saturated_pixel_pct": round(float(np.mean(mean_proj >= 0.995 * max_code) * 100.0), 4),
            "well_fill_p99_pct": round(min(100.0, (sig_p99_fg / max(1.0, max_code)) * 100.0), 2),
            "dynamic_range_db": round(
                20.0 * math.log10(max(max(0.0, sig_p99_fg - dark_offset) / max(read_noise, 1e-3), 1.0)), 2
            ),
            "effective_gain_adu_per_e": effective_gain,
            "snr_fg_mean_corrected": round(snr_mean_c, 2),
            "snr_fg_median_corrected": round(snr_med_c, 2),
            "snr_fg_p95_corrected": round(snr_p95_c, 2),
            "snr_fg_mean_corrected_db": round(20.0 * math.log10(max(snr_mean_c, 1e-6)), 2),
            "snr_fg_mean_raw": round(snr_mean_r, 2),
            "snr_fg_median_raw": round(snr_med_r, 2),
            "snr_fg_p95_raw": round(snr_p95_r, 2),
            "snr_fg_mean_raw_db": round(20.0 * math.log10(max(snr_mean_r, 1e-6)), 2),
            "cnr": round(cnr, 2),
            "sbr": round(sig_mean_fg / max(bg_mean, 1e-3), 2),
            "weber_contrast": round((sig_mean_fg - bg_mean) / max(bg_mean, 1e-3), 2),
            "michelson_contrast": round(max(0.0, p99_all - p1_all) / max(p99_all + p1_all, 1e-3), 4),
            "rms_contrast": round(float(np.std(fg_sig) / max(float(np.mean(fg_sig)), 1e-3)), 4),
            "hist_bin_centers_adu": hist_centers,
            "hist_counts": hist_counts,
            "snr_hist_bin_centers": snr_hist_centers,
            "snr_hist_counts": snr_hist_counts,
            "ptc_points": ptc_points,
            "saved_tiffs": saved_tiffs,
            "previews": previews,
        }
    )


def _svg_frame(
    w: int,
    h: int,
    pad: tuple[int, int, int, int],
    body: str,
    xl: str,
    yl: str,
    xr: tuple[float, float],
    fmt: str = ".0f",
) -> str:
    pl, pr, pt, pb = pad
    pw, ph, (x0, x1) = w - pl - pr, h - pt - pb, xr
    ym = pt + ph / 2
    ticks = "".join(
        f'<text x="{pl + k / 4 * pw:.1f}" y="{h - 17}" text-anchor="middle" fill="#a1a1aa"'
        f' font-size="10">{x0 + (x1 - x0) * k / 4:{fmt}}</text>'
        for k in range(5)
    )
    ax = 'stroke="#52525b"'
    lbl = 'text-anchor="middle" fill="#a1a1aa" font-size="11"'
    return (
        f'<svg viewBox="0 0 {w} {h}" class="chart-svg">'
        f'<line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt + ph}" {ax} />'
        f'<line x1="{pl}" y1="{pt + ph}" x2="{w - pr}" y2="{pt + ph}" {ax} />{body}{ticks}'
        f'<text x="{pl + pw / 2:.1f}" y="{h - 3}" {lbl}>{html.escape(xl)}</text>'
        f'<text x="14" y="{ym:.1f}" {lbl} transform="rotate(-90 14 {ym:.1f})">{html.escape(yl)}</text></svg>'
    )


def _svg_line_chart(
    x_vals: Sequence[float | int],
    series: Sequence[tuple[str, Sequence[float], str, bool]],
    xl: str,
    yl: str,
    w: int = 540,
    h: int = 240,
) -> str:
    if not x_vals or not series or not series[0][1]:
        return '<div class="empty-chart">Insufficient data points</div>'
    all_y = [v for _, ys, _, _ in series for v in ys if math.isfinite(v)]
    if not all_y:
        return '<div class="empty-chart">No finite values</div>'
    pad = (58, 20, 26, 40)
    pw, ph = w - pad[0] - pad[1], h - pad[2] - pad[3]
    x0, x1 = float(min(x_vals)), float(max(x_vals))
    x1 = x1 if x1 > x0 else x0 + 1.0
    y0, y1 = float(min(all_y)), float(max(all_y))
    y1 = y1 if y1 > y0 else y0 + 1.0
    py_pad = 0.08 * (y1 - y0)
    y0, y1 = (max(0.0, y0 - py_pad) if y0 >= 0 else y0 - py_pad), y1 + py_pad
    sx = lambda x: pad[0] + (x - x0) / (x1 - x0) * pw  # noqa: E731
    sy = lambda y: pad[2] + ph - (y - y0) / (y1 - y0) * ph  # noqa: E731
    parts = [
        f'<line x1="{pad[0]}" y1="{(py := sy(gy)):.1f}" x2="{w - pad[1]}" y2="{py:.1f}"'
        f' stroke="#27272a" stroke-dasharray="3,3" />'
        f'<text x="{pad[0] - 6}" y="{py + 4:.1f}" text-anchor="end" fill="#a1a1aa" font-size="10">{gy:.1f}</text>'
        for i in range(5)
        for gy in (y0 + (y1 - y0) * (i / 4.0),)
    ]
    lx = float(pad[0] + 6)
    for lbl, ys, col, dashed in series:
        pts = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(x_vals, ys, strict=False))
        d = ' stroke-dasharray="5,4"' if dashed else ""
        parts.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="2"{d} points="{pts}" />'
            f'<line x1="{lx}" y1="13" x2="{lx + 16}" y2="13" stroke="{col}" stroke-width="2"{d} />'
            f'<text x="{lx + 20}" y="17" fill="#d4d4d8" font-size="11">{html.escape(lbl)}</text>'
        )
        lx += 28 + len(lbl) * 6.5
    return _svg_frame(w, h, pad, "".join(parts), xl, yl, (x0, x1))


def _svg_bar_histogram(
    centers: Sequence[float], counts: Sequence[int], col: str, xl: str, w: int = 540, h: int = 220
) -> str:
    if not centers or not counts:
        return '<div class="empty-chart">No histogram data</div>'
    pad = (52, 16, 18, 38)
    pw, ph, n = w - pad[0] - pad[1], h - pad[2] - pad[3], len(centers)
    logs = [math.log10(c + 1) for c in counts]
    ymax, bw = max(1.0, *logs), max(2.0, pw / max(1, n) - 1.5)
    bars = "".join(
        f'<rect x="{pad[0] + i / max(1, n) * pw:.1f}" y="{pad[2] + ph - lc / ymax * ph:.1f}"'
        f' width="{bw:.1f}" height="{lc / ymax * ph:.1f}" fill="{col}" opacity="0.82" rx="1" />'
        for i, lc in enumerate(logs)
    )
    return _svg_frame(w, h, pad, bars, xl, "Pixels (log10)", (centers[0], centers[-1]), ".1f")


def _render_camera_section(cam_id: str, c: CameraSnrAnalysis, output_dir: Path) -> str:
    b = " &bull; "
    hl = (
        f"{c.bleach_half_life_frames:.0f} frames ({c.bleach_half_life_exposure_s:.2f} s)"
        if c.bleach_half_life_frames is not None and c.bleach_half_life_exposure_s is not None
        else "Negligible bleaching"
    )
    hl_esc = html.escape(hl)
    bin_desc = f"HW {c.hardware_binning}x{c.hardware_binning} | GPU {c.gpu_bin_y}x{c.gpu_bin_x} ({c.gpu_pre_filter})"
    gain_str = f"{c.effective_gain_adu_per_e:.3f} ADU/e-" if c.effective_gain_adu_per_e is not None else "N/A"
    snr_c_3 = f"{c.snr_fg_mean_corrected:.2f}</strong> / {c.snr_fg_median_corrected:.2f} / {c.snr_fg_p95_corrected:.2f}"
    snr_r_3 = f"{c.snr_fg_mean_raw:.2f} / {c.snr_fg_median_raw:.2f} / {c.snr_fg_p95_raw:.2f}"
    k_snr_sub = f"{c.snr_fg_mean_corrected_db:.1f} dB (Raw: {c.snr_fg_mean_raw:.1f})"
    k_con_sub = f"Weber: {c.weber_contrast:.2f}{b}Michelson: {c.michelson_contrast:.3f}"
    k_sig_sub = f"Well Fill P99: {c.well_fill_p99_pct:.1f}%{b}Sat: {c.saturated_pixel_pct:.3f}%"
    kpis = [
        (
            " highlight",
            "Bleach-Corrected SNR (FG Mean / P95)",
            f"{c.snr_fg_mean_corrected:.1f} / {c.snr_fg_p95_corrected:.1f}",
            k_snr_sub,
        ),
        (
            "",
            f"Photobleaching ({c.num_signal_frames} frames)",
            f"{c.bleach_total_pct:+.2f}%",
            f"{c.bleach_rate_per_frame_pct:.3f}%/frame{b}t1/2: {hl_esc}",
        ),
        ("", "Contrast (CNR / SBR)", f"{c.cnr:.1f} / {c.sbr:.2f}x", k_con_sub),
        ("", "Signal P95 / Well Fill", f"{c.signal_p95_fg_adu:.0f} ADU", k_sig_sub),
        (
            "",
            "Sharpness (Tenengrad)",
            f"{c.tenengrad_mean_fg:.1f}",
            f"In-Focus Mask: {c.sharp_in_focus_area_pct:.1f}% of FOV",
        ),
    ]
    kpi_html = "".join(
        f'<div class="kpi-card{cls}"><span class="kpi-label">{lbl}</span>'
        f'<span class="kpi-val">{val}</span><span class="kpi-sub">{sub}</span></div>'
        for cls, lbl, val, sub in kpis
    )

    c_mean = (
        f"FG Mean: {c.signal_mean_fg_adu:.1f} ADU{b}BG Mean: {c.background_mean_adu:.1f} ADU"
        f"{b}Max: {c.signal_max_adu:.0f} ADU"
    )
    c_noise = f"Read Floor: {c.read_noise_adu:.2f} ADU{b}BG Temporal &sigma;: {c.background_noise_temporal_adu:.2f} ADU"
    c_snr = (
        f"FG Median SNR: {c.snr_fg_median_corrected:.1f}{b}"
        f"P95 SNR: {c.snr_fg_p95_corrected:.1f} ({c.snr_fg_mean_corrected_db:.1f} dB)"
    )
    c_bleach = (
        f"Initial FG: {c.bleach_initial_fg_adu:.1f} ADU &rarr; Final FG: {c.bleach_final_fg_adu:.1f} ADU"
        f"{b}R&sup2;={c.bleach_fit_r2:.3f}"
    )
    c_masks = (
        f'<span style="color:#4ade80">&#9632;</span> Sharp Focus ({c.sharp_in_focus_area_pct:.1f}%){b}'
        f'<span style="color:#38bdf8">&#9632;</span> Foreground ({c.foreground_area_pct:.1f}%){b}'
        f'<span style="color:#94a3b8">&#9632;</span> Background (Otsu={c.otsu_threshold_adu:.0f})'
    )
    c_sharp = (
        f"Tenengrad P95: {c.tenengrad_p95_fg:.1f}{b}Laplacian Var: {c.laplacian_variance_fg:.1f}"
        f"{b}Brenner: {c.brenner_score_fg:.1f}"
    )
    maps = [
        ("Mean Projection &mu;(y, x)", "Viridis", "mean", c_mean),
        ("Bleaching-Detrended Noise &sigma;_detrended(y, x)", "Magma", "std_detrended", c_noise),
        ("Bleaching-Corrected Temporal SNR Map", "Inferno", "snr_corrected", c_snr),
        ("Spatial Photobleaching Map (% Loss)", "Plasma", "bleach_map", c_bleach),
        ("Signal, Background &amp; In-Focus Sharpness Masks", "Overlay", "masks", c_masks),
        ("Local Normalized Sharpness Map", "Turbo", "sharpness", c_sharp),
    ]
    map_html = "".join(
        f'<div class="map-card"><div class="map-title">{t} <span class="tag">{tag}</span></div>'
        f'<img src="{c.previews.get(k, "")}" alt="{html.escape(tag)}" /><div class="map-caption">{cap}</div></div>'
        for t, tag, k, cap in maps
    )

    bleach_s = [
        ("Foreground Mean (ADU)", c.bleach_fg_mean_trace, "#38bdf8", False),
        ("Exp Decay Fit", c.bleach_fit_trace, "#f43f5e", True),
        ("Background Mean", c.bleach_bg_mean_trace, "#a1a1aa", True),
    ]
    ptc_s = [
        ("Detrended Var (Shot+Read)", [p.var_detrended for p in c.ptc_points], "#10b981", False),
        ("Raw Var (incl. Bleaching)", [p.var_raw for p in c.ptc_points], "#f59e0b", True),
    ]
    bleach_svg = _svg_line_chart(
        c.bleach_frame_indices, bleach_s, "Frame Index (Static Sample)", "Dark-Subtracted Intensity (ADU)"
    )
    ptc_svg = _svg_line_chart(
        [p.mean_adu for p in c.ptc_points], ptc_s, "Mean Signal Above Dark (ADU)", "Temporal Variance (ADU^2)"
    )
    hist_svg = _svg_bar_histogram(c.hist_bin_centers_adu, c.hist_counts, "#38bdf8", "Pixel Intensity (ADU)")
    snr_svg = _svg_bar_histogram(
        c.snr_hist_bin_centers, c.snr_hist_counts, "#a855f7", "Foreground Bleaching-Corrected SNR"
    )
    charts = [
        (f"Stationary Photobleaching Decay Curve (N={c.num_signal_frames} frames)", bleach_svg),
        (f"Photon Transfer Curve (Detrended vs. Raw Variance){b}Est. Gain: {html.escape(gain_str)}", ptc_svg),
        ("Mean Projection Intensity Distribution", hist_svg),
        ("Foreground Bleaching-Corrected Temporal SNR Distribution", snr_svg),
    ]
    chart_html = "".join(f'<div class="chart-card"><div class="chart-title">{t}</div>{svg}</div>' for t, svg in charts)

    sig_row = f"{c.signal_p95_fg_adu:.1f} / {c.signal_p99_fg_adu:.1f} / {c.signal_max_adu:.1f} ADU"
    dark_row = f"{c.dark_offset_adu:.2f} / {c.read_noise_adu:.3f} ADU ({c.num_dark_frames} dark)"
    b_row = f"{c.bleach_total_pct:+.3f}% ({c.bleach_rate_per_frame_pct:.4f}% / frame, R&sup2;={c.bleach_fit_r2:.3f})"
    dr_row = f"{c.dynamic_range_db:.2f} dB / {c.well_fill_p99_pct:.2f}% (Sat: {c.saturated_pixel_pct:.4f}%)"
    tables = [
        (
            "Signal, Noise &amp; Photobleaching Metrics",
            [
                ("Signal Mean / Median (Foreground)", f"{c.signal_mean_fg_adu:.2f} / {c.signal_median_fg_adu:.2f} ADU"),
                ("Signal P95 / P99 / Max", sig_row),
                ("Dark Offset / Read Noise Floor", dark_row),
                (
                    "Bleach-Corrected Temporal SNR (Mean / Median / P95)",
                    f"<strong>{snr_c_3} ({c.snr_fg_mean_corrected_db:.2f} dB)",
                ),
                ("Uncorrected Raw Temporal SNR (Mean / Median / P95)", f"{snr_r_3} ({c.snr_fg_mean_raw_db:.2f} dB)"),
                ("Photobleaching Total Loss / Rate per Frame", b_row),
                ("Photobleaching Half-Life (t1/2)", hl_esc),
            ],
        ),
        (
            "Contrast, Dynamic Range &amp; Sharpness Metrics",
            [
                ("Contrast-to-Noise Ratio (CNR)", f"<strong>{c.cnr:.2f}</strong>"),
                ("Signal-to-Background Ratio (SBR) / Weber Contrast", f"{c.sbr:.2f}x / {c.weber_contrast:.2f}"),
                ("Michelson / RMS Contrast", f"{c.michelson_contrast:.4f} / {c.rms_contrast:.4f}"),
                ("Dynamic Range / Sensor Well Fill (P99)", dr_row),
                ("Tenengrad Sharpness (FG Mean / P95)", f"{c.tenengrad_mean_fg:.2f} / {c.tenengrad_p95_fg:.2f}"),
                (
                    "Laplacian Variance / Brenner Focus Score",
                    f"{c.laplacian_variance_fg:.2f} / {c.brenner_score_fg:.2f}",
                ),
                ("Saved TIFF Projections", f"<code>{html.escape(str(output_dir))}</code>"),
            ],
        ),
    ]
    tables_html = "".join(
        f'<table class="metrics-table"><thead><tr><th colspan="2">{title}</th></tr></thead>'
        f"<tbody>{''.join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rows)}</tbody></table>"
        for title, rows in tables
    )
    meta_bits = (
        f"Frame: <strong>{c.width} &times; {c.height} px</strong> ({html.escape(c.dtype)}, {c.valid_bits}-bit){b}"
        f"Sampling: <strong>{c.pixel_size_um[0]:.3f} &times; {c.pixel_size_um[1]:.3f} &mu;m/px</strong>{b}"
        f"Exposure: <strong>{c.exposure_time_ms:.2f} ms</strong>{b}Binning: <strong>{html.escape(bin_desc)}</strong>"
    )
    return (
        f'<section class="cam-section"><div class="cam-header"><div>'
        f"<h2>Camera: <code>{html.escape(cam_id)}</code>"
        f' <span class="sub">Channel: {html.escape(c.channel_id)}</span></h2>'
        f'<p class="meta-line">{meta_bits}</p></div><div class="kpi-strip">{kpi_html}</div></div>'
        f'<div class="map-grid">{map_html}</div><div class="charts-grid">{chart_html}</div>'
        f'<div class="tables-grid">{tables_html}</div></section>'
    )


def render_snr_html_report(
    *,
    report_id: str,
    timestamp: datetime,
    instrument_name: str,
    profile_id: str,
    stage_xyz_um: tuple[float, float, float] | None,
    laser_summary: dict[str, dict[str, Any]],
    cameras: dict[str, CameraSnrAnalysis],
    output_dir: Path,
) -> Path:
    """Write ``snr_report.html`` and companion ``snr_metrics.json`` under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "snr_report.html"
    xyz_str = (
        f"X={stage_xyz_um[0]:.1f} um, Y={stage_xyz_um[1]:.1f} um, Z={stage_xyz_um[2]:.1f} um"
        if stage_xyz_um
        else "Stationary"
    )
    badges = " ".join(
        f'<span class="badge badge-laser">{html.escape(lid)}: {info.get("wavelength", "")}nm'
        f" @ {info.get('power_setpoint_mw', info.get('power_mw', '?'))} mW</span>"
        for lid, info in laser_summary.items()
    )
    meta = {"report_id": report_id, "instrument_name": instrument_name, "profile_id": profile_id}
    html_doc = _REPORT_TEMPLATE.substitute(
        **{k: html.escape(v) for k, v in meta.items()},
        xyz_str=html.escape(xyz_str),
        captured_str=html.escape(timestamp.strftime("%Y-%m-%d %H:%M:%S")),
        laser_badges=badges,
        cam_sections="".join(_render_camera_section(cid, cam, output_dir) for cid, cam in cameras.items()),
    )
    report_path.write_text(html_doc, encoding="utf-8")
    payload = {
        **meta,
        "timestamp": timestamp.isoformat(),
        "stage_xyz_um": list(stage_xyz_um) if stage_xyz_um is not None else None,
        "lasers": laser_summary,
        "cameras": {cid: cam.model_dump(exclude={"previews"}) for cid, cam in cameras.items()},
    }
    (output_dir / "snr_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def read_snr_report_html(report_root: Path, report_id: str) -> str:
    """Read a previously generated ``snr_report.html`` under ``report_root``."""
    if "/" in report_id or "\\" in report_id or ".." in report_id:
        raise FileNotFoundError(report_id)
    report_path = report_root / report_id / "snr_report.html"
    if not report_path.is_file():
        raise FileNotFoundError(f"SNR report '{report_id}' not found")
    return report_path.read_text(encoding="utf-8")
