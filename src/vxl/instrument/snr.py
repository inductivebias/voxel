"""Stationary multi-frame SNR, Photobleaching, Contrast, and Sharpness diagnostic engine & HTML report generator."""

from __future__ import annotations

import base64
import html
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import BaseModel, Field

from vxl._utils.tiff import write_tiff_2d

log = logging.getLogger(__name__)


class PtcPoint(BaseModel):
    """Single bin on the Photon Transfer Curve (variance vs. dark-subtracted mean intensity)."""

    mean_adu: float
    var_detrended: float
    var_raw: float
    snr_detrended: float
    pixel_count: int


class CameraSnrAnalysis(BaseModel):
    """Comprehensive stationary SNR & photobleaching diagnostic metrics for a single camera."""

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

    # Signal strength & noise floor
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

    # Temporal SNR (Bleaching-Corrected vs Raw)
    snr_fg_mean_corrected: float
    snr_fg_median_corrected: float
    snr_fg_p95_corrected: float
    snr_fg_mean_corrected_db: float
    snr_fg_mean_raw: float
    snr_fg_median_raw: float
    snr_fg_p95_raw: float
    snr_fg_mean_raw_db: float

    # Photobleaching metrics
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

    # Contrast metrics
    cnr: float
    sbr: float
    weber_contrast: float
    michelson_contrast: float
    rms_contrast: float

    # Sharpness metrics
    tenengrad_mean_fg: float
    tenengrad_p95_fg: float
    laplacian_variance_fg: float
    brenner_score_fg: float
    sharp_in_focus_area_pct: float

    # Histograms & PTC
    hist_bin_centers_adu: list[float] = Field(default_factory=list)
    hist_counts: list[int] = Field(default_factory=list)
    snr_hist_bin_centers: list[float] = Field(default_factory=list)
    snr_hist_counts: list[int] = Field(default_factory=list)
    ptc_points: list[PtcPoint] = Field(default_factory=list)

    # Saved TIFF paths
    saved_tiffs: dict[str, str] = Field(default_factory=dict)

    # Base64 PNG previews for the self-contained HTML report
    preview_mean_png: str = ""
    preview_std_detrended_png: str = ""
    preview_snr_corrected_png: str = ""
    preview_bleach_map_png: str = ""
    preview_masks_png: str = ""
    preview_sharpness_png: str = ""


class SnrReportResult(BaseModel):
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


def _encode_png_data_url(bgr_or_gray: np.ndarray, *, max_width: int = 720) -> str:
    """Resize image to at most ``max_width`` and encode as a ``data:image/png;base64,...`` URL."""
    h, w = bgr_or_gray.shape[:2]
    if w > max_width and w > 0:
        scale = max_width / float(w)
        new_h = max(1, round(h * scale))
        bgr_or_gray = cv2.resize(bgr_or_gray, (max_width, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", bgr_or_gray)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_heatmap_png(
    arr: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: int = cv2.COLORMAP_INFERNO,
    max_width: int = 720,
) -> str:
    """Render a 2D float array as a colormapped base64 PNG data URL."""
    clean = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(clean, 1.0)) if vmin is None else float(vmin)
    hi = float(np.percentile(clean, 99.5)) if vmax is None else float(vmax)
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    norm = np.clip((clean - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0 + 0.5).astype(np.uint8)
    colored = cv2.applyColorMap(u8, colormap)
    return _encode_png_data_url(colored, max_width=max_width)


def _render_masks_overlay_png(
    mean_img: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    sharp_mask: np.ndarray,
    *,
    max_width: int = 720,
) -> str:
    """Render a color overlay showing Foreground (cyan), In-Focus Sharp region (emerald green), and Background (amber/slate)."""
    clean = np.nan_to_num(mean_img.astype(np.float32), nan=0.0)
    lo = float(np.percentile(clean, 1.0))
    hi = float(np.percentile(clean, 99.5))
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    gray_u8 = (np.clip((clean - lo) / (hi - lo), 0.0, 1.0) * 200.0).astype(np.uint8)
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    # Tint background region subtle blue-slate, foreground cyan, sharp foreground bright green
    bgr[bg_mask] = np.clip(bgr[bg_mask] * 0.6 + np.array([75, 45, 30], dtype=np.float32), 0, 255).astype(np.uint8)
    fg_only = fg_mask & (~sharp_mask)
    bgr[fg_only] = np.clip(bgr[fg_only] * 0.65 + np.array([180, 140, 20], dtype=np.float32), 0, 255).astype(np.uint8)
    bgr[sharp_mask] = np.clip(bgr[sharp_mask] * 0.6 + np.array([90, 220, 40], dtype=np.float32), 0, 255).astype(
        np.uint8
    )
    return _encode_png_data_url(bgr, max_width=max_width)


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
    """Compute stationary multi-frame SNR, photobleaching, contrast, and sharpness diagnostics for one camera.

    Accounts for photobleaching using both frame-by-frame exponential decay fitting and a successive-difference
    (Allan deviation) temporal noise estimator ``sigma_detrended(y, x)`` that isolates true shot + read noise
    from monotonic photobleaching drift.
    """
    stack = np.asarray(frames)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(f"Expected 3D frame stack with N >= 2 frames, got shape {stack.shape}")

    n_frames, height, width = int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2])
    stack_f32 = stack.astype(np.float32)

    # 1. Mean and raw temporal standard deviation projections
    mean_proj = np.mean(stack_f32, axis=0, dtype=np.float32)
    std_raw_proj = np.std(stack_f32, axis=0, ddof=1, dtype=np.float32)

    # 2. Bleaching-detrended temporal noise via successive frame differences (Allan deviation estimator):
    # Var(I_{t+1} - I_t) = 2 * sigma_noise^2, suppressing slow monotonic photobleaching drift by ~O(N^2).
    diffs = np.diff(stack_f32, axis=0)
    std_detrended_proj = np.sqrt(np.mean(diffs * diffs, axis=0, dtype=np.float32) * 0.5)

    # 3. Dark offset & read noise (from dark_frames if provided, else estimated from 5th percentile background)
    n_dark = 0
    if dark_frames is not None and np.asarray(dark_frames).ndim == 3 and np.asarray(dark_frames).shape[0] >= 1:
        d_stack = np.asarray(dark_frames, dtype=np.float32)
        n_dark = int(d_stack.shape[0])
        dark_mean_map = np.mean(d_stack, axis=0, dtype=np.float32)
        dark_offset_adu = float(np.median(dark_mean_map))
        if n_dark >= 2:
            d_diffs = np.diff(d_stack, axis=0)
            read_noise_map = np.sqrt(np.mean(d_diffs * d_diffs, axis=0, dtype=np.float32) * 0.5)
        else:
            read_noise_map = np.std(d_stack, axis=0, dtype=np.float32)
        read_noise_adu = float(np.median(read_noise_map))
        sig_mean_proj = np.maximum(0.0, mean_proj - dark_mean_map)
    else:
        p1, p5 = np.percentile(mean_proj, [1.0, 5.0])
        bg_floor_mask = mean_proj <= p5
        dark_offset_adu = float(p1)
        read_noise_adu = float(np.median(std_detrended_proj[bg_floor_mask])) if np.any(bg_floor_mask) else 1.0
        sig_mean_proj = np.maximum(0.0, mean_proj - dark_offset_adu)

    # 4. Foreground (M_sig) & Background (M_bg) segmentation via Otsu thresholding on 8-bit normalized mean
    lo_p, hi_p = float(np.percentile(mean_proj, 0.5)), float(np.percentile(mean_proj, 99.8))
    span = max(1.0, hi_p - lo_p)
    norm_u8 = np.clip((mean_proj - lo_p) * (255.0 / span), 0.0, 255.0).astype(np.uint8)
    hist_u8 = np.bincount(norm_u8.ravel(), minlength=256).astype(np.float64)
    prob = hist_u8 / max(1.0, float(norm_u8.size))
    omega = np.cumsum(prob)
    mu_cum = np.cumsum(prob * np.arange(256, dtype=np.float64))
    mu_total = mu_cum[-1]
    denom_otsu = omega * (1.0 - omega)
    sigma_b2 = np.where(denom_otsu > 1e-12, ((mu_total * omega - mu_cum) ** 2) / denom_otsu, 0.0)
    max_sb2 = float(np.max(sigma_b2))
    if max_sb2 > 0.0:
        # Average across tied/plateau maxima so bimodal histograms with a wide gap split at the valley midpoint
        plateau_bins = np.flatnonzero(sigma_b2 >= max_sb2 * 0.999)
        otsu_u8 = float(np.mean(plateau_bins))
    else:
        otsu_u8 = 128.0
    otsu_threshold_adu = max(
        float(lo_p + (otsu_u8 / 255.0) * span),
        float(dark_offset_adu + 4.0 * read_noise_adu),
    )
    fg_mask = mean_proj > otsu_threshold_adu
    if not np.any(fg_mask):
        fg_mask = mean_proj >= float(np.median(mean_proj))

    p25 = float(np.percentile(mean_proj, 25.0))
    bg_mask = (mean_proj <= min(otsu_threshold_adu, p25)) & (~fg_mask)
    if not np.any(bg_mask):
        bg_mask = ~fg_mask
    if not np.any(bg_mask):
        bg_mask = mean_proj <= float(np.median(mean_proj))

    foreground_area_pct = float(np.mean(fg_mask) * 100.0)

    # 5. Bleaching-Corrected vs. Raw Temporal SNR Maps
    eps = 1e-3
    noise_floor_detrended = np.maximum(std_detrended_proj, eps)
    noise_floor_raw = np.maximum(std_raw_proj, eps)
    snr_corrected_map = sig_mean_proj / noise_floor_detrended
    snr_raw_map = sig_mean_proj / noise_floor_raw

    fg_snr_corr = snr_corrected_map[fg_mask]
    fg_snr_raw = snr_raw_map[fg_mask]
    snr_fg_mean_corrected = float(np.mean(fg_snr_corr))
    snr_fg_median_corrected = float(np.median(fg_snr_corr))
    snr_fg_p95_corrected = float(np.percentile(fg_snr_corr, 95.0))
    snr_fg_mean_corrected_db = float(20.0 * math.log10(max(snr_fg_mean_corrected, 1e-6)))

    snr_fg_mean_raw = float(np.mean(fg_snr_raw))
    snr_fg_median_raw = float(np.median(fg_snr_raw))
    snr_fg_p95_raw = float(np.percentile(fg_snr_raw, 95.0))
    snr_fg_mean_raw_db = float(20.0 * math.log10(max(snr_fg_mean_raw, 1e-6)))

    # 6. Photobleaching Trace & Exponential Fit across static frames t = 0..N-1
    bleach_fg_mean_trace: list[float] = []
    bleach_fg_p95_trace: list[float] = []
    bleach_bg_mean_trace: list[float] = []
    for t in range(n_frames):
        frame_t = stack_f32[t]
        fg_vals = np.maximum(0.0, frame_t[fg_mask] - dark_offset_adu)
        bg_vals = frame_t[bg_mask]
        bleach_fg_mean_trace.append(float(np.mean(fg_vals)))
        bleach_fg_p95_trace.append(float(np.percentile(fg_vals, 95.0)))
        bleach_bg_mean_trace.append(float(np.mean(bg_vals)))

    t_arr = np.arange(n_frames, dtype=np.float64)
    y_arr = np.maximum(np.asarray(bleach_fg_mean_trace, dtype=np.float64), 1e-3)
    log_y = np.log(y_arr)
    # Linear regression on ln(I(t)) = ln(I0) - k * t
    slope, intercept = np.polyfit(t_arr, log_y, 1)
    k_decay = max(0.0, float(-slope))
    i0_fit = float(math.exp(intercept))
    fit_trace = [float(i0_fit * math.exp(-k_decay * t)) for t in range(n_frames)]

    ss_res = float(np.sum((y_arr - np.asarray(fit_trace)) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    bleach_fit_r2 = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0

    bleach_initial_fg_adu = float(bleach_fg_mean_trace[0])
    bleach_final_fg_adu = float(bleach_fg_mean_trace[-1])
    bleach_total_pct = (
        float((bleach_initial_fg_adu - bleach_final_fg_adu) / max(bleach_initial_fg_adu, 1e-3) * 100.0)
        if bleach_initial_fg_adu > 0
        else 0.0
    )
    bleach_rate_per_frame_pct = float((1.0 - math.exp(-k_decay)) * 100.0)
    if k_decay > 1e-6:
        half_life_frames: float | None = round(float(math.log(2.0) / k_decay), 2)
        half_life_s: float | None = round(float(half_life_frames * (exposure_time_ms / 1000.0)), 3)
    else:
        half_life_frames = None
        half_life_s = None

    # Spatial per-pixel photobleaching map (% loss between first quarter and last quarter of stack)
    q = max(1, n_frames // 4)
    first_q = np.mean(stack_f32[:q], axis=0, dtype=np.float32) - dark_offset_adu
    last_q = np.mean(stack_f32[-q:], axis=0, dtype=np.float32) - dark_offset_adu
    bleach_map_pct = np.where(
        fg_mask & (first_q > 1.0),
        np.clip((first_q - last_q) / np.maximum(first_q, 1.0) * 100.0, -25.0, 100.0),
        0.0,
    ).astype(np.float32)

    # 7. Signal Strength, Dynamic Range & Contrast Metrics
    fg_pixels = mean_proj[fg_mask]
    bg_pixels = mean_proj[bg_mask]
    signal_mean_fg_adu = float(np.mean(fg_pixels))
    signal_median_fg_adu = float(np.median(fg_pixels))
    signal_p95_fg_adu = float(np.percentile(fg_pixels, 95.0))
    signal_p99_fg_adu = float(np.percentile(fg_pixels, 99.0))
    signal_max_adu = float(np.max(mean_proj))

    background_mean_adu = float(np.mean(bg_pixels))
    background_std_spatial_adu = float(np.std(bg_pixels))
    background_noise_temporal_adu = float(np.median(std_detrended_proj[bg_mask]))

    max_code = float((1 << valid_bits) - 1)
    saturated_pixel_pct = float(np.mean(mean_proj >= (0.995 * max_code)) * 100.0)
    well_fill_p99_pct = float(min(100.0, (signal_p99_fg_adu / max(1.0, max_code)) * 100.0))
    peak_sig_above_dark = max(0.0, signal_p99_fg_adu - dark_offset_adu)
    dynamic_range_db = float(20.0 * math.log10(max(peak_sig_above_dark / max(read_noise_adu, 1e-3), 1.0)))

    fg_temp_var = float(np.median(std_detrended_proj[fg_mask]) ** 2)
    bg_temp_var = float(background_noise_temporal_adu**2)
    cnr = float(max(0.0, signal_mean_fg_adu - background_mean_adu) / math.sqrt(max(fg_temp_var + bg_temp_var, 1e-6)))
    sbr = float(signal_mean_fg_adu / max(background_mean_adu, 1e-3))
    weber_contrast = float((signal_mean_fg_adu - background_mean_adu) / max(background_mean_adu, 1e-3))
    p99_all, p1_all = float(np.percentile(mean_proj, 99.0)), float(np.percentile(mean_proj, 1.0))
    michelson_contrast = float(max(0.0, p99_all - p1_all) / max(p99_all + p1_all, 1e-3))
    rms_contrast = float(np.std(sig_mean_proj[fg_mask]) / max(np.mean(sig_mean_proj[fg_mask]), 1e-3))

    # 8. Sharpness Metrics, Local Sharpness Map & Binary Sharpness Mask
    gx = cv2.Sobel(mean_proj, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mean_proj, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(mean_proj, cv2.CV_32F, ksize=3)

    # Brenner focus measure (2-pixel stride differences)
    brenner_map = np.zeros_like(mean_proj, dtype=np.float32)
    if width > 2 and height > 2:
        dx2 = mean_proj[:, 2:] - mean_proj[:, :-2]
        dy2 = mean_proj[2:, :] - mean_proj[:-2, :]
        brenner_map[:, :-2] += dx2 * dx2
        brenner_map[:-2, :] += dy2 * dy2

    # Smooth local gradient energy with a 15x15 box filter normalized by local intensity to create a sharpness map
    local_grad = cv2.blur(grad_mag, (15, 15))
    local_mean = np.maximum(cv2.blur(sig_mean_proj, (15, 15)), max(10.0, background_noise_temporal_adu * 3.0))
    sharpness_map = local_grad / local_mean

    fg_sharp_vals = sharpness_map[fg_mask]
    sharp_thresh = float(np.percentile(fg_sharp_vals, 55.0)) if fg_sharp_vals.size > 0 else 0.0
    sharp_mask = fg_mask & (sharpness_map >= sharp_thresh)
    sharp_in_focus_area_pct = float(np.mean(sharp_mask) * 100.0)

    tenengrad_mean_fg = float(np.mean(grad_mag[fg_mask]))
    tenengrad_p95_fg = float(np.percentile(grad_mag[fg_mask], 95.0))
    laplacian_variance_fg = float(np.var(lap[fg_mask]))
    brenner_score_fg = float(np.mean(brenner_map[fg_mask]))

    # 9. Photon Transfer Curve (PTC: Variance vs. Dark-Subtracted Mean Intensity)
    var_detrended_map = std_detrended_proj * std_detrended_proj
    var_raw_map = std_raw_proj * std_raw_proj
    ptc_points: list[PtcPoint] = []
    valid_px = (sig_mean_proj > 1.0) & (mean_proj < 0.98 * max_code)
    effective_gain: float | None = None
    if np.count_nonzero(valid_px) >= 128:
        s_vals = sig_mean_proj[valid_px]
        vd_vals = var_detrended_map[valid_px]
        vr_vals = var_raw_map[valid_px]
        quantiles = np.linspace(2.0, 98.0, 21)
        edges = np.unique(np.percentile(s_vals, quantiles))
        for b_idx in range(len(edges) - 1):
            lo_e, hi_e = float(edges[b_idx]), float(edges[b_idx + 1])
            in_bin = (s_vals >= lo_e) & (s_vals <= hi_e if b_idx == len(edges) - 2 else s_vals < hi_e)
            cnt = int(np.count_nonzero(in_bin))
            if cnt < 16:
                continue
            m_adu = float(np.median(s_vals[in_bin]))
            v_det = float(np.median(vd_vals[in_bin]))
            v_raw = float(np.median(vr_vals[in_bin]))
            snr_det = float(m_adu / math.sqrt(max(v_det, 1e-6)))
            ptc_points.append(
                PtcPoint(
                    mean_adu=round(m_adu, 2),
                    var_detrended=round(v_det, 3),
                    var_raw=round(v_raw, 3),
                    snr_detrended=round(snr_det, 2),
                    pixel_count=cnt,
                )
            )
        if len(ptc_points) >= 4:
            x_ptc = np.array([p.mean_adu for p in ptc_points], dtype=np.float64)
            y_ptc = np.array([p.var_detrended for p in ptc_points], dtype=np.float64)
            gain_slope, _ = np.polyfit(x_ptc, y_ptc, 1)
            if 0.001 < gain_slope < 50.0:
                effective_gain = round(float(gain_slope), 4)

    # 10. Intensity & SNR Histograms (48 bins)
    hist_counts_arr, hist_edges = np.histogram(
        mean_proj, bins=48, range=(float(np.min(mean_proj)), float(np.max(mean_proj)) + 1.0)
    )
    hist_centers = [round(float(0.5 * (hist_edges[i] + hist_edges[i + 1])), 1) for i in range(len(hist_counts_arr))]

    snr_max_hist = max(1.0, float(np.percentile(fg_snr_corr, 99.5)))
    snr_counts_arr, snr_edges = np.histogram(fg_snr_corr, bins=40, range=(0.0, snr_max_hist))
    snr_centers = [round(float(0.5 * (snr_edges[i] + snr_edges[i + 1])), 2) for i in range(len(snr_counts_arr))]

    # 11. Save TIFF projections if output_dir is given
    saved_tiffs: dict[str, str] = {}
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        meta_common = {
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
        mean_tif = write_tiff_2d(
            out_root / f"{camera_id}_mean.tif",
            mean_proj.astype(stack.dtype if stack.dtype in (np.uint8, np.uint16) else np.float32),
            pixel_size_um=pixel_size_um,
            metadata={**meta_common, "projection": "mean"},
        )
        std_det_tif = write_tiff_2d(
            out_root / f"{camera_id}_std_detrended.tif",
            std_detrended_proj,
            pixel_size_um=pixel_size_um,
            metadata={**meta_common, "projection": "std_bleaching_detrended"},
        )
        snr_corr_tif = write_tiff_2d(
            out_root / f"{camera_id}_snr_bleach_corrected.tif",
            snr_corrected_map,
            pixel_size_um=pixel_size_um,
            metadata={**meta_common, "projection": "snr_bleaching_corrected"},
        )
        bleach_tif = write_tiff_2d(
            out_root / f"{camera_id}_bleach_pct_map.tif",
            bleach_map_pct,
            pixel_size_um=pixel_size_um,
            metadata={**meta_common, "projection": "photobleaching_pct_map"},
        )
        saved_tiffs = {
            "mean": str(mean_tif),
            "std_detrended": str(std_det_tif),
            "snr_corrected": str(snr_corr_tif),
            "bleach_pct_map": str(bleach_tif),
        }

    # 12. Render base64 PNG maps for the HTML report
    preview_mean_png = _render_heatmap_png(mean_proj, colormap=cv2.COLORMAP_VIRIDIS)
    preview_std_detrended_png = _render_heatmap_png(std_detrended_proj, colormap=cv2.COLORMAP_MAGMA)
    preview_snr_corrected_png = _render_heatmap_png(snr_corrected_map, vmin=0.0, colormap=cv2.COLORMAP_INFERNO)
    preview_bleach_map_png = _render_heatmap_png(
        bleach_map_pct,
        vmin=0.0,
        vmax=max(5.0, float(np.percentile(bleach_map_pct[fg_mask], 98.0)) if np.any(fg_mask) else 10.0),
        colormap=cv2.COLORMAP_PLASMA,
    )
    preview_masks_png = _render_masks_overlay_png(mean_proj, fg_mask, bg_mask, sharp_mask)
    preview_sharpness_png = _render_heatmap_png(sharpness_map, vmin=0.0, colormap=cv2.COLORMAP_TURBO)

    return CameraSnrAnalysis(
        camera_id=camera_id,
        channel_id=channel_id,
        profile_id=profile_id,
        width=width,
        height=height,
        dtype=str(stack.dtype),
        valid_bits=valid_bits,
        num_signal_frames=n_frames,
        num_dark_frames=n_dark,
        exposure_time_ms=exposure_time_ms,
        pixel_size_um=pixel_size_um,
        hardware_binning=hardware_binning,
        gpu_bin_y=gpu_bin_y,
        gpu_bin_x=gpu_bin_x,
        gpu_pre_filter=gpu_pre_filter,
        gpu_filter_size=gpu_filter_size,
        dark_offset_adu=round(dark_offset_adu, 2),
        read_noise_adu=round(read_noise_adu, 3),
        background_mean_adu=round(background_mean_adu, 2),
        background_std_spatial_adu=round(background_std_spatial_adu, 3),
        background_noise_temporal_adu=round(background_noise_temporal_adu, 3),
        signal_mean_fg_adu=round(signal_mean_fg_adu, 2),
        signal_median_fg_adu=round(signal_median_fg_adu, 2),
        signal_p95_fg_adu=round(signal_p95_fg_adu, 2),
        signal_p99_fg_adu=round(signal_p99_fg_adu, 2),
        signal_max_adu=round(signal_max_adu, 2),
        otsu_threshold_adu=round(otsu_threshold_adu, 2),
        foreground_area_pct=round(foreground_area_pct, 2),
        saturated_pixel_pct=round(saturated_pixel_pct, 4),
        well_fill_p99_pct=round(well_fill_p99_pct, 2),
        dynamic_range_db=round(dynamic_range_db, 2),
        effective_gain_adu_per_e=effective_gain,
        snr_fg_mean_corrected=round(snr_fg_mean_corrected, 2),
        snr_fg_median_corrected=round(snr_fg_median_corrected, 2),
        snr_fg_p95_corrected=round(snr_fg_p95_corrected, 2),
        snr_fg_mean_corrected_db=round(snr_fg_mean_corrected_db, 2),
        snr_fg_mean_raw=round(snr_fg_mean_raw, 2),
        snr_fg_median_raw=round(snr_fg_median_raw, 2),
        snr_fg_p95_raw=round(snr_fg_p95_raw, 2),
        snr_fg_mean_raw_db=round(snr_fg_mean_raw_db, 2),
        bleach_initial_fg_adu=round(bleach_initial_fg_adu, 2),
        bleach_final_fg_adu=round(bleach_final_fg_adu, 2),
        bleach_total_pct=round(bleach_total_pct, 3),
        bleach_rate_per_frame_pct=round(bleach_rate_per_frame_pct, 4),
        bleach_half_life_frames=half_life_frames,
        bleach_half_life_exposure_s=half_life_s,
        bleach_fit_r2=round(bleach_fit_r2, 4),
        bleach_frame_indices=list(range(1, n_frames + 1)),
        bleach_fg_mean_trace=[round(v, 2) for v in bleach_fg_mean_trace],
        bleach_fg_p95_trace=[round(v, 2) for v in bleach_fg_p95_trace],
        bleach_bg_mean_trace=[round(v, 2) for v in bleach_bg_mean_trace],
        bleach_fit_trace=[round(v, 2) for v in fit_trace],
        cnr=round(cnr, 2),
        sbr=round(sbr, 2),
        weber_contrast=round(weber_contrast, 2),
        michelson_contrast=round(michelson_contrast, 4),
        rms_contrast=round(rms_contrast, 4),
        tenengrad_mean_fg=round(tenengrad_mean_fg, 2),
        tenengrad_p95_fg=round(tenengrad_p95_fg, 2),
        laplacian_variance_fg=round(laplacian_variance_fg, 2),
        brenner_score_fg=round(brenner_score_fg, 2),
        sharp_in_focus_area_pct=round(sharp_in_focus_area_pct, 2),
        hist_bin_centers_adu=hist_centers,
        hist_counts=[int(c) for c in hist_counts_arr],
        snr_hist_bin_centers=snr_centers,
        snr_hist_counts=[int(c) for c in snr_counts_arr],
        ptc_points=ptc_points,
        saved_tiffs=saved_tiffs,
        preview_mean_png=preview_mean_png,
        preview_std_detrended_png=preview_std_detrended_png,
        preview_snr_corrected_png=preview_snr_corrected_png,
        preview_bleach_map_png=preview_bleach_map_png,
        preview_masks_png=preview_masks_png,
        preview_sharpness_png=preview_sharpness_png,
    )


def _svg_line_chart(
    x_vals: list[float] | list[int],
    series: list[tuple[str, list[float], str, bool]],
    *,
    x_label: str,
    y_label: str,
    width: int = 540,
    height: int = 240,
) -> str:
    """Render a clean inline SVG line chart for bleaching curves or PTC plots."""
    if not x_vals or not series or not series[0][1]:
        return '<div class="empty-chart">Insufficient data points</div>'
    pad_l, pad_r, pad_t, pad_b = 58, 20, 26, 40
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    all_y = [v for _, ys, _, _ in series for v in ys if math.isfinite(v)]
    if not all_y:
        return '<div class="empty-chart">No finite values</div>'
    xmin, xmax = float(min(x_vals)), float(max(x_vals))
    if xmax <= xmin:
        xmax = xmin + 1.0
    ymin, ymax = float(min(all_y)), float(max(all_y))
    if ymax <= ymin:
        ymax = ymin + 1.0
    pad_y = 0.08 * (ymax - ymin)
    ymin = max(0.0, ymin - pad_y) if ymin >= 0 else ymin - pad_y
    ymax = ymax + pad_y

    def sx(x: float) -> float:
        return pad_l + ((x - xmin) / (xmax - xmin)) * pw

    def sy(y: float) -> float:
        return pad_t + ph - ((y - ymin) / (ymax - ymin)) * ph

    grid_lines: list[str] = []
    for i in range(5):
        gy = ymin + (ymax - ymin) * (i / 4.0)
        py = sy(gy)
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{py:.1f}" x2="{width - pad_r}" y2="{py:.1f}" stroke="#27272a" stroke-dasharray="3,3" />'
            f'<text x="{pad_l - 6}" y="{py + 4:.1f}" text-anchor="end" fill="#a1a1aa" font-size="10">{gy:.1f}</text>'
        )

    for i in range(5):
        gx = xmin + (xmax - xmin) * (i / 4.0)
        px = sx(gx)
        grid_lines.append(
            f'<text x="{px:.1f}" y="{height - 18}" text-anchor="middle" fill="#a1a1aa" font-size="10">{gx:.0f}</text>'
        )

    polylines: list[str] = []
    legend_items: list[str] = []
    lx = pad_l + 6
    for label, ys, color, dashed in series:
        pts = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(x_vals, ys, strict=False))
        dash_attr = ' stroke-dasharray="5,4"' if dashed else ""
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash_attr} points="{pts}" />')
        legend_items.append(
            f'<line x1="{lx}" y1="13" x2="{lx + 16}" y2="13" stroke="{color}" stroke-width="2"{dash_attr} />'
            f'<text x="{lx + 20}" y="17" fill="#d4d4d8" font-size="11">{html.escape(label)}</text>'
        )
        lx += 28 + len(label) * 6.5

    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg">'
        + "".join(grid_lines)
        + f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#52525b" />'
        + f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{width - pad_r}" y2="{pad_t + ph}" stroke="#52525b" />'
        + "".join(polylines)
        + "".join(legend_items)
        + f'<text x="{pad_l + pw / 2:.1f}" y="{height - 3}" text-anchor="middle" fill="#a1a1aa" font-size="11">{html.escape(x_label)}</text>'
        + f'<text x="14" y="{pad_t + ph / 2:.1f}" text-anchor="middle" fill="#a1a1aa" font-size="11" transform="rotate(-90 14 {pad_t + ph / 2:.1f})">{html.escape(y_label)}</text>'
        + "</svg>"
    )


def _svg_bar_histogram(
    centers: list[float],
    counts: list[int],
    *,
    color: str = "#38bdf8",
    x_label: str,
    y_label: str = "Pixels (log10)",
    width: int = 540,
    height: int = 220,
) -> str:
    """Render a log-scaled inline SVG bar histogram."""
    if not centers or not counts:
        return '<div class="empty-chart">No histogram data</div>'
    pad_l, pad_r, pad_t, pad_b = 52, 16, 18, 38
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b
    log_counts = [math.log10(c + 1) for c in counts]
    ymax = max(1.0, max(log_counts))
    n = len(centers)
    bar_w = max(2.0, pw / max(1, n) - 1.5)

    bars: list[str] = []
    for i, lc in enumerate(log_counts):
        bh = (lc / ymax) * ph
        bx = pad_l + (i / max(1, n)) * pw
        by = pad_t + ph - bh
        bars.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.82" rx="1" />'
        )

    xmin, xmax = centers[0], centers[-1]
    ticks = [
        f'<text x="{pad_l + (k / 4.0) * pw:.1f}" y="{height - 17}" text-anchor="middle" fill="#a1a1aa" font-size="10">{xmin + (xmax - xmin) * (k / 4.0):.1f}</text>'
        for k in range(5)
    ]

    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg">'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#52525b" />'
        f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{width - pad_r}" y2="{pad_t + ph}" stroke="#52525b" />'
        + "".join(bars)
        + "".join(ticks)
        + f'<text x="{pad_l + pw / 2:.1f}" y="{height - 3}" text-anchor="middle" fill="#a1a1aa" font-size="11">{html.escape(x_label)}</text>'
        + f'<text x="14" y="{pad_t + ph / 2:.1f}" text-anchor="middle" fill="#a1a1aa" font-size="11" transform="rotate(-90 14 {pad_t + ph / 2:.1f})">{html.escape(y_label)}</text>'
        + "</svg>"
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
    """Render and write a self-contained HTML diagnostic report ``snr_report.html``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "snr_report.html"

    xyz_str = (
        f"X={stage_xyz_um[0]:.1f} µm, Y={stage_xyz_um[1]:.1f} µm, Z={stage_xyz_um[2]:.1f} µm"
        if stage_xyz_um is not None
        else "Stationary"
    )
    laser_badges = " ".join(
        f'<span class="badge badge-laser">{html.escape(lid)}: {info.get("wavelength", "")}nm @ {info.get("power_setpoint_mw", info.get("power_mw", "?"))} mW</span>'
        for lid, info in laser_summary.items()
    )

    cam_sections: list[str] = []
    for cam_id, cam in cameras.items():
        half_life_str = (
            f"{cam.bleach_half_life_frames:.0f} frames ({cam.bleach_half_life_exposure_s:.2f} s)"
            if cam.bleach_half_life_frames is not None and cam.bleach_half_life_exposure_s is not None
            else "Negligible bleaching"
        )
        bin_desc = f"HW {cam.hardware_binning}×{cam.hardware_binning} | GPU {cam.gpu_bin_y}×{cam.gpu_bin_x} (filter: {cam.gpu_pre_filter})"
        gain_str = f"{cam.effective_gain_adu_per_e:.3f} ADU/e⁻" if cam.effective_gain_adu_per_e is not None else "N/A"

        bleach_svg = _svg_line_chart(
            cam.bleach_frame_indices,
            [
                ("Foreground Mean (ADU)", cam.bleach_fg_mean_trace, "#38bdf8", False),
                ("Exp Decay Fit", cam.bleach_fit_trace, "#f43f5e", True),
                ("Background Mean", cam.bleach_bg_mean_trace, "#a1a1aa", True),
            ],
            x_label="Frame Index (Static Sample)",
            y_label="Dark-Subtracted Intensity (ADU)",
        )

        ptc_x = [p.mean_adu for p in cam.ptc_points]
        ptc_var_det = [p.var_detrended for p in cam.ptc_points]
        ptc_var_raw = [p.var_raw for p in cam.ptc_points]
        ptc_svg = _svg_line_chart(
            ptc_x,
            [
                ("Detrended Var (Shot+Read)", ptc_var_det, "#10b981", False),
                ("Raw Var (incl. Bleaching)", ptc_var_raw, "#f59e0b", True),
            ],
            x_label="Mean Signal Above Dark (ADU)",
            y_label="Temporal Variance (ADU²)",
        )

        hist_svg = _svg_bar_histogram(
            cam.hist_bin_centers_adu,
            cam.hist_counts,
            color="#38bdf8",
            x_label="Pixel Intensity (ADU)",
        )
        snr_hist_svg = _svg_bar_histogram(
            cam.snr_hist_bin_centers,
            cam.snr_hist_counts,
            color="#a855f7",
            x_label="Foreground Bleaching-Corrected SNR",
        )

        cam_sections.append(f"""
        <section class="cam-section">
          <div class="cam-header">
            <div>
              <h2>Camera: <code>{html.escape(cam_id)}</code> <span class="sub">Channel: {html.escape(cam.channel_id)}</span></h2>
              <p class="meta-line">
                Frame: <strong>{cam.width} × {cam.height} px</strong> ({html.escape(cam.dtype)}, {cam.valid_bits}-bit) &bull;
                Sampling: <strong>{cam.pixel_size_um[0]:.3f} × {cam.pixel_size_um[1]:.3f} µm/px</strong> &bull;
                Exposure: <strong>{cam.exposure_time_ms:.2f} ms</strong> &bull;
                Binning: <strong>{html.escape(bin_desc)}</strong>
              </p>
            </div>
            <div class="kpi-strip">
              <div class="kpi-card highlight">
                <span class="kpi-label">Bleach-Corrected SNR (FG Mean / P95)</span>
                <span class="kpi-val">{cam.snr_fg_mean_corrected:.1f} / {cam.snr_fg_p95_corrected:.1f}</span>
                <span class="kpi-sub">{cam.snr_fg_mean_corrected_db:.1f} dB (Raw: {cam.snr_fg_mean_raw:.1f})</span>
              </div>
              <div class="kpi-card">
                <span class="kpi-label">Photobleaching ({cam.num_signal_frames} frames)</span>
                <span class="kpi-val">{cam.bleach_total_pct:+.2f}%</span>
                <span class="kpi-sub">{cam.bleach_rate_per_frame_pct:.3f}%/frame &bull; t½: {html.escape(half_life_str)}</span>
              </div>
              <div class="kpi-card">
                <span class="kpi-label">Contrast (CNR / SBR)</span>
                <span class="kpi-val">{cam.cnr:.1f} / {cam.sbr:.2f}×</span>
                <span class="kpi-sub">Weber: {cam.weber_contrast:.2f} &bull; Michelson: {cam.michelson_contrast:.3f}</span>
              </div>
              <div class="kpi-card">
                <span class="kpi-label">Signal P95 / Well Fill</span>
                <span class="kpi-val">{cam.signal_p95_fg_adu:.0f} ADU</span>
                <span class="kpi-sub">Well Fill P99: {cam.well_fill_p99_pct:.1f}% &bull; Sat: {cam.saturated_pixel_pct:.3f}%</span>
              </div>
              <div class="kpi-card">
                <span class="kpi-label">Sharpness (Tenengrad)</span>
                <span class="kpi-val">{cam.tenengrad_mean_fg:.1f}</span>
                <span class="kpi-sub">In-Focus Mask: {cam.sharp_in_focus_area_pct:.1f}% of FOV</span>
              </div>
            </div>
          </div>

          <div class="map-grid">
            <div class="map-card">
              <div class="map-title">Mean Projection µ(y, x) <span class="tag">Viridis</span></div>
              <img src="{cam.preview_mean_png}" alt="Mean Projection" />
              <div class="map-caption">FG Mean: {cam.signal_mean_fg_adu:.1f} ADU &bull; BG Mean: {cam.background_mean_adu:.1f} ADU &bull; Max: {cam.signal_max_adu:.0f} ADU</div>
            </div>
            <div class="map-card">
              <div class="map-title">Bleaching-Detrended Noise σ_detrended(y, x) <span class="tag">Magma</span></div>
              <img src="{cam.preview_std_detrended_png}" alt="Detrended Temporal Std" />
              <div class="map-caption">Read Floor: {cam.read_noise_adu:.2f} ADU &bull; BG Temporal σ: {cam.background_noise_temporal_adu:.2f} ADU</div>
            </div>
            <div class="map-card">
              <div class="map-title">Bleaching-Corrected Temporal SNR Map <span class="tag">Inferno</span></div>
              <img src="{cam.preview_snr_corrected_png}" alt="Bleach-Corrected SNR Map" />
              <div class="map-caption">FG Median SNR: {cam.snr_fg_median_corrected:.1f} &bull; P95 SNR: {cam.snr_fg_p95_corrected:.1f} ({cam.snr_fg_mean_corrected_db:.1f} dB)</div>
            </div>
            <div class="map-card">
              <div class="map-title">Spatial Photobleaching Map (% Loss) <span class="tag">Plasma</span></div>
              <img src="{cam.preview_bleach_map_png}" alt="Photobleaching Map" />
              <div class="map-caption">Initial FG: {cam.bleach_initial_fg_adu:.1f} ADU &rarr; Final FG: {cam.bleach_final_fg_adu:.1f} ADU (R²={cam.bleach_fit_r2:.3f})</div>
            </div>
            <div class="map-card">
              <div class="map-title">Signal, Background &amp; In-Focus Sharpness Masks <span class="tag">Overlay</span></div>
              <img src="{cam.preview_masks_png}" alt="Segmentation Masks" />
              <div class="map-caption"><span style="color:#4ade80">&#9632;</span> Sharp Focus ({cam.sharp_in_focus_area_pct:.1f}%) &bull; <span style="color:#38bdf8">&#9632;</span> Foreground ({cam.foreground_area_pct:.1f}%) &bull; <span style="color:#94a3b8">&#9632;</span> Background (Otsu={cam.otsu_threshold_adu:.0f})</div>
            </div>
            <div class="map-card">
              <div class="map-title">Local Normalized Sharpness Map <span class="tag">Turbo</span></div>
              <img src="{cam.preview_sharpness_png}" alt="Sharpness Map" />
              <div class="map-caption">Tenengrad P95: {cam.tenengrad_p95_fg:.1f} &bull; Laplacian Var: {cam.laplacian_variance_fg:.1f} &bull; Brenner: {cam.brenner_score_fg:.1f}</div>
            </div>
          </div>

          <div class="charts-grid">
            <div class="chart-card">
              <div class="chart-title">Stationary Photobleaching Decay Curve (N={cam.num_signal_frames} frames)</div>
              {bleach_svg}
            </div>
            <div class="chart-card">
              <div class="chart-title">Photon Transfer Curve (Detrended vs. Raw Variance) &bull; Est. Gain: {html.escape(gain_str)}</div>
              {ptc_svg}
            </div>
            <div class="chart-card">
              <div class="chart-title">Mean Projection Intensity Distribution</div>
              {hist_svg}
            </div>
            <div class="chart-card">
              <div class="chart-title">Foreground Bleaching-Corrected Temporal SNR Distribution</div>
              {snr_hist_svg}
            </div>
          </div>

          <div class="tables-grid">
            <table class="metrics-table">
              <thead><tr><th colspan="2">Signal, Noise &amp; Photobleaching Metrics</th></tr></thead>
              <tbody>
                <tr><td>Signal Mean / Median (Foreground)</td><td>{cam.signal_mean_fg_adu:.2f} / {cam.signal_median_fg_adu:.2f} ADU</td></tr>
                <tr><td>Signal P95 / P99 / Max</td><td>{cam.signal_p95_fg_adu:.1f} / {cam.signal_p99_fg_adu:.1f} / {cam.signal_max_adu:.1f} ADU</td></tr>
                <tr><td>Dark Offset / Read Noise Floor</td><td>{cam.dark_offset_adu:.2f} ADU / {cam.read_noise_adu:.3f} ADU ({cam.num_dark_frames} dark frames)</td></tr>
                <tr><td>Bleach-Corrected Temporal SNR (Mean / Median / P95)</td><td><strong>{cam.snr_fg_mean_corrected:.2f}</strong> / {cam.snr_fg_median_corrected:.2f} / {cam.snr_fg_p95_corrected:.2f} ({cam.snr_fg_mean_corrected_db:.2f} dB)</td></tr>
                <tr><td>Uncorrected Raw Temporal SNR (Mean / Median / P95)</td><td>{cam.snr_fg_mean_raw:.2f} / {cam.snr_fg_median_raw:.2f} / {cam.snr_fg_p95_raw:.2f} ({cam.snr_fg_mean_raw_db:.2f} dB)</td></tr>
                <tr><td>Photobleaching Total Loss / Rate per Frame</td><td>{cam.bleach_total_pct:+.3f}% ({cam.bleach_rate_per_frame_pct:.4f}% / frame, R²={cam.bleach_fit_r2:.3f})</td></tr>
                <tr><td>Photobleaching Half-Life (t½)</td><td>{html.escape(half_life_str)}</td></tr>
              </tbody>
            </table>
            <table class="metrics-table">
              <thead><tr><th colspan="2">Contrast, Dynamic Range &amp; Sharpness Metrics</th></tr></thead>
              <tbody>
                <tr><td>Contrast-to-Noise Ratio (CNR)</td><td><strong>{cam.cnr:.2f}</strong></td></tr>
                <tr><td>Signal-to-Background Ratio (SBR) / Weber Contrast</td><td>{cam.sbr:.2f}× / {cam.weber_contrast:.2f}</td></tr>
                <tr><td>Michelson / RMS Contrast</td><td>{cam.michelson_contrast:.4f} / {cam.rms_contrast:.4f}</td></tr>
                <tr><td>Dynamic Range / Sensor Well Fill (P99)</td><td>{cam.dynamic_range_db:.2f} dB / {cam.well_fill_p99_pct:.2f}% (Saturated: {cam.saturated_pixel_pct:.4f}%)</td></tr>
                <tr><td>Tenengrad Sharpness (FG Mean / P95)</td><td>{cam.tenengrad_mean_fg:.2f} / {cam.tenengrad_p95_fg:.2f}</td></tr>
                <tr><td>Laplacian Variance / Brenner Focus Score</td><td>{cam.laplacian_variance_fg:.2f} / {cam.brenner_score_fg:.2f}</td></tr>
                <tr><td>Saved TIFF Projections</td><td><code>{html.escape(str(output_dir))}</code></td></tr>
              </tbody>
            </table>
          </div>
        </section>
        """)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voxel SNR &amp; Photobleaching Report — {html.escape(report_id)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #09090b;
      --card: #121216;
      --card-alt: #18181d;
      --border: #27272a;
      --text: #f4f4f5;
      --muted: #a1a1aa;
      --accent: #38bdf8;
      --emerald: #10b981;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px 32px 48px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
      font-size: 13px;
      line-height: 1.45;
    }}
    header.report-header {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding-bottom: 18px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{ margin: 0 0 6px; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; }}
    h2 {{ margin: 0 0 4px; font-size: 16px; font-weight: 600; }}
    h2 .sub {{ font-weight: 400; color: var(--muted); font-size: 13px; margin-left: 8px; }}
    .meta-line {{ margin: 0; color: var(--muted); font-size: 12px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 500;
      background: var(--card-alt);
      border: 1px solid var(--border);
      color: var(--text);
    }}
    .badge-laser {{ border-color: #0284c7; background: rgba(2, 132, 199, 0.15); color: #7dd3fc; }}
    .cam-section {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 28px;
    }}
    .cam-header {{
      display: flex;
      flex-direction: column;
      gap: 14px;
      margin-bottom: 18px;
    }}
    .kpi-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
      gap: 10px;
    }}
    .kpi-card {{
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .kpi-card.highlight {{ border-color: rgba(56, 189, 248, 0.45); background: rgba(56, 189, 248, 0.06); }}
    .kpi-label {{ font-size: 11px; color: var(--muted); }}
    .kpi-val {{ font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .kpi-sub {{ font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }}
    .map-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .map-card, .chart-card {{
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .map-title, .chart-title {{
      font-size: 12px;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .tag {{
      font-size: 10px;
      font-weight: 500;
      padding: 1px 7px;
      border-radius: 4px;
      background: #27272a;
      color: #d4d4d8;
    }}
    .map-card img {{
      width: 100%;
      height: auto;
      max-height: 300px;
      object-fit: contain;
      background: #050507;
      border-radius: 4px;
      border: 1px solid #1f1f23;
    }}
    .map-caption {{ font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .chart-svg {{ width: 100%; height: auto; display: block; }}
    .tables-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 14px;
    }}
    .metrics-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      font-variant-numeric: tabular-nums;
    }}
    .metrics-table th {{
      text-align: left;
      padding: 8px 12px;
      background: #1f1f26;
      font-size: 12px;
      border-bottom: 1px solid var(--border);
    }}
    .metrics-table td {{
      padding: 6px 12px;
      border-bottom: 1px solid #222228;
      font-size: 12px;
    }}
    .metrics-table td:first-child {{ color: var(--muted); }}
    .metrics-table td:last-child {{ text-align: right; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; color: #e4e4e7; }}
  </style>
</head>
<body>
  <header class="report-header">
    <div>
      <h1>Stationary SNR, Photobleaching &amp; Image Quality Diagnostics</h1>
      <p class="meta-line">
        Instrument: <strong>{html.escape(instrument_name)}</strong> &bull;
        Profile: <code>{html.escape(profile_id)}</code> &bull;
        Position: <strong>{html.escape(xyz_str)}</strong> &bull;
        Captured: <strong>{html.escape(timestamp.strftime("%Y-%m-%d %H:%M:%S"))}</strong>
      </p>
    </div>
    <div class="badges">
      <span class="badge">Static Sample Mode</span>
      {laser_badges}
    </div>
  </header>
  {"".join(cam_sections)}
</body>
</html>
"""
    report_path.write_text(html_doc, encoding="utf-8")

    # Also write JSON metrics alongside HTML report (excluding large base64 PNG strings for clean machine readability)
    metrics_payload = {
        "report_id": report_id,
        "timestamp": timestamp.isoformat(),
        "instrument_name": instrument_name,
        "profile_id": profile_id,
        "stage_xyz_um": list(stage_xyz_um) if stage_xyz_um is not None else None,
        "lasers": laser_summary,
        "cameras": {
            cid: cam.model_dump(
                exclude={
                    "preview_mean_png",
                    "preview_std_detrended_png",
                    "preview_snr_corrected_png",
                    "preview_bleach_map_png",
                    "preview_masks_png",
                    "preview_sharpness_png",
                }
            )
            for cid, cam in cameras.items()
        },
    }
    (output_dir / "snr_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return report_path
