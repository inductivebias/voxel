"""Stationary multi-frame SNR, photobleaching, contrast, and sharpness diagnostics + HTML report."""

from __future__ import annotations

import base64
import html
import json
import math
from dataclasses import dataclass
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

    # Temporal SNR (bleaching-corrected vs raw)
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

    # Histograms, PTC, saved TIFFs & embedded HTML previews
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


@dataclass(slots=True)
class _BleachMetrics:
    initial_fg: float
    final_fg: float
    total_pct: float
    rate_per_frame_pct: float
    half_life_frames: float | None
    half_life_s: float | None
    fit_r2: float
    fg_mean_trace: list[float]
    fg_p95_trace: list[float]
    bg_mean_trace: list[float]
    fit_trace: list[float]
    map_pct: np.ndarray


@dataclass(slots=True)
class _SharpnessMetrics:
    tenengrad_mean: float
    tenengrad_p95: float
    laplacian_var: float
    brenner_score: float
    in_focus_pct: float
    sharpness_map: np.ndarray
    sharp_mask: np.ndarray


def _encode_png_data_url(bgr_or_gray: np.ndarray, *, max_width: int = 720) -> str:
    h, w = bgr_or_gray.shape[:2]
    if w > max_width > 0:
        bgr_or_gray = cv2.resize(
            bgr_or_gray, (max_width, max(1, round(h * (max_width / w)))), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".png", bgr_or_gray)
    if not ok:
        return ""
    return f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('ascii')}"


def _render_heatmap_png(
    arr: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> str:
    clean = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(clean, 1.0)) if vmin is None else float(vmin)
    hi = float(np.percentile(clean, 99.5)) if vmax is None else float(vmax)
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    u8 = (np.clip((clean - lo) / (hi - lo), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return _encode_png_data_url(cv2.applyColorMap(u8, colormap))


def _render_masks_overlay_png(
    mean_img: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    sharp_mask: np.ndarray,
) -> str:
    clean = np.nan_to_num(mean_img.astype(np.float32), nan=0.0)
    lo, hi = float(np.percentile(clean, 1.0)), float(np.percentile(clean, 99.5))
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    bgr = cv2.cvtColor((np.clip((clean - lo) / (hi - lo), 0.0, 1.0) * 200.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for mask, scale, tint in (
        (bg_mask, 0.6, (75.0, 45.0, 30.0)),
        (fg_mask & ~sharp_mask, 0.65, (180.0, 140.0, 20.0)),
        (sharp_mask, 0.6, (90.0, 220.0, 40.0)),
    ):
        bgr[mask] = np.clip(bgr[mask] * scale + np.array(tint, dtype=np.float32), 0, 255).astype(np.uint8)
    return _encode_png_data_url(bgr)


def _dark_stats(
    mean_proj: np.ndarray,
    std_detrended: np.ndarray,
    dark_frames: np.ndarray | None,
) -> tuple[int, float, float, np.ndarray]:
    if dark_frames is not None and np.asarray(dark_frames).ndim == 3 and np.asarray(dark_frames).shape[0] >= 1:
        d_stack = np.asarray(dark_frames, dtype=np.float32)
        n_dark = int(d_stack.shape[0])
        dark_mean = np.mean(d_stack, axis=0, dtype=np.float32)
        if n_dark >= 2:
            d_diffs = np.diff(d_stack, axis=0)
            read_map = np.sqrt(np.mean(d_diffs * d_diffs, axis=0, dtype=np.float32) * 0.5)
        else:
            read_map = np.std(d_stack, axis=0, dtype=np.float32)
        return n_dark, float(np.median(dark_mean)), float(np.median(read_map)), np.maximum(0.0, mean_proj - dark_mean)
    p1, p5 = np.percentile(mean_proj, [1.0, 5.0])
    bg_floor = mean_proj <= p5
    read_noise = float(np.median(std_detrended[bg_floor])) if np.any(bg_floor) else 1.0
    return 0, float(p1), read_noise, np.maximum(0.0, mean_proj - float(p1))


def _segment_masks(
    mean_proj: np.ndarray,
    dark_offset: float,
    read_noise: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    lo_p, hi_p = float(np.percentile(mean_proj, 0.5)), float(np.percentile(mean_proj, 99.8))
    span = max(1.0, hi_p - lo_p)
    norm_u8 = np.clip((mean_proj - lo_p) * (255.0 / span), 0.0, 255.0).astype(np.uint8)
    prob = np.bincount(norm_u8.ravel(), minlength=256).astype(np.float64) / max(1.0, float(norm_u8.size))
    omega = np.cumsum(prob)
    mu_cum = np.cumsum(prob * np.arange(256, dtype=np.float64))
    denom = omega * (1.0 - omega)
    sigma_b2 = np.divide((mu_cum[-1] * omega - mu_cum) ** 2, denom, out=np.zeros_like(denom), where=denom > 1e-12)
    max_sb2 = float(np.max(sigma_b2))
    otsu_u8 = float(np.mean(np.flatnonzero(sigma_b2 >= max_sb2 * 0.999))) if max_sb2 > 0.0 else 128.0
    otsu_thresh = max(float(lo_p + (otsu_u8 / 255.0) * span), float(dark_offset + 4.0 * read_noise))

    fg_mask = mean_proj > otsu_thresh
    if not np.any(fg_mask):
        fg_mask = mean_proj >= float(np.median(mean_proj))
    p25 = float(np.percentile(mean_proj, 25.0))
    bg_mask = (mean_proj <= min(otsu_thresh, p25)) & ~fg_mask
    if not np.any(bg_mask):
        bg_mask = ~fg_mask
    if not np.any(bg_mask):
        bg_mask = mean_proj <= float(np.median(mean_proj))
    return otsu_thresh, fg_mask, bg_mask


def _compute_bleaching(
    stack_f32: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    dark_offset: float,
    exposure_ms: float,
) -> _BleachMetrics:
    n_frames = int(stack_f32.shape[0])
    fg_vals = np.maximum(0.0, stack_f32[:, fg_mask] - dark_offset)
    fg_mean = [float(v) for v in np.mean(fg_vals, axis=1)]
    fg_p95 = [float(v) for v in np.percentile(fg_vals, 95.0, axis=1)]
    bg_mean = [float(v) for v in np.mean(stack_f32[:, bg_mask], axis=1)]

    t_arr = np.arange(n_frames, dtype=np.float64)
    y_arr = np.maximum(np.asarray(fg_mean, dtype=np.float64), 1e-3)
    slope, intercept = np.polyfit(t_arr, np.log(y_arr), 1)
    k_decay = max(0.0, float(-slope))
    i0_fit = float(math.exp(intercept))
    fit_trace = [float(i0_fit * math.exp(-k_decay * t)) for t in range(n_frames)]

    ss_res = float(np.sum((y_arr - np.asarray(fit_trace)) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    fit_r2 = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0
    initial_fg, final_fg = fg_mean[0], fg_mean[-1]
    total_pct = (initial_fg - final_fg) / max(initial_fg, 1e-3) * 100.0 if initial_fg > 0 else 0.0
    rate_pct = (1.0 - math.exp(-k_decay)) * 100.0
    half_frames = round(math.log(2.0) / k_decay, 2) if k_decay > 1e-6 else None
    half_s = round(half_frames * (exposure_ms / 1000.0), 3) if half_frames is not None else None

    q = max(1, n_frames // 4)
    first_q = np.mean(stack_f32[:q], axis=0, dtype=np.float32) - dark_offset
    last_q = np.mean(stack_f32[-q:], axis=0, dtype=np.float32) - dark_offset
    bleach_map = np.where(
        fg_mask & (first_q > 1.0),
        np.clip((first_q - last_q) / np.maximum(first_q, 1.0) * 100.0, -25.0, 100.0),
        0.0,
    ).astype(np.float32)
    return _BleachMetrics(
        initial_fg=initial_fg,
        final_fg=final_fg,
        total_pct=total_pct,
        rate_per_frame_pct=rate_pct,
        half_life_frames=half_frames,
        half_life_s=half_s,
        fit_r2=fit_r2,
        fg_mean_trace=fg_mean,
        fg_p95_trace=fg_p95,
        bg_mean_trace=bg_mean,
        fit_trace=fit_trace,
        map_pct=bleach_map,
    )


def _compute_sharpness(
    mean_proj: np.ndarray,
    sig_mean: np.ndarray,
    fg_mask: np.ndarray,
    bg_noise: float,
) -> _SharpnessMetrics:
    gx = cv2.Sobel(mean_proj, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mean_proj, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    lap = cv2.Laplacian(mean_proj, cv2.CV_32F, ksize=3)

    height, width = mean_proj.shape
    brenner_map = np.zeros_like(mean_proj, dtype=np.float32)
    if width > 2 and height > 2:
        dx2 = mean_proj[:, 2:] - mean_proj[:, :-2]
        dy2 = mean_proj[2:, :] - mean_proj[:-2, :]
        brenner_map[:, :-2] += dx2 * dx2
        brenner_map[:-2, :] += dy2 * dy2

    local_grad = cv2.blur(grad_mag, (15, 15))
    local_mean = np.maximum(cv2.blur(sig_mean, (15, 15)), max(10.0, bg_noise * 3.0))
    sharpness_map = local_grad / local_mean
    fg_sharp = sharpness_map[fg_mask]
    sharp_mask = fg_mask & (sharpness_map >= (float(np.percentile(fg_sharp, 55.0)) if fg_sharp.size > 0 else 0.0))
    return _SharpnessMetrics(
        tenengrad_mean=float(np.mean(grad_mag[fg_mask])),
        tenengrad_p95=float(np.percentile(grad_mag[fg_mask], 95.0)),
        laplacian_var=float(np.var(lap[fg_mask])),
        brenner_score=float(np.mean(brenner_map[fg_mask])),
        in_focus_pct=float(np.mean(sharp_mask) * 100.0),
        sharpness_map=sharpness_map,
        sharp_mask=sharp_mask,
    )


def _compute_ptc(
    mean_proj: np.ndarray,
    sig_mean: np.ndarray,
    std_detrended: np.ndarray,
    std_raw: np.ndarray,
    max_code: float,
) -> tuple[list[PtcPoint], float | None]:
    valid_px = (sig_mean > 1.0) & (mean_proj < 0.98 * max_code)
    if np.count_nonzero(valid_px) < 128:
        return [], None
    s_vals = sig_mean[valid_px]
    vd_vals = (std_detrended * std_detrended)[valid_px]
    vr_vals = (std_raw * std_raw)[valid_px]
    edges = np.unique(np.percentile(s_vals, np.linspace(2.0, 98.0, 21)))
    points: list[PtcPoint] = []
    for idx in range(len(edges) - 1):
        lo_e, hi_e = float(edges[idx]), float(edges[idx + 1])
        in_bin = (s_vals >= lo_e) & (s_vals <= hi_e if idx == len(edges) - 2 else s_vals < hi_e)
        cnt = int(np.count_nonzero(in_bin))
        if cnt < 16:
            continue
        m_adu = float(np.median(s_vals[in_bin]))
        v_det = float(np.median(vd_vals[in_bin]))
        v_raw = float(np.median(vr_vals[in_bin]))
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


def _write_projections(
    out_root: Path,
    camera_id: str,
    stack_dtype: np.dtype[Any],
    pixel_size_um: tuple[float, float],
    meta_common: dict[str, Any],
    arrays: dict[str, tuple[str, np.ndarray, str]],
) -> dict[str, str]:
    out_root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for key, (suffix, arr, proj_name) in arrays.items():
        data = arr.astype(stack_dtype if stack_dtype in (np.uint8, np.uint16) else np.float32) if key == "mean" else arr
        path = write_tiff_2d(
            out_root / f"{camera_id}_{suffix}.tif",
            data,
            pixel_size_um=pixel_size_um,
            metadata={**meta_common, "projection": proj_name},
        )
        saved[key] = str(path)
    return saved


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
    """Compute stationary multi-frame SNR, photobleaching, contrast, and sharpness diagnostics.

    Uses both exponential decay fitting and an Allan-deviation (successive-difference) temporal
    noise estimator to isolate shot + read noise from monotonic photobleaching drift.
    """
    stack = np.asarray(frames)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError(f"Expected 3D frame stack with N >= 2 frames, got shape {stack.shape}")

    n_frames, height, width = int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2])
    stack_f32 = stack.astype(np.float32)
    mean_proj = np.mean(stack_f32, axis=0, dtype=np.float32)
    std_raw = np.std(stack_f32, axis=0, ddof=1, dtype=np.float32)
    diffs = np.diff(stack_f32, axis=0)
    std_detrended = np.sqrt(np.mean(diffs * diffs, axis=0, dtype=np.float32) * 0.5)

    n_dark, dark_offset, read_noise, sig_mean = _dark_stats(mean_proj, std_detrended, dark_frames)
    otsu_thresh, fg_mask, bg_mask = _segment_masks(mean_proj, dark_offset, read_noise)

    snr_corr_map = sig_mean / np.maximum(std_detrended, 1e-3)
    snr_raw_map = sig_mean / np.maximum(std_raw, 1e-3)
    fg_snr_corr, fg_snr_raw = snr_corr_map[fg_mask], snr_raw_map[fg_mask]
    snr_mean_c, snr_mean_r = float(np.mean(fg_snr_corr)), float(np.mean(fg_snr_raw))

    bleach = _compute_bleaching(stack_f32, fg_mask, bg_mask, dark_offset, exposure_time_ms)

    fg_px, bg_px = mean_proj[fg_mask], mean_proj[bg_mask]
    sig_mean_fg, sig_p99_fg = float(np.mean(fg_px)), float(np.percentile(fg_px, 99.0))
    bg_mean = float(np.mean(bg_px))
    bg_temp_noise = float(np.median(std_detrended[bg_mask]))
    max_code = float((1 << valid_bits) - 1)
    fg_temp_var = float(np.median(std_detrended[fg_mask]) ** 2)
    cnr = max(0.0, sig_mean_fg - bg_mean) / math.sqrt(max(fg_temp_var + bg_temp_noise**2, 1e-6))
    p99_all, p1_all = float(np.percentile(mean_proj, 99.0)), float(np.percentile(mean_proj, 1.0))

    sharp = _compute_sharpness(mean_proj, sig_mean, fg_mask, bg_temp_noise)
    ptc_points, effective_gain = _compute_ptc(mean_proj, sig_mean, std_detrended, std_raw, max_code)

    hist_counts, hist_edges = np.histogram(
        mean_proj, bins=48, range=(float(np.min(mean_proj)), float(np.max(mean_proj)) + 1.0)
    )
    snr_counts, snr_edges = np.histogram(
        fg_snr_corr, bins=40, range=(0.0, max(1.0, float(np.percentile(fg_snr_corr, 99.5))))
    )

    saved_tiffs: dict[str, str] = {}
    if output_dir is not None:
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
        saved_tiffs = _write_projections(
            Path(output_dir),
            camera_id,
            stack.dtype,
            pixel_size_um,
            meta_common,
            {
                "mean": ("mean", mean_proj, "mean"),
                "std_detrended": ("std_detrended", std_detrended, "std_bleaching_detrended"),
                "snr_corrected": ("snr_bleach_corrected", snr_corr_map, "snr_bleaching_corrected"),
                "bleach_pct_map": ("bleach_pct_map", bleach.map_pct, "photobleaching_pct_map"),
            },
        )

    bleach_vmax = max(5.0, float(np.percentile(bleach.map_pct[fg_mask], 98.0)) if np.any(fg_mask) else 10.0)
    previews = {
        "mean": _render_heatmap_png(mean_proj, colormap=cv2.COLORMAP_VIRIDIS),
        "std_detrended": _render_heatmap_png(std_detrended, colormap=cv2.COLORMAP_MAGMA),
        "snr_corrected": _render_heatmap_png(snr_corr_map, vmin=0.0, colormap=cv2.COLORMAP_INFERNO),
        "bleach_map": _render_heatmap_png(bleach.map_pct, vmin=0.0, vmax=bleach_vmax, colormap=cv2.COLORMAP_PLASMA),
        "masks": _render_masks_overlay_png(mean_proj, fg_mask, bg_mask, sharp.sharp_mask),
        "sharpness": _render_heatmap_png(sharp.sharpness_map, vmin=0.0, colormap=cv2.COLORMAP_TURBO),
    }

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
        dark_offset_adu=round(dark_offset, 2),
        read_noise_adu=round(read_noise, 3),
        background_mean_adu=round(bg_mean, 2),
        background_std_spatial_adu=round(float(np.std(bg_px)), 3),
        background_noise_temporal_adu=round(bg_temp_noise, 3),
        signal_mean_fg_adu=round(sig_mean_fg, 2),
        signal_median_fg_adu=round(float(np.median(fg_px)), 2),
        signal_p95_fg_adu=round(float(np.percentile(fg_px, 95.0)), 2),
        signal_p99_fg_adu=round(sig_p99_fg, 2),
        signal_max_adu=round(float(np.max(mean_proj)), 2),
        otsu_threshold_adu=round(otsu_thresh, 2),
        foreground_area_pct=round(float(np.mean(fg_mask) * 100.0), 2),
        saturated_pixel_pct=round(float(np.mean(mean_proj >= 0.995 * max_code) * 100.0), 4),
        well_fill_p99_pct=round(min(100.0, (sig_p99_fg / max(1.0, max_code)) * 100.0), 2),
        dynamic_range_db=round(
            20.0 * math.log10(max(max(0.0, sig_p99_fg - dark_offset) / max(read_noise, 1e-3), 1.0)), 2
        ),
        effective_gain_adu_per_e=effective_gain,
        snr_fg_mean_corrected=round(snr_mean_c, 2),
        snr_fg_median_corrected=round(float(np.median(fg_snr_corr)), 2),
        snr_fg_p95_corrected=round(float(np.percentile(fg_snr_corr, 95.0)), 2),
        snr_fg_mean_corrected_db=round(20.0 * math.log10(max(snr_mean_c, 1e-6)), 2),
        snr_fg_mean_raw=round(snr_mean_r, 2),
        snr_fg_median_raw=round(float(np.median(fg_snr_raw)), 2),
        snr_fg_p95_raw=round(float(np.percentile(fg_snr_raw, 95.0)), 2),
        snr_fg_mean_raw_db=round(20.0 * math.log10(max(snr_mean_r, 1e-6)), 2),
        bleach_initial_fg_adu=round(bleach.initial_fg, 2),
        bleach_final_fg_adu=round(bleach.final_fg, 2),
        bleach_total_pct=round(bleach.total_pct, 3),
        bleach_rate_per_frame_pct=round(bleach.rate_per_frame_pct, 4),
        bleach_half_life_frames=bleach.half_life_frames,
        bleach_half_life_exposure_s=bleach.half_life_s,
        bleach_fit_r2=round(bleach.fit_r2, 4),
        bleach_frame_indices=list(range(1, n_frames + 1)),
        bleach_fg_mean_trace=[round(v, 2) for v in bleach.fg_mean_trace],
        bleach_fg_p95_trace=[round(v, 2) for v in bleach.fg_p95_trace],
        bleach_bg_mean_trace=[round(v, 2) for v in bleach.bg_mean_trace],
        bleach_fit_trace=[round(v, 2) for v in bleach.fit_trace],
        cnr=round(cnr, 2),
        sbr=round(sig_mean_fg / max(bg_mean, 1e-3), 2),
        weber_contrast=round((sig_mean_fg - bg_mean) / max(bg_mean, 1e-3), 2),
        michelson_contrast=round(max(0.0, p99_all - p1_all) / max(p99_all + p1_all, 1e-3), 4),
        rms_contrast=round(float(np.std(sig_mean[fg_mask]) / max(float(np.mean(sig_mean[fg_mask])), 1e-3)), 4),
        tenengrad_mean_fg=round(sharp.tenengrad_mean, 2),
        tenengrad_p95_fg=round(sharp.tenengrad_p95, 2),
        laplacian_variance_fg=round(sharp.laplacian_var, 2),
        brenner_score_fg=round(sharp.brenner_score, 2),
        sharp_in_focus_area_pct=round(sharp.in_focus_pct, 2),
        hist_bin_centers_adu=[
            round(float(0.5 * (hist_edges[i] + hist_edges[i + 1])), 1) for i in range(len(hist_counts))
        ],
        hist_counts=[int(c) for c in hist_counts],
        snr_hist_bin_centers=[
            round(float(0.5 * (snr_edges[i] + snr_edges[i + 1])), 2) for i in range(len(snr_counts))
        ],
        snr_hist_counts=[int(c) for c in snr_counts],
        ptc_points=ptc_points,
        saved_tiffs=saved_tiffs,
        previews=previews,
    )


def _svg_frame(
    width: int,
    height: int,
    pad: tuple[int, int, int, int],
    body: str,
    x_label: str,
    y_label: str,
) -> str:
    pad_l, pad_r, pad_t, pad_b = pad
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    ymid = pad_t + ph / 2
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg">'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + ph}" stroke="#52525b" />'
        f'<line x1="{pad_l}" y1="{pad_t + ph}" x2="{width - pad_r}" y2="{pad_t + ph}" stroke="#52525b" />'
        f"{body}"
        f'<text x="{pad_l + pw / 2:.1f}" y="{height - 3}" text-anchor="middle" fill="#a1a1aa"'
        f' font-size="11">{html.escape(x_label)}</text>'
        f'<text x="14" y="{ymid:.1f}" text-anchor="middle" fill="#a1a1aa" font-size="11"'
        f' transform="rotate(-90 14 {ymid:.1f})">{html.escape(y_label)}</text>'
        "</svg>"
    )


def _svg_line_chart(
    x_vals: Sequence[float | int],
    series: Sequence[tuple[str, Sequence[float], str, bool]],
    *,
    x_label: str,
    y_label: str,
    width: int = 540,
    height: int = 240,
) -> str:
    if not x_vals or not series or not series[0][1]:
        return '<div class="empty-chart">Insufficient data points</div>'
    all_y = [v for _, ys, _, _ in series for v in ys if math.isfinite(v)]
    if not all_y:
        return '<div class="empty-chart">No finite values</div>'

    pad_l, pad_r, pad_t, pad_b = 58, 20, 26, 40
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    xmin, xmax = float(min(x_vals)), float(max(x_vals))
    xmax = xmax if xmax > xmin else xmin + 1.0
    ymin, ymax = float(min(all_y)), float(max(all_y))
    ymax = ymax if ymax > ymin else ymin + 1.0
    pad_y = 0.08 * (ymax - ymin)
    ymin = max(0.0, ymin - pad_y) if ymin >= 0 else ymin - pad_y
    ymax += pad_y

    def sx(x: float) -> float:
        return pad_l + ((x - xmin) / (xmax - xmin)) * pw

    def sy(y: float) -> float:
        return pad_t + ph - ((y - ymin) / (ymax - ymin)) * ph

    parts: list[str] = []
    for i in range(5):
        gy = ymin + (ymax - ymin) * (i / 4.0)
        py = sy(gy)
        parts.append(
            f'<line x1="{pad_l}" y1="{py:.1f}" x2="{width - pad_r}" y2="{py:.1f}"'
            ' stroke="#27272a" stroke-dasharray="3,3" />'
            f'<text x="{pad_l - 6}" y="{py + 4:.1f}" text-anchor="end" fill="#a1a1aa" font-size="10">{gy:.1f}</text>'
        )
        gx = xmin + (xmax - xmin) * (i / 4.0)
        parts.append(
            f'<text x="{sx(gx):.1f}" y="{height - 18}" text-anchor="middle" fill="#a1a1aa" font-size="10">'
            f"{gx:.0f}</text>"
        )

    lx = float(pad_l + 6)
    for label, ys, color, dashed in series:
        pts = " ".join(f"{sx(float(x)):.1f},{sy(float(y)):.1f}" for x, y in zip(x_vals, ys, strict=False))
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} points="{pts}" />')
        parts.append(
            f'<line x1="{lx}" y1="13" x2="{lx + 16}" y2="13" stroke="{color}" stroke-width="2"{dash} />'
            f'<text x="{lx + 20}" y="17" fill="#d4d4d8" font-size="11">{html.escape(label)}</text>'
        )
        lx += 28 + len(label) * 6.5

    return _svg_frame(width, height, (pad_l, pad_r, pad_t, pad_b), "".join(parts), x_label, y_label)


def _svg_bar_histogram(
    centers: Sequence[float],
    counts: Sequence[int],
    *,
    color: str = "#38bdf8",
    x_label: str,
    y_label: str = "Pixels (log10)",
    width: int = 540,
    height: int = 220,
) -> str:
    if not centers or not counts:
        return '<div class="empty-chart">No histogram data</div>'
    pad_l, pad_r, pad_t, pad_b = 52, 16, 18, 38
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    log_counts = [math.log10(c + 1) for c in counts]
    ymax = max(1.0, *log_counts)
    n = len(centers)
    bar_w = max(2.0, pw / max(1, n) - 1.5)

    bars = [
        f'<rect x="{pad_l + (i / max(1, n)) * pw:.1f}" y="{pad_t + ph - (lc / ymax) * ph:.1f}"'
        f' width="{bar_w:.1f}" height="{(lc / ymax) * ph:.1f}" fill="{color}" opacity="0.82" rx="1" />'
        for i, lc in enumerate(log_counts)
    ]
    xmin, xmax = centers[0], centers[-1]
    ticks = [
        f'<text x="{pad_l + (k / 4.0) * pw:.1f}" y="{height - 17}" text-anchor="middle" fill="#a1a1aa"'
        f' font-size="10">{xmin + (xmax - xmin) * (k / 4.0):.1f}</text>'
        for k in range(5)
    ]
    return _svg_frame(width, height, (pad_l, pad_r, pad_t, pad_b), "".join(bars + ticks), x_label, y_label)


def _render_table(title: str, rows: Sequence[tuple[str, str]]) -> str:
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return (
        f'<table class="metrics-table"><thead><tr><th colspan="2">{title}</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def _render_camera_section(cam_id: str, cam: CameraSnrAnalysis, output_dir: Path) -> str:
    half_life = (
        f"{cam.bleach_half_life_frames:.0f} frames ({cam.bleach_half_life_exposure_s:.2f} s)"
        if cam.bleach_half_life_frames is not None and cam.bleach_half_life_exposure_s is not None
        else "Negligible bleaching"
    )
    bin_desc = (
        f"HW {cam.hardware_binning}x{cam.hardware_binning} | GPU {cam.gpu_bin_y}x{cam.gpu_bin_x} ({cam.gpu_pre_filter})"
    )
    gain_str = f"{cam.effective_gain_adu_per_e:.3f} ADU/e-" if cam.effective_gain_adu_per_e is not None else "N/A"

    kpis = [
        (
            "highlight",
            "Bleach-Corrected SNR (FG Mean / P95)",
            f"{cam.snr_fg_mean_corrected:.1f} / {cam.snr_fg_p95_corrected:.1f}",
            f"{cam.snr_fg_mean_corrected_db:.1f} dB (Raw: {cam.snr_fg_mean_raw:.1f})",
        ),
        (
            "",
            f"Photobleaching ({cam.num_signal_frames} frames)",
            f"{cam.bleach_total_pct:+.2f}%",
            f"{cam.bleach_rate_per_frame_pct:.3f}%/frame &bull; t1/2: {html.escape(half_life)}",
        ),
        (
            "",
            "Contrast (CNR / SBR)",
            f"{cam.cnr:.1f} / {cam.sbr:.2f}x",
            f"Weber: {cam.weber_contrast:.2f} &bull; Michelson: {cam.michelson_contrast:.3f}",
        ),
        (
            "",
            "Signal P95 / Well Fill",
            f"{cam.signal_p95_fg_adu:.0f} ADU",
            f"Well Fill P99: {cam.well_fill_p99_pct:.1f}% &bull; Sat: {cam.saturated_pixel_pct:.3f}%",
        ),
        (
            "",
            "Sharpness (Tenengrad)",
            f"{cam.tenengrad_mean_fg:.1f}",
            f"In-Focus Mask: {cam.sharp_in_focus_area_pct:.1f}% of FOV",
        ),
    ]
    kpi_html = "".join(
        f'<div class="kpi-card {cls}"><span class="kpi-label">{lbl}</span>'
        f'<span class="kpi-val">{val}</span><span class="kpi-sub">{sub}</span></div>'
        for cls, lbl, val, sub in kpis
    )

    maps = [
        (
            "Mean Projection &mu;(y, x)",
            "Viridis",
            cam.previews.get("mean", ""),
            (
                f"FG Mean: {cam.signal_mean_fg_adu:.1f} ADU &bull; BG Mean: {cam.background_mean_adu:.1f} ADU"
                f" &bull; Max: {cam.signal_max_adu:.0f} ADU"
            ),
        ),
        (
            "Bleaching-Detrended Noise &sigma;_detrended(y, x)",
            "Magma",
            cam.previews.get("std_detrended", ""),
            (
                f"Read Floor: {cam.read_noise_adu:.2f} ADU"
                f" &bull; BG Temporal &sigma;: {cam.background_noise_temporal_adu:.2f} ADU"
            ),
        ),
        (
            "Bleaching-Corrected Temporal SNR Map",
            "Inferno",
            cam.previews.get("snr_corrected", ""),
            (
                f"FG Median SNR: {cam.snr_fg_median_corrected:.1f} &bull; P95 SNR: {cam.snr_fg_p95_corrected:.1f}"
                f" ({cam.snr_fg_mean_corrected_db:.1f} dB)"
            ),
        ),
        (
            "Spatial Photobleaching Map (% Loss)",
            "Plasma",
            cam.previews.get("bleach_map", ""),
            (
                f"Initial FG: {cam.bleach_initial_fg_adu:.1f} ADU &rarr; Final FG: {cam.bleach_final_fg_adu:.1f} ADU"
                f" (R&sup2;={cam.bleach_fit_r2:.3f})"
            ),
        ),
        (
            "Signal, Background &amp; In-Focus Sharpness Masks",
            "Overlay",
            cam.previews.get("masks", ""),
            (
                f'<span style="color:#4ade80">&#9632;</span> Sharp Focus ({cam.sharp_in_focus_area_pct:.1f}%) &bull; '
                f'<span style="color:#38bdf8">&#9632;</span> Foreground ({cam.foreground_area_pct:.1f}%) &bull; '
                f'<span style="color:#94a3b8">&#9632;</span> Background (Otsu={cam.otsu_threshold_adu:.0f})'
            ),
        ),
        (
            "Local Normalized Sharpness Map",
            "Turbo",
            cam.previews.get("sharpness", ""),
            (
                f"Tenengrad P95: {cam.tenengrad_p95_fg:.1f} &bull; Laplacian Var: {cam.laplacian_variance_fg:.1f}"
                f" &bull; Brenner: {cam.brenner_score_fg:.1f}"
            ),
        ),
    ]
    map_html = "".join(
        f'<div class="map-card"><div class="map-title">{title} <span class="tag">{tag}</span></div>'
        f'<img src="{src}" alt="{html.escape(tag)}" /><div class="map-caption">{cap}</div></div>'
        for title, tag, src, cap in maps
    )

    charts = [
        (
            f"Stationary Photobleaching Decay Curve (N={cam.num_signal_frames} frames)",
            _svg_line_chart(
                cam.bleach_frame_indices,
                [
                    ("Foreground Mean (ADU)", cam.bleach_fg_mean_trace, "#38bdf8", False),
                    ("Exp Decay Fit", cam.bleach_fit_trace, "#f43f5e", True),
                    ("Background Mean", cam.bleach_bg_mean_trace, "#a1a1aa", True),
                ],
                x_label="Frame Index (Static Sample)",
                y_label="Dark-Subtracted Intensity (ADU)",
            ),
        ),
        (
            f"Photon Transfer Curve (Detrended vs. Raw Variance) &bull; Est. Gain: {html.escape(gain_str)}",
            _svg_line_chart(
                [p.mean_adu for p in cam.ptc_points],
                [
                    ("Detrended Var (Shot+Read)", [p.var_detrended for p in cam.ptc_points], "#10b981", False),
                    ("Raw Var (incl. Bleaching)", [p.var_raw for p in cam.ptc_points], "#f59e0b", True),
                ],
                x_label="Mean Signal Above Dark (ADU)",
                y_label="Temporal Variance (ADU^2)",
            ),
        ),
        (
            "Mean Projection Intensity Distribution",
            _svg_bar_histogram(
                cam.hist_bin_centers_adu,
                cam.hist_counts,
                color="#38bdf8",
                x_label="Pixel Intensity (ADU)",
            ),
        ),
        (
            "Foreground Bleaching-Corrected Temporal SNR Distribution",
            _svg_bar_histogram(
                cam.snr_hist_bin_centers,
                cam.snr_hist_counts,
                color="#a855f7",
                x_label="Foreground Bleaching-Corrected SNR",
            ),
        ),
    ]
    chart_html = "".join(f'<div class="chart-card"><div class="chart-title">{t}</div>{svg}</div>' for t, svg in charts)

    left_table = _render_table(
        "Signal, Noise &amp; Photobleaching Metrics",
        [
            ("Signal Mean / Median (Foreground)", f"{cam.signal_mean_fg_adu:.2f} / {cam.signal_median_fg_adu:.2f} ADU"),
            (
                "Signal P95 / P99 / Max",
                f"{cam.signal_p95_fg_adu:.1f} / {cam.signal_p99_fg_adu:.1f} / {cam.signal_max_adu:.1f} ADU",
            ),
            (
                "Dark Offset / Read Noise Floor",
                f"{cam.dark_offset_adu:.2f} ADU / {cam.read_noise_adu:.3f} ADU ({cam.num_dark_frames} dark)",
            ),
            (
                "Bleach-Corrected Temporal SNR (Mean / Median / P95)",
                (
                    f"<strong>{cam.snr_fg_mean_corrected:.2f}</strong> / {cam.snr_fg_median_corrected:.2f}"
                    f" / {cam.snr_fg_p95_corrected:.2f} ({cam.snr_fg_mean_corrected_db:.2f} dB)"
                ),
            ),
            (
                "Uncorrected Raw Temporal SNR (Mean / Median / P95)",
                (
                    f"{cam.snr_fg_mean_raw:.2f} / {cam.snr_fg_median_raw:.2f}"
                    f" / {cam.snr_fg_p95_raw:.2f} ({cam.snr_fg_mean_raw_db:.2f} dB)"
                ),
            ),
            (
                "Photobleaching Total Loss / Rate per Frame",
                (
                    f"{cam.bleach_total_pct:+.3f}%"
                    f" ({cam.bleach_rate_per_frame_pct:.4f}% / frame, R&sup2;={cam.bleach_fit_r2:.3f})"
                ),
            ),
            ("Photobleaching Half-Life (t1/2)", html.escape(half_life)),
        ],
    )
    right_table = _render_table(
        "Contrast, Dynamic Range &amp; Sharpness Metrics",
        [
            ("Contrast-to-Noise Ratio (CNR)", f"<strong>{cam.cnr:.2f}</strong>"),
            ("Signal-to-Background Ratio (SBR) / Weber Contrast", f"{cam.sbr:.2f}x / {cam.weber_contrast:.2f}"),
            ("Michelson / RMS Contrast", f"{cam.michelson_contrast:.4f} / {cam.rms_contrast:.4f}"),
            (
                "Dynamic Range / Sensor Well Fill (P99)",
                f"{cam.dynamic_range_db:.2f} dB / {cam.well_fill_p99_pct:.2f}% (Sat: {cam.saturated_pixel_pct:.4f}%)",
            ),
            ("Tenengrad Sharpness (FG Mean / P95)", f"{cam.tenengrad_mean_fg:.2f} / {cam.tenengrad_p95_fg:.2f}"),
            (
                "Laplacian Variance / Brenner Focus Score",
                f"{cam.laplacian_variance_fg:.2f} / {cam.brenner_score_fg:.2f}",
            ),
            ("Saved TIFF Projections", f"<code>{html.escape(str(output_dir))}</code>"),
        ],
    )

    return (
        f'<section class="cam-section"><div class="cam-header"><div>'
        f'<h2>Camera: <code>{html.escape(cam_id)}</code>'
        f' <span class="sub">Channel: {html.escape(cam.channel_id)}</span></h2>'
        f'<p class="meta-line">Frame: <strong>{cam.width} &times; {cam.height} px</strong>'
        f" ({html.escape(cam.dtype)}, {cam.valid_bits}-bit) &bull; "
        f"Sampling: <strong>{cam.pixel_size_um[0]:.3f} &times; {cam.pixel_size_um[1]:.3f} &mu;m/px</strong> &bull; "
        f"Exposure: <strong>{cam.exposure_time_ms:.2f} ms</strong> &bull; "
        f"Binning: <strong>{html.escape(bin_desc)}</strong></p></div>"
        f'<div class="kpi-strip">{kpi_html}</div></div>'
        f'<div class="map-grid">{map_html}</div>'
        f'<div class="charts-grid">{chart_html}</div>'
        f'<div class="tables-grid">{left_table}{right_table}</div></section>'
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
        if stage_xyz_um is not None
        else "Stationary"
    )
    laser_badges = " ".join(
        f'<span class="badge badge-laser">{html.escape(lid)}: {info.get("wavelength", "")}nm'
        f' @ {info.get("power_setpoint_mw", info.get("power_mw", "?"))} mW</span>'
        for lid, info in laser_summary.items()
    )
    html_doc = _REPORT_TEMPLATE.substitute(
        report_id=html.escape(report_id),
        instrument_name=html.escape(instrument_name),
        profile_id=html.escape(profile_id),
        xyz_str=html.escape(xyz_str),
        captured_str=html.escape(timestamp.strftime("%Y-%m-%d %H:%M:%S")),
        laser_badges=laser_badges,
        cam_sections="".join(_render_camera_section(cid, cam, output_dir) for cid, cam in cameras.items()),
    )
    report_path.write_text(html_doc, encoding="utf-8")

    metrics_payload = {
        "report_id": report_id,
        "timestamp": timestamp.isoformat(),
        "instrument_name": instrument_name,
        "profile_id": profile_id,
        "stage_xyz_um": list(stage_xyz_um) if stage_xyz_um is not None else None,
        "lasers": laser_summary,
        "cameras": {cid: cam.model_dump(exclude={"previews"}) for cid, cam in cameras.items()},
    }
    (output_dir / "snr_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    return report_path


def read_snr_report_html(report_root: Path, report_id: str) -> str:
    """Read a previously generated ``snr_report.html`` under ``report_root``."""
    if "/" in report_id or "\\" in report_id or ".." in report_id:
        raise FileNotFoundError(report_id)
    report_path = report_root / report_id / "snr_report.html"
    if not report_path.is_file():
        raise FileNotFoundError(f"SNR report '{report_id}' not found")
    return report_path.read_text(encoding="utf-8")
