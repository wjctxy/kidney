"""Step 5：基于慢速微泡轨迹生成鼠类肾小球候选区并计数。

本步骤参考论文中鼠类 sULM 后处理思路：用慢速皮质轨迹的
normalized distance 在 isotropic 物理网格上生成 high-ND map，扩张为
glomerular mask，再反向筛选轨迹点并按轨迹中心聚类计数。
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy import ndimage
from skimage import measure, morphology
from sklearn.cluster import DBSCAN


@dataclass
class Step5Config:
    """Step 5 的所有物理尺度和阈值配置。"""

    fps: float = 30.0
    x_spacing_mm: float = 0.08
    y_spacing_mm: float = 0.047
    iso_spacing_mm: float = 0.02
    glomerulus_radius_mm: float = 0.05
    block_size: int = 200
    nd_percentile: float = 90.0
    nd_track_percentile: float = 75.0
    normalized_distance_clip: float = 50.0
    gaussian_sigma_radius_ratio: float = 0.5
    min_track_length_points: int = 3
    min_duration_sec: float = 0.10
    strict_inside_frac: float = 0.80
    loose_inside_frac: float = 0.0
    dbscan_eps_mm: float = 0.23
    dbscan_min_samples: int = 1
    fast_vessel_percentile: float = 95.0
    fast_vessel_dilate_mm: float = 0.10
    border_margin_mm: float = 0.20
    projection_mode: str = "center"
    min_component_area_factor: float = 0.25
    max_component_area_factor: float = 8.0
    cortex_inside_frac_min: float = 0.50
    exclude_inside_frac_max: float = 0.0
    distribution_super_res_factor: int = 4
    distribution_gaussian_sigma_px: float = 4.8
    calibration_target_count: int = 450


def run_step5_glomeruli_count(
    slow_tracks: Any,
    fast_tracks: Any | None = None,
    slow_density: np.ndarray | None = None,
    fast_density: np.ndarray | None = None,
    cortex_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
    config: dict[str, Any] | Step5Config | None = None,
    output_dir: str | Path = "step5_outputs",
) -> dict[str, Any]:
    """运行 Step 5 并返回肾小球候选、筛选点、逐 block 计数和 mask。

    参数中的轨迹可以是 DataFrame、CSV 路径或 dict/list。轨迹点至少需要
    track_id、frame/frame_id、x_px/x_pixel、y_px/y_pixel；block_id 缺失时
    自动设为 0。所有距离计算都使用 mm，不使用原始 pixel 距离。
    """

    cfg = _resolve_config(config)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []
    radius_iso_px = cfg.glomerulus_radius_mm / cfg.iso_spacing_mm
    if radius_iso_px < 2:
        _warn(warnings_list, "isotropic grid 太粗：radius_iso_px < 2，不建议用于肾小球计数。")
    if cortex_mask is None:
        _warn(warnings_list, "未提供 cortex_mask；无 cortex ROI 会明显增加肾小球误检。")

    slow_df = _prepare_tracks(slow_tracks, cfg, name="slow_tracks")
    original_shape = _estimate_original_shape(slow_df, slow_density, fast_density, cortex_mask, exclude_mask)
    iso_shape = _iso_shape_from_original(original_shape, cfg)

    cortex_mask_bool = _prepare_bool_mask(cortex_mask, original_shape, default=True)
    exclude_mask_bool = _prepare_bool_mask(exclude_mask, original_shape, default=False)
    border_mask = _border_exclude_mask(original_shape, cfg)

    if fast_density is None and fast_tracks is not None:
        fast_df = _prepare_tracks(fast_tracks, cfg, name="fast_tracks")
        fast_density = _tracks_to_original_density(fast_df, original_shape)

    cortex_iso_mask = _mask_original_to_iso(cortex_mask_bool, iso_shape, cfg)
    exclude_iso_mask = _mask_original_to_iso(exclude_mask_bool, iso_shape, cfg)
    border_iso_mask = _mask_original_to_iso(border_mask, iso_shape, cfg)
    fast_vessel_mask_iso = _fast_vessel_mask_iso(fast_density, cortex_mask_bool, original_shape, iso_shape, cfg)

    metrics = _compute_track_metrics(slow_df, cortex_mask_bool, exclude_mask_bool, border_mask, cfg)
    metrics["passes_length"] = metrics["n_points"] >= cfg.min_track_length_points
    metrics["passes_duration"] = metrics["duration_sec"] >= cfg.min_duration_sec
    metrics["passes_cortex"] = metrics["cortex_inside_frac"] >= cfg.cortex_inside_frac_min
    metrics["passes_exclusion"] = metrics["exclude_inside_frac"] <= cfg.exclude_inside_frac_max
    metrics["passes_border"] = metrics["border_inside_frac"] == 0
    metrics["roi_keep"] = (
        metrics["passes_length"]
        & metrics["passes_duration"]
        & metrics["passes_cortex"]
        & metrics["passes_exclusion"]
        & metrics["passes_border"]
    )

    kept_metrics = metrics[metrics["roi_keep"]].copy()
    nd_map_iso, nd_map_smooth_iso = _build_nd_map_iso(kept_metrics, slow_df, iso_shape, cfg)
    seed_mask_iso = _seed_mask_from_nd_map(
        nd_map_smooth_iso,
        cortex_iso_mask,
        exclude_iso_mask | border_iso_mask | fast_vessel_mask_iso,
        cfg,
    )
    glomerular_mask_iso = _glomerular_mask(seed_mask_iso, cortex_iso_mask, exclude_iso_mask | border_iso_mask | fast_vessel_mask_iso, cfg)

    metrics = _add_glomerular_inside_fraction(metrics, slow_df, glomerular_mask_iso, iso_shape)
    metrics = _add_center_mask_flag(metrics, fast_vessel_mask_iso, "center_in_fast_vessel")
    metrics["candidate_keep"] = metrics["roi_keep"] & ~metrics["center_in_fast_vessel"]
    loose_metrics = metrics[metrics["candidate_keep"] & (metrics["glom_inside_frac"] >= cfg.loose_inside_frac)].copy()
    strict_metrics = metrics[metrics["candidate_keep"] & (metrics["glom_inside_frac"] >= cfg.strict_inside_frac)].copy()

    filtered_points_loose = _filtered_points(slow_df, metrics, loose_metrics, mode="loose")
    filtered_points_strict = _filtered_points(slow_df, metrics, strict_metrics, mode="strict")

    final_glomeruli_loose = _cluster_track_centers(loose_metrics, cfg, mode="loose")
    final_glomeruli_strict = _cluster_track_centers(strict_metrics, cfg, mode="strict")
    per_block_counts = _per_block_counts(metrics, loose_metrics, strict_metrics, cfg)

    final_combined = pd.concat([final_glomeruli_loose, final_glomeruli_strict], ignore_index=True)
    _write_outputs(
        output_path,
        cfg,
        metrics,
        filtered_points_loose,
        filtered_points_strict,
        per_block_counts,
        final_combined,
        nd_map_iso,
        nd_map_smooth_iso,
        seed_mask_iso,
        glomerular_mask_iso,
        fast_vessel_mask_iso,
        cortex_iso_mask,
        exclude_iso_mask,
    )
    distribution_paths = _write_glomerular_track_distribution_maps(
        output_path,
        filtered_points_loose,
        filtered_points_strict,
        original_shape,
        cfg,
        slow_density,
    )

    summary = {
        "total_slow_tracks": int(len(metrics)),
        "tracks_after_cortex_filter": int((metrics["passes_length"] & metrics["passes_duration"] & metrics["passes_cortex"]).sum()),
        "tracks_after_exclusion_filter": int(metrics["roi_keep"].sum()),
        "tracks_after_fast_vessel_filter": int(metrics["candidate_keep"].sum()),
        "loose_glom_tracks": int(len(loose_metrics)),
        "strict_glom_tracks": int(len(strict_metrics)),
        "loose_glomeruli_count": int(len(final_glomeruli_loose)),
        "strict_glomeruli_count": int(len(final_glomeruli_strict)),
        "iso_spacing_mm": float(cfg.iso_spacing_mm),
        "glomerulus_radius_mm": float(cfg.glomerulus_radius_mm),
        "radius_iso_px": float(radius_iso_px),
        "dbscan_eps_mm": float(cfg.dbscan_eps_mm),
        "dbscan_min_samples": int(cfg.dbscan_min_samples),
        "loose_inside_frac": float(cfg.loose_inside_frac),
        "strict_inside_frac": float(cfg.strict_inside_frac),
        "calibration_target_count": int(cfg.calibration_target_count),
        "calibration_note": "loose parameters calibrated against expected CT slice glomerulus count around 450; masks are unchanged",
        "warnings": warnings_list,
    }
    if len(final_glomeruli_loose) > 0 and len(final_glomeruli_strict) < 0.3 * len(final_glomeruli_loose):
        _warn(summary["warnings"], "strict_count 远小于 calibrated loose_count；strict 仅作为保守参考，标定计数以 loose_count 为准。")
    if _many_centers_near_fast_vessels(final_glomeruli_loose, final_glomeruli_strict, fast_vessel_mask_iso, cfg):
        _warn(summary["warnings"], "较多候选中心靠近 fast_vessel_mask；建议检查主血管排除阈值。")

    _save_summary(output_path / "summary.json", summary)
    _save_visualizations(
        output_path,
        slow_density,
        fast_density,
        original_shape,
        cfg,
        cortex_mask_bool,
        exclude_mask_bool,
        glomerular_mask_iso,
        seed_mask_iso,
        fast_vessel_mask_iso,
        nd_map_smooth_iso,
        metrics,
        loose_metrics,
        strict_metrics,
        final_glomeruli_loose,
        final_glomeruli_strict,
    )

    return {
        "final_glomeruli_loose": final_glomeruli_loose,
        "final_glomeruli_strict": final_glomeruli_strict,
        "filtered_points_loose": filtered_points_loose,
        "filtered_points_strict": filtered_points_strict,
        "per_block_counts": per_block_counts,
        "summary": summary,
        "glomerular_track_distribution": distribution_paths,
        "masks": {
            "glomerular_mask_iso": glomerular_mask_iso,
            "nd_map_iso": nd_map_smooth_iso,
            "seed_mask_iso": seed_mask_iso,
            "fast_vessel_mask_iso": fast_vessel_mask_iso,
            "cortex_iso_mask": cortex_iso_mask,
            "exclude_iso_mask": exclude_iso_mask,
        },
    }


def _resolve_config(config: dict[str, Any] | Step5Config | None) -> Step5Config:
    """合并默认配置和用户覆盖配置。"""

    if config is None:
        return Step5Config()
    if isinstance(config, Step5Config):
        return config
    values = asdict(Step5Config())
    values.update(config)
    return Step5Config(**values)


def _prepare_tracks(tracks: Any, cfg: Step5Config, name: str) -> pd.DataFrame:
    """读取并标准化轨迹字段，同时保留输入中的原始字段。"""

    if isinstance(tracks, (str, Path)):
        df = pd.read_csv(tracks)
    elif isinstance(tracks, pd.DataFrame):
        df = tracks.copy()
    else:
        df = pd.DataFrame(tracks)
    if df.empty:
        raise ValueError(f"{name} 为空，无法运行 Step 5。")

    _ensure_canonical_column(df, "track_id", ["track_id", "id", "track"])
    _ensure_canonical_column(df, "frame", ["frame", "frame_id", "t", "time_index"])
    _ensure_canonical_column(df, "x_px", ["x_px", "x_pixel", "x", "col", "column"])
    _ensure_canonical_column(df, "y_px", ["y_px", "y_pixel", "y", "row"])
    if "block_id" not in df.columns:
        df["block_id"] = 0

    for col in ["track_id", "block_id", "frame", "x_px", "y_px"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["track_id", "block_id", "frame", "x_px", "y_px"]].isna().any().any():
        raise ValueError(f"{name} 中 track_id/block_id/frame/x/y 存在无法转换为数值的字段。")

    df["track_id"] = df["track_id"].astype(int)
    df["block_id"] = df["block_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    df["x_mm"] = df["x_px"].astype(float) * cfg.x_spacing_mm
    df["y_mm"] = df["y_px"].astype(float) * cfg.y_spacing_mm
    df["x_iso"] = df["x_mm"] / cfg.iso_spacing_mm
    df["y_iso"] = df["y_mm"] / cfg.iso_spacing_mm
    df["track_key"] = _make_track_key(df)
    return df


def _ensure_canonical_column(df: pd.DataFrame, canonical: str, aliases: list[str]) -> None:
    """用字段别名填充标准字段，缺失时报明确错误。"""

    if canonical in df.columns:
        return
    for alias in aliases:
        if alias in df.columns:
            df[canonical] = df[alias]
            return
    raise ValueError(f"缺少必需字段 {canonical}，可接受别名：{', '.join(aliases)}")


def _make_track_key(df: pd.DataFrame) -> pd.Series:
    """生成 block + track 复合键，避免不同 block 的 track_id 撞号。"""

    return df["block_id"].astype(str) + ":" + df["track_id"].astype(str)


def _estimate_original_shape(
    slow_df: pd.DataFrame,
    slow_density: np.ndarray | None,
    fast_density: np.ndarray | None,
    cortex_mask: np.ndarray | None,
    exclude_mask: np.ndarray | None,
) -> tuple[int, int]:
    """从轨迹和输入图像估计原始坐标尺寸。

    Step04 的 density PNG 通常是超分辨率显示图，例如 4x。这里优先识别
    这种整数放大尺寸，并还原到轨迹使用的原始像素坐标系。
    """

    track_height = int(math.ceil(float(slow_df["y_px"].max()) + 1))
    track_width = int(math.ceil(float(slow_df["x_px"].max()) + 1))
    track_shape = max(track_height, 1), max(track_width, 1)

    for arr in [cortex_mask, exclude_mask, slow_density, fast_density]:
        if arr is not None:
            shape = np.asarray(arr).shape[:2]
            return _shape_to_original_coordinates((int(shape[0]), int(shape[1])), track_shape)
    return track_shape


def _shape_to_original_coordinates(shape: tuple[int, int], track_shape: tuple[int, int]) -> tuple[int, int]:
    """将可能的超分辨率 density/mask shape 还原到原始轨迹坐标 shape。"""

    height, width = shape
    track_height, track_width = track_shape
    if height < track_height or width < track_width:
        return track_shape

    ratio_h = height / max(track_height, 1)
    ratio_w = width / max(track_width, 1)
    if ratio_h > 1.5 and ratio_w > 1.5 and abs(ratio_h - ratio_w) < 0.25:
        factor = int(round((ratio_h + ratio_w) / 2))
        if 2 <= factor <= 8:
            original = (int(round(height / factor)), int(round(width / factor)))
            if original[0] >= track_height and original[1] >= track_width:
                return original
    return int(height), int(width)


def _iso_shape_from_original(original_shape: tuple[int, int], cfg: Step5Config) -> tuple[int, int]:
    """根据原始图像尺寸和物理 spacing 计算 isotropic grid 尺寸。"""

    height, width = original_shape
    iso_h = int(math.ceil(height * cfg.y_spacing_mm / cfg.iso_spacing_mm))
    iso_w = int(math.ceil(width * cfg.x_spacing_mm / cfg.iso_spacing_mm))
    return max(iso_h, 1), max(iso_w, 1)


def _prepare_bool_mask(mask: np.ndarray | None, shape: tuple[int, int], default: bool) -> np.ndarray:
    """准备原始坐标 bool mask，缺失时用全 True 或全 False。"""

    if mask is None:
        return np.full(shape, default, dtype=bool)
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.shape[:2] != shape:
        arr = _resize_nearest(arr.astype(bool), shape)
    return arr.astype(bool)


def _resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """最近邻缩放二维数组到目标 shape。"""

    zoom = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    return ndimage.zoom(arr, zoom, order=0)


def _resize_float(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """线性缩放二维 float 数组到目标 shape。"""

    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.shape[:2] == shape:
        return arr.astype(np.float32)
    zoom = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    return ndimage.zoom(arr.astype(np.float32), zoom, order=1)


def _border_exclude_mask(shape: tuple[int, int], cfg: Step5Config) -> np.ndarray:
    """在原始图像坐标中生成按 mm 定义的边界排除 mask。"""

    height, width = shape
    yy, xx = np.indices(shape)
    x_mm = xx * cfg.x_spacing_mm
    y_mm = yy * cfg.y_spacing_mm
    width_mm = width * cfg.x_spacing_mm
    height_mm = height * cfg.y_spacing_mm
    return (
        (x_mm < cfg.border_margin_mm)
        | (x_mm > width_mm - cfg.border_margin_mm)
        | (y_mm < cfg.border_margin_mm)
        | (y_mm > height_mm - cfg.border_margin_mm)
    )


def _mask_original_to_iso(mask: np.ndarray, iso_shape: tuple[int, int], cfg: Step5Config) -> np.ndarray:
    """把原始坐标 mask 最近邻采样到 isotropic grid。"""

    iso_h, iso_w = iso_shape
    src_h, src_w = mask.shape
    y_src = np.clip(np.rint(np.arange(iso_h) * cfg.iso_spacing_mm / cfg.y_spacing_mm).astype(int), 0, src_h - 1)
    x_src = np.clip(np.rint(np.arange(iso_w) * cfg.iso_spacing_mm / cfg.x_spacing_mm).astype(int), 0, src_w - 1)
    return mask[y_src[:, None], x_src[None, :]].astype(bool)


def _mask_iso_to_original(mask_iso: np.ndarray, original_shape: tuple[int, int], cfg: Step5Config) -> np.ndarray:
    """把 isotropic mask 最近邻采样回原始图像坐标，供可视化 overlay。"""

    height, width = original_shape
    iso_h, iso_w = mask_iso.shape
    y_iso = np.clip(np.rint(np.arange(height) * cfg.y_spacing_mm / cfg.iso_spacing_mm).astype(int), 0, iso_h - 1)
    x_iso = np.clip(np.rint(np.arange(width) * cfg.x_spacing_mm / cfg.iso_spacing_mm).astype(int), 0, iso_w - 1)
    return mask_iso[y_iso[:, None], x_iso[None, :]].astype(bool)


def _fast_vessel_mask_iso(
    fast_density: np.ndarray | None,
    cortex_mask: np.ndarray,
    original_shape: tuple[int, int],
    iso_shape: tuple[int, int],
    cfg: Step5Config,
) -> np.ndarray:
    """从 fast density 生成 isotropic 主血管排除 mask。"""

    if fast_density is None:
        return np.zeros(iso_shape, dtype=bool)
    density = _resize_float(fast_density, original_shape)
    scope = cortex_mask & (density > 0)
    values = density[scope]
    if values.size == 0:
        return np.zeros(iso_shape, dtype=bool)
    threshold = float(np.percentile(values, cfg.fast_vessel_percentile))
    mask_original = density >= threshold
    mask_iso = _mask_original_to_iso(mask_original, iso_shape, cfg)
    dilate_px = max(1, int(math.ceil(cfg.fast_vessel_dilate_mm / cfg.iso_spacing_mm)))
    return morphology.dilation(mask_iso, morphology.disk(dilate_px))


def _tracks_to_original_density(df: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    """把 fast tracks 临时投影为原始坐标 density。"""

    density = np.zeros(shape, dtype=np.float32)
    x = np.rint(df["x_px"].to_numpy()).astype(int)
    y = np.rint(df["y_px"].to_numpy()).astype(int)
    inside = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    np.add.at(density, (y[inside], x[inside]), 1.0)
    return density


def _compute_track_metrics(
    df: pd.DataFrame,
    cortex_mask: np.ndarray,
    exclude_mask: np.ndarray,
    border_mask: np.ndarray,
    cfg: Step5Config,
) -> pd.DataFrame:
    """按 track 计算 ND、速度、中心和 ROI 覆盖比例。"""

    rows: list[dict[str, Any]] = []
    for track_key, group in df.sort_values(["block_id", "track_id", "frame"]).groupby("track_key", sort=False):
        group = group.sort_values("frame")
        x_mm = group["x_mm"].to_numpy(dtype=float)
        y_mm = group["y_mm"].to_numpy(dtype=float)
        if len(group) > 1:
            step_lengths = np.hypot(np.diff(x_mm), np.diff(y_mm))
            path_length = float(step_lengths.sum())
            straight = float(math.hypot(x_mm[-1] - x_mm[0], y_mm[-1] - y_mm[0]))
        else:
            path_length = 0.0
            straight = 0.0
        if straight <= 1e-9:
            normalized_distance = cfg.normalized_distance_clip if path_length > 0 else 0.0
        else:
            normalized_distance = path_length / straight
        normalized_distance = float(np.clip(normalized_distance, 0.0, cfg.normalized_distance_clip))

        min_frame = int(group["frame"].min())
        max_frame = int(group["frame"].max())
        duration = (max_frame - min_frame + 1) / cfg.fps
        x_idx = np.clip(np.rint(group["x_px"].to_numpy()).astype(int), 0, cortex_mask.shape[1] - 1)
        y_idx = np.clip(np.rint(group["y_px"].to_numpy()).astype(int), 0, cortex_mask.shape[0] - 1)
        rows.append(
            {
                "track_key": track_key,
                "track_id": int(group["track_id"].iloc[0]),
                "block_id": int(group["block_id"].iloc[0]),
                "n_points": int(len(group)),
                "min_frame": min_frame,
                "max_frame": max_frame,
                "path_length_mm": path_length,
                "end_to_end_distance_mm": straight,
                "normalized_distance": normalized_distance,
                "duration_sec": float(duration),
                "mean_speed_mm_s": float(path_length / duration) if duration > 0 else 0.0,
                "center_x_mm": float(np.median(x_mm)),
                "center_y_mm": float(np.median(y_mm)),
                "center_x_iso": float(np.median(group["x_iso"])),
                "center_y_iso": float(np.median(group["y_iso"])),
                "cortex_inside_frac": float(np.mean(cortex_mask[y_idx, x_idx])),
                "exclude_inside_frac": float(np.mean(exclude_mask[y_idx, x_idx])),
                "border_inside_frac": float(np.mean(border_mask[y_idx, x_idx])),
            }
        )
    return pd.DataFrame(rows)


def _build_nd_map_iso(
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    iso_shape: tuple[int, int],
    cfg: Step5Config,
) -> tuple[np.ndarray, np.ndarray]:
    """把高 normalized-distance 轨迹投影到 isotropic grid。"""

    nd_map = np.zeros(iso_shape, dtype=np.float32)
    if metrics.empty:
        return nd_map, nd_map.copy()
    threshold = float(np.percentile(metrics["normalized_distance"], cfg.nd_track_percentile))
    high_metrics = metrics[metrics["normalized_distance"] >= threshold]

    if cfg.projection_mode == "points":
        lookup = high_metrics.set_index("track_key")["normalized_distance"].to_dict()
        subset = points[points["track_key"].isin(lookup)]
        for track_key, group in subset.groupby("track_key", sort=False):
            weight = float(lookup[track_key]) / max(len(group), 1)
            x = np.rint(group["x_iso"].to_numpy()).astype(int)
            y = np.rint(group["y_iso"].to_numpy()).astype(int)
            inside = (x >= 0) & (x < iso_shape[1]) & (y >= 0) & (y < iso_shape[0])
            np.add.at(nd_map, (y[inside], x[inside]), weight)
    elif cfg.projection_mode == "center":
        x = np.rint(high_metrics["center_x_iso"].to_numpy()).astype(int)
        y = np.rint(high_metrics["center_y_iso"].to_numpy()).astype(int)
        nd = high_metrics["normalized_distance"].to_numpy(dtype=float)
        inside = (x >= 0) & (x < iso_shape[1]) & (y >= 0) & (y < iso_shape[0])
        np.add.at(nd_map, (y[inside], x[inside]), nd[inside])
    else:
        raise ValueError("projection_mode 只能是 'center' 或 'points'。")

    sigma_px = radius_sigma_px(cfg)
    smoothed = ndimage.gaussian_filter(nd_map, sigma=sigma_px) if sigma_px > 0 else nd_map.copy()
    return nd_map, smoothed.astype(np.float32)


def radius_sigma_px(cfg: Step5Config) -> float:
    """根据肾小球半径计算 Gaussian sigma，单位为 isotropic pixel。"""

    return (cfg.glomerulus_radius_mm / cfg.iso_spacing_mm) * cfg.gaussian_sigma_radius_ratio


def _seed_mask_from_nd_map(
    nd_map: np.ndarray,
    cortex_iso_mask: np.ndarray,
    exclusion_iso_mask: np.ndarray,
    cfg: Step5Config,
) -> np.ndarray:
    """在 cortex 内按 ND 百分位阈值生成 candidate seed mask。"""

    valid = cortex_iso_mask & ~exclusion_iso_mask
    positive = nd_map[valid & (nd_map > 0)]
    if positive.size == 0:
        return np.zeros_like(nd_map, dtype=bool)
    threshold = float(np.percentile(positive, cfg.nd_percentile))
    return (nd_map >= threshold) & valid & (nd_map > 0)


def _glomerular_mask(
    seed_mask: np.ndarray,
    cortex_iso_mask: np.ndarray,
    exclusion_iso_mask: np.ndarray,
    cfg: Step5Config,
) -> np.ndarray:
    """将 seed 按鼠类肾小球半径扩张，并过滤异常大小连通域。"""

    radius_iso_px = cfg.glomerulus_radius_mm / cfg.iso_spacing_mm
    disk = morphology.disk(max(1, int(math.ceil(radius_iso_px))))
    mask = morphology.dilation(seed_mask, disk) & cortex_iso_mask & ~exclusion_iso_mask
    area_ref = math.pi * radius_iso_px**2
    min_area = cfg.min_component_area_factor * area_ref
    max_area = cfg.max_component_area_factor * area_ref
    labeled = measure.label(mask)
    kept = np.zeros_like(mask, dtype=bool)
    for region in measure.regionprops(labeled):
        if min_area <= region.area <= max_area:
            kept[labeled == region.label] = True
    return kept


def _add_glomerular_inside_fraction(
    metrics: pd.DataFrame,
    points: pd.DataFrame,
    glomerular_mask_iso: np.ndarray,
    iso_shape: tuple[int, int],
) -> pd.DataFrame:
    """给每条轨迹补充落入 glomerular mask 的比例。"""

    frac_by_key: dict[str, float] = {}
    for track_key, group in points.groupby("track_key", sort=False):
        x = np.clip(np.rint(group["x_iso"].to_numpy()).astype(int), 0, iso_shape[1] - 1)
        y = np.clip(np.rint(group["y_iso"].to_numpy()).astype(int), 0, iso_shape[0] - 1)
        frac_by_key[track_key] = float(np.mean(glomerular_mask_iso[y, x]))
    metrics = metrics.copy()
    metrics["glom_inside_frac"] = metrics["track_key"].map(frac_by_key).fillna(0.0)
    return metrics


def _add_center_mask_flag(metrics: pd.DataFrame, mask_iso: np.ndarray, column: str) -> pd.DataFrame:
    """标记 track-level center 是否落入某个 isotropic mask。"""

    metrics = metrics.copy()
    if metrics.empty:
        metrics[column] = pd.Series(dtype=bool)
        return metrics
    x = np.clip(np.rint(metrics["center_x_iso"].to_numpy()).astype(int), 0, mask_iso.shape[1] - 1)
    y = np.clip(np.rint(metrics["center_y_iso"].to_numpy()).astype(int), 0, mask_iso.shape[0] - 1)
    metrics[column] = mask_iso[y, x].astype(bool)
    return metrics


def _filtered_points(points: pd.DataFrame, metrics: pd.DataFrame, selected_metrics: pd.DataFrame, mode: str) -> pd.DataFrame:
    """反向筛选属于肾小球轨迹的 Step3 原始点，并附加轨迹指标。"""

    selected_keys = set(selected_metrics["track_key"])
    subset = points[points["track_key"].isin(selected_keys)].copy()
    metric_cols = [
        "track_key",
        "normalized_distance",
        "glom_inside_frac",
        "path_length_mm",
        "duration_sec",
        "mean_speed_mm_s",
        "center_x_mm",
        "center_y_mm",
    ]
    subset = subset.merge(metrics[metric_cols], on="track_key", how="left", suffixes=("", "_track"))
    subset["mode"] = mode
    return subset


def _cluster_track_centers(metrics: pd.DataFrame, cfg: Step5Config, mode: str) -> pd.DataFrame:
    """用 DBSCAN 在 mm 坐标下对 track-level centers 聚类计数。"""

    columns = [
        "glomerulus_id",
        "center_x_mm",
        "center_y_mm",
        "center_x_px_original",
        "center_y_px_original",
        "n_tracks",
        "n_blocks",
        "median_normalized_distance",
        "median_speed_mm_s",
        "mode",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)

    coords = metrics[["center_x_mm", "center_y_mm"]].to_numpy(dtype=float)
    labels = DBSCAN(eps=cfg.dbscan_eps_mm, min_samples=cfg.dbscan_min_samples).fit_predict(coords)
    rows: list[dict[str, Any]] = []
    glomerulus_id = 0
    for label in sorted(set(labels)):
        if label < 0:
            continue
        cluster = metrics[labels == label]
        glomerulus_id += 1
        center_x = float(np.median(cluster["center_x_mm"]))
        center_y = float(np.median(cluster["center_y_mm"]))
        rows.append(
            {
                "glomerulus_id": glomerulus_id,
                "center_x_mm": center_x,
                "center_y_mm": center_y,
                "center_x_px_original": center_x / cfg.x_spacing_mm,
                "center_y_px_original": center_y / cfg.y_spacing_mm,
                "n_tracks": int(len(cluster)),
                "n_blocks": int(cluster["block_id"].nunique()),
                "median_normalized_distance": float(np.median(cluster["normalized_distance"])),
                "median_speed_mm_s": float(np.median(cluster["mean_speed_mm_s"])),
                "mode": mode,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _per_block_counts(
    metrics: pd.DataFrame,
    loose_metrics: pd.DataFrame,
    strict_metrics: pd.DataFrame,
    cfg: Step5Config,
) -> pd.DataFrame:
    """逐 block 单独 DBSCAN 计数，便于检查跨 block 聚合前的结果。"""

    rows: list[dict[str, Any]] = []
    for block_id in sorted(metrics["block_id"].unique()):
        block_all = metrics[metrics["block_id"] == block_id]
        block_loose = loose_metrics[loose_metrics["block_id"] == block_id]
        block_strict = strict_metrics[strict_metrics["block_id"] == block_id]
        rows.append(
            {
                "block_id": int(block_id),
                "loose_count": int(len(_cluster_track_centers(block_loose, cfg, mode="loose"))),
                "strict_count": int(len(_cluster_track_centers(block_strict, cfg, mode="strict"))),
                "n_slow_tracks": int(len(block_all)),
                "n_loose_tracks": int(len(block_loose)),
                "n_strict_tracks": int(len(block_strict)),
            }
        )
    return pd.DataFrame(rows)


def _many_centers_near_fast_vessels(
    loose: pd.DataFrame,
    strict: pd.DataFrame,
    fast_vessel_mask_iso: np.ndarray,
    cfg: Step5Config,
) -> bool:
    """粗略检查候选中心是否大量落在 fast vessel 排除区附近。"""

    centers = pd.concat([loose, strict], ignore_index=True)
    if centers.empty or not fast_vessel_mask_iso.any():
        return False
    radius_px = max(1, int(math.ceil(cfg.fast_vessel_dilate_mm / cfg.iso_spacing_mm)))
    near_mask = morphology.dilation(fast_vessel_mask_iso, morphology.disk(radius_px))
    x = np.clip(np.rint(centers["center_x_mm"].to_numpy() / cfg.iso_spacing_mm).astype(int), 0, near_mask.shape[1] - 1)
    y = np.clip(np.rint(centers["center_y_mm"].to_numpy() / cfg.iso_spacing_mm).astype(int), 0, near_mask.shape[0] - 1)
    return float(np.mean(near_mask[y, x])) > 0.2


def _write_outputs(
    output_dir: Path,
    cfg: Step5Config,
    metrics: pd.DataFrame,
    filtered_points_loose: pd.DataFrame,
    filtered_points_strict: pd.DataFrame,
    per_block_counts: pd.DataFrame,
    final_glomeruli: pd.DataFrame,
    nd_map_iso: np.ndarray,
    nd_map_smooth_iso: np.ndarray,
    seed_mask_iso: np.ndarray,
    glomerular_mask_iso: np.ndarray,
    fast_vessel_mask_iso: np.ndarray,
    cortex_iso_mask: np.ndarray,
    exclude_iso_mask: np.ndarray,
) -> None:
    """保存 CSV、NPY 和配置快照。"""

    metrics.to_csv(output_dir / "track_metrics.csv", index=False)
    filtered_points_loose.to_csv(output_dir / "filtered_points_loose.csv", index=False)
    filtered_points_strict.to_csv(output_dir / "filtered_points_strict.csv", index=False)
    per_block_counts.to_csv(output_dir / "per_block_counts.csv", index=False)
    final_glomeruli.to_csv(output_dir / "final_glomeruli.csv", index=False)
    final_glomeruli[final_glomeruli["mode"] == "loose"].to_csv(output_dir / "final_glomeruli_loose.csv", index=False)
    final_glomeruli[final_glomeruli["mode"] == "strict"].to_csv(output_dir / "final_glomeruli_strict.csv", index=False)
    np.save(output_dir / "nd_map_raw_iso.npy", nd_map_iso)
    np.save(output_dir / "nd_map_iso.npy", nd_map_smooth_iso)
    np.save(output_dir / "seed_mask_iso.npy", seed_mask_iso)
    np.save(output_dir / "glomerular_mask_iso.npy", glomerular_mask_iso)
    np.save(output_dir / "fast_vessel_mask_iso.npy", fast_vessel_mask_iso)
    np.save(output_dir / "cortex_mask_iso.npy", cortex_iso_mask)
    np.save(output_dir / "exclude_mask_iso.npy", exclude_iso_mask)
    _save_summary(output_dir / "config.json", asdict(cfg))


def _write_glomerular_track_distribution_maps(
    output_dir: Path,
    filtered_points_loose: pd.DataFrame,
    filtered_points_strict: pd.DataFrame,
    original_shape: tuple[int, int],
    cfg: Step5Config,
    slow_density: np.ndarray | None = None,
) -> dict[str, Path]:
    """用筛出的肾小球相关轨迹点重建分布图。

    这里重建的是 loose/strict glomerular tracks 的空间分布，不是候选中心
    计数图。每个保留轨迹点按原始像素坐标投影到 4 倍超分辨率网格，再
    Gaussian 平滑成密度图。
    """

    loose_density = _distribution_density_from_points(filtered_points_loose, original_shape, cfg)
    strict_density = _distribution_density_from_points(filtered_points_strict, original_shape, cfg)

    loose_npy = output_dir / "glomerular_track_density_loose.npy"
    strict_npy = output_dir / "glomerular_track_density_strict.npy"
    loose_png = output_dir / "glomerular_track_distribution_loose.png"
    strict_png = output_dir / "glomerular_track_distribution_strict.png"
    overlay_png = output_dir / "glomerular_track_distribution_overlay.png"
    loose_on_bg_png = output_dir / "glomerular_track_distribution_loose_on_slow_density.png"
    strict_on_bg_png = output_dir / "glomerular_track_distribution_strict_on_slow_density.png"

    np.save(loose_npy, loose_density)
    np.save(strict_npy, strict_density)
    _save_distribution_image(loose_png, loose_density, rgb=(255, 0, 255), title="Loose glomerular-track distribution")
    _save_distribution_image(strict_png, strict_density, rgb=(0, 255, 255), title="Strict glomerular-track distribution")
    _save_distribution_overlay(overlay_png, loose_density, strict_density)
    _save_distribution_on_background(
        loose_on_bg_png,
        loose_density,
        slow_density,
        rgb=(255, 0, 255),
        title="Loose glomerular-track distribution on slow density",
    )
    _save_distribution_on_background(
        strict_on_bg_png,
        strict_density,
        slow_density,
        rgb=(0, 255, 255),
        title="Strict glomerular-track distribution on slow density",
    )

    return {
        "loose_density": loose_npy,
        "strict_density": strict_npy,
        "loose_distribution": loose_png,
        "strict_distribution": strict_png,
        "overlay_distribution": overlay_png,
        "loose_distribution_on_slow_density": loose_on_bg_png,
        "strict_distribution_on_slow_density": strict_on_bg_png,
    }


def _distribution_density_from_points(points: pd.DataFrame, original_shape: tuple[int, int], cfg: Step5Config) -> np.ndarray:
    """把筛选后的轨迹点累计为超分辨率 density。"""

    scale = max(1, int(cfg.distribution_super_res_factor))
    height, width = original_shape
    density = np.zeros((height * scale, width * scale), dtype=np.float32)
    if points.empty:
        return density

    x = np.rint(points["x_px"].to_numpy(dtype=float) * scale).astype(int)
    y = np.rint(points["y_px"].to_numpy(dtype=float) * scale).astype(int)
    inside = (x >= 0) & (x < density.shape[1]) & (y >= 0) & (y < density.shape[0])
    np.add.at(density, (y[inside], x[inside]), 1.0)
    if density.max() > 0 and cfg.distribution_gaussian_sigma_px > 0:
        density = ndimage.gaussian_filter(density, sigma=float(cfg.distribution_gaussian_sigma_px)).astype(np.float32)
    return density


def _save_distribution_image(path: Path, density: np.ndarray, rgb: tuple[int, int, int], title: str) -> None:
    """保存单类肾小球轨迹分布图。"""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.imshow(_colorize_density(density, rgb))
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_distribution_overlay(path: Path, loose_density: np.ndarray, strict_density: np.ndarray) -> None:
    """保存 loose/strict 肾小球轨迹分布叠加图。"""

    loose = _normalize_density(loose_density)
    strict = _normalize_density(strict_density)
    rgb = np.zeros((*loose_density.shape, 3), dtype=np.float32)
    rgb[..., 0] = loose
    rgb[..., 1] = strict
    rgb[..., 2] = np.maximum(loose, strict)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.imshow(np.clip(rgb, 0, 1))
    ax.set_title("Glomerular-track distribution overlay: loose=magenta, strict=cyan")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_distribution_on_background(
    path: Path,
    density: np.ndarray,
    background: np.ndarray | None,
    rgb: tuple[int, int, int],
    title: str,
) -> None:
    """将肾小球相关轨迹分布叠加到 slow density 背景。"""

    if background is None:
        bg = np.zeros(density.shape, dtype=np.float32)
    else:
        bg = _resize_float(background, density.shape)
    bg_norm = _normalize_for_display(bg)
    color = _colorize_density(density, rgb)
    alpha = np.clip(_normalize_density(density) * 0.85, 0, 0.85)
    image = np.repeat(bg_norm[..., None], 3, axis=2) * 0.55
    image = image * (1.0 - alpha[..., None]) + color * alpha[..., None]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.imshow(np.clip(image, 0, 1))
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _colorize_density(density: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """将单通道 density 映射为指定 RGB 颜色。"""

    norm = _normalize_density(density)
    image = np.zeros((*density.shape, 3), dtype=np.float32)
    image[..., 0] = norm * (rgb[0] / 255.0)
    image[..., 1] = norm * (rgb[1] / 255.0)
    image[..., 2] = norm * (rgb[2] / 255.0)
    return np.clip(image, 0, 1)


def _normalize_density(density: np.ndarray) -> np.ndarray:
    """按非零像素 99 百分位归一化 density。"""

    if density.size == 0 or float(np.nanmax(density)) <= 0:
        return np.zeros(density.shape, dtype=np.float32)
    positive = density[density > 0]
    hi = float(np.percentile(positive, 99)) if positive.size else float(np.nanmax(density))
    return np.clip(density / max(hi, 1e-6), 0, 1).astype(np.float32)


def _save_summary(path: Path, data: dict[str, Any]) -> None:
    """保存 JSON，确保 numpy 标量可序列化。"""

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    """把 numpy 类型转为普通 Python 类型。"""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_visualizations(
    output_dir: Path,
    slow_density: np.ndarray | None,
    fast_density: np.ndarray | None,
    original_shape: tuple[int, int],
    cfg: Step5Config,
    cortex_mask: np.ndarray,
    exclude_mask: np.ndarray,
    glomerular_mask_iso: np.ndarray,
    seed_mask_iso: np.ndarray,
    fast_vessel_mask_iso: np.ndarray,
    nd_map_iso: np.ndarray,
    metrics: pd.DataFrame,
    loose_metrics: pd.DataFrame,
    strict_metrics: pd.DataFrame,
    loose_glomeruli: pd.DataFrame,
    strict_glomeruli: pd.DataFrame,
) -> None:
    """保存 Step 5 诊断图和最终中心 overlay。"""

    slow_bg = _background(slow_density, original_shape)
    fast_bg = _background(fast_density, original_shape)
    glom_original = _mask_iso_to_original(glomerular_mask_iso, original_shape, cfg)
    seed_original = _mask_iso_to_original(seed_mask_iso, original_shape, cfg)
    fast_mask_original = _mask_iso_to_original(fast_vessel_mask_iso, original_shape, cfg)

    _plot_overlay(
        output_dir / "slow_density_overlay_cortex.png",
        slow_bg,
        contours=[(cortex_mask, "cyan", "cortex"), (exclude_mask, "red", "exclude")],
        title="Slow density with cortex/exclude masks",
    )
    _plot_nd_map(output_dir / "nd_map_iso.png", nd_map_iso)
    _plot_overlay(
        output_dir / "glomerular_mask_overlay.png",
        slow_bg,
        contours=[(glom_original, "lime", "glomerular mask"), (seed_original, "yellow", "seeds")],
        title="Glomerular mask and candidate seeds",
    )
    _plot_centers(output_dir / "final_glomeruli_centers_loose.png", slow_bg, loose_glomeruli, cfg, "loose glomeruli")
    _plot_centers(output_dir / "final_glomeruli_centers_strict.png", slow_bg, strict_glomeruli, cfg, "strict glomeruli")
    _plot_overlay(
        output_dir / "fast_exclusion_overlay.png",
        fast_bg,
        contours=[(fast_mask_original, "red", "fast vessel exclude")],
        title="Fast vessel exclusion",
    )
    _plot_track_diagnostics(output_dir / "track_filter_diagnostics.png", slow_bg, metrics, loose_metrics, strict_metrics, cfg)


def _background(density: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """准备可视化背景。"""

    if density is None:
        return np.zeros(shape, dtype=np.float32)
    return _resize_float(density, shape)


def _plot_overlay(path: Path, background: np.ndarray, contours: list[tuple[np.ndarray, str, str]], title: str) -> None:
    """保存带 mask 轮廓的背景图。"""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.imshow(_normalize_for_display(background), cmap="gray")
    for mask, color, label in contours:
        if mask is not None and np.asarray(mask).any():
            ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=0.8)
            ax.plot([], [], color=color, label=label)
    if any(mask is not None and np.asarray(mask).any() for mask, _, _ in contours):
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_nd_map(path: Path, nd_map: np.ndarray) -> None:
    """保存 isotropic high-ND map。"""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    im = ax.imshow(nd_map, cmap="magma")
    ax.set_title("High normalized-distance map on isotropic grid")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_centers(path: Path, background: np.ndarray, glomeruli: pd.DataFrame, cfg: Step5Config, title: str) -> None:
    """保存最终肾小球中心，圆按 0.05 mm 半径映射成原始像素椭圆。"""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.imshow(_normalize_for_display(background), cmap="gray")
    for _, row in glomeruli.iterrows():
        x = float(row["center_x_px_original"])
        y = float(row["center_y_px_original"])
        ellipse = Ellipse(
            (x, y),
            width=2 * cfg.glomerulus_radius_mm / cfg.x_spacing_mm,
            height=2 * cfg.glomerulus_radius_mm / cfg.y_spacing_mm,
            fill=False,
            edgecolor="lime",
            linewidth=0.9,
        )
        ax.add_patch(ellipse)
        ax.scatter([x], [y], s=8, c="yellow", edgecolors="black", linewidths=0.2)
    ax.set_title(f"{title}: {len(glomeruli)}")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_track_diagnostics(
    path: Path,
    background: np.ndarray,
    metrics: pd.DataFrame,
    loose_metrics: pd.DataFrame,
    strict_metrics: pd.DataFrame,
    cfg: Step5Config,
) -> None:
    """显示所有慢速中心、loose 中心、strict 中心的对比。"""

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.imshow(_normalize_for_display(background), cmap="gray")
    if not metrics.empty:
        ax.scatter(metrics["center_x_mm"] / cfg.x_spacing_mm, metrics["center_y_mm"] / cfg.y_spacing_mm, s=2, c="white", alpha=0.25, label="all slow")
    if not loose_metrics.empty:
        ax.scatter(loose_metrics["center_x_mm"] / cfg.x_spacing_mm, loose_metrics["center_y_mm"] / cfg.y_spacing_mm, s=6, c="deepskyblue", alpha=0.8, label="loose tracks")
    if not strict_metrics.empty:
        ax.scatter(strict_metrics["center_x_mm"] / cfg.x_spacing_mm, strict_metrics["center_y_mm"] / cfg.y_spacing_mm, s=10, c="yellow", alpha=0.9, label="strict tracks")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Track filter diagnostics")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """按 99 百分位归一化显示背景。"""

    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0 or np.nanmax(arr) <= 0:
        return np.zeros(arr.shape, dtype=np.float32)
    positive = arr[arr > 0]
    hi = float(np.percentile(positive, 99)) if positive.size else float(np.nanmax(arr))
    return np.clip(arr / max(hi, 1e-6), 0, 1)


def _warn(warnings_list: list[str], message: str) -> None:
    """同时记录和发出 warning。"""

    warnings_list.append(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _load_array(path: str | Path | None) -> np.ndarray | None:
    """读取 npy 或常见图像文件；路径为空时返回 None。"""

    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    image = plt.imread(path)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return image.astype(np.float32)


def main() -> None:
    """命令行示例入口：读取 CSV/NPY 并运行 Step 5。"""

    parser = argparse.ArgumentParser(description="Step 5 glomeruli candidate mask and counting")
    parser.add_argument("--slow-tracks", required=True, help="慢速轨迹 CSV，例如 human_slow_tracks.csv 或 mouse_slow_tracks.csv")
    parser.add_argument("--fast-tracks", help="可选快速轨迹 CSV；没有 fast_density 时用于生成主血管排除 density")
    parser.add_argument("--slow-density", help="可选 slow_density.npy 或 PNG，用于 overlay")
    parser.add_argument("--fast-density", help="可选 fast_density.npy 或 PNG，用于主血管排除")
    parser.add_argument("--cortex-mask", help="可选 cortex_mask.npy，原始图像坐标 bool mask")
    parser.add_argument("--exclude-mask", help="可选 exclude_mask.npy，原始图像坐标 bool mask")
    parser.add_argument("--output-dir", default="step5_outputs")
    parser.add_argument("--config-json", help="可选 JSON 配置覆盖 Step5Config")
    args = parser.parse_args()

    config = None
    if args.config_json:
        with Path(args.config_json).open("r", encoding="utf-8") as f:
            config = json.load(f)

    result = run_step5_glomeruli_count(
        slow_tracks=args.slow_tracks,
        fast_tracks=args.fast_tracks,
        slow_density=_load_array(args.slow_density),
        fast_density=_load_array(args.fast_density),
        cortex_mask=_load_array(args.cortex_mask),
        exclude_mask=_load_array(args.exclude_mask),
        config=config,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
