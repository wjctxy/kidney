"""检测点追踪、速度分组、密度图和指标计算。

本模块是人类链路和小鼠链路的共享第三步核心逻辑。前两步只负责产生
detections.csv；这里统一把检测点变成轨迹，再根据轨迹速度划分高/低速。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import ulm_config as config


DETECTION_FIELDS = [
    "frame_id",
    "x_pixel",
    "y_pixel",
    "x_physical",
    "y_physical",
    "response_intensity",
    "signed_intensity",
    "intensity",
    "method",
]

TRACK_FIELDS = [
    "track_id",
    "frame_id",
    "time_s",
    "x_pixel",
    "y_pixel",
    "x_physical",
    "y_physical",
    "velocity",
]

METRIC_FIELDS = [
    "track_id",
    "point_count",
    "duration_s",
    "path_length",
    "straight_distance",
    "normalized_distance",
    "dwell_time_s",
    "mean_velocity",
    "max_velocity",
    "dispersion",
    "speed_group",
]


@dataclass
class Track:
    """保存一条微泡轨迹的状态，包括轨迹点和连续丢失帧数。"""

    track_id: int
    points: list[dict[str, float | int | str]] = field(default_factory=list)
    missing: int = 0

    @property
    def last(self) -> dict[str, float | int | str]:
        """返回轨迹最后一个点，用于和下一帧候选点计算匹配距离。"""

        return self.points[-1]


def write_detections(rows: list[dict[str, float | int | str]], path: str | Path) -> Path:
    """把检测点列表写成统一 detections.csv 格式。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def track_detections(
    detections_csv: str | Path,
    metadata: dict,
    output_dir: str | Path,
    prefix: str,
) -> Path:
    """只执行匈牙利轨迹追踪，写出 tracks.csv 和 tracking_summary.txt。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections = _load_detections(detections_csv)
    tracks = _track_detections(detections, metadata)
    tracks_csv = output_dir / f"{prefix}_tracks.csv"
    _write_tracks(tracks, tracks_csv)
    _write_tracking_summary(tracks, output_dir / "tracking_summary.txt", detections)
    return tracks_csv


def reconstruct_density_and_metrics(
    tracks_csv: str | Path,
    metadata: dict,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """读取 tracks.csv，执行速度分组、密度图重建、指标计算和 summary 输出。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = _load_tracks_csv(tracks_csv)
    low_tracks, high_tracks = _split_tracks_by_speed(tracks)
    low_csv = output_dir / f"{prefix}_low_speed_tracks.csv"
    high_csv = output_dir / f"{prefix}_high_speed_tracks.csv"
    metrics_csv = output_dir / f"{prefix}_metrics.csv"
    summary_txt = output_dir / f"{prefix}_summary.txt"

    _write_tracks(low_tracks, low_csv)
    _write_tracks(high_tracks, high_csv)
    shape = (int(metadata["height"]), int(metadata["width"]))
    density_paths = _write_density_maps(tracks, low_tracks, high_tracks, shape, output_dir, prefix)
    metrics = _compute_metrics(tracks)
    _write_metrics(metrics, metrics_csv)
    _write_summary(metrics, summary_txt)

    return {
        "low_tracks": low_csv,
        "high_tracks": high_csv,
        "density_total": density_paths["total"],
        "density_low_speed": density_paths["low"],
        "density_high_speed": density_paths["high"],
        "density_speed_overlay": density_paths["speed_overlay"],
        "metrics": metrics_csv,
        "summary": summary_txt,
    }


def track_and_reconstruct(
    detections_csv: str | Path,
    metadata: dict,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """兼容旧入口：从 detections.csv 追踪后继续生成密度图和指标。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_csv = track_detections(detections_csv, metadata, output_dir, prefix)
    outputs = reconstruct_density_and_metrics(tracks_csv, metadata, output_dir, prefix)
    return {"tracks": tracks_csv, **outputs}


def _load_detections(path: str | Path) -> dict[int, list[dict[str, float | int | str]]]:
    """读取 detections.csv，并按 frame_id 分组，便于逐帧追踪。"""

    grouped: dict[int, list[dict[str, float | int | str]]] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = {
                "frame_id": int(row["frame_id"]),
                "x_pixel": float(row["x_pixel"]),
                "y_pixel": float(row["y_pixel"]),
                "x_physical": float(row["x_physical"]),
                "y_physical": float(row["y_physical"]),
                "intensity": float(row.get("response_intensity") or row.get("intensity") or 0.0),
                "method": row["method"],
            }
            grouped.setdefault(int(item["frame_id"]), []).append(item)
    return grouped


def _track_detections(
    detections: dict[int, list[dict[str, float | int | str]]],
    metadata: dict,
) -> list[Track]:
    """维护 active/finished 轨迹集合，把逐帧检测点连接成完整轨迹。"""

    active: list[Track] = []
    finished: list[Track] = []
    next_id = 1

    for frame_id in sorted(detections):
        points = detections[frame_id]
        assigned_tracks, assigned_points = _assign(active, points)

        for track_idx, point_idx in zip(assigned_tracks, assigned_points):
            track = active[track_idx]
            point = points[point_idx]
            point["time_s"] = float(point["frame_id"]) / float(metadata["fps"])
            point["velocity"] = _velocity(track.last, point)
            track.points.append(point)
            track.missing = 0

        assigned_track_set = set(assigned_tracks)
        assigned_point_set = set(assigned_points)
        still_active: list[Track] = []
        for idx, track in enumerate(active):
            if idx not in assigned_track_set:
                track.missing += 1
            if track.missing <= config.MAX_MISSING_FRAMES:
                still_active.append(track)
            else:
                finished.append(track)
        active = still_active

        for idx, point in enumerate(points):
            if idx in assigned_point_set:
                continue
            point["velocity"] = 0.0
            point["time_s"] = float(point["frame_id"]) / float(metadata["fps"])
            active.append(Track(next_id, [point]))
            next_id += 1

    finished.extend(active)
    return [track for track in finished if len(track.points) >= config.MIN_TRACK_LENGTH]


def _assign(active: list[Track], points: list[dict[str, float | int | str]]) -> tuple[list[int], list[int]]:
    """用距离矩阵和匈牙利算法匹配当前活跃轨迹与当前帧检测点。"""

    if not active or not points:
        return [], []

    last_xy = np.array([[float(t.last["x_pixel"]), float(t.last["y_pixel"])] for t in active])
    point_xy = np.array([[float(p["x_pixel"]), float(p["y_pixel"])] for p in points])

    # 匈牙利算法只处理相邻帧的最小代价匹配；最大位移约束负责过滤不合理连接。
    cost = cdist(last_xy, point_xy)
    cost[cost > config.MAX_FRAME_DISPLACEMENT_PX] = 1e9
    rows, cols = linear_sum_assignment(cost)

    keep_rows: list[int] = []
    keep_cols: list[int] = []
    for row, col in zip(rows, cols):
        if cost[row, col] < 1e9:
            keep_rows.append(int(row))
            keep_cols.append(int(col))
    return keep_rows, keep_cols


def _velocity(p0: dict[str, float | int | str], p1: dict[str, float | int | str]) -> float:
    """根据相邻轨迹点的物理位移和时间间隔计算速度。"""

    dt = float(p1["time_s"]) - float(p0["time_s"]) if "time_s" in p0 else 0.0
    if dt <= 0:
        return 0.0
    dx = float(p1["x_physical"]) - float(p0["x_physical"])
    dy = float(p1["y_physical"]) - float(p0["y_physical"])
    return math.hypot(dx, dy) / dt


def _write_tracks(tracks: list[Track], path: str | Path) -> None:
    """把轨迹对象展开为每行一个轨迹点的 tracks.csv。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACK_FIELDS)
        writer.writeheader()
        for track in tracks:
            for point in track.points:
                row = {field: point.get(field, "") for field in TRACK_FIELDS}
                row["track_id"] = track.track_id
                writer.writerow(row)


def _split_tracks_by_speed(tracks: list[Track]) -> tuple[list[Track], list[Track]]:
    """按轨迹平均速度把轨迹分为低速组和高速组。"""

    low: list[Track] = []
    high: list[Track] = []
    for track in tracks:
        velocities = [float(p["velocity"]) for p in track.points[1:]]
        mean_velocity = float(np.mean(velocities)) if velocities else 0.0
        if mean_velocity >= config.SPEED_THRESHOLD:
            high.append(track)
        else:
            low.append(track)
    return low, high


def _load_tracks_csv(path: str | Path) -> list[Track]:
    """从 tracks.csv 读取轨迹点，并恢复为 Track 对象列表。"""

    grouped: dict[int, Track] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            track_id = int(row["track_id"])
            point = {
                "frame_id": int(row["frame_id"]),
                "time_s": float(row["time_s"]),
                "x_pixel": float(row["x_pixel"]),
                "y_pixel": float(row["y_pixel"]),
                "x_physical": float(row["x_physical"]),
                "y_physical": float(row["y_physical"]),
                "velocity": float(row["velocity"]),
            }
            grouped.setdefault(track_id, Track(track_id)).points.append(point)
    return list(grouped.values())


def _write_tracking_summary(
    tracks: list[Track],
    path: str | Path,
    detections: dict[int, list[dict[str, float | int | str]]],
) -> None:
    """写出 Step 03 追踪摘要，不包含速度分组、密度图或指标统计。"""

    detection_count = sum(len(points) for points in detections.values())
    lines = [
        "ULM TRACKING SUMMARY",
        f"frames_with_detections: {len(detections)}",
        f"detections: {detection_count}",
        f"valid_tracks: {len(tracks)}",
        f"min_track_length: {config.MIN_TRACK_LENGTH}",
        f"max_frame_displacement_px: {config.MAX_FRAME_DISPLACEMENT_PX}",
        f"max_missing_frames: {config.MAX_MISSING_FRAMES}",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_density_maps(
    tracks: list[Track],
    low_tracks: list[Track],
    high_tracks: list[Track],
    shape: tuple[int, int],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """分别写出总密度图、低速密度图和高速密度图。"""

    total = _density_from_tracks(tracks, shape)
    low = _density_from_tracks(low_tracks, shape)
    high = _density_from_tracks(high_tracks, shape)
    total_path = output_dir / f"{prefix}_density_total.png"
    low_path = output_dir / f"{prefix}_density_low_speed.png"
    high_path = output_dir / f"{prefix}_density_high_speed.png"
    overlay_path = output_dir / f"{prefix}_density_speed_overlay.png"
    cv2.imwrite(str(total_path), _colorize(total, (255, 255, 255)))
    cv2.imwrite(str(low_path), _colorize(low, (255, 0, 255)))
    cv2.imwrite(str(high_path), _colorize(high, (0, 255, 0)))
    cv2.imwrite(str(overlay_path), _speed_overlay(low, high))
    return {"total": total_path, "low": low_path, "high": high_path, "speed_overlay": overlay_path}


def _density_from_tracks(tracks: list[Track], shape: tuple[int, int]) -> np.ndarray:
    """把轨迹点累计到超分辨率网格中，形成血流密度矩阵。"""

    h, w = shape
    scale = config.SUPER_RES_FACTOR
    density = np.zeros((h * scale, w * scale), dtype=np.float32)
    for track in tracks:
        for point in track.points:
            x = int(round(float(point["x_pixel"]) * scale))
            y = int(round(float(point["y_pixel"]) * scale))
            if 0 <= y < density.shape[0] and 0 <= x < density.shape[1]:
                density[y, x] += 1.0
    if density.max() > 0:
        density = cv2.GaussianBlur(density, (0, 0), sigmaX=1.2 * scale)
    return density


def _colorize(density: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """把单通道密度矩阵按指定 RGB 颜色映射为 OpenCV 可写的 BGR 图像。"""

    norm = _normalize_density_uint8(density).astype(np.float32) / 255.0
    bgr = np.zeros((*density.shape, 3), dtype=np.float32)
    bgr[..., 0] = norm * float(rgb[2])
    bgr[..., 1] = norm * float(rgb[1])
    bgr[..., 2] = norm * float(rgb[0])
    return np.clip(bgr, 0, 255).astype(np.uint8)


def _speed_overlay(low_density: np.ndarray, high_density: np.ndarray) -> np.ndarray:
    """生成黑底速度合成图：低速为紫色，高速为绿色。"""

    low = _normalize_density_uint8(low_density).astype(np.float32) / 255.0
    high = _normalize_density_uint8(high_density).astype(np.float32) / 255.0
    bgr = np.zeros((*low_density.shape, 3), dtype=np.float32)
    bgr[..., 0] = low * 255.0
    bgr[..., 1] = high * 255.0
    bgr[..., 2] = low * 255.0
    return np.clip(bgr, 0, 255).astype(np.uint8)


def _normalize_density_uint8(density: np.ndarray) -> np.ndarray:
    """把密度矩阵按非零像素 99 百分位归一化到 uint8。"""

    if density.max() <= 0:
        return np.zeros(density.shape, dtype=np.uint8)
    positive = density[density > 0]
    hi = float(np.percentile(positive, 99)) if positive.size else 1.0
    return np.clip(density / max(hi, 1e-6) * 255.0, 0, 255).astype(np.uint8)


def _compute_metrics(tracks: list[Track]) -> list[dict[str, float | int | str]]:
    """计算每条轨迹的长度、归一化距离、滞留时长、速度和离散度。"""

    rows: list[dict[str, float | int | str]] = []
    low_tracks, high_tracks = _split_tracks_by_speed(tracks)
    speed_group = {id(t): "low" for t in low_tracks}
    speed_group.update({id(t): "high" for t in high_tracks})

    for track in tracks:
        points = track.points
        distances = [
            math.hypot(
                float(points[i]["x_physical"]) - float(points[i - 1]["x_physical"]),
                float(points[i]["y_physical"]) - float(points[i - 1]["y_physical"]),
            )
            for i in range(1, len(points))
        ]
        path_length = float(sum(distances))
        straight = float(
            math.hypot(
                float(points[-1]["x_physical"]) - float(points[0]["x_physical"]),
                float(points[-1]["y_physical"]) - float(points[0]["y_physical"]),
            )
        )
        velocities = [float(p["velocity"]) for p in points[1:]]
        rows.append(
            {
                "track_id": track.track_id,
                "point_count": len(points),
                "duration_s": float(points[-1]["time_s"]) - float(points[0]["time_s"]),
                "path_length": path_length,
                "straight_distance": straight,
                "normalized_distance": path_length / straight if straight > 0 else 0.0,
                "dwell_time_s": float(points[-1]["time_s"]) - float(points[0]["time_s"]),
                "mean_velocity": float(np.mean(velocities)) if velocities else 0.0,
                "max_velocity": float(np.max(velocities)) if velocities else 0.0,
                "dispersion": _dispersion(points),
                "speed_group": speed_group[id(track)],
            }
        )
    return rows


def _dispersion(points: list[dict[str, float | int | str]]) -> float:
    """计算轨迹方向一致性，返回方向落在中位方向容差内的比例。"""

    angles: list[float] = []
    for idx in range(1, len(points)):
        dx = float(points[idx]["x_physical"]) - float(points[idx - 1]["x_physical"])
        dy = float(points[idx]["y_physical"]) - float(points[idx - 1]["y_physical"])
        if dx == 0 and dy == 0:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)))
    if not angles:
        return 0.0
    median = float(np.median(angles))
    same_direction = sum(
        abs(((angle - median + 180.0) % 360.0) - 180.0) <= config.DIRECTION_TOLERANCE_DEG
        for angle in angles
    )
    return same_direction / len(angles)


def _write_metrics(rows: list[dict[str, float | int | str]], path: str | Path) -> None:
    """把逐轨迹指标写入 metrics.csv。"""

    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(rows: list[dict[str, float | int | str]], path: str | Path) -> None:
    """把关键总体指标汇总成便于人工查看的 summary.txt。"""

    low = [row for row in rows if row["speed_group"] == "low"]
    high = [row for row in rows if row["speed_group"] == "high"]
    lines = [
        "ULM 轨迹指标摘要",
        f"total_tracks: {len(rows)}",
        f"low_speed_tracks: {len(low)}",
        f"high_speed_tracks: {len(high)}",
        f"mean_velocity: {_mean(rows, 'mean_velocity'):.6f}",
        f"mean_normalized_distance: {_mean(rows, 'normalized_distance'):.6f}",
        f"mean_dwell_time_s: {_mean(rows, 'dwell_time_s'):.6f}",
        f"mean_dispersion: {_mean(rows, 'dispersion'):.6f}",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    """计算某个指标字段的均值；没有数据时返回 0。"""

    values = [float(row[key]) for row in rows]
    return float(np.mean(values)) if values else 0.0
