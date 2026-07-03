"""检测点追踪、速度分组、密度图和指标计算。

本模块是人类链路和小鼠链路的共享第三步核心逻辑。前两步只负责产生
detections.csv；这里统一把检测点变成轨迹，再根据轨迹速度划分高/低速。
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm

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
    "mean_displacement_px_per_frame",
    "track_enclosed_area_pixels",
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
    killed: bool = False                # True 表示该轨迹因 missing 超限被杀死（Akebia unmatched source）
    started_from_unmatched: bool = False  # True 表示该轨迹始于匈牙利未匹配的孤儿点（Akebia unmatched target）

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


def write_frame_detections(
    n_frames: int,
    detector: Callable[[int], list[dict[str, float | int | str]]],
    path: str | Path,
    desc: str = "detecting",
) -> tuple[Path, int]:
    """逐帧调用 detector，汇总检测点并写出统一 detections.csv。"""

    rows: list[dict[str, float | int | str]] = []
    for frame_id in tqdm(range(n_frames), desc=desc, unit="frame"):
        rows.extend(detector(frame_id))
    return write_detections(rows, path), len(rows)


def track_detections(
    detections_csv: str | Path,
    metadata: dict,
    output_dir: str | Path,
    prefix: str,
    max_frame_displacement_px: float | None = None,
    min_track_length: int | None = None,
    max_missing_frames: int | None = None,
    max_gap_closing_frames: int | None = None,
) -> Path:
    """只执行匈牙利轨迹追踪，写出 tracks.csv 和 tracking_summary.txt。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections = _load_detections(detections_csv)
    tracks = _track_detections(
        detections,
        metadata,
        max_frame_displacement_px=max_frame_displacement_px,
        min_track_length=min_track_length,
        max_missing_frames=max_missing_frames,
        max_gap_closing_frames=max_gap_closing_frames,
    )
    tracks_csv = output_dir / f"{prefix}_tracks.csv"
    _write_tracks(tracks, tracks_csv)
    _write_tracking_summary(
        tracks,
        output_dir / "tracking_summary.txt",
        detections,
        min_track_length=min_track_length,
        max_frame_displacement_px=max_frame_displacement_px,
        max_missing_frames=max_missing_frames,
    )
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


def reconstruct_profile_density_and_metrics(
    rapid_tracks_csv: str | Path,
    slow_tracks_csv: str | Path,
    metadata: dict,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """Akebia human 式重建：直接把 Rapid/Slow profile 当作两类轨迹。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rapid_tracks = _load_tracks_csv(rapid_tracks_csv)
    slow_tracks = _load_tracks_csv(slow_tracks_csv)
    all_tracks = rapid_tracks + slow_tracks

    rapid_csv = output_dir / f"{prefix}_rapid_tracks.csv"
    slow_csv = output_dir / f"{prefix}_slow_tracks.csv"
    metrics_csv = output_dir / f"{prefix}_metrics.csv"
    summary_txt = output_dir / f"{prefix}_summary.txt"

    _write_tracks(rapid_tracks, rapid_csv)
    _write_tracks(slow_tracks, slow_csv)

    shape = (int(metadata["height"]), int(metadata["width"]))
    density_paths = _write_profile_density_maps(all_tracks, rapid_tracks, slow_tracks, shape, output_dir, prefix)

    rapid_metrics = _compute_metrics(rapid_tracks)
    for row in rapid_metrics:
        row["speed_group"] = "rapid_profile"
    slow_metrics = _compute_metrics(slow_tracks)
    for row in slow_metrics:
        row["speed_group"] = "slow_profile"
    metrics = rapid_metrics + slow_metrics
    _write_metrics(metrics, metrics_csv)
    _write_profile_summary(rapid_metrics, slow_metrics, summary_txt)

    return {
        "rapid_tracks": rapid_csv,
        "slow_tracks": slow_csv,
        "density_total": density_paths["total"],
        "density_rapid": density_paths["rapid"],
        "density_slow": density_paths["slow"],
        "density_profile_overlay": density_paths["profile_overlay"],
        "metrics": metrics_csv,
        "summary": summary_txt,
    }


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
    max_frame_displacement_px: float | None = None,
    min_track_length: int | None = None,
    max_missing_frames: int | None = None,
    max_gap_closing_frames: int | None = None,
) -> list[Track]:
    """维护 active/finished 轨迹集合，把逐帧检测点连接成完整轨迹。"""

    max_frame_displacement_px = float(max_frame_displacement_px if max_frame_displacement_px is not None else config.MAX_FRAME_DISPLACEMENT_PX)
    min_track_length = int(min_track_length if min_track_length is not None else config.MIN_TRACK_LENGTH)
    max_missing_frames = int(max_missing_frames if max_missing_frames is not None else config.MAX_MISSING_FRAMES)
    max_gap_closing_frames = int(max_gap_closing_frames if max_gap_closing_frames is not None else getattr(config, "MAX_GAP_CLOSING_FRAMES", 0))

    active: list[Track] = []
    finished: list[Track] = []
    next_id = 1

    for frame_id in tqdm(sorted(detections), desc="tracking", unit="frame"):
        points = detections[frame_id]
        assigned_tracks, assigned_points = _assign(active, points, max_frame_displacement_px)

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
            if track.missing <= max_missing_frames:
                still_active.append(track)
            else:
                track.killed = True  # Akebia 式：被匈牙利遗漏导致死亡
                finished.append(track)
        active = still_active

        for idx, point in enumerate(points):
            if idx in assigned_point_set:
                continue
            point["velocity"] = 0.0
            point["time_s"] = float(point["frame_id"]) / float(metadata["fps"])
            new_track = Track(next_id, [point])
            new_track.started_from_unmatched = True  # Akebia unmatched target
            active.append(new_track)
            next_id += 1

    finished.extend(active)

    # ── Akebia 式 gap-closing：只对被杀的尾 ↔ 孤儿头做跨帧桥接 ──
    if max_gap_closing_frames > 0:
        finished = _gap_closing_merge(finished, max_gap_closing_frames, max_frame_displacement_px)

    return [track for track in finished if len(track.points) >= min_track_length]


def _gap_closing_merge(tracks: list[Track], max_gap: int, max_dist: float) -> list[Track]:
    """Akebia 式 gap-closing：只对被匈牙利遗漏杀死的 track 尾端与孤儿起始 track 头端做最近邻桥接。"""

    if max_gap <= 0 or len(tracks) <= 1:
        return tracks

    # 收集被杀轨迹尾端 (unmatched sources) 和孤儿轨迹头端 (unmatched targets)
    killed_tails: list[dict] = []
    orphan_heads: list[dict] = []
    for idx, track in enumerate(tracks):
        if len(track.points) == 0:
            continue
        if track.killed:
            end_pt = track.points[-1]
            killed_tails.append({
                "track_idx": idx,
                "frame_id": int(end_pt["frame_id"]),
                "x_pixel": float(end_pt["x_pixel"]),
                "y_pixel": float(end_pt["y_pixel"]),
            })
        if track.started_from_unmatched:
            start_pt = track.points[0]
            orphan_heads.append({
                "track_idx": idx,
                "frame_id": int(start_pt["frame_id"]),
                "x_pixel": float(start_pt["x_pixel"]),
                "y_pixel": float(start_pt["y_pixel"]),
            })

    if not killed_tails or not orphan_heads:
        return tracks

    # 对 orphan_heads 按 frame_id 建索引，只查 in-range 帧，避免 O(killed×all_orphans)
    heads_by_frame: dict[int, list[dict]] = {}
    for head in orphan_heads:
        heads_by_frame.setdefault(head["frame_id"], []).append(head)

    merged = list(tracks)
    used_heads: set[int] = set()  # Akebia 式：已用头不得重复桥接

    for tail in sorted(killed_tails, key=lambda t: t["frame_id"]):
        candidates = []
        # 只查 tail.frame_id+1 到 tail.frame_id+max_gap 的帧
        for target_frame in range(tail["frame_id"] + 1, tail["frame_id"] + max_gap + 1):
            for head in heads_by_frame.get(target_frame, []):
                head_idx = head["track_idx"]
                if head_idx in used_heads:
                    continue
                dist = math.hypot(head["x_pixel"] - tail["x_pixel"], head["y_pixel"] - tail["y_pixel"])
                if dist <= max_dist:
                    candidates.append((dist, head))

        if not candidates:
            continue

        # 最近邻桥接
        candidates.sort(key=lambda c: c[0])
        best_head = candidates[0][1]
        head_idx = best_head["track_idx"]

        tail_track = merged[tail["track_idx"]]
        head_track = merged[head_idx]
        gap_frames = best_head["frame_id"] - tail["frame_id"] - 1

        # 缺失帧补 NaN 占位
        for offset in range(1, gap_frames + 1):
            tail_track.points.append({
                "frame_id": tail["frame_id"] + offset,
                "time_s": 0.0,
                "x_pixel": float('nan'),
                "y_pixel": float('nan'),
                "x_physical": float('nan'),
                "y_physical": float('nan'),
                "velocity": float('nan'),
            })

        tail_track.points.extend(head_track.points)
        head_track.points = []
        used_heads.add(head_idx)

    return [t for t in merged if len(t.points) > 0]


def _assign(
    active: list[Track],
    points: list[dict[str, float | int | str]],
    max_frame_displacement_px: float,
) -> tuple[list[int], list[int]]:
    """用距离矩阵和匈牙利算法匹配当前活跃轨迹与当前帧检测点。"""

    if not active or not points:
        return [], []

    last_xy = np.array([[float(t.last["x_pixel"]), float(t.last["y_pixel"])] for t in active])
    point_xy = np.array([[float(p["x_pixel"]), float(p["y_pixel"])] for p in points])

    # 匈牙利算法只处理相邻帧的最小代价匹配；最大位移约束负责过滤不合理连接。
    cost = cdist(last_xy, point_xy)
    cost[cost > max_frame_displacement_px] = 1e9
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
    """按肾小球尺度把轨迹分为低速组和高速组。

    平均位移超过一个肾小球直径的轨迹归为高速；平均位移不超过一个肾小球
    直径的慢速候选，还必须满足轨迹包围盒面积不超过 15 个像素。
    """

    low: list[Track] = []
    high: list[Track] = []
    for track in tracks:
        speed_group = _classify_track_speed_group(track)
        if speed_group == "high":
            high.append(track)
        elif speed_group == "low":
            low.append(track)
    return low, high


def _classify_track_speed_group(track: Track) -> str:
    """返回 high、low 或 rejected_slow_size，用于 Step04 密度图和指标。"""

    mean_displacement = _mean_displacement_px_per_frame(track.points)
    enclosed_area = _track_enclosed_area_pixels(track.points)
    if mean_displacement > config.GLOMERULUS_DIAMETER_PX:
        return "high"
    if enclosed_area > config.GLOMERULUS_MAX_ENCLOSED_AREA_PIXELS:
        return "rejected_slow_size"
    return "low"


def _mean_displacement_px_per_frame(points: list[dict[str, float | int | str]]) -> float:
    """计算轨迹相邻点平均像素位移，单位是 px/frame。"""

    displacements: list[float] = []
    for idx in range(1, len(points)):
        frame_delta = int(points[idx]["frame_id"]) - int(points[idx - 1]["frame_id"])
        if frame_delta <= 0:
            continue
        dx = float(points[idx]["x_pixel"]) - float(points[idx - 1]["x_pixel"])
        dy = float(points[idx]["y_pixel"]) - float(points[idx - 1]["y_pixel"])
        displacements.append(math.hypot(dx, dy) / frame_delta)
    return float(np.mean(displacements)) if displacements else 0.0


def _track_enclosed_area_pixels(points: list[dict[str, float | int | str]]) -> float:
    """计算轨迹点像素包围盒面积，用于剔除过大的慢速路径。"""

    xs = [float(point["x_pixel"]) for point in points]
    ys = [float(point["y_pixel"]) for point in points]
    if not xs or not ys:
        return 0.0
    width = max(xs) - min(xs) + 1.0
    height = max(ys) - min(ys) + 1.0
    return float(width * height)


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
    min_track_length: int | None = None,
    max_frame_displacement_px: float | None = None,
    max_missing_frames: int | None = None,
) -> None:
    """写出 Step 03 追踪摘要，不包含速度分组、密度图或指标统计。"""

    detection_count = sum(len(points) for points in detections.values())
    lines = [
        "ULM TRACKING SUMMARY",
        f"frames_with_detections: {len(detections)}",
        f"detections: {detection_count}",
        f"valid_tracks: {len(tracks)}",
        f"min_track_length: {int(min_track_length if min_track_length is not None else config.MIN_TRACK_LENGTH)}",
        f"max_frame_displacement_px: {float(max_frame_displacement_px if max_frame_displacement_px is not None else config.MAX_FRAME_DISPLACEMENT_PX)}",
        f"max_missing_frames: {int(max_missing_frames if max_missing_frames is not None else config.MAX_MISSING_FRAMES)}",
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


def _write_profile_density_maps(
    tracks: list[Track],
    rapid_tracks: list[Track],
    slow_tracks: list[Track],
    shape: tuple[int, int],
    output_dir: Path,
    prefix: str,
) -> dict[str, Path]:
    """写出 Akebia profile 密度图：rapid、slow、合计和合成图。"""

    total = _density_from_tracks(tracks, shape)
    rapid = _density_from_tracks(rapid_tracks, shape)
    slow = _density_from_tracks(slow_tracks, shape)
    total_path = output_dir / f"{prefix}_density_total.png"
    rapid_path = output_dir / f"{prefix}_density_rapid.png"
    slow_path = output_dir / f"{prefix}_density_slow.png"
    overlay_path = output_dir / f"{prefix}_density_profile_overlay.png"
    cv2.imwrite(str(total_path), _colorize(total, (255, 255, 255)))
    cv2.imwrite(str(rapid_path), _colorize(rapid, (0, 255, 0)))
    cv2.imwrite(str(slow_path), _colorize(slow, (255, 0, 255)))
    cv2.imwrite(str(overlay_path), _speed_overlay(slow, rapid))
    return {"total": total_path, "rapid": rapid_path, "slow": slow_path, "profile_overlay": overlay_path}


def _density_from_tracks(tracks: list[Track], shape: tuple[int, int]) -> np.ndarray:
    """把轨迹点累计到超分辨率网格中，形成血流密度矩阵（跳过 NaN 占位点）。"""

    h, w = shape
    scale = config.SUPER_RES_FACTOR
    density = np.zeros((h * scale, w * scale), dtype=np.float32)
    for track in tqdm(tracks, desc="density from tracks", unit="track"):
        for point in track.points:
            px = float(point["x_pixel"])
            py = float(point["y_pixel"])
            if np.isnan(px) or np.isnan(py):  # 跳过 gap-closing NaN 占位点
                continue
            x = int(round(px * scale))
            y = int(round(py * scale))
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

    for track in tqdm(tracks, desc="computing metrics", unit="track"):
        points = track.points
        distances = [
            math.hypot(
                float(points[i]["x_physical"]) - float(points[i - 1]["x_physical"]),
                float(points[i]["y_physical"]) - float(points[i - 1]["y_physical"]),
            )
            for i in range(1, len(points))
            if not (
                np.isnan(float(points[i]["x_physical"]))
                or np.isnan(float(points[i]["y_physical"]))
                or np.isnan(float(points[i - 1]["x_physical"]))
                or np.isnan(float(points[i - 1]["y_physical"]))
            )
        ]
        path_length = float(sum(distances))
        straight = float(
            math.hypot(
                float(points[-1]["x_physical"]) - float(points[0]["x_physical"]),
                float(points[-1]["y_physical"]) - float(points[0]["y_physical"]),
            )
        )
        velocities = [float(p["velocity"]) for p in points[1:]]
        mean_displacement = _mean_displacement_px_per_frame(points)
        enclosed_area = _track_enclosed_area_pixels(points)
        rows.append(
            {
                "track_id": track.track_id,
                "point_count": len(points),
                "duration_s": float(points[-1]["time_s"]) - float(points[0]["time_s"]),
                "path_length": path_length,
                "mean_displacement_px_per_frame": mean_displacement,
                "track_enclosed_area_pixels": enclosed_area,
                "straight_distance": straight,
                "normalized_distance": path_length / straight if straight > 0 else 0.0,
                "dwell_time_s": float(points[-1]["time_s"]) - float(points[0]["time_s"]),
                "mean_velocity": float(np.mean(velocities)) if velocities else 0.0,
                "max_velocity": float(np.max(velocities)) if velocities else 0.0,
                "dispersion": _dispersion(points),
                "speed_group": _classify_track_speed_group(track),
            }
        )
    return rows


def _dispersion(points: list[dict[str, float | int | str]]) -> float:
    """计算轨迹方向一致性，返回方向落在中位方向容差内的比例。"""

    angles: list[float] = []
    for idx in range(1, len(points)):
        dx = float(points[idx]["x_physical"]) - float(points[idx - 1]["x_physical"])
        dy = float(points[idx]["y_physical"]) - float(points[idx - 1]["y_physical"])
        if np.isnan(dx) or np.isnan(dy):  # 跳过 gap-closing NaN
            continue
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
    rejected = [row for row in rows if row["speed_group"] == "rejected_slow_size"]
    lines = [
        "ULM 轨迹指标摘要",
        f"total_tracks: {len(rows)}",
        f"low_speed_tracks: {len(low)}",
        f"high_speed_tracks: {len(high)}",
        f"rejected_slow_size_tracks: {len(rejected)}",
        f"low_speed_definition: mean_displacement_px_per_frame <= {config.GLOMERULUS_DIAMETER_PX}",
        f"high_speed_definition: mean_displacement_px_per_frame > {config.GLOMERULUS_DIAMETER_PX}",
        f"low_speed_max_enclosed_area_pixels: <= {config.GLOMERULUS_MAX_ENCLOSED_AREA_PIXELS}",
        f"mean_displacement_px_per_frame: {_mean(rows, 'mean_displacement_px_per_frame'):.6f}",
        f"mean_track_enclosed_area_pixels: {_mean(rows, 'track_enclosed_area_pixels'):.6f}",
        f"mean_velocity: {_mean(rows, 'mean_velocity'):.6f}",
        f"mean_normalized_distance: {_mean(rows, 'normalized_distance'):.6f}",
        f"mean_dwell_time_s: {_mean(rows, 'dwell_time_s'):.6f}",
        f"mean_dispersion: {_mean(rows, 'dispersion'):.6f}",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _write_profile_summary(
    rapid_rows: list[dict[str, float | int | str]],
    slow_rows: list[dict[str, float | int | str]],
    path: str | Path,
) -> None:
    """写出 Akebia human Rapid/Slow profile 摘要。"""

    rows = rapid_rows + slow_rows
    lines = [
        "ULM Akebia-profile 轨迹指标摘要",
        f"total_tracks: {len(rows)}",
        f"rapid_profile_tracks: {len(rapid_rows)}",
        f"slow_profile_tracks: {len(slow_rows)}",
        "rapid_profile_definition: Akebia human Rapid parameters, bandpass [1, 5.5] Hz, maxLinkingDistance=15, minLength=5",
        "slow_profile_definition: Human Slow parameters, bandpass [0.05, 1.0] Hz with cortex-limited detections when a cortex mask is available, maxLinkingDistance=4, minLength=10",
        f"rapid_mean_displacement_px_per_frame: {_mean(rapid_rows, 'mean_displacement_px_per_frame'):.6f}",
        f"slow_mean_displacement_px_per_frame: {_mean(slow_rows, 'mean_displacement_px_per_frame'):.6f}",
        f"rapid_mean_velocity: {_mean(rapid_rows, 'mean_velocity'):.6f}",
        f"slow_mean_velocity: {_mean(slow_rows, 'mean_velocity'):.6f}",
        f"mean_displacement_px_per_frame: {_mean(rows, 'mean_displacement_px_per_frame'):.6f}",
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
