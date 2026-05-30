"""Microbubble candidate detection by thresholding and connected components."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import step00_config as config


CSV_FIELDS = [
    "frame_id",
    "time_s",
    "x_pixel",
    "y_pixel",
    "x_physical",
    "y_physical",
    "intensity",
    "area",
    "channel",
]


def detect_bubbles(
    frames_path: str | Path,
    output_csv: str | Path,
    channel: str,
    frame_time_ms: float = config.FRAME_TIME_MS,
) -> Path:
    frames = np.load(frames_path, mmap_mode="r")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for frame_id in range(frames.shape[0]):
            frame = frames[frame_id]
            for row in _detect_frame(frame, frame_id, channel, frame_time_ms):
                writer.writerow(row)
    return output_csv


def _detect_frame(frame: np.ndarray, frame_id: int, channel: str, frame_time_ms: float) -> list[dict[str, float | int | str]]:
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    threshold = float(blurred.mean() + config.THRESHOLD_STD_FACTOR * blurred.std())
    threshold = max(threshold, 8.0)
    binary = (blurred >= threshold).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    rows = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < config.MIN_BLOB_AREA or area > config.MAX_BLOB_AREA:
            continue
        x, y = centroids[label]
        xi = int(round(x))
        yi = int(round(y))
        if yi < 0 or yi >= frame.shape[0] or xi < 0 or xi >= frame.shape[1]:
            continue
        rows.append(
            {
                "frame_id": frame_id,
                "time_s": frame_id * frame_time_ms / 1000.0,
                "x_pixel": float(x),
                "y_pixel": float(y),
                "x_physical": float(x) * config.PHYSICAL_DELTA_X,
                "y_physical": float(y) * config.PHYSICAL_DELTA_Y,
                "intensity": int(frame[yi, xi]),
                "area": area,
                "channel": channel,
            }
        )
    return rows


def make_detection_preview(frames_path: str | Path, points_csv: str | Path, output_mp4: str | Path, fps: float) -> None:
    frames = np.load(frames_path, mmap_mode="r")
    points = _load_points_by_frame(points_csv)
    count = min(frames.shape[0], config.VIDEO_PREVIEW_MAX_FRAMES)
    h, w = frames.shape[1:]
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return
    for idx in range(count):
        canvas = cv2.cvtColor(frames[idx], cv2.COLOR_GRAY2BGR)
        for x, y in points.get(idx, []):
            cv2.circle(canvas, (int(round(x)), int(round(y))), 3, (0, 0, 255), 1)
        writer.write(canvas)
    writer.release()


def _load_points_by_frame(points_csv: str | Path) -> dict[int, list[tuple[float, float]]]:
    grouped: dict[int, list[tuple[float, float]]] = {}
    with Path(points_csv).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_id = int(row["frame_id"])
            grouped.setdefault(frame_id, []).append((float(row["x_pixel"]), float(row["y_pixel"])))
    return grouped
