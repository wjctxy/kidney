"""Visualization helpers for detection and tracking previews."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import step00_config as config


def make_tracks_preview(
    base_frames_path: str | Path,
    track_csvs: list[str | Path],
    output_mp4: str | Path,
    fps: float = config.FPS,
) -> None:
    frames = np.load(base_frames_path, mmap_mode="r")
    grouped = _load_tracks(track_csvs)
    count = min(frames.shape[0], config.VIDEO_PREVIEW_MAX_FRAMES)
    h, w = frames.shape[1:]
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return
    for frame_id in range(count):
        canvas = cv2.cvtColor(frames[frame_id], cv2.COLOR_GRAY2BGR)
        for channel, pts in grouped.get(frame_id, []):
            color = (255, 0, 255) if channel == "low" else (0, 255, 0)
            for x, y in pts:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 2, color, -1)
        writer.write(canvas)
    writer.release()


def _load_tracks(track_csvs: list[str | Path]) -> dict[int, list[tuple[str, list[tuple[float, float]]]]]:
    grouped: dict[int, list[tuple[str, list[tuple[float, float]]]]] = {}
    for path in track_csvs:
        path = Path(path)
        if not path.exists():
            continue
        by_track: dict[str, list[dict[str, float | str]]] = {}
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                by_track.setdefault(row["track_id"], []).append(
                    {
                        "frame": int(row["frame_id"]),
                        "x": float(row["x_pixel"]),
                        "y": float(row["y_pixel"]),
                        "channel": row["channel"],
                    }
                )
        for points in by_track.values():
            points.sort(key=lambda p: p["frame"])
            for i, point in enumerate(points):
                history = points[max(0, i - 8) : i + 1]
                grouped.setdefault(point["frame"], []).append(
                    (
                        str(point["channel"]),
                        [(float(p["x"]), float(p["y"])) for p in history],
                    )
                )
    return grouped
