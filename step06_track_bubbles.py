"""Track microbubble detections with nearest-neighbor/Hungarian assignment."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import step00_config as config


TRACK_FIELDS = [
    "track_id",
    "frame_id",
    "time_s",
    "x_pixel",
    "y_pixel",
    "x_physical",
    "y_physical",
    "velocity",
    "channel",
]


@dataclass
class Track:
    track_id: int
    channel: str
    points: list[dict[str, float | int | str]] = field(default_factory=list)
    missing: int = 0

    @property
    def last(self) -> dict[str, float | int | str]:
        return self.points[-1]


def track_bubbles(points_csv: str | Path, output_csv: str | Path, channel: str) -> Path:
    detections = _load_detections(points_csv)
    active: list[Track] = []
    finished: list[Track] = []
    next_id = 1

    for frame_id in sorted(detections):
        points = detections[frame_id]
        assigned_tracks, assigned_points = _assign(active, points)

        for track_idx, point_idx in zip(assigned_tracks, assigned_points):
            track = active[track_idx]
            point = points[point_idx]
            point["velocity"] = _velocity(track.last, point)
            track.points.append(point)
            track.missing = 0

        assigned_track_set = set(assigned_tracks)
        assigned_point_set = set(assigned_points)
        still_active = []
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
            active.append(Track(next_id, channel, [point]))
            next_id += 1

    finished.extend(active)
    valid = [t for t in finished if len(t.points) >= config.MIN_TRACK_LENGTH]
    return _write_tracks(valid, output_csv)


def _load_detections(points_csv: str | Path) -> dict[int, list[dict[str, float | int | str]]]:
    grouped: dict[int, list[dict[str, float | int | str]]] = {}
    with Path(points_csv).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = {
                "frame_id": int(row["frame_id"]),
                "time_s": float(row["time_s"]),
                "x_pixel": float(row["x_pixel"]),
                "y_pixel": float(row["y_pixel"]),
                "x_physical": float(row["x_physical"]),
                "y_physical": float(row["y_physical"]),
                "channel": row["channel"],
            }
            grouped.setdefault(item["frame_id"], []).append(item)
    return grouped


def _assign(active: list[Track], points: list[dict[str, float | int | str]]) -> tuple[list[int], list[int]]:
    if not active or not points:
        return [], []
    cost = np.full((len(active), len(points)), 1e9, dtype=np.float32)
    for i, track in enumerate(active):
        for j, point in enumerate(points):
            dist = math.hypot(
                float(track.last["x_pixel"]) - float(point["x_pixel"]),
                float(track.last["y_pixel"]) - float(point["y_pixel"]),
            )
            if dist <= config.MAX_FRAME_DISPLACEMENT_PX:
                cost[i, j] = dist
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
    except Exception:
        rows, cols = _greedy_assignment(cost)
    keep_rows = []
    keep_cols = []
    for r, c in zip(rows, cols):
        if cost[r, c] < 1e9:
            keep_rows.append(int(r))
            keep_cols.append(int(c))
    return keep_rows, keep_cols


def _greedy_assignment(cost: np.ndarray) -> tuple[list[int], list[int]]:
    rows = []
    cols = []
    used_r = set()
    used_c = set()
    for r, c in sorted(np.ndindex(cost.shape), key=lambda rc: cost[rc]):
        if cost[r, c] >= 1e9 or r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        rows.append(r)
        cols.append(c)
    return rows, cols


def _velocity(p0: dict[str, float | int | str], p1: dict[str, float | int | str]) -> float:
    dt = float(p1["time_s"]) - float(p0["time_s"])
    if dt <= 0:
        return 0.0
    dist = math.hypot(
        float(p1["x_physical"]) - float(p0["x_physical"]),
        float(p1["y_physical"]) - float(p0["y_physical"]),
    )
    return dist / dt


def _write_tracks(tracks: list[Track], output_csv: str | Path) -> Path:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACK_FIELDS)
        writer.writeheader()
        for track in tracks:
            for point in track.points:
                row = {field: point.get(field, "") for field in TRACK_FIELDS}
                row["track_id"] = track.track_id
                writer.writerow(row)
    return output_csv
