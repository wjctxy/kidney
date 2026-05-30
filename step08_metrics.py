"""Track-level ULM metrics."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import step00_config as config


FIELDS = [
    "track_id",
    "channel",
    "point_count",
    "duration_s",
    "path_length",
    "straight_distance",
    "normalized_distance",
    "dwell_time_s",
    "mean_velocity",
    "max_velocity",
    "dispersion",
]


def compute_metrics(low_tracks_csv: str | Path, high_tracks_csv: str | Path, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(_metrics_for_file(low_tracks_csv))
    rows.extend(_metrics_for_file(high_tracks_csv))

    metrics_csv = output_dir / "track_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _write_summary(rows, output_dir / "summary_metrics.txt")
    _plot_hist(rows, "mean_velocity", output_dir / "velocity_hist.png", "Mean velocity")
    _plot_hist(rows, "dwell_time_s", output_dir / "dwell_time_hist.png", "Dwell time (s)")
    return metrics_csv


def _load_tracks(path: str | Path) -> dict[tuple[str, str], list[dict[str, float | str]]]:
    tracks: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    path = Path(path)
    if not path.exists():
        return tracks
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["channel"], row["track_id"])
            item = {
                "x": float(row["x_physical"]),
                "y": float(row["y_physical"]),
                "time": float(row["time_s"]),
                "velocity": float(row["velocity"]),
            }
            tracks.setdefault(key, []).append(item)
    return tracks


def _metrics_for_file(path: str | Path) -> list[dict[str, float | int | str]]:
    rows = []
    for (channel, track_id), points in _load_tracks(path).items():
        if len(points) < config.MIN_TRACK_LENGTH:
            continue
        distances = [
            math.hypot(points[i]["x"] - points[i - 1]["x"], points[i]["y"] - points[i - 1]["y"])
            for i in range(1, len(points))
        ]
        path_length = float(sum(distances))
        straight = float(math.hypot(points[-1]["x"] - points[0]["x"], points[-1]["y"] - points[0]["y"]))
        velocities = [float(p["velocity"]) for p in points[1:]]
        rows.append(
            {
                "track_id": track_id,
                "channel": channel,
                "point_count": len(points),
                "duration_s": points[-1]["time"] - points[0]["time"],
                "path_length": path_length,
                "straight_distance": straight,
                "normalized_distance": path_length / straight if straight > 0 else 0.0,
                "dwell_time_s": len(points) * config.FRAME_TIME_MS / 1000.0,
                "mean_velocity": float(np.mean(velocities)) if velocities else 0.0,
                "max_velocity": float(np.max(velocities)) if velocities else 0.0,
                "dispersion": _dispersion(points),
            }
        )
    return rows


def _dispersion(points: list[dict[str, float | str]]) -> float:
    angles = []
    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i - 1]["x"]
        dy = points[i]["y"] - points[i - 1]["y"]
        if dx == 0 and dy == 0:
            continue
        angles.append(math.degrees(math.atan2(dy, dx)))
    if not angles:
        return 0.0
    median = float(np.median(angles))
    same = sum(abs(((a - median + 180) % 360) - 180) <= config.DIRECTION_TOLERANCE_DEG for a in angles)
    return same / len(angles)


def _write_summary(rows: list[dict[str, float | int | str]], path: Path) -> None:
    lows = [r for r in rows if r["channel"] == "low"]
    highs = [r for r in rows if r["channel"] == "high"]
    lines = [
        "ULM SUMMARY METRICS",
        "=" * 80,
        f"total_tracks: {len(rows)}",
        f"low_speed_tracks: {len(lows)}",
        f"high_speed_tracks: {len(highs)}",
        f"mean_velocity: {_mean(rows, 'mean_velocity'):.6f}",
        f"mean_dwell_time_s: {_mean(rows, 'dwell_time_s'):.6f}",
        f"mean_normalized_distance: {_mean(rows, 'normalized_distance'):.6f}",
        f"mean_dispersion: {_mean(rows, 'dispersion'):.6f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return float(np.mean(values)) if values else 0.0


def _plot_hist(rows: list[dict[str, float | int | str]], key: str, path: Path, title: str) -> None:
    values = [float(r[key]) for r in rows]
    plt.figure(figsize=(6, 4))
    if values:
        plt.hist(values, bins=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
