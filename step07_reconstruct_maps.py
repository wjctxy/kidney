"""Reconstruct low/high-speed microbubble trajectory density maps."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import step00_config as config


def reconstruct_density_maps(
    low_tracks_csv: str | Path,
    high_tracks_csv: str | Path,
    output_dir: str | Path,
    shape: tuple[int, int],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    low = _density_from_tracks(low_tracks_csv, shape)
    high = _density_from_tracks(high_tracks_csv, shape)

    cv2.imwrite(str(output_dir / "low_speed_density.png"), _colorize(low, (255, 0, 255)))
    cv2.imwrite(str(output_dir / "high_speed_density.png"), _colorize(high, (0, 255, 0)))
    combined = np.maximum(_colorize(low, (255, 0, 255)), _colorize(high, (0, 255, 0)))
    cv2.imwrite(str(output_dir / "combined_density.png"), combined)
    cv2.imwrite(str(output_dir / "tracks_overlay.png"), combined)


def _density_from_tracks(tracks_csv: str | Path, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    scale = config.SUPER_RES_FACTOR
    density = np.zeros((h * scale, w * scale), dtype=np.float32)
    path = Path(tracks_csv)
    if not path.exists():
        return density
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x = int(round(float(row["x_pixel"]) * scale))
            y = int(round(float(row["y_pixel"]) * scale))
            if 0 <= y < density.shape[0] and 0 <= x < density.shape[1]:
                density[y, x] += 1.0
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=1.2 * scale)
    return density


def _colorize(density: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    if density.max() <= 0:
        norm = np.zeros(density.shape, dtype=np.uint8)
    else:
        norm = np.clip(density / np.percentile(density[density > 0], 99) * 255, 0, 255).astype(np.uint8)
    bgr = np.zeros((*density.shape, 3), dtype=np.uint8)
    bgr[..., 0] = norm * rgb[2] // 255
    bgr[..., 1] = norm * rgb[1] // 255
    bgr[..., 2] = norm * rgb[0] // 255
    return bgr
