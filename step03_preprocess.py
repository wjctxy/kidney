"""ROI grayscale conversion, background subtraction, and denoising."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.lib.format import open_memmap

import step00_config as config
from step01_dicom_io import DicomFrameReader
from step02_extract_roi import crop_roi


def rgb_to_gray01(frame_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


def _save_gray_png(path: Path, image01: np.ndarray) -> None:
    image8 = np.clip(image01 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), image8)


def preprocess_frames(
    dicom_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    max_frames: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    reader = DicomFrameReader(dicom_path)
    frame_count = min(reader.frame_count, max_frames or reader.frame_count)
    h = config.ROI_Y1 - config.ROI_Y0
    w = config.ROI_X1 - config.ROI_X0

    raw = open_memmap(cache_dir / "gray_roi_frames.npy", mode="w+", dtype=np.uint8, shape=(frame_count, h, w))
    for idx in range(frame_count):
        gray = rgb_to_gray01(crop_roi(reader.read_frame(idx)))
        raw[idx] = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    raw.flush()

    raw_read = np.load(cache_dir / "gray_roi_frames.npy", mmap_mode="r")
    background = np.median(raw_read.astype(np.float32) / 255.0, axis=0)
    _save_gray_png(output_dir / "background.png", background)

    pre = open_memmap(cache_dir / "preprocessed_frames.npy", mode="w+", dtype=np.uint8, shape=(frame_count, h, w))
    preview = []
    for idx in range(frame_count):
        frame = raw_read[idx].astype(np.float32) / 255.0
        enhanced = np.clip(frame - background, 0.0, 1.0)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        pre[idx] = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
        if idx < min(frame_count, config.VIDEO_PREVIEW_MAX_FRAMES):
            preview.append(pre[idx].copy())
    pre.flush()
    _write_gray_video(preview, output_dir / "preprocessed_preview.mp4", reader.fps)
    return cache_dir / "preprocessed_frames.npy"


def _write_gray_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
