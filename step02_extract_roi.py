"""Extract CEUS ROI frames from the multi-frame DICOM."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import step00_config as config
from step01_dicom_io import DicomFrameReader


def crop_roi(frame: np.ndarray) -> np.ndarray:
    return frame[config.ROI_Y0 : config.ROI_Y1, config.ROI_X0 : config.ROI_X1]


def _write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return
    for frame in frames:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()


def extract_roi_previews(
    dicom_path: str | Path,
    output_dir: str | Path,
    max_preview_frames: int = config.VIDEO_PREVIEW_MAX_FRAMES,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reader = DicomFrameReader(dicom_path)

    preview_indices = [
        i for i in config.PREVIEW_FRAMES if 0 <= i < reader.frame_count
    ]
    for idx in preview_indices:
        roi = crop_roi(reader.read_frame(idx))
        cv2.imwrite(str(out / f"frame_{idx:04d}.png"), cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))

    video_frames = []
    count = min(reader.frame_count, max_preview_frames)
    for idx in range(count):
        video_frames.append(crop_roi(reader.read_frame(idx)))
    _write_video(video_frames, out / "roi_preview.mp4", reader.fps)
