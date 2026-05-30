"""Temporal filtering for low-speed and high-speed microbubble signals."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.lib.format import open_memmap

import step00_config as config


def _butter_bandpass_filter(data: np.ndarray, band: tuple[float, float], fps: float) -> np.ndarray:
    try:
        from scipy.signal import butter, filtfilt

        nyq = fps * 0.5
        low = max(0.001, band[0] / nyq)
        high = min(0.999, band[1] / nyq)
        b, a = butter(2, (low, high), btype="bandpass")
        return filtfilt(b, a, data, axis=0).astype(np.float32)
    except Exception:
        return _fft_bandpass_filter(data, band, fps)


def _fft_bandpass_filter(data: np.ndarray, band: tuple[float, float], fps: float) -> np.ndarray:
    freq = np.fft.rfftfreq(data.shape[0], d=1.0 / fps)
    spectrum = np.fft.rfft(data, axis=0)
    mask = (freq >= band[0]) & (freq <= band[1])
    spectrum[~mask] = 0
    return np.fft.irfft(spectrum, n=data.shape[0], axis=0).astype(np.float32)


def _normalize_stack_to_uint8(stack: np.ndarray) -> np.ndarray:
    stack = np.abs(stack)
    hi = float(np.percentile(stack, 99.5))
    if hi <= 0:
        return np.zeros(stack.shape, dtype=np.uint8)
    return np.clip(stack * (255.0 / hi), 0, 255).astype(np.uint8)


def filter_frames(
    preprocessed_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    fps: float = config.FPS,
    row_block: int = 64,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames = np.load(preprocessed_path, mmap_mode="r")
    n, h, w = frames.shape
    low_path = cache_dir / "low_speed_frames.npy"
    high_path = cache_dir / "high_speed_frames.npy"
    low_out = open_memmap(low_path, mode="w+", dtype=np.uint8, shape=(n, h, w))
    high_out = open_memmap(high_path, mode="w+", dtype=np.uint8, shape=(n, h, w))

    for y0 in range(0, h, row_block):
        y1 = min(h, y0 + row_block)
        block = frames[:, y0:y1, :].astype(np.float32) / 255.0
        low_out[:, y0:y1, :] = _normalize_stack_to_uint8(
            _butter_bandpass_filter(block, config.LOW_BAND_HZ, fps)
        )
        high_out[:, y0:y1, :] = _normalize_stack_to_uint8(
            _butter_bandpass_filter(block, config.HIGH_BAND_HZ, fps)
        )
    low_out.flush()
    high_out.flush()

    _write_preview_video(np.load(low_path, mmap_mode="r"), output_dir / "low_speed_preview.mp4", fps)
    _write_preview_video(np.load(high_path, mmap_mode="r"), output_dir / "high_speed_preview.mp4", fps)
    return low_path, high_path


def _write_preview_video(frames: np.ndarray, path: Path, fps: float) -> None:
    count = min(frames.shape[0], config.VIDEO_PREVIEW_MAX_FRAMES)
    h, w = frames.shape[1:]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return
    for idx in range(count):
        writer.write(cv2.cvtColor(frames[idx], cv2.COLOR_GRAY2BGR))
    writer.release()
