"""
Visualize RGB color ultrasound multi-frame DICOM files exported by VINNO.

Usage:
    python DICOM/show.py path/to/video.dcm

Keyboard:
    Space      play / pause
    Left/Right previous / next frame
    Home/End   first / last frame
    Esc        close
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from pydicom.pixels import pixel_array as read_pixel_array


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert integer/float image data to uint8 for display."""
    if image.dtype == np.uint8:
        return image

    image_float = image.astype(np.float32)
    finite = np.isfinite(image_float)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    lo = float(image_float[finite].min())
    hi = float(image_float[finite].max())
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)

    image_float = (image_float - lo) * (255.0 / (hi - lo))
    return np.clip(image_float, 0, 255).astype(np.uint8)


def _prepare_frame(frame: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()

    if frame.ndim == 3 and photometric.startswith("YBR"):
        frame = convert_color_space(frame, photometric, "RGB")

    # MONOCHROME1 means low values are bright, so invert for normal display.
    if frame.ndim == 2 and photometric == "MONOCHROME1":
        frame = frame.max() - frame

    return _normalize_to_uint8(np.asarray(frame))


class LazyDicomFrames:
    """Read and cache only the frames that are displayed."""

    def __init__(self, dicom_path: Path, cache_size: int = 80):
        self.dicom_path = dicom_path
        self.ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)
        self.frame_count = int(getattr(self.ds, "NumberOfFrames", 1) or 1)
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, index: int) -> np.ndarray:
        index = int(np.clip(index, 0, self.frame_count - 1))
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]

        try:
            # This is the important part: decode one frame instead of calling
            # ds.pixel_array, which expands the whole video into memory.
            frame = read_pixel_array(str(self.dicom_path), index=index)
        except MemoryError as exc:
            raise RuntimeError(
                "Not enough memory to decode this DICOM frame. Try closing other "
                "programs, using 64-bit Python, or exporting a shorter/lower "
                "resolution DICOM clip."
            ) from exc
        except Exception as exc:
            transfer_syntax = getattr(self.ds.file_meta, "TransferSyntaxUID", "unknown")
            raise RuntimeError(
                "Failed to decode DICOM pixel data. Transfer Syntax UID: "
                f"{transfer_syntax}\n"
                "If the file is compressed, install decoder plugins and retry:\n"
                "    pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
                "If it still fails, try:\n"
                "    pip install python-gdcm"
            ) from exc

        frame = _prepare_frame(frame, self.ds)
        self._cache[index] = frame
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return frame


def open_dicom_frames(dicom_path: Path, cache_size: int) -> LazyDicomFrames:
    frames = LazyDicomFrames(dicom_path, cache_size=cache_size)
    frames.get(0)
    return frames


class DicomVideoViewer:
    def __init__(self, frames: LazyDicomFrames, dicom_path: Path):
        self.frames = frames
        self.ds = frames.ds
        self.dicom_path = dicom_path
        self.index = 0
        self.playing = True
        self.slider_dragging = False

        self.frame_count = frames.frame_count
        self.interval_ms = self._get_frame_interval_ms()

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title(f"DICOM Viewer - {dicom_path.name}")
        plt.subplots_adjust(bottom=0.18)

        self.image = self.ax.imshow(self.frames.get(self.index), cmap="gray")
        self.ax.axis("off")
        self.title = self.ax.set_title("")

        slider_ax = self.fig.add_axes((0.15, 0.06, 0.72, 0.035))
        self.slider = Slider(
            slider_ax,
            "Frame",
            0,
            max(0, self.frame_count - 1),
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(self._on_slider_changed)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)

        self.animation = FuncAnimation(
            self.fig,
            self._tick,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        self._draw_frame()

    def _get_frame_interval_ms(self) -> int:
        frame_time = getattr(self.ds, "FrameTime", None)
        if frame_time:
            try:
                return max(1, int(float(frame_time)))
            except ValueError:
                pass

        cine_rate = getattr(self.ds, "CineRate", None)
        if cine_rate:
            try:
                return max(1, int(1000.0 / float(cine_rate)))
            except (ValueError, ZeroDivisionError):
                pass

        return 40

    def _draw_frame(self) -> None:
        self.image.set_data(self.frames.get(self.index))
        status = "Playing" if self.playing else "Paused"
        self.title.set_text(
            f"{self.dicom_path.name} | {status} | "
            f"{self.index + 1}/{self.frame_count} | interval {self.interval_ms} ms"
        )
        self.fig.canvas.draw_idle()

    def _set_index(self, index: int, update_slider: bool = True) -> None:
        self.index = int(np.clip(index, 0, self.frame_count - 1))
        if update_slider:
            self.slider.set_val(self.index)
        else:
            self._draw_frame()

    def _tick(self, _frame_number: int) -> None:
        if self.playing and self.frame_count > 1 and not self.slider_dragging:
            self._set_index((self.index + 1) % self.frame_count)

    def _on_slider_changed(self, value: float) -> None:
        self._set_index(int(value), update_slider=False)

    def _on_mouse_press(self, event) -> None:
        if event.inaxes == self.slider.ax:
            self.slider_dragging = True

    def _on_mouse_release(self, event) -> None:
        self.slider_dragging = False

    def _on_key_press(self, event) -> None:
        if event.key == " ":
            self.playing = not self.playing
            self._draw_frame()
        elif event.key == "right":
            self.playing = False
            self._set_index(self.index + 1)
        elif event.key == "left":
            self.playing = False
            self._set_index(self.index - 1)
        elif event.key == "home":
            self.playing = False
            self._set_index(0)
        elif event.key == "end":
            self.playing = False
            self._set_index(self.frame_count - 1)
        elif event.key == "escape":
            plt.close(self.fig)

    def show(self) -> None:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize single-frame or multi-frame color ultrasound DICOM files."
    )
    parser.add_argument("dicom_file", type=Path, help="Path to .dcm file")
    parser.add_argument(
        "--cache-size",
        type=int,
        default=80,
        help="Number of decoded frames kept in memory. Default: 80",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dicom_path = args.dicom_file.expanduser().resolve()
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    frames = open_dicom_frames(dicom_path, cache_size=args.cache_size)
    viewer = DicomVideoViewer(frames, dicom_path)
    viewer.show()


if __name__ == "__main__":
    main()
