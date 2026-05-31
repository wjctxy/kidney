"""ULM 数据接口工具。

统一约定：
- 帧序列只用 numpy 数组保存，形状为 [T, H, W]，dtype=float32。
- metadata.json 保存帧率、像素尺寸、ROI 等结构化信息。
- PNG/MP4 只作为人工查看结果，不作为任何算法 step 的输入。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from numpy.lib.format import open_memmap
from pydicom.pixel_data_handlers.util import convert_color_space
from pydicom.pixels import pixel_array as read_pixel_array

import ulm_config as config


def dataset_dir(kind: str) -> Path:
    """根据链路类型返回数据目录，kind 只能是 human 或 mouse。"""

    if kind == "human":
        return config.HUMAN_DIR
    if kind == "mouse":
        return config.MOUSE_DIR
    raise ValueError("kind 必须是 human 或 mouse")


def default_dicom_path(kind: str) -> Path:
    """返回指定链路默认 DICOM 文件路径。"""

    return dataset_dir(kind) / config.DICOM_NAME


def step_dir(kind: str, step_name: str) -> Path:
    """返回指定链路下某个 step 的输出目录，并保证目录存在。"""

    path = dataset_dir(kind) / step_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_metadata_path(kind: str) -> Path:
    """返回指定链路默认 metadata.json 路径。"""

    return step_dir(kind, "step00_preprocess") / "metadata.json"


def default_frames_path(kind: str) -> Path:
    """返回指定链路默认 frames.npy 路径。"""

    return step_dir(kind, "step00_preprocess") / "frames.npy"


def load_metadata(path: str | Path) -> dict[str, Any]:
    """读取 metadata.json，并返回普通 Python 字典。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    """把 metadata 字典保存为 UTF-8 JSON 文件。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_frames(path: str | Path, mmap_mode: str | None = "r") -> np.ndarray:
    """读取 frames.npy，并校验其必须是 [T,H,W] 三维数组。"""

    frames = np.load(path, mmap_mode=mmap_mode)
    if frames.ndim != 3:
        raise ValueError(f"frames 必须是 [T,H,W]，当前形状为 {frames.shape}")
    return frames


def read_dicom_metadata(dicom_path: str | Path) -> tuple[pydicom.Dataset, dict[str, Any]]:
    """从 DICOM 标签中提取 ROI、帧率、像素尺寸和帧数等元数据。"""

    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    x0, y0, x1, y1 = _read_ultrasound_roi(ds)
    frame_time_ms = float(getattr(ds, "FrameTime", config.DEFAULT_FRAME_TIME_MS))
    fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else config.DEFAULT_FPS
    pixel_size_x, pixel_size_y = _read_pixel_size(ds)
    number_of_frames = int(getattr(ds, "NumberOfFrames", 1))

    metadata = {
        "source_dicom": str(Path(dicom_path).resolve()),
        "number_of_frames": number_of_frames,
        "source_height": int(getattr(ds, "Rows")),
        "source_width": int(getattr(ds, "Columns")),
        "roi": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "height": int(y1 - y0),
        "width": int(x1 - x0),
        "fps": fps,
        "frame_time_ms": frame_time_ms,
        "pixel_size_x": pixel_size_x,
        "pixel_size_y": pixel_size_y,
        "frame_array": "float32 [T,H,W], normalized to [0,1]",
    }
    return ds, metadata


def dicom_to_frames(
    dicom_path: str | Path,
    frames_path: str | Path,
    metadata_path: str | Path,
    max_frames: int | None = None,
) -> tuple[Path, Path]:
    """把冗余 DICOM cine-loop 转换为统一算法输入。

    当前样例左半幅包含设备界面和冗余信息。这里优先读取 DICOM 的
    SequenceOfUltrasoundRegions，裁剪出真实超声图像区域，再转成灰度。
    """

    dicom_path = Path(dicom_path)
    frames_path = Path(frames_path)
    metadata_path = Path(metadata_path)
    ds, metadata = read_dicom_metadata(dicom_path)
    n_frames = int(metadata["number_of_frames"])
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    metadata["frames_saved"] = n_frames

    frames_path.parent.mkdir(parents=True, exist_ok=True)
    out = open_memmap(
        frames_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_frames, int(metadata["height"]), int(metadata["width"])),
    )

    roi = metadata["roi"]
    for frame_id in range(n_frames):
        frame = read_pixel_array(dicom_path, index=frame_id)
        gray = _prepare_frame(frame, ds)
        cropped = gray[roi["y0"] : roi["y1"], roi["x0"] : roi["x1"]]
        out[frame_id] = cropped.astype(np.float32) / 255.0
    out.flush()

    save_metadata(metadata, metadata_path)
    return frames_path, metadata_path


def _read_ultrasound_roi(ds: pydicom.Dataset) -> tuple[int, int, int, int]:
    """优先读取 DICOM 超声区域标签；缺失时返回配置中的默认 ROI。"""

    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if regions:
        region = regions[0]
        return (
            int(getattr(region, "RegionLocationMinX0", config.DEFAULT_ROI[0])),
            int(getattr(region, "RegionLocationMinY0", config.DEFAULT_ROI[1])),
            int(getattr(region, "RegionLocationMaxX1", config.DEFAULT_ROI[2])),
            int(getattr(region, "RegionLocationMaxY1", config.DEFAULT_ROI[3])),
        )
    return config.DEFAULT_ROI


def _read_pixel_size(ds: pydicom.Dataset) -> tuple[float, float]:
    """从 DICOM 超声区域标签读取单像素物理尺寸；缺失时使用默认值。"""

    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if regions:
        region = regions[0]
        return (
            float(getattr(region, "PhysicalDeltaX", config.DEFAULT_PIXEL_SIZE_X)),
            float(getattr(region, "PhysicalDeltaY", config.DEFAULT_PIXEL_SIZE_Y)),
        )
    return config.DEFAULT_PIXEL_SIZE_X, config.DEFAULT_PIXEL_SIZE_Y


def _prepare_frame(frame: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """将单帧 DICOM 像素转为 uint8 灰度图。"""

    if frame.ndim == 3 and frame.shape[-1] == 3:
        photo = getattr(ds, "PhotometricInterpretation", "")
        if str(photo).startswith("YBR"):
            frame = convert_color_space(frame, photo, "RGB")
        gray = (
            0.299 * frame[..., 0].astype(np.float32)
            + 0.587 * frame[..., 1].astype(np.float32)
            + 0.114 * frame[..., 2].astype(np.float32)
        )
    else:
        gray = frame.astype(np.float32)

    if gray.max() <= 255 and gray.min() >= 0:
        return gray.astype(np.uint8)

    lo = float(np.percentile(gray, 1))
    hi = float(np.percentile(gray, 99))
    if hi <= lo:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip((gray - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
