"""Step 0：DICOM 预处理，生成统一算法输入。

Input:
    human_dcm/11.0.dcm 或 mouse_dcm/11.0.dcm

Output:
    human_dcm/step00_preprocess/frames.npy: float32 [T,H,W]，单通道 CEUS score/灰度、裁剪 ROI、归一化到 [0,1]
    human_dcm/step00_preprocess/metadata.json: fps、像素尺寸、ROI、帧数等元数据
    human_dcm/step00_preprocess/preview_frame_000.png: 预处理后首帧预览

直接运行示例:
    python step00_preprocess.py --kind human
    python step00_preprocess.py --kind mouse
    python step00_preprocess.py --kind human --max-frames 0  # 处理全部帧
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ulm_io
from ulm_visualization import save_frame_png_raw_scale


DEFAULT_MAX_FRAMES = 600


def run(kind: str, dicom: Path | None = None, max_frames: int | None = DEFAULT_MAX_FRAMES) -> tuple[Path, Path]:
    """执行 step0：把 DICOM 转换为单通道 frames.npy 和 metadata.json。

    彩色伪彩 CEUS 会提取橙黄色造影增强 score；单通道 DICOM 保留灰度强度。
    """

    data_dir = ulm_io.step_dir(kind, "step00_preprocess")
    dicom_path = dicom or ulm_io.default_dicom_path(kind)
    frames_path = data_dir / "frames.npy"
    metadata_path = data_dir / "metadata.json"
    frame_limit = None if max_frames is not None and max_frames <= 0 else max_frames

    frames_path, metadata_path = ulm_io.dicom_to_frames(
        dicom_path=dicom_path,
        frames_path=frames_path,
        metadata_path=metadata_path,
        max_frames=frame_limit,
    )
    frames = ulm_io.load_frames(frames_path)
    preview_path = data_dir / "preview_frame_000.png"
    save_frame_png_raw_scale(frames[0], preview_path, "Step 0 preprocessed frame")
    print(f"frames: {frames_path}")
    print(f"metadata: {metadata_path}")
    print(f"preview: {preview_path}")
    return frames_path, metadata_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数，使本 step 可以直接通过 python 脚本运行。"""

    parser = argparse.ArgumentParser(description="Step 0: DICOM 预处理")
    parser.add_argument("--kind", choices=["human", "mouse"], default="human")
    parser.add_argument("--dicom", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="默认 600；设为 0 表示处理全部帧。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.kind, args.dicom, args.max_frames)
