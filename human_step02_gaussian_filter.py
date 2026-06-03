"""Human Step 02：二维高斯滤波。

Input:
    human_dcm/step01_bandpass/human_filtered.npy: float32 [T,H,W]
    human_dcm/step00_preprocess/metadata.json

Output:
    human_dcm/step02_gaussian_filter/human_smoothed.npy: signed float32 [T,H,W]，只作为局部极大值 guide
    human_dcm/step02_gaussian_filter/signed_smoothed_frame_XXXX.png
    human_dcm/step02_gaussian_filter/bandpass_vs_gaussian_frame_comparison.png
    human_dcm/step02_gaussian_filter/gaussian_stats.json

算法说明:
    对齐 Akebia human 的 ULM_localization2D_interp.m：
    高斯平滑只用于 imregionalmax/局部极大值搜索，不替代原始带通帧的强度。
    Step 03 会用本 step 的 smoothed guide 找峰位置，再回到 Step 01 的 bandpass 帧取强度和做阈值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import gaussian_filter

import ulm_config as config
import ulm_io
from ulm_visualization import save_frame_png_raw_scale, save_signed_frame_comparison, symmetric_limits


def gaussian_smooth_frame(frame: np.ndarray) -> np.ndarray:
    """对单帧 signed bandpass 结果做二维高斯平滑，作为 Akebia 式局部极大值 guide。"""

    return gaussian_filter(frame, sigma=config.HUMAN_GAUSSIAN_SIGMA).astype(np.float32)


def summarize_array(prefix: str, arr: np.ndarray) -> dict[str, float]:
    """计算 signed 数组统计量，用于核对高斯平滑前后的真实数值范围。"""

    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p1": float(np.percentile(arr, 1)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p99": float(np.percentile(arr, 99)),
    }


def run(
    frames_path: Path | None = None,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """遍历人类 Step 01 输出帧，生成局部极大值搜索用的 human_smoothed.npy。"""

    input_dir = ulm_io.step_dir("human", "step01_bandpass")
    default_output_dir = ulm_io.step_dir("human", "step02_gaussian_filter")
    frames_path = frames_path or (input_dir / "human_filtered.npy")
    metadata_path = metadata_path or ulm_io.default_metadata_path("human")
    output_path = output_path or (default_output_dir / "human_smoothed.npy")
    output_dir = output_path.parent
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ = ulm_io.load_metadata(metadata_path)
    frames = ulm_io.load_frames(frames_path)
    smoothed = open_memmap(output_path, mode="w+", dtype=np.float32, shape=frames.shape)
    for frame_id in range(frames.shape[0]):
        smoothed[frame_id] = gaussian_smooth_frame(frames[frame_id])
    smoothed.flush()

    smooth_view = ulm_io.load_frames(output_path)
    preview_idx = frames.shape[0] // 2
    vmin, vmax = symmetric_limits(smooth_view[preview_idx])
    save_frame_png_raw_scale(
        smooth_view[preview_idx],
        output_dir / f"signed_smoothed_frame_{preview_idx:04d}.png",
        f"Akebia-style Gaussian local-max guide frame {preview_idx}",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    save_signed_frame_comparison(
        frames[preview_idx],
        smooth_view[preview_idx],
        output_dir / "bandpass_vs_gaussian_frame_comparison.png",
        f"Step 1 signed bandpass frame {preview_idx}",
        f"Step 2 Gaussian local-max guide frame {preview_idx}",
    )
    stats_path = output_dir / "gaussian_stats.json"
    stats = {
        **summarize_array("input_bandpass", frames),
        **summarize_array("smoothed", smooth_view),
        "input_bandpass_abs_p95": float(np.percentile(np.abs(frames), 95)),
        "input_bandpass_abs_p99": float(np.percentile(np.abs(frames), 99)),
        "smoothed_abs_p95": float(np.percentile(np.abs(smooth_view), 95)),
        "smoothed_abs_p99": float(np.percentile(np.abs(smooth_view), 99)),
        "gaussian_sigma": float(config.HUMAN_GAUSSIAN_SIGMA),
        "akebia_reference": "ULM_localization2D_interp: imregionalmax(imgaussfilt(frame, sigma)) uses sigma=1 for human profiles",
        "n_frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"human_smoothed: {output_path} shape={frames.shape}")
    print(f"signed_frame: {output_dir / f'signed_smoothed_frame_{preview_idx:04d}.png'}")
    print(f"comparison: {output_dir / 'bandpass_vs_gaussian_frame_comparison.png'}")
    print(f"stats: {stats_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析 Human Step 02 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Human Step 02: 二维高斯滤波")
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.frames, args.metadata, args.output)
