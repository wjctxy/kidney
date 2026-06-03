"""Human Step 01：时域带通滤波。

Input:
    human_dcm/step00_preprocess/frames.npy: float32 [T,H,W]
    human_dcm/step00_preprocess/metadata.json

Output:
    human_dcm/step01_bandpass/human_filtered.npy: float32 [T,H,W]
    human_dcm/step01_bandpass/signed_bandpass_frame_XXXX.png
    human_dcm/step01_bandpass/raw_vs_signed_bandpass_frame_comparison.png
    human_dcm/step01_bandpass/bandpass_std_projection.png
    human_dcm/step01_bandpass/bandpass_temporal_curve_center.png
    human_dcm/step01_bandpass/bandpass_temporal_curve_max_std.png
    human_dcm/step01_bandpass/bandpass_stats.json

算法说明:
    对每个空间位置 (x,y) 的时间曲线 I(x,y,t) 做 Akebia human Rapid profile
    同形式的一阶 Butterworth 因果带通：butter(order=1) + lfilter。
    低频部分多为组织、呼吸和探头慢变化，高频部分多为随机噪声；
    带通后保留微泡运动对应的中间频段。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json  # 输出统计结果

import numpy as np
from numpy.lib.format import open_memmap
from scipy.signal import butter, lfilter

import ulm_config as config
import ulm_io
from ulm_visualization import (
    save_frame_comparison,
    save_frame_png_raw_scale,
    save_temporal_curve,
    save_temporal_projection_comparison,
    symmetric_limits,
)


def temporal_bandpass(
    frames: np.ndarray,
    fps: float,
    lowcut: float,
    highcut: float,
    order: int = config.HUMAN_BANDPASS_ORDER,
) -> np.ndarray:
    """沿时间轴做 Akebia human 形式的因果 Butterworth 带通滤波。"""

    if frames.ndim != 3:
        raise ValueError(f"frames 必须是 [T,H,W]，当前形状为 {frames.shape}")  # 显式检查输入形状
    if fps <= 0:
        raise ValueError(f"fps 必须为正数，当前为 {fps}")  # 显式检查 fps
    if lowcut <= 0 or highcut <= 0:
        raise ValueError("lowcut 和 highcut 必须为正数")  # 显式检查频段
    if highcut <= lowcut:
        raise ValueError("highcut 必须大于 lowcut")  # 显式检查频段顺序
    nyquist = fps * 0.5
    if lowcut >= nyquist or highcut >= nyquist:
        raise ValueError(f"截止频率必须小于 Nyquist({nyquist:.3f} Hz)")  # 避免静默截断
    low = lowcut / nyquist  # 归一化低截止频率
    high = highcut / nyquist  # 归一化高截止频率
    b, a = butter(order, [low, high], btype="bandpass")
    # Akebia human 在 BlockInterp2Track.m 中用 MATLAB filter(..., [], 3)。
    # 这里等价地沿 Python 的时间轴 axis=0 做单向 IIR 因果滤波，不做零相位反向补偿。
    return lfilter(b, a, frames, axis=0).astype(np.float32)


def summarize_array(name: str, arr: np.ndarray) -> dict[str, float]:
    """计算数组基础统计量，便于判断滤波前后真实幅度变化。"""

    return {
        f"{name}_mean": float(np.mean(arr)),
        f"{name}_std": float(np.std(arr)),
        f"{name}_min": float(np.min(arr)),
        f"{name}_max": float(np.max(arr)),
        f"{name}_p1": float(np.percentile(arr, 1)),
        f"{name}_p50": float(np.percentile(arr, 50)),
        f"{name}_p99": float(np.percentile(arr, 99)),
    }


def run(
    frames_path: Path | None = None,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
    row_block: int = 32,
) -> Path:
    """读取人类 frames.npy，分块执行时域带通滤波，并保存 human_filtered.npy。"""

    frames_path = frames_path or ulm_io.default_frames_path("human")
    metadata_path = metadata_path or ulm_io.default_metadata_path("human")
    default_output_dir = ulm_io.step_dir("human", "step01_bandpass")  # step01 默认输出目录
    output_path = output_path or (default_output_dir / "human_filtered.npy")  # 默认输出文件
    output_dir = output_path.parent  # 传入自定义 output_path 时，PNG/stats 跟随同一输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    metadata = ulm_io.load_metadata(metadata_path)
    frames = ulm_io.load_frames(frames_path)
    n_frames, height, width = frames.shape
    out = open_memmap(output_path, mode="w+", dtype=np.float32, shape=frames.shape)  # memmap 写盘

    for y0 in range(0, height, row_block):
        y1 = min(height, y0 + row_block)
        block = frames[:, y0:y1, :]
        out[:, y0:y1, :] = temporal_bandpass(
            block,
            fps=float(metadata["fps"]),
            lowcut=config.HUMAN_BANDPASS_LOW_HZ,
            highcut=config.HUMAN_BANDPASS_HIGH_HZ,
            order=config.HUMAN_BANDPASS_ORDER,
        )
    out.flush()
    filtered = ulm_io.load_frames(output_path)  # 重新加载输出用于可视化
    preview_idx = n_frames // 2  # 因果 IIR 前几帧有启动瞬态，中间帧更适合人工检查
    vmin, vmax = symmetric_limits(filtered[preview_idx])
    save_frame_png_raw_scale(  # signed 单帧预览图
        filtered[preview_idx],
        output_dir / f"signed_bandpass_frame_{preview_idx:04d}.png",
        f"Signed bandpass frame {preview_idx}",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    save_frame_comparison(  # 单帧对比图
        frames[preview_idx],
        filtered[preview_idx],
        output_dir / "raw_vs_signed_bandpass_frame_comparison.png",
        f"Raw frame {preview_idx}",
        f"Signed bandpass frame {preview_idx}",
    )
    save_temporal_projection_comparison(  # 时域投影对比图（STD）
        frames,
        filtered,
        output_dir / "bandpass_std_projection.png",
        mode="std",
    )
    save_temporal_curve(  # 中心像素时间曲线
        frames,
        filtered,
        metadata,
        output_dir / "bandpass_temporal_curve_center.png",
        pick="center",
    )
    save_temporal_curve(  # 动态最强像素时间曲线
        frames,
        filtered,
        metadata,
        output_dir / "bandpass_temporal_curve_max_std.png",
        pick="max_std",
    )
    stats_path = output_dir / "bandpass_stats.json"  # 输出统计 JSON
    stats = {
        **summarize_array("raw", frames),
        **summarize_array("bandpass", filtered),
        "bandpass_abs_p95": float(np.percentile(np.abs(filtered), 95)),
        "bandpass_abs_p99": float(np.percentile(np.abs(filtered), 99)),
        "fps": float(metadata["fps"]),
        "lowcut": float(config.HUMAN_BANDPASS_LOW_HZ),
        "highcut": float(config.HUMAN_BANDPASS_HIGH_HZ),
        "filter_order": int(config.HUMAN_BANDPASS_ORDER),
        "filter_method": "butterworth_causal_lfilter",
        "akebia_reference": "akebia_human Rapid: butter(1), filter(), bandpassBounds=[1, 5.5] Hz",
        "n_frames": int(n_frames),
    }
    with stats_path.open("w", encoding="utf-8") as f:  # 保存统计 JSON
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"human_filtered: {output_path} shape=({n_frames}, {height}, {width})")
    print(f"signed_frame: {output_dir / f'signed_bandpass_frame_{preview_idx:04d}.png'}")
    print(f"comparison: {output_dir / 'raw_vs_signed_bandpass_frame_comparison.png'}")
    print(f"projection(std): {output_dir / 'bandpass_std_projection.png'}")
    print(f"curve(center): {output_dir / 'bandpass_temporal_curve_center.png'}")
    print(f"curve(max_std): {output_dir / 'bandpass_temporal_curve_max_std.png'}")
    print(f"stats: {stats_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    """解析 Human Step 01 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Human Step 01: 时域带通滤波")
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--row-block", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.frames, args.metadata, args.output, args.row_block)
