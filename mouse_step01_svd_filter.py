"""Mouse Step 01：时空 SVD 滤波。

Input:
    mouse_dcm/step00_preprocess/frames.npy: float32 [T,H,W]
    mouse_dcm/step00_preprocess/metadata.json

Output:
    mouse_dcm/step01_svd_filter/mouse_filtered.npy: float32 [T,H,W]
    mouse_dcm/step01_svd_filter/mouse_singular_values.csv
    mouse_dcm/step01_svd_filter/svd_frame_comparison.png
    mouse_dcm/step01_svd_filter/singular_values.png

算法说明:
    将 [T,H,W] 重排为 [H*W,T] 后做 SVD。大奇异值通常对应组织背景，
    中间奇异值更可能对应微泡运动，小奇异值多为随机噪声。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from sklearn.utils.extmath import randomized_svd

import ulm_config as config
import ulm_io
from ulm_visualization import save_frame_comparison, save_frame_png_raw_scale, save_singular_values_plot, symmetric_limits


def svd_filter(
    frames: np.ndarray,
    low_rank_cut: int = config.SVD_LOW_RANK_CUT,
    high_rank_cut: int = config.SVD_HIGH_RANK_CUT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 [T,H,W] 帧序列做 SVD 滤波，返回滤波帧、奇异值和保留掩码。"""

    n_frames, height, width = frames.shape
    matrix = frames.reshape(n_frames, height * width).T.astype(np.float32)
    n_components = min(high_rank_cut, min(matrix.shape) - 1)

    # randomized_svd 避免对完整大矩阵做全量分解，仍保留 SVD 的中间奇异值思想。
    u, singular_values, vt = randomized_svd(matrix, n_components=n_components, random_state=0)
    keep = np.zeros_like(singular_values, dtype=bool)
    keep[low_rank_cut:n_components] = True
    filtered_matrix = (u[:, keep] * singular_values[keep]) @ vt[keep]
    filtered = filtered_matrix.T.reshape(n_frames, height, width).astype(np.float32)
    return filtered, singular_values, keep


def run(
    frames_path: Path | None = None,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
    singular_csv: Path | None = None,
    low_rank_cut: int = config.SVD_LOW_RANK_CUT,
    high_rank_cut: int = config.SVD_HIGH_RANK_CUT,
) -> tuple[Path, Path]:
    """读取小鼠 frames.npy，执行 SVD 滤波，并写出滤波帧和奇异值表。"""

    data_dir = ulm_io.step_dir("mouse", "step01_svd_filter")
    frames_path = frames_path or ulm_io.default_frames_path("mouse")
    metadata_path = metadata_path or ulm_io.default_metadata_path("mouse")
    output_path = output_path or (data_dir / "mouse_filtered.npy")
    singular_csv = singular_csv or (data_dir / "mouse_singular_values.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _ = ulm_io.load_metadata(metadata_path)
    frames = ulm_io.load_frames(frames_path)
    filtered, singular_values, keep = svd_filter(frames, low_rank_cut, high_rank_cut)

    out = open_memmap(output_path, mode="w+", dtype=np.float32, shape=filtered.shape)
    out[:] = filtered
    out.flush()
    _write_singular_values(singular_values, keep, singular_csv)
    vmin, vmax = symmetric_limits(filtered[0])
    save_frame_png_raw_scale(
        filtered[0],
        data_dir / "filtered_frame_000.png",
        "Mouse Step 01 signed SVD filtered frame",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
    )
    save_frame_comparison(
        frames[0],
        filtered[0],
        data_dir / "svd_frame_comparison.png",
        "Step 0 input",
        "Step 1 SVD output",
    )
    save_singular_values_plot(singular_values, keep, data_dir / "singular_values.png")
    print(f"mouse_filtered: {output_path} shape={filtered.shape}")
    print(f"singular_values: {singular_csv}")
    print(f"preview: {data_dir / 'svd_frame_comparison.png'}")
    return output_path, singular_csv


def _write_singular_values(values: np.ndarray, keep: np.ndarray, path: Path) -> None:
    """把奇异值及其是否保留写入 CSV，便于检查 SVD 阈值区间。"""

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "singular_value", "kept"])
        writer.writeheader()
        for idx, value in enumerate(values):
            writer.writerow({"rank": idx, "singular_value": float(value), "kept": bool(keep[idx])})


def parse_args() -> argparse.Namespace:
    """解析 Mouse Step 01 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Mouse Step 01: SVD 滤波")
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--singular-csv", type=Path, default=None)
    parser.add_argument("--low-rank-cut", type=int, default=config.SVD_LOW_RANK_CUT)
    parser.add_argument("--high-rank-cut", type=int, default=config.SVD_HIGH_RANK_CUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.frames,
        args.metadata,
        args.output,
        args.singular_csv,
        args.low_rank_cut,
        args.high_rank_cut,
    )
