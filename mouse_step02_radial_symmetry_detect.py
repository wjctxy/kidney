"""Mouse Step 02：径向对称法亚像素定位。

Input:
    mouse_dcm/step01_svd_filter/mouse_filtered.npy: float32 [T,H,W]
    mouse_dcm/step00_preprocess/metadata.json

Output:
    mouse_dcm/step02_detect/mouse_detections.csv
    mouse_dcm/step02_detect/detections_frame_000.png

算法说明:
    先用局部峰值找到候选亮斑，再围绕候选点裁剪 patch。
    patch 内灰度梯度大致指向微泡中心，使用加权最小二乘估计亚像素中心。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

import ulm_config as config
import ulm_io
from ulm_tracking import write_frame_detections
from ulm_visualization import save_detection_preview


def detect_frame(frame: np.ndarray, frame_id: int, metadata: dict) -> list[dict[str, float | int | str]]:
    """先找候选峰值，再用径向对称法细化单帧微泡中心。"""

    # SVD 重建后可能存在正负响应；定位时使用响应幅值寻找候选亮斑。
    smooth = gaussian_filter(np.abs(frame), sigma=config.GAUSSIAN_SIGMA)
    threshold = float(smooth.mean() + config.PEAK_THRESHOLD_STD * smooth.std())
    peaks = peak_local_max(
        smooth,
        min_distance=config.PEAK_MIN_DISTANCE,
        threshold_abs=threshold,
        exclude_border=config.RADIAL_PATCH_RADIUS,
    )

    rows: list[dict[str, float | int | str]] = []
    for y, x in peaks:
        refined = radial_symmetry_center(smooth, int(y), int(x), config.RADIAL_PATCH_RADIUS)
        if refined is None:
            continue
        x_sub, y_sub = refined
        rows.append(
            {
                "frame_id": frame_id,
                "x_pixel": x_sub,
                "y_pixel": y_sub,
                "x_physical": x_sub * float(metadata["pixel_size_x"]),
                "y_physical": y_sub * float(metadata["pixel_size_y"]),
                "intensity": float(smooth[int(round(y_sub)), int(round(x_sub))]),
                "method": "radial_symmetry",
            }
        )
    return rows


def radial_symmetry_center(frame: np.ndarray, y: int, x: int, radius: int) -> tuple[float, float] | None:
    """在候选点周围的 patch 内估计亚像素级径向对称中心。"""

    patch = frame[y - radius : y + radius + 1, x - radius : x + radius + 1]
    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        return None

    gy, gx = np.gradient(patch.astype(np.float64))
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    weight = np.hypot(gx, gy)
    valid = weight > np.percentile(weight, 60)
    if valid.sum() < 4:
        return None

    # 梯度方向线的法向约束：[-gy, gx] dot ([cx,cy] - [x_i,y_i]) = 0。
    a = np.column_stack((-gy[valid].ravel(), gx[valid].ravel()))
    b = a[:, 0] * xx[valid].ravel() + a[:, 1] * yy[valid].ravel()
    w = weight[valid].ravel()
    aw = a * w[:, None]
    bw = b * w
    try:
        cx, cy = np.linalg.lstsq(aw, bw, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    if abs(cx) > radius or abs(cy) > radius:
        return None
    return float(x + cx), float(y + cy)


def run(
    frames_path: Path | None = None,
    metadata_path: Path | None = None,
    output_csv: Path | None = None,
) -> Path:
    """遍历小鼠 SVD 滤波帧，生成 mouse_detections.csv。"""

    input_dir = ulm_io.step_dir("mouse", "step01_svd_filter")
    output_dir = ulm_io.step_dir("mouse", "step02_detect")
    frames_path = frames_path or (input_dir / "mouse_filtered.npy")
    metadata_path = metadata_path or ulm_io.default_metadata_path("mouse")
    output_csv = output_csv or (output_dir / "mouse_detections.csv")

    frames = ulm_io.load_frames(frames_path)
    metadata = ulm_io.load_metadata(metadata_path)
    path, detection_count = write_frame_detections(
        frames.shape[0],
        lambda frame_id: detect_frame(frames[frame_id], frame_id, metadata),
        output_csv,
        desc="mouse detection",
    )
    preview_path = output_dir / "detections_frame_000.png"
    save_detection_preview(frames[0], path, preview_path, frame_id=0)
    print(f"detections: {path} count={detection_count}")
    print(f"preview: {preview_path}")
    return path


def parse_args() -> argparse.Namespace:
    """解析 Mouse Step 02 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Mouse Step 02: 径向对称法定位")
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.frames, args.metadata, args.output)
