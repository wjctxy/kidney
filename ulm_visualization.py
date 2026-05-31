"""ULM 中间结果可视化工具。

这些函数只生成 PNG 供人工检查，不改变任何算法 step 的输入输出契约。
算法模块之间仍然只传 npy/json/csv。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def symmetric_limits(image: np.ndarray) -> tuple[float, float]:
    """根据 signed 图像的最大绝对值返回对称显示范围。"""

    lim = float(max(abs(np.nanmin(image)), abs(np.nanmax(image))))
    if lim == 0.0 or not np.isfinite(lim):
        lim = 1.0
    return -lim, lim


def save_frame_png_raw_scale(
    frame: np.ndarray,
    path: str | Path,
    title: str,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """按原始数值范围保存单帧 PNG，不做 abs、裁剪或百分位归一化。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_frame_comparison(
    before: np.ndarray,
    after: np.ndarray,
    path: str | Path,
    before_title: str = "Raw frame",
    after_title: str = "Signed bandpass frame",
) -> Path:
    """保存 raw frame 与 signed bandpass frame 对比图，直接显示真实数值。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = symmetric_limits(after)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw_image = axes[0].imshow(before, cmap="gray")
    axes[0].set_title(before_title)
    axes[0].axis("off")
    fig.colorbar(raw_image, ax=axes[0], fraction=0.046, pad=0.04)

    bandpass_image = axes[1].imshow(after, cmap="seismic", vmin=vmin, vmax=vmax)
    axes[1].set_title(after_title)
    axes[1].axis("off")
    fig.colorbar(bandpass_image, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_signed_to_nonnegative_comparison(
    before: np.ndarray,
    after: np.ndarray,
    path: str | Path,
    before_title: str,
    after_title: str,
) -> Path:
    """保存 signed 输入与非负输出对比图；左侧发散色图，右侧灰度图。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    before_vmin, before_vmax = symmetric_limits(before)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    before_image = axes[0].imshow(before, cmap="seismic", vmin=before_vmin, vmax=before_vmax)
    axes[0].set_title(before_title)
    axes[0].axis("off")
    fig.colorbar(before_image, ax=axes[0], fraction=0.046, pad=0.04)

    after_image = axes[1].imshow(after, cmap="gray", vmin=0.0, vmax=float(np.nanmax(after)))
    axes[1].set_title(after_title)
    axes[1].axis("off")
    fig.colorbar(after_image, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_signed_frame_comparison(
    before: np.ndarray,
    after: np.ndarray,
    path: str | Path,
    before_title: str,
    after_title: str,
) -> Path:
    """保存两个 signed frame 的对比图，使用统一对称色标，不做 abs 或 normalization。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lim = float(
        max(
            abs(np.nanmin(before)),
            abs(np.nanmax(before)),
            abs(np.nanmin(after)),
            abs(np.nanmax(after)),
        )
    )
    if lim == 0.0 or not np.isfinite(lim):
        lim = 1.0
    vmin = -lim
    vmax = lim

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    before_image = axes[0].imshow(before, cmap="seismic", vmin=vmin, vmax=vmax)
    axes[0].set_title(before_title)
    axes[0].axis("off")
    fig.colorbar(before_image, ax=axes[0], fraction=0.046, pad=0.04)

    after_image = axes[1].imshow(after, cmap="seismic", vmin=vmin, vmax=vmax)
    axes[1].set_title(after_title)
    axes[1].axis("off")
    fig.colorbar(after_image, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_temporal_projection_comparison(
    raw_frames: np.ndarray,
    filtered_frames: np.ndarray,
    path: str | Path,
    mode: str = "std",
) -> Path:
    """保存时域投影对比图；std 非负用灰度，mean signed 用发散色图。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "std":
        raw_proj = np.std(raw_frames, axis=0)
        filtered_proj = np.std(filtered_frames, axis=0)
        raw_cmap = "gray"
        filtered_cmap = "gray"
        raw_limits = (None, None)
        filtered_limits = (None, None)
        title = "Temporal STD projection"
    elif mode == "mean":
        raw_proj = np.mean(raw_frames, axis=0)
        filtered_proj = np.mean(filtered_frames, axis=0)
        raw_cmap = "gray"
        filtered_cmap = "seismic"
        raw_limits = (None, None)
        filtered_limits = symmetric_limits(filtered_proj)
        title = "Temporal mean projection"
    else:
        raise ValueError(f"Unknown projection mode: {mode}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    raw_image = axes[0].imshow(raw_proj, cmap=raw_cmap, vmin=raw_limits[0], vmax=raw_limits[1])
    axes[0].set_title(f"Raw {mode}")
    axes[0].axis("off")
    fig.colorbar(raw_image, ax=axes[0], fraction=0.046, pad=0.04)

    filtered_image = axes[1].imshow(
        filtered_proj,
        cmap=filtered_cmap,
        vmin=filtered_limits[0],
        vmax=filtered_limits[1],
    )
    axes[1].set_title(f"Bandpass {mode}")
    axes[1].axis("off")
    fig.colorbar(filtered_image, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_temporal_curve(
    raw_frames: np.ndarray,
    filtered_frames: np.ndarray,
    metadata: dict,
    path: str | Path,
    x: int | None = None,
    y: int | None = None,
    pick: str = "center",
) -> Path:
    """保存指定像素的原始强度和 signed bandpass 时间曲线，二者使用独立 y 轴。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pick == "center":
        y = raw_frames.shape[1] // 2 if y is None else y
        x = raw_frames.shape[2] // 2 if x is None else x
    elif pick == "max_std":
        activity = np.std(filtered_frames, axis=0)
        y, x = np.unravel_index(np.argmax(activity), activity.shape)
    else:
        raise ValueError(f"Unknown pick mode: {pick}")
    t = np.arange(raw_frames.shape[0]) / float(metadata["fps"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, raw_frames[:, y, x], label="raw", linewidth=1.2)
    axes[0].set_ylabel("raw intensity")
    axes[0].legend()
    axes[1].plot(t, filtered_frames[:, y, x], label="signed bandpass", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("signed bandpass intensity")
    axes[1].legend()
    fig.suptitle(f"Temporal curve at pixel ({x}, {y}), pick={pick}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_detection_preview(
    frame: np.ndarray,
    detections_csv: str | Path,
    path: str | Path,
    frame_id: int = 0,
    max_points: int = 2000,
    title: str | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """在指定帧原始数值图上叠加检测点，不对背景图做 abs 或百分位归一化。"""

    points = _load_points_for_frame(detections_csv, frame_id, max_points)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
    if points:
        xs, ys = zip(*points)
        ax.scatter(xs, ys, s=8, facecolors="none", edgecolors="red", linewidths=0.6)
    ax.set_title(title or f"Detections on frame {frame_id}")
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_detection_pipeline_preview(
    signed_frame: np.ndarray,
    response_frame: np.ndarray,
    detections_csv: str | Path,
    path: str | Path,
    frame_id: int,
) -> Path:
    """保存 Step 03 三联图：signed frame、positive response、positive peaks。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signed_vmin, signed_vmax = symmetric_limits(signed_frame)
    response_vmax = float(np.nanmax(response_frame))
    if response_vmax <= 0.0 or not np.isfinite(response_vmax):
        response_vmax = 1.0
    points = _load_points_for_frame(detections_csv, frame_id, max_points=2000)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    signed_image = axes[0].imshow(signed_frame, cmap="seismic", vmin=signed_vmin, vmax=signed_vmax)
    axes[0].set_title(f"Step 2 signed Gaussian frame {frame_id}")
    axes[0].axis("off")
    fig.colorbar(signed_image, ax=axes[0], fraction=0.046, pad=0.04)

    response_image = axes[1].imshow(response_frame, cmap="gray", vmin=0.0, vmax=response_vmax)
    axes[1].set_title("Positive response = max(frame, 0)")
    axes[1].axis("off")
    fig.colorbar(response_image, ax=axes[1], fraction=0.046, pad=0.04)

    detection_image = axes[2].imshow(response_frame, cmap="gray", vmin=0.0, vmax=response_vmax)
    if points:
        xs, ys = zip(*points)
        axes[2].scatter(xs, ys, s=8, facecolors="none", edgecolors="red", linewidths=0.6)
    axes[2].set_title("Detected positive peaks")
    axes[2].axis("off")
    fig.colorbar(detection_image, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_singular_values_plot(values: np.ndarray, keep: np.ndarray, path: str | Path) -> Path:
    """保存 SVD 奇异值曲线，并标出当前被保留的 rank 区间。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ranks = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ranks, values, marker="o", linewidth=1.2, label="singular value")
    if keep.any():
        ax.scatter(ranks[keep], values[keep], color="red", s=24, label="kept")
    ax.set_xlabel("rank")
    ax.set_ylabel("singular value")
    ax.set_title("SVD singular values")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _load_points_for_frame(
    detections_csv: str | Path,
    frame_id: int,
    max_points: int,
) -> list[tuple[float, float]]:
    """从 detections.csv 中读取指定帧的检测点坐标，最多返回 max_points 个。"""

    points: list[tuple[float, float]] = []
    with Path(detections_csv).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["frame_id"]) != frame_id:
                continue
            points.append((float(row["x_pixel"]), float(row["y_pixel"])))
            if len(points) >= max_points:
                break
    return points
