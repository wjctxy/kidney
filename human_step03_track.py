"""Human Step 03：正向峰值检测与匈牙利轨迹追踪。

Input:
    human_dcm/step01_bandpass/human_filtered.npy
    human_dcm/step02_gaussian_filter/human_smoothed.npy
    human_dcm/step00_preprocess/metadata.json

Output:
    human_dcm/step03_track/human_detections.csv
    human_dcm/step03_track/detections_positive_response_frame_XXXX.png
    human_dcm/step03_track/detections_on_signed_frame_XXXX.png
    human_dcm/step03_track/detection_pipeline_frame_XXXX.png
    human_dcm/step03_track/human_tracks.csv
    human_dcm/step03_track/tracking_summary.txt

算法说明:
    本 step 只负责 Akebia human 式局部极大值检测和匈牙利轨迹追踪：
    用 Step 02 的 Gaussian guide 找局部极大值位置，用 Step 01 原始 bandpass 帧取强度并做阈值。
    高/低速分组、密度图和指标计算全部放在 Human Step 04。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage.feature import peak_local_max

import ulm_config as config
import ulm_io
from ulm_tracking import track_detections, write_frame_detections
from ulm_visualization import save_detection_pipeline_preview, save_detection_preview, symmetric_limits


DEFAULT_PREVIEW_FRAME = 300


def build_detection_response(frame: np.ndarray) -> np.ndarray:
    """只保留 signed bandpass 图中的正向波动，构建非负检测响应图。"""

    return np.maximum(frame, 0.0).astype(np.float32)


def resolve_preview_frame_id(n_frames: int, requested: int = DEFAULT_PREVIEW_FRAME) -> int:
    """返回预览帧编号；请求帧越界时退回到中间帧。"""

    if 0 <= requested < n_frames:
        return requested
    return n_frames // 2


def detect_frame(
    frame: np.ndarray,
    smooth_guide: np.ndarray,
    frame_id: int,
    metadata: dict,
) -> list[dict[str, float | int | str]]:
    """用平滑帧找峰位置，再用原始 bandpass 帧取强度并筛选。

    对应 Akebia human:
      isABubble = imregionalmax(imgaussfilt(frame, sigma)) .* frame;
      isABubble = (isABubble >= threshold);
    """

    response = build_detection_response(frame)
    threshold = float(response.mean() + config.HUMAN_PEAK_THRESHOLD_STD * response.std())
    peaks = peak_local_max(
        smooth_guide,
        min_distance=config.HUMAN_LOCAL_MAX_MIN_DISTANCE,
        threshold_abs=None,
        exclude_border=False,
    )

    rows: list[dict[str, float | int | str]] = []
    for y, x in peaks:
        if response[y, x] < threshold:
            continue
        rows.append(
            {
                "frame_id": frame_id,
                "x_pixel": float(x),
                "y_pixel": float(y),
                "x_physical": float(x) * float(metadata["pixel_size_x"]),
                "y_physical": float(y) * float(metadata["pixel_size_y"]),
                "response_intensity": float(response[y, x]),
                "signed_intensity": float(frame[y, x]),
                "intensity": float(smooth_guide[y, x]),
                "method": "akebia_human_gaussian_localmax",
            }
        )
    return rows


def detect_bubbles(
    frames_path: Path,
    smooth_frames_path: Path,
    metadata: dict,
    output_csv: Path,
    output_dir: Path,
    preview_frame: int = DEFAULT_PREVIEW_FRAME,
) -> Path:
    """读取 bandpass 帧和平滑 guide，写出 Akebia human 式检测 CSV 和预览图。"""

    frames = ulm_io.load_frames(frames_path)
    smooth_frames = ulm_io.load_frames(smooth_frames_path)
    if smooth_frames.shape != frames.shape:
        raise ValueError(f"smooth guide shape {smooth_frames.shape} must match frames shape {frames.shape}")

    detections_csv, detection_count = write_frame_detections(
        frames.shape[0],
        lambda frame_id: detect_frame(frames[frame_id], smooth_frames[frame_id], frame_id, metadata),
        output_csv,
        desc="human detection",
    )
    preview_frame_id = resolve_preview_frame_id(frames.shape[0], preview_frame)
    signed_preview = frames[preview_frame_id]
    preview_response = build_detection_response(signed_preview)
    response_vmax = float(np.nanmax(preview_response))
    if response_vmax <= 0.0 or not np.isfinite(response_vmax):
        response_vmax = 1.0
    positive_path = output_dir / f"detections_positive_response_frame_{preview_frame_id:04d}.png"
    save_detection_preview(
        preview_response,
        detections_csv,
        positive_path,
        frame_id=preview_frame_id,
        title=f"Detections on positive response frame {preview_frame_id}",
        cmap="gray",
        vmin=0.0,
        vmax=response_vmax,
    )
    signed_vmin, signed_vmax = symmetric_limits(signed_preview)
    signed_path = output_dir / f"detections_on_signed_frame_{preview_frame_id:04d}.png"
    save_detection_preview(
        signed_preview,
        detections_csv,
        signed_path,
        frame_id=preview_frame_id,
        title=f"Detections on signed Gaussian frame {preview_frame_id}",
        cmap="seismic",
        vmin=signed_vmin,
        vmax=signed_vmax,
    )
    pipeline_path = output_dir / f"detection_pipeline_frame_{preview_frame_id:04d}.png"
    save_detection_pipeline_preview(
        signed_preview,
        preview_response,
        detections_csv,
        pipeline_path,
        frame_id=preview_frame_id,
    )
    print(f"human_detections: {detections_csv} count={detection_count}")
    print(f"detections_positive_response_preview: {positive_path}")
    print(f"detections_signed_preview: {signed_path}")
    print(f"detection_pipeline_preview: {pipeline_path}")
    return detections_csv


def run(
    frames_path: Path | None = None,
    smooth_frames_path: Path | None = None,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
    preview_frame: int = DEFAULT_PREVIEW_FRAME,
    profile: str = "rapid",
) -> dict[str, Path]:
    """运行 Human Step 03：Akebia human 式检测后进行匈牙利轨迹追踪。"""

    if profile not in config.HUMAN_PROFILES:
        raise ValueError(f"profile must be one of {sorted(config.HUMAN_PROFILES)}")

    frames_path = frames_path or (ulm_io.step_dir("human", "step01_bandpass") / "human_filtered.npy")
    smooth_frames_path = smooth_frames_path or (ulm_io.step_dir("human", "step02_gaussian_filter") / "human_smoothed.npy")
    metadata_path = metadata_path or ulm_io.default_metadata_path("human")
    output_dir = output_dir or ulm_io.step_dir("human", "step03_track")
    metadata = ulm_io.load_metadata(metadata_path)

    detections_csv = detect_bubbles(
        frames_path,
        smooth_frames_path,
        metadata,
        output_dir / "human_detections.csv",
        output_dir,
        preview_frame,
    )
    profile_params = config.HUMAN_PROFILES[profile]
    tracks_csv = track_detections(
        detections_csv,
        metadata,
        output_dir,
        prefix="human",
        max_frame_displacement_px=float(profile_params["max_frame_displacement_px"]),
        min_track_length=int(profile_params["min_track_length"]),
        max_missing_frames=int(profile_params["max_missing_frames"]),
        max_gap_closing_frames=int(profile_params["max_gap_closing_frames"]),
    )
    tracking_summary = output_dir / "tracking_summary.txt"
    print(f"human_tracks: {tracks_csv}")
    print(f"tracking_summary: {tracking_summary}")
    return {"detections": detections_csv, "tracks": tracks_csv, "tracking_summary": tracking_summary}


def parse_args() -> argparse.Namespace:
    """解析 Human Step 03 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Human Step 03: 正向峰值检测与轨迹追踪")
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--smooth-frames", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--preview-frame", type=int, default=DEFAULT_PREVIEW_FRAME)
    parser.add_argument("--profile", choices=sorted(config.HUMAN_PROFILES), default="rapid")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.frames, args.smooth_frames, args.metadata, args.output_dir, args.preview_frame, args.profile)
