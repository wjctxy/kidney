"""Run Akebia-style Human Rapid and Slow pipelines from the same Step00 input.

默认行为：
    扫描项目根目录的 .dcm/.dicom 文件，从 source_frame_start 开始截取 frame_count 帧。

输出：
    human_dcm/step00_preprocess/<label>/frames.npy
    human_dcm/step01_bandpass/<label>/rapid|slow/
    human_dcm/step02_gaussian_filter/<label>/rapid|slow/
    human_dcm/step03_track/<label>/rapid|slow/
    human_dcm/step04_density_metrics/<label>/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import human_step01_bandpass
import human_step02_gaussian_filter
import human_step03_track
import human_step04_density_metrics
import ulm_config as config
import ulm_io
from ulm_visualization import save_frame_png_raw_scale


DEFAULT_FRAME_COUNT = 200
DICOM_SUFFIXES = {".dcm", ".dicom"}


def run(
    label: str | None = None,
    frames_path: Path | None = None,
    metadata_path: Path | None = None,
    preview_frame: int = 100,
    source_frame_start: int = 0,
    frame_count: int | None = DEFAULT_FRAME_COUNT,
) -> dict[str, Path]:
    """Run Rapid and Slow human profiles and combine their profile density maps."""

    base = ulm_io.dataset_dir("human")
    if (frames_path is None) != (metadata_path is None):
        raise ValueError("--frames 和 --metadata 必须同时提供，或者都不提供以自动从 DICOM 预处理")

    if frames_path is None and metadata_path is None:
        dicom_path = _resolve_root_dicom()
        label = label or _default_label(dicom_path, source_frame_start, frame_count)
        frames_path, metadata_path = _prepare_step00_window(
            dicom_path=dicom_path,
            label=label,
            source_frame_start=source_frame_start,
            frame_count=frame_count,
        )
    else:
        label = label or frames_path.parent.name

    profile_outputs: dict[str, dict[str, Path]] = {}
    for profile in ("rapid", "slow"):
        print(f"=== Human {profile} Step01 ===")
        step01_dir = base / "step01_bandpass" / label / profile
        filtered = human_step01_bandpass.run(
            frames_path=frames_path,
            metadata_path=metadata_path,
            output_path=step01_dir / "human_filtered.npy",
            profile=profile,
        )

        print(f"=== Human {profile} Step02 ===")
        step02_dir = base / "step02_gaussian_filter" / label / profile
        smoothed = human_step02_gaussian_filter.run(
            frames_path=filtered,
            metadata_path=metadata_path,
            output_path=step02_dir / "human_smoothed.npy",
        )

        print(f"=== Human {profile} Step03 ===")
        step03_dir = base / "step03_track" / label / profile
        profile_outputs[profile] = human_step03_track.run(
            frames_path=filtered,
            smooth_frames_path=smoothed,
            metadata_path=metadata_path,
            output_dir=step03_dir,
            preview_frame=preview_frame,
            profile=profile,
        )

    print("=== Human profile Step04 ===")
    step04_dir = base / "step04_density_metrics" / label
    combined = human_step04_density_metrics.run(
        rapid_tracks_csv=profile_outputs["rapid"]["tracks"],
        slow_tracks_csv=profile_outputs["slow"]["tracks"],
        metadata_path=metadata_path,
        output_dir=step04_dir,
    )

    outputs = {
        "rapid_tracks": profile_outputs["rapid"]["tracks"],
        "slow_tracks": profile_outputs["slow"]["tracks"],
        **combined,
    }
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Akebia-style Human Rapid and Slow profiles")
    parser.add_argument("--label", default=None)
    parser.add_argument("--source-frame-start", type=int, default=0, help="源 DICOM 起始帧，0-based，默认 0。")
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT, help="从起始帧开始保存的帧数，默认 200；设为 0 表示保存到末尾。")
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--preview-frame", type=int, default=100)
    return parser.parse_args()


def _prepare_step00_window(
    dicom_path: Path,
    label: str,
    source_frame_start: int,
    frame_count: int | None,
) -> tuple[Path, Path]:
    """从源 DICOM 指定起始帧截取窗口，生成 human Step00 输入。"""

    frame_limit = None if frame_count is not None and frame_count <= 0 else frame_count
    step00_dir = ulm_io.dataset_dir("human") / "step00_preprocess" / label
    frames_path = step00_dir / "frames.npy"
    metadata_path = step00_dir / "metadata.json"
    frames_path, metadata_path = ulm_io.dicom_to_frames(
        dicom_path=dicom_path,
        frames_path=frames_path,
        metadata_path=metadata_path,
        max_frames=frame_limit,
        source_frame_start=source_frame_start,
    )
    frames = ulm_io.load_frames(frames_path)
    preview_path = step00_dir / "preview_frame_000.png"
    save_frame_png_raw_scale(frames[0], preview_path, "Step 0 preprocessed frame")
    print(f"source_dicom: {dicom_path}")
    print(f"source_frame_start: {source_frame_start}")
    print(f"frames: {frames_path} shape={frames.shape}")
    print(f"metadata: {metadata_path}")
    print(f"preview: {preview_path}")
    return frames_path, metadata_path


def _resolve_root_dicom() -> Path:
    """扫描项目根目录，要求存在且只存在一个 DICOM 文件。"""

    candidates = sorted(
        path
        for path in config.BASE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in DICOM_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(f"项目根目录未找到 DICOM 文件：{config.BASE_DIR}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(f"项目根目录只能放一个 DICOM 文件，当前找到 {len(candidates)} 个：{names}")
    return candidates[0]


def _default_label(dicom_path: Path, source_frame_start: int, frame_count: int | None) -> str:
    """根据 DICOM 文件名和源帧窗口生成稳定输出标签。"""

    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", dicom_path.stem).strip("._-") or "dicom"
    if frame_count is not None and frame_count > 0:
        source_frame_end = source_frame_start + frame_count - 1
        return f"{safe_stem}_{source_frame_start}_{source_frame_end}"
    return f"{safe_stem}_{source_frame_start}_end"


if __name__ == "__main__":
    args = parse_args()
    run(
        label=args.label,
        frames_path=args.frames,
        metadata_path=args.metadata,
        preview_frame=args.preview_frame,
        source_frame_start=args.source_frame_start,
        frame_count=args.frame_count,
    )
