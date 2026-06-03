"""Run Akebia-style Human Rapid and Slow pipelines from the same Step00 input.

默认读取：
    human_dcm/step00_preprocess/stable_200_400/frames.npy
    human_dcm/step00_preprocess/stable_200_400/metadata.json

输出：
    human_dcm/step01_bandpass/stable_200_400/rapid|slow/
    human_dcm/step02_gaussian_filter/stable_200_400/rapid|slow/
    human_dcm/step03_track/stable_200_400/rapid|slow/
    human_dcm/step04_density_metrics/stable_200_400/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import human_step01_bandpass
import human_step02_gaussian_filter
import human_step03_track
import human_step04_density_metrics
import ulm_io


DEFAULT_LABEL = "stable_200_400"


def run(
    label: str = DEFAULT_LABEL,
    frames_path: Path | None = None,
    metadata_path: Path | None = None,
    preview_frame: int = 100,
) -> dict[str, Path]:
    """Run Rapid and Slow human profiles and combine their profile density maps."""

    base = ulm_io.dataset_dir("human")
    step00_dir = base / "step00_preprocess" / label
    frames_path = frames_path or (step00_dir / "frames.npy")
    metadata_path = metadata_path or (step00_dir / "metadata.json")

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
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--frames", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--preview-frame", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.label, args.frames, args.metadata, args.preview_frame)
