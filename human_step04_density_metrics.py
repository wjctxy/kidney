"""Human Step 04：速度分组、密度图重建和指标计算。

Input:
    human_dcm/step03_track/human_tracks.csv
    human_dcm/step00_preprocess/metadata.json

Output:
    human_dcm/step04_density_metrics/human_low_speed_tracks.csv
    human_dcm/step04_density_metrics/human_high_speed_tracks.csv
    human_dcm/step04_density_metrics/human_density_total.png
    human_dcm/step04_density_metrics/human_density_low_speed.png
    human_dcm/step04_density_metrics/human_density_high_speed.png
    human_dcm/step04_density_metrics/human_density_speed_overlay.png
    human_dcm/step04_density_metrics/human_metrics.csv
    human_dcm/step04_density_metrics/human_summary.txt

算法说明:
    本 step 不再检测微泡，也不再做匈牙利匹配。
    默认兼容旧逻辑：读取单个 tracks.csv 后按速度/尺度分组。
    若同时提供 rapid_tracks_csv 和 slow_tracks_csv，则按 Akebia human profile
    直接合成 rapid/slow 密度图，不再用速度阈值二次切分。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ulm_io
from ulm_tracking import reconstruct_density_and_metrics, reconstruct_profile_density_and_metrics


def run(
    tracks_csv: Path | None = None,
    rapid_tracks_csv: Path | None = None,
    slow_tracks_csv: Path | None = None,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """运行 Human Step 04：速度分组或 Akebia profile 密度图重建和指标计算。"""

    tracks_csv = tracks_csv or (ulm_io.step_dir("human", "step03_track") / "human_tracks.csv")
    metadata_path = metadata_path or ulm_io.default_metadata_path("human")
    output_dir = output_dir or ulm_io.step_dir("human", "step04_density_metrics")
    metadata = ulm_io.load_metadata(metadata_path)

    if rapid_tracks_csv is not None and slow_tracks_csv is not None:
        outputs = reconstruct_profile_density_and_metrics(rapid_tracks_csv, slow_tracks_csv, metadata, output_dir, prefix="human")
        _print_outputs(outputs, ("rapid_tracks", "slow_tracks", "density_total", "density_rapid", "density_slow", "density_profile_overlay"))
    else:
        outputs = reconstruct_density_and_metrics(tracks_csv, metadata, output_dir, prefix="human")
        _print_outputs(outputs, ("low_tracks", "high_tracks", "density_total", "density_low_speed", "density_high_speed", "density_speed_overlay"))
    return outputs


def _print_outputs(outputs: dict[str, Path], keys: tuple[str, ...]) -> None:
    """按固定顺序打印本 step 产生的输出路径。"""

    labels = {"low_tracks": "low_speed_tracks", "high_tracks": "high_speed_tracks"}
    for key in (*keys, "metrics", "summary"):
        print(f"{labels.get(key, key)}: {outputs[key]}")


def parse_args() -> argparse.Namespace:
    """解析 Human Step 04 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Human Step 04: 速度分组、密度图和指标")
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--rapid-tracks", type=Path, default=None)
    parser.add_argument("--slow-tracks", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.tracks, args.rapid_tracks, args.slow_tracks, args.metadata, args.output_dir)
