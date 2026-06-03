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
    它读取 Step 03 生成的轨迹，根据轨迹平均速度划分高/低速，并累计轨迹点生成密度图。
    低速组按肾小球尺度筛选：慢速候选的平均位移不超过一个肾小球直径，
    且轨迹包围盒面积不超过 15 个像素。本 step 不做前 180 帧伪影扣除。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ulm_io
from ulm_tracking import reconstruct_density_and_metrics


def run(
    tracks_csv: Path | None = None,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """运行 Human Step 04：速度分组、密度图重建和指标计算。"""

    tracks_csv = tracks_csv or (ulm_io.step_dir("human", "step03_track") / "human_tracks.csv")
    metadata_path = metadata_path or ulm_io.default_metadata_path("human")
    output_dir = output_dir or ulm_io.step_dir("human", "step04_density_metrics")
    metadata = ulm_io.load_metadata(metadata_path)
    outputs = reconstruct_density_and_metrics(tracks_csv, metadata, output_dir, prefix="human")
    print(f"low_speed_tracks: {outputs['low_tracks']}")
    print(f"high_speed_tracks: {outputs['high_tracks']}")
    print(f"density_total: {outputs['density_total']}")
    print(f"density_low_speed: {outputs['density_low_speed']}")
    print(f"density_high_speed: {outputs['density_high_speed']}")
    print(f"density_speed_overlay: {outputs['density_speed_overlay']}")
    print(f"metrics: {outputs['metrics']}")
    print(f"summary: {outputs['summary']}")
    return outputs


def parse_args() -> argparse.Namespace:
    """解析 Human Step 04 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Human Step 04: 速度分组、密度图和指标")
    parser.add_argument("--tracks", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.tracks, args.metadata, args.output_dir)
