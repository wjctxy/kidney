"""Mouse Step 03：匈牙利追踪、速度分组、密度图和指标。

Input:
    mouse_dcm/step02_detect/mouse_detections.csv
    mouse_dcm/step00_preprocess/metadata.json

Output:
    mouse_dcm/step03_track_reconstruct/mouse_tracks.csv
    mouse_dcm/step03_track_reconstruct/mouse_low_speed_tracks.csv
    mouse_dcm/step03_track_reconstruct/mouse_high_speed_tracks.csv
    mouse_dcm/step03_track_reconstruct/mouse_density_total.png
    mouse_dcm/step03_track_reconstruct/mouse_density_low_speed.png
    mouse_dcm/step03_track_reconstruct/mouse_density_high_speed.png
    mouse_dcm/step03_track_reconstruct/mouse_metrics.csv
    mouse_dcm/step03_track_reconstruct/mouse_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ulm_io
from ulm_tracking import track_and_reconstruct


def run(
    detections_csv: Path | None = None,
    metadata_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """读取小鼠检测点，执行追踪、速度分组、密度图重建和指标计算。"""

    detections_csv = detections_csv or (ulm_io.step_dir("mouse", "step02_detect") / "mouse_detections.csv")
    metadata_path = metadata_path or ulm_io.default_metadata_path("mouse")
    output_dir = output_dir or ulm_io.step_dir("mouse", "step03_track_reconstruct")
    metadata = ulm_io.load_metadata(metadata_path)
    outputs = track_and_reconstruct(detections_csv, metadata, output_dir, prefix="mouse")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


def parse_args() -> argparse.Namespace:
    """解析 Mouse Step 03 的命令行参数。"""

    parser = argparse.ArgumentParser(description="Mouse Step 03: 追踪、重建和指标")
    parser.add_argument("--detections", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.detections, args.metadata, args.output_dir)
