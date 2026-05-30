"""One-command VINNO DICOM ULM processing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import step00_config as config
from step01_dicom_io import DicomFrameReader, export_metadata
from step02_extract_roi import extract_roi_previews
from step03_preprocess import preprocess_frames
from step04_filter_signal import filter_frames
from step05_detect_bubbles import detect_bubbles, make_detection_preview
from step06_track_bubbles import track_bubbles
from step07_reconstruct_maps import reconstruct_density_maps
from step07_visualize_results import make_tracks_preview
from step08_metrics import compute_metrics


def run_pipeline(dicom_path: Path, output_dir: Path, max_frames: int | None = None) -> None:
    metadata_dir = output_dir / "metadata"
    roi_dir = output_dir / "roi"
    preprocess_dir = output_dir / "preprocess"
    filter_dir = output_dir / "filter"
    cache_dir = output_dir / "cache"
    detection_dir = output_dir / "detections"
    track_dir = output_dir / "tracks"
    map_dir = output_dir / "maps"
    metric_dir = output_dir / "metrics"

    print("[1/9] Exporting DICOM metadata")
    metadata = export_metadata(dicom_path, metadata_dir)
    reader = DicomFrameReader(dicom_path)

    print("[2/9] Extracting ROI previews")
    extract_roi_previews(dicom_path, roi_dir)

    print("[3/9] Preprocessing ROI frames")
    preprocessed_path = preprocess_frames(dicom_path, preprocess_dir, cache_dir, max_frames=max_frames)

    print("[4/9] Temporal filtering")
    low_path, high_path = filter_frames(preprocessed_path, filter_dir, cache_dir, fps=float(metadata["fps"]))

    print("[5/9] Detecting microbubble centers")
    low_points = detect_bubbles(low_path, detection_dir / "low_speed_points.csv", "low", reader.frame_time_ms)
    high_points = detect_bubbles(high_path, detection_dir / "high_speed_points.csv", "high", reader.frame_time_ms)
    make_detection_preview(low_path, low_points, detection_dir / "detection_preview.mp4", reader.fps)

    print("[6/9] Tracking microbubbles")
    low_tracks = track_bubbles(low_points, track_dir / "low_speed_tracks.csv", "low")
    high_tracks = track_bubbles(high_points, track_dir / "high_speed_tracks.csv", "high")
    make_tracks_preview(preprocessed_path, [low_tracks, high_tracks], track_dir / "tracks_preview.mp4", reader.fps)

    print("[7/9] Reconstructing density maps")
    shape = (config.ROI_Y1 - config.ROI_Y0, config.ROI_X1 - config.ROI_X0)
    reconstruct_density_maps(low_tracks, high_tracks, map_dir, shape)

    print("[8/9] Computing metrics")
    compute_metrics(low_tracks, high_tracks, metric_dir)

    print("[9/9] Done")
    print(f"Results saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VINNO DICOM ULM pipeline.")
    parser.add_argument("--dicom", type=Path, default=config.DICOM_PATH, help="Input DICOM file.")
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.dicom, args.output, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
