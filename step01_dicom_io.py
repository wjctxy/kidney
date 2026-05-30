"""DICOM metadata export and lazy frame reading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from pydicom.pixels import pixel_array as read_pixel_array


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    lo = float(np.nanmin(image))
    hi = float(np.nanmax(image))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    image = (image - lo) * (255.0 / (hi - lo))
    return np.clip(image, 0, 255).astype(np.uint8)


def prepare_frame(frame: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if frame.ndim == 3 and photometric.startswith("YBR"):
        frame = convert_color_space(frame, photometric, "RGB")
    if frame.ndim == 2 and photometric == "MONOCHROME1":
        frame = frame.max() - frame
    return normalize_uint8(np.asarray(frame))


class DicomFrameReader:
    def __init__(self, dicom_path: str | Path):
        self.path = Path(dicom_path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"DICOM file not found: {self.path}")
        self.ds = pydicom.dcmread(str(self.path), stop_before_pixels=True)
        self.frame_count = int(getattr(self.ds, "NumberOfFrames", 1) or 1)
        self.rows = int(getattr(self.ds, "Rows", 0) or 0)
        self.columns = int(getattr(self.ds, "Columns", 0) or 0)
        self.frame_time_ms = float(getattr(self.ds, "FrameTime", 33.3333333333333))
        self.fps = 1000.0 / self.frame_time_ms if self.frame_time_ms > 0 else 30.0

    def read_frame(self, index: int) -> np.ndarray:
        index = int(np.clip(index, 0, self.frame_count - 1))
        frame = read_pixel_array(str(self.path), index=index)
        return prepare_frame(frame, self.ds)

    def metadata(self) -> dict[str, Any]:
        file_meta = getattr(self.ds, "file_meta", None)
        transfer_syntax = getattr(file_meta, "TransferSyntaxUID", "N/A")
        region = self._region_metadata()
        return {
            "path": str(self.path),
            "file_size_mb": self.path.stat().st_size / (1024 * 1024),
            "modality": _json_value(getattr(self.ds, "Modality", "N/A")),
            "manufacturer": _json_value(getattr(self.ds, "Manufacturer", "N/A")),
            "model": _json_value(getattr(self.ds, "ManufacturerModelName", "N/A")),
            "rows": self.rows,
            "columns": self.columns,
            "number_of_frames": self.frame_count,
            "frame_time_ms": self.frame_time_ms,
            "fps": self.fps,
            "samples_per_pixel": _json_value(getattr(self.ds, "SamplesPerPixel", "N/A")),
            "photometric_interpretation": _json_value(
                getattr(self.ds, "PhotometricInterpretation", "N/A")
            ),
            "transfer_syntax_uid": str(transfer_syntax),
            "transfer_syntax_name": getattr(transfer_syntax, "name", "N/A"),
            "physical_delta_x": region.get("physical_delta_x"),
            "physical_delta_y": region.get("physical_delta_y"),
            "region": region,
        }

    def _region_metadata(self) -> dict[str, Any]:
        seq = getattr(self.ds, "SequenceOfUltrasoundRegions", None)
        if not seq:
            return {}
        item = seq[0]
        return {
            "x0": _json_value(getattr(item, "RegionLocationMinX0", None)),
            "y0": _json_value(getattr(item, "RegionLocationMinY0", None)),
            "x1": _json_value(getattr(item, "RegionLocationMaxX1", None)),
            "y1": _json_value(getattr(item, "RegionLocationMaxY1", None)),
            "physical_delta_x": _json_value(getattr(item, "PhysicalDeltaX", None)),
            "physical_delta_y": _json_value(getattr(item, "PhysicalDeltaY", None)),
            "physical_units_x": _json_value(getattr(item, "PhysicalUnitsXDirection", None)),
            "physical_units_y": _json_value(getattr(item, "PhysicalUnitsYDirection", None)),
        }


def export_metadata(dicom_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = DicomFrameReader(dicom_path)
    metadata = reader.metadata()

    txt_lines = ["DICOM METADATA", "=" * 80]
    txt_lines.extend(f"{key}: {value}" for key, value in metadata.items())
    (output_dir / "dicom_info.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    (output_dir / "dicom_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata

