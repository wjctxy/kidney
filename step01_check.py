"""
Export DICOM metadata to a txt file.

Default example:
    python check.py

Use another DICOM file:
    python check.py D:\Data\Data杂\DICOM\11.0.dcm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pydicom


DEFAULT_DICOM_FILE = Path(__file__).resolve().parent / "11.0.dcm"


def format_value(value, max_length: int = 220) -> str:
    text = str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def get_keyword(ds: pydicom.Dataset, keyword: str, default: str = "N/A") -> str:
    return format_value(getattr(ds, keyword, default))


def build_summary_lines(ds: pydicom.Dataset, dicom_path: Path) -> list[str]:
    file_meta = getattr(ds, "file_meta", None)
    transfer_syntax = getattr(file_meta, "TransferSyntaxUID", "N/A")

    rows = getattr(ds, "Rows", "N/A")
    columns = getattr(ds, "Columns", "N/A")
    samples_per_pixel = getattr(ds, "SamplesPerPixel", "N/A")
    bits_allocated = getattr(ds, "BitsAllocated", "N/A")
    number_of_frames = getattr(ds, "NumberOfFrames", 1)

    lines = [
        "DICOM FILE SUMMARY",
        "=" * 80,
        f"File path: {dicom_path}",
        f"File size: {dicom_path.stat().st_size / (1024 * 1024):.2f} MB",
        "",
        "Patient / Study",
        "-" * 80,
        f"PatientName: {get_keyword(ds, 'PatientName')}",
        f"PatientID: {get_keyword(ds, 'PatientID')}",
        f"PatientSex: {get_keyword(ds, 'PatientSex')}",
        f"PatientBirthDate: {get_keyword(ds, 'PatientBirthDate')}",
        f"StudyDate: {get_keyword(ds, 'StudyDate')}",
        f"StudyTime: {get_keyword(ds, 'StudyTime')}",
        f"StudyDescription: {get_keyword(ds, 'StudyDescription')}",
        f"SeriesDescription: {get_keyword(ds, 'SeriesDescription')}",
        "",
        "Device / Acquisition",
        "-" * 80,
        f"Modality: {get_keyword(ds, 'Modality')}",
        f"Manufacturer: {get_keyword(ds, 'Manufacturer')}",
        f"ManufacturerModelName: {get_keyword(ds, 'ManufacturerModelName')}",
        f"DeviceSerialNumber: {get_keyword(ds, 'DeviceSerialNumber')}",
        f"SoftwareVersions: {get_keyword(ds, 'SoftwareVersions')}",
        f"InstitutionName: {get_keyword(ds, 'InstitutionName')}",
        "",
        "Image / Video",
        "-" * 80,
        f"Rows x Columns: {rows} x {columns}",
        f"NumberOfFrames: {number_of_frames}",
        f"SamplesPerPixel: {samples_per_pixel}",
        f"PhotometricInterpretation: {get_keyword(ds, 'PhotometricInterpretation')}",
        f"PlanarConfiguration: {get_keyword(ds, 'PlanarConfiguration')}",
        f"BitsAllocated: {bits_allocated}",
        f"BitsStored: {get_keyword(ds, 'BitsStored')}",
        f"HighBit: {get_keyword(ds, 'HighBit')}",
        f"PixelRepresentation: {get_keyword(ds, 'PixelRepresentation')}",
        f"FrameTime: {get_keyword(ds, 'FrameTime')}",
        f"CineRate: {get_keyword(ds, 'CineRate')}",
        "",
        "Encoding",
        "-" * 80,
        f"TransferSyntaxUID: {transfer_syntax}",
        f"TransferSyntaxName: {getattr(transfer_syntax, 'name', 'N/A')}",
        f"SOPClassUID: {get_keyword(ds, 'SOPClassUID')}",
        f"SOPInstanceUID: {get_keyword(ds, 'SOPInstanceUID')}",
        f"StudyInstanceUID: {get_keyword(ds, 'StudyInstanceUID')}",
        f"SeriesInstanceUID: {get_keyword(ds, 'SeriesInstanceUID')}",
        "",
    ]

    return lines


def iter_dataset_elements(ds: pydicom.Dataset) -> Iterable[str]:
    for elem in ds.iterall():
        if elem.tag == (0x7FE0, 0x0010):
            value = "<PixelData skipped>"
        else:
            value = format_value(elem.value)

        keyword = elem.keyword or "NoKeyword"
        yield (
            f"{elem.tag} | {elem.VR:<2} | {keyword:<35} | "
            f"{elem.name:<45} | {value}"
        )


def export_dicom_info(dicom_path: Path, output_path: Path | None = None) -> Path:
    dicom_path = dicom_path.expanduser().resolve()
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True)

    if output_path is None:
        output_path = Path(__file__).resolve().parent / f"{dicom_path.stem}_dicom_info.txt"
    else:
        output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = build_summary_lines(ds, dicom_path)
    lines.extend(
        [
            "ALL DICOM TAGS",
            "=" * 80,
            "Tag         | VR | Keyword                             | Name                                          | Value",
            "-" * 80,
        ]
    )
    lines.extend(iter_dataset_elements(ds))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.with_suffix(".json").write_text(
        json.dumps(build_json_summary(ds, dicom_path), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def build_json_summary(ds: pydicom.Dataset, dicom_path: Path) -> dict:
    file_meta = getattr(ds, "file_meta", None)
    transfer_syntax = getattr(file_meta, "TransferSyntaxUID", "N/A")
    return {
        "file_path": str(dicom_path),
        "file_size_mb": dicom_path.stat().st_size / (1024 * 1024),
        "modality": str(getattr(ds, "Modality", "N/A")),
        "manufacturer": str(getattr(ds, "Manufacturer", "N/A")),
        "model": str(getattr(ds, "ManufacturerModelName", "N/A")),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "number_of_frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
        "frame_time_ms": float(getattr(ds, "FrameTime", 0) or 0),
        "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "N/A")),
        "transfer_syntax_uid": str(transfer_syntax),
        "transfer_syntax_name": getattr(transfer_syntax, "name", "N/A"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DICOM metadata to txt.")
    parser.add_argument(
        "dicom_file",
        nargs="?",
        type=Path,
        default=DEFAULT_DICOM_FILE,
        help=f"DICOM file path. Default: {DEFAULT_DICOM_FILE}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output txt path. Default: DICOM/<dicom_name>_dicom_info.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_dicom_info(args.dicom_file, args.output)
    print(f"DICOM information saved to: {output_path}")
    print(f"DICOM JSON summary saved to: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
