"""样本 DICOM 与 mask 的统一路径管理。

约定：
    all_masks/<sample_id>/<sample_id>.dcm
    all_masks/<sample_id>/cortex_mask.npy
    all_masks/<sample_id>/exclude_mask.npy
    all_masks/<sample_id>/cortex_mask_overlay.png
    all_masks/<sample_id>/exclude_mask_overlay.png

human_dcm/ 只保存当前正在运行的最新中间结果，不按样本编号长期归档。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import ulm_config as config


ALL_MASKS_DIR = config.BASE_DIR / "all_masks"
DICOM_SUFFIXES = {".dcm", ".dicom"}


@dataclass(frozen=True)
class SamplePaths:
    """一个样本在 all_masks 中的 DICOM 与 mask 路径。"""

    sample_id: str
    sample_dir: Path
    dicom_path: Path
    cortex_mask: Path
    exclude_mask: Path
    cortex_overlay: Path
    exclude_overlay: Path


def sample_id_from_dicom(dicom_path: str | Path) -> str:
    """从 DICOM 文件名提取样本编号，例如 21.0.dcm -> 21.0。"""

    return Path(dicom_path).stem


def sample_paths(sample_id: str, all_masks_dir: str | Path = ALL_MASKS_DIR) -> SamplePaths:
    """返回样本在 all_masks 下的标准路径。"""

    root = Path(all_masks_dir)
    sample_dir = root / sample_id
    return SamplePaths(
        sample_id=sample_id,
        sample_dir=sample_dir,
        dicom_path=_find_sample_dicom_path(sample_dir, sample_id),
        cortex_mask=sample_dir / "cortex_mask.npy",
        exclude_mask=sample_dir / "exclude_mask.npy",
        cortex_overlay=sample_dir / "cortex_mask_overlay.png",
        exclude_overlay=sample_dir / "exclude_mask_overlay.png",
    )


def resolve_sample(
    sample: str | Path | None = None,
    all_masks_dir: str | Path = ALL_MASKS_DIR,
) -> SamplePaths:
    """根据样本编号或 DICOM 路径解析样本；sample 为空时要求 all_masks 只有一个样本。"""

    root = Path(all_masks_dir)
    root.mkdir(parents=True, exist_ok=True)
    if sample is None:
        samples = list_samples(root)
        if not samples:
            raise FileNotFoundError(f"未在 {root} 中找到样本 DICOM。")
        if len(samples) > 1:
            names = ", ".join(item.sample_id for item in samples)
            raise RuntimeError(f"all_masks 中有多个样本，请用 --sample 指定：{names}")
        return samples[0]

    sample_path = Path(sample)
    if sample_path.suffix.lower() in DICOM_SUFFIXES:
        sample_id = sample_id_from_dicom(sample_path)
        paths = sample_paths(sample_id, root)
        if sample_path.exists():
            register_dicom(sample_path, root)
        if not paths.dicom_path.exists():
            raise FileNotFoundError(f"样本 DICOM 不存在：{paths.dicom_path}")
        return paths

    paths = sample_paths(str(sample), root)
    if not paths.dicom_path.exists():
        raise FileNotFoundError(f"样本 {sample} 未找到 DICOM：{paths.dicom_path}")
    return paths


def list_samples(all_masks_dir: str | Path = ALL_MASKS_DIR) -> list[SamplePaths]:
    """列出 all_masks 中已有 DICOM 的样本。"""

    root = Path(all_masks_dir)
    if not root.exists():
        return []
    sample_ids: set[str] = set()
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in DICOM_SUFFIXES:
            sample_ids.add(path.stem)
        elif path.is_dir():
            if _find_sample_dicom_path(path, path.name).exists():
                sample_ids.add(path.name)
    return [sample_paths(sample_id, root) for sample_id in sorted(sample_ids)]


def register_dicom(dicom_path: str | Path, all_masks_dir: str | Path = ALL_MASKS_DIR) -> SamplePaths:
    """把 DICOM 登记到 all_masks/<sample_id>/，同名文件已存在时不重复拷贝。"""

    source = Path(dicom_path)
    if not source.exists():
        raise FileNotFoundError(source)
    sample_id = sample_id_from_dicom(source)
    target_dir = Path(all_masks_dir) / sample_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{sample_id}{source.suffix.lower() if source.suffix else '.dcm'}"
    if not target.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return sample_paths(sample_id, all_masks_dir)


def archive_dicom_into_sample_dir(
    paths: SamplePaths,
    all_masks_dir: str | Path = ALL_MASKS_DIR,
    move: bool = True,
) -> SamplePaths:
    """把 all_masks 根目录中的 DICOM 收进 all_masks/<sample_id>/ 样本目录。"""

    root = Path(all_masks_dir)
    paths.sample_dir.mkdir(parents=True, exist_ok=True)
    source = Path(paths.dicom_path)
    target = paths.sample_dir / f"{paths.sample_id}{source.suffix.lower() if source.suffix else '.dcm'}"
    if source.exists() and source.resolve() != target.resolve():
        if not target.exists():
            if move:
                shutil.move(str(source), str(target))
            else:
                shutil.copy2(source, target)
        elif move and source.parent == root:
            source.unlink()
    return sample_paths(paths.sample_id, root)


def mask_dir_for_sample(sample_id: str, all_masks_dir: str | Path = ALL_MASKS_DIR) -> Path:
    """返回该样本的 mask 目录。"""

    return Path(all_masks_dir) / sample_id


def has_required_masks(paths: SamplePaths) -> bool:
    """判断 cortex 和 exclude mask 是否都已存在。"""

    return paths.cortex_mask.exists() and paths.exclude_mask.exists()


def missing_mask_modes(paths: SamplePaths) -> list[str]:
    """返回缺失的 Step05 mask 类型。"""

    missing: list[str] = []
    if not paths.cortex_mask.exists():
        missing.append("cortex")
    if not paths.exclude_mask.exists():
        missing.append("exclude")
    return missing


def _find_sample_dicom_path(sample_dir: Path, sample_id: str) -> Path:
    """在样本目录中寻找 DICOM；不存在时返回标准目标路径。"""

    standard = sample_dir / f"{sample_id}.dcm"
    if standard.exists():
        return standard
    root_level = sample_dir.parent / f"{sample_id}.dcm"
    if root_level.exists():
        return root_level
    if sample_dir.exists():
        candidates = sorted(path for path in sample_dir.iterdir() if path.suffix.lower() in DICOM_SUFFIXES)
        if candidates:
            return candidates[0]
    return standard
