"""Step 0：DICOM 预处理，并用 TIC 选择连续稳定造影窗口。

TIC(time-intensity curve) 是每一帧 ROI 内平均 CEUS score 随时间变化的曲线。
如果 masks/ 目录下存在 cortex_mask.npy，则 TIC 只统计 cortex mask 内部像素；
否则 TIC 退回到完整 ROI 平均值。
本 step 先对完整 DICOM 计算 TIC，再选择一段 TIC 强度足够、且波动尽量小的
连续源帧窗口，最后把该窗口转成后续算法统一使用的 frames.npy。

Input:
    human_dcm/11.0.dcm 或 mouse_dcm/11.0.dcm

Output:
    human_dcm/step00_preprocess/frames.npy: float32 [T,H,W]，单通道 CEUS score/灰度、裁剪 ROI、归一化到 [0,1]
    human_dcm/step00_preprocess/metadata.json: fps、像素尺寸、ROI、TIC 选帧信息等元数据
    human_dcm/step00_preprocess/tic_raw.npy: float32 [source_T]，完整源 DICOM 的 TIC
    human_dcm/step00_preprocess/tic_selection.png: TIC 与被选连续窗口的可视化
    human_dcm/step00_preprocess/preview_frame_000.png: 预处理后首帧预览

直接运行示例:
    python step00_preprocess.py --kind human
    python step00_preprocess.py --kind mouse
    python step00_preprocess.py --kind human --target-frames 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap

import ulm_config as config
import ulm_io
import ulm_sample_manager
from ulm_visualization import save_frame_png_raw_scale


DEFAULT_TARGET_FRAMES = 200
DEFAULT_MIN_FRAMES = 200
DEFAULT_SMOOTH_WINDOW = 15
DEFAULT_BASELINE_SECONDS = 2.0
DEFAULT_MIN_ENHANCEMENT = 0.01


def run(
    kind: str,
    dicom: Path | None = None,
    mask_dir: Path | None = None,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    baseline_seconds: float = DEFAULT_BASELINE_SECONDS,
    min_enhancement: float = DEFAULT_MIN_ENHANCEMENT,
    target_tic: float | None = None,
) -> tuple[Path, Path]:
    """执行 Step0：计算 TIC，选择稳定连续窗口，并保存统一算法输入。"""

    data_dir = ulm_io.step_dir(kind, "step00_preprocess")
    dicom_path = dicom or ulm_io.default_dicom_path(kind)
    frames_path = data_dir / "frames.npy"
    metadata_path = data_dir / "metadata.json"

    ds, metadata = ulm_io.read_dicom_metadata(dicom_path)
    color_space = ulm_io._resolve_color_space_handling(dicom_path, ds)
    metadata["color_space_handling"] = color_space
    n_source_frames = int(metadata["number_of_frames"])
    roi = metadata["roi"]
    fps = float(metadata["fps"]) if float(metadata["fps"]) > 0 else config.DEFAULT_FPS
    tic_mask, tic_mask_info = load_cortex_tic_mask((int(metadata["height"]), int(metadata["width"])), mask_dir=mask_dir)
    metadata["tic_mask"] = tic_mask_info

    print(f"[step00] computing TIC for {n_source_frames} source frame(s)")
    if tic_mask is None:
        print("[step00] TIC uses full ROI because no cortex mask was found")
    else:
        print(f"[step00] TIC uses cortex mask: {tic_mask_info['path']}")
    tic = compute_tic(dicom_path, roi, n_source_frames, color_space, tic_mask)
    tic_path = data_dir / "tic_raw.npy"
    np.save(tic_path, tic)

    selection = select_stable_tic_window(
        tic=tic,
        fps=fps,
        target_frames=target_frames,
        min_frames=min_frames,
        smooth_window=smooth_window,
        baseline_seconds=baseline_seconds,
        min_enhancement=min_enhancement,
        target_tic=target_tic,
    )
    selected_frames = selection["selected_source_frames"]
    if not selected_frames:
        raise ValueError("TIC 选帧没有得到任何帧。")

    save_tic_selection_plot(
        tic=tic,
        smoothed=selection["tic_smoothed"],
        selection=selection,
        output_path=data_dir / "tic_selection.png",
    )

    print(f"[step00] saving {len(selected_frames)} selected frame(s) to frames.npy")
    save_selected_frames(
        dicom_path=dicom_path,
        metadata=metadata,
        selected_frames=selected_frames,
        frames_path=frames_path,
    )

    metadata.update(
        {
            "frames_saved": int(len(selected_frames)),
            "source_frame_start": int(selection["selected_start"]),
            "source_frame_end_exclusive": int(selection["selected_end"] + 1),
            "source_frame_end_inclusive": int(selection["selected_end"]),
            "frame_selection_method": "tic_absolute_stable_window",
            "source_frame_indices": selected_frames,
            "selected_source_frames": selected_frames,
            "selected_ranges": {
                "selected_start": int(selection["selected_start"]),
                "selected_end": int(selection["selected_end"]),
                "selected_source_frames": selected_frames,
            },
            "frame_selection_parameters": selection["parameters"],
            "frame_selection_stats": selection["stats"],
            "frame_array_note": (
                "frames.npy contains one continuous source-frame window selected by TIC absolute "
                "enhancement and stability; source_frame_indices maps output frame index to original DICOM frame index."
            ),
        }
    )
    ulm_io.save_metadata(metadata, metadata_path)
    save_selection_json(selection, data_dir / "selected_ranges.json")

    frames = ulm_io.load_frames(frames_path)
    preview_path = data_dir / "preview_frame_000.png"
    save_frame_png_raw_scale(frames[0], preview_path, "Step 0 TIC-selected preprocessed frame")
    projection_paths = save_step00_temporal_projections(frames, data_dir)

    print(f"frames: {frames_path}")
    print(f"metadata: {metadata_path}")
    print(f"preview: {preview_path}")
    for name, path in projection_paths.items():
        print(f"{name}: {path}")
    print(f"selected_ranges: {data_dir / 'selected_ranges.json'}")
    print(f"tic: {tic_path}")
    print(f"tic_selection: {data_dir / 'tic_selection.png'}")
    return frames_path, metadata_path


def compute_tic(
    dicom_path: Path,
    roi: dict[str, int],
    n_frames: int,
    color_space: dict[str, Any],
    tic_mask: np.ndarray | None = None,
) -> np.ndarray:
    """计算完整源 DICOM 的 TIC；有 cortex mask 时只统计 mask 内像素。"""

    tic = np.zeros(n_frames, dtype=np.float32)
    for frame_id in range(n_frames):
        frame = ulm_io._read_frame_pixels(dicom_path, frame_id, color_space)
        prepared = ulm_io._prepare_frame(frame)
        cropped = prepared[roi["y0"] : roi["y1"], roi["x0"] : roi["x1"]]
        normalized = cropped.astype(np.float32) / 255.0
        if tic_mask is None:
            tic[frame_id] = float(np.mean(normalized))
        else:
            tic[frame_id] = float(np.mean(normalized[tic_mask]))
    return tic


def load_cortex_tic_mask(
    target_shape: tuple[int, int],
    mask_dir: Path | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """读取 cortex_mask，并转换到 Step00 裁剪帧尺寸；不存在时返回 None。"""

    return load_cortex_tic_mask_from_dir(target_shape, mask_dir or (config.BASE_DIR / "masks"))


def load_cortex_tic_mask_from_dir(target_shape: tuple[int, int], mask_dir: Path | None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """从指定目录读取 cortex_mask，并转换到 Step00 裁剪帧尺寸。"""

    if mask_dir is None:
        return load_cortex_tic_mask(target_shape)
    candidates = [Path(mask_dir) / "cortex_mask.npy", Path(mask_dir) / "cortex_mask"]
    existing = next((path for path in candidates if path.exists()), None)
    if existing is None:
        return None, {
            "mode": "full_roi",
            "reason": "no_cortex_mask_found",
            "searched": [str(path) for path in candidates],
        }

    source_mask = np.load(existing).astype(bool)
    if source_mask.ndim != 2:
        raise ValueError(f"cortex mask 必须是二维数组，当前 shape={source_mask.shape}")
    resized_mask = resize_bool_mask(source_mask, target_shape)
    included_pixels = int(np.count_nonzero(resized_mask))
    if included_pixels == 0:
        raise ValueError(f"cortex mask 下采样到 Step00 尺寸后为空：{existing}")

    return resized_mask, {
        "mode": "cortex_mask",
        "path": str(existing.resolve()),
        "source_shape": [int(source_mask.shape[0]), int(source_mask.shape[1])],
        "target_shape": [int(target_shape[0]), int(target_shape[1])],
        "included_pixels": included_pixels,
        "included_fraction": float(included_pixels / max(int(target_shape[0]) * int(target_shape[1]), 1)),
    }


def resize_bool_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """把 bool mask 转成目标尺寸；整数倍超分辨率 mask 用块平均下采样。"""

    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    source_h, source_w = int(mask.shape[0]), int(mask.shape[1])
    if (source_h, source_w) == (target_h, target_w):
        return mask.astype(bool, copy=True)

    if source_h % target_h == 0 and source_w % target_w == 0:
        scale_y = source_h // target_h
        scale_x = source_w // target_w
        blocks = mask.reshape(target_h, scale_y, target_w, scale_x)
        return (np.mean(blocks, axis=(1, 3)) >= 0.5).astype(bool)

    y_idx = np.clip(((np.arange(target_h) + 0.5) * source_h / target_h).astype(int), 0, source_h - 1)
    x_idx = np.clip(((np.arange(target_w) + 0.5) * source_w / target_w).astype(int), 0, source_w - 1)
    return mask[np.ix_(y_idx, x_idx)].astype(bool)


def save_selected_frames(
    dicom_path: Path,
    metadata: dict[str, Any],
    selected_frames: list[int],
    frames_path: Path,
) -> None:
    """把选中的连续源帧窗口转换为 float32 [T,H,W] 并保存到 frames.npy。"""

    roi = metadata["roi"]
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    out = open_memmap(
        frames_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected_frames), int(metadata["height"]), int(metadata["width"])),
    )
    for out_id, source_frame_id in enumerate(selected_frames):
        frame = ulm_io._read_frame_pixels(dicom_path, source_frame_id, metadata["color_space_handling"])
        prepared = ulm_io._prepare_frame(frame)
        cropped = prepared[roi["y0"] : roi["y1"], roi["x0"] : roi["x1"]]
        out[out_id] = cropped.astype(np.float32) / 255.0
    out.flush()


def select_stable_tic_window(
    tic: np.ndarray,
    fps: float,
    target_frames: int,
    min_frames: int,
    smooth_window: int,
    baseline_seconds: float,
    min_enhancement: float,
    target_tic: float | None,
) -> dict[str, Any]:
    """按绝对帧数和 TIC 稳定性选择连续窗口，不使用总帧数比例。"""

    if tic.ndim != 1 or tic.size == 0:
        raise ValueError("tic 必须是一维非空数组。")

    total_frames = int(tic.size)
    target_frames = max(1, int(target_frames))
    min_frames = max(1, int(min_frames))
    window_frames = min(total_frames, max(target_frames, min_frames))
    if total_frames < min_frames:
        print(f"[step00] warning: source DICOM has only {total_frames} frame(s), below min_frames={min_frames}")

    smoothed = smooth_1d(tic, smooth_window)
    baseline, baseline_end = estimate_baseline_by_seconds(smoothed, fps, baseline_seconds)
    enhancement_threshold = float(baseline + float(min_enhancement))
    peak_frame = int(np.argmax(smoothed))
    peak_value = float(smoothed[peak_frame])

    starts = np.arange(0, total_frames - window_frames + 1, dtype=np.int32)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int32)

    scores = score_candidate_windows(
        smoothed=smoothed,
        starts=starts,
        window_frames=window_frames,
        enhancement_threshold=enhancement_threshold,
        target_tic=target_tic,
    )
    best_pos = int(np.argmin(scores))
    selected_start = int(starts[best_pos])
    selected_end = int(selected_start + window_frames - 1)
    selected_frames = list(range(selected_start, selected_end + 1))
    selected_tic = smoothed[selected_start : selected_end + 1]
    below_threshold_fraction = float(np.mean(selected_tic < enhancement_threshold))
    selected_mean = float(np.mean(selected_tic))
    selected_std = float(np.std(selected_tic))
    selected_cv = float(selected_std / max(abs(selected_mean), 1e-6))

    return {
        "tic_frame_count": total_frames,
        "tic_smoothed": smoothed,
        "baseline": float(baseline),
        "baseline_end_frame": int(baseline_end),
        "enhancement_threshold": enhancement_threshold,
        "peak_frame": peak_frame,
        "peak_value": peak_value,
        "selected_start": selected_start,
        "selected_end": selected_end,
        "selected_source_frames": selected_frames,
        "parameters": {
            "target_frames": int(target_frames),
            "min_frames": int(min_frames),
            "actual_window_frames": int(window_frames),
            "smooth_window": int(smooth_window),
            "baseline_seconds": float(baseline_seconds),
            "min_enhancement": float(min_enhancement),
            "target_tic": target_tic,
        },
        "stats": {
            "selected_tic_mean": selected_mean,
            "selected_tic_std": selected_std,
            "selected_tic_cv": selected_cv,
            "selected_tic_min": float(np.min(selected_tic)),
            "selected_tic_max": float(np.max(selected_tic)),
            "selected_tic_start": float(selected_tic[0]),
            "selected_tic_end": float(selected_tic[-1]),
            "below_threshold_fraction": below_threshold_fraction,
            "window_score": float(scores[best_pos]),
        },
    }


def score_candidate_windows(
    smoothed: np.ndarray,
    starts: np.ndarray,
    window_frames: int,
    enhancement_threshold: float,
    target_tic: float | None,
) -> np.ndarray:
    """给每个候选连续窗口打分，低分代表 TIC 更稳定且增强强度更合适。"""

    scores = np.zeros(starts.shape, dtype=np.float64)
    for i, start in enumerate(starts):
        end = int(start) + int(window_frames)
        window = smoothed[int(start) : end]
        mean_value = float(np.mean(window))
        std_value = float(np.std(window))
        drift = float(abs(window[-1] - window[0]))
        below_fraction = float(np.mean(window < enhancement_threshold))
        score = std_value + 0.5 * drift + 5.0 * below_fraction
        if target_tic is None:
            score -= 0.02 * mean_value
        else:
            score += abs(mean_value - float(target_tic))
        scores[i] = score
    return scores


def smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    """用居中移动平均平滑一维 TIC，降低单帧噪声对选窗的影响。"""

    arr = np.asarray(values, dtype=np.float32)
    window = max(1, int(window))
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(arr, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def estimate_baseline_by_seconds(smoothed: np.ndarray, fps: float, baseline_seconds: float) -> tuple[float, int]:
    """用开头固定秒数估计背景 baseline，而不是用总帧数比例。"""

    n = int(smoothed.size)
    baseline_frames = int(round(max(float(baseline_seconds), 0.0) * max(float(fps), 1e-6)))
    end = max(1, min(n, baseline_frames if baseline_frames > 0 else 1))
    return float(np.median(smoothed[:end])), end - 1


def save_tic_selection_plot(
    tic: np.ndarray,
    smoothed: np.ndarray,
    selection: dict[str, Any],
    output_path: Path,
) -> None:
    """保存 TIC 选窗诊断图，橙色区域为写入 frames.npy 的连续源帧窗口。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    frames = np.arange(tic.size)
    ax.plot(frames, tic, color="0.65", linewidth=1.0, label="raw TIC")
    ax.plot(frames, smoothed, color="tab:red", linewidth=1.5, label="smoothed TIC")
    ax.axvspan(selection["selected_start"], selection["selected_end"], color="tab:orange", alpha=0.18, label="saved window")
    ax.axvline(selection["peak_frame"], color="purple", linewidth=1.2, label="peak")
    ax.axhline(selection["baseline"], color="tab:blue", linewidth=1.0, alpha=0.7, label="baseline")
    ax.axhline(
        selection["enhancement_threshold"],
        color="tab:green",
        linewidth=1.0,
        alpha=0.8,
        label="baseline + min enhancement",
    )
    ax.set_xlabel("source DICOM frame index")
    ax.set_ylabel("mean CEUS score")
    ax.set_title("Step00 TIC stable continuous-window selection")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_selection_json(selection: dict[str, Any], path: Path) -> None:
    """保存可复查的 TIC 选帧信息，去掉不能直接 JSON 序列化的平滑数组。"""

    serializable = dict(selection)
    serializable.pop("tic_smoothed", None)
    path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")


def save_step00_temporal_projections(frames: np.ndarray, output_dir: Path) -> dict[str, Path]:
    """保存 Step00 选窗内 200 帧的时间聚合图，供 mask-only 画 mask 使用。"""

    projections = {
        "mean": np.mean(frames, axis=0).astype(np.float32),
        "max": np.max(frames, axis=0).astype(np.float32),
        "std": np.std(frames, axis=0).astype(np.float32),
    }
    paths: dict[str, Path] = {}
    for name, projection in projections.items():
        npy_path = output_dir / f"step00_{name}_projection.npy"
        png_path = output_dir / f"step00_{name}_projection.png"
        np.save(npy_path, projection)
        save_plain_projection_png(projection, png_path)
        paths[f"step00_{name}_projection_npy"] = npy_path
        paths[f"step00_{name}_projection_png"] = png_path
    return paths


def save_plain_projection_png(image: np.ndarray, path: Path) -> None:
    """保存无坐标轴、无 colorbar 的 projection PNG，避免画 mask 时坐标错位。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    display = normalize_projection_for_display(image)
    plt.imsave(path, display, cmap="gray", vmin=0.0, vmax=1.0)


def normalize_projection_for_display(image: np.ndarray) -> np.ndarray:
    """把 projection 映射到 [0,1] 仅用于 PNG 人工预览。"""

    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.zeros(arr.shape, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def run_human_full_pipeline(
    sample: str | Path | None,
    all_masks_dir: Path,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    baseline_seconds: float = DEFAULT_BASELINE_SECONDS,
    min_enhancement: float = DEFAULT_MIN_ENHANCEMENT,
    target_tic: float | None = None,
    count_mode: str = "healthy_calibration",
    dbscan_eps_mm: float | None = None,
    rerun_after_mask: bool = True,
    clean_outputs: bool = True,
) -> dict[str, Any]:
    """从 all_masks 中取一个人类样本，运行 Step00-05，并按需交互式补 mask。"""

    sample_paths = _resolve_or_register_sample(sample, all_masks_dir)
    had_required_masks = ulm_sample_manager.has_required_masks(sample_paths)
    if clean_outputs:
        clean_human_current_outputs()

    frames_path, metadata_path = _run_step00_for_sample(
        sample_paths=sample_paths,
        target_frames=target_frames,
        min_frames=min_frames,
        smooth_window=smooth_window,
        baseline_seconds=baseline_seconds,
        min_enhancement=min_enhancement,
        target_tic=target_tic,
    )
    _run_human_steps01_to04(frames_path, metadata_path)

    missing_modes = ulm_sample_manager.missing_mask_modes(sample_paths)
    if missing_modes:
        _draw_masks(sample_paths, missing_modes)
        sample_paths = ulm_sample_manager.archive_dicom_into_sample_dir(sample_paths, all_masks_dir)
        if rerun_after_mask:
            print("[pipeline] mask 已补齐，重新从 Step00 开始使用 cortex TIC 跑最终结果")
            if clean_outputs:
                clean_human_current_outputs()
            frames_path, metadata_path = _run_step00_for_sample(
                sample_paths=sample_paths,
                target_frames=target_frames,
                min_frames=min_frames,
                smooth_window=smooth_window,
                baseline_seconds=baseline_seconds,
                min_enhancement=min_enhancement,
                target_tic=target_tic,
            )
            _run_human_steps01_to04(frames_path, metadata_path)

    step5_outputs = _run_step05_for_sample(sample_paths, count_mode=count_mode, dbscan_eps_mm=dbscan_eps_mm)
    sample_paths = ulm_sample_manager.archive_dicom_into_sample_dir(sample_paths, all_masks_dir)
    summary_path = config.BASE_DIR / "human_dcm" / "step05_glomeruli_count" / "summary.json"
    _save_healthy_run_summary(sample_paths.sample_id, summary_path)
    return {
        "sample_id": sample_paths.sample_id,
        "dicom": str(sample_paths.dicom_path),
        "had_required_masks": had_required_masks,
        "step5": step5_outputs,
        "summary": str(summary_path),
    }


def run_human_mask_only(
    sample: str | Path | None,
    all_masks_dir: Path,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    baseline_seconds: float = DEFAULT_BASELINE_SECONDS,
    min_enhancement: float = DEFAULT_MIN_ENHANCEMENT,
    target_tic: float | None = None,
    clean_outputs: bool = True,
) -> dict[str, Any]:
    """只为指定人类 DICOM 重画 Step05 mask，不继续执行 Step05 计数。"""

    sample_paths = _resolve_or_register_sample(sample, all_masks_dir)
    if clean_outputs:
        clean_human_current_outputs()

    frames_path, metadata_path = _run_step00_for_sample(
        sample_paths=sample_paths,
        target_frames=target_frames,
        min_frames=min_frames,
        smooth_window=smooth_window,
        baseline_seconds=baseline_seconds,
        min_enhancement=min_enhancement,
        target_tic=target_tic,
    )
    _run_human_steps01_to04(frames_path, metadata_path)

    _draw_masks(sample_paths, ["cortex", "exclude"])

    sample_paths = ulm_sample_manager.archive_dicom_into_sample_dir(sample_paths, all_masks_dir)
    print(f"[pipeline] DICOM 已归档到样本目录：{sample_paths.dicom_path}")
    return {
        "sample_id": sample_paths.sample_id,
        "dicom": str(sample_paths.dicom_path),
        "mask_dir": str(sample_paths.sample_dir),
        "cortex_mask": str(sample_paths.cortex_mask),
        "exclude_mask": str(sample_paths.exclude_mask),
        "step04_density_slow": str(config.HUMAN_DIR / "step04_density_metrics" / "human_density_slow.png"),
    }


def clean_human_current_outputs() -> None:
    """清理当前 human_dcm 的 Step00-05 输出，只保留最新一次中间结果。"""

    if not config.HUMAN_DIR.exists():
        return
    for path in config.HUMAN_DIR.iterdir():
        if not path.is_dir() or not path.name.startswith("step"):
            continue
        if path.exists():
            shutil.rmtree(path)


def _resolve_or_register_sample(sample: str | Path | None, all_masks_dir: Path) -> ulm_sample_manager.SamplePaths:
    """解析样本；如果传入 DICOM 路径，则登记到 all_masks/<sample_id>/。"""

    if sample is not None and Path(sample).suffix.lower() in ulm_sample_manager.DICOM_SUFFIXES:
        return ulm_sample_manager.register_dicom(Path(sample), all_masks_dir)
    return ulm_sample_manager.resolve_sample(sample, all_masks_dir)


def _run_step00_for_sample(
    sample_paths: ulm_sample_manager.SamplePaths,
    target_frames: int,
    min_frames: int,
    smooth_window: int,
    baseline_seconds: float,
    min_enhancement: float,
    target_tic: float | None,
) -> tuple[Path, Path]:
    """对指定样本运行 Step00；若已有 cortex mask，则 TIC 只统计 mask 内部。"""

    print(f"[pipeline] Step00 sample={sample_paths.sample_id} dicom={sample_paths.dicom_path}")
    return run(
        kind="human",
        dicom=sample_paths.dicom_path,
        mask_dir=sample_paths.sample_dir,
        target_frames=target_frames,
        min_frames=min_frames,
        smooth_window=smooth_window,
        baseline_seconds=baseline_seconds,
        min_enhancement=min_enhancement,
        target_tic=target_tic,
    )


def _run_human_steps01_to04(frames_path: Path, metadata_path: Path) -> None:
    """运行 Human rapid/slow Step01-03，并合成 Step04 密度和轨迹指标。"""

    import human_step01_bandpass
    import human_step02_gaussian_filter
    import human_step03_track
    import human_step04_density_metrics

    base = config.HUMAN_DIR
    profile_tracks: dict[str, Path] = {}
    for profile in ("rapid", "slow"):
        print(f"[pipeline] Step01 profile={profile}")
        filtered_path = human_step01_bandpass.run(
            frames_path=frames_path,
            metadata_path=metadata_path,
            output_path=base / "step01_bandpass" / profile / "human_filtered.npy",
            profile=profile,
        )
        print(f"[pipeline] Step02 profile={profile}")
        smoothed_path = human_step02_gaussian_filter.run(
            frames_path=filtered_path,
            metadata_path=metadata_path,
            output_path=base / "step02_gaussian_filter" / profile / "human_smoothed.npy",
        )
        print(f"[pipeline] Step03 profile={profile}")
        outputs = human_step03_track.run(
            frames_path=filtered_path,
            smooth_frames_path=smoothed_path,
            metadata_path=metadata_path,
            output_dir=base / "step03_track" / profile,
            preview_frame=100,
            profile=profile,
        )
        profile_tracks[profile] = outputs["tracks"]

    print("[pipeline] Step04")
    human_step04_density_metrics.run(
        rapid_tracks_csv=profile_tracks["rapid"],
        slow_tracks_csv=profile_tracks["slow"],
        metadata_path=metadata_path,
        output_dir=base / "step04_density_metrics",
    )


def _draw_masks(sample_paths: ulm_sample_manager.SamplePaths, mask_modes: list[str]) -> None:
    """使用当前流程输出作为背景，交互式绘制指定类型的样本 mask。"""

    from human_step05_glomeruli_count import draw_mask_from_image

    for mode in mask_modes:
        output_path = sample_paths.cortex_mask if mode == "cortex" else sample_paths.exclude_mask
        overlay_path = sample_paths.cortex_overlay if mode == "cortex" else sample_paths.exclude_overlay
        image_path = select_mask_background(mode)
        print(f"[pipeline] 使用背景 {image_path} 开始交互式绘制 {mode} mask：{output_path}")
        draw_mask_from_image(image_path, output_path, overlay_path, mode)


def select_mask_background(mode: str) -> Path:
    """按 mask 类型选择交互绘制背景图。"""

    if mode == "cortex":
        return _required_background(config.HUMAN_DIR / "step00_preprocess" / "step00_mean_projection.npy")
    if mode == "exclude":
        return select_exclude_mask_background()
    raise ValueError("mode 必须是 cortex 或 exclude")


def select_exclude_mask_background() -> Path:
    """优先用 Step04 轨迹聚合图画排除区，密度太弱时回退到 Step00 mean projection。"""

    candidates = [
        config.HUMAN_DIR / "step04_density_metrics" / "human_density_profile_overlay.png",
        config.HUMAN_DIR / "step04_density_metrics" / "human_density_total.png",
        config.HUMAN_DIR / "step04_density_metrics" / "human_density_rapid.png",
    ]
    for candidate in candidates:
        if candidate.exists() and image_has_enough_signal(candidate):
            return candidate
    return _required_background(config.HUMAN_DIR / "step00_preprocess" / "step00_mean_projection.npy")


def image_has_enough_signal(path: Path) -> bool:
    """判断 PNG 是否有足够非黑像素作为画 exclude mask 的背景。"""

    image = plt.imread(path)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0 or float(np.nanmax(arr)) <= 0:
        return False
    threshold = max(float(np.nanmax(arr)) * 0.03, 1e-6)
    return float(np.mean(arr > threshold)) >= 0.005


def _required_background(path: Path) -> Path:
    """返回必须存在的绘制背景路径。"""

    if not path.exists():
        raise FileNotFoundError(f"mask 绘制背景不存在：{path}")
    return path


def _run_step05_for_sample(
    sample_paths: ulm_sample_manager.SamplePaths,
    count_mode: str,
    dbscan_eps_mm: float | None,
) -> dict[str, Any]:
    """使用样本目录中的 mask 运行 Step05 计数。"""

    from human_step05_glomeruli_count import run_step5_from_cli_defaults

    step5_config: dict[str, Any] = {}
    if count_mode == "diagnostic":
        step5_config["calibration_enabled"] = False
    if dbscan_eps_mm is not None:
        step5_config["dbscan_eps_mm"] = float(dbscan_eps_mm)

    print(f"[pipeline] Step05 count_mode={count_mode}")
    return run_step5_from_cli_defaults(
        base_dir=config.HUMAN_DIR,
        masks_dir=sample_paths.sample_dir,
        step04_dir=config.HUMAN_DIR / "step04_density_metrics",
        output_dir=config.HUMAN_DIR / "step05_glomeruli_count",
        slow_tracks=config.HUMAN_DIR / "step04_density_metrics" / "human_slow_tracks.csv",
        fast_tracks=config.HUMAN_DIR / "step04_density_metrics" / "human_rapid_tracks.csv",
        slow_density=config.HUMAN_DIR / "step04_density_metrics" / "human_density_slow.png",
        fast_density=config.HUMAN_DIR / "step04_density_metrics" / "human_density_rapid.png",
        cortex_mask=sample_paths.cortex_mask,
        exclude_mask=sample_paths.exclude_mask,
        step5_config=step5_config,
    )


def _save_healthy_run_summary(sample_id: str, summary_path: Path) -> None:
    """保存健康标定所需的轻量 summary；不复制中间 npy/png。"""

    if not summary_path.exists():
        return
    archive_dir = config.BASE_DIR / "archive" / "healthy_step05_runs"
    archive_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["sample_id"] = sample_id
    (archive_dir / f"{sample_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize_healthy_step05_runs(
    runs_dir: Path,
    output_path: Path | None = None,
    target_count: int = 450,
    rewrite_sample_json: bool = True,
) -> dict[str, Any]:
    """汇总健康样例 Step05 JSON，生成全局平均参数和两部分样本报告。"""

    runs_dir = Path(runs_dir)
    output_path = output_path or (runs_dir / "global_arguments.json")
    runs = load_healthy_run_records(runs_dir)
    if not runs:
        raise FileNotFoundError(f"未找到健康样例 JSON：{runs_dir}")

    per_sample_closest = {
        run["sample_id"]: closest_count_record(run["sampled_counts"], target_count)
        for run in runs
    }
    global_parameters = average_global_parameters(runs, per_sample_closest)
    global_effects = {
        run["sample_id"]: effect_at_global_parameters(run, global_parameters)
        for run in runs
    }
    global_counts = [int(effect["glomeruli_count"]) for effect in global_effects.values()]
    closest_counts = [int(item["glomeruli_count"]) for item in per_sample_closest.values()]

    report = {
        "mode": "healthy_step05_global_arguments",
        "runs_dir": str(runs_dir.resolve()),
        "target_healthy_glomeruli_count": int(target_count),
        "sample_ids": [run["sample_id"] for run in runs],
        "global_parameters_definition": "每个样本先取 count 最接近 target_count 的参数；global_parameters 为这些样本参数的算术平均。",
        "global_parameters": global_parameters,
        "global_parameters_effect": {
            "meaning": "使用 global_parameters 中的 dbscan_eps_mm，在每个样本 sampled_counts 中取最近 eps 对应的估计效果。",
            "by_sample": global_effects,
            "glomeruli_count_mean": float(np.mean(global_counts)) if global_counts else None,
            "glomeruli_count_min": int(min(global_counts)) if global_counts else None,
            "glomeruli_count_max": int(max(global_counts)) if global_counts else None,
        },
        "closest_to_target_parameters": {
            "meaning": "每个样本独立选择 count 最接近健康目标 450 的参数。",
            "by_sample": per_sample_closest,
            "glomeruli_count_mean": float(np.mean(closest_counts)) if closest_counts else None,
            "glomeruli_count_min": int(min(closest_counts)) if closest_counts else None,
            "glomeruli_count_max": int(max(closest_counts)) if closest_counts else None,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if rewrite_sample_json:
        for run in runs:
            sample_id = run["sample_id"]
            sample_report = {
                "sample_id": sample_id,
                "target_healthy_glomeruli_count": int(target_count),
                "part1_global_parameters_effect": {
                    "description": "全局平均参数下的估计效果；最重要字段是 effect.glomeruli_count。",
                    "global_parameters": global_parameters,
                    "effect": global_effects[sample_id],
                },
                "part2_closest_to_450_parameters": {
                    "description": "该样本独立扫描中，计数最接近健康肾小球数量 450 的参数。",
                    "parameters": per_sample_closest[sample_id]["parameters"],
                    "effect": {
                        "glomeruli_count": int(per_sample_closest[sample_id]["glomeruli_count"]),
                        "distance_to_target": int(per_sample_closest[sample_id]["distance_to_target"]),
                        "matched_eps_mm": float(per_sample_closest[sample_id]["matched_eps_mm"]),
                    },
                    "sampled_counts": run["sampled_counts"],
                },
                "source_metrics": run["source_metrics"],
            }
            run["path"].write_text(json.dumps(sample_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"global_arguments: {output_path}")
    print(f"sample_ids: {', '.join(report['sample_ids'])}")
    print(f"global_parameters: {json.dumps(global_parameters, ensure_ascii=False)}")
    for sample_id, effect in global_effects.items():
        print(f"{sample_id}: global_count={effect['glomeruli_count']} closest_count={per_sample_closest[sample_id]['glomeruli_count']}")
    return report


def load_healthy_run_records(runs_dir: Path) -> list[dict[str, Any]]:
    """读取 healthy_step05_runs 中的 raw 或两部分格式 JSON。"""

    records: list[dict[str, Any]] = []
    for path in sorted(Path(runs_dir).glob("*.json")):
        if path.name == "global_arguments.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(data.get("sample_id") or path.name.removesuffix("_summary.json"))
        sampled_counts = extract_sampled_counts(data)
        if not sampled_counts:
            continue
        records.append(
            {
                "path": path,
                "sample_id": sample_id,
                "sampled_counts": sampled_counts,
                "parameters": extract_run_parameters(data, sampled_counts),
                "source_metrics": extract_source_metrics(data),
            }
        )
    return records


def extract_sampled_counts(data: dict[str, Any]) -> list[dict[str, float | int]]:
    """从 raw summary 或两部分格式 summary 中提取 eps-count 扫描表。"""

    sampled = data.get("calibration_info", {}).get("sampled_counts")
    if sampled is None:
        sampled = data.get("part2_closest_to_450_parameters", {}).get("sampled_counts")
    result: list[dict[str, float | int]] = []
    for item in sampled or []:
        if "eps_mm" in item and "count" in item:
            result.append({"eps_mm": float(item["eps_mm"]), "count": int(item["count"])})
    return result


def extract_run_parameters(data: dict[str, Any], sampled_counts: list[dict[str, float | int]]) -> dict[str, float]:
    """提取可平均的 Step05 参数；dbscan_eps_mm 使用该样本最接近 450 的 eps。"""

    global_params = data.get("part1_global_parameters_effect", {}).get("global_parameters", {})
    if "part2_closest_to_450_parameters" in data:
        params = data["part2_closest_to_450_parameters"].get("parameters", {})
    else:
        params = {}
    closest = closest_count_record(sampled_counts, int(data.get("calibration_target_count", 450)))
    numeric_keys = [
        "exclude_inside_frac_max",
        "dbscan_min_samples",
        "iso_spacing_mm",
        "glomerulus_radius_mm",
        "calibration_target_count",
        "calibration_min_count",
    ]
    extracted: dict[str, float] = {"dbscan_eps_mm": float(closest["matched_eps_mm"])}
    for key in numeric_keys:
        value = params.get(key, data.get(key, global_params.get(key)))
        if isinstance(value, (int, float, np.integer, np.floating)):
            extracted[key] = float(value)
    calibration = data.get("calibration_info", {})
    for key, source_key in [
        ("dbscan_eps_min_mm", "eps_min_mm"),
        ("dbscan_eps_max_mm", "eps_max_mm"),
        ("dbscan_eps_steps", "eps_steps"),
    ]:
        value = params.get(key, calibration.get(source_key, global_params.get(key)))
        if isinstance(value, (int, float, np.integer, np.floating)):
            extracted[key] = float(value)
    return extracted


def extract_source_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """保留与计数解释直接相关的轨迹数量指标。"""

    keys = [
        "total_slow_tracks",
        "tracks_after_cortex_filter",
        "tracks_after_exclusion_filter",
        "tracks_after_fast_vessel_filter",
        "glomerular_tracks",
        "warnings",
    ]
    if "source_metrics" in data:
        return data["source_metrics"]
    return {key: data.get(key) for key in keys if key in data}


def closest_count_record(sampled_counts: list[dict[str, float | int]], target_count: int) -> dict[str, Any]:
    """从扫描表中选择 count 最接近目标值的 eps。"""

    if not sampled_counts:
        raise ValueError("sampled_counts 为空，无法选择最接近目标的参数。")
    best = min(
        sampled_counts,
        key=lambda item: (abs(int(item["count"]) - int(target_count)), float(item["eps_mm"])),
    )
    return {
        "parameters": {"dbscan_eps_mm": float(best["eps_mm"])},
        "matched_eps_mm": float(best["eps_mm"]),
        "glomeruli_count": int(best["count"]),
        "distance_to_target": abs(int(best["count"]) - int(target_count)),
    }


def average_global_parameters(
    runs: list[dict[str, Any]],
    per_sample_closest: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """对所有样本对应参数求算术平均，得到全局参数。"""

    keys = sorted(set().union(*(set(run["parameters"].keys()) for run in runs)))
    result: dict[str, float] = {}
    for key in keys:
        values = [float(run["parameters"][key]) for run in runs if key in run["parameters"]]
        if values:
            result[key] = float(np.mean(values))
    result["dbscan_eps_mm"] = float(np.mean([item["matched_eps_mm"] for item in per_sample_closest.values()]))
    return result


def effect_at_global_parameters(run: dict[str, Any], global_parameters: dict[str, float]) -> dict[str, Any]:
    """用全局 eps 在样本扫描表中找最近 eps 的计数，表示全局参数效果。"""

    target_eps = float(global_parameters["dbscan_eps_mm"])
    sampled = run["sampled_counts"]
    matched = min(sampled, key=lambda item: abs(float(item["eps_mm"]) - target_eps))
    return {
        "glomeruli_count": int(matched["count"]),
        "matched_eps_mm": float(matched["eps_mm"]),
        "target_global_eps_mm": target_eps,
        "eps_distance_mm": abs(float(matched["eps_mm"]) - target_eps),
        "source_metrics": run["source_metrics"],
    }


def parse_args() -> argparse.Namespace:
    """解析 Step0 命令行参数。"""

    parser = argparse.ArgumentParser(description="Step 0: TIC 稳定连续窗口 DICOM 预处理")
    parser.add_argument("--kind", choices=["human", "mouse"], default="human")
    parser.add_argument("--dicom", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None, help="Step00 TIC 使用的 mask 目录；默认仍查找 masks/。")
    parser.add_argument("--target-frames", type=int, default=DEFAULT_TARGET_FRAMES, help="默认保存 200 个连续源帧。")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES, help="保底连续帧数，默认 200。")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="TIC 移动平均窗口，默认 15。")
    parser.add_argument("--baseline-seconds", type=float, default=DEFAULT_BASELINE_SECONDS, help="用开头多少秒估计 baseline，默认 2。")
    parser.add_argument(
        "--min-enhancement",
        type=float,
        default=DEFAULT_MIN_ENHANCEMENT,
        help="相对 baseline 的最小绝对 TIC 增强，默认 0.01。",
    )
    parser.add_argument(
        "--target-tic",
        type=float,
        default=None,
        help="可选：希望窗口均值接近的绝对 TIC 强度；不传则优先稳定并轻微偏好高增强。",
    )
    parser.add_argument(
        "--run-full-pipeline",
        action="store_true",
        help="仅 human 使用：从 all_masks 样本运行 Step00-05，并按需进入 Step05 交互式 mask 绘制。",
    )
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="仅 human 使用：为指定 DICOM 跑到 Step04 并调用 Step05 交互式重画 mask，完成后归档 DICOM，不执行 Step05 计数。",
    )
    parser.add_argument("--sample", default=None, help="all_masks 中的样本编号，例如 21.0；也可以用 --dicom 传 DICOM 路径。")
    parser.add_argument("--all-masks-dir", type=Path, default=ulm_sample_manager.ALL_MASKS_DIR, help="样本 DICOM 和 mask 的长期目录。")
    parser.add_argument(
        "--count-mode",
        choices=["healthy_calibration", "diagnostic"],
        default="healthy_calibration",
        help="Step05 计数模式：健康样例标定或病肾诊断固定参数。",
    )
    parser.add_argument("--dbscan-eps-mm", type=float, default=None, help="diagnostic 模式或复现实验使用的固定 Step05 DBSCAN eps。")
    parser.add_argument("--no-rerun-after-mask", action="store_true", help="首次绘制 mask 后不重新从 Step00 跑最终流程。")
    parser.add_argument("--keep-existing-outputs", action="store_true", help="全流程运行前不清理 human_dcm 当前 Step 输出。")
    parser.add_argument(
        "--summarize-healthy-runs",
        action="store_true",
        help="汇总 archive/healthy_step05_runs/*.json，生成 global_arguments.json，并把样本 JSON 改为两部分格式。",
    )
    parser.add_argument(
        "--healthy-runs-dir",
        type=Path,
        default=config.BASE_DIR / "archive" / "healthy_step05_runs",
        help="健康样例 Step05 JSON 目录。",
    )
    parser.add_argument("--healthy-global-output", type=Path, default=None, help="global_arguments.json 输出路径；默认写入 healthy-runs-dir。")
    parser.add_argument("--healthy-target-count", type=int, default=450, help="健康肾小球目标数量，默认 450。")
    parser.add_argument("--no-rewrite-healthy-json", action="store_true", help="只生成 global_arguments.json，不重写每个样本 summary JSON。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.summarize_healthy_runs:
        summarize_healthy_step05_runs(
            runs_dir=args.healthy_runs_dir,
            output_path=args.healthy_global_output,
            target_count=args.healthy_target_count,
            rewrite_sample_json=not args.no_rewrite_healthy_json,
        )
    elif args.mask_only:
        if args.kind != "human":
            raise ValueError("--mask-only 当前只支持 human 链路。")
        run_human_mask_only(
            sample=args.sample or args.dicom,
            all_masks_dir=args.all_masks_dir,
            target_frames=args.target_frames,
            min_frames=args.min_frames,
            smooth_window=args.smooth_window,
            baseline_seconds=args.baseline_seconds,
            min_enhancement=args.min_enhancement,
            target_tic=args.target_tic,
            clean_outputs=not args.keep_existing_outputs,
        )
    elif args.run_full_pipeline:
        if args.kind != "human":
            raise ValueError("--run-full-pipeline 当前只支持 human 链路。")
        run_human_full_pipeline(
            sample=args.sample or args.dicom,
            all_masks_dir=args.all_masks_dir,
            target_frames=args.target_frames,
            min_frames=args.min_frames,
            smooth_window=args.smooth_window,
            baseline_seconds=args.baseline_seconds,
            min_enhancement=args.min_enhancement,
            target_tic=args.target_tic,
            count_mode=args.count_mode,
            dbscan_eps_mm=args.dbscan_eps_mm,
            rerun_after_mask=not args.no_rerun_after_mask,
            clean_outputs=not args.keep_existing_outputs,
        )
    else:
        run(
            kind=args.kind,
            dicom=args.dicom,
            mask_dir=args.mask_dir,
            target_frames=args.target_frames,
            min_frames=args.min_frames,
            smooth_window=args.smooth_window,
            baseline_seconds=args.baseline_seconds,
            min_enhancement=args.min_enhancement,
            target_tic=args.target_tic,
        )
