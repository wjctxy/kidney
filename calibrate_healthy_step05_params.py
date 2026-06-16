"""从健康样例 Step05 摘要中选择公共诊断参数。

输入来自：
    archive/healthy_step05_runs/<sample_id>_summary.json

输出：
    archive/healthy_common_step05_params.json

这个脚本只统计健康样例，不读取每个 DICOM 的中间 npy/png。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ulm_config as config


DEFAULT_RUNS_DIR = config.BASE_DIR / "archive" / "healthy_step05_runs"
DEFAULT_OUTPUT = config.BASE_DIR / "archive" / "healthy_common_step05_params.json"


def calibrate(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    output: Path = DEFAULT_OUTPUT,
    min_count: int = 400,
    max_count: int = 600,
    target_count: int = 450,
) -> dict[str, Any]:
    """选择使健康样例稳定超过 min_count 且接近 target_count 的公共 eps。"""

    summaries = load_run_summaries(runs_dir)
    if not summaries:
        raise FileNotFoundError(f"未找到健康样例 Step05 summary：{runs_dir}")

    table = build_count_table(summaries)
    if not table:
        raise ValueError("summary 中没有 calibration_info.sampled_counts，无法统计公共 eps。")

    above_min_best = choose_above_min_eps(table, min_count=min_count, target_count=target_count)
    balanced_best = choose_balanced_eps(table, min_count=min_count, max_count=max_count, target_count=target_count)
    is_balanced = all(min_count <= value <= max_count for value in above_min_best["counts_by_sample"].values())
    result = {
        "mode": "healthy_common_step05_params",
        "runs_dir": str(runs_dir),
        "n_samples": len(summaries),
        "sample_ids": sorted(summaries),
        "selection_rule": (
            "above_min_best 优先保证所有健康样例 count >= min_count；"
            "balanced_best 优先让所有健康样例同时靠近 target_count，避免一个样例被拆成异常高计数。"
        ),
        "min_count": int(min_count),
        "max_count": int(max_count),
        "target_count": int(target_count),
        "selected_dbscan_eps_mm": float(above_min_best["eps_mm"]),
        "counts_by_sample": above_min_best["counts_by_sample"],
        "min_selected_count": int(above_min_best["min_count"]),
        "mean_selected_count": float(above_min_best["mean_count"]),
        "all_samples_within_healthy_range": bool(is_balanced),
        "recommended_for_diagnostic": bool(is_balanced),
        "above_min_best": above_min_best,
        "balanced_best": balanced_best,
        "recommended_step5_config": {
            "calibration_enabled": False,
            "dbscan_eps_mm": float(above_min_best["eps_mm"]),
            "exclude_inside_frac_max": _common_value(summaries, "exclude_inside_frac_max"),
        },
        "eps_table": table,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"selected_dbscan_eps_mm: {result['selected_dbscan_eps_mm']}")
    print(f"counts_by_sample: {json.dumps(result['counts_by_sample'], ensure_ascii=False)}")
    print(f"recommended_for_diagnostic: {result['recommended_for_diagnostic']}")
    print(f"balanced_best: {json.dumps(result['balanced_best'], ensure_ascii=False)}")
    print(f"output: {output}")
    return result


def load_run_summaries(runs_dir: Path) -> dict[str, dict[str, Any]]:
    """读取每个健康样例的 Step05 summary。"""

    summaries: dict[str, dict[str, Any]] = {}
    if not runs_dir.exists():
        return summaries
    for path in sorted(runs_dir.glob("*_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(summary.get("sample_id") or path.name.removesuffix("_summary.json"))
        summaries[sample_id] = summary
    return summaries


def build_count_table(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """把多个 summary 的 eps->count 扫描结果合成表格。"""

    counts_by_sample: dict[str, dict[float, int]] = {}
    for sample_id, summary in summaries.items():
        sampled = _sampled_counts(summary)
        counts_by_sample[sample_id] = {round(float(item["eps_mm"]), 12): int(item["count"]) for item in sampled}

    common_eps = sorted(set.intersection(*(set(values) for values in counts_by_sample.values()))) if counts_by_sample else []
    table: list[dict[str, Any]] = []
    for eps in common_eps:
        counts = {sample_id: values[eps] for sample_id, values in counts_by_sample.items()}
        table.append(
            {
                "eps_mm": float(eps),
                "counts_by_sample": counts,
                "min_count": int(min(counts.values())),
                "mean_count": float(sum(counts.values()) / len(counts)),
                "below_400_samples": int(sum(value < 400 for value in counts.values())),
            }
        )
    return table


def choose_above_min_eps(table: list[dict[str, Any]], min_count: int, target_count: int) -> dict[str, Any]:
    """选择所有健康样例尽量不低于 min_count 的 eps。"""

    def score(row: dict[str, Any]) -> tuple[int, int, float, float]:
        below_samples = int(sum(value < min_count for value in row["counts_by_sample"].values()))
        below_gap = int(sum(max(0, min_count - value) for value in row["counts_by_sample"].values()))
        target_gap = abs(float(row["mean_count"]) - float(target_count))
        return below_samples, below_gap, target_gap, float(row["eps_mm"])

    return min(table, key=score)


def choose_balanced_eps(table: list[dict[str, Any]], min_count: int, max_count: int, target_count: int) -> dict[str, Any]:
    """选择健康样例之间最均衡、最接近 target_count 的 eps。"""

    def score(row: dict[str, Any]) -> tuple[int, int, int, float, float]:
        counts = list(row["counts_by_sample"].values())
        below_gap = int(sum(max(0, min_count - value) for value in counts))
        above_gap = int(sum(max(0, value - max_count) for value in counts))
        max_target_gap = int(max(abs(value - target_count) for value in counts))
        mean_gap = abs(float(row["mean_count"]) - float(target_count))
        return below_gap + above_gap, max_target_gap, below_gap, mean_gap, float(row["eps_mm"])

    return min(table, key=score)


def _common_value(summaries: dict[str, dict[str, Any]], key: str) -> Any:
    """如果所有 summary 的某个字段一致，则返回该值，否则返回逐样本字典。"""

    values = {sample_id: _summary_value(summary, key) for sample_id, summary in summaries.items()}
    unique = set(json.dumps(value, sort_keys=True) for value in values.values())
    if len(unique) == 1:
        return next(iter(values.values()))
    return values


def _sampled_counts(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """兼容 raw summary 和两部分 summary 的 sampled_counts 位置。"""

    sampled = summary.get("calibration_info", {}).get("sampled_counts")
    if sampled is None:
        sampled = summary.get("part2_closest_to_450_parameters", {}).get("sampled_counts", [])
    return list(sampled or [])


def _summary_value(summary: dict[str, Any], key: str) -> Any:
    """兼容 raw summary 和两部分 summary 的参数位置。"""

    if key in summary:
        return summary.get(key)
    global_params = summary.get("part1_global_parameters_effect", {}).get("global_parameters", {})
    if key in global_params:
        return global_params.get(key)
    closest_params = summary.get("part2_closest_to_450_parameters", {}).get("parameters", {})
    return closest_params.get(key)


def parse_args() -> argparse.Namespace:
    """解析公共参数统计脚本参数。"""

    parser = argparse.ArgumentParser(description="Calibrate common Step05 params from healthy samples")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-count", type=int, default=400)
    parser.add_argument("--max-count", type=int, default=600)
    parser.add_argument("--target-count", type=int, default=450)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    calibrate(args.runs_dir, args.output, args.min_count, args.max_count, args.target_count)
