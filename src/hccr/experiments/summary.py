"""Seed distributions and paired experiment comparisons."""

from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime
from typing import Any

from hccr.experiments.spec import ExperimentSpec, _jsonable_spec


def _summarize(
    spec: ExperimentSpec, records: list[dict[str, Any]], started_at: str
) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    variants = {}
    metric_names = (
        "top1",
        "top5",
        "macro_recall",
        "head_recall",
        "mid_recall",
        "tail_recall",
        "expected_calibration_error",
    )
    for variant in spec.variants:
        rows = [record for record in completed if record["variant"] == variant.name]
        completed_seeds = [row["seed"] for row in rows]
        failed_seeds = [
            seed
            for seed in spec.seeds
            if seed not in completed_seeds
            and any(
                record["variant"] == variant.name
                and record["seed"] == seed
                and record["status"] == "failed"
                for record in records
            )
        ]
        variants[variant.name] = {
            "planned_seeds": list(spec.seeds),
            "completed_seeds": completed_seeds,
            "failed_seeds": failed_seeds,
            "seed_metrics": {
                str(row["seed"]): row["selection_metrics"] for row in rows
            },
            "metrics": {
                metric: _distribution(
                    [
                        float(row["selection_metrics"][metric])
                        for row in rows
                        if metric in row["selection_metrics"]
                    ]
                )
                for metric in metric_names
            },
            "batch1_p95_ms": _aggregate_profile_latency(rows),
        }
    baseline_name = spec.variants[0].name
    comparisons = {
        name: _compare_variant_aggregates(variants[baseline_name], aggregate)
        for name, aggregate in variants.items()
        if name != baseline_name
    }
    planned_keys = {
        f"{variant.name}/seed-{seed}"
        for variant in spec.variants
        for seed in spec.seeds
    }
    completed_keys = {
        record["key"] for record in completed if record["key"] in planned_keys
    }
    failed_keys = {
        record["key"]
        for record in records
        if record["status"] == "failed" and record["key"] not in completed_keys
    }
    status = (
        "completed"
        if completed_keys == planned_keys
        else "partial"
        if completed_keys
        else "failed"
    )
    return {
        "schema_version": 3,
        "status": status,
        "jobs": {
            "planned": len(planned_keys),
            "completed": len(completed_keys),
            "failed": len(failed_keys),
            "missing": len(planned_keys - completed_keys - failed_keys),
        },
        "experiment_id": spec.experiment_id,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "spec": _jsonable_spec(spec),
        "baseline_variant": baseline_name,
        "records": records,
        "variants": variants,
        "comparisons": comparisons,
    }


def _aggregate_profile_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for device in ("cpu", "cuda"):
        values = []
        for row in rows:
            profile = row["profiles"].get(device, {})
            if profile.get("status") == "completed":
                benchmark = next(
                    item
                    for item in profile["trained_head"]["inference_benchmarks"]
                    if item["batch_size"] == 1
                )
                values.append(benchmark["latency_p95_ms"])
        result[device] = {"trained_head": _distribution(values)}
    return result


def _compare_variant_aggregates(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    paired = {
        metric: _paired_metric_delta(baseline, candidate, metric)
        for metric in ("top1", "macro_recall", "tail_recall")
    }
    result = {
        "top1_mean_delta": paired["top1"]["mean"],
        "macro_recall_mean_delta": paired["macro_recall"]["mean"],
        "tail_recall_mean_delta": paired["tail_recall"]["mean"],
        "paired_deltas": paired,
        "latency_p95_ratio": {},
    }
    for device in ("cpu", "cuda"):
        baseline_latency = _mean(baseline["batch1_p95_ms"][device]["trained_head"])
        candidate_latency = _mean(candidate["batch1_p95_ms"][device]["trained_head"])
        result["latency_p95_ratio"][device] = (
            candidate_latency / baseline_latency
            if baseline_latency and candidate_latency
            else None
        )
    return result


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "sample_std": None,
            "standard_error": None,
            "ci95_low": None,
            "ci95_high": None,
            "min": None,
            "max": None,
            "values": [],
        }
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else None
    standard_error = (
        sample_std / math.sqrt(len(values)) if sample_std is not None else None
    )
    margin = (
        _t_critical_95(len(values) - 1) * standard_error
        if standard_error is not None
        else None
    )
    return {
        "count": len(values),
        "mean": mean,
        "std": sample_std,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _paired_metric_delta(
    baseline: dict[str, Any], candidate: dict[str, Any], metric: str
) -> dict[str, Any]:
    baseline_by_seed = baseline["seed_metrics"]
    candidate_by_seed = candidate["seed_metrics"]
    baseline_seeds = set(baseline_by_seed)
    candidate_seeds = set(candidate_by_seed)
    shared = sorted(baseline_seeds.intersection(candidate_seeds), key=int)
    paired_seeds = [
        seed
        for seed in shared
        if metric in baseline_by_seed[seed] and metric in candidate_by_seed[seed]
    ]
    result = _distribution(
        [
            float(candidate_by_seed[seed][metric])
            - float(baseline_by_seed[seed][metric])
            for seed in paired_seeds
        ]
    )
    result.update(
        {
            "seeds": [int(seed) for seed in paired_seeds],
            "complete": (
                baseline_seeds == candidate_seeds
                and len(paired_seeds) == len(baseline_seeds)
            ),
            "baseline_only_seeds": sorted(map(int, baseline_seeds - candidate_seeds)),
            "candidate_only_seeds": sorted(map(int, candidate_seeds - baseline_seeds)),
        }
    )
    return result


def _t_critical_95(degrees_of_freedom: int) -> float:
    table = {
        1: 12.7062047364,
        2: 4.3026527297,
        3: 3.1824463053,
        4: 2.7764451052,
        5: 2.5705818356,
        6: 2.4469118488,
        7: 2.3646242510,
        8: 2.3060041352,
        9: 2.2621571629,
        10: 2.2281388520,
        11: 2.2009851601,
        12: 2.1788128297,
        13: 2.1603686565,
        14: 2.1447866879,
        15: 2.1314495456,
        16: 2.1199052992,
        17: 2.1098155778,
        18: 2.1009220402,
        19: 2.0930240544,
        20: 2.0859634473,
        21: 2.0796138447,
        22: 2.0738730679,
        23: 2.0686576104,
        24: 2.0638985616,
        25: 2.0595385528,
        26: 2.0555294386,
        27: 2.0518305165,
        28: 2.0484071418,
        29: 2.0452296421,
        30: 2.0422724563,
    }
    if degrees_of_freedom in table:
        return table[degrees_of_freedom]
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be positive")
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def _mean(distribution: dict[str, Any]) -> float | None:
    value = distribution.get("mean")
    return float(value) if value is not None else None
