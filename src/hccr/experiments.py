"""Utilities for comparing architecture runs against an explicit quality gate."""

from __future__ import annotations

import csv
from pathlib import Path


def compare_runs(
    summary_path: Path,
    baseline_run_id: str,
    candidate_run_id: str,
    min_top1_gain: float = 0.0,
    max_p95_latency_ratio: float = 1.0,
) -> dict[str, float | bool | str]:
    """Return a transparent accept/reject verdict for one architecture candidate."""
    with summary_path.open(newline="", encoding="utf-8") as file:
        rows = {row["run_id"]: row for row in csv.DictReader(file)}
    baseline = rows[baseline_run_id]
    candidate = rows[candidate_run_id]
    top1_gain = float(candidate["top1"]) - float(baseline["top1"])
    latency_ratio = float(candidate["latency_p95_ms"]) / float(
        baseline["latency_p95_ms"]
    )
    return {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "top1_gain": top1_gain,
        "p95_latency_ratio": latency_ratio,
        "accepted": (
            top1_gain >= min_top1_gain and latency_ratio <= max_p95_latency_ratio
        ),
    }
