"""Re-run resource profiling for completed checkpoints.

The original run directories are treated as immutable.  Each profile is written
below ``--output-root/runs/<run-id>`` together with checkpoint provenance, and
an aggregate JSON/CSV summary is written at the output root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from hccr.experiment_runner import _model_from_run
from hccr.training.diagnostics import profile_model
from hccr.utils.experiment import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run resource profiling for completed HCCR checkpoints."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--exclude-run", action="append", default=[])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--benchmark-iterations", type=int, default=200)
    parser.add_argument("--benchmark-repetitions", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested, but CUDA is unavailable")
    return requested


def discover_runs(runs_root: Path, selected_ids: list[str]) -> list[Path]:
    if selected_ids:
        candidates = [runs_root / run_id for run_id in selected_ids]
    else:
        candidates = sorted(path for path in runs_root.iterdir() if path.is_dir())
    completed = []
    for path in candidates:
        required = (path / "checkpoint.pt", path / "checkpoint_metadata.json")
        if all(item.is_file() for item in required):
            completed.append(path)
        elif selected_ids:
            raise FileNotFoundError(f"incomplete checkpoint run: {path}")
    return completed


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batch_one(profile: dict[str, Any], optimized: bool = False) -> dict[str, Any]:
    benchmarks = (
        profile["optimized_inference"]["benchmarks"]
        if optimized
        else profile["inference_benchmarks"]
    )
    return next(row for row in benchmarks if int(row["batch_size"]) == 1)


def profile_run(
    run_dir: Path,
    output_root: Path,
    device: str,
    warmup_iterations: int,
    benchmark_iterations: int,
    benchmark_repetitions: int,
    overwrite: bool,
) -> dict[str, Any]:
    run_id = run_dir.name
    destination = output_root / "runs" / run_id / device
    profile_path = destination / "resource_profile.json"
    provenance_path = destination / "provenance.json"
    if profile_path.is_file() and provenance_path.is_file() and not overwrite:
        print(f"resume {run_id}: existing profile retained", flush=True)
        profile = load_json(profile_path)
        provenance = load_json(provenance_path)
        return summary_row(provenance, profile)

    metadata = load_json(run_dir / "checkpoint_metadata.json")
    config_path = run_dir / "config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    image_size = int(metadata["preprocess"]["image_size"])
    checkpoint = run_dir / "checkpoint.pt"
    print(
        f"profiling {run_id} model={metadata['model']['name']} "
        f"seed={config.get('seed')} device={device}",
        flush=True,
    )
    model = _model_from_run(run_dir, device)
    profile = profile_model(
        model=model,
        image_size=image_size,
        device=device,
        output_dir=destination,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        benchmark_repetitions=benchmark_repetitions,
        full_class_num_classes=int(metadata["model"]["num_classes"]),
    )
    provenance = {
        "schema_version": 1,
        "profiled_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "seed": config.get("seed"),
        "source_run_directory": str(run_dir.resolve()),
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha256(checkpoint),
        "source_checkpoint_metadata": str(
            (run_dir / "checkpoint_metadata.json").resolve()
        ),
        "model": metadata["model"],
        "preprocess": metadata["preprocess"],
        "profiling_device": device,
    }
    write_json(provenance_path, provenance)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return summary_row(provenance, profile)


def summary_row(provenance: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    eager = batch_one(profile)
    optimized = batch_one(profile, optimized=True)
    model = provenance["model"]
    return {
        "run_id": provenance["run_id"],
        "seed": provenance.get("seed"),
        "model": model["name"],
        "classification_head": model.get("classification_head"),
        "reparameterize_depthwise": model.get("reparameterize_depthwise", False),
        "device": profile["device"],
        "parameter_count": profile["parameter_count"],
        "estimated_macs": profile["estimated_macs"],
        "estimated_flops": profile["estimated_flops"],
        "flop_coverage_complete": profile["flop_coverage"]["complete"],
        "eager_batch1_p50_ms": eager["latency_p50_ms"],
        "eager_batch1_p95_ms": eager["latency_p95_ms"],
        "optimized_parameter_count": profile["optimized_inference"]["parameter_count"],
        "optimized_estimated_macs": profile["optimized_inference"]["estimated_macs"],
        "optimized_estimated_flops": profile["optimized_inference"]["estimated_flops"],
        "optimized_flop_coverage_complete": profile["optimized_inference"][
            "flop_coverage"
        ]["complete"],
        "optimized_batch1_p50_ms": optimized["latency_p50_ms"],
        "optimized_batch1_p95_ms": optimized["latency_p95_ms"],
        "checkpoint_sha256": provenance["source_checkpoint_sha256"],
    }


def variant_key(row: dict[str, Any]) -> str:
    suffix = (
        "|reparam=" + str(row["reparameterize_depthwise"]).lower()
        if row["model"] == "efficient_hccr"
        else ""
    )
    return f"{row['model']}|head={row['classification_head']}{suffix}"


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        variants.setdefault(variant_key(row), []).append(row)
    result = {}
    for variant, variant_rows in sorted(variants.items()):
        latency_fields = (
            "eager_batch1_p50_ms",
            "eager_batch1_p95_ms",
            "optimized_batch1_p50_ms",
            "optimized_batch1_p95_ms",
        )
        result[variant] = {
            "run_ids": [row["run_id"] for row in variant_rows],
            "seeds": sorted(row["seed"] for row in variant_rows),
            "parameter_count": variant_rows[0]["parameter_count"],
            "estimated_macs": variant_rows[0]["estimated_macs"],
            "estimated_flops": variant_rows[0]["estimated_flops"],
            "latency_ms": {
                field: {
                    "mean": statistics.fmean(row[field] for row in variant_rows),
                    "std": (
                        statistics.stdev(row[field] for row in variant_rows)
                        if len(variant_rows) > 1
                        else 0.0
                    ),
                    "n": len(variant_rows),
                }
                for field in latency_fields
            },
        }
    return result


def write_summary(
    output_root: Path,
    rows: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    arguments: argparse.Namespace,
    device: str,
) -> None:
    rows = sorted(rows, key=lambda row: (row["model"], row["seed"], row["run_id"]))
    payload = {
        "schema_version": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "device": device,
            "warmup_iterations": arguments.warmup_iterations,
            "benchmark_iterations": arguments.benchmark_iterations,
            "benchmark_repetitions": arguments.benchmark_repetitions,
            "source_runs_root": str(arguments.runs_root.resolve()),
        },
        "included_run_count": len(rows),
        "excluded_runs": excluded,
        "runs": rows,
        "variants": aggregate(rows),
    }
    write_json(output_root / "summary.json", payload)
    csv_path = output_root / "summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    arguments = parse_args()
    if (
        min(
            arguments.warmup_iterations,
            arguments.benchmark_iterations,
            arguments.benchmark_repetitions,
        )
        < 1
    ):
        raise ValueError("profiling iteration counts must be positive")
    device = resolve_device(arguments.device)
    discovered = discover_runs(arguments.runs_root, arguments.run_id)
    excluded_ids = set(arguments.exclude_run)
    discovered_ids = {path.name for path in discovered}
    unknown_exclusions = excluded_ids - discovered_ids
    if unknown_exclusions:
        raise ValueError(f"excluded run IDs were not discovered: {unknown_exclusions}")
    selected = [path for path in discovered if path.name not in excluded_ids]
    if not selected:
        raise ValueError("no checkpoint runs selected for profiling")
    excluded = []
    for path in discovered:
        if path.name in excluded_ids:
            metadata = load_json(path / "checkpoint_metadata.json")
            excluded.append(
                {
                    "run_id": path.name,
                    "reason": "explicitly_excluded_multi_branch",
                    "model": metadata["model"],
                }
            )
    write_json(
        arguments.output_root / "selection.json",
        {
            "selected_run_ids": [path.name for path in selected],
            "excluded_runs": excluded,
        },
    )
    rows = [
        profile_run(
            path,
            arguments.output_root,
            device,
            arguments.warmup_iterations,
            arguments.benchmark_iterations,
            arguments.benchmark_repetitions,
            arguments.overwrite,
        )
        for path in selected
    ]
    write_summary(arguments.output_root, rows, excluded, arguments, device)
    summary_path = arguments.output_root / "summary.json"
    print(f"completed {len(rows)} profiles; summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
