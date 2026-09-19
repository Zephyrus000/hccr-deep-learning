"""Reproducible multi-run CPU benchmark over checkpoints and an LMDB sample."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from hccr.benchmarking import (
    BENCHMARK_SCHEMA_VERSION,
    aggregate_timing_summaries,
    build_benchmark_protocol,
    summarize_timings,
)
from hccr.data.dataset import HCCRDataset
from hccr.models import (
    load_model_from_run,
    optimize_model_for_inference,
    read_checkpoint_metadata,
)
from hccr.preprocessing import EvalPreprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple experiment checkpoints on one CPU host."
    )
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="*",
        help="Specific run directories; defaults to all runs under --experiments.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/casia_hwdb/manifest.csv"),
    )
    parser.add_argument(
        "--lmdb",
        type=Path,
        default=Path("data/processed/casia_hwdb/images.lmdb"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--mode",
        choices=("eager", "optimized"),
        default="eager",
        help="eager for the main comparison; optimized for deployment studies",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("experiments/experiment_summary.csv"),
    )
    parser.add_argument("--selection-seed", type=int, default=20260918)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/cpu_lmdb_inference_benchmark"),
    )
    return parser


def benchmark_model(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup_iterations: int,
    timed_iterations: int,
    repetitions: int,
) -> dict[str, Any]:
    """Benchmark a prepared model sample using the shared result schema."""
    repeat_summaries = []
    model.eval()
    with torch.inference_mode():
        for repetition_index in range(repetitions):
            for _ in range(warmup_iterations):
                model(sample)
            timings = []
            for _ in range(timed_iterations):
                started_at = time.perf_counter_ns()
                model(sample)
                timings.append((time.perf_counter_ns() - started_at) / 1_000_000)
            summary = summarize_timings(timings, batch_size=1)
            summary.update(
                {
                    "repetition_index": repetition_index,
                    "latency_std_ms": (
                        statistics.stdev(timings) if len(timings) > 1 else 0.0
                    ),
                    "latency_min_ms": min(timings),
                    "latency_max_ms": max(timings),
                    "timed_pass_count": len(timings),
                }
            )
            repeat_summaries.append(summary)
    return aggregate_timing_summaries(repeat_summaries)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if (
        min(
            arguments.threads,
            arguments.warmup,
            arguments.iterations,
            arguments.repetitions,
        )
        < 1
    ):
        raise ValueError("threads, warmup, iterations and repetitions must be positive")
    torch.set_num_threads(arguments.threads)
    torch.set_num_interop_threads(1)
    selected_manifest = _resolve_run_manifest(arguments)
    preprocess = _shared_preprocess(selected_manifest)
    image_size = int(preprocess["image_size"])
    dataset = HCCRDataset(
        arguments.manifest,
        arguments.split,
        EvalPreprocessor(**preprocess),
        storage_backend="lmdb",
        lmdb_path=arguments.lmdb,
        metadata_mode="index",
    )
    if not dataset:
        raise ValueError("benchmark split contains no samples")
    sample_index = arguments.sample_index % len(dataset)
    sample = dataset[sample_index][0].unsqueeze(0)
    execution_order = list(selected_manifest)
    random.Random(arguments.selection_seed).shuffle(execution_order)
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_protocol": build_benchmark_protocol(
            device="cpu",
            batch_size=1,
            warmup_iterations=arguments.warmup,
            timed_iterations=arguments.iterations,
            repetitions=arguments.repetitions,
            scope="model_forward",
            metadata={
                "dtype": "float32",
                "execution_mode": arguments.mode,
                "optimized": arguments.mode == "optimized",
                "optimization_transforms": (
                    []
                    if arguments.mode == "eager"
                    else ["optimize_model_for_inference"]
                ),
                "intra_op_threads": torch.get_num_threads(),
                "inter_op_threads": torch.get_num_interop_threads(),
                "input_shape": [1, 1, image_size, image_size],
                "preprocessing": preprocess,
                "dataset_backend": "lmdb",
                "manifest_path": str(arguments.manifest),
                "lmdb_path": str(arguments.lmdb),
                "split": arguments.split,
                "sample_index": sample_index,
                "cpu_model": platform.processor(),
                "operating_system": platform.platform(),
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "aggregation_convention": (
                    "checkpoint=median of repetition summaries; "
                    "model=mean and sample SD over selected seeds"
                ),
            },
        ),
        "results": [],
        "selected_run_manifest": selected_manifest,
        "execution_order": [item["run_id"] for item in execution_order],
    }
    for selection in execution_order:
        run_dir = Path(selection["run_path"])
        checkpoint = run_dir / "checkpoint.pt"
        model = load_model_from_run(run_dir)
        model_name = str(selection["model"])
        model = _prepare_model_for_mode(model, arguments.mode)
        result = {
            "run_id": run_dir.name,
            "seed": selection.get("seed"),
            "model_name": model_name,
            "checkpoint": str(checkpoint),
            "precision": "float32",
            "mode": arguments.mode,
            "optimized": arguments.mode == "optimized",
            "shared_inference_behavior": [
                "model_eval",
                "torch_inference_mode",
                *(
                    ["cache_normalized_classifier_weight"]
                    if getattr(model, "classification_head", None)
                    in {"cosface", "arcface"}
                    else []
                ),
            ],
            "optimization_transforms": (
                [] if arguments.mode == "eager" else ["optimize_model_for_inference"]
            ),
            "batch_size": 1,
            "scope": "model_forward",
            **benchmark_model(
                model,
                sample,
                arguments.warmup,
                arguments.iterations,
                arguments.repetitions,
            ),
        }
        report["results"].append(result)
        print(json.dumps(result))
    _write_report(arguments.output, report)
    return 0


def _prepare_model_for_mode(model: torch.nn.Module, mode: str) -> torch.nn.Module:
    """Apply only the transformations authorized by the benchmark mode."""
    if mode == "eager":
        return model.eval()
    if mode == "optimized":
        return optimize_model_for_inference(model).eval()
    raise ValueError(f"unsupported benchmark mode: {mode}")


def _resolve_run_manifest(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    if arguments.runs:
        return [_explicit_run_entry(path) for path in arguments.runs]
    return _select_runs(arguments.summary, arguments.experiments)


def _explicit_run_entry(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metadata = read_checkpoint_metadata(run_dir)
    checkpoint = run_dir / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "model": metadata["model"]["name"],
        "seed": config.get("seed"),
        "run_id": run_dir.name,
        "run_path": str(run_dir),
        "reparameterize_depthwise": metadata["model"].get(
            "reparameterize_depthwise"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": True,
        "preprocess": metadata["preprocess"],
        "selection": "explicit",
    }


def _select_runs(summary: Path, experiments: Path) -> list[dict[str, Any]]:
    required = {
        "resnet18",
        "efficientnet_b0",
        "efficient_hccr",
        "mobilenet_v3_small",
        "shufflenet_v2_x1_0",
    }
    with summary.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    manifest = []
    for model in sorted(required):
        for seed in (7, 17, 29):
            candidates = [
                row
                for row in rows
                if row["model"] == model
                and int(row["seed"]) == seed
                and (
                    model != "efficient_hccr"
                    or row["reparameterize_depthwise"].lower() == "false"
                )
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"expected exactly one candidate for {model} seed {seed}, "
                    f"found {len(candidates)}"
                )
            row = candidates[0]
            run = (experiments / row["run_id"]).resolve()
            checkpoint = run / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            metadata = read_checkpoint_metadata(run)
            manifest.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_id": row["run_id"],
                    "run_path": str(run),
                    "reparameterize_depthwise": row[
                        "reparameterize_depthwise"
                    ],
                    "checkpoint": str(checkpoint),
                    "checkpoint_exists": True,
                    "preprocess": metadata["preprocess"],
                    "selection": "summary_matrix",
                }
            )
    print(json.dumps({"selected_run_manifest": manifest}, indent=2))
    return manifest


def _shared_preprocess(manifest: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not manifest:
        raise ValueError("benchmark run manifest is empty")
    expected = dict(manifest[0]["preprocess"])
    mismatched = [
        item["run_id"]
        for item in manifest[1:]
        if dict(item["preprocess"]) != expected
    ]
    if mismatched:
        raise ValueError(
            "all CPU benchmark runs must share preprocessing; mismatched runs: "
            + ", ".join(mismatched)
        )
    return expected


def _write_report(output: Path, report: dict[str, Any]) -> None:
    json_path = output.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["results"]:
        return
    csv_rows = [
        {key: value for key, value in result.items() if key != "repeat_summaries"}
        for result in report["results"]
    ]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    summary_rows = []
    for model in sorted({r["model_name"] for r in report["results"]}):
        vals = [r for r in report["results"] if r["model_name"] == model]

        def metric(records: list[dict[str, Any]], key: str) -> list[float]:
            return [
                statistics.median(
                    repetition[key] for repetition in value["repeat_summaries"]
                )
                for value in records
            ]

        means = metric(vals, "latency_mean_ms")
        p50s = metric(vals, "latency_p50_ms")
        p95s = metric(vals, "latency_p95_ms")
        summary_rows.append(
            {
                "model": model,
                "seeds": ",".join(
                    str(seed)
                    for seed in sorted(
                        value["seed"]
                        for value in vals
                        if value.get("seed") is not None
                    )
                ),
                "mean_latency_mean_ms": statistics.mean(means),
                "mean_latency_std_ms": _sample_std(means),
                "p50_mean_ms": statistics.mean(p50s),
                "p50_std_ms": _sample_std(p50s),
                "p95_mean_ms": statistics.mean(p95s),
                "p95_std_ms": _sample_std(p95s),
            }
        )
    summary_path = output.with_name(output.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
