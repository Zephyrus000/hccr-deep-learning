"""Reproducible multi-run CPU benchmark over checkpoints and an LMDB sample."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from collections.abc import Sequence
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
from hccr.models import build_model, optimize_model_for_inference
from hccr.preprocessing import EvalPreprocessor

IMAGE_SIZE = 96


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
    parser.add_argument("--summary", type=Path, default=Path("experiments/experiment_summary.csv"))
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
            summary.update({"repetition_index": repetition_index,
                            "latency_std_ms": statistics.stdev(timings),
                            "latency_min_ms": min(timings), "latency_max_ms": max(timings),
                            "timed_pass_count": len(timings)})
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
    dataset = HCCRDataset(
        arguments.manifest,
        arguments.split,
        EvalPreprocessor(IMAGE_SIZE),
        storage_backend="lmdb",
        lmdb_path=arguments.lmdb,
        metadata_mode="index",
    )
    if not dataset:
        raise ValueError("benchmark split contains no samples")
    sample_index = arguments.sample_index % len(dataset)
    sample = dataset[sample_index][0].unsqueeze(0)
    selected_manifest = _select_runs(arguments.summary, arguments.experiments)
    selected = [item["run_id"] for item in selected_manifest]
    execution_order = list(selected)
    import random
    random.Random(arguments.selection_seed).shuffle(execution_order)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
                "execution_mode": arguments.mode, "optimized": arguments.mode == "optimized",
                "optimization_transforms": (
                    []
                    if arguments.mode == "eager"
                    else ["optimize_model_for_inference"]
                ),
                "intra_op_threads": torch.get_num_threads(),
                "inter_op_threads": torch.get_num_interop_threads(),
                "input_shape": [1, 1, IMAGE_SIZE, IMAGE_SIZE],
                "preprocessing": (
                    f"EvalPreprocessor(image_size={IMAGE_SIZE}, margin=4)"
                ),
                "dataset_backend": "lmdb",
                "manifest_path": str(arguments.manifest),
                "lmdb_path": str(arguments.lmdb),
                "split": arguments.split,
                "sample_index": sample_index,
                "cpu_model": platform.processor(), "operating_system": platform.platform(),
                "python_version": platform.python_version(), "pytorch_version": torch.__version__,
                "aggregation_convention": "checkpoint=median of five repetition means/p50/p95; model=mean and sample SD over seeds",
            },
        ),
        "results": [],
        "selected_run_manifest": selected_manifest, "execution_order": execution_order,
    }
    for run_id in execution_order:
        config_path = arguments.experiments / run_id / "config.json"
        run_dir = config_path.parent
        checkpoint = run_dir / "checkpoint.pt"
        if not checkpoint.is_file():
            continue
        model, model_name = _model_from_run(run_dir)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model = _prepare_model_for_mode(model, arguments.mode)
        result = {
            "run_id": run_dir.name,
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


def _config_paths(experiments: Path, runs: Sequence[Path] | None) -> list[Path]:
    if runs:
        return [run / "config.json" for run in runs]
    direct_config = experiments / "config.json"
    if direct_config.is_file():
        return [direct_config]
    return sorted(experiments.glob("*/config.json"))

def _select_runs(summary: Path, experiments: Path) -> list[str]:
    required = {"resnet18", "efficientnet_b0", "efficient_hccr", "mobilenet_v3_small", "shufflenet_v2_x1_0"}
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    chosen = []
    manifest = []
    for model in sorted(required):
        for seed in (7, 17, 29):
            candidates = [r for r in rows if r["model"] == model and int(r["seed"]) == seed
                          and (model != "efficient_hccr" or r["reparameterize_depthwise"].lower() == "false")]
            if len(candidates) != 1:
                raise ValueError(f"expected exactly one candidate for {model} seed {seed}, found {len(candidates)}")
            r = candidates[0]; run = experiments / r["run_id"]; checkpoint = run / "checkpoint.pt"
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            chosen.append(r["run_id"]); manifest.append({"model": model, "seed": seed, "run_id": r["run_id"], "reparameterize_depthwise": r["reparameterize_depthwise"], "checkpoint": str(checkpoint), "checkpoint_exists": True})
    print(json.dumps({"selected_run_manifest": manifest}, indent=2))
    return manifest


def _model_from_run(run_dir: Path) -> tuple[torch.nn.Module, str]:
    metadata = json.loads(
        (run_dir / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    stored = metadata["model"]
    model_name = str(stored["name"])
    model_keys = (
        {
            "in_channels",
            "width",
            "backbone_output_channels",
            "embedding_dim",
            "stage_depths",
            "stem_stride",
            "reparameterize_depthwise",
            "dropout",
            "classification_head",
            "logit_scale",
            "angular_margin",
        }
        if model_name == "efficient_hccr"
        else {
            "in_channels",
            "classification_head",
            "embedding_dim",
            "logit_scale",
            "angular_margin",
        }
    )
    kwargs = {key: stored[key] for key in model_keys if key in stored}
    if "stage_depths" in kwargs:
        kwargs["stage_depths"] = tuple(kwargs["stage_depths"])
    model = build_model(model_name, num_classes=int(stored["num_classes"]), **kwargs)
    return model, model_name


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
        def metric(key): return [statistics.median(x[key] for x in v["repeat_summaries"]) for v in vals]
        means, p50s, p95s = metric("latency_mean_ms"), metric("latency_p50_ms"), metric("latency_p95_ms")
        summary_rows.append({"model": model, "seeds": "7,17,29", "mean_latency_mean_ms": statistics.mean(means), "mean_latency_std_ms": statistics.stdev(means), "p50_mean_ms": statistics.mean(p50s), "p50_std_ms": statistics.stdev(p50s), "p95_mean_ms": statistics.mean(p95s), "p95_std_ms": statistics.stdev(p95s)})
    with output.with_name(output.stem + "_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys()); writer.writeheader(); writer.writerows(summary_rows)


if __name__ == "__main__":
    raise SystemExit(main())
