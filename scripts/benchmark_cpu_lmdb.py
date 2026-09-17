"""Reproducible multi-run CPU benchmark over checkpoints and an LMDB sample."""

from __future__ import annotations

import argparse
import csv
import json
import time
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
        for _ in range(repetitions):
            for _ in range(warmup_iterations):
                model(sample)
            timings = []
            for _ in range(timed_iterations):
                started_at = time.perf_counter_ns()
                model(sample)
                timings.append((time.perf_counter_ns() - started_at) / 1_000_000)
            repeat_summaries.append(summarize_timings(timings, batch_size=1))
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
    report = {
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
                "mode": arguments.mode,
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
            },
        ),
        "results": [],
    }
    for config_path in _config_paths(arguments.experiments, arguments.runs):
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


if __name__ == "__main__":
    raise SystemExit(main())
