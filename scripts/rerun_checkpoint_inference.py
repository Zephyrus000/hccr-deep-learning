"""Re-evaluate completed checkpoints and summarize seed distributions.

This utility intentionally reconstructs each model from its checkpoint metadata
and evaluates the frozen ``test`` split with the stored preprocessing and
precision.  It does not modify the original training run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hccr.data.dataset import HCCRDataset, select_class_subset
from hccr.data.manifest import read_manifest
from hccr.evaluation.evaluator import evaluate
from hccr.models import load_model_from_run
from hccr.preprocessing import EvalPreprocessor
from hccr.training.precision import resolve_precision

METRICS = (
    "top1",
    "top5",
    "macro_recall",
    "head_recall",
    "mid_recall",
    "tail_recall",
    "mean_confidence",
    "expected_calibration_error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run held-out inference for completed HCCR checkpoints."
    )
    parser.add_argument("--runs-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def variant_label(config: dict) -> str:
    model = str(config["model"])
    if model != "efficient_hccr":
        return model
    depths = "-".join(str(value) for value in config["stage_depths"])
    return (
        f"efficient_hccr|reparam={config['reparameterize_depthwise']}"
        f"|width={config['width']}|stages={depths}"
        f"|head={config['classification_head']}"
    )


def training_support(manifest: Path) -> Counter[int]:
    return Counter(
        int(row["class_id"])
        for row in read_manifest(manifest)
        if row["split"] == "train"
    )


def evaluation_loader(
    manifest: Path, config: dict, num_workers: int, batch_size: int
) -> DataLoader:
    class_id_map = None
    if config.get("max_classes") is not None:
        class_id_map = select_class_subset(
            manifest, int(config["max_classes"]), int(config["class_subset_seed"])
        )
    dataset = HCCRDataset(
        manifest,
        "test",
        EvalPreprocessor(int(config["image_size"])),
        class_id_map,
        storage_backend=str(config.get("dataset_backend", "auto")),
        lmdb_path=(
            Path(config["lmdb_path"]) if config.get("lmdb_path") is not None else None
        ),
        metadata_mode="index",
    )
    options: dict[str, object] = {"pin_memory": True}
    if num_workers > 0:
        options.update(
            {
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 1,
            }
        )
    return DataLoader(dataset, batch_size=batch_size, **options)


def evaluate_run(arguments: argparse.Namespace, run_id: str) -> dict:
    run_dir = arguments.runs_root / run_id
    checkpoint = run_dir / "checkpoint.pt"
    config_path = run_dir / "config.json"
    metadata_path = run_dir / "checkpoint_metadata.json"
    if (
        not checkpoint.is_file()
        or not config_path.is_file()
        or not metadata_path.is_file()
    ):
        raise FileNotFoundError(f"missing completed-run artifact for {run_id}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = Path(config["manifest_path"])
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest missing for {run_id}: {manifest}")
    output_dir = arguments.output_root / run_id
    loader = evaluation_loader(
        manifest, config, arguments.num_workers, arguments.batch_size
    )
    model = load_model_from_run(run_dir, arguments.device)
    requested_precision = str(metadata["training"]["precision"]["requested"])
    precision = resolve_precision(requested_precision, arguments.device)
    metrics = evaluate(
        model,
        loader,
        arguments.device,
        output_dir=output_dir,
        class_support=training_support(manifest),
        evaluation_name="test_rerun",
        precision=precision,
    )
    result = {
        "run_id": run_id,
        "seed": int(config["seed"]),
        "variant": variant_label(config),
        "model": metadata["model"],
        "source_checkpoint": str(checkpoint),
        "test_rerun": metrics,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    del model, loader
    if arguments.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(
        f"completed {run_id} seed={result['seed']} "
        f"top1={metrics['top1']:.6f} top5={metrics['top5']:.6f}",
        flush=True,
    )
    return result


def summarize(output_root: Path) -> dict:
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output_root.glob("*/result.json"))
    ]
    if not results:
        raise ValueError(f"no result.json files found below {output_root}")
    variants: dict[str, list[dict]] = {}
    for result in results:
        variants.setdefault(result["variant"], []).append(result)
    summary = {"runs": results, "variants": {}}
    for variant, rows in sorted(variants.items()):
        summary["variants"][variant] = {
            "seeds": sorted(row["seed"] for row in rows),
            "metrics": {
                metric: {
                    "mean": statistics.fmean(
                        row["test_rerun"][metric] for row in rows
                    ),
                    "std": (
                        statistics.stdev(row["test_rerun"][metric] for row in rows)
                        if len(rows) > 1
                        else 0.0
                    ),
                    "n": len(rows),
                }
                for metric in METRICS
            },
        }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("variant", "metric", "mean", "std", "n")
        )
        writer.writeheader()
        for variant, aggregate in summary["variants"].items():
            for metric, values in aggregate["metrics"].items():
                writer.writerow({"variant": variant, "metric": metric, **values})
    print(json.dumps(summary["variants"], indent=2), flush=True)
    return summary


def main() -> int:
    arguments = parse_args()
    if arguments.summarize_only:
        summarize(arguments.output_root)
        return 0
    if not arguments.run_id:
        raise ValueError("provide at least one --run-id, or pass --summarize-only")
    for run_id in arguments.run_id:
        evaluate_run(arguments, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
