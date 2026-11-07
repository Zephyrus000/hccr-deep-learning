"""Pinned full-endpoint evaluation of a trained HOG-CNN fusion artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as VF

from hccr.artifacts.checkpoints import file_digest
from hccr.data.lmdb_store import LMDBImageStore
from hccr.models.hog_fusion import load_hog_fusion_run
from hccr.preprocessing.pipeline import EvalPreprocessor
from hccr.training.precision import resolve_precision

OFFICIAL_COUNTS = {"test": 533675, "icdar": 224419}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class HoldoutImages(Dataset):
    """Retain only endpoint identities while opening images lazily via LMDB."""

    def __init__(self, manifest, split, labels, preprocess, manifest_sha, count):
        if split not in OFFICIAL_COUNTS:
            raise ValueError("only test and icdar endpoints are supported")
        self.ids, self.writers, targets = [], [], []
        writer_pool = {}
        with Path(manifest).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != split:
                    continue
                target = int(row["class_id"])
                if labels.get(str(target)) != row["unicode_label"]:
                    raise ValueError("endpoint class order differs from model labels")
                writer = row["writer_id"].strip()
                if not writer:
                    raise ValueError("endpoint writer ID is missing")
                self.ids.append(row["sample_id"])
                self.writers.append(writer_pool.setdefault(writer, writer))
                targets.append(target)
        if len(self.ids) != count:
            raise ValueError(f"{split} requires {count} rows, found {len(self.ids)}")
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("endpoint sample IDs must be nonempty and unique")
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = EvalPreprocessor(**preprocess)
        self.store = LMDBImageStore(
            Path(manifest).with_name("images.lmdb"), manifest_sha
        )
        _ = self.store.metadata
        self.store.close()
        digest = hashlib.sha256()
        for sample_id in self.ids:
            digest.update(sample_id.encode("utf-8") + b"\n")
        self.ordered_ids_sha256 = digest.hexdigest()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image = self.transform(self.store.read_image(self.ids[index]))
        return VF.to_dtype(VF.to_image(image), torch.float32, scale=True), int(
            self.targets[index]
        )


def prediction_indices(logits):
    """Keep argmax tie-breaking separate from top-k membership/order."""
    return (
        logits.argmax(dim=1).cpu().numpy(),
        logits.topk(min(5, logits.shape[1]), dim=1).indices.cpu().numpy(),
    )


def summarize_predictions(targets, predictions, losses, num_classes, *, top1=None):
    """Compute exact whole-endpoint metrics from saved top-five predictions."""
    counts = np.bincount(targets, minlength=num_classes)
    present = counts > 0
    metrics = {}
    top1 = (
        {name: top[:, 0] for name, top in predictions.items()} if top1 is None else top1
    )
    for name, top in predictions.items():
        correct = top1[name] == targets
        class_correct = np.bincount(targets[correct], minlength=num_classes)
        metrics[name] = {
            "top1": float(correct.mean()),
            "top5": float(np.any(top == targets[:, None], axis=1).mean()),
            "macro_recall": float((class_correct[present] / counts[present]).mean()),
            "cross_entropy": float(np.asarray(losses[name], dtype=np.float64).mean()),
        }
    baseline = top1["baseline"] == targets
    for name in ("fusion", "hog_off"):
        correct = top1[name] == targets
        repairs = int((~baseline & correct).sum())
        damage = int((baseline & ~correct).sum())
        metrics[name].update(
            repairs=repairs,
            damage=damage,
            delta_pp=100 * (repairs - damage) / len(targets),
            prediction_changes=int((top1[name] != top1["baseline"]).sum()),
        )
    return {
        "samples": len(targets),
        "classes_present": int(present.sum()),
        "metrics": metrics,
        "natural_minus_hog_off_pp": 100
        * (metrics["fusion"]["top1"] - metrics["hog_off"]["top1"]),
    }


def pin_evaluation(run_dir, manifest, output, settings):
    """Freeze checkpoint bytes and the evaluation plan before endpoint reads."""
    run_dir, manifest, output = map(Path, (run_dir, manifest, output))
    metadata = json.loads((run_dir / "checkpoint_metadata.json").read_text("utf-8"))
    if (
        metadata["model"]["name"] != "hog_cnn_fusion"
        or int(metadata["selection"]["epoch"]) < 1
        or metadata["selection"].get("split") != "validation"
    ):
        raise ValueError("a trained fusion selected without test labels is required")
    labels = json.loads((run_dir / "labels.json").read_text("utf-8"))
    if json.loads(manifest.with_name("labels.json").read_text("utf-8")) != labels:
        raise ValueError("local labels differ from checkpoint labels")
    expected_classes = int(metadata["model"]["cnn"]["num_classes"])
    if set(labels["labels"]) != {str(i) for i in range(expected_classes)}:
        raise ValueError("checkpoint labels must contain all contiguous class IDs")
    plan = {
        "schema_version": 1,
        "source_run": str(run_dir.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_digest(manifest),
        "frozen_files": {
            name: file_digest(run_dir / name)
            for name in ("checkpoint.pt", "checkpoint_metadata.json", "labels.json")
        },
        "model_code_sha256": file_digest(
            Path(__file__).parents[1] / "models/hog_fusion.py"
        ),
        "evaluator_code_sha256": file_digest(Path(__file__)),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "selection": metadata["selection"],
        "preprocess": metadata["preprocess"],
        "arithmetic": {
            "backbone": metadata["model"]["backbone_precision"],
            "hog_fusion_head": "float32",
            "tf32": False,
        },
        "settings": settings,
        "selection_after_holdout": "forbidden",
        "hog_off_intervention": "zero_64D_projected_hog_keep_all_fitted_weights",
        "hog_off_limitation": "can_be_out_of_training_distribution",
    }
    output.mkdir(parents=True, exist_ok=True)
    plan_file = output / "evaluation_plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text("utf-8")) != plan:
        raise ValueError("evaluation identity differs; use another output directory")
    if not plan_file.exists() and any(output.iterdir()):
        raise ValueError("refusing to adopt an output directory without a pinned plan")
    _write_json(plan_file, plan)
    frozen = output / "frozen_model"
    frozen.mkdir(exist_ok=True)
    for name, digest in plan["frozen_files"].items():
        destination = frozen / name
        if destination.exists():
            if file_digest(destination) != digest:
                raise ValueError("frozen evaluation artifact was changed")
        else:
            temporary = frozen / f"{name}.tmp"
            shutil.copyfile(run_dir / name, temporary)
            if file_digest(temporary) != digest:
                raise ValueError("source artifact changed while being pinned")
            os.replace(temporary, destination)
    return plan, metadata, labels


def evaluate_endpoint(model, dataset, output, device, batch_size, workers, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt_file = output / "metrics.json"
    archive = output / "predictions.npz"
    if receipt_file.exists():
        prior = json.loads(receipt_file.read_text("utf-8"))
        if (
            prior.get("status") != "completed"
            or prior["samples"] != len(dataset)
            or prior["ordered_ids_sha256"] != dataset.ordered_ids_sha256
            or file_digest(archive) != prior["predictions_sha256"]
        ):
            raise ValueError("completed endpoint receipt differs from current support")
        return prior
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        **({"multiprocessing_context": "spawn"} if workers else {}),
    )
    names = ("baseline", "fusion", "hog_off")
    predictions = {name: [] for name in names}
    top1 = {name: [] for name in names}
    losses = {name: [] for name in names}
    seen = 0
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = targets.to(device, non_blocking=True)
            cnn, hog = model.extract(images)
            baseline = model.cnn.classifier(cnn)
            fusion = model.forward_cached(cnn, hog)
            ablated = model.fusion(
                torch.cat((cnn, cnn.new_zeros((len(cnn), model.hog_dim))), dim=1)
            )
            logits = {
                "baseline": baseline,
                "fusion": fusion,
                "hog_off": model.cnn.classifier(ablated),
            }
            for name, value in logits.items():
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f"nonfinite {name} logits")
                first, five = prediction_indices(value)
                top1[name].append(first)
                predictions[name].append(five)
                losses[name].append(
                    F.cross_entropy(value, labels, reduction="none").cpu().numpy()
                )
            seen += len(targets)
            if batch_index % 100 == 0 or seen == len(dataset):
                report(completed=seen, total=len(dataset))
    if seen != len(dataset):
        raise RuntimeError("endpoint evaluation did not consume every example")
    predictions = {name: np.concatenate(value) for name, value in predictions.items()}
    top1 = {name: np.concatenate(value) for name, value in top1.items()}
    losses = {name: np.concatenate(value) for name, value in losses.items()}
    metrics = summarize_predictions(
        dataset.targets,
        predictions,
        losses,
        model.cnn.classifier.out_features,
        top1=top1,
    )
    temporary = archive.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            targets=dataset.targets,
            sample_ids=np.asarray(dataset.ids),
            writers=np.asarray(dataset.writers),
            **{f"{name}_top5": value for name, value in predictions.items()},
            **{f"{name}_top1": value for name, value in top1.items()},
            **{f"{name}_loss": value for name, value in losses.items()},
        )
    os.replace(temporary, archive)
    metrics.update(
        status="completed",
        full_endpoint=True,
        failures=0,
        elapsed_seconds=time.perf_counter() - started,
        elapsed_scope="full_loader_and_three_paired_heads_not_single_model_latency",
        ordered_ids_sha256=dataset.ordered_ids_sha256,
        predictions_sha256=file_digest(archive),
        writers=len(set(dataset.writers)),
        identity_reconciliation="manifest_namespace_ids_only",
    )
    _write_json(receipt_file, metrics)
    dataset.store.close()
    return metrics


def benchmark_forward(model, image, *, warmup=20, iterations=100, repetitions=5):
    """Alternating batch-one eager timing, including HOG only for fusion."""
    device = image.device
    precision = resolve_precision(model.backbone_precision, str(device))

    def baseline():
        with precision.autocast():
            features = model.cnn._forward_embedding(image)
        return model.cnn.classifier(features.float())

    functions = {"baseline": baseline, "fusion": lambda: model(image)}

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model.eval()
    timings = {name: [] for name in functions}
    with torch.inference_mode():
        for function in functions.values():
            for _ in range(warmup):
                function()
        synchronize()
        for repetition in range(repetitions):
            order = (
                list(functions) if repetition % 2 == 0 else list(reversed(functions))
            )
            for name in order:
                block = []
                for _ in range(iterations):
                    synchronize()
                    start = time.perf_counter_ns()
                    result = functions[name]()
                    synchronize()
                    block.append((time.perf_counter_ns() - start) / 1e6)
                    if not torch.isfinite(result).all():
                        raise FloatingPointError("nonfinite benchmark output")
                timings[name].append(block)
    summary = {}
    for name, blocks in timings.items():
        p50 = [float(np.percentile(block, 50)) for block in blocks]
        p95 = [float(np.percentile(block, 95)) for block in blocks]
        summary[name] = {
            "median_block_p50_ms": float(np.median(p50)),
            "median_block_p95_ms": float(np.median(p95)),
            "block_p50_range_ms": [min(p50), max(p50)],
            "block_p95_range_ms": [min(p95), max(p95)],
            "raw_blocks_ms": blocks,
        }
    return {
        "status": "completed",
        "batch_size": 1,
        "warmup": warmup,
        "iterations_per_block": iterations,
        "repetitions": repetitions,
        "scope": "eager_forward_on_preprocessed_device_tensor_including_hog_for_fusion",
        "excluded": ["PIL_preprocessing", "disk_loading", "host_to_device_transfer"],
        "paired_order": "baseline_fusion_then_fusion_baseline_alternating",
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "failures": 0,
        "limitation": "host_not_isolated_no_kernel_limiter_profile_no_speedup_claim",
        "measurements": summary,
    }


def run_hog_evaluation(args, *, expected_counts=None):
    counts = OFFICIAL_COUNTS if expected_counts is None else expected_counts
    if set(counts) != set(OFFICIAL_COUNTS) or any(n < 1 for n in counts.values()):
        raise ValueError("both complete endpoints must have positive declared counts")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output = Path(args.output_dir).resolve()
    plan, metadata, labels = pin_evaluation(
        args.run_dir,
        args.manifest,
        output,
        {
            "device": args.device,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "cpu_threads": args.cpu_threads,
            "expected_counts": counts,
            "latency": not args.skip_latency,
        },
    )
    model = load_hog_fusion_run(output / "frozen_model", args.device)
    results = {}
    sample = None
    for split, count in counts.items():
        dataset = HoldoutImages(
            args.manifest,
            split,
            labels["labels"],
            metadata["preprocess"],
            plan["manifest_sha256"],
            count,
        )

        def progress(*, completed, total, endpoint=split):
            status = {
                "stage": "evaluate",
                "split": endpoint,
                "completed": completed,
                "total": total,
            }
            _write_json(output / "status.json", status)
            print(json.dumps(status), flush=True)

        results[split] = evaluate_endpoint(
            model,
            dataset,
            output / split,
            args.device,
            args.batch_size,
            args.num_workers,
            progress,
        )
        if sample is None and not args.skip_latency:
            sample = dataset[0][0][None].to(args.device)
        dataset.store.close()
    if not args.skip_latency:
        latency_path = output / "forward_latency.json"
        if latency_path.exists():
            latency = json.loads(latency_path.read_text("utf-8"))
        else:
            latency = benchmark_forward(model, sample)
            _write_json(latency_path, latency)
        results["forward_latency"] = latency
    report = {
        "status": "completed",
        "selection_frozen_before_images": True,
        "plan_sha256": file_digest(output / "evaluation_plan.json"),
        "results": results,
    }
    _write_json(output / "report.json", report)
    _write_json(output / "status.json", {"stage": "completed", "splits": list(counts)})
    return report
