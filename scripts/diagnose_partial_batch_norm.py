"""Isolate final-batch BatchNorm effects on a saved model, without optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from hccr.data.dataset import HCCRDataset
from hccr.models import load_model_from_run
from hccr.preprocessing import EvalPreprocessor
from hccr.preprocessing.batch import TensorBatchAugmenter
from hccr.training.precision import resolve_precision


def parameter_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def validation_accuracy(model, batches, precision) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in batches:
        with precision.autocast():
            predicted = model(images).argmax(dim=1)
        correct += int(predicted.eq(targets).sum())
        total += len(targets)
    return correct / total


@torch.no_grad()
def diagnose_partial_batch_norm(arguments: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    config = json.loads((arguments.run_dir / "config.json").read_text())
    manifest = Path(config["manifest_path"])
    batch_size = int(config["batch_size"])
    seed = int(config["seed"])
    precision = resolve_precision(config["precision"], arguments.device)
    model = load_model_from_run(arguments.run_dir, arguments.device).eval()
    initial_digest = parameter_digest(model)
    transform = EvalPreprocessor(int(config["image_size"]))
    fit = HCCRDataset(manifest, "train", transform, storage_backend="lmdb")
    validation = HCCRDataset(manifest, "validation", transform, storage_backend="lmdb")
    count = min(arguments.validation_samples, len(validation))
    indices = torch.randperm(
        len(validation), generator=torch.Generator().manual_seed(20261023)
    )[:count].tolist()
    batches = [
        (images.to(arguments.device), targets.to(arguments.device))
        for images, targets, _ in DataLoader(
            Subset(validation, indices), batch_size=batch_size, num_workers=0
        )
    ]
    report = {
        "source_run": str(arguments.run_dir),
        "fit_samples": len(fit),
        "validation_samples": count,
        "parameter_digest": initial_digest,
        "baseline_top1": validation_accuracy(model, batches, precision),
        "optimizer_steps": 0,
        "scope": (
            "Fixed saved weights; replay exact epoch sample order and augmentation "
            "RNG tail. This isolates BatchNorm buffer sensitivity, not the full "
            "historical optimizer trajectory. No test/calibration data are used."
        ),
        "epochs": [],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in arguments.epochs:
        candidate = deepcopy(model).eval()
        layers = [
            layer
            for layer in candidate.modules()
            if isinstance(layer, nn.modules.batchnorm._BatchNorm)
            and layer.momentum is not None
        ]
        for layer in layers:
            layer.train()
        order = torch.randperm(
            len(fit), generator=torch.Generator().manual_seed(seed + epoch)
        )
        tail_start = max(
            0, (len(order) // batch_size - arguments.tail_batches) * batch_size
        )
        augmenter = TensorBatchAugmenter(
            int(config["image_size"]),
            rotation_degrees=config["train_rotation_degrees"],
            translate_ratio=config["train_translate_ratio"],
            blur_probability=config["train_blur_probability"],
            seed=seed,
        )
        augmenter.set_epoch(epoch)
        for _ in range(0, tail_start, batch_size):
            torch.rand((batch_size, 5), generator=augmenter._generator)
        tail = DataLoader(
            Subset(fit, order[tail_start:].tolist()),
            batch_size=batch_size,
            num_workers=0,
        )
        partial = None
        for images, _targets, _metadata in tail:
            # Match the production uint8 cache conversion before augmentation.
            images = images.mul(255).round().clamp_(0, 255).to(torch.uint8)
            images, _ = augmenter(images.to(arguments.device).float().mul_(1.0 / 255))
            if len(images) < batch_size:
                partial = images
                break
            with precision.autocast():
                candidate(images)
        if partial is None:
            raise ValueError("Selected fit split has no partial final batch")
        saved = {name: value.clone() for name, value in candidate.named_buffers()}
        before = validation_accuracy(candidate, batches, precision)
        row = {
            "epoch_sample_order": epoch,
            "replayed_full_batches": (len(order) - tail_start) // batch_size,
            "partial_batch_size": len(partial),
            "before_partial_top1": before,
        }
        momentums = [layer.momentum for layer in layers]
        for policy in ("legacy", "sample_weighted"):
            for name, value in candidate.named_buffers():
                value.copy_(saved[name])
            for layer, momentum in zip(layers, momentums, strict=True):
                layer.train()
                layer.momentum = (
                    1 - (1 - momentum) ** (len(partial) / batch_size)
                    if policy == "sample_weighted"
                    else momentum
                )
            with precision.autocast():
                candidate(partial)
            row[f"{policy}_top1"] = validation_accuracy(candidate, batches, precision)
            for layer, momentum in zip(layers, momentums, strict=True):
                layer.momentum = momentum
        row["parameters_unchanged"] = parameter_digest(candidate) == initial_digest
        if not row["parameters_unchanged"]:
            raise RuntimeError("Diagnostic unexpectedly changed model weights")
        report["epochs"].append(row)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    fit.close()
    validation.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, nargs="+", default=[12, 13, 14])
    parser.add_argument("--tail-batches", type=int, default=128)
    parser.add_argument("--validation-samples", type=int, default=8192)
    args = parser.parse_args()
    if min(args.tail_batches, args.validation_samples) < 1 or min(args.epochs) < 0:
        parser.error("Sample/batch counts must be positive and epochs nonnegative")
    diagnose_partial_batch_norm(args)
