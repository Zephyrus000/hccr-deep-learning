"""Compare local LR stability from identical saved weights on fit/validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diagnose_partial_batch_norm import parameter_digest, validation_accuracy
from torch.utils.data import DataLoader, Subset

from hccr.data.dataset import HCCRDataset
from hccr.models import load_model_from_run
from hccr.preprocessing import EvalPreprocessor
from hccr.preprocessing.batch import TensorBatchAugmenter
from hccr.training.precision import resolve_precision
from hccr.training.trainer import _batch_norm_tail_momentum


def diagnose_learning_rate(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(20261023)
    config = json.loads((args.run_dir / "config.json").read_text())
    manifest = Path(config["manifest_path"])
    batch_size = config["batch_size"]
    model = load_model_from_run(args.run_dir, args.device)
    source_digest = parameter_digest(model)
    precision = resolve_precision(config["precision"], args.device)
    transform = EvalPreprocessor(config["image_size"])
    fit = HCCRDataset(manifest, "train", transform, storage_backend="lmdb")
    validation = HCCRDataset(manifest, "validation", transform, storage_backend="lmdb")
    chosen = torch.randperm(
        len(validation), generator=torch.Generator().manual_seed(20261023)
    )[:8192].tolist()
    validation_batches = [
        (x.to(args.device), y.to(args.device))
        for x, y, _ in DataLoader(Subset(validation, chosen), batch_size=batch_size)
    ]
    report = {
        "learning_rate": args.learning_rate,
        "source_parameter_digest": source_digest,
        "source_run": str(args.run_dir),
        "validation_samples": len(chosen),
        "baseline_top1": validation_accuracy(model, validation_batches, precision),
        "scope": (
            "Local stability probe from a development checkpoint with reset AdamW; "
            "not a from-scratch hyperparameter search or final paper run."
        ),
        "measurements": [],
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=config["weight_decay"]
    )
    order = torch.randperm(
        len(fit), generator=torch.Generator().manual_seed(config["seed"] + 15)
    )[: args.batches * batch_size].tolist()
    augmenter = TensorBatchAugmenter(
        config["image_size"],
        rotation_degrees=config["train_rotation_degrees"],
        translate_ratio=config["train_translate_ratio"],
        blur_probability=config["train_blur_probability"],
        seed=config["seed"],
    )
    augmenter.set_epoch(15)
    layers = [
        (layer, layer.momentum)
        for layer in model.modules()
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
        and layer.momentum is not None
    ]
    losses = []
    for step, (images, targets, _) in enumerate(
        DataLoader(Subset(fit, order), batch_size=batch_size), start=1
    ):
        model.train()
        images = images.mul(255).round().to(torch.uint8).to(args.device)
        images, _ = augmenter(images.float().mul_(1 / 255))
        targets = targets.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with (
            _batch_norm_tail_momentum(layers, len(images), batch_size),
            precision.autocast(),
        ):
            logits = model(images, targets, 1.0)
        loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite loss during LR stability probe")
        precision.backward(loss)
        precision.step(optimizer)
        losses.append(float(loss.detach()))
        if step % 128 == 0 or step == args.batches:
            row = {
                "optimizer_steps": step,
                "mean_loss": sum(losses) / len(losses),
                "validation_top1": validation_accuracy(
                    model, validation_batches, precision
                ),
            }
            report["measurements"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            losses.clear()
    fit.close()
    validation.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batches", type=int, default=512)
    args = parser.parse_args()
    if args.learning_rate <= 0 or args.batches < 1:
        parser.error("Learning rate and batch count must be positive")
    diagnose_learning_rate(args)
