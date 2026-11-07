"""Matched CNN-only/HOG/TDA fine-tuning, with plain CNN deployment artifacts."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from hccr.artifacts.checkpoints import file_digest
from hccr.models.artifacts import load_model_from_run, read_checkpoint_metadata
from hccr.models.descriptor_supervision import DescriptorSupervision, SupervisionConfig
from hccr.preprocessing.batch import TensorBatchAugmenter
from hccr.training.descriptor_targets import (
    IndexedImages,
    TargetScaler,
    load_tda_cache,
    prepare_tda_cache,
    target_identity,
)
from hccr.training.hog_fusion import write_json
from hccr.training.input_cache import build_training_cache
from hccr.training.precision import resolve_precision
from hccr.training.trainer import _batch_norm_tail_momentum


@dataclass(frozen=True)
class DescriptorTrainingConfig:
    source_run: str
    manifest: str
    output_dir: str
    method: str = "none"
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    min_learning_rate: float = 1e-6
    hog_weight: float = 0.1
    tda_weight: float = 0.1
    auxiliary_warmup_epochs: int = 2
    seed: int = 7
    device: str = "cuda"
    precision: str = "source"
    num_workers: int = 2
    cpu_threads: int = 4
    tda_cache: str | None = None
    augmentation: str = "none"
    training_cache: str = "none"
    training_cache_max_gib: float = 20.0
    training_cache_reserve_gib: float = 4.0
    require_completed_teacher: bool = False

    def __post_init__(self):
        self.supervision()
        if (
            min(self.epochs, self.batch_size, self.cpu_threads) < 1
            or self.num_workers < 0
        ):
            raise ValueError("epochs/batch size/threads must be positive; workers >= 0")
        if not 0 < self.min_learning_rate <= self.learning_rate < float("inf"):
            raise ValueError("require 0 < minimum learning rate <= learning rate < inf")
        if not 0 <= self.weight_decay < float("inf"):
            raise ValueError("weight decay must be finite and nonnegative")
        if self.augmentation not in {"none", "gpu"}:
            raise ValueError("augmentation must be unaugmented (none) or gpu")
        if self.training_cache not in {"none", "cuda"}:
            raise ValueError("training cache must be none or cuda")
        if (self.augmentation == "gpu" or self.training_cache == "cuda") and (
            torch.device(self.device).type != "cuda"
        ):
            raise ValueError("GPU augmentation and CUDA cache require a CUDA device")
        if self.method in {"tda", "both"} and self.tda_cache is None:
            raise ValueError("TDA requires an explicit image-target cache directory")

    def supervision(self) -> SupervisionConfig:
        return SupervisionConfig(
            self.method, self.hog_weight, self.tda_weight, self.auxiliary_warmup_epochs
        )


def read_supervision_config(
    path: Path, method: str, *, seed: int | None = None, **overrides
) -> DescriptorTrainingConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if method not in payload["variants"]:
        raise ValueError(f"unknown supervision variant: {method}")
    values = {**payload["common"], **payload["variants"][method]}
    if "source_runs" in payload:
        source_runs = payload["source_runs"]
        if seed not in source_runs:
            raise ValueError(f"choose --seed from {list(source_runs)}")
        values["source_run"] = source_runs[seed]
        values["output_dir"] = values["output_dir"].format(seed=seed)
    if seed is not None:
        values["seed"] = seed
    values.update({key: value for key, value in overrides.items() if value is not None})
    if not values.get("source_run"):
        raise ValueError(
            f"seed {seed} requires --source-run with a native FP32 teacher"
        )
    values.setdefault("method", method)
    return DescriptorTrainingConfig(**values)


def load_supervision_data(config: DescriptorTrainingConfig):
    source, manifest = Path(config.source_run), Path(config.manifest)
    metadata = read_checkpoint_metadata(source)
    labels_payload = json.loads((source / "labels.json").read_text(encoding="utf-8"))
    if (
        json.loads(manifest.with_name("labels.json").read_text(encoding="utf-8"))
        != labels_payload
    ):
        raise ValueError("manifest vocabulary differs from the source checkpoint")
    labels = labels_payload["labels"]
    if set(labels) != {str(i) for i in range(metadata["model"]["num_classes"])}:
        raise ValueError("source labels must have contiguous class IDs")
    digest = file_digest(manifest)
    datasets = {
        split: IndexedImages(manifest, split, labels, metadata["preprocess"], digest)
        for split in ("train", "validation")
    }
    if set(datasets["train"].writers) & set(datasets["validation"].writers):
        raise ValueError("train and validation writers overlap")
    if set(datasets["train"].ids) & set(datasets["validation"].ids):
        raise ValueError("train and validation sample identities overlap")
    return metadata, datasets, digest


def prepare_supervision_targets(config: DescriptorTrainingConfig) -> dict:
    """Explicit preparation command, separate from model training."""
    if config.method not in {"tda", "both"}:
        raise ValueError("only TDA variants need disk target preparation")
    metadata, datasets, digest = load_supervision_data(config)
    receipts = {}
    try:
        for split, dataset in datasets.items():
            identity = target_identity(dataset, digest, metadata["preprocess"])
            values = prepare_tda_cache(
                dataset,
                Path(config.tda_cache) / split,
                identity,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )
            receipts[split] = {"samples": len(values), "identity": identity}
        return receipts
    finally:
        for dataset in datasets.values():
            dataset.store.close()


@torch.no_grad()
def evaluate_supervision(model, dataset, config, precision) -> dict:
    """Validation only; auxiliary targets never select the checkpoint."""
    model.eval()
    correct, samples, loss_sum = 0, 0, 0.0
    for images, targets, _ in DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=str(config.device).startswith("cuda"),
    ):
        images, targets = images.to(config.device), targets.to(config.device)
        with precision.autocast():
            logits = model(images)
        correct += int((logits.argmax(1) == targets).sum())
        loss_sum += float(F.cross_entropy(logits.float(), targets, reduction="sum"))
        samples += len(images)
    if samples != len(dataset) or not samples:
        raise RuntimeError("validation did not consume the complete split")
    return {
        "samples": samples,
        "top1": correct / samples,
        "cross_entropy": loss_sum / samples,
    }


def save_supervision_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def train_supervision_epoch(
    model,
    dataset,
    config,
    precision,
    optimizer,
    epoch,
    tda,
    scaler,
    cached=None,
    augmenter=None,
):
    model.train()
    if cached is None:
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            generator=torch.Generator().manual_seed(config.seed + epoch),
            pin_memory=str(config.device).startswith("cuda"),
        )
    else:
        cached.set_epoch(epoch)
        loader = cached.indexed_batches()
    if augmenter is not None:
        augmenter.set_epoch(epoch)
    separate_tda_view = augmenter is not None and scaler is not None
    tda_weight = config.tda_weight * model.config.scale(epoch)
    batch_norms = [
        (module, module.momentum)
        for module in model.cnn.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
        and module.momentum is not None
    ]
    totals = dict.fromkeys(("loss", "classification_loss", "hog_loss", "tda_loss"), 0.0)
    samples, steps = 0, 0
    active = (
        torch.from_numpy(scaler.active).to(config.device)
        if scaler is not None
        else None
    )
    for images, targets, indices in loader:
        clean, targets = images.to(config.device), targets.to(config.device)
        images = augmenter(clean)[0] if augmenter is not None else clean
        tda_batch = None
        if scaler is not None:
            tda_batch = (
                tda[indices]
                if isinstance(tda, torch.Tensor)
                else scaler.transform(tda[indices.numpy()]).to(config.device)
            )
        optimizer.zero_grad(set_to_none=True)
        with (
            _batch_norm_tail_momentum(batch_norms, len(images), config.batch_size),
            precision.autocast(),
        ):
            losses = model.training_losses(
                images,
                targets,
                tda_targets=tda_batch,
                tda_active=active,
                separate_tda_view=separate_tda_view,
                epoch=epoch,
            )
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("nonfinite supervision training loss")
        precision.backward(losses["loss"])
        if separate_tda_view:
            # Sequential backward keeps only one view's activations live at a time.
            for module, _ in batch_norms:
                module.eval()
            try:
                with precision.autocast():
                    clean_loss = model.clean_topology_loss(clean, tda_batch, active)
                if not torch.isfinite(clean_loss):
                    raise RuntimeError("nonfinite clean-view topology loss")
                precision.backward(tda_weight * clean_loss)
            finally:
                for module, _ in batch_norms:
                    module.train()
            losses["tda_loss"] = clean_loss.detach()
            losses["loss"] = losses["loss"].detach() + tda_weight * clean_loss.detach()
        precision.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        steps += int(precision.step(optimizer))
        for name in totals:
            totals[name] += float(losses[name].detach()) * len(images)
        samples += len(images)
    if samples != len(dataset) or not samples:
        raise RuntimeError("training did not consume the complete split")
    return {
        **{name: value / samples for name, value in totals.items()},
        "samples": samples,
        "optimizer_steps": steps,
    }


def run_descriptor_training(
    config: DescriptorTrainingConfig, *, resume: bool = False
) -> dict:
    """Run only when explicitly invoked; never opens test/ICDAR images."""
    output = Path(config.output_dir)
    if output.exists() and not resume:
        raise FileExistsError("output exists; use --resume for an identical run")
    if resume and not (output / "latest.pt").is_file():
        raise ValueError("resume requires a completed epoch in latest.pt")
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = Path(config.source_run)
    if config.require_completed_teacher:
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        source_curves = json.loads((source / "curves.json").read_text(encoding="utf-8"))
        source_metrics = json.loads(
            (source / "metrics.json").read_text(encoding="utf-8")
        )
        if (
            source_config.get("seed") != config.seed
            or source_config.get("epochs") != config.epochs
            or [row["epoch"] for row in source_curves["epochs"]]
            != list(range(1, config.epochs + 1))
            or source_metrics.get("validation", {}).get("selected_checkpoint")
            != "checkpoint.pt"
        ):
            raise ValueError(
                "source must be a completed matched-seed teacher with checkpoint.pt"
            )
    metadata, datasets, digest = load_supervision_data(config)
    try:
        source_precision = metadata["training"]["precision"]["resolved"]
        requested = (
            source_precision if config.precision == "source" else config.precision
        )
        precision = resolve_precision(requested, config.device)
        if precision.resolved != source_precision:
            raise ValueError(
                "fine-tuning precision must match source checkpoint arithmetic"
            )
        tda, scaler = None, None
        target_provenance = None
        if config.method in {"tda", "both"}:
            identity = target_identity(
                datasets["train"], digest, metadata["preprocess"]
            )
            cache = Path(config.tda_cache) / "train"
            tda = load_tda_cache(cache, identity)
            scaler = TargetScaler.fit(tda)
            target_provenance = {
                "identity": identity,
                "sha256": file_digest(cache / "targets.npy"),
                "scaler": scaler.metadata(),
            }
        cnn = load_model_from_run(source, config.device)
        model = DescriptorSupervision(
            cnn, config.supervision(), metadata["preprocess"]["image_size"]
        ).to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.epochs, eta_min=config.min_learning_rate
        )
        implementation = {
            str(path.relative_to(Path(__file__).parents[1])): file_digest(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("descriptor_targets.py"),
                Path(__file__).parents[1] / "models/descriptor_supervision.py",
                Path(__file__).parents[1] / "models/hog_fusion.py",
                Path(__file__).parents[1] / "models/efficient_hccr.py",
                Path(__file__).parents[1] / "models/heads.py",
                Path(__file__).with_name("hog_fusion.py"),
                Path(__file__).with_name("trainer.py"),
                Path(__file__).with_name("precision.py"),
                Path(__file__).with_name("input_cache.py"),
                Path(__file__).parents[1] / "preprocessing/batch.py",
                Path(__file__).parents[1] / "preprocessing/pipeline.py",
                Path(__file__).parents[1] / "data/lmdb_store.py",
            )
        }
        identity = {
            "config": asdict(config),
            "manifest_sha256": digest,
            "source_checkpoint_sha256": file_digest(source / "checkpoint.pt"),
            "source_metadata_sha256": file_digest(source / "checkpoint_metadata.json"),
            "labels_sha256": file_digest(source / "labels.json"),
            "implementation": implementation,
            "precision": precision.metadata(),
            "versions": {
                name: version(name)
                for name in ("torch", "numpy", "gudhi")
                if name != "gudhi" or tda is not None
            },
            "data": {
                split: {
                    "samples": len(ds),
                    "writers": len(set(ds.writers)),
                    "ordered_ids_sha256": ds.ordered_ids_sha256,
                }
                for split, ds in datasets.items()
            },
            "tda": target_provenance,
        }
        augmenter = (
            TensorBatchAugmenter(metadata["preprocess"]["image_size"], seed=config.seed)
            if config.augmentation == "gpu"
            else None
        )
        identity["augmentation"] = (
            augmenter.metadata() if augmenter is not None else None
        )
        cached, cache_info = None, None
        history, start, best_epoch = [], 0, 0
        state = None
        if resume:
            state = torch.load(
                output / "latest.pt", map_location="cpu", weights_only=True
            )
            if state["identity"] != identity:
                raise ValueError(
                    "resume configuration, source, runtime or data changed"
                )
            start = state["epoch"]
        if start < config.epochs and config.training_cache == "cuda":
            cached, cache_info = build_training_cache(
                datasets["train"],
                config.batch_size,
                config.device,
                placement="cuda",
                max_gib=config.training_cache_max_gib,
                reserve_gib=config.training_cache_reserve_gib,
                seed=config.seed,
                num_workers=config.num_workers,
                model=model,
            )
            if tda is not None:
                tda = scaler.transform(tda).to(config.device)
        output.mkdir(parents=True, exist_ok=resume)
        if cache_info is not None:
            write_json(output / "training_cache.json", cache_info)
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if precision.scaler is not None:
                precision.scaler.load_state_dict(state["scaler"])
            torch.set_rng_state(state["rng"])
            if str(config.device).startswith("cuda"):
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            history = state["history"]
            best, best_epoch = state["best"], state["best_epoch"]
            selected_cnn = state["selected_cnn"]
        else:
            # Align classification dropout RNG across matched methods.
            torch.manual_seed(config.seed)
            best = evaluate_supervision(
                model, datasets["validation"], config, precision
            )
            selected_cnn = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.cnn.state_dict().items()
            }
            write_json(output / "identity.json", identity)
            write_json(
                output / "labels.json",
                json.loads((source / "labels.json").read_text(encoding="utf-8")),
            )
        for epoch in range(start + 1, config.epochs + 1):
            train = train_supervision_epoch(
                model,
                datasets["train"],
                config,
                precision,
                optimizer,
                epoch,
                tda,
                scaler,
                cached,
                augmenter,
            )
            validation = evaluate_supervision(
                model, datasets["validation"], config, precision
            )
            history.append(
                {
                    "epoch": epoch,
                    "train": train,
                    "validation": validation,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "auxiliary_scale": config.supervision().scale(epoch),
                }
            )
            if (validation["top1"], -validation["cross_entropy"]) > (
                best["top1"],
                -best["cross_entropy"],
            ):
                best, best_epoch = validation, epoch
                selected_cnn = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.cnn.state_dict().items()
                }
            scheduler.step()
            save_supervision_state(
                output / "latest.pt",
                {
                    "identity": identity,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": precision.scaler.state_dict()
                    if precision.scaler
                    else None,
                    "epoch": epoch,
                    "history": history,
                    "best": best,
                    "best_epoch": best_epoch,
                    "selected_cnn": selected_cnn,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if str(config.device).startswith("cuda")
                    else [],
                },
            )
            write_json(output / "curves.json", {"epochs": history})
            print(json.dumps(history[-1]), flush=True)
        save_supervision_state(output / "checkpoint.pt", selected_cnn)
        deployed_metadata = deepcopy(metadata)
        deployed_metadata["training"] = {
            "precision": precision.metadata(),
            "supervision": asdict(config),
            "selected_epoch": best_epoch,
            "source_run": str(source.resolve()),
            "selection_split": "validation",
            "augmentation": config.augmentation,
        }
        write_json(output / "checkpoint_metadata.json", deployed_metadata)
        result = {
            "status": "completed",
            "method": config.method,
            "epochs": config.epochs,
            "selected_epoch": best_epoch,
            "validation": best,
            "test_status": "not_run",
            "icdar_status": "not_run",
            "deployment_parameters": sum(p.numel() for p in model.cnn.parameters()),
            "auxiliary_parameters": sum(
                p.numel()
                for name, p in model.named_parameters()
                if not name.startswith("cnn.")
            ),
        }
        write_json(output / "result.json", result)
        return result
    finally:
        for dataset in datasets.values():
            dataset.store.close()
