"""Rerunnable stage 2 PH pilot with an explicit held-out evaluation gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from heapq import nsmallest
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Subset
from torchvision.transforms.v2 import functional as vision
from tqdm.auto import tqdm

from hccr.config.loader import load_yaml
from hccr.data.dataset import resolve_data_root
from hccr.data.gnt import load_source_image
from hccr.data.lmdb_store import LMDBImageStore
from hccr.models import load_model_from_run, read_checkpoint_metadata
from hccr.models.heads import AngularMarginClassifier
from hccr.preprocessing import EvalPreprocessor
from hccr.tda.cubical import cubical_features, distribution_statistics
from hccr.tda.features import (
    VARIANTS,
    CacheWriter,
    FittedBlocks,
    FusedRows,
    auxiliary_block,
    digest_file,
    implementation_identity,
    load_cache,
    signature,
    write_json,
)
from hccr.tda.projection import (
    FrozenStage2Adapter,
    ProjectionBudget,
    ScalarProjection,
    calibrate_maps,
    fit_projection,
    load_projection,
    projection_digest,
    save_projection,
)
from hccr.training.precision import resolve_precision


@dataclass(frozen=True)
class PilotConfig:
    run_dir: str
    manifest: str
    output_dir: str = "artifacts/stage2-ph"
    seed: int = 7
    seeds: tuple[int, ...] = (7, 17, 29)
    class_subset_seed: int = 7
    max_classes: int = 1000
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_holdout_samples: int | None = None
    device: str = "auto"
    dataset_backend: str = "auto"
    batch_size: int = 256
    num_workers: int = 0
    projection_epochs: int = 20
    projection_learning_rate: float = 3e-4
    projection_weight_decay: float = 1e-4
    calibration_samples: int = 4096
    head_epochs: int = 8
    head_batch_size: int = 2048
    head_learning_rate: float = 0.001
    head_weight_decay: float = 0.0001
    precision: str = "float32"
    alpha_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    variants: tuple[str, ...] = VARIANTS


class CompactPlan(Sequence):
    """Keep only packed UTF-8 sample IDs and integer labels for LMDB workers."""

    def __init__(self, rows, class_map):
        sample_ids, targets = [], []
        for row in rows:
            sample_ids.append(row["sample_id"].encode("utf-8"))
            targets.append(class_map[row["unicode_label"]])
        self.sample_ids = np.asarray(sample_ids, dtype="S")
        self.targets = np.asarray(targets, dtype=np.int32)
        self.labels = {value: key for key, value in class_map.items()}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        return {
            "sample_id": self.sample_ids[index].decode("utf-8"),
            "unicode_label": self.labels[int(self.targets[index])],
        }


class PlannedImages(Dataset):
    """A bounded sample plan using the existing GNT/LMDB and image transforms."""

    def __init__(
        self,
        manifest: Path,
        rows: Sequence[dict],
        transform,
        class_map: dict,
        backend: str,
        manifest_sha256: str,
    ) -> None:
        self.rows, self.transform, self.class_map = rows, transform, class_map
        store = manifest.with_name("images.lmdb")
        self.store = (
            LMDBImageStore(store, manifest_sha256)
            if backend == "lmdb" or (backend == "auto" and store.is_file())
            else None
        )
        self.root = resolve_data_root(manifest, rows) if self.store is None else None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = (
            self.store.read_image(row["sample_id"])
            if self.store
            else load_source_image(self.root, row)
        )
        image = self.transform(image)
        tensor = vision.to_dtype(vision.to_image(image), torch.float32, scale=True)
        return tensor, self.class_map[row["unicode_label"]], index


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict:
    """Reduce batches to class counters; never retain an N-by-classes matrix."""
    batches = (
        (logits[start : start + 4096] for start in range(0, len(logits), 4096))
        if isinstance(logits, np.ndarray)
        else logits
    )
    support = np.zeros(int(targets.max()) + 1, dtype=np.int64)
    errors = np.zeros_like(support)
    offset = 0
    for batch in batches:
        predictions = batch.argmax(axis=1) if batch.ndim == 2 else batch
        labels = targets[offset : offset + len(predictions)]
        if len(labels) != len(predictions):
            raise ValueError("prediction and target counts differ")
        support += np.bincount(labels, minlength=len(support))
        errors += np.bincount(labels[predictions != labels], minlength=len(support))
        offset += len(labels)
    if offset != len(targets) or offset == 0:
        raise ValueError("prediction and target counts differ or are empty")
    classes = np.flatnonzero(support)
    return {
        "top1": float(1 - errors.sum() / offset),
        "samples": len(targets),
        "class_errors": {str(int(label)): int(errors[label]) for label in classes},
        "class_support": {str(int(label)): int(support[label]) for label in classes},
    }


class FeatureBatches(Dataset):
    """Read and fuse only the rows in one shuffled head-training batch."""

    def __init__(self, features, targets):
        self.features, self.targets = features, targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, indices):
        return (
            torch.from_numpy(
                np.array(self.features[indices], dtype=np.float32, copy=True)
            ),
            torch.from_numpy(
                np.array(self.targets[indices], dtype=np.int64, copy=True)
            ),
        )


def train_common_head(
    training: np.ndarray,
    targets: np.ndarray,
    validation: np.ndarray,
    validation_targets: np.ndarray,
    classes: int,
    config: PilotConfig,
    scale: float,
    margin: float,
    device: str,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(config.seed)
    head = AngularMarginClassifier(
        training.shape[1], classes, "cosface", scale, margin
    ).to(device)
    precision = resolve_precision(config.precision, device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.head_learning_rate,
        weight_decay=config.head_weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        FeatureBatches(training, targets),
        batch_size=None,
        sampler=BatchSampler(
            RandomSampler(range(len(training)), generator=generator),
            config.head_batch_size,
            drop_last=False,
        ),
        num_workers=0,
        generator=generator,
    )
    best_accuracy, best_state, history = -1.0, None, []
    for epoch in range(config.head_epochs):
        head.train()
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with precision.autocast():
                loss = F.cross_entropy(head(features, labels), labels)
            precision.backward(loss)
            precision.step(optimizer)
        metrics = classification_metrics(
            head_logits(head, validation, device, config.head_batch_size),
            validation_targets,
        )
        history.append({"epoch": epoch + 1, "top1": metrics["top1"]})
        if metrics["top1"] > best_accuracy:
            best_accuracy = metrics["top1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
    head.load_state_dict(best_state)
    head.eval()
    return head, {
        "validation_top1": best_accuracy,
        "selected_epoch": max(history, key=lambda item: item["top1"])["epoch"],
        "history": history,
        "precision": precision.metadata(),
        "parameters": sum(p.numel() for p in head.parameters()),
    }


@torch.no_grad()
def head_logits(
    head: nn.Module, values: np.ndarray, device: str, batch_size: int
) -> Iterator[np.ndarray]:
    """Yield one inference batch at a time to the streaming metric reducer."""
    head.eval()
    for start in range(0, len(values), batch_size):
        yield (
            head(torch.from_numpy(values[start : start + batch_size].copy()).to(device))
            .float()
            .cpu()
            .numpy()
        )


class Stage2Study:
    """One seed and fixed candidates; holdouts open only after all selections."""

    def __init__(self, config: PilotConfig) -> None:
        self.config = config
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        self.manifest = Path(config.manifest).resolve()
        self.run_dir = Path(config.run_dir).resolve()
        self.output = Path(config.output_dir).resolve() / f"seed-{config.seed}"
        self.output.mkdir(parents=True, exist_ok=True)
        self.metadata = read_checkpoint_metadata(self.run_dir)
        self.adapter = FrozenStage2Adapter(
            load_model_from_run(self.run_dir, self.device)
        ).eval()
        self.preprocess = dict(self.metadata["preprocess"])
        unsupported = set(self.preprocess) - {"image_size", "margin"}
        if unsupported:
            raise ValueError(
                f"unsupported checkpoint preprocessing: {sorted(unsupported)}"
            )
        self.transform = EvalPreprocessor(**self.preprocess)
        labels = json.loads((self.run_dir / "labels.json").read_text(encoding="utf-8"))[
            "labels"
        ]
        self.checkpoint_labels = {int(key): value for key, value in labels.items()}
        if len(self.checkpoint_labels) != self.metadata["model"]["num_classes"]:
            raise ValueError(
                "checkpoint labels do not match its classifier; "
                "supply original run artifacts"
            )
        train_labels, writers, missing, seen = set(), {}, 0, 0
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "train":
                    train_labels.add(row["unicode_label"])
                writer = row["writer_id"].strip()
                seen += 1
                missing += not bool(writer)
                if writer:
                    writers.setdefault(writer, set()).add(row["split"])
        if any(len(splits) > 1 for splits in writers.values()):
            raise ValueError("writer leakage between manifest splits; rebuild manifest")
        self.writer_provenance = {
            "missing_writer_samples": missing,
            "samples": seen,
            "writer_disjoint_verified": not missing if writers else None,
            "basis": "manifest_identifiers_not_biometric_identity",
            "baseline_pretraining_provenance": self.metadata.get("test", {}).get(
                "writer_provenance"
            ),
        }
        candidates = sorted(
            class_id
            for class_id, label in self.checkpoint_labels.items()
            if label in train_labels
        )
        if not 2 <= config.max_classes <= len(candidates):
            raise ValueError(
                f"max_classes must be in [2, {len(candidates)}] "
                "for shared TRAIN/checkpoint labels"
            )
        self.original_ids = sorted(
            Random(config.class_subset_seed).sample(candidates, config.max_classes)
        )
        self.class_map = {
            self.checkpoint_labels[class_id]: index
            for index, class_id in enumerate(self.original_ids)
        }
        self.identity = {
            "schema_version": 1,
            "checkpoint": digest_file(self.run_dir / "checkpoint.pt"),
            "checkpoint_metadata": digest_file(
                self.run_dir / "checkpoint_metadata.json"
            ),
            "checkpoint_labels": digest_file(self.run_dir / "labels.json"),
            "manifest": digest_file(self.manifest),
            "class_mapping": self.class_map,
            "original_checkpoint_class_ids": self.original_ids,
            "candidate_policy": "pilot_classes_only",
            "preprocessing": self.preprocess,
            "stage": "stage2",
            "extraction_precision": "float32",
            "execution_device": self.device,
            "cuda_device_name": (
                torch.cuda.get_device_name(self.device)
                if self.device.startswith("cuda")
                else None
            ),
            "cuda_runtime": torch.version.cuda,
            "deterministic_algorithms": True,
            "implementation": implementation_identity(),
            "seed": config.seed,
        }
        report_path = self.manifest.with_name("audit_report.json")
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("source_kind") == "gnt"
                and report.get("manifest_digest") != self.identity["manifest"]
            ):
                raise ValueError("manifest changed after its GNT audit")
        self.plans = {}
        write_json(
            self.output / "pilot_mapping.json",
            {
                "class_mapping": self.class_map,
                "original_checkpoint_class_ids": self.original_ids,
                "writer_provenance": self.writer_provenance,
            },
        )

    def dataset(self, split: str) -> PlannedImages:
        if split in {"test", "icdar"}:
            self.selected_comparison()
        limit = {
            "train": self.config.max_train_samples,
            "validation": self.config.max_validation_samples,
        }.get(split, self.config.max_holdout_samples)
        if split not in self.plans:
            with self.manifest.open(newline="", encoding="utf-8") as handle:
                rows = (
                    row
                    for row in csv.DictReader(handle)
                    if row["split"] == split and row["unicode_label"] in self.class_map
                )
                if limit is not None:
                    rows = nsmallest(
                        limit,
                        rows,
                        key=lambda row: hashlib.sha256(
                            f"{self.config.class_subset_seed}\0{row['sample_id']}".encode()
                        ).digest(),
                    )
                use_lmdb = self.config.dataset_backend == "lmdb" or (
                    self.config.dataset_backend == "auto"
                    and self.manifest.with_name("images.lmdb").is_file()
                )
                rows = CompactPlan(rows, self.class_map) if use_lmdb else list(rows)
            if not rows:
                raise ValueError(f"empty {split} sample plan")
            if split == "train" and {row["unicode_label"] for row in rows} != set(
                self.class_map
            ):
                raise ValueError(
                    "bounded train plan omits candidate classes; "
                    "increase max_train_samples"
                )
            self.plans[split] = rows
        return PlannedImages(
            self.manifest,
            self.plans[split],
            self.transform,
            self.class_map,
            self.config.dataset_backend,
            self.identity["manifest"],
        )

    def loader(self, dataset: Dataset) -> DataLoader:
        options = {"num_workers": self.config.num_workers}
        if self.config.num_workers:
            options.update(
                multiprocessing_context="spawn",
                persistent_workers=True,
                prefetch_factor=1,
            )
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False, **options
        )

    def fit_projection(self) -> Path:
        target = self.output / "projection.pt"
        if target.exists():
            raise FileExistsError(
                f"projection exists: {target}; reuse it or choose a new output_dir"
            )
        train, validation = self.dataset("train"), self.dataset("validation")
        train_loader, validation_loader = self.loader(train), self.loader(validation)
        budget = ProjectionBudget(
            self.config.projection_epochs,
            self.config.projection_learning_rate,
            self.config.projection_weight_decay,
            self.config.seed,
            self.config.precision,
        )
        projection, evidence = fit_projection(
            self.adapter,
            train_loader,
            validation_loader,
            len(self.class_map),
            self.device,
            budget,
        )
        # Drop persistent training/validation worker pools before calibration starts.
        del train_loader, validation_loader
        calibration_indices = sorted(
            Random(self.config.class_subset_seed).sample(
                range(len(train)), min(self.config.calibration_samples, len(train))
            )
        )
        calibration_loader = self.loader(Subset(train, calibration_indices))
        calibration = calibrate_maps(
            self.adapter,
            projection,
            calibration_loader,
            [row["sample_id"] for row in train.rows],
            self.device,
            self.config.calibration_samples,
        )
        fixed = (
            ScalarProjection(projection.in_channels)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        fixed_calibration = calibrate_maps(
            self.adapter,
            fixed,
            calibration_loader,
            [row["sample_id"] for row in train.rows],
            self.device,
            self.config.calibration_samples,
        )
        metadata = {
            "identity": self.identity,
            "training": evidence,
            "calibration": calibration,
            "fixed_calibration": fixed_calibration,
            "train_ids": [row["sample_id"] for row in train.rows],
            "validation_ids": [row["sample_id"] for row in validation.rows],
        }
        save_projection(target, projection, metadata)
        write_json(self.output / "projection.json", metadata)
        return target

    def projection(self) -> tuple[ScalarProjection, dict]:
        projection, metadata = load_projection(
            self.output / "projection.pt", self.device
        )
        if metadata["identity"] != self.identity:
            raise ValueError(
                "projection identity changed; refit into a new output directory"
            )
        for split in ("train", "validation"):
            ids = [row["sample_id"] for row in self.dataset(split).rows]
            if metadata[f"{split}_ids"] != ids:
                raise ValueError(
                    f"projection {split} sample plan changed; refit projection"
                )
        return projection, metadata

    def extract(self, split: str) -> dict[str, np.ndarray]:
        if split not in {"train", "validation"}:
            raise ValueError(
                "extract accepts only train/validation; "
                "use evaluate after all selections"
            )
        return self._extract(split)

    @torch.no_grad()
    def _extract(self, split: str) -> dict[str, np.ndarray]:
        projection, metadata = self.projection()
        dataset = self.dataset(split)
        cache_identity = {
            **self.identity,
            "split": split,
            "sample_ids": [row["sample_id"] for row in dataset.rows],
            "projection": projection_digest(projection),
            "fixed_projection": "channel_mean_1_over_C",
            "calibration": metadata["calibration"],
            "fixed_calibration": metadata["fixed_calibration"],
        }
        cache_dir = self.output / "features" / split
        if cache_dir.exists():
            return load_cache(cache_dir, cache_identity)
        fixed = (
            ScalarProjection(projection.in_channels)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        inspection = {}
        timing = {
            "backbone_seconds": 0.0,
            "projection_seconds": 0.0,
            "ph_cpu_seconds": 0.0,
        }
        with CacheWriter(cache_dir, cache_identity, len(dataset)) as writer:
            for images, targets, _ in tqdm(
                self.loader(dataset), desc=f"stage2 extract {split}"
            ):
                started = perf_counter()
                embedding, stage, original_logits = self.adapter(images.to(self.device))
                batch = {
                    "cnn": embedding.float().cpu().numpy(),
                    "gap": stage.float().mean(dim=(2, 3)).cpu().numpy(),
                    "original_predictions": original_logits[:, self.original_ids]
                    .argmax(dim=1)
                    .cpu()
                    .numpy(),
                    "targets": targets.numpy(),
                }
                timing["backbone_seconds"] += perf_counter() - started
                started = perf_counter()
                maps = (
                    (
                        projection(stage)[:, 0].double()
                        / metadata["calibration"]["scale"]
                    )
                    .cpu()
                    .numpy()
                )
                fixed_maps = (
                    (
                        fixed(stage)[:, 0].double()
                        / metadata["fixed_calibration"]["scale"]
                    )
                    .cpu()
                    .numpy()
                )
                timing["projection_seconds"] += perf_counter() - started
                batch["flat"] = maps.reshape(len(maps), -1).astype(np.float32)
                batch["statistics"] = np.stack(
                    [distribution_statistics(array) for array in maps]
                ).astype(np.float32)
                started = perf_counter()
                results = [cubical_features(array) for array in maps]
                fixed_results = [cubical_features(array) for array in fixed_maps]
                batch["h0_h1"] = np.stack(
                    [result.features for result in results]
                ).astype(np.float32)
                batch["mean_h0_h1"] = np.stack(
                    [result.features for result in fixed_results]
                ).astype(np.float32)
                timing["ph_cpu_seconds"] += perf_counter() - started
                writer.append(batch)
                if not inspection:
                    inspection["inspection_learned_map"] = maps[:1].copy()
                    inspection["inspection_mean_map"] = fixed_maps[:1].copy()
                    for dimension in (0, 1):
                        inspection[f"inspection_learned_h{dimension}"] = results[
                            0
                        ].diagrams[dimension]
                        inspection[f"inspection_mean_h{dimension}"] = fixed_results[
                            0
                        ].diagrams[dimension]
            writer.add_inspection(inspection)
        write_json(
            self.output / f"extraction_{split}.json",
            {
                "timing": timing,
                "timing_scope": "observed_component_wall_time_not_deployment_latency",
                "inspection_sample_id": dataset.rows[0]["sample_id"],
                "samples": len(dataset),
                "cache_storage": "batch_written_npy_memmap",
                "original_output": "candidate_restricted_top1_predictions_no_logits",
                "identity": signature(cache_identity),
            },
        )
        return load_cache(cache_dir, cache_identity)

    def training_signature(self) -> str:
        options = asdict(self.config)
        options.pop("output_dir")
        return signature(
            {
                "identity": self.identity,
                "config": options,
                "projection_file": digest_file(self.output / "projection.pt"),
            }
        )

    def compare(self) -> dict:
        if (self.output / "selection.json").exists():
            raise FileExistsError(
                "selection already exists; use evaluate or choose a new output_dir"
            )
        with self.extract("train") as train, self.extract("validation") as validation:
            return self._compare(train, validation)

    def _compare(self, train, validation) -> dict:
        selections = {}
        scale = float(self.metadata["model"].get("logit_scale", 32))
        margin = float(self.metadata["model"].get("angular_margin", 0.1))
        started = perf_counter()
        for variant in self.config.variants:
            if variant == "original":
                selections[variant] = {
                    "validation": classification_metrics(
                        validation["original_predictions"], validation["targets"]
                    ),
                    "refitted": False,
                }
                continue
            train_aux, val_aux = auxiliary_block(train, variant), auxiliary_block(
                validation, variant
            )
            blocks = FittedBlocks.fit(
                train["cnn"], train_aux, pca=variant in {"gap", "flat"}
            )
            trials, best = [], None
            for trial, alpha in enumerate(
                (0.0,) if variant == "cnn" else self.config.alpha_grid
            ):
                head, evidence = train_common_head(
                    FusedRows(train["cnn"], train_aux, blocks, alpha),
                    np.array(train["targets"], copy=True),
                    FusedRows(validation["cnn"], val_aux, blocks, alpha),
                    np.array(validation["targets"], copy=True),
                    len(self.class_map),
                    self.config,
                    scale,
                    margin,
                    self.device,
                )
                metrics = classification_metrics(
                    head_logits(
                        head,
                        FusedRows(validation["cnn"], val_aux, blocks, alpha),
                        self.device,
                        self.config.head_batch_size,
                    ),
                    validation["targets"],
                )
                evidence.update(alpha=alpha, validation=metrics)
                trials.append(evidence)
                if best is None or metrics["top1"] > best["validation"]["top1"]:
                    path = self.output / "heads" / f"{variant}-trial-{trial}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            name: value.detach().cpu()
                            for name, value in head.state_dict().items()
                        },
                        path,
                    )
                    best = {
                        "alpha": alpha,
                        "validation": metrics,
                        "head": str(path.relative_to(self.output)),
                        "head_sha256": digest_file(path),
                        "parameters": evidence["parameters"],
                        "selected_epoch": evidence["selected_epoch"],
                    }
            selections[variant] = {
                **best,
                "trials": trials,
                "transforms": blocks.state(),
                "refitted": True,
                "head_family": "cosface",
                "scale": scale,
                "margin": margin,
            }
        report = {
            "schema_version": 1,
            "status": "all_variants_selected",
            "training_signature": self.training_signature(),
            "selection_rule": "head_checkpoint_and_alpha_max_validation_top1_first_tie",
            "alpha_grid": list(self.config.alpha_grid),
            "variants": selections,
            "head_budget": {
                key: value
                for key, value in asdict(self.config).items()
                if key.startswith("head_")
            },
            "projection_training": json.loads(
                (self.output / "projection.json").read_text(encoding="utf-8")
            )["training"],
            "identity": self.identity,
            "writer_provenance": self.writer_provenance,
            "head_search_wall_seconds": perf_counter() - started,
            "backbone_parameters": sum(
                parameter.numel()
                for name, parameter in self.adapter.baseline.named_parameters()
                if not name.startswith("classifier.")
            ),
            "original_classifier_parameters": sum(
                parameter.numel()
                for parameter in self.adapter.baseline.classifier.parameters()
            ),
            "research_accuracy_gain_established": False,
        }
        for item in selections.values():
            item["delta_against_non_ph_controls"] = {
                control: item["validation"]["top1"]
                - selections[control]["validation"]["top1"]
                for control in ("cnn", "gap", "flat", "statistics")
                if control in selections
            }
        write_json(self.output / "selection.json", report)
        return report

    def selected_comparison(self) -> dict:
        path = self.output / "selection.json"
        if not path.is_file():
            raise ValueError(
                "finish compare for every variant before opening test or ICDAR"
            )
        selection = json.loads(path.read_text(encoding="utf-8"))
        if selection["status"] != "all_variants_selected" or set(
            selection["variants"]
        ) != set(self.config.variants):
            raise ValueError("incomplete variant selection; holdout evaluation denied")
        if selection["training_signature"] != self.training_signature():
            raise ValueError(
                "selection/config/projection changed; holdout evaluation denied"
            )
        return selection

    def evaluate(self, splits: tuple[str, ...] = ("test",)) -> dict:
        selection = self.selected_comparison()
        if not splits or any(split not in {"test", "icdar"} for split in splits):
            raise ValueError("held-out evaluation splits must be test or icdar")
        reports = {}
        for split in splits:
            with self._extract(split) as cache:
                reports[split] = self._evaluate_split(cache, selection)
        report = {
            "training_signature": selection["training_signature"],
            "held_out": reports,
            "candidate_policy": "pilot_classes_only",
            "bounded_holdout": self.config.max_holdout_samples is not None,
            "selection_unchanged": True,
        }
        write_json(self.output / "heldout.json", report)
        return report

    def _evaluate_split(self, cache, selection) -> dict:
        metrics = {}
        for variant, chosen in selection["variants"].items():
            if variant == "original":
                logits = cache["original_predictions"]
            else:
                head_path = self.output / chosen["head"]
                if digest_file(head_path) != chosen["head_sha256"]:
                    raise ValueError(f"selected head checkpoint changed: {head_path}")
                head = AngularMarginClassifier(
                    cache["cnn"].shape[1] + 14,
                    len(self.class_map),
                    "cosface",
                    chosen["scale"],
                    chosen["margin"],
                ).to(self.device)
                head.load_state_dict(
                    torch.load(head_path, map_location=self.device, weights_only=True)
                )
                blocks = FittedBlocks.from_state(chosen["transforms"])
                values = FusedRows(
                    cache["cnn"],
                    auxiliary_block(cache, variant),
                    blocks,
                    chosen["alpha"],
                )
                logits = head_logits(
                    head, values, self.device, self.config.head_batch_size
                )
            metrics[variant] = classification_metrics(logits, cache["targets"])
        return metrics


def load_config(path: Path, seed: int | None = None) -> PilotConfig:
    raw = load_yaml(path)
    unknown = set(raw) - {field.name for field in fields(PilotConfig)}
    if unknown:
        raise ValueError(f"unknown stage2 config keys: {sorted(unknown)}")
    for key in ("seeds", "alpha_grid", "variants"):
        if key in raw:
            raw[key] = tuple(raw[key])
    if seed is not None:
        raw["seed"] = seed
    config = PilotConfig(**raw)
    if (
        config.projection_epochs < 1
        or config.head_epochs < 1
        or config.batch_size < 1
        or config.head_batch_size < 1
        or config.calibration_samples < 1
    ):
        raise ValueError("training/calibration budgets must be positive")
    if config.num_workers < 0 or config.dataset_backend not in {
        "auto",
        "filesystem",
        "lmdb",
    }:
        raise ValueError("invalid workers or dataset_backend")
    if not config.alpha_grid or any(
        not np.isfinite(alpha) or alpha <= 0 for alpha in config.alpha_grid
    ):
        raise ValueError("alpha_grid must contain positive finite weights")
    if (
        not config.variants
        or len(set(config.variants)) != len(config.variants)
        or not set(config.variants) <= set(VARIANTS)
    ):
        raise ValueError(f"variants must be unique members of {VARIANTS}")
    for key in ("max_train_samples", "max_validation_samples", "max_holdout_samples"):
        if getattr(config, key) is not None and getattr(config, key) < 1:
            raise ValueError(f"{key} must be positive when set")
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "project",
            "extract",
            "compare",
            "evaluate",
            "run",
            "smoke",
            "real-smoke",
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiment/stage2_ph.yaml")
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "smoke":
        from hccr.tda.smoke import synthetic_smoke

        print(json.dumps(synthetic_smoke(arguments.output_dir), indent=2))
        return 0
    config = load_config(arguments.config, arguments.seed)
    if arguments.command == "real-smoke":
        from hccr.tda.smoke import real_smoke

        print(json.dumps(real_smoke(config, arguments.output_dir), indent=2))
        return 0
    study = Stage2Study(config)
    if arguments.command in {"project", "run"}:
        study.fit_projection()
    if arguments.command in {"extract", "run"}:
        for split in arguments.splits:
            study.extract(split)
    if arguments.command in {"compare", "run"}:
        study.compare()
    if arguments.command == "evaluate":
        study.evaluate(tuple(arguments.splits))
    print(json.dumps({"output_dir": str(study.output), "command": arguments.command}))
    return 0
