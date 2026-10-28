"""Validate a fixed compression-sensitive class set on independent writers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hccr.artifacts.checkpoints import file_digest
from hccr.data.dataset import HCCRDataset
from hccr.data.manifest import read_manifest, validate_training_manifest
from hccr.models.allocation import mapping_digest
from hccr.models.artifacts import load_model_from_run, read_checkpoint_metadata
from hccr.preprocessing import EvalPreprocessor
from hccr.training.proposal import (
    manifest_class_labels,
    pilot_budget,
    resolve_residual_mask,
    state_digest,
    validate_official_source,
)


def _rates(counts: np.ndarray) -> dict[str, Any]:
    support, harm, help_count, dense_correct, low_correct = map(int, counts)
    net = harm - help_count
    return {
        "support": support,
        "harm": harm,
        "help": help_count,
        "signed_net_damage_count": net,
        "signed_net_damage_rate": net / support if support else None,
        "dense_accuracy": dense_correct / support if support else None,
        "low_rank_accuracy": low_correct / support if support else None,
    }


def summarize_compression_sensitivity(
    predictions: Iterable[Mapping[str, Any]],
    selected_classes: Sequence[int],
    class_labels: Sequence[str],
    *,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20261024,
) -> dict[str, Any]:
    """Aggregate correctness pairs, without selecting or modifying any classes.

    Each record contains class_id, dense_correct, low_rank_correct, writer_id,
    and source_version. Rates are fractions, not percentage points. Bootstrap
    whole writers within their source-version signatures, retaining all paired
    outcomes and their sample support when a writer is drawn more than once.
    """
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 0:
        raise ValueError("bootstrap_repetitions must be a nonnegative integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    labels = list(class_labels)
    selected = list(selected_classes)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("class_labels must be nonempty and unique")
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(c) is not int or not 0 <= c < len(labels) for c in selected)
    ):
        raise ValueError("selected_classes must be unique valid class IDs")
    chosen = np.zeros(len(labels), dtype=bool)
    chosen[selected] = True
    by_writer: dict[str, np.ndarray] = {}
    versions: dict[str, set[str]] = {}
    for row in predictions:
        c = row["class_id"]
        writer, version = row["writer_id"], row["source_version"]
        dense, low = row["dense_correct"], row["low_rank_correct"]
        if type(c) is not int or not 0 <= c < len(labels):
            raise ValueError("prediction class_id is outside the class mapping")
        if type(dense) is not bool or type(low) is not bool:
            raise ValueError("paired correctness values must be booleans")
        if (
            not isinstance(writer, str)
            or not writer.strip()
            or not isinstance(version, str)
            or not version.strip()
        ):
            raise ValueError("writer and source-version identifiers are required")
        if writer not in by_writer:
            by_writer[writer] = np.zeros((len(labels), 5), dtype=np.int64)
            versions[writer] = set()
        versions[writer].add(version)
        by_writer[writer][c] += (1, dense and not low, low and not dense, dense, low)
    if not by_writer:
        raise ValueError("validation predictions must be nonempty")
    writers = sorted(by_writer)
    counts = np.stack([by_writer[w] for w in writers])
    per_class = counts.sum(axis=0)
    groups = {
        "selected": _rates(per_class[chosen].sum(axis=0)),
        "outside": _rates(per_class[~chosen].sum(axis=0)),
        "overall": _rates(per_class.sum(axis=0)),
    }
    inside, outside = (
        groups[name]["signed_net_damage_rate"] for name in ("selected", "outside")
    )
    gap = inside - outside if inside is not None and outside is not None else None
    signed = per_class[:, 1] - per_class[:, 2]
    positive = np.maximum(signed, 0)
    signed_total, positive_total = int(signed.sum()), int(positive.sum())
    strata: dict[tuple[str, ...], list[int]] = {}
    for index, writer in enumerate(writers):
        strata.setdefault(tuple(sorted(versions[writer])), []).append(index)
    intervals = None
    if bootstrap_repetitions:
        # Only group support/harm/help are needed for the prespecified gap CIs.
        grouped = np.stack(
            [counts[:, chosen, :3].sum(axis=1), counts[:, ~chosen, :3].sum(axis=1)],
            axis=1,
        )
        rng = np.random.default_rng(bootstrap_seed)
        draws = np.zeros((bootstrap_repetitions, 2, 3), dtype=np.int64)
        for indices in strata.values():
            sampled = rng.choice(indices, size=(bootstrap_repetitions, len(indices)))
            draws += grouped[sampled].sum(axis=1)
        rates = np.divide(
            draws[:, :, 1] - draws[:, :, 2],
            draws[:, :, 0],
            out=np.full((bootstrap_repetitions, 2), np.nan),
            where=draws[:, :, 0] > 0,
        )
        values = {
            "selected_signed_net_damage_rate": rates[:, 0],
            "outside_signed_net_damage_rate": rates[:, 1],
            "signed_net_damage_gap": rates[:, 0] - rates[:, 1],
        }
        intervals = {}
        for name, array in values.items():
            finite = array[np.isfinite(array)]
            intervals[name] = {
                "lower": float(np.quantile(finite, 0.025)) if finite.size else None,
                "upper": float(np.quantile(finite, 0.975)) if finite.size else None,
                "valid_repetitions": int(finite.size),
            }
    return {
        "groups": groups,
        "signed_net_damage_gap": gap,
        "per_class": [
            {
                "class_id": c,
                "unicode_label": label,
                "selected": bool(chosen[c]),
                **_rates(per_class[c]),
                "positive_net_damage_count": int(positive[c]),
            }
            for c, label in enumerate(labels)
        ],
        "concentration": {
            "signed_selected_contribution_ratio": float(signed[chosen].sum())
            / signed_total
            if signed_total > 0
            else None,
            "signed_ratio_definition": (
                "selected sum(harm-help) / overall sum(harm-help); defined only "
                "for positive overall net damage; may be outside [0,1]"
            ),
            "positive_selected_share": float(positive[chosen].sum()) / positive_total
            if positive_total > 0
            else None,
            "positive_share_definition": "sum_selected max(class harm-help,0) / "
            "sum_all max(class harm-help,0); undefined when denominator is zero",
            "positive_net_damage_count": positive_total,
            "descriptive_only": True,
        },
        "bootstrap": {
            "method": "source_version_signature_stratified_writer_cluster_percentile",
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
            "confidence_level": 0.95,
            "intervals": intervals,
            "writer_count": len(writers),
            "strata": [
                {"source_versions": list(signature), "writers": len(indices)}
                for signature, indices in sorted(strata.items())
            ],
            "singleton_strata": sum(len(indices) == 1 for indices in strata.values()),
            "uncertainty_scope": (
                "conditional on this fixed trained pilot pair and mask; "
                "does not estimate training-seed or mask-selection uncertainty"
            ),
        },
        "definitions": {
            "harm": "dense correct and low rank incorrect",
            "help": "dense incorrect and low rank correct",
            "signed_net_damage_rate": "(harm-help)/support; support weighted",
            "signed_net_damage_gap": "selected rate minus outside rate",
        },
    }


def analyze_compression_sensitivity(
    dense_run: Path,
    low_rank_run: Path,
    mask: Path,
    manifest: Path,
    output: Path,
    *,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    storage_backend: str = "lmdb",
    lmdb_path: Path | None = None,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20261024,
) -> dict[str, Any]:
    """Evaluate validation images once for both frozen pilots and a fixed mask."""
    dense_run, low_rank_run, mask, manifest, output = map(
        Path, (dense_run, low_rank_run, mask, manifest, output)
    )
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 0:
        raise ValueError("bootstrap_repetitions must be a nonnegative integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    paths = [manifest, mask]
    for run in (dense_run, low_rank_run):
        paths.extend(
            run / name
            for name in (
                "checkpoint.pt",
                "checkpoint_metadata.json",
                "labels.json",
                "initialization.json",
            )
        )
    paths.extend(
        p
        for p in (
            manifest.with_name("audit_report.json"),
            manifest.with_name("class_mapping_full.json"),
        )
        if p.is_file()
    )
    if output.resolve() in {p.resolve() for p in paths}:
        raise ValueError("output must not overwrite analysis inputs")
    inputs = {p: file_digest(p) for p in paths}
    rows = read_manifest(manifest)
    validate_training_manifest(manifest, rows)
    validate_official_source(manifest, rows)
    labels = manifest_class_labels(rows)
    allocation = resolve_residual_mask(mask, manifest, rows)
    validation = [row for row in rows if row["split"] == "validation"]
    validation_writers = {row["writer_id"] for row in validation}
    calibration_writers = set(allocation["provenance"]["calibration_writer_ids"])
    fit_writers = {row["writer_id"] for row in rows if row["split"] == "train"}
    if (
        not validation
        or any(not w.strip() for w in validation_writers)
        or validation_writers & (calibration_writers | fit_writers)
        or any(row.get("official_partition") != "train" for row in validation)
    ):
        raise ValueError(
            "validation requires independent known writers on official train"
        )
    models, budgets = {}, {}
    for role, run in (("dense", dense_run), ("low_rank", low_rank_run)):
        recorded = allocation["provenance"][role]
        metadata = read_checkpoint_metadata(run)
        budgets[role] = pilot_budget(metadata)
        if metadata.get("manifest_digest") != inputs[manifest]:
            raise ValueError("pilot manifest differs from analysis manifest")
        if inputs[run / "checkpoint.pt"] != recorded["checkpoint_digest"]:
            raise ValueError("pilot checkpoint differs from fixed-mask pair")
        if budgets[role] != recorded["head_fit_budget"]:
            raise ValueError("pilot fitting budget differs from fixed-mask pair")
        if metadata["model"].get("classifier_structure", "dense") != role or metadata[
            "model"
        ].get("num_classes") != len(labels):
            raise ValueError("pilot classifier role or class count differs")
        if (
            role == "low_rank"
            and metadata["model"].get("classifier_rank") != recorded["classifier_rank"]
        ):
            raise ValueError("low-rank pilot rank differs from fixed-mask pair")
        saved_labels = json.loads((run / "labels.json").read_text(encoding="utf-8"))
        if metadata.get("labels_digest") != inputs[
            run / "labels.json"
        ] or saved_labels.get("labels") != {str(i): v for i, v in enumerate(labels)}:
            raise ValueError("pilot labels differ from manifest class mapping")
        model = load_model_from_run(run, device=device).eval()
        backbone = state_digest(model.state_dict(), exclude_classifier=True)
        initialization = json.loads(
            (run / "initialization.json").read_text(encoding="utf-8")
        )
        if backbone != recorded["backbone_digest"] or backbone != initialization.get(
            "backbone_digest"
        ):
            raise ValueError("actual frozen backbone differs from paired provenance")
        models[role] = model
    if budgets["dense"] != budgets["low_rank"]:
        raise ValueError("dense and low-rank pilots have different fitting budgets")
    preprocess = budgets["dense"]["preprocess"]
    dataset = HCCRDataset(
        manifest,
        "validation",
        EvalPreprocessor(preprocess["image_size"], preprocess["margin"]),
        storage_backend=storage_backend,
        lmdb_path=lmdb_path,
        metadata_mode="index",
    )
    try:
        protected_images = {
            (dataset.root / source).resolve()
            for source in {row["source_file"] for row in rows}
        }
        protected_images.add(dataset.lmdb_path.resolve())
        if output.resolve() in protected_images:
            raise ValueError("output must not overwrite image inputs")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        def predictions():
            with (
                torch.inference_mode(),
                torch.autocast(device_type=torch.device(device).type, enabled=False),
            ):
                for images, targets, indices in loader:
                    images = images.to(device)
                    correct = {}
                    for role, model in models.items():
                        logits = model(images)
                        if (
                            logits.shape != (len(indices), len(labels))
                            or not torch.isfinite(logits).all()
                        ):
                            raise ValueError(
                                "pilot returned invalid target-free logits"
                            )
                        correct[role] = logits.argmax(1).cpu().eq(targets).tolist()
                    for offset, index in enumerate(indices.tolist()):
                        row = dataset.rows[index]
                        if int(row["class_id"]) != int(targets[offset]):
                            raise ValueError("dataset target/index alignment failed")
                        yield {
                            "class_id": int(row["class_id"]),
                            "writer_id": row["writer_id"],
                            "source_version": row["source_version"],
                            "dense_correct": correct["dense"][offset],
                            "low_rank_correct": correct["low_rank"][offset],
                        }

        summary = summarize_compression_sensitivity(
            predictions(),
            allocation["residual_class_indices"],
            labels,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
        )
        resolved_storage = dataset.storage_backend
    finally:
        dataset.close()
    for path, digest in inputs.items():
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"analysis input changed during evaluation: {path.name}")
    endpoint_hash = hashlib.sha256(
        json.dumps(validation, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "analysis": "fixed_mask_independent_writer_compression_sensitivity",
        "split": "validation",
        "official_partition": "train",
        "mask_reselected": False,
        "test_images_opened": False,
        "pilot_seed": allocation["provenance"]["dense"]["pilot_seed"],
        "classifier_rank": allocation["provenance"]["low_rank"]["classifier_rank"],
        "backbone_digest": allocation["provenance"]["dense"]["backbone_digest"],
        "head_fit_budget": budgets["dense"],
        "interpretation": (
            "validation diagnostic with a fixed trained pair and calibration mask; "
            "writer separation does not undo prior validation-based model selection"
        ),
        "class_mapping_digest": mapping_digest(labels),
        "selected_class_ids": allocation["residual_class_indices"],
        "validation_writer_ids": sorted(validation_writers),
        "calibration_writer_ids": sorted(calibration_writers),
        "validation_calibration_writer_disjoint": True,
        "validation_fit_writer_disjoint": True,
        "validation_rows_digest": endpoint_hash,
        "source_version_sample_counts": dict(
            Counter(r["source_version"] for r in validation)
        ),
        "input_files": {str(p.resolve()): digest for p, digest in inputs.items()},
        "analysis_source_digest": file_digest(Path(__file__)),
        "runtime": {
            "device": device,
            "precision": "float32",
            "batch_size": batch_size,
            "num_workers": num_workers,
            "storage_backend": resolved_storage,
        },
        **summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
