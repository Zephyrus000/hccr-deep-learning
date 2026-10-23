"""Provenance and initialization boundaries for the official classifier study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from hccr.artifacts.checkpoints import file_digest
from hccr.models.allocation import _validate_metadata, mapping_digest
from hccr.models.artifacts import load_model_from_run, read_checkpoint_metadata


def fit_rows_digest(rows: list[dict[str, str]]) -> str:
    """Hash every fit-row field, independent of CSV row and column order."""
    fit = sorted(
        (row for row in rows if row["split"] == "train"), key=lambda r: r["sample_id"]
    )
    payload = json.dumps(fit, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pilot_budget(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate a completed frozen pilot and describe its paired fitting budget."""
    training, selection = metadata.get("training", {}), metadata.get("selection", {})
    if not isinstance(training, dict) or not isinstance(selection, dict):
        raise ValueError("Pilot training and selection metadata must be objects")
    if (
        training.get("freeze_backbone") is not True
        or selection.get("policy") != "final"
    ):
        raise ValueError("Calibration requires a frozen-backbone final-epoch pilot")
    if training.get("max_train_samples") is not None or training.get(
        "overfit_check", False
    ):
        raise ValueError("Calibration pilots must fit the full declared fit partition")
    for field in (
        "seed",
        "optimizer",
        "weight_decay",
        "completed_optimizer_steps",
        "total_optimizer_steps",
        "resolved_epochs",
    ):
        if field not in training:
            raise ValueError(f"Pilot metadata missing training.{field}")
    if type(training["seed"]) is not int:
        raise ValueError("Pilot seed must be an integer")
    if (
        type(training["resolved_epochs"]) is not int
        or training["resolved_epochs"] <= 0
        or not isinstance(training["optimizer"], str)
        or not training["optimizer"]
        or not isinstance(training["weight_decay"], (int, float))
        or not math.isfinite(training["weight_decay"])
        or training["weight_decay"] < 0
    ):
        raise ValueError("Pilot optimizer and epoch budget are invalid")
    steps = training["completed_optimizer_steps"]
    if (
        type(steps) is not int
        or steps <= 0
        or type(training["total_optimizer_steps"]) is not int
        or steps != training["total_optimizer_steps"]
    ):
        raise ValueError("Pilot did not complete its declared optimizer budget")
    if selection.get("epoch") != training["resolved_epochs"]:
        raise ValueError("Pilot checkpoint is not the final declared epoch")
    objective, preprocess = metadata.get("model", {}), metadata.get("preprocess", {})
    if not isinstance(objective, dict) or objective.get("classification_head") not in {
        "normalized_softmax",
        "cosface",
        "arcface",
    }:
        raise ValueError("Calibration pilots require a normalized classifier objective")
    scale, margin = objective.get("logit_scale"), objective.get("angular_margin")
    if (
        type(scale) not in (int, float)
        or not math.isfinite(scale)
        or scale <= 0
        or type(margin) not in (int, float)
        or not math.isfinite(margin)
        or margin < 0
    ):
        raise ValueError("Pilot objective scale and margin are invalid")
    epsilon = objective.get("normalization_eps")
    if epsilon is not None and (
        type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0
    ):
        raise ValueError("Pilot objective normalization epsilon is invalid")
    if not isinstance(preprocess, dict):
        raise ValueError("Pilot preprocessing metadata must be an object")
    size, border = preprocess.get("image_size"), preprocess.get("margin")
    if type(size) is not int or type(border) is not int or not 0 <= 2 * border < size:
        raise ValueError("Pilot preprocessing size and margin are invalid")
    budget = {
        key: value
        for key, value in training.items()
        if key not in {"initialization_run", "allocation_digest"}
    }
    budget["checkpoint_selection"] = {
        key: selection[key] for key in ("epoch", "policy")
    }
    budget["objective"] = {
        key: metadata["model"].get(key)
        for key in (
            "classification_head",
            "logit_scale",
            "angular_margin",
            "normalization_eps",
        )
    }
    budget["preprocess"] = metadata["preprocess"]
    return budget


def classifier_options(config: Any) -> dict[str, Any]:
    """Constructor options shared by training and checkpoint reconstruction."""
    return {
        name: getattr(config, name)
        for name in (
            "classifier_structure",
            "classifier_rank",
            "residual_class_indices",
            "gate_kind",
            "gate_use_gap",
            "gate_detach_gap",
            "normalization_eps",
        )
    }


def manifest_class_labels(rows: list[dict[str, str]]) -> list[str]:
    labels: dict[int, str] = {}
    for row in rows:
        index, label = int(row["class_id"]), row["unicode_label"]
        if index in labels and labels[index] != label:
            raise ValueError("one class_id has multiple Unicode labels")
        labels[index] = label
    if sorted(labels) != list(range(len(labels))) or len(set(labels.values())) != len(
        labels
    ):
        raise ValueError("manifest must have a unique contiguous class mapping")
    return [labels[index] for index in range(len(labels))]


def state_digest(
    state: dict[str, torch.Tensor], *, exclude_classifier: bool = False
) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if exclude_classifier and name.startswith("classifier."):
            continue
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_official_source(path: Path, rows: list[dict[str, str]]) -> None:
    from hccr.data.casia_manifest import OFFICIAL_PROTOCOLS
    from hccr.data.manifest import validate_training_manifest

    report_path = path.with_name("audit_report.json")
    if not report_path.is_file():
        raise ValueError("official proposal runs require an audit_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("source_kind") != "gnt" or report.get(
        "manifest_digest"
    ) != file_digest(path):
        raise ValueError("official proposal runs require the audited GNT manifest")
    if (
        report.get("protocol") not in OFFICIAL_PROTOCOLS
        or report.get("schema_version") != 4
    ):
        raise ValueError(
            "official proposal runs require a version-4 official protocol audit"
        )
    provenance = validate_training_manifest(path, rows)
    if provenance.get("hwdb_writer_disjoint_verified") is not True:
        raise ValueError("official HWDB writer separation has not been verified")
    for row in rows:
        if not row.get("writer_id") or row.get("source_version") not in {
            "HWDB1.0",
            "HWDB1.1",
            "HWDB1.2",
            "ICDAR2013",
        }:
            raise ValueError(
                "official source rows require writer and source-version provenance"
            )
        if not row["source_file"].lower().endswith(".gnt"):
            raise ValueError("official proposal runs accept GNT source records only")
        expected = (
            "train"
            if row["split"] in {"train", "calibration", "validation"}
            else row["split"]
        )
        if row.get("official_partition") != expected:
            raise ValueError(
                "internal split disagrees with the official source partition"
            )
    labels = manifest_class_labels(rows)
    for split in ("train", "calibration"):
        support = {int(row["class_id"]) for row in rows if row["split"] == split}
        if support != set(range(len(labels))):
            raise ValueError(f"official {split} split must cover every class")


def resolve_residual_mask(
    path: Path, manifest_path: Path, rows: list[dict[str, str]]
) -> dict[str, Any]:
    mask = json.loads(path.read_text(encoding="utf-8"))
    labels = manifest_class_labels(rows)
    if mask.get("class_labels") != labels or mask.get(
        "class_mapping_digest"
    ) != mapping_digest(labels):
        raise ValueError("residual mask class mapping differs from training manifest")
    if mask.get("manifest_digest") != file_digest(manifest_path):
        raise ValueError("residual mask was selected from a different manifest")
    indices = mask.get("residual_class_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or not 0 <= i < len(labels) for i in indices)
    ):
        raise ValueError("residual mask must contain valid class indices")
    if len(set(indices)) != len(indices):
        raise ValueError("residual mask has duplicate class indices")
    # The selector persists the validated pilot metadata; do not accept bare masks.
    provenance = mask.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("residual mask provenance must be an object")
    pilots = {
        name: provenance[name] for name in ("dense", "low_rank") if name in provenance
    }
    if len(pilots) != 2:
        raise ValueError("residual mask is missing calibration pilot provenance")
    expected = {
        "class_labels": labels,
        "class_mapping_digest": mapping_digest(labels),
        "manifest_digest": file_digest(manifest_path),
        "fit_manifest_digest": fit_rows_digest(rows),
    }
    if mask.get("fit_manifest_digest") != expected["fit_manifest_digest"]:
        raise ValueError("residual mask fit rows differ from the training manifest")
    fit_writers = {row["writer_id"] for row in rows if row["split"] == "train"}
    for role, metadata in pilots.items():
        if not isinstance(metadata, dict):
            raise ValueError("residual pilot provenance must contain metadata objects")
        _validate_metadata(metadata)
        if metadata["model_role"] != role:
            raise ValueError("residual pilot provenance has incorrect model roles")
        for field, value in expected.items():
            if metadata[field] != value:
                raise ValueError(f"residual pilot {field} differs from the manifest")
        for field in ("backbone_digest", "checkpoint_digest"):
            _require_sha256(metadata[field], field)
        if set(metadata.get("fit_writer_ids", ())) != fit_writers:
            raise ValueError("residual pilots used different fit writers")
        budget = metadata["head_fit_budget"]
        if any(
            not isinstance(budget.get(field), dict)
            for field in ("checkpoint_selection", "objective", "preprocess")
        ):
            raise ValueError("residual pilot head_fit_budget is incomplete")
        pilot_budget(
            {
                "training": budget,
                "selection": budget["checkpoint_selection"],
                "model": budget["objective"],
                "preprocess": budget["preprocess"],
            }
        )
        if metadata["pilot_seed"] != budget["seed"]:
            raise ValueError("residual pilot seed differs from its fitting budget")
    for field in ("backbone_digest", "head_fit_budget", "pilot_seed"):
        if pilots["dense"][field] != pilots["low_rank"][field]:
            raise ValueError(f"residual pilot metadata mismatch: {field}")
    if (
        type(pilots["low_rank"].get("classifier_rank")) is not int
        or pilots["low_rank"]["classifier_rank"] <= 0
    ):
        raise ValueError("residual pilot requires a positive low-rank classifier rank")
    prediction_digests = provenance.get("prediction_file_digests")
    if not isinstance(prediction_digests, dict):
        raise ValueError("residual mask is missing prediction-file digests")
    for role in pilots:
        _require_sha256(prediction_digests.get(role), f"{role} prediction file digest")
    calibration_writers = {
        row["writer_id"] for row in rows if row["split"] == "calibration"
    }
    if (
        not calibration_writers
        or not fit_writers
        or any(not writer.strip() for writer in calibration_writers | fit_writers)
        or calibration_writers & fit_writers
    ):
        raise ValueError("residual mask requires disjoint fit and calibration writers")
    if set(provenance.get("calibration_writer_ids", ())) != calibration_writers:
        raise ValueError("residual mask calibration writers differ from the manifest")
    return mask


def _require_sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a SHA-256 digest")


def validate_allocation_initialization(
    allocation: dict[str, Any], run_dir: Path, initialization: dict[str, Any]
) -> None:
    """Bind a resolved mask to the pilot checkpoint actually used to initialize."""
    pilot = allocation["provenance"]["low_rank"]
    if (
        file_digest(Path(run_dir) / "checkpoint.pt") != pilot["checkpoint_digest"]
        or initialization.get("checkpoint_digest") != pilot["checkpoint_digest"]
    ):
        raise ValueError("allocation initialization checkpoint differs from its pilot")
    if initialization.get("backbone_digest") != pilot["backbone_digest"]:
        raise ValueError("allocation initialization backbone differs from its pilot")
    metadata = read_checkpoint_metadata(Path(run_dir))
    if (
        metadata["model"].get("classifier_structure", "dense") != "low_rank"
        or metadata["model"].get("classifier_rank") != pilot["classifier_rank"]
    ):
        raise ValueError(
            "allocation initialization requires the recorded low-rank head"
        )
    if metadata.get("manifest_digest") != pilot["manifest_digest"]:
        raise ValueError("allocation initialization manifest differs from its pilot")
    if metadata.get("training", {}).get("seed") != pilot["pilot_seed"]:
        raise ValueError("allocation initialization seed differs from its pilot")
    if pilot_budget(metadata) != pilot["head_fit_budget"]:
        raise ValueError(
            "allocation initialization fitting budget differs from its pilot"
        )


def initialize_training_model(
    model: torch.nn.Module,
    run_dir: Path,
    manifest_path: Path,
    class_labels: list[str],
) -> dict[str, Any]:
    """Load a matching pilot backbone and transfer or factorize its head."""
    metadata = read_checkpoint_metadata(run_dir)
    if metadata.get("manifest_digest") != file_digest(manifest_path):
        raise ValueError("initialization checkpoint uses a different manifest")
    if metadata.get("labels_digest") != file_digest(run_dir / "labels.json"):
        raise ValueError("initialization labels changed after checkpoint creation")
    label_payload = json.loads((run_dir / "labels.json").read_text(encoding="utf-8"))[
        "labels"
    ]
    stored_labels = [label_payload[str(i)] for i in range(len(label_payload))]
    if stored_labels != class_labels:
        raise ValueError("initialization checkpoint uses a different class mapping")
    source = load_model_from_run(run_dir)
    source_state = source.state_dict()
    target_state = model.state_dict()
    source_backbone = {
        k: v for k, v in source_state.items() if not k.startswith("classifier.")
    }
    target_backbone = {
        k: v for k, v in target_state.items() if not k.startswith("classifier.")
    }
    if source_backbone.keys() != target_backbone.keys() or any(
        source_backbone[k].shape != v.shape for k, v in target_backbone.items()
    ):
        raise ValueError("initialization must preserve the exact backbone architecture")
    model.load_state_dict(source_backbone, strict=False)
    source_head, target_head = source.classifier, model.classifier
    source_structure = getattr(
        source,
        "classifier_structure",
        metadata["model"].get("classifier_structure", "dense"),
    )
    target_structure = getattr(model, "classifier_structure", "dense")
    if source_structure == target_structure:
        target_head.load_state_dict(source_head.state_dict())
        mode = "matching_checkpoint"
    elif source_structure == "dense" and hasattr(target_head, "initialize_from_dense"):
        if getattr(source, "classification_head", None) == "softmax":
            raise ValueError(
                "dense SVD initialization requires a normalized dense teacher"
            )
        target_head.initialize_from_dense(source_head.weight)
        mode = "normalized_dense_svd"
    elif source_structure == "low_rank" and target_structure in {
        "static_residual",
        "dynamic_residual",
    }:
        for name in ("coefficients", "basis"):
            source_value, target_value = (
                getattr(source_head, name),
                getattr(target_head, name),
            )
            if source_value.shape != target_value.shape:
                raise ValueError("residual head rank must match the low-rank pilot")
            with torch.no_grad():
                target_value.copy_(source_value)
        mode = "shared_low_rank_pilot"
    else:
        raise ValueError(
            f"unsupported head initialization: {source_structure} -> {target_structure}"
        )
    return {
        "source_run": str(run_dir.resolve()),
        "checkpoint_digest": file_digest(run_dir / "checkpoint.pt"),
        "backbone_digest": state_digest(source_backbone),
        "mode": mode,
        "manifest_digest": file_digest(manifest_path),
        "class_mapping_digest": mapping_digest(class_labels),
    }
