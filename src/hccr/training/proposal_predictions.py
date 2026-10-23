"""Export auditable predictions from frozen-backbone calibration pilots."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from hccr.artifacts.checkpoints import file_digest
from hccr.data.dataset import HCCRDataset
from hccr.data.manifest import read_manifest, validate_training_manifest
from hccr.models.allocation import (
    directional_svd_errors,
    mapping_digest,
    write_prediction_file,
)
from hccr.models.artifacts import load_model_from_run, read_checkpoint_metadata
from hccr.preprocessing import EvalPreprocessor
from hccr.training.proposal import (
    fit_rows_digest,
    manifest_class_labels,
    state_digest,
    validate_official_source,
)
from hccr.training.proposal import (
    pilot_budget as _pilot_budget,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_calibration_predictions(
    run_dir: Path,
    manifest_path: Path,
    output: Path,
    *,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    storage_backend: str = "auto",
    lmdb_path: Path | None = None,
    fit_frequency_output: Path | None = None,
    directional_output: Path | None = None,
    directional_rank: int | None = None,
) -> dict[str, Any]:
    """Run only the calibration split; derive provenance from the saved artifacts."""
    run_dir, manifest_path, output = Path(run_dir), Path(manifest_path), Path(output)
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    targets = [
        output,
        *[p for p in (fit_frequency_output, directional_output) if p is not None],
    ]
    protected = [
        manifest_path,
        run_dir / "checkpoint.pt",
        run_dir / "checkpoint_metadata.json",
        run_dir / "labels.json",
        run_dir / "initialization.json",
        manifest_path.with_name("audit_report.json"),
        manifest_path.with_name("class_mapping_full.json"),
    ]
    if len({Path(p).resolve() for p in targets}) != len(targets) or any(
        Path(p).resolve() in {q.resolve() for q in protected} for p in targets
    ):
        raise ValueError(
            "Export destinations must be distinct from each other and inputs"
        )
    input_digests = {path: file_digest(path) for path in protected if path.is_file()}
    metadata = read_checkpoint_metadata(run_dir)
    budget = _pilot_budget(metadata)
    manifest_hash = file_digest(manifest_path)
    if metadata.get("manifest_digest") != manifest_hash:
        raise ValueError("Checkpoint manifest digest differs from selected manifest")
    rows = read_manifest(manifest_path)
    validate_training_manifest(manifest_path, rows)
    validate_official_source(manifest_path, rows)
    labels = manifest_class_labels(rows)
    label_path = run_dir / "labels.json"
    if metadata.get("labels_digest") != file_digest(label_path):
        raise ValueError("Checkpoint labels digest differs from labels.json")
    saved_labels = json.loads(label_path.read_text(encoding="utf-8"))["labels"]
    if saved_labels != {str(index): label for index, label in enumerate(labels)}:
        raise ValueError("Checkpoint labels do not match the manifest class-ID order")
    model_config = metadata["model"]
    role = model_config.get("classifier_structure", "dense")
    if role not in {"dense", "low_rank"}:
        raise ValueError("Calibration exports require dense or uniform low_rank pilots")
    if model_config["num_classes"] != len(labels):
        raise ValueError("Checkpoint class count differs from manifest")
    if directional_output is not None and (role != "dense" or directional_rank is None):
        raise ValueError(
            "Directional control requires a dense pilot and directional_rank"
        )
    if directional_rank is not None and directional_output is None:
        raise ValueError("directional_rank requires directional_output")
    calibration = [row for row in rows if row["split"] == "calibration"]
    fit = [row for row in rows if row["split"] == "train"]
    fit_writers = sorted({row["writer_id"] for row in fit})
    if (
        not calibration
        or not fit
        or any(
            not row["writer_id"].strip() or row.get("official_partition") != "train"
            for row in [*fit, *calibration]
        )
    ):
        raise ValueError(
            "Fit and calibration require known writers and official train rows"
        )
    if set(fit_writers) & {row["writer_id"] for row in calibration}:
        raise ValueError("Calibration writers overlap fit writers")
    checkpoint_hash = file_digest(run_dir / "checkpoint.pt")
    model = load_model_from_run(run_dir, device=device).eval()
    if file_digest(run_dir / "checkpoint.pt") != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading")
    backbone_hash = state_digest(model.state_dict(), exclude_classifier=True)
    initialization_path = run_dir / "initialization.json"
    if not initialization_path.is_file():
        raise ValueError("Frozen pilot requires initialization.json provenance")
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    if initialization.get("backbone_digest") != backbone_hash:
        raise ValueError("Frozen pilot backbone differs from its initialization")
    directional_values = None
    if directional_output is not None:
        directional_values = directional_svd_errors(
            model.classifier.weight, directional_rank
        )
    prediction_metadata = {
        "type": "metadata",
        "schema_version": 1,
        "split": "calibration",
        "official_partition": "train",
        "class_labels": labels,
        "class_mapping_digest": mapping_digest(labels),
        "manifest_digest": manifest_hash,
        "fit_manifest_digest": fit_rows_digest(rows),
        "fit_writer_ids": fit_writers,
        "backbone_digest": backbone_hash,
        "head_fit_budget": budget,
        "pilot_seed": metadata["training"]["seed"],
        "model_role": role,
        "classifier_rank": int(model.classifier.basis.shape[0])
        if role == "low_rank"
        else None,
        "checkpoint_digest": checkpoint_hash,
    }
    preprocess = metadata["preprocess"]
    dataset = HCCRDataset(
        manifest_path,
        "calibration",
        EvalPreprocessor(int(preprocess["image_size"]), int(preprocess["margin"])),
        storage_backend=storage_backend,
        lmdb_path=lmdb_path,
        metadata_mode="index",
    )
    predictions = []
    try:
        image_inputs = {
            (dataset.root / source).resolve()
            for source in {row["source_file"] for row in rows}
        }
        image_inputs.add(dataset.lmdb_path.resolve())
        if any(Path(path).resolve() in image_inputs for path in targets):
            raise ValueError("Export destinations must not overwrite image inputs")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        with torch.inference_mode():
            for images, targets, indices in loader:
                logits = model(images.to(device))
                if (
                    logits.shape != (len(indices), len(labels))
                    or not torch.isfinite(logits).all()
                ):
                    raise ValueError("Pilot returned invalid target-free logits")
                predicted = logits.argmax(dim=1).cpu().tolist()
                for index, target, prediction in zip(
                    indices.tolist(), targets.tolist(), predicted, strict=True
                ):
                    row = dataset.rows[index]
                    if int(row["class_id"]) != target:
                        raise ValueError("Dataset index and target alignment failed")
                    predictions.append(
                        {
                            "sample_id": row["sample_id"],
                            "writer_id": row["writer_id"],
                            "class_id": target,
                            "prediction": prediction,
                        }
                    )
    finally:
        dataset.close()
    for path, digest in input_digests.items():
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"Input changed while exporting: {path.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
        write_prediction_file(temporary, prediction_metadata, predictions)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    control = {
        "class_mapping_digest": mapping_digest(labels),
        "provenance": {
            "checkpoint_digest": checkpoint_hash,
            "pilot_seed": metadata["training"]["seed"],
        },
    }
    if fit_frequency_output is not None:
        counts = Counter(int(row["class_id"]) for row in fit)
        _atomic_json(
            Path(fit_frequency_output),
            {
                **control,
                "split": "fit",
                "official_partition": "train",
                "manifest_digest": fit_rows_digest(rows),
                "values": [counts[c] for c in range(len(labels))],
            },
        )
    if directional_output is not None:
        _atomic_json(
            Path(directional_output),
            {
                **control,
                "rank": directional_rank,
                "values": directional_values,
                "definition": "one_minus_cosine_after_row_normalized_dense_SVD",
            },
        )
    return prediction_metadata
