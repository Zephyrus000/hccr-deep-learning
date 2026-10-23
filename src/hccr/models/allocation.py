"""Select residual classes from paired, writer-held-out pilot predictions."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def mapping_digest(class_labels: Sequence[str]) -> str:
    """Hash class labels in class-ID order using the prediction-file convention."""
    payload = json.dumps(list(class_labels), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("schema_version") != 1 or metadata.get("type") != "metadata":
        raise ValueError(
            "Prediction metadata must use schema_version=1 and type=metadata"
        )
    if metadata.get("split") != "calibration":
        raise ValueError("Only calibration predictions may select residual classes")
    if metadata.get("official_partition") != "train":
        raise ValueError("Calibration must belong to the official train partition")
    labels = metadata.get("class_labels")
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or len(label) != 1 for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("class_labels must contain unique single Unicode characters")
    if metadata.get("class_mapping_digest") != mapping_digest(labels):
        raise ValueError("class_mapping_digest does not match class_labels")
    for field in (
        "manifest_digest",
        "fit_manifest_digest",
        "backbone_digest",
        "checkpoint_digest",
    ):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError(f"Missing {field}")
    writers = metadata.get("fit_writer_ids")
    if (
        not isinstance(writers, list)
        or not writers
        or any(not isinstance(writer, str) or not writer.strip() for writer in writers)
        or len(set(writers)) != len(writers)
    ):
        raise ValueError("fit_writer_ids must be a nonempty unique list")
    if type(metadata.get("pilot_seed")) is not int:
        raise ValueError("pilot_seed must be an integer")
    if (
        not isinstance(metadata.get("head_fit_budget"), dict)
        or not metadata["head_fit_budget"]
    ):
        raise ValueError("head_fit_budget must describe the paired fitting protocol")
    if metadata.get("model_role") not in ("dense", "low_rank"):
        raise ValueError("model_role must be dense or low_rank")
    rank = metadata.get("classifier_rank")
    if (
        metadata["model_role"] == "low_rank"
        and "classifier_rank" in metadata
        and (type(rank) is not int or rank <= 0)
    ):
        raise ValueError("low_rank classifier_rank must be a positive integer")


def _validated_rows(
    rows: Iterable[Mapping[str, Any]], metadata: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    fit_writers = set(metadata["fit_writer_ids"])
    classes = len(metadata["class_labels"])
    for row in rows:
        sample_id, writer = row.get("sample_id"), row.get("writer_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("Each prediction requires a nonempty sample_id")
        if sample_id in result:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        if not isinstance(writer, str) or not writer.strip():
            raise ValueError(f"Missing writer_id for {sample_id}")
        if writer in fit_writers:
            raise ValueError(f"Calibration writer overlaps fit writers: {writer}")
        for field in ("class_id", "prediction"):
            if type(row.get(field)) is not int or not 0 <= row[field] < classes:
                raise ValueError(f"Invalid {field} for {sample_id}")
        if "split" in row and row["split"] != "calibration":
            raise ValueError(f"Non-calibration row: {sample_id}")
        if "official_partition" in row and row["official_partition"] != "train":
            raise ValueError(f"Non-training row: {sample_id}")
        result[sample_id] = dict(row)
    if not result:
        raise ValueError("Prediction file has no samples")
    missing = set(range(classes)) - {row["class_id"] for row in result.values()}
    if missing:
        raise ValueError(f"Calibration lacks class support: {sorted(missing)}")
    return result


def write_prediction_file(
    path: str | Path, metadata: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]
) -> None:
    """Write validated predictions with metadata as the first JSONL line."""
    _validate_metadata(metadata)
    checked = _validated_rows(rows, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(dict(metadata), ensure_ascii=False, allow_nan=False) + "\n"
        )
        for sample_id in sorted(checked):
            stream.write(
                json.dumps(checked[sample_id], ensure_ascii=False, allow_nan=False)
                + "\n"
            )


@dataclass(frozen=True)
class PairedPredictions:
    dense_metadata: dict[str, Any]
    low_rank_metadata: dict[str, Any]
    dense: dict[str, dict[str, Any]]
    low_rank: dict[str, dict[str, Any]]
    source_digests: dict[str, str]


def _read_predictions(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    contents = Path(path).read_bytes()
    lines = contents.decode("utf-8-sig").splitlines()
    if not lines:
        raise ValueError("Empty prediction file")
    metadata = json.loads(lines[0])
    _validate_metadata(metadata)
    rows = _validated_rows(
        (json.loads(line) for line in lines[1:] if line.strip()), metadata
    )
    return metadata, rows, hashlib.sha256(contents).hexdigest()


def load_paired_predictions(
    dense_path: str | Path, low_rank_path: str | Path
) -> PairedPredictions:
    dense_meta, dense, dense_digest = _read_predictions(dense_path)
    low_meta, low, low_digest = _read_predictions(low_rank_path)
    if dense_meta["model_role"] != "dense" or low_meta["model_role"] != "low_rank":
        raise ValueError("Expected dense and low_rank prediction files in that order")
    for field in (
        "class_labels",
        "class_mapping_digest",
        "manifest_digest",
        "fit_manifest_digest",
        "backbone_digest",
        "head_fit_budget",
        "pilot_seed",
    ):
        if dense_meta[field] != low_meta[field]:
            raise ValueError(f"Paired prediction metadata mismatch: {field}")
    if set(dense_meta["fit_writer_ids"]) != set(low_meta["fit_writer_ids"]):
        raise ValueError("Paired prediction metadata mismatch: fit_writer_ids")
    if dense.keys() != low.keys():
        raise ValueError("Paired prediction sample IDs differ")
    for sample_id in dense:
        for field in ("writer_id", "class_id"):
            if dense[sample_id][field] != low[sample_id][field]:
                raise ValueError(f"Paired {field} mismatch for {sample_id}")
    return PairedPredictions(
        dense_meta,
        low_meta,
        dense,
        low,
        {"dense": dense_digest, "low_rank": low_digest},
    )


def directional_svd_errors(weights: Any, rank: int) -> list[float]:
    """Return 1-cosine errors after SVD of row-normalized dense class weights."""
    import torch

    weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    if type(rank) is not int or weights.ndim != 2 or not 0 < rank <= min(weights.shape):
        raise ValueError("rank must fit a two-dimensional dense weight matrix")
    norms = weights.norm(dim=1, keepdim=True)
    if not torch.isfinite(weights).all() or (norms <= 0).any():
        raise ValueError("Dense class weights must be finite and nonzero")
    normalized = weights / norms
    u, singular, vh = torch.linalg.svd(normalized, full_matrices=False)
    reconstructed = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    reconstructed_norms = reconstructed.norm(dim=1, keepdim=True)
    directions = reconstructed / reconstructed_norms.clamp_min(
        torch.finfo(weights.dtype).eps
    )
    return (1 - (normalized * directions).sum(dim=1).clamp(-1, 1)).tolist()


def select_residual_classes(
    paired: PairedPredictions,
    k: int,
    *,
    method: str = "penalty",
    seed: int = 7,
    eta: float = 0.0,
    frequencies: Sequence[float] | None = None,
    directional_errors: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Select exactly k classes; ordered ties use Unicode, never numeric class ID."""
    labels = paired.dense_metadata["class_labels"]
    classes = len(labels)
    if type(k) is not int or not 0 < k <= classes:
        raise ValueError("k must be between 1 and the class count")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    if method not in (
        "penalty",
        "pair",
        "pair_only",
        "random",
        "frequency",
        "raw_difficulty",
        "directional_svd",
    ):
        raise ValueError(f"Unknown allocation method: {method}")
    support, harmed, helped, errors = (Counter() for _ in range(4))
    dense_confusion, low_confusion = Counter(), Counter()
    for sample_id, row in paired.dense.items():
        target, dense_prediction = row["class_id"], row["prediction"]
        low_prediction = paired.low_rank[sample_id]["prediction"]
        support[target] += 1
        harmed[target] += int(dense_prediction == target and low_prediction != target)
        helped[target] += int(dense_prediction != target and low_prediction == target)
        errors[target] += int(low_prediction != target)
        if dense_prediction != target:
            dense_confusion[target, dense_prediction] += 1
        if low_prediction != target:
            low_confusion[target, low_prediction] += 1
    signed = [(harmed[c] - helped[c]) / (support[c] + 2) for c in range(classes)]
    penalty = [max(value, 0.0) for value in signed]
    extra = {
        (c, j): max((count - dense_confusion[c, j]) / (support[c] + 2), 0.0)
        for (c, j), count in low_confusion.items()
    }
    pair_scores = [0.0] * classes
    for c, j in extra:
        value = (extra[c, j] + extra.get((j, c), 0.0)) / 2
        pair_scores[c] = max(pair_scores[c], value)
        pair_scores[j] = max(pair_scores[j], value)
    scores = list(penalty)
    if method == "pair":
        scores = [d + eta * v for d, v in zip(penalty, pair_scores, strict=True)]
    elif method == "pair_only":
        scores = pair_scores
    elif method == "raw_difficulty":
        scores = [errors[c] / (support[c] + 2) for c in range(classes)]
    elif method in ("frequency", "directional_svd"):
        values = frequencies if method == "frequency" else directional_errors
        if values is None or len(values) != classes:
            raise ValueError(f"{method} requires one value per class in class-ID order")
        scores = [float(value) for value in values]
        if any(not math.isfinite(value) or value < 0 for value in scores):
            raise ValueError(f"{method} values must be finite and nonnegative")
    order = sorted(range(classes), key=lambda c: (-scores[c], ord(labels[c])))
    if method == "random":
        order = sorted(range(classes), key=lambda c: ord(labels[c]))
        random.Random(seed).shuffle(order)
    selected = order[:k]
    return {
        "schema_version": 1,
        "method": method,
        "k": k,
        "seed": seed,
        "eta": eta,
        "residual_class_indices": sorted(selected),
        "selection_order": selected,
        "class_labels": labels,
        "class_mapping_digest": paired.dense_metadata["class_mapping_digest"],
        "manifest_digest": paired.dense_metadata["manifest_digest"],
        "fit_manifest_digest": paired.dense_metadata["fit_manifest_digest"],
        "split": "calibration",
        "official_partition": "train",
        "zero_penalty_selected": [c for c in selected if penalty[c] == 0],
        "provenance": {
            "dense": paired.dense_metadata,
            "low_rank": paired.low_rank_metadata,
            "prediction_file_digests": paired.source_digests,
            "calibration_writer_ids": sorted(
                {row["writer_id"] for row in paired.dense.values()}
            ),
        },
        "class_statistics": [
            {
                "class_id": c,
                "label": labels[c],
                "support": support[c],
                "harmed": harmed[c],
                "helped": helped[c],
                "signed_delta": signed[c],
                "penalty": penalty[c],
                "pair_score": pair_scores[c],
                "low_rank_errors": errors[c],
                "selection_score": scores[c],
            }
            for c in range(classes)
        ],
    }
