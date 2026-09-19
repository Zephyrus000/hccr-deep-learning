"""Build immutable, versioned evidence views over historical run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVIDENCE_BUNDLE_SCHEMA_VERSION = 1
RUN_EVIDENCE_SCHEMA_VERSION = 1
SWEEP_EVIDENCE_SCHEMA_VERSION = 1
DERIVED_EVIDENCE_SCHEMA_VERSION = 1

_RUN_REQUIRED_FILES = (
    "config.json",
    "metadata.json",
    "checkpoint_metadata.json",
    "metrics.json",
    "resource_profile.json",
)
_RAW_SUFFIXES = {".csv", ".json", ".png", ".txt"}


def build_evidence_bundle(
    source_root: Path,
    output_root: Path,
    *,
    run_ids: Iterable[str] | None = None,
    sweep_ids: Iterable[str] | None = None,
    derived_roots: Iterable[Path] | None = None,
) -> dict[str, Any]:
    """Create a portable evidence bundle without mutating source artifacts."""
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"experiment root does not exist: {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"evidence output must be empty: {output_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("evidence output must be outside the experiment root")

    selected_runs = set(run_ids or ())
    selected_sweeps = set(sweep_ids or ())
    resolved_derived_roots = [Path(path).resolve() for path in derived_roots or ()]
    missing_derived = [path for path in resolved_derived_roots if not path.is_dir()]
    if missing_derived:
        raise FileNotFoundError(f"derived evidence roots not found: {missing_derived}")
    collection_ids = [path.name for path in resolved_derived_roots]
    if len(collection_ids) != len(set(collection_ids)):
        raise ValueError("derived evidence root names must be unique")
    run_dirs = _discover_run_dirs(source_root, selected_runs)
    sweep_dirs = _discover_sweep_dirs(source_root, selected_sweeps)
    if selected_runs - {path.name for path in run_dirs}:
        missing = sorted(selected_runs - {path.name for path in run_dirs})
        raise FileNotFoundError(f"run IDs not found or incomplete: {missing}")
    if selected_sweeps - {path.name for path in sweep_dirs}:
        missing = sorted(selected_sweeps - {path.name for path in sweep_dirs})
        raise FileNotFoundError(f"sweep IDs not found: {missing}")

    output_root.mkdir(parents=True, exist_ok=True)
    records = [
        _write_run_evidence(source_root, output_root, run_dir)
        for run_dir in run_dirs
    ]
    sweeps = [
        _write_sweep_evidence(source_root, output_root, sweep_dir)
        for sweep_dir in sweep_dirs
    ]
    derived = [
        _write_derived_evidence(output_root, derived_root)
        for derived_root in resolved_derived_roots
    ]
    standalone = _copy_standalone_evidence(source_root, output_root)
    manifest = {
        "artifact_type": "hccr.paper_evidence_bundle",
        "schema_version": EVIDENCE_BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(source_root),
        "policy": {
            "raw_artifacts": "copied_without_content_changes",
            "checkpoints": "checksum_only_not_copied",
            "missing_values": "null_with_availability_status",
            "normalization": "derived_view_preserves_source_sha256",
        },
        "runs": records,
        "sweeps": sweeps,
        "derived_collections": derived,
        "standalone_artifacts": standalone,
    }
    _write_json(output_root / "evidence_manifest.json", manifest)
    validate_evidence_bundle(output_root)
    return manifest


def validate_evidence_bundle(bundle_root: Path) -> dict[str, int]:
    """Validate bundle schemas and every copied raw-artifact checksum."""
    bundle_root = bundle_root.resolve()
    manifest = _read_json(bundle_root / "evidence_manifest.json")
    _require_envelope(
        manifest,
        artifact_type="hccr.paper_evidence_bundle",
        schema_version=EVIDENCE_BUNDLE_SCHEMA_VERSION,
    )
    checked_sources = 0
    groups = (
        ("runs", "hccr.run_evidence", RUN_EVIDENCE_SCHEMA_VERSION),
        ("sweeps", "hccr.sweep_evidence", SWEEP_EVIDENCE_SCHEMA_VERSION),
        (
            "derived_collections",
            "hccr.derived_evidence_collection",
            DERIVED_EVIDENCE_SCHEMA_VERSION,
        ),
        ("standalone_artifacts", "hccr.standalone_evidence", 1),
    )
    for group_name, artifact_type, schema_version in groups:
        for entry in manifest.get(group_name, []):
            evidence_path = bundle_root / entry["evidence_path"]
            evidence = _read_json(evidence_path)
            _require_envelope(
                evidence,
                artifact_type=artifact_type,
                schema_version=schema_version,
            )
            for source in evidence.get("sources", {}).values():
                raw_path = bundle_root / source["bundle_path"]
                if not raw_path.is_file():
                    raise FileNotFoundError(f"missing bundled source: {raw_path}")
                if _sha256(raw_path) != source["sha256"]:
                    raise ValueError(f"source checksum mismatch: {raw_path}")
                checked_sources += 1
            checkpoint = evidence.get("checkpoint")
            if checkpoint is not None and not checkpoint.get("sha256"):
                raise ValueError(f"checkpoint checksum missing: {evidence_path}")
    return {
        "runs": len(manifest.get("runs", [])),
        "sweeps": len(manifest.get("sweeps", [])),
        "derived_collections": len(manifest.get("derived_collections", [])),
        "standalone_artifacts": len(manifest.get("standalone_artifacts", [])),
        "checked_sources": checked_sources,
    }


def normalize_resource_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return one stable resource view for legacy and current profiles."""
    mac_coverage = profile.get("mac_coverage")
    flop_coverage = profile.get("flop_coverage")
    full_class = profile.get("full_class_projection") or {}
    eager = list(profile.get("inference_benchmarks") or [])
    optimized = profile.get("optimized_inference") or {}
    end_to_end = profile.get("end_to_end_batch1_benchmark")
    eager_batch_one = _batch_one(eager)
    return {
        "source_variant": (
            "resource-profile-v2"
            if "complexity_protocol" in profile or "flop_coverage" in profile
            else "resource-profile-v1"
        ),
        "model": {
            "name": profile.get("model_name"),
            "classification_head": profile.get("classification_head"),
            "embedding_dim": profile.get("embedding_dim"),
            "backbone_output_channels": profile.get("backbone_output_channels"),
        },
        "parameters": {
            "total": profile.get("parameter_count"),
            "trainable": profile.get("trainable_parameter_count"),
            "backbone": profile.get("backbone_parameter_count"),
            "embedding_projection": profile.get(
                "embedding_projection_parameter_count"
            ),
            "classifier": profile.get("classifier_parameter_count"),
            "head": profile.get("head_parameter_count"),
        },
        "complexity": {
            "macs": _complexity_measure(profile, "macs", mac_coverage),
            "flops": _complexity_measure(profile, "flops", flop_coverage),
            "protocol": profile.get("complexity_protocol"),
            "full_class": {
                "num_classes": full_class.get("num_classes"),
                "parameter_count": full_class.get("total_parameter_count"),
                "macs": full_class.get("total_macs"),
                "flops": full_class.get("total_flops"),
            },
        },
        "latency": {
            "protocol": profile.get("benchmark_protocol"),
            "eager": eager,
            "optimized": list(optimized.get("benchmarks") or []),
            "end_to_end_batch1": {
                "availability": (
                    "measured" if end_to_end is not None else "not_measured"
                ),
                "value": end_to_end,
            },
        },
        "memory": {
            "peak_inference_cuda_memory_mib": (
                eager_batch_one.get("peak_cuda_memory_mib")
                if eager_batch_one is not None
                else None
            )
        },
        "device": profile.get("device"),
        "device_metadata": profile.get("device_metadata"),
    }


def _write_run_evidence(
    source_root: Path, output_root: Path, run_dir: Path
) -> dict[str, Any]:
    raw_dir = output_root / "raw" / "runs" / run_dir.name
    sources = _copy_sources(run_dir, raw_dir, output_root)
    config = _read_json(run_dir / "config.json")
    metadata = _read_json(run_dir / "metadata.json")
    checkpoint_metadata = _read_json(run_dir / "checkpoint_metadata.json")
    metrics = _read_json(run_dir / "metrics.json")
    resource_profile = _read_json(run_dir / "resource_profile.json")
    checkpoint_path = run_dir / "checkpoint.pt"
    evidence = {
        "artifact_type": "hccr.run_evidence",
        "schema_version": RUN_EVIDENCE_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "sources": sources,
        "checkpoint": (
            {
                "source_path": _relative_or_absolute(checkpoint_path, source_root),
                "sha256": _sha256(checkpoint_path),
                "copied": False,
            }
            if checkpoint_path.is_file()
            else None
        ),
        "provenance": {
            "git_commit": metadata.get("git_commit"),
            "git": metadata.get("git"),
            "environment": metadata.get("environment"),
            "manifest_digest": (
                checkpoint_metadata.get("manifest_digest")
                or metadata.get("manifest_digest")
            ),
            "labels_digest": checkpoint_metadata.get("labels_digest"),
            "writer_provenance": metadata.get("writer_provenance"),
        },
        "protocol": {
            "model": checkpoint_metadata.get("model"),
            "preprocess": checkpoint_metadata.get("preprocess"),
            "training": checkpoint_metadata.get("training"),
            "evaluation_policy": config.get("evaluation_policy"),
            "reproducibility_mode": config.get("reproducibility_mode"),
            "seed": config.get("seed"),
            "precision": metadata.get("precision") or metrics.get("precision"),
        },
        "results": {
            "validation": metrics.get("validation"),
            "test": metrics.get("test"),
            "resources": normalize_resource_profile(resource_profile),
        },
    }
    evidence_path = output_root / "runs" / run_dir.name / "run_evidence.json"
    _write_json(evidence_path, evidence)
    return {
        "run_id": run_dir.name,
        "evidence_path": evidence_path.relative_to(output_root).as_posix(),
        "source_artifact_count": len(sources),
        "checkpoint_sha256": (
            evidence["checkpoint"]["sha256"] if evidence["checkpoint"] else None
        ),
    }


def _write_sweep_evidence(
    source_root: Path, output_root: Path, sweep_dir: Path
) -> dict[str, Any]:
    raw_dir = output_root / "raw" / "sweeps" / sweep_dir.name
    sources = _copy_sources(sweep_dir, raw_dir, output_root, recursive=True)
    plan = _optional_json(sweep_dir / "plan.json")
    status = _optional_json(sweep_dir / "status.json")
    summary = _optional_json(sweep_dir / "summary.json")
    evidence = {
        "artifact_type": "hccr.sweep_evidence",
        "schema_version": SWEEP_EVIDENCE_SCHEMA_VERSION,
        "experiment_id": sweep_dir.name,
        "sources": sources,
        "plan": plan,
        "status": status,
        "summary": summary,
        "source_schemas": {
            name: source["detected_schema"] for name, source in sources.items()
        },
    }
    evidence_path = output_root / "sweeps" / sweep_dir.name / "sweep_evidence.json"
    _write_json(evidence_path, evidence)
    return {
        "experiment_id": sweep_dir.name,
        "evidence_path": evidence_path.relative_to(output_root).as_posix(),
        "source_artifact_count": len(sources),
    }


def _write_derived_evidence(
    output_root: Path, derived_root: Path
) -> dict[str, Any]:
    collection_id = derived_root.name
    raw_dir = output_root / "raw" / "derived" / collection_id
    sources = _copy_sources(
        derived_root, raw_dir, output_root, recursive=True
    )
    profiles = []
    for profile_path in sorted(
        derived_root.glob("runs/*/*/resource_profile.json")
    ):
        provenance_path = profile_path.with_name("provenance.json")
        provenance = _optional_json(provenance_path)
        run_id = profile_path.parent.parent.name
        profiles.append(
            {
                "run_id": run_id,
                "device": profile_path.parent.name,
                "source": profile_path.relative_to(derived_root).as_posix(),
                "provenance": provenance,
                "resources": normalize_resource_profile(_read_json(profile_path)),
            }
        )
    evidence = {
        "artifact_type": "hccr.derived_evidence_collection",
        "schema_version": DERIVED_EVIDENCE_SCHEMA_VERSION,
        "collection_id": collection_id,
        "source_root": str(derived_root),
        "sources": sources,
        "summary": _optional_json(derived_root / "summary.json"),
        "profiles": profiles,
    }
    evidence_path = (
        output_root / "derived" / collection_id / "derived_evidence.json"
    )
    _write_json(evidence_path, evidence)
    return {
        "collection_id": collection_id,
        "evidence_path": evidence_path.relative_to(output_root).as_posix(),
        "source_artifact_count": len(sources),
        "normalized_profile_count": len(profiles),
    }


def _copy_standalone_evidence(
    source_root: Path, output_root: Path
) -> list[dict[str, Any]]:
    entries = []
    for source_path in sorted(path for path in source_root.iterdir() if path.is_file()):
        if source_path.suffix.lower() not in {".csv", ".json"}:
            continue
        raw_path = output_root / "raw" / "standalone" / source_path.name
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, raw_path)
        source = _source_descriptor(source_path, raw_path, output_root)
        evidence = {
            "artifact_type": "hccr.standalone_evidence",
            "schema_version": 1,
            "name": source_path.name,
            "sources": {source_path.name: source},
            "row_count": _csv_row_count(source_path),
        }
        evidence_path = (
            output_root / "standalone" / f"{source_path.name}.evidence.json"
        )
        _write_json(evidence_path, evidence)
        entries.append(
            {
                "name": source_path.name,
                "evidence_path": evidence_path.relative_to(output_root).as_posix(),
            }
        )
    return entries


def _copy_sources(
    source_dir: Path,
    raw_dir: Path,
    output_root: Path,
    *,
    recursive: bool = False,
) -> dict[str, dict[str, Any]]:
    iterator = source_dir.rglob("*") if recursive else source_dir.iterdir()
    sources = {}
    for source_path in sorted(path for path in iterator if path.is_file()):
        if source_path.suffix.lower() not in _RAW_SUFFIXES:
            continue
        relative = source_path.relative_to(source_dir)
        raw_path = raw_dir / relative
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        source_hash = _sha256(source_path)
        shutil.copy2(source_path, raw_path)
        if _sha256(raw_path) != source_hash:
            raise ValueError(f"raw artifact changed while copying: {source_path}")
        key = relative.as_posix()
        sources[key] = _source_descriptor(source_path, raw_path, output_root)
    return sources


def _source_descriptor(
    source_path: Path, raw_path: Path, output_root: Path
) -> dict[str, Any]:
    return {
        "source_path": str(source_path.resolve()),
        "bundle_path": raw_path.relative_to(output_root).as_posix(),
        "sha256": _sha256(source_path),
        "size_bytes": source_path.stat().st_size,
        "detected_schema": _detect_schema(source_path),
    }


def _detect_schema(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as file:
            fields = next(csv.reader(file), [])
        digest = hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()[:12]
        return f"csv-header-sha256:{digest}"
    if path.suffix.lower() != ".json":
        return "unversioned-binary"
    payload = _read_json(path)
    declared = payload.get("schema_version")
    if declared is not None:
        return f"declared-v{declared}"
    name = path.name
    if name == "resource_profile.json":
        return (
            "resource-profile-v2"
            if "complexity_protocol" in payload or "flop_coverage" in payload
            else "resource-profile-v1"
        )
    if name == "status.json":
        return "sweep-status-v2" if "records" in payload else "sweep-status-v1"
    known = {
        "config.json": "training-config-v1",
        "metadata.json": "run-metadata-v1",
        "plan.json": "sweep-plan-v1",
        "batch_training_plan.json": "batch-training-plan-v1",
        "labels.json": "labels-v1",
    }
    return known.get(name, "unversioned-json")


def _complexity_measure(
    profile: Mapping[str, Any], suffix: str, coverage: Any
) -> dict[str, Any]:
    total_key = f"estimated_{suffix}"
    measured = total_key in profile and profile.get(total_key) is not None
    return {
        "availability": "measured" if measured else "not_measured",
        "total": profile.get(total_key),
        "backbone": profile.get(f"estimated_backbone_{suffix}"),
        "embedding_projection": profile.get(
            f"estimated_embedding_projection_{suffix}"
        ),
        "classifier": profile.get(f"estimated_classifier_{suffix}"),
        "head": profile.get(f"estimated_head_{suffix}"),
        "coverage": coverage,
    }


def _batch_one(rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((row for row in rows if int(row.get("batch_size", -1)) == 1), None)


def _discover_run_dirs(source_root: Path, selected: set[str]) -> list[Path]:
    candidates = [source_root / run_id for run_id in selected] if selected else []
    if not candidates:
        candidates = sorted(path for path in source_root.iterdir() if path.is_dir())
    return [
        path
        for path in candidates
        if path.is_dir()
        and all((path / filename).is_file() for filename in _RUN_REQUIRED_FILES)
    ]


def _discover_sweep_dirs(source_root: Path, selected: set[str]) -> list[Path]:
    sweeps_root = source_root / "sweeps"
    if not sweeps_root.is_dir():
        return []
    candidates = [sweeps_root / sweep_id for sweep_id in selected] if selected else []
    if not candidates:
        candidates = sorted(path for path in sweeps_root.iterdir() if path.is_dir())
    return [path for path in candidates if path.is_dir()]


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _csv_row_count(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader, None)
        return sum(1 for _ in reader)


def _require_envelope(
    payload: Mapping[str, Any], *, artifact_type: str, schema_version: int
) -> None:
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"unexpected artifact type: {payload.get('artifact_type')}")
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"unsupported schema version: {payload.get('schema_version')}")


def _optional_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
