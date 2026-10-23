"""Sequential experiment execution with resumable state."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from hccr.artifacts import write_json
from hccr.artifacts.checkpoints import file_digest
from hccr.benchmarking.profiling import _device_metadata, profile_model
from hccr.cli import build_parser as build_hccr_parser
from hccr.experiments.spec import (
    ExperimentJob,
    ExperimentSpec,
    _jsonable_spec,
    _validate_hardware_requirements,
)
from hccr.experiments.summary import _summarize
from hccr.models import load_model_from_run

_NEGATED_BOOLEAN_TRAIN_OPTIONS = {
    "reparameterize_depthwise": "--no-reparameterize-depthwise",
    "persistent_workers": "--no-persistent-workers",
    "gate_use_gap": "--no-gate-use-gap",
    "gate_detach_gap": "--no-gate-detach-gap",
}


def build_jobs(spec: ExperimentSpec) -> list[ExperimentJob]:
    jobs = []
    for variant in spec.variants:
        for seed in spec.seeds:
            train_args = {**spec.base_args, **variant.args, "seed": seed}
            command = _training_command(spec, train_args)
            build_hccr_parser().parse_args(list(command[command.index("train") :]))
            jobs.append(
                ExperimentJob(
                    key=f"{variant.name}/seed-{seed}",
                    variant=variant.name,
                    seed=seed,
                    train_args=train_args,
                    command=command,
                )
            )
    return jobs


class ExperimentRunner:
    """Own one sweep specification and its execution lifecycle."""

    def __init__(self, spec: ExperimentSpec) -> None:
        self.spec = spec

    def run_experiments(
        self,
        *,
        dry_run: bool = False,
        resume: bool = False,
        show_output: bool = False,
        continue_on_error: bool = False,
    ) -> dict[str, Any]:
        spec = self.spec
        jobs = build_jobs(spec)
        if not dry_run:
            _validate_hardware_requirements(spec)
        experiment_dir = spec.output_dir / "sweeps" / spec.experiment_id
        state_path = experiment_dir / "status.json"
        if experiment_dir.exists() and not (resume or dry_run):
            raise FileExistsError(
                f"experiment directory already exists; use --resume: {experiment_dir}"
            )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "logs").mkdir(exist_ok=True)
        write_json(
            experiment_dir / "plan.json",
            {
                "spec": _jsonable_spec(spec),
                "jobs": [{**asdict(job), "command": list(job.command)} for job in jobs],
            },
        )
        if dry_run:
            result = {"status": "dry_run_complete", "jobs": len(jobs)}
            write_json(state_path, result)
            return result
        prior = _read_json(state_path) if resume and state_path.is_file() else {}
        records = [_normalize_record(record) for record in prior.get("records", [])]
        completed = {
            record["key"] for record in records if record["status"] == "completed"
        }
        started_at = prior.get("started_at", datetime.now(UTC).isoformat())
        for job in jobs:
            if job.key in completed:
                continue
            _write_status(state_path, started_at, records, "training", job.key)
            try:
                run_dir = _run_training_job(spec, job, experiment_dir, show_output)
                run_result = _load_run_result(run_dir / "metrics.json")
                profiles = _profile_run(spec, job, run_dir, experiment_dir)
                records.append(
                    {
                        "key": job.key,
                        "variant": job.variant,
                        "seed": job.seed,
                        "status": "completed",
                        "run_id": run_dir.name,
                        "run_path": str(run_dir),
                        **run_result,
                        "profiles": profiles,
                    }
                )
            except Exception as error:
                records.append(
                    {
                        "key": job.key,
                        "variant": job.variant,
                        "seed": job.seed,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                _write_status(state_path, started_at, records, "failed", job.key)
                if not continue_on_error:
                    raise
            _write_status(state_path, started_at, records, "running")
        summary = _summarize(spec, records, started_at)
        write_json(experiment_dir / "summary.json", summary)
        _write_status(state_path, started_at, records, summary["status"])
        return summary


def run_experiments(
    spec: ExperimentSpec,
    *,
    dry_run: bool = False,
    resume: bool = False,
    show_output: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    return ExperimentRunner(spec).run_experiments(
        dry_run=dry_run,
        resume=resume,
        show_output=show_output,
        continue_on_error=continue_on_error,
    )


def _training_command(
    spec: ExperimentSpec, train_args: dict[str, Any]
) -> tuple[str, ...]:
    command = [sys.executable]
    if spec.torchrun_nproc_per_node is not None:
        command.extend(
            [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={spec.torchrun_nproc_per_node}",
            ]
        )
    command.extend(
        [
            "-m",
            "hccr",
            "train",
            "--manifest",
            str(spec.manifest),
            "--output-dir",
            str(spec.output_dir),
        ]
    )
    for key, value in train_args.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
            elif negated_option := _NEGATED_BOOLEAN_TRAIN_OPTIONS.get(key):
                command.append(negated_option)
        elif value is not None:
            command.append(option)
            if isinstance(value, (list, tuple)):
                command.extend(str(item) for item in value)
            else:
                command.append(str(value))
    return tuple(command)


def _run_training_job(
    spec: ExperimentSpec,
    job: ExperimentJob,
    experiment_dir: Path,
    show_output: bool,
) -> Path:
    before = _run_directories(spec.output_dir)
    log_path = experiment_dir / "logs" / f"{job.variant}-seed-{job.seed}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            job.command,
            check=True,
            stdout=None if show_output else log_file,
            stderr=None if show_output else subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        )
    created = _run_directories(spec.output_dir) - before
    if len(created) != 1:
        raise RuntimeError(
            f"expected one run directory for {job.key}, found {len(created)}"
        )
    return created.pop()


def _profile_run(
    spec: ExperimentSpec,
    job: ExperimentJob,
    run_dir: Path,
    experiment_dir: Path,
) -> dict[str, Any]:
    profiles = {}
    warmup = int(job.train_args.get("benchmark_warmup_iterations", 20))
    iterations = int(job.train_args.get("benchmark_iterations", 200))
    repetitions = int(job.train_args.get("benchmark_repetitions", 5))
    metadata = _read_json(run_dir / "checkpoint_metadata.json")
    image_size = int(metadata["preprocess"]["image_size"])
    metrics = _read_json(run_dir / "metrics.json")
    checkpoint_name = metrics.get("validation", {}).get(
        "selected_checkpoint", "checkpoint"
    )
    checkpoint_name = str(checkpoint_name).removesuffix(".pt")
    if checkpoint_name not in {"checkpoint", "checkpoint_recalibrated"}:
        raise ValueError(f"unsupported selected checkpoint: {checkpoint_name}")
    checkpoint_digest = file_digest(run_dir / f"{checkpoint_name}.pt")
    cached_path = run_dir / "resource_profile.json"
    cached = _read_json(cached_path) if cached_path.is_file() else {}
    expected_protocol = {
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "repetitions": repetitions,
    }
    for device in spec.profile_devices:
        if device == "cuda" and not torch.cuda.is_available():
            profiles[device] = {"status": "skipped", "reason": "CUDA unavailable"}
            continue
        cache_matches = (
            cached.get("schema_version") == 2
            and cached.get("measurement_stage") == "selected_checkpoint"
            and cached.get("checkpoint_sha256") == checkpoint_digest
            and cached.get("device_metadata") == _device_metadata(device)
            and all(
                cached.get("benchmark_protocol", {}).get(key) == value
                for key, value in expected_protocol.items()
            )
            and cached.get("full_class_projection", {}).get("num_classes")
            == (spec.full_class_num_classes or metadata["model"]["num_classes"])
        )
        if cache_matches:
            profiles[device] = {
                "status": "completed",
                "trained_head": cached,
                "source": str(cached_path),
            }
            continue
        destination = experiment_dir / "profiles" / job.variant / f"seed-{job.seed}"
        trained_model = load_model_from_run(run_dir, device)
        if checkpoint_name != "checkpoint":
            trained_model.load_state_dict(
                torch.load(
                    run_dir / f"{checkpoint_name}.pt",
                    map_location=device,
                    weights_only=True,
                )
            )
        trained_profile = profile_model(
            trained_model,
            image_size,
            device,
            destination / device / "trained_head",
            warmup,
            iterations,
            repetitions,
            full_class_num_classes=spec.full_class_num_classes
            or metadata["model"]["num_classes"],
        )
        trained_profile["checkpoint_name"] = checkpoint_name
        trained_profile["measurement_stage"] = "selected_checkpoint"
        trained_profile["checkpoint_sha256"] = checkpoint_digest
        write_json(
            destination / device / "trained_head" / "resource_profile.json",
            trained_profile,
        )
        device_profiles = {"status": "completed", "trained_head": trained_profile}
        del trained_model
        profiles[device] = device_profiles
        if device == "cuda":
            torch.cuda.empty_cache()
    return profiles


def _run_directories(output_dir: Path) -> set[Path]:
    return {
        path.resolve()
        for path in output_dir.glob("20*")
        if path.is_dir() and (path / "metadata.json").is_file()
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_result(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    validation = payload.get("validation")
    if isinstance(validation, dict) and isinstance(validation.get("best"), dict):
        test = payload.get("test")
        test_status = (
            test.get("status", "not_run") if isinstance(test, dict) else "not_run"
        )
        test_metrics = test.get("metrics") if isinstance(test, dict) else None
        if test_metrics is not None and not isinstance(test_metrics, dict):
            raise ValueError(f"invalid test.metrics object in {path}")
        icdar = payload.get("icdar")
        icdar_metrics = icdar.get("metrics") if isinstance(icdar, dict) else None
        if icdar_metrics is not None and not isinstance(icdar_metrics, dict):
            raise ValueError(f"invalid icdar.metrics object in {path}")
        selected = validation.get("selected", validation["best"])
        if not isinstance(selected, dict):
            raise ValueError(f"invalid validation.selected object in {path}")
        return {
            "metrics_schema": f"nested-v{payload.get('schema_version', 1)}",
            "selection_split": str(validation.get("split", "validation")),
            "selection_metrics": selected,
            "test_status": test_status,
            "test_metrics": test_metrics,
            "icdar_status": (
                icdar.get("status", "not_run") if isinstance(icdar, dict) else "not_run"
            ),
            "icdar_metrics": icdar_metrics,
        }
    legacy = payload.get("best")
    if isinstance(legacy, dict):
        return {
            "metrics_schema": "legacy-best",
            "selection_split": "validation",
            "selection_metrics": legacy,
            "test_status": "unknown",
            "test_metrics": None,
        }
    raise ValueError(
        f"unsupported metrics schema in {path}; expected validation.best or legacy best"
    )


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("status") != "completed" or "selection_metrics" in record:
        return record
    legacy_metrics = record.get("metrics")
    if not isinstance(legacy_metrics, dict):
        raise ValueError(f"completed record {record.get('key')} has no metrics")
    normalized = dict(record)
    normalized.pop("metrics", None)
    normalized.update(
        {
            "metrics_schema": "legacy-record",
            "selection_split": "validation",
            "selection_metrics": legacy_metrics,
            "test_status": "unknown",
            "test_metrics": None,
        }
    )
    return normalized


def _write_status(
    path: Path,
    started_at: str,
    records: list[dict[str, Any]],
    status: str,
    active_job: str | None = None,
) -> None:
    write_json(
        path,
        {
            "status": status,
            "started_at": started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "active_job": active_job,
            "records": records,
        },
    )
