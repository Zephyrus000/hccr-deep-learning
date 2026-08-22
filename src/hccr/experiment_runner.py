"""Configuration-driven sequential training and ablation orchestration."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from hccr.cli import build_parser as build_hccr_parser
from hccr.config.loader import load_yaml
from hccr.models import build_model
from hccr.preprocessing import EvalPreprocessor
from hccr.training.diagnostics import profile_model
from hccr.utils.experiment import write_json


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    manifest: Path
    output_dir: Path
    seeds: tuple[int, ...]
    base_args: dict[str, Any]
    variants: tuple[ExperimentVariant, ...]
    profile_devices: tuple[str, ...] = ("cpu",)
    full_class_num_classes: int | None = 7186


@dataclass(frozen=True)
class ExperimentJob:
    key: str
    variant: str
    seed: int
    train_args: dict[str, Any]
    command: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ordered HCCR seed sweeps and one-factor ablations."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--set",
        dest="base_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument(
        "--variant",
        dest="variant_overrides",
        action="append",
        default=[],
        metavar="JSON",
        help=(
            "JSON/YAML mapping with name and args; use the YAML config for "
            "complex variants."
        ),
    )
    parser.add_argument("--profile-devices", choices=("cpu", "cuda"), nargs="+")
    parser.add_argument("--full-class-num-classes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--show-output", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def load_experiment_spec(
    arguments: argparse.Namespace, project_root: Path
) -> ExperimentSpec:
    raw = load_yaml(arguments.config) if arguments.config else {}
    experiment_id = arguments.experiment_id or raw.get("experiment_id")
    if not experiment_id or any(character in experiment_id for character in "/\\"):
        raise ValueError("experiment_id must be a non-path name")
    manifest_value = arguments.manifest or raw.get(
        "manifest", "data/processed/casia_hwdb/manifest.csv"
    )
    output_value = arguments.output_dir or raw.get("output_dir", "experiments")
    seeds = tuple(arguments.seeds or raw.get("seeds", (7, 17, 29)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    base_args = dict(raw.get("base_args", {}))
    base_args.update(_parse_assignments(arguments.base_overrides))
    raw_variants = list(raw.get("variants", []))
    raw_variants.extend(_parse_variant(value) for value in arguments.variant_overrides)
    if not raw_variants:
        raw_variants = [{"name": "default", "args": {}}]
    variants = tuple(
        ExperimentVariant(str(item["name"]), dict(item.get("args", {})))
        for item in raw_variants
    )
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("variant names must be unique")
    forbidden = {"manifest", "output_dir", "seed"}
    for source_name, values in [
        ("base_args", base_args),
        *((f"variant {variant.name}", variant.args) for variant in variants),
    ]:
        overlap = forbidden.intersection(values)
        if overlap:
            raise ValueError(
                f"{source_name} cannot override reserved arguments: {sorted(overlap)}"
            )
    profile_devices = tuple(
        arguments.profile_devices or raw.get("profile_devices", ("cpu",))
    )
    full_class_num_classes = (
        arguments.full_class_num_classes
        if arguments.full_class_num_classes is not None
        else raw.get("full_class_num_classes", 7186)
    )
    if full_class_num_classes is not None and full_class_num_classes < 2:
        raise ValueError("full_class_num_classes must be at least 2 or null")
    return ExperimentSpec(
        experiment_id=experiment_id,
        manifest=_resolve_from_root(project_root, Path(manifest_value)),
        output_dir=_resolve_from_root(project_root, Path(output_value)),
        seeds=seeds,
        base_args=base_args,
        variants=variants,
        profile_devices=profile_devices,
        full_class_num_classes=full_class_num_classes,
    )


def build_jobs(spec: ExperimentSpec) -> list[ExperimentJob]:
    jobs = []
    for variant in spec.variants:
        for seed in spec.seeds:
            train_args = {**spec.base_args, **variant.args, "seed": seed}
            command = _training_command(spec, train_args)
            build_hccr_parser().parse_args(list(command[3:]))
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


def run_experiments(
    spec: ExperimentSpec,
    *,
    dry_run: bool = False,
    resume: bool = False,
    show_output: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    jobs = build_jobs(spec)
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
    records = list(prior.get("records", []))
    completed = {record["key"] for record in records if record["status"] == "completed"}
    started_at = prior.get("started_at", datetime.now(UTC).isoformat())
    for job in jobs:
        if job.key in completed:
            continue
        _write_status(state_path, started_at, records, "training", job.key)
        try:
            run_dir = _run_training_job(spec, job, experiment_dir, show_output)
            metrics = _read_json(run_dir / "metrics.json")["best"]
            profiles = _profile_run(spec, job, run_dir, experiment_dir)
            records.append(
                {
                    "key": job.key,
                    "variant": job.variant,
                    "seed": job.seed,
                    "status": "completed",
                    "run_id": run_dir.name,
                    "run_path": str(run_dir),
                    "metrics": metrics,
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
    _write_status(state_path, started_at, records, "completed")
    return summary


def _training_command(
    spec: ExperimentSpec, train_args: dict[str, Any]
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "-m",
        "hccr",
        "train",
        "--manifest",
        str(spec.manifest),
        "--output-dir",
        str(spec.output_dir),
    ]
    for key, value in train_args.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
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
            cwd=Path(__file__).resolve().parents[2],
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
    transform = EvalPreprocessor(**metadata["preprocess"])
    image_size = int(metadata["preprocess"]["image_size"])
    for device in spec.profile_devices:
        if device == "cuda" and not torch.cuda.is_available():
            profiles[device] = {"status": "skipped", "reason": "CUDA unavailable"}
            continue
        destination = experiment_dir / "profiles" / job.variant / f"seed-{job.seed}"
        trained_model = _model_from_run(run_dir, device)
        trained_profile = profile_model(
            trained_model,
            image_size,
            device,
            destination / device / "trained_head",
            warmup,
            iterations,
            repetitions,
            transform,
            spec.full_class_num_classes or metadata["model"]["num_classes"],
        )
        device_profiles = {"status": "completed", "trained_head": trained_profile}
        del trained_model
        if spec.full_class_num_classes is not None:
            full_model = _model_from_run(run_dir, device, spec.full_class_num_classes)
            device_profiles["full_class_head"] = profile_model(
                full_model,
                image_size,
                device,
                destination / device / "full_class_head",
                warmup,
                iterations,
                repetitions,
                transform,
                spec.full_class_num_classes,
            )
            del full_model
        profiles[device] = device_profiles
        if device == "cuda":
            torch.cuda.empty_cache()
    return profiles


def _model_from_run(
    run_dir: Path, device: str, num_classes_override: int | None = None
) -> torch.nn.Module:
    metadata = _read_json(run_dir / "checkpoint_metadata.json")
    stored = metadata["model"]
    model_keys = {
        "in_channels",
        "width",
        "stage_depths",
        "dropout",
        "classification_head",
        "logit_scale",
        "angular_margin",
    }
    kwargs = {key: stored[key] for key in model_keys if key in stored}
    for tuple_key in ("stage_depths",):
        if tuple_key in kwargs:
            kwargs[tuple_key] = tuple(kwargs[tuple_key])
    target_classes = num_classes_override or int(stored["num_classes"])
    model = build_model(stored["name"], num_classes=target_classes, **kwargs)
    state = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    if target_classes == int(stored["num_classes"]):
        model.load_state_dict(state)
    else:
        backbone_state = {
            name: value
            for name, value in state.items()
            if not name.startswith("classifier.")
        }
        incompatible = model.load_state_dict(backbone_state, strict=False)
        if incompatible.unexpected_keys or any(
            not key.startswith("classifier.") for key in incompatible.missing_keys
        ):
            raise RuntimeError("full-class model reconstruction changed backbone keys")
    return model.to(device)


def _summarize(
    spec: ExperimentSpec, records: list[dict[str, Any]], started_at: str
) -> dict[str, Any]:
    completed = [record for record in records if record["status"] == "completed"]
    variants = {}
    metric_names = (
        "top1",
        "top5",
        "macro_recall",
        "head_recall",
        "mid_recall",
        "tail_recall",
        "expected_calibration_error",
    )
    for variant in spec.variants:
        rows = [record for record in completed if record["variant"] == variant.name]
        variants[variant.name] = {
            "completed_seeds": [row["seed"] for row in rows],
            "metrics": {
                metric: _distribution(
                    [
                        float(row["metrics"][metric])
                        for row in rows
                        if metric in row["metrics"]
                    ]
                )
                for metric in metric_names
            },
            "batch1_p95_ms": _aggregate_profile_latency(rows),
        }
    baseline_name = spec.variants[0].name
    comparisons = {
        name: _compare_variant_aggregates(variants[baseline_name], aggregate)
        for name, aggregate in variants.items()
        if name != baseline_name
    }
    return {
        "status": "completed",
        "experiment_id": spec.experiment_id,
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "spec": _jsonable_spec(spec),
        "baseline_variant": baseline_name,
        "records": records,
        "variants": variants,
        "comparisons": comparisons,
    }


def _aggregate_profile_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for device in ("cpu", "cuda"):
        values = []
        full_values = []
        optimized_values = []
        optimized_full_values = []
        for row in rows:
            profile = row["profiles"].get(device, {})
            if profile.get("status") != "completed":
                continue
            values.append(
                profile["trained_head"]["inference_benchmarks"][0]["latency_p95_ms"]
            )
            optimized = profile["trained_head"].get("optimized_inference")
            if optimized is not None:
                optimized_values.append(optimized["benchmarks"][0]["latency_p95_ms"])
            if "full_class_head" in profile:
                full_values.append(
                    profile["full_class_head"]["inference_benchmarks"][0][
                        "latency_p95_ms"
                    ]
                )
                optimized_full = profile["full_class_head"].get("optimized_inference")
                if optimized_full is not None:
                    optimized_full_values.append(
                        optimized_full["benchmarks"][0]["latency_p95_ms"]
                    )
        result[device] = {
            "trained_head": _distribution(values),
            "full_class_head": _distribution(full_values),
            "optimized_trained_head": _distribution(optimized_values),
            "optimized_full_class_head": _distribution(optimized_full_values),
        }
    return result


def _compare_variant_aggregates(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    result = {
        "top1_mean_delta": _difference(
            candidate["metrics"]["top1"], baseline["metrics"]["top1"]
        ),
        "macro_recall_mean_delta": _difference(
            candidate["metrics"]["macro_recall"],
            baseline["metrics"]["macro_recall"],
        ),
        "tail_recall_mean_delta": _difference(
            candidate["metrics"]["tail_recall"],
            baseline["metrics"]["tail_recall"],
        ),
        "latency_p95_ratio": {},
        "optimized_latency_p95_ratio": {},
    }
    for device in ("cpu", "cuda"):
        baseline_latency = _mean(baseline["batch1_p95_ms"][device]["trained_head"])
        candidate_latency = _mean(candidate["batch1_p95_ms"][device]["trained_head"])
        result["latency_p95_ratio"][device] = (
            candidate_latency / baseline_latency
            if baseline_latency and candidate_latency
            else None
        )
        optimized_baseline = _mean(
            baseline["batch1_p95_ms"][device]["optimized_trained_head"]
        )
        optimized_candidate = _mean(
            candidate["batch1_p95_ms"][device]["optimized_trained_head"]
        )
        result["optimized_latency_p95_ratio"][device] = (
            optimized_candidate / optimized_baseline
            if optimized_baseline and optimized_candidate
            else None
        )
    return result


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _mean(distribution: dict[str, Any]) -> float | None:
    value = distribution.get("mean")
    return float(value) if value is not None else None


def _difference(candidate: dict[str, Any], baseline: dict[str, Any]) -> float | None:
    candidate_mean = _mean(candidate)
    baseline_mean = _mean(baseline)
    if candidate_mean is None or baseline_mean is None:
        return None
    return candidate_mean - baseline_mean


def _parse_assignments(values: list[str]) -> dict[str, Any]:
    import yaml

    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE: {value}")
        key, encoded = value.split("=", 1)
        result[key.replace("-", "_")] = yaml.safe_load(encoded)
    return result


def _parse_variant(value: str) -> dict[str, Any]:
    import yaml

    parsed = yaml.safe_load(value)
    if not isinstance(parsed, dict) or "name" not in parsed:
        raise ValueError("--variant must be a mapping with name and optional args")
    return parsed


def _resolve_from_root(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _run_directories(output_dir: Path) -> set[Path]:
    return {
        path.resolve()
        for path in output_dir.glob("20*")
        if path.is_dir() and (path / "metadata.json").is_file()
    }


def _jsonable_spec(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "manifest": str(spec.manifest),
        "output_dir": str(spec.output_dir),
        "seeds": list(spec.seeds),
        "base_args": spec.base_args,
        "variants": [asdict(variant) for variant in spec.variants],
        "profile_devices": list(spec.profile_devices),
        "full_class_num_classes": spec.full_class_num_classes,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    spec = load_experiment_spec(arguments, project_root)
    run_experiments(
        spec,
        dry_run=arguments.dry_run,
        resume=arguments.resume,
        show_output=arguments.show_output,
        continue_on_error=arguments.continue_on_error,
    )
    return 0
