"""Plan exact experiment jobs and run at most one child per selected GPU."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from hccr.artifacts.checkpoints import file_digest
from hccr.cli import build_parser
from hccr.commands.train import config_from_arguments
from hccr.experiments.cli import build_parser as experiment_parser
from hccr.experiments.runner import _load_run_result, build_jobs
from hccr.experiments.spec import load_experiment_spec
from hccr.training.workflow import _validate_training_config


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted([*root.glob("src/**/*.py"), *root.glob("scripts/**/*.py")]):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(file_digest(path)))
    return digest.hexdigest()


def _plan(configs: list[Path], output: Path, root: Path) -> tuple[list[dict], str]:
    jobs, digests = {}, {}
    code_digest = _code_digest(root)
    for path in configs:
        spec = load_experiment_spec(
            experiment_parser().parse_args(["--config", str(path)]), root
        )
        if spec.torchrun_nproc_per_node is not None:
            raise ValueError(
                "GPU queue requires independent jobs; torchrun/DDP is unsupported"
            )
        for job in build_jobs(spec):
            parsed = build_parser().parse_args(
                list(job.command[job.command.index("train") :])
            )
            config = config_from_arguments(parsed)
            _validate_training_config(config)
            if config.distributed or config.device not in {"auto", "cuda"}:
                raise ValueError("GPU queue requires single-process CUDA/auto training")
            scientific = {
                key: str((root / value).resolve()) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
                if key != "output_dir"
            }
            inputs = [spec.manifest]
            if config.residual_mask is not None:
                inputs.append((root / config.residual_mask).resolve())
            if config.initialize_from_run is not None:
                inputs.extend(
                    (root / config.initialize_from_run / name).resolve()
                    for name in (
                        "checkpoint.pt",
                        "checkpoint_metadata.json",
                        "labels.json",
                    )
                )
            hashes = {}
            for source in inputs:
                if source not in digests:
                    digests[source] = file_digest(source) if source.is_file() else None
                hashes[str(source)] = digests[source]
            identity = {
                "training": scientific,
                "input_digests": hashes,
                "code_digest": code_digest,
                "profile_devices": spec.profile_devices,
                "full_class_num_classes": spec.full_class_num_classes,
            }
            fingerprint = hashlib.sha256(
                json.dumps(identity, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            alias = {"config": str(path), "variant": job.variant, "seed": job.seed}
            requirement = {
                "count": spec.required_cuda_device_count,
                "name": spec.required_cuda_device_name,
            }
            if fingerprint in jobs:
                jobs[fingerprint]["aliases"].append(alias)
                jobs[fingerprint]["requirements"].append(requirement)
                continue
            directory = output / "jobs" / fingerprint
            experiment_id = "queue-" + fingerprint[:20]
            child = {
                "experiment_id": experiment_id,
                "manifest": str(spec.manifest),
                "output_dir": str(directory / "runs"),
                "seeds": [job.seed],
                "profile_devices": list(spec.profile_devices),
                "full_class_num_classes": spec.full_class_num_classes,
                "base_args": {
                    key: value for key, value in job.train_args.items() if key != "seed"
                },
                "variants": [{"name": "job", "args": {}}],
            }
            jobs[fingerprint] = {
                "fingerprint": fingerprint,
                "identity": identity,
                "aliases": [alias],
                "requirements": [requirement],
                "config": str(directory / "job.yaml"),
                "child_status": str(
                    directory / "runs" / "sweeps" / experiment_id / "status.json"
                ),
                "log": str(directory / "queue.log"),
                "child": child,
                "status": "pending",
            }
    return list(jobs.values()), code_digest


def _preflight(jobs: list[dict], gpus: list[int]) -> list[dict]:
    count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    mapping = (
        {int(value): index for index, value in enumerate(visible.split(","))}
        if visible
        else {gpu: gpu for gpu in gpus}
    )
    hardware = []
    for gpu in gpus:
        logical = mapping.get(gpu, -1)
        if not 0 <= logical < count:
            raise RuntimeError(
                f"Requested physical GPU {gpu} is not visible; "
                f"found {count} CUDA devices"
            )
        with torch.cuda.device(logical):
            hardware.append(
                {
                    "physical_id": gpu,
                    "name": torch.cuda.get_device_name(logical),
                    "bfloat16_supported": torch.cuda.is_bf16_supported(),
                }
            )
    for job in jobs:
        for alias in job["aliases"]:
            if not Path(alias["config"]).is_file():
                raise FileNotFoundError(alias["config"])
        for path, expected in job["identity"]["input_digests"].items():
            if (
                expected is None
                or not Path(path).is_file()
                or file_digest(Path(path)) != expected
            ):
                raise ValueError(f"Missing or changed experiment input: {path}")
        for requirement in job["requirements"]:
            if requirement["count"] and len(gpus) < requirement["count"]:
                raise RuntimeError(
                    "Selected GPU count is below an experiment hardware requirement"
                )
            if requirement["name"] and any(
                requirement["name"].casefold() not in device["name"].casefold()
                for device in hardware
            ):
                raise RuntimeError(
                    "Selected GPU names do not match experiment requirements"
                )
        if job["identity"]["training"]["precision"] == "bfloat16" and not all(
            device["bfloat16_supported"] for device in hardware
        ):
            raise RuntimeError(
                "bfloat16 was requested but a selected GPU does not support it"
            )
    return hardware


def _artifacts(job: dict) -> dict[str, str]:
    status_path = Path(job["child_status"])
    status = _json(status_path)
    records = [
        record
        for record in status.get("records", [])
        if record.get("status") == "completed"
    ]
    if status.get("status") != "completed" or len(records) != 1:
        raise ValueError("Child experiment has no verified completed job")
    seed = job["identity"]["training"]["seed"]
    if records[0].get("key") != f"job/seed-{seed}" or records[0].get("seed") != seed:
        raise ValueError("Completed child identity does not match its queued seed")
    run = Path(records[0]["run_path"]).resolve()
    if not run.is_relative_to(Path(job["child"]["output_dir"]).resolve()):
        raise ValueError("Child run path escapes its job directory")
    metrics = _json(run / "metrics.json")
    _load_run_result(run / "metrics.json")
    metadata = _json(run / "checkpoint_metadata.json")
    if (
        metadata.get("training", {}).get("seed") != seed
        or metadata.get("manifest_digest")
        != job["identity"]["input_digests"][job["child"]["manifest"]]
    ):
        raise ValueError(
            "Completed checkpoint seed/manifest differs from queued inputs"
        )
    selected = str(
        metrics.get("validation", {}).get("selected_checkpoint", "checkpoint")
    ).removesuffix(".pt")
    if selected not in {"checkpoint", "checkpoint_recalibrated"}:
        raise ValueError("Unsupported selected checkpoint")
    paths = {
        status_path,
        run / "metrics.json",
        run / "labels.json",
        run / "checkpoint.pt",
        run / "checkpoint_metadata.json",
        run / f"{selected}.pt",
        run / f"{selected}_metadata.json",
    }
    return {str(path): file_digest(path) for path in paths}


def _execute(job: dict, gpu: int, root: Path) -> dict:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    try:
        for source, expected in job["identity"]["input_digests"].items():
            if file_digest(Path(source)) != expected:
                raise ValueError(f"Experiment input changed before launch: {source}")
        if _code_digest(root) != job["identity"]["code_digest"]:
            raise ValueError("Experiment source code changed before launch")
        with Path(job["log"]).open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "hccr.experiments",
                    "--config",
                    job["config"],
                    "--resume",
                ],
                cwd=root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"Child exited with code {result.returncode}")
        return {"status": "completed", "gpu": gpu, "artifacts": _artifacts(job)}
    except Exception as error:
        return {
            "status": "failed",
            "gpu": gpu,
            "error": f"{type(error).__name__}: {error}; inspect {job['log']}",
        }


def run_gpu_queue(
    config_paths: list[Path],
    gpus: list[int],
    output_dir: Path,
    *,
    execute: bool = False,
    resume: bool = False,
    project_root: Path | None = None,
) -> dict:
    """Write a plan by default; execution launches isolated experiment children."""
    if (
        not config_paths
        or not gpus
        or len(set(gpus)) != len(gpus)
        or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
    ):
        raise ValueError("Provide configurations and unique nonnegative GPU indices")
    root = (project_root or Path(__file__).resolve().parents[3]).resolve()
    output = Path(output_dir).resolve()
    jobs, code_digest = _plan(
        [Path(path).resolve() for path in config_paths], output, root
    )
    status_path = output / "status.json"
    prior = _json(status_path) if status_path.is_file() else None
    if prior and not resume:
        raise FileExistsError("Queue output already exists; use --resume")
    if prior and {job["fingerprint"] for job in prior["jobs"]} != {
        job["fingerprint"] for job in jobs
    }:
        raise ValueError("Queue inputs/code changed; use a new output directory")
    state = {
        "schema_version": 1,
        "status": "planned",
        "code_digest": code_digest,
        "gpus": gpus,
        "jobs": jobs,
        "hardware": _preflight(jobs, gpus) if execute else [],
    }
    previous = {job["fingerprint"]: job for job in prior["jobs"]} if prior else {}
    pending = []
    for job in jobs:
        saved = previous.get(job["fingerprint"], {})
        if saved.get("status") == "completed" or saved.get("verification_failed"):
            try:
                if not saved.get("artifacts") or _artifacts(job) != saved["artifacts"]:
                    raise ValueError("Completed artifacts differ from saved hashes")
                job.update(status="completed", artifacts=saved["artifacts"])
            except (OSError, ValueError, KeyError) as error:
                job.update(
                    status="failed",
                    verification_failed=True,
                    artifacts=saved.get("artifacts", {}),
                    error=(
                        "Restore completed artifacts or use a new output "
                        f"directory: {error}"
                    ),
                )
        else:
            pending.append(job)
        _write(Path(job["config"]), yaml.safe_dump(job["child"], sort_keys=False))

    def save():
        _write(status_path, json.dumps(state, indent=2, allow_nan=False) + "\n")

    save()
    if not execute:
        return state
    state["status"] = "running"
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        remaining, active = iter(pending), {}

        def dispatch(gpu):
            job = next(remaining, None)
            if job is not None:
                job.update(status="running", gpu=gpu)
                save()
                active[pool.submit(_execute, job, gpu, root)] = (job, gpu)

        for gpu in gpus:
            dispatch(gpu)
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                job, gpu = active.pop(future)
                job.update(future.result())
                save()
                dispatch(gpu)
    state["status"] = (
        "completed" if all(job["status"] == "completed" for job in jobs) else "failed"
    )
    save()
    return state
