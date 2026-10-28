"""Prepare or resume the fixed-mask classifier study without opening test endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

from hccr.artifacts.checkpoints import file_digest
from hccr.models.artifacts import read_checkpoint_metadata
from hccr.ops.gpu_queue import run_gpu_queue

if __package__:
    from scripts.prepare_classifier_stage import prepare_classifier_stage
else:
    from prepare_classifier_stage import prepare_classifier_stage

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _study_lock(output: Path):
    """Let the OS release ownership after a crash; the lock file can remain."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "study.lock").open("a+b") as stream:
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def teacher_identity(run: Path, manifest_digest: str) -> dict:
    """A selected checkpoint may precede epoch 20; curves prove run completion."""
    run = run.resolve()
    meta = read_checkpoint_metadata(run)
    model, training = meta["model"], meta["training"]
    if (
        meta.get("manifest_digest") != manifest_digest
        or model.get("classifier_structure", "dense") != "dense"
        or model.get("classification_head") != "cosface"
        or model.get("name") != "efficient_hccr"
        or model.get("num_classes") != 3755
        or model.get("embedding_dim") != 320
        or model.get("logit_scale") != 32.0
        or model.get("angular_margin") != 0.1
        or training.get("freeze_backbone", False)
    ):
        raise ValueError(
            "Study requires paired official EfficientHCCR dense CosFace teachers"
        )
    protocol = {
        "resolved_epochs": 20,
        "optimizer": "adamw",
        "base_learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 256,
        "reference_batch_size": 256,
        "optimizer_step_policy": "configured_epochs",
    }
    if (
        any(training.get(key) != value for key, value in protocol.items())
        or training.get("max_train_samples") is not None
        or training.get("overfit_check", False)
    ):
        raise ValueError("Teacher differs from the locked 20-epoch full-fit recipe")
    metrics = _read(run / "metrics.json")
    if any(
        metrics.get(name, {}).get("status") != "not_run" for name in ("test", "icdar")
    ):
        raise ValueError("Teacher must leave both test endpoints unopened")
    epochs = _read(run / "curves.json")["epochs"]
    resolved = training["resolved_epochs"]
    if (
        not epochs
        or int(epochs[-1]["epoch"]) != resolved
        or epochs[-1]["optimizer_step_end"] != training["total_optimizer_steps"]
        or len(epochs) != resolved
    ):
        raise ValueError("Teacher run did not finish its declared fitting budget")
    labels = _read(run / "labels.json").get("labels", {})
    if len(labels) != model["num_classes"]:
        raise ValueError("Teacher label count differs from its classifier")
    return {
        "run": str(run),
        "seed": training["seed"],
        "classes": model["num_classes"],
        "embedding_dim": model["embedding_dim"],
        "selected_epoch": meta["selection"]["epoch"],
        "completed_epochs": resolved,
        "model": model,
        "teacher_protocol": protocol,
        "digests": {
            name: file_digest(run / name)
            for name in (
                "checkpoint.pt",
                "checkpoint_metadata.json",
                "labels.json",
                "metrics.json",
                "curves.json",
            )
        },
    }


def prepare_study(
    teacher_runs: list[Path],
    manifest: Path,
    output: Path,
    *,
    rank: int = 64,
    k: int = 500,
    precision: str = "float32",
    gate_controls: bool = False,
    allocation_controls: bool = False,
    resume: bool = False,
) -> dict:
    manifest, output = manifest.resolve(), output.resolve()
    digest = file_digest(manifest)
    teachers = sorted(
        (teacher_identity(run, digest) for run in teacher_runs), key=lambda t: t["seed"]
    )
    seeds = [teacher["seed"] for teacher in teachers]
    if seeds != [7, 17, 29]:
        raise ValueError("Primary study requires exactly teacher seeds 7, 17 and 29")
    if len({(t["classes"], t["embedding_dim"]) for t in teachers}) != 1:
        raise ValueError("Teacher architectures/vocabularies differ")
    if len({json.dumps(t["model"], sort_keys=True) for t in teachers}) != 1:
        raise ValueError("Teacher model architecture/objective settings differ")
    if len({t["digests"]["labels.json"] for t in teachers}) != 1:
        raise ValueError("Teacher class ordering differs")
    settings = {
        "manifest": str(manifest),
        "manifest_digest": digest,
        "teachers": teachers,
        "rank": rank,
        "k": k,
        "precision": precision,
        "gate_controls": gate_controls,
        "allocation_controls": allocation_controls,
        "dataset_backend": "lmdb",
        "test_endpoints": "unopened",
    }
    identity = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    status_path = output / "study.json"
    prior = _read(status_path) if status_path.exists() else None
    if prior and not resume:
        raise FileExistsError("Study exists; use --resume")
    if prior and prior.get("identity") != identity:
        raise ValueError("Study inputs/settings changed; use a new output directory")
    for path, expected in (prior or {}).get("artifacts", {}).items():
        if not Path(path).is_file() or file_digest(Path(path)) != expected:
            raise ValueError(f"Study output changed or disappeared: {path}")
    configurations = []
    for teacher in teachers:
        config = prepare_classifier_stage(
            "pilots",
            Path(teacher["run"]),
            manifest,
            rank=rank,
            near_budget_k=k,
            runtime_profile="gpu",
            precision=precision,
        )
        config["output_dir"] = str(output / "pilot-runs")
        path = output / "configs" / f"seed-{teacher['seed']}-pilots.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configurations.append(str(path))
    state = prior or {
        "schema_version": 1,
        "identity": identity,
        "settings": settings,
        "status": "planned",
        "artifacts": {},
        "completed_stages": [],
    }
    state["pilot_configs"] = configurations
    state["planned_jobs"] = {
        "pilots": 9,
        "final": 15 + (15 if gate_controls else 0) + (18 if allocation_controls else 0),
    }
    _write(status_path, state)
    return state


def completed_pilots(queue: dict, rank: int) -> dict[int, dict[str, Path]]:
    if queue["status"] != "completed":
        raise RuntimeError(
            "Pilot queue is incomplete; allocation/final fitting is blocked"
        )
    result: dict[int, dict[str, Path]] = {}
    for job in queue["jobs"]:
        if job["status"] != "completed":
            raise RuntimeError("A pilot job is incomplete")
        child = _read(Path(job["child_status"]))
        records = [r for r in child["records"] if r["status"] == "completed"]
        if len(records) != 1:
            raise ValueError("Pilot has no unique completed artifact")
        run = Path(records[0]["run_path"]).resolve()
        if not run.is_relative_to(Path(job["child"]["output_dir"]).resolve()):
            raise ValueError("Pilot run escapes its queued output directory")
        meta = read_checkpoint_metadata(run)
        seed = meta["training"]["seed"]
        model = meta["model"]
        role = (
            "dense"
            if model["classifier_structure"] == "dense"
            else ("low_rank" if model["classifier_rank"] == rank else "near_budget")
        )
        by_role = result.setdefault(seed, {})
        if role in by_role:
            raise ValueError("Duplicate fitted pilot role")
        by_role[role] = run
    if sorted(result) != [7, 17, 29] or any(
        set(runs) != {"dense", "low_rank", "near_budget"} for runs in result.values()
    ):
        raise ValueError("Missing one or more paired pilot references")
    return result


def _command(script: str, arguments: list[str], output: Path) -> None:
    with (output / "flow.log").open("a", encoding="utf-8") as stream:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *arguments],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _remember(state: dict, paths: list[Path]) -> None:
    state["artifacts"].update({str(path): file_digest(path) for path in paths})


def run_study(state: dict, output: Path, gpus: list[int], through: str) -> dict:
    settings = state["settings"]
    queue = run_gpu_queue(
        [Path(p) for p in state["pilot_configs"]],
        gpus,
        output / "pilot-queue",
        execute=True,
        resume=(output / "pilot-queue/status.json").exists(),
    )
    pilots = completed_pilots(queue, settings["rank"])
    state["status"] = "pilots_completed"
    if "pilots" not in state["completed_stages"]:
        state["completed_stages"].append("pilots")
    _write(output / "study.json", state)
    if through == "pilots":
        return state
    configs = []
    for seed, runs in sorted(pilots.items()):
        folder = output / f"seed-{seed}"
        folder.mkdir(exist_ok=True)
        dense, low = folder / "dense.jsonl", folder / "low-rank.jsonl"
        frequency, directional = folder / "frequency.json", folder / "directional.json"
        export_paths = [dense, low, frequency, directional]
        if not all(str(p) in state["artifacts"] for p in export_paths):
            common = [
                "--manifest",
                settings["manifest"],
                "--device",
                f"cuda:{gpus[0]}",
                "--storage-backend",
                "lmdb",
                "--num-workers",
                "8",
            ]
            _command(
                "export_calibration_predictions.py",
                [
                    "--run-dir",
                    str(runs["dense"]),
                    "--output",
                    str(dense),
                    "--fit-frequency-output",
                    str(frequency),
                    "--directional-output",
                    str(directional),
                    "--directional-rank",
                    str(settings["rank"]),
                    *common,
                ],
                output,
            )
            _command(
                "export_calibration_predictions.py",
                [
                    "--run-dir",
                    str(runs["low_rank"]),
                    "--output",
                    str(low),
                    *common,
                ],
                output,
            )
            _remember(state, export_paths)
            _write(output / "study.json", state)
        masks = [("penalty", 7)]
        if settings["allocation_controls"]:
            masks += [("random", s) for s in (7, 17, 29)]
            masks += [
                (method, 7)
                for method in ("frequency", "raw_difficulty", "directional_svd")
            ]
        for method, mask_seed in masks:
            mask = folder / f"{method}-{mask_seed}.json"
            if str(mask) not in state["artifacts"]:
                args = [
                    "--dense-predictions",
                    str(dense),
                    "--low-rank-predictions",
                    str(low),
                    "--k",
                    str(settings["k"]),
                    "--method",
                    method,
                    "--seed",
                    str(mask_seed),
                    "--output",
                    str(mask),
                ]
                if method in {"frequency", "directional_svd"}:
                    args += [
                        "--control-scores",
                        str(frequency if method == "frequency" else directional),
                    ]
                _command("select_residual_classes.py", args, output)
                _remember(state, [mask])
                _write(output / "study.json", state)
            if method == "penalty":
                report = folder / "h1-validation.json"
                if str(report) not in state["artifacts"]:
                    _command(
                        "analyze_compression_sensitivity.py",
                        [
                            "--dense-run",
                            str(runs["dense"]),
                            "--low-rank-run",
                            str(runs["low_rank"]),
                            "--mask",
                            str(mask),
                            "--manifest",
                            settings["manifest"],
                            "--output",
                            str(report),
                            "--device",
                            f"cuda:{gpus[0]}",
                            "--storage-backend",
                            "lmdb",
                            "--num-workers",
                            "8",
                        ],
                        output,
                    )
                    _remember(state, [report])
            if through == "allocation":
                continue
            config = prepare_classifier_stage(
                "residuals",
                runs["low_rank"],
                Path(settings["manifest"]),
                mask=mask,
                dense_pilot_run=runs["dense"],
                near_budget_pilot_run=runs["near_budget"],
                runtime_profile="gpu",
                precision=settings["precision"],
                include_gate_controls=settings["gate_controls"] and method == "penalty",
            )
            if method != "penalty":
                config["variants"] = [
                    v for v in config["variants"] if v["name"] == "static_residual"
                ]
            config["output_dir"] = str(output / "final-runs")
            path = folder / f"final-{method}-{mask_seed}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            configs.append(path)
        _write(output / "study.json", state)
    state["status"] = "allocation_completed"
    if "allocation" not in state["completed_stages"]:
        state["completed_stages"].append("allocation")
    if configs:
        final_queue = run_gpu_queue(
            configs,
            gpus,
            output / "final-queue",
            execute=through == "final",
            resume=(output / "final-queue/status.json").exists(),
        )
        state["status"] = (
            "completed" if final_queue["status"] == "completed" else "final_planned"
        )
        if final_queue["status"] == "failed":
            raise RuntimeError("Final queue failed; inspect its status and logs")
        state["final_configs"] = [str(p) for p in configs]
    _write(output / "study.json", state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument(
        "--precision", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--gate-controls", action="store_true")
    parser.add_argument("--allocation-controls", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--through",
        choices=("pilots", "allocation", "final-plan", "final"),
        default="pilots",
    )
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    with _study_lock(output):
        state = prepare_study(
            args.teacher_run,
            args.manifest,
            output,
            rank=args.rank,
            k=args.k,
            precision=args.precision,
            gate_controls=args.gate_controls,
            allocation_controls=args.allocation_controls,
            resume=args.resume,
        )
        if args.execute:
            try:
                state = run_study(state, output, args.gpus, args.through)
            except Exception as error:
                state.update(status="failed", error=f"{type(error).__name__}: {error}")
                _write(output / "study.json", state)
                raise
    print(
        json.dumps(
            {
                "status": state["status"],
                "jobs": state["planned_jobs"],
                "status_file": str(output / "study.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
