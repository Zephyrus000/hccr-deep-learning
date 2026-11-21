"""Fill free GPUs with ready jobs from the 3,755-class CosFace follow-up panel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from run_hccr_compression import MANIFEST_SHA, ROOT

SEEDS = (7, 17, 29)
EXTRA_SEEDS = (41, 53)
DRIVER = ROOT / "scripts/run_hccr_compression.py"


def plan(inputs, output, rank64):
    jobs = {}

    def add(key, action, seed=None, deps=(), **options):
        jobs[key] = {
            "action": action,
            "seed": seed,
            "dependencies": list(deps),
            "options": options,
            "status": "pending",
        }

    def fit(variant, seed, source=None, dep=None):
        key = f"{variant}-{seed}"
        add(key, "fit", seed, (dep,) if dep else (), variant=variant, source=source)
        return key

    for seed in SEEDS:
        fit("embedding104", seed, inputs / f"pilot104-{seed}")
    for seed in EXTRA_SEEDS:
        fit("teacher", seed)
    for seed in SEEDS:
        pilot = f"pilot160-{seed}"
        add(pilot, "pilot", seed, variant="rank160", source=inputs / f"teacher-{seed}")
        fit("rank160", seed, dep=pilot)
    for seed in SEEDS:
        fit("embedding64", seed, inputs / f"pilot64-{seed}")
    for seed in EXTRA_SEEDS:
        for variant in ("dense", "rank104"):
            pilot = f"pilot-{variant}-{seed}"
            add(pilot, "pilot", seed, (f"teacher-{seed}",), variant=variant)
            fit(variant, seed, dep=pilot)
    for key, job in list(jobs.items()):
        if job["action"] == "fit" and job["options"]["variant"] != "teacher":
            add(f"evaluate-{key}", "evaluate", job["seed"], (key,))
    add("rank64-29", "external", 29, source=rank64 / "experiments/result.json")
    add("evaluate-rank64-29", "evaluate", 29, ("rank64-29",))
    for seed in SEEDS:
        add(f"svd104-{seed}", "svd", seed, source=inputs / f"dense-{seed}")
    for variant in ("dense", "rank104", "rank64"):
        for device in ("cuda", "cpu"):
            add(
                f"deploy-{variant}-{device}",
                "deploy",
                7,
                source=inputs / f"{variant}-7",
                device=device,
            )
    deps = tuple(
        f"evaluate-{variant}-{seed}"
        for variant in ("dense", "rank104")
        for seed in EXTRA_SEEDS
    )
    for split in ("test", "icdar"):
        add(
            f"stats-{split}",
            "stats",
            deps=deps,
            split=split,
            source=inputs,
            reports=output,
        )
    return json.loads(json.dumps(jobs, default=str))


def run_directory(folder):
    candidates = sorted(folder.glob("*/checkpoint_metadata.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one checkpoint in {folder}")
    run = candidates[0].parent
    metadata = json.loads(candidates[0].read_bytes())
    curves = json.loads((run / "curves.json").read_bytes())["epochs"]
    if (
        not (run / "checkpoint.pt").is_file()
        or metadata["manifest_digest"] != MANIFEST_SHA
        or [item["epoch"] for item in curves] != list(range(1, 101))
        or metadata["training"]["precision"]["resolved"] != "float32"
    ):
        raise ValueError(f"Incomplete or incompatible FP32 checkpoint in {run}")
    return str(run)


def idle_gpus():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        timeout=10,
        text=True,
    )
    cards = [tuple(map(int, line.split(","))) for line in result.stdout.splitlines()]
    return {
        index for index, free, total in cards if free >= 23100 and total - free < 768
    }


def command(key, job, jobs, args, cpu):
    action, options = job["action"], job["options"]
    target = args.output / key
    result = [sys.executable, str(DRIVER), action]
    result += [
        "--output",
        str(
            target if action not in ("deploy", "stats") else target.with_suffix(".json")
        ),
    ]
    if action != "stats":
        result += ["--manifest", str(args.manifest)]
    source = options.get("source")
    if job["dependencies"] and action != "stats":
        source = jobs[job["dependencies"][0]]["run"]
    if action in ("fit", "pilot"):
        result += ["--variant", options["variant"]]
    if action == "fit":
        result += ["--seed", str(job["seed"])]
        if source:
            result += ["--initialize-from", source]
    elif action == "pilot":
        result += ["--teacher", source, "--runner", str(args.runner)]
    elif action == "stats":
        split = options["split"]
        result += ["--split", split]
        for variant in ("dense", "rank104"):
            paths = [
                Path(source) / f"{variant}-{seed}" / f"{split}.json" for seed in SEEDS
            ]
            paths += [
                Path(options["reports"])
                / f"evaluate-{variant}-{seed}"
                / f"{split}.json"
                for seed in EXTRA_SEEDS
            ]
            result += [f"--{variant}", *map(str, paths)]
    else:
        result += ["--run", source]
        if action == "deploy":
            result += ["--device", options["device"]]
            if options["device"] == "cpu":
                result += ["--core", str(cpu)]
    return result


def eligible(jobs):
    return [
        key
        for key, job in jobs.items()
        if job["status"] == "pending"
        and all(jobs[dep]["status"] == "completed" for dep in job["dependencies"])
        and job["action"] != "external"
    ]


def supervise(args):
    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = args.output / "queue.json"
        if report.exists():
            raise FileExistsError(f"Queue already exists at {report}")
        jobs = plan(args.inputs, args.output, args.rank64)
        queue = {"status": "waiting_for_lmdb", "pid": os.getpid(), "jobs": jobs}
        running = {}
        logs = args.output / "logs"
        logs.mkdir()

        def publish():
            queue["updated_at_utc"] = datetime.now(UTC).isoformat()
            temp = report.with_suffix(".tmp")
            temp.write_text(json.dumps(queue, indent=2) + "\n")
            temp.replace(report)

        while True:
            disk = args.rank64 / ".ops"
            status = json.loads((disk / "status.json").read_bytes())
            phase = status["status"]
            external = jobs["rank64-29"]
            if phase == "completed" and external["status"] != "completed":
                external.update(
                    status="completed", run=run_directory(args.rank64 / "experiments")
                )
            elif phase.startswith("failed_"):
                external.update(status="failed", error=phase)
            elif phase == "training":
                external.update(
                    status="running", gpu=int(status["gpu"]), pid=status["pid"]
                )
            for key, (child, _) in list(running.items()):
                rc = child.poll()
                if rc is None:
                    continue
                entry = jobs[key]
                entry.update(status="completed" if rc == 0 else "failed", exit_code=rc)
                if rc == 0 and entry["action"] in ("fit", "pilot"):
                    try:
                        entry["run"] = run_directory(args.output / key)
                    except (OSError, KeyError, ValueError) as error:
                        entry.update(status="failed", error=str(error))
                del running[key]
            for entry in jobs.values():
                if entry["status"] == "pending" and any(
                    jobs[dep]["status"] in ("failed", "blocked")
                    for dep in entry["dependencies"]
                ):
                    entry.update(status="blocked", error="required job failed")
            if all(
                job["status"] in ("completed", "failed", "blocked")
                for job in jobs.values()
            ):
                queue["status"] = (
                    "completed"
                    if all(job["status"] == "completed" for job in jobs.values())
                    else "completed_with_errors"
                )
                publish()
                return
            if not args.manifest.with_name("images.lmdb").is_file() or phase not in (
                "preflight",
                "training",
                "completed",
                "failed_training",
                "failed_preflight",
            ):
                if phase.startswith("failed_"):
                    queue.update(status="blocked", error=phase)
                    publish()
                    return
                publish()
                time.sleep(3)
                continue
            queue["status"] = "active"
            try:
                available = idle_gpus() - {gpu for _, gpu in running.values()}
                queue.pop("gpu_probe_error", None)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                available = set()
                queue["gpu_probe_error"] = type(error).__name__
            if external["status"] in ("pending", "running"):
                available.discard(int(status.get("gpu") or 0))
            cpu_running = any(gpu is None for _, gpu in running.values())
            ready = sorted(
                eligible(jobs),
                key=lambda key: jobs[key]["action"] in ("svd", "deploy"),
            )
            for key in ready:
                entry = jobs[key]
                cpu_job = (
                    entry["action"] == "stats"
                    or entry["options"].get("device") == "cpu"
                )
                if cpu_job:
                    if cpu_running or (
                        entry["action"] == "deploy"
                        and (
                            external["status"] == "running"
                            or any(gpu is not None for _, gpu in running.values())
                            or any(jobs[item]["action"] == "fit" for item in ready)
                        )
                    ):
                        continue
                    gpu = None
                elif available:
                    gpu = min(available)
                else:
                    continue
                cmd = command(key, entry, jobs, args, min(os.sched_getaffinity(0)))
                env = {
                    **os.environ,
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": str(gpu) if gpu is not None else "",
                    "NVIDIA_TF32_OVERRIDE": "0",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                    "OMP_NUM_THREADS": "2",
                    "MKL_NUM_THREADS": "2",
                    "PYTHONUNBUFFERED": "1",
                }
                log = logs / f"{key}.log"
                try:
                    with log.open("ab") as stream:
                        child = subprocess.Popen(
                            cmd,
                            cwd=ROOT,
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                        )
                except OSError as error:
                    entry.update(status="failed", error=str(error))
                    continue
                entry.update(status="running", gpu=gpu, pid=child.pid, log=str(log))
                running[key] = (child, gpu)
                if gpu is not None:
                    available.remove(gpu)
                else:
                    cpu_running = True
            publish()
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "inputs", "output", "rank64", "runner"):
        parser.add_argument(
            f"--{name}", type=lambda value: Path(value).resolve(), required=True
        )
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(plan(args.inputs, args.output, args.rank64), indent=2))
    else:
        supervise(args)


if __name__ == "__main__":
    main()
