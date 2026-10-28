"""Prepare the audited main queue: teachers, objectives, then other backbones."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hccr.ops.gpu_queue import run_gpu_queue
from hccr.ops.vast_bootstrap import configure_runtime, hardware_report


def prepare_official_classifier_queue(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    runtime_dir = args.runtime_dir.resolve()
    hardware = hardware_report()
    configs = []
    reports = {}
    for group in ("teacher", "objectives", "backbones"):
        output = runtime_dir / f"{group}.yaml"
        reports[group] = configure_runtime(
            root / "configs" / "experiment" / f"official_classifier_{group}.yaml",
            output,
            args.manifest.resolve(),
            hardware,
            preset="bf16",
            concurrent_jobs=len(args.gpus),
            cache_backend="auto",
            classes=3755,
        )
        configs.append(output)
    state = run_gpu_queue(
        configs,
        args.gpus,
        args.output_dir,
        project_root=root,
        stage_configs=True,
    )
    jobs = state["jobs"]
    if len(jobs) != 30:
        raise RuntimeError("Expected 30 unique main-study jobs")
    if any(job["identity"]["training"]["epochs"] != 20 for job in jobs):
        raise RuntimeError("Main-study queue must use the declared 20-epoch recipe")
    first = [job["identity"]["training"] for job in jobs[:3]]
    if any(
        job["model"] != "efficient_hccr" or job["classification_head"] != "cosface"
        for job in first
    ) or [job["seed"] for job in first] != [7, 17, 29]:
        raise RuntimeError("Teacher seed priority is incorrect")
    report = {
        "configs": [str(path) for path in configs],
        "runtime": reports,
        "code_digest": state["code_digest"],
        "queue_output": str(args.output_dir.resolve()),
        "jobs": len(jobs),
        "stage_configs": True,
        "stage_sizes": [3, 9, 18],
        "ordered_fingerprints": [job["fingerprint"] for job in jobs],
        "training_started": False,
    }
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    if not args.manifest.is_file() or not args.gpus:
        parser.error("An existing audited manifest and at least one GPU are required")
    prepare_official_classifier_queue(args)
