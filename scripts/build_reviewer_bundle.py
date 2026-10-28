"""Build a portable, verified reviewer snapshot without moving working artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def rows_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def slug(value):
    result = re.sub(r"[^a-z0-9-]+", "-", str(value).lower()).strip("-")
    if not result:
        raise ValueError("Empty artifact name")
    return result


def variant_name(row):
    variant = row["variant"]
    replacements = {
        "efficient_hccr": "efficienthccrnet",
        "mobilenet_v3_small": "mobilenetv3-small", "mobilenetv4_conv_small": "mobilenetv4-conv-small",
        "shufflenet_v2_x1_0": "shufflenetv2-1x", "repvit_m0_9": "repvit-m0-9",
        "uniform_low_rank": "uniform-low-rank-r064", "uniform_near_budget": "uniform-low-rank-r104",
    }
    for old, new in replacements.items():
        variant = variant.replace(old, new)
    variant = variant.replace("__final-penalty-7", "")
    variant = variant.replace("__final-random-", "-random-mask-")
    variant = variant.replace("__final-frequency-7", "-frequency-mask")
    variant = variant.replace("__final-raw_difficulty-7", "-raw-difficulty-mask")
    variant = variant.replace("__final-directional_svd-7", "-directional-svd-mask")
    return slug(variant)


@dataclass(frozen=True)
class FileEntry:
    source: Path
    target: str
    role: str
    run_id: str = ""
    expected_sha256: str | None = None


class BundlePlan:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.entries = {}

    def add(self, source, target, role, run_id="", expected_sha256=None):
        source = Path(source).resolve()
        if not source.is_relative_to(self.workspace):
            raise ValueError(f"Source outside workspace: {source}")
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Source must be a regular existing file: {source}")
        parts = {part.lower() for part in source.relative_to(self.workspace).parts}
        if (parts & {".ssh", "__pycache__", ".git", ".venv"}
                or any(part.startswith(".env") for part in parts)
                or source.suffix.lower() in {".lmdb", ".zip", ".tar", ".pem", ".key", ".partial"}
                or source.name.startswith(("id_rsa", "id_ed25519"))):
            raise ValueError(f"Not a reviewer artifact: {source.name}")
        relative = PurePosixPath(str(target).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or ":" in str(relative):
            raise ValueError("Bundle target must stay inside its release")
        key = str(relative)
        if key in self.entries:
            existing = self.entries[key]
            if existing.source != source:
                raise ValueError(f"Target collision: {key}")
            if existing.expected_sha256 and expected_sha256 and existing.expected_sha256 != expected_sha256:
                raise ValueError(f"Conflicting frozen digests: {key}")
            if expected_sha256 and not existing.expected_sha256:
                self.entries[key] = FileEntry(source, key, existing.role, existing.run_id, expected_sha256)
            return key
        self.entries[key] = FileEntry(source, key, role, run_id, expected_sha256)
        return key

    def optional(self, source, target, role, run_id="", expected_sha256=None):
        if Path(source).is_file():
            return self.add(source, target, role, run_id, expected_sha256)
        return None

    def curated_tree(self, source, target, role, suffixes=(".json", ".csv", ".md", ".png", ".pdf", ".tex")):
        source = Path(source)
        if not source.is_dir():
            return
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes and "__pycache__" not in path.parts:
                self.add(path, PurePosixPath(target) / path.relative_to(source).as_posix(), role)


def collect_runs(workspace):
    downloads = workspace / "artifacts/downloads/vastai-20261026"
    rows = rows_csv(downloads / "consolidated/heldout-results.csv")
    if len(rows) != 78 or len({r["fingerprint"] for r in rows}) != 78:
        raise ValueError("Expected all 78 baseline/final runs")
    for row in rows:
        row["group"] = "baselines" if row["group"] == "baseline_objective" else "final"
        row["run"] = str(Path(row["training_metrics"]).parent)
    pilot_root = downloads / "57418784/primary/repo/experiments/classifier-study/pilot-queue/jobs"
    for path in sorted(pilot_root.rglob("checkpoint_metadata.json")):
        meta = read_json(path)
        model, training = meta["model"], meta["training"]
        variant = "dense" if model["classifier_structure"] == "dense" else f"low-rank-r{model['classifier_rank']:03d}"
        metrics = read_json(path.with_name("metrics.json"))
        rows.append({"group": "pilots", "variant": variant, "seed": str(training["seed"]),
                     "fingerprint": path.parts[path.parts.index("jobs") + 1],
                     "run": str(path.parent), "training_metrics": str(path.with_name("metrics.json")),
                     "checkpoint_sha256": sha256(path.with_name("checkpoint.pt")),
                     "validation_top1": metrics["validation"]["selected"]["top1"]})
    if len(rows) != 87:
        raise ValueError("Expected 30 baselines, 9 pilots and 48 final runs")
    counts = {g: sum(r["group"] == g for r in rows) for g in ("baselines", "pilots", "final")}
    if counts != {"baselines": 30, "pilots": 9, "final": 48}:
        raise ValueError(f"Unexpected run coverage: {counts}")
    return rows


def add_calibration_predictions(plan, study, catalog):
    records = []
    for seed in (7, 17, 29):
        for filename, variant, role in (("dense", "dense", "dense"), ("low-rank", "low-rank-r064", "low_rank")):
            run_id = f"pilots/{variant}/seed-{seed:02d}"
            run = next(row for row in catalog if row["run_id"] == run_id)
            source = study / f"seed-{seed}" / f"{filename}.jsonl"
            with source.open(encoding="utf-8-sig") as stream:
                header = json.loads(next(stream))
            checkpoint = plan.entries[run["checkpoint"]]
            if (header.get("type") != "metadata" or header.get("split") != "calibration"
                    or header.get("official_partition") != "train" or header.get("pilot_seed") != seed
                    or header.get("model_role") != role
                    or header.get("classifier_rank") != (64 if role == "low_rank" else None)
                    or header.get("checkpoint_digest") != checkpoint.expected_sha256):
                raise ValueError(f"Calibration prediction checkpoint/protocol mismatch: {source}")
            target = f"inference/calibration/{variant}/seed-{seed:02d}/predictions.jsonl"
            plan.add(source, target, "pilot_calibration_predictions_fp32", run_id)
            records.append({"run_id": run_id, "predictions": target,
                            "checkpoint": run["checkpoint"], "checkpoint_sha256": header["checkpoint_digest"],
                            "precision": "float32", "split": "calibration", "official_partition": "train",
                            "manifest_digest": header["manifest_digest"]})
    return records


def make_plan(workspace):
    workspace = Path(workspace).resolve()
    plan = BundlePlan(workspace)
    runs = collect_runs(workspace)
    catalog = []
    for row in runs:
        run = Path(row["run"])
        meta = read_json(run / "checkpoint_metadata.json")
        key = f"{row['group']}/{variant_name(row)}/seed-{int(row['seed']):02d}"
        profile = read_json(run / "resource_profile.json")
        info = {"run_id": key, "group": row["group"], "variant": row["variant"],
                "seed": int(row["seed"]), "fingerprint": row["fingerprint"],
                "whole_model_parameters": profile["parameter_count"],
                "configured_epochs": meta["training"]["configured_epochs"],
                "validation_top1": float(row["validation_top1"]),
                "test_top1": float(row["test_top1"]) if row.get("test_top1") else None,
                "icdar_top1": float(row["icdar_top1"]) if row.get("icdar_top1") else None,
                "inference_precision": "not_evaluated" if row["group"] == "pilots" else "bfloat16_backbone_fp32_normalized_head",
                "checkpoint": f"checkpoints/{key}/checkpoint.pt",
                "training_metrics": f"metrics/{key}/metrics.json",
                "readme": f"runs/{key}/README.md"}
        for name in ("checkpoint.pt", "checkpoint_metadata.json", "labels.json"):
            plan.add(run / name, f"checkpoints/{key}/{name}", "checkpoint", key,
                     row["checkpoint_sha256"] if name == "checkpoint.pt" else None)
        for name in ("metrics.json", "curves.json", "config.json", "metadata.json", "initialization.json",
                     "batch_training_plan.json", "input_pipeline.json", "validation_stability.json",
                     "run.log", "learning_curves.png"):
            plan.optional(run / name, f"metrics/{key}/{name}", "training", key)
        plan.add(run / "resource_profile.json", f"benchmarks/training-gpu-fp32/{key}/resource_profile.json",
                 "provisional_training_gpu_profile", key)
        for split, endpoint in (("test", "hwdb-test"), ("icdar", "icdar-2013")):
            if not row.get(split + "_metrics"):
                continue
            source = Path(row[split + "_metrics"])
            report = read_json(source)
            if (report["checkpoint_sha256"] != row["checkpoint_sha256"]
                    or report["precision"] != "bfloat16"
                    or report["samples"] != {"test": 533675, "icdar": 224419}[split]
                    or report["test_used_for_selection"] is not False):
                raise ValueError(f"Invalid frozen endpoint report: {key}/{split}")
            target = f"inference/recognition-bf16/{key}/{endpoint}"
            plan.add(source, target + "/metrics.json", "recognition_inference", key)
            plan.add(source.with_name("predictions.npz"), target + "/predictions.npz",
                     "recognition_inference", key, report["predictions_sha256"])
            info[split + "_inference"] = target + "/metrics.json"
        catalog.append(info)

    artifacts = workspace / "artifacts"
    downloads = artifacts / "downloads/vastai-20261026"
    post = artifacts / "local-post-training/20261026-retry-af8b8078/results"
    verified = artifacts / "local-post-training/20261026-onnx-repair-verified/results"
    remaining = artifacts / "local-post-training/remaining-work-20261026"
    study = downloads / "57418784/primary/repo/experiments/classifier-study"
    plan.calibration_map = add_calibration_predictions(plan, study, catalog)
    for source, target in ((study / "study.json", "study.json"),
                           (study / "pilot-queue/status.json", "pilot-queue-status.json"),
                           (study / "final-queue/status.json", "final-queue-status.json")):
        plan.add(source, f"provenance/study/{target}", "study_ledger")
    plan.curated_tree(study / "configs", "reproduce/configs/pilots", "study_configuration", (".yaml", ".yml"))
    core = next(downloads.rglob("classifier-core-heldout-bf16"))
    for split, endpoint in (("test", "hwdb-test"), ("icdar", "icdar-2013")):
        plan.add(core / f"{split}-index-gpu-0.csv", f"provenance/dataset/{endpoint}-index.csv", "endpoint_index")
    for name in ("final-study-artifacts.json", "heldout-coverage-audit.json"):
        plan.add(downloads / "consolidated" / name, f"provenance/original-ledgers/{name}", "provenance")
    for name in ("classifier-baselines-heldout-bf16", "classifier-core-heldout-bf16", "classifier-ablations-heldout-bf16"):
        folder = next(downloads.rglob(name))
        for leaf in ("protocol.json", "summary.json"):
            plan.add(folder / leaf, f"provenance/recognition/{name}/{leaf}", "recognition_protocol")
    for seed in (7, 17, 29):
        folder = study / f"seed-{seed}"
        for name in ("h1-validation.json", "penalty-7.json", "random-7.json", "random-17.json", "random-29.json",
                     "frequency-7.json", "raw_difficulty-7.json", "directional_svd-7.json", "frequency.json", "directional.json"):
            plan.optional(folder / name, f"analysis/allocation/seed-{seed:02d}/{name}", "allocation")
        for path in sorted(folder.glob("*.yaml")):
            plan.add(path, f"reproduce/configs/final/seed-{seed:02d}/{path.name}", "study_configuration")
    for name in ("paired-comparisons.json", "paired-comparisons.csv", "prediction-summary.json",
                 "h1-existing-and-mask-stability.json", "gate-training-trajectories.csv", "gate-trajectories.png",
                 "telemetry-scope.json", "local-data-reconciliation.json", "scope-and-remaining.json"):
        plan.optional(post / name, f"analysis/summary/{name}", "analysis")
    for folder in ("gate-counterfactual", "precision-and-equivalence"):
        plan.curated_tree(post / folder, f"analysis/{folder}", "analysis")
    precision = remaining / "precision"
    for name in ("completion.json", "claim-evidence-review.md"):
        plan.optional(precision / name, f"analysis/precision/{name}", "precision")
    for name in ("run-20261026b", "stem-20261026b"):
        plan.curated_tree(precision / name, f"analysis/precision/{name}", "precision")
    evidence = remaining / "evidence"
    plan.curated_tree(evidence / "results", "paper/tables-and-figures", "paper_evidence")
    for name in ("qa-receipt.json", "deployment-endpoints-snapshot.json"):
        plan.optional(evidence / name, f"provenance/evidence/{name}", "provenance")
    for name in ("research_proposal_revised.tex", "current_study_results.md"):
        plan.optional(workspace / "docs" / name, f"paper/{name}", "paper")

    # Keep the measured native CPU blocks and the repaired ONNX blocks separate.
    for label, source, include_onnx in (("native-pytorch", post, False), ("onnx-verified", verified, True)):
        plan.add(source / "cpu-benchmark-protocol.json", f"benchmarks/cpu/{label}/protocol.json", "cpu_protocol")
        plan.add(source / "cpu-benchmarks.csv", f"provenance/original-cpu-tables/{label}.csv", "historical_cpu_table")
        for path in sorted((source / "cpu-raw").glob("*.json")):
            if ("-onnx_" in path.name) == include_onnx:
                plan.add(path, f"benchmarks/cpu/{label}/raw/{path.name}", "cpu_raw")
    plan.optional(verified / "verification.json", "provenance/onnx/verification.json", "verification")
    plan.curated_tree(verified / "onnx", "deployment/onnx", "deployment", (".onnx", ".json"))
    for seed in (7, 17, 29):
        for kind in ("fp32", "int8"):
            name = f"seed-{seed}-dense_reference-head-{kind}.pt"
            plan.add(post / "native-exports" / name, f"deployment/native-pytorch/seed-{seed:02d}/head-{kind}.pt", "deployment")
    plan.add(post / "deployment-artifacts.json", "provenance/deployment/native-export-ledger.json", "provenance")
    plan.add(post / "environment.json", "provenance/runtime/native-environment.json", "runtime")

    deployment = remaining / "deployment-heldouts"
    frozen_protocol = read_json(deployment / "protocol.json")
    protocol_hash = sha256(deployment / "protocol.json")
    frozen_metrics = {str((workspace / row["path"]).resolve()): row["sha256"]
                      for row in read_json(evidence / "deployment-endpoints-snapshot.json")["completed_receipts"]}
    frozen_artifacts = {(item["seed"], item["family"]): item for item in frozen_protocol["artifacts"]}
    for name in ("protocol.json", "protocol-sha256.json", "input-ready.json", "execution-proof-initial.json"):
        plan.optional(deployment / name, f"provenance/deployment/{name}", "deployment_protocol")
    completed = []
    for seed in (7, 17, 29):
        for family in ("native_fp32", "native_head_int8", "onnx_fp32", "onnx_mixed_int8"):
            for split, endpoint in (("test", "hwdb-test"), ("icdar", "icdar-2013")):
                folder = deployment / f"evaluations/seed-{seed}/{family}/{split}"
                path = folder / "metrics.json"
                if not path.is_file() or str(path.resolve()) not in frozen_metrics:
                    continue
                report = read_json(path)
                if report.get("status") != "completed" or report["samples"] != {"test": 533675, "icdar": 224419}[split]:
                    continue
                if (report["seed"], report["family"], report["split"]) != (seed, family, split):
                    raise ValueError("Deployment evaluation identity mismatch")
                if (report["artifact"] != frozen_artifacts[(seed, family)]
                        or report["protocol_file_sha256"] != protocol_hash):
                    raise ValueError("Deployment evaluation does not match the frozen artifact/protocol")
                target = f"inference/deployment/{family}/seed-{seed:02d}/{endpoint}"
                plan.add(path, target + "/metrics.json", "deployment_inference",
                         expected_sha256=frozen_metrics[str(path.resolve())])
                for filename, expected in report["outputs_sha256"].items():
                    if Path(filename).name != filename:
                        raise ValueError("Unsafe prediction output name")
                    plan.add(folder / filename, target + "/" + filename, "deployment_inference", expected_sha256=expected)
                completed.append({"seed": seed, "family": family, "endpoint": endpoint, "metrics": target + "/metrics.json"})
    for split, endpoint in (("test", "hwdb-test"), ("icdar", "icdar-2013")):
        for path in (deployment / "cache").glob(f"{split}*targets*.npy"):
            plan.add(path, f"provenance/dataset/deployment-{endpoint}-targets.npy", "endpoint_targets")
    # Source and reproduction code are copied explicitly, never the workspace's
    # .env, SSH credentials, raw data, Vast.ai downloads, image caches or logs.
    for path in sorted((workspace / "src/hccr").rglob("*.py")):
        plan.add(path, "reproduce/" + path.relative_to(workspace).as_posix(), "source")
    for name in ("pyproject.toml", "README.md"):
        plan.add(workspace / name, "reproduce/" + name, "source")
    script_names = ["run_classifier_study.py", "prepare_classifier_stage.py", "analyze_compression_sensitivity.py",
                    "export_calibration_predictions.py", "benchmark_classifier_quantization.py",
                    "run_local_post_training.py", "rerun_local_onnx.py", "evaluate_frozen_deployment_heldouts.py",
                    "resume_frozen_deployment_heldouts.py", "run_precision_forensics.py", "localize_stem_precision.py",
                    "summarize_precision_forensics.py", "build_reviewer_bundle.py"]
    for name in script_names:
        plan.optional(workspace / "scripts" / name, "reproduce/scripts/" + name, "source")
    for name in ("build_evidence.py", "test_evidence.py"):
        plan.optional(evidence / name, "reproduce/evidence/" + name, "source")
    for name in ("test_reviewer_bundle.py", "test_local_post_training.py", "test_post_training_analysis.py",
                 "test_precision_forensics.py", "test_frozen_deployment_heldouts.py"):
        plan.optional(workspace / "tests" / name, "reproduce/tests/" + name, "test")
    sources = {entry.source: entry.target for entry in plan.entries.values()}
    deployment_map = []
    for item in frozen_protocol["artifacts"]:
        source = Path(item["path"]).resolve()
        if source not in sources:
            raise ValueError(f"Frozen deployment artifact is absent from bundle plan: {source}")
        target = sources[source]
        plan.add(source, target, "deployment", expected_sha256=item["sha256"])
        deployment_map.append({"seed": item["seed"], "family": item["family"],
                               "path": target, "sha256": item["sha256"], "engine": item["engine"]})
    plan.deployment_map = deployment_map
    return plan, catalog, completed


def rebase_markdown(text, original, target, plan):
    mapping = {entry.source: entry.target for entry in plan.entries.values()}
    def replace(match):
        label, link = match.group(1), match.group(2).strip().strip("<>")
        if re.match(r"^(https?://|mailto:|#)", link):
            return match.group(0)
        raw, separator, fragment = link.partition("#")
        raw = re.sub(r":\d+(?:-\d+)?$", "", raw)
        candidates = [Path(raw)] if Path(raw).is_absolute() else [original.parent / raw, plan.workspace / raw]
        for candidate in candidates:
            try:
                destination = mapping.get(candidate.resolve())
            except OSError:
                continue
            if destination:
                relative = os.path.relpath(destination, str(PurePosixPath(target).parent)).replace("\\", "/")
                return f"[{label}]({relative}{separator}{fragment})"
        # Preserve the source report's wording without a machine-specific link.
        # FILES.csv retains the original report path for provenance lookup.
        return label
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", replace, text)


def copy_entry(entry, destination, plan):
    target = destination / entry.target
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError("Copy destination escapes the release")
    target.parent.mkdir(parents=True, exist_ok=True)
    before = entry.source.stat()
    source_hash = sha256(entry.source)
    if entry.expected_sha256 is not None and source_hash != entry.expected_sha256:
        raise ValueError(f"Frozen source hash mismatch: {entry.source}")
    if entry.source.suffix.lower() == ".md":
        target.write_text(rebase_markdown(entry.source.read_text(encoding="utf-8-sig"), entry.source, entry.target, plan), encoding="utf-8")
    else:
        shutil.copyfile(entry.source, target)
    after = entry.source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed during snapshot; rebuild when stable: {entry.source}")
    copied_hash = sha256(target)
    if entry.source.suffix.lower() != ".md" and copied_hash != source_hash:
        raise ValueError(f"Copied artifact hash mismatch: {target}")
    return {"path": entry.target, "sha256": copied_hash, "bytes": target.stat().st_size,
            "role": entry.role, "run_id": entry.run_id,
            "source": entry.source.relative_to(plan.workspace).as_posix(), "source_sha256": source_hash,
            "transformation": "portable_markdown_links" if entry.source.suffix.lower() == ".md" else "byte_identical_copy"}


def link_from(page, target):
    return os.path.relpath(target, str(PurePosixPath(page).parent)).replace("\\", "/")


def create_navigation(destination, catalog, deployment, calibration):
    write_csv(destination / "RUNS.csv", catalog)
    write_json(destination / "inference/calibration/INDEX.json", {"precision": "float32", "evaluations": calibration})
    (destination / "inference/calibration/README.md").write_text(
        "# Pilot calibration predictions\n\n"
        "Six full calibration prediction files pair dense and rank-64 pilots for seeds 7, 17 and 29. "
        "Each JSONL starts with the original provenance header. The builder verifies its checkpoint digest against the copied pilot weights. "
        "The rank-104 pilots are a separate budget control and do not supply these predictions.\n\n"
        "Calibration inference uses FP32 model evaluation without autocast. The head-fit training budget in the header records BF16 training. "
        "These are different phases. The BF16 recognition results live separately under recognition-bf16. "
        "Calibration uses writer-separated official-training records for fixed-H allocation and is not a held-out test score.\n\n"
        "[Prediction/checkpoint index](INDEX.json) · [Allocation summaries](../../analysis/allocation) · "
        "[Study ledgers](../../provenance/study) · [All runs](../../RUNS.csv)\n", encoding="utf-8")
    for run in catalog:
        page = run["readme"]
        path = destination / page
        path.parent.mkdir(parents=True, exist_ok=True)
        links = [("Loadable checkpoint and sidecars", run["checkpoint"]), ("Training metrics", run["training_metrics"])]
        for key, title in (("test_inference", "HWDB test inference"), ("icdar_inference", "ICDAR-2013 inference")):
            if run.get(key):
                links.append((title, run[key]))
        lines = [f"# {run['variant']}, seed {run['seed']}", "", f"Stage: {run['group']}. Whole-model parameters: {run['whole_model_parameters']:,}.",
                 "", "\n".join(f"- [{title}]({link_from(page, target)})" for title, target in links), "",
                 "Training metrics retain the original validation-only policy. Separate post-training inference reports provide held-out accuracy.",
                 "Pilot checkpoints are intermediate head fits and have no test/ICDAR inference." if run["group"] == "pilots" else
                 "Recognition accuracy uses BF16 backbone inference and FP32 normalized heads. CPU FP32 and quantized results are separate deployment evaluations.",
                 "", f"Original configuration identity: `{run['fingerprint']}`.", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
    pending = 24 - len(deployment)
    write_json(destination / "DEPLOYMENT-COVERAGE.json", {"completed": len(deployment), "expected": 24,
               "pending": pending, "status": "completed" if pending == 0 else "partial_snapshot",
               "evaluations": deployment, "not_a_claim_that_live_jobs_are_stopped": True})
    start = ["# HCCR reviewer artifacts", "", "## Start here", "",
             "- [Model accuracy and parameter table](paper/tables-and-figures/supervisor-accuracy.md)",
             "- [CPU latency tables](paper/tables-and-figures/supervisor-cpu-latency.md)",
             "- [All 87 runs](RUNS.csv)", "- [Deployment endpoint coverage](DEPLOYMENT-COVERAGE.json)",
             "- [Pilot calibration predictions and fixed-H inputs](inference/calibration/README.md)",
             "- [Current scientific findings and limitations](paper/current_study_results.md)",
             "- [File checksums and source provenance](FILES.csv)", "",
             "## Folder layout", "", "```text", "checkpoints/   baselines, pilots, final / model-variant / seed-NN",
             "metrics/       original training configuration, curves and validation metrics",
             "inference/     recognition-bf16 and exact deployment-artifact endpoints",
             "benchmarks/    CPU raw timings/protocols; provisional training-GPU profiles are separate",
             "analysis/      H1/allocation, paired effects, gating and precision diagnostics",
             "deployment/    reloadable native INT8 and verified ONNX artifacts",
             "paper/         proposal, evidence tables and figures",
             "provenance/    dataset indices, frozen protocols, runtime and audit receipts",
             "runs/          per-run navigation pages", "reproduce/     source and selected verification scripts", "```", "",
             "Each run uses the same group/model/seed path in checkpoints, metrics and inference. For example:", "",
             "- [EfficientHCCRNet + CosFace, seed 7](runs/baselines/efficienthccrnet-cosface/seed-07/README.md)",
             "- [Matched dense reference, seed 7](runs/final/dense-reference/seed-07/README.md)", "",
             "## Coverage and interpretation", "",
             "The snapshot contains 30 baseline/objective runs, 9 pilots and 48 final/ablation runs. The 78 baseline/final checkpoints have 156 completed BF16 recognition evaluations. Pilots support allocation/calibration and validation, not held-out leaderboard claims.",
             f"Deployment evaluation includes {len(deployment)} of 24 complete artifact/endpoint passes at this snapshot. {pending} passes were not yet published here. Incomplete outputs, caches, test fixtures and runtime locks are excluded.",
             "The final dense reference follows the matched 20+10+30 stage lineage; it differs from the 20-epoch CosFace baseline. CPU timing and recognition accuracy have separate precision/runtime policies. Panel diagnostics with 4096 images cover only two writers and are not full validation or test scores.",
             "Six FP32 pilot calibration prediction files support fixed-H allocation. Their original headers identify the dense and rank-64 pilot checkpoints. Frequency/directional controls, study ledgers and pilot/final configurations are included.",
             "Original JSON receipts preserve historical container paths. Use FILES.csv to find the corresponding portable snapshot files. Old/provisional ONNX CPU rows survive only in provenance/original-cpu-tables and are excluded from the canonical paper latency table.",
             "Raw CASIA/ICDAR images, LMDB stores, ZIP/TAR downloads, temporary files and cloud access material are not included. Existing working artifacts were copied, not moved or hard-linked.",
             "", "## Verify and load", "", "From the project workspace:", "", "```powershell",
             "python scripts/build_reviewer_bundle.py --verify <this-release-directory>", "```", "",
             "The included reproduce/scripts/build_reviewer_bundle.py can also verify this standalone directory. To reconstruct a checkpoint with the included source installed:", "", "```python",
             "from pathlib import Path", "from hccr.models.artifacts import load_model_from_run",
             'model = load_model_from_run(Path("checkpoints/final/dense-reference/seed-07"), "cpu").eval()', "```", ""]
    (destination / "README.md").write_text("\n".join(start), encoding="utf-8")
    (destination / "reproduce/REVIEWER-GUIDE.md").write_text(
        "# Reproduce and inspect\n\n"
        "Start with ../RUNS.csv and ../deployment/ARTIFACTS.json to locate models. "
        "Each checkpoint leaf contains the original weights, model metadata and label mapping.\n\n"
        "The source is copied under src/hccr. The original pyproject.toml declares the package; "
        "install this folder with its modern extra to reconstruct the modern baselines. "
        "Runtime versions are recorded under ../provenance/runtime and the individual protocols.\n\n"
        "From the reviewer release directory, verify every file without loading weights:\n\n"
        "```powershell\npython reproduce/scripts/build_reviewer_bundle.py --verify .\n```\n\n"
        "Raw licensed datasets and preprocessed image caches are excluded. "
        "Prediction arrays and endpoint label/index files are provided to audit accuracy without images. "
        "Archived JSON receipts preserve original machine/container locations; FILES.csv provides the portable path mapping. "
        "Research execution scripts preserve their original workflow and are not claimed to be turnkey without the recorded source data and configuration.\n",
        encoding="utf-8")
    for name, description in {
        "checkpoints": "Each leaf is directly loadable with load_model_from_run and includes metadata and labels.",
        "metrics": "Original training artifacts. test/not_run in these files describes the training phase, not the later inference coverage.",
        "inference": "Completed predictions and metrics only. BF16 recognition and deployment-artifact accuracy are intentionally separate.",
        "benchmarks": "Use paper/tables-and-figures for canonical summaries. Raw CPU blocks and their declared policies remain auditable. Training GPU profiles are provisional FP32 measurements and omit p99.",
        "analysis": "Independent-writer H1, paired comparisons, gate interventions and precision forensics. Panel/sample support and uncertainty limitations remain explicit.",
        "deployment": "Exported models and reports. Native FP32 references are the dense-reference checkpoints. ONNX FP32 retains BN with ORT_DISABLE_ALL; mixed INT8 uses ORT_ENABLE_ALL, so their timing difference does not isolate quantization.",
        "provenance": "Original audit records may reference source machine paths. FILES.csv maps workspace-relative original files to portable reviewer paths. Scientific JSON, checkpoint and prediction files are byte-identical copies. Markdown links are rebased for portability, so original source-report receipts still describe the source Markdown hashes. FILES.csv records source_sha256, the copied sha256 and transformation separately; MANIFEST.json and CHECKSUMS.sha256 verify the published files.",
    }.items():
        path = destination / name / "README.md"; path.parent.mkdir(exist_ok=True)
        path.write_text(f"# {name.capitalize()}\n\n{description}\n\n[All runs](../RUNS.csv) · [Start here](../README.md)\n", encoding="utf-8")


def markdown_link_errors(root):
    errors = []
    for path in root.rglob("*.md"):
        if "reproduce" in path.relative_to(root).parts:
            continue
        for raw in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or re.match(r"^(https?://|mailto:)", target):
                continue
            destination = (path.parent / target).resolve()
            if not destination.is_relative_to(root.resolve()) or not destination.exists():
                errors.append({"file": path.relative_to(root).as_posix(), "target": target})
    return errors


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = read_json(root / "MANIFEST.json")
    expected = {row["path"]: row for row in manifest["files"]}
    for name, info in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file() or path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError(f"Bundle file integrity failed: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual - {"VERIFICATION.json"} != set(expected) | {"MANIFEST.json"}:
        raise ValueError("Bundle contains unlisted or missing files")
    links = markdown_link_errors(root)
    if links:
        raise ValueError(f"Broken reviewer links: {links[:5]}")
    return {"status": "verified", "files": len(expected), "bytes": sum(r["bytes"] for r in expected.values()),
            "runs": manifest["coverage"]["runs"], "recognition_evaluations": 156,
            "deployment_completed": manifest["coverage"]["deployment_completed"],
            "portable_markdown_links": "passed", "source_paths_moved": False,
            "checked_at_utc": datetime.now(timezone.utc).isoformat()}


def build(workspace, output, dry_run=False):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    reviewer_root = workspace / "artifacts/reviewer"
    if not output.is_relative_to(reviewer_root) or output.parent != reviewer_root:
        raise ValueError("Release directory must be a direct child of artifacts/reviewer")
    if output.exists():
        raise FileExistsError("Use a fresh release name; prior reviewer snapshots are preserved")
    plan, catalog, deployment = make_plan(workspace)
    summary = {"files_to_copy": len(plan.entries), "source_bytes": sum(e.source.stat().st_size for e in plan.entries.values()),
               "runs": len(catalog), "recognition_evaluations": 156, "deployment_completed": len(deployment),
               "deployment_expected": 24, "output": str(output)}
    if dry_run:
        return summary
    reviewer_root.mkdir(parents=True, exist_ok=True)
    owner = reviewer_root / ".build.lock"
    handle = os.open(owner, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, str(os.getpid()).encode()); os.close(handle)
    build_root = reviewer_root / ".builds" / output.name
    payload = build_root / "payload"
    if build_root.exists():
        owner.unlink()
        raise FileExistsError("A prior build receipt exists; choose a fresh release name")
    payload.mkdir(parents=True)
    status = {"status": "copying", "pid": os.getpid(), **summary, "copied": 0}
    write_json(build_root / "status.json", status)
    try:
        files = []
        for index, entry in enumerate(plan.entries.values(), start=1):
            files.append(copy_entry(entry, payload, plan))
            if index % 100 == 0:
                status["copied"] = index; write_json(build_root / "status.json", status)
                print(f"Copied and hashed {index}/{len(plan.entries)}", flush=True)
        create_navigation(payload, catalog, deployment, plan.calibration_map)
        write_json(payload / "deployment/ARTIFACTS.json", plan.deployment_map)
        # Include all navigation/index files as verified generated artifacts.
        known = {r["path"] for r in files}
        for path in sorted(payload.rglob("*")):
            relative = path.relative_to(payload).as_posix()
            if path.is_file() and relative not in known:
                files.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size,
                              "role": "generated_navigation", "run_id": "", "source": "generated", "source_sha256": "",
                              "transformation": "generated"})
        write_csv(payload / "FILES.csv", files)
        for name in ("FILES.csv",):
            path = payload / name
            files.append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size,
                          "role": "generated_index", "run_id": "", "source": "generated", "source_sha256": "",
                          "transformation": "generated"})
        (payload / "CHECKSUMS.sha256").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in files), encoding="utf-8")
        path = payload / "CHECKSUMS.sha256"
        files.append({"path": "CHECKSUMS.sha256", "sha256": sha256(path), "bytes": path.stat().st_size,
                      "role": "generated_checksums", "run_id": "", "source": "generated", "source_sha256": "",
                      "transformation": "generated"})
        write_json(payload / "MANIFEST.json", {"schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "release": output.name,
            "coverage": {"runs": 87, "baselines": 30, "pilots": 9, "final": 48,
                         "recognition_evaluations": 156, "calibration_evaluations": len(plan.calibration_map),
                         "deployment_completed": len(deployment), "deployment_expected": 24},
            "files": files, "limitations": ["deployment snapshot may be partial", "CPU host isolation not certified",
                  "FP32/BF16 portability and INT8 accuracy limitations retained", "pilot endpoints not scheduled"]})
        write_json(payload / "VERIFICATION.json", verify_bundle(payload))
        # Directory rename is restricted to checked absolute paths in reviewer_root.
        if not payload.resolve().is_relative_to(reviewer_root) or not output.resolve().is_relative_to(reviewer_root):
            raise ValueError("Publish paths escape reviewer_root")
        payload.rename(output)
        status.update(status="completed", copied=len(plan.entries), verification=read_json(output / "VERIFICATION.json"))
        write_json(build_root / "status.json", status)
        write_json(reviewer_root / "LATEST.json", {"release": output.name, "readme": output.name + "/README.md", "status": "verified"})
        (reviewer_root / "README.md").write_text(f"# Reviewer snapshots\n\n[Open the latest verified snapshot]({output.name}/README.md).\n\nEach release is a portable copy. Original working artifacts remain at their source paths. Run the bundle builder with a new output name to include newly completed evaluations.\n", encoding="utf-8")
        return status
    except Exception as exc:
        status.update(status="failed", error=str(exc)); write_json(build_root / "status.json", status)
        raise
    finally:
        owner.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_bundle(args.verify)
    else:
        output = args.output or args.workspace / "artifacts/reviewer/iciit2027-primary-20261026-v1"
        result = build(args.workspace, output, args.dry_run)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
