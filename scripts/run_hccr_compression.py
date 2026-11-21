"""FP32 CosFace compression follow-ups for the audited 3,755-class HWDB panel."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "dense": ("dense", 320, 64),
    "rank160": ("low_rank", 320, 160),
    "rank104": ("low_rank", 320, 104),
    "rank64": ("low_rank", 320, 64),
    "embedding104": ("dense", 104, 64),
    "embedding64": ("dense", 64, 64),
}
COUNTS = {"test": 533675, "icdar": 224419}
MANIFEST_SHA = "f45c3e68bde7019336d5e835cea06fb911049ebdf2a063eb4381ad2f085011bf"


def read(path):
    return json.loads(path.read_bytes())


def verify_manifest(path):
    from hccr.artifacts.checkpoints import file_digest

    if file_digest(path) != MANIFEST_SHA:
        raise ValueError("This panel requires the audited HWDB1.0–1.1 manifest")


def fit(args):
    import yaml

    from hccr.cli import build_parser
    from hccr.commands.train import config_from_arguments
    from hccr.experiments.cli import build_parser as experiment_parser
    from hccr.experiments.runner import build_jobs
    from hccr.experiments.spec import load_experiment_spec
    from hccr.training.proposal import pilot_budget
    from hccr.training.workflow import run_training

    spec = yaml.safe_load(
        (ROOT / "configs/experiment/official_classifier_100ep.yaml").read_bytes()
    )
    spec.update(
        experiment_id=f"compression-{args.variant}-seed-{args.seed}",
        manifest=str(args.manifest.resolve()),
        output_dir=str(args.output.resolve()),
        seeds=[args.seed],
    )
    shared = spec["base_args"]
    shared.update(
        next(
            v["args"] for v in spec["variants"] if v["name"] == "efficient_hccr_cosface"
        )
    )
    shared.update(
        prefetch_factor=1, persistent_workers=True, worker_timeout_seconds=120.0
    )
    if args.variant != "teacher":
        if args.initialize_from is None:
            raise ValueError(
                "The final continuation requires a matching 100-epoch pilot"
            )
        provenance = read(args.initialize_from / "checkpoint_metadata.json")
        pilot_budget(provenance)
        structure, dim, rank = VARIANTS[args.variant]
        expected = ("low_rank", dim) if dim != 320 else (structure, rank)
        actual = provenance["model"]
        if (
            provenance["training"]["seed"] != args.seed
            or provenance["training"]["configured_epochs"] != 100
            or actual["classification_head"] != "cosface"
            or actual["embedding_dim"] != 320
            or actual["classifier_structure"] != expected[0]
            or (expected[0] == "low_rank" and actual["classifier_rank"] != expected[1])
        ):
            raise ValueError(
                "Pilot seed, rank, objective, or budget disagrees with the variant"
            )
        shared.update(
            classifier_structure=structure,
            embedding_dim=dim,
            classifier_rank=rank,
            initialize_from_run=str(args.initialize_from.resolve()),
        )
    elif args.initialize_from is not None:
        raise ValueError("A teacher starts from the seeded random initialization")
    spec["variants"] = [{"name": args.variant, "args": {}}]
    args.output.mkdir(parents=True, exist_ok=True)
    recipe = args.output / "experiment.yaml"
    serialized = yaml.safe_dump(spec, sort_keys=False)
    if recipe.exists() and recipe.read_text(encoding="utf-8") != serialized:
        raise ValueError("Choose a new output directory for a changed recipe")
    recipe.write_text(serialized, encoding="utf-8")
    parsed = experiment_parser().parse_args(["--config", str(recipe)])
    (job,) = build_jobs(load_experiment_spec(parsed, ROOT))
    parsed = build_parser().parse_args(list(job.command[job.command.index("train") :]))
    parameters = config_from_arguments(parsed)
    if args.dry_run:
        print(recipe)
        return
    verify_manifest(args.manifest)
    if any(args.output.glob("*/checkpoint_metadata.json")):
        raise FileExistsError("Training output already contains a checkpoint")
    run_training(parameters)


def pilot(args):
    runner = args.runner.resolve()
    pinned = runner.parent.parent
    sys.path[:0] = [str(pinned / "src"), str(pinned)]
    import torch

    module_spec = importlib.util.spec_from_file_location("iciit_fp32_runner", runner)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    verify_manifest(args.manifest)
    arithmetic = module.pin_arithmetic()
    if torch.cuda.device_count() != 1 or (
        arithmetic["torch"],
        arithmetic["cuda"],
        arithmetic["cudnn"],
        arithmetic["gpu"],
    ) != ("2.15.0+cu132", "13.2", 92600, "NVIDIA GeForce RTX 4090"):
        raise ValueError("Use the ICIIT FP32 runtime with one visible GPU")
    teacher = args.teacher.resolve()
    manifest = args.manifest.resolve()
    recipe = module.load_teacher_recipe(teacher, manifest)
    identity = module.teacher_identity(teacher, recipe.seed, manifest)
    rank = VARIANTS[args.variant][2]
    spec = module.prepare_classifier_stage(
        "pilots", teacher, manifest, seed=recipe.seed, rank=rank, device="cuda"
    )
    spec["variants"] = [
        v
        for v in spec["variants"]
        if v["name"]
        == ("dense_pilot" if args.variant == "dense" else f"rank{rank}_pilot")
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = output / args.variant
    if run.exists():
        raise FileExistsError(run)
    spec["output_dir"] = str(output)
    module.apply_teacher_recipe(spec, recipe, pilot=True)
    (config,) = module.config_jobs(spec, output / "pilot.yaml").values()
    rows = module.read_manifest(manifest)
    writer_info = module.validate_training_manifest(manifest, rows)
    module.validate_official_source(manifest, rows)
    labels = module.manifest_class_labels(rows)
    del rows
    module.workflow._configure_reproducibility(config)
    model = module.load_model_from_run(teacher, device="cuda").eval()
    features = {
        split: module.extract_features(model, config, split)
        for split in ("train", "validation")
    }
    cache = {
        "backbone_digest": module.state_digest(
            model.state_dict(), exclude_classifier=True
        ),
        "teacher_checkpoint_digest": identity["artifacts"]["checkpoint.pt"],
        "manifest_digest": MANIFEST_SHA,
        "splits": {key: value.report for key, value in features.items()},
        "arithmetic": arithmetic,
        "extract_workers": config.num_workers,
    }
    del model
    module.fit_pilot(config, teacher, run, features, labels, writer_info, cache)


def open_run(run, manifest):
    from hccr.models import read_checkpoint_metadata

    verify_manifest(manifest)
    metadata = read_checkpoint_metadata(run)
    architecture = metadata["model"]
    if (
        metadata["manifest_digest"] != MANIFEST_SHA
        or architecture["name"] != "efficient_hccr"
        or architecture["classification_head"] != "cosface"
        or architecture["num_classes"] != 3755
        or metadata["training"]["precision"]["resolved"] != "float32"
    ):
        raise ValueError(
            "Expected a matching 3,755-class FP32 EfficientHCCR CosFace checkpoint"
        )
    return metadata


def evaluate_or_svd(args):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from hccr.artifacts import write_json
    from hccr.artifacts.checkpoints import file_digest
    from hccr.data.dataset import HCCRDataset
    from hccr.evaluation.evaluator import evaluate
    from hccr.models import load_model_from_run
    from hccr.models.heads import build_classifier
    from hccr.preprocessing import EvalPreprocessor

    meta = open_run(args.run, args.manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model_from_run(args.run, args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint_sha256": file_digest(args.run / "checkpoint.pt"),
        "manifest_sha256": MANIFEST_SHA,
        "seed": meta["training"]["seed"],
        "batch_size": 32,
        "device": args.device,
        "precision": "float32",
    }
    if args.command == "svd":
        if model.classifier_structure != "dense" or model.embedding_dim != 320:
            raise ValueError("SVD requires the dense 320-D CosFace reference")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dense = model.classifier.weight.detach().cpu()
        normalized = F.normalize(dense, dim=1, eps=model.normalization_eps)
        spectrum = torch.linalg.svdvals(normalized).double()
        energy = spectrum.square().cumsum(0) / spectrum.square().sum()
        report.update(
            svd_rank=104,
            matrix="row-normalized CosFace weights",
            energy_at_104=float(energy[103]),
        )
        write_json(
            args.output / "spectrum.json",
            {
                **report,
                "spectrum": spectrum.tolist(),
                "cumulative_energy": energy.tolist(),
            },
        )
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), layout="constrained")
        ranks = list(range(1, len(spectrum) + 1))
        for ax, values, label in zip(
            axes,
            (spectrum, energy),
            ("Singular value", "Cumulative energy"),
            strict=True,
        ):
            ax.plot(ranks, values)
            ax.axvline(104, color="gray", linestyle="--")
            ax.set(xlabel="Rank", ylabel=label)
        fig.savefig(args.output / "spectrum.png", dpi=200)
        plt.close(fig)
        if args.spectrum_only:
            return
        head = build_classifier(
            model.embedding_dim,
            3755,
            "cosface",
            meta["model"]["logit_scale"],
            meta["model"]["angular_margin"],
            classifier_structure="low_rank",
            classifier_rank=104,
            normalization_eps=model.normalization_eps,
        )
        head.initialize_from_dense(dense)
        model.classifier = head.to(args.device).eval()
    for split, count in COUNTS.items():
        prep = EvalPreprocessor(**meta["preprocess"])
        data = HCCRDataset(
            args.manifest, split, prep, storage_backend="lmdb", metadata_mode="index"
        )
        if len(data) != count:
            raise ValueError(f"{split} sample count mismatch")
        loader = DataLoader(data, batch_size=32, num_workers=10)
        metrics = evaluate(model, loader, args.device, evaluation_name=split)
        write_json(
            args.output / f"{split}.json",
            {**report, "split": split, "samples": count, "metrics": metrics},
        )
        data.close()


def deployment(args):
    if args.device == "cpu":
        if args.core is None:
            raise ValueError("Supply --core for a one-core CPU measurement")
        os.sched_setaffinity(0, {args.core})

    import torch

    from hccr.artifacts import write_json
    from hccr.artifacts.checkpoints import file_digest
    from hccr.benchmarking import summarize_timings
    from hccr.data.dataset import HCCRDataset
    from hccr.models import load_model_from_run
    from hccr.preprocessing import EvalPreprocessor

    meta = open_run(args.run, args.manifest)
    arch = meta["model"]
    if arch["embedding_dim"] != 320 or (
        arch["classifier_structure"] != "dense"
        and (arch["classifier_structure"], arch["classifier_rank"])
        not in {("low_rank", 64), ("low_rank", 104)}
    ):
        raise ValueError("Deployment panel covers Dense, Rank 104 and Rank 64")
    if args.device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model_from_run(args.run, args.device).eval()
    data = HCCRDataset(
        args.manifest,
        "icdar",
        EvalPreprocessor(**meta["preprocess"]),
        storage_backend="lmdb",
        metadata_mode="index",
    )
    assert len(data) == COUNTS["icdar"]
    indices = Random(20261122).sample(range(len(data)), 3000)
    images = [data[i][0].unsqueeze(0) for i in indices]
    synchronize = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    timings = []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i, image in enumerate(images[:100] + images):
            sample = image.to(args.device)
            synchronize()
            start = time.perf_counter()
            model(sample)
            synchronize()
            if i >= 100:
                timings.append((time.perf_counter() - start) * 1000)
    checkpoint = args.run / "checkpoint.pt"
    report = {
        "checkpoint_sha256": file_digest(checkpoint),
        "manifest_sha256": MANIFEST_SHA,
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name()
            if args.device == "cuda"
            else platform.processor() or platform.machine()
        ),
        "platform": platform.platform(),
        "cpu_core": args.core if args.device == "cpu" else None,
        "torch": str(torch.__version__),
        "batch_size": 1,
        "warmup_iterations": 100,
        "iterations": len(timings),
        "precision": "float32",
        "latency_unit": "ms",
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if args.device == "cuda"
        else None,
        "sample_ids": [data.rows[i]["sample_id"] for i in indices],
        **summarize_timings(timings),
    }
    write_json(args.output, report)
    data.close()


def reliability(args):
    from hccr.artifacts import write_json

    def collect(paths):
        scores = {}
        for path in paths:
            row = read(path)
            if row["manifest_sha256"] != MANIFEST_SHA or row["split"] != args.split:
                raise ValueError(f"Incompatible heldout receipt: {path}")
            if row["samples"] != COUNTS[args.split] or row["precision"] != "float32":
                raise ValueError(f"Unexpected sample count or precision: {path}")
            seed = row.get("seed")
            if seed is None:
                seed = int(path.parent.name.removeprefix("seed-"))
            if seed in scores:
                raise ValueError(f"Duplicate seed {seed}")
            scores[seed] = row["metrics"]["top1"] * 100
        return scores

    def summary(values):
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        half = 2.7764451051977987 * std / 5**0.5
        return {"mean_pct": mean, "std_pp": std, "ci95_pct": [mean - half, mean + half]}

    dense, rank104 = collect(args.dense), collect(args.rank104)
    if len(dense) != 5 or dense.keys() != rank104.keys():
        raise ValueError("Use five matched, independent seeds per model")
    seeds = sorted(dense)
    paired = summary([rank104[seed] - dense[seed] for seed in seeds])
    report = {
        "split": args.split,
        "seed_scores_pct": {"dense": dense, "rank104": rank104},
        "seeds": seeds,
        "dense": summary(dense.values()),
        "rank104": summary(rank104.values()),
        "rank104_minus_dense_pp": paired,
    }
    if args.margin_pp is not None:
        if not args.margin_pp > 0:
            raise ValueError("Supply a positive, preselected non-inferiority margin")
        report["noninferiority"] = {
            "margin_pp": args.margin_pp,
            "passes": paired["ci95_pct"][0] > -args.margin_pp,
            "test": "paired lower one-sided 97.5% t bound",
        }
    write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit")
    train.add_argument("--variant", choices=["teacher", *VARIANTS], required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--initialize-from", type=Path)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(action=fit)
    head = commands.add_parser("pilot")
    head.add_argument(
        "--variant", choices=("dense", "rank160", "rank104", "rank64"), required=True
    )
    head.add_argument("--teacher", type=Path, required=True)
    head.add_argument("--runner", type=Path, required=True)
    head.set_defaults(action=pilot)
    for name, action in (
        ("evaluate", evaluate_or_svd),
        ("svd", evaluate_or_svd),
        ("deploy", deployment),
    ):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
        if name == "deploy":
            command.add_argument("--core", type=int)
        if name == "svd":
            command.add_argument("--spectrum-only", action="store_true")
        command.set_defaults(action=action)
    stats = commands.add_parser("stats")
    stats.add_argument("--split", choices=COUNTS, required=True)
    stats.add_argument("--dense", nargs=5, type=Path, required=True)
    stats.add_argument("--rank104", nargs=5, type=Path, required=True)
    stats.add_argument("--margin-pp", type=float)
    stats.set_defaults(action=reliability)
    for name, command in commands.choices.items():
        command.add_argument("--output", type=Path, required=True)
        if name != "stats":
            command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command != "pilot":
        sys.path.insert(0, str(ROOT / "src"))
    args.action(args)


if __name__ == "__main__":
    main()
