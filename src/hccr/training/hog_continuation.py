"""Extend a completed HOG pilot without changing its cache or original receipts."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hccr.artifacts.checkpoints import file_digest
from hccr.models.hog_fusion import load_hog_fusion_run
from hccr.training.hog_fusion import (
    FusionImages,
    cached_batches,
    evaluate_pair,
    write_json,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_fusion_artifact(model, output, metadata, labels, epoch, scores, policy):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "checkpoint.pt.tmp")
    os.replace(output / "checkpoint.pt.tmp", output / "checkpoint.pt")
    meta = deepcopy(metadata)
    meta["selection"] = {
        "epoch": epoch,
        "split": "validation",
        "metrics": scores,
        "policy": policy,
    }
    meta["test"] = {"status": "not_run"}
    meta["icdar"] = {"status": "not_run"}
    write_json(output / "checkpoint_metadata.json", meta)
    write_json(output / "labels.json", labels)


def open_verified_cache(parent, split, dataset, identity):
    root = Path(parent) / "cache" / split
    receipt = read_json(root / "metadata.json")
    expected = {
        **identity,
        "ordered_ids_sha256": dataset.ordered_ids_sha256,
        "samples": len(dataset),
        "dtype": "float32",
    }
    if (
        receipt["identity"] != expected
        or not receipt["complete"]
        or receipt["completed_rows"] != len(dataset)
    ):
        raise ValueError("parent feature-cache support or identity is incomplete")
    arrays = {}
    for name, dimension in (("cnn", 320), ("hog", 4356)):
        path = root / f"{name}.npy"
        if file_digest(path) != receipt["sha256"][name]:
            raise ValueError("parent feature-cache checksum differs")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (len(dataset), dimension):
            raise ValueError("parent feature-cache schema differs")
        arrays[name] = array
    return arrays


def continue_hog_training(args):
    parent = Path(args.parent_run).resolve()
    output = Path(args.output_dir).resolve()
    manifest = Path(args.manifest).resolve()
    if parent == output or parent in output.parents:
        raise ValueError("continuation must use a separate sibling output directory")
    if output.exists() and not args.resume:
        raise FileExistsError("continuation output exists; use --resume")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    prior = read_json(parent / "identity.json")
    metadata = read_json(parent / "checkpoint_metadata.json")
    labels = read_json(parent / "labels.json")
    code_root = Path(__file__).parents[1]
    if (
        prior["manifest_sha256"] != file_digest(manifest)
        or prior["labels_sha256"]
        != file_digest(Path(metadata["source_run"]) / "labels.json")
        or prior["model_code_sha256"] != file_digest(code_root / "models/hog_fusion.py")
        or prior["runner_code_sha256"]
        != file_digest(code_root / "training/hog_fusion.py")
        or prior["torch"] != str(torch.__version__)
        or prior["gpu"]
        != (
            torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda")
            else None
        )
        or prior["device"] != args.device
        or read_json(manifest.with_name("labels.json")) != labels
    ):
        raise ValueError("continuation differs from the pinned pilot data/code/runtime")
    original_state = torch.load(
        parent / "latest.pt", map_location=args.device, weights_only=True
    )
    if args.total_epochs <= original_state["epoch"]:
        raise ValueError("total epochs must extend the completed pilot")
    identity = {
        "parent": str(parent),
        "parent_identity": prior,
        "parent_latest_sha256": file_digest(parent / "latest.pt"),
        "parent_selected_sha256": file_digest(parent / "checkpoint.pt"),
        "parent_metadata_sha256": file_digest(parent / "checkpoint_metadata.json"),
        "parent_labels_sha256": file_digest(parent / "labels.json"),
        "total_epochs": args.total_epochs,
        "continuation_code_sha256": file_digest(Path(__file__)),
        "start_epoch": original_state["epoch"] + 1,
        "selection": "validation_best_available_trained_and_epoch_zero_fallback",
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "identity.json").exists() and (
        read_json(output / "identity.json") != identity
    ):
        raise ValueError("continuation identity changed")
    write_json(output / "identity.json", identity)

    def report(stage, **values):
        status = {"stage": stage, **values}
        write_json(output / "status.json", status)
        print(json.dumps(status), flush=True)

    report("verify_parent_caches")
    datasets = {
        split: FusionImages(
            manifest,
            split,
            labels["labels"],
            metadata["preprocess"],
            prior["manifest_sha256"],
        )
        for split in ("train", "validation")
    }
    if set(datasets["train"].writers) & set(datasets["validation"].writers):
        raise ValueError("training and validation writers overlap")
    caches = {
        split: open_verified_cache(parent, split, dataset, prior)
        for split, dataset in datasets.items()
    }
    model = load_hog_fusion_run(parent, args.device)
    frozen = {k: v.detach().cpu().clone() for k, v in model.cnn.state_dict().items()}
    model.load_state_dict(original_state["model"])
    optimizer = torch.optim.AdamW(
        [
            {"params": model.hog_projection.parameters(), "weight_decay": 1e-4},
            {"params": model.hog_normalization.parameters(), "weight_decay": 0.0},
            {"params": model.fusion.parameters(), "weight_decay": 0.0},
        ],
        lr=prior["learning_rate"],
    )
    optimizer.load_state_dict(original_state["optimizer"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    metadata = deepcopy(metadata)
    metadata["identity"] = identity
    metadata["training"]["epochs"] = args.total_epochs
    metadata["training"]["continuation_from_epoch"] = original_state["epoch"]
    metadata["training"]["parent_run"] = str(parent)
    pilot_best = max(
        original_state["curves"],
        key=lambda row: (row["fusion_top1"], -row["fusion_cross_entropy"]),
    )["epoch"]
    metadata["training"]["selection_limitation"] = (
        f"best pilot trained epoch was {pilot_best}; only saved pilot epoch "
        f"{original_state['epoch']} and subsequent epochs have available weights"
    )
    state_path = output / "latest.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=args.device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first_epoch = state["epoch"] + 1
        best = tuple(state["best_trained"])
        best_epoch, curves = state["best_epoch"], state["curves"]
    else:
        first_epoch = original_state["epoch"] + 1
        curves = original_state["curves"]
        scores, _, _ = evaluate_pair(
            model,
            caches["validation"],
            datasets["validation"].targets,
            args.device,
            prior["batch_size"],
        )
        best, best_epoch = (
            (scores["fusion_top1"], -scores["fusion_cross_entropy"]),
            original_state["epoch"],
        )
        save_fusion_artifact(
            model,
            output / "trained_best",
            metadata,
            labels,
            best_epoch,
            scores,
            "max_validation_top1_then_min_cross_entropy_available_trained_epochs",
        )
    for epoch in range(first_epoch, args.total_epochs + 1):
        model.train()
        seen, total_loss = 0, 0.0
        for cnn, hog, targets in cached_batches(
            caches["train"],
            datasets["train"].targets,
            args.device,
            prior["batch_size"],
            rng=np.random.default_rng(prior["seed"] + epoch),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model.forward_cached(cnn, hog, targets), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite continuation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            seen += len(targets)
            total_loss += loss.item() * len(targets)
            if seen % (prior["batch_size"] * 200) < prior["batch_size"]:
                report(
                    "train", epoch=epoch, completed=seen, total=len(datasets["train"])
                )
        scores, _, _ = evaluate_pair(
            model,
            caches["validation"],
            datasets["validation"].targets,
            args.device,
            prior["batch_size"],
        )
        candidate = (scores["fusion_top1"], -scores["fusion_cross_entropy"])
        if candidate > best:
            best, best_epoch = candidate, epoch
            save_fusion_artifact(
                model,
                output / "trained_best",
                metadata,
                labels,
                epoch,
                scores,
                "max_validation_top1_then_min_cross_entropy_available_trained_epochs",
            )
        curves.append({"epoch": epoch, "train_loss": total_loss / seen, **scores})
        write_json(output / "curves.json", {"epochs": curves})
        save_fusion_artifact(
            model,
            output / "last",
            metadata,
            labels,
            epoch,
            scores,
            "fixed_final_epoch_not_selected_by_test",
        )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "curves": curves,
                "best_trained": best,
                "best_epoch": best_epoch,
            },
            output / "latest.pt.tmp",
        )
        os.replace(output / "latest.pt.tmp", state_path)
        report("epoch_complete", epoch=epoch, best_trained_epoch=best_epoch, **scores)
    if any(
        not torch.equal(frozen[name], value.detach().cpu())
        for name, value in model.cnn.state_dict().items()
    ):
        raise RuntimeError("frozen CNN or CosFace state changed")
    trained = load_hog_fusion_run(output / "trained_best", args.device)
    scores, baseline, fusion = evaluate_pair(
        trained,
        caches["validation"],
        datasets["validation"].targets,
        args.device,
        prior["batch_size"],
    )
    without_hog, _, _ = evaluate_pair(
        trained,
        caches["validation"],
        datasets["validation"].targets,
        args.device,
        prior["batch_size"],
        ablate_hog=True,
    )
    scores.update(
        best_trained_epoch=best_epoch,
        total_epochs=args.total_epochs,
        frozen_state_unchanged=True,
        without_hog=without_hog,
        selection_limitation=metadata["training"]["selection_limitation"],
    )
    write_json(output / "validation_metrics.json", scores)
    np.savez_compressed(
        output / "validation_predictions.npz",
        baseline=baseline,
        fusion=fusion,
        targets=datasets["validation"].targets,
        sample_ids=np.asarray(datasets["validation"].ids),
        writers=np.asarray(datasets["validation"].writers),
    )
    original = read_json(parent / "initial_metrics.json")
    fallback = (scores["fusion_top1"], -scores["fusion_cross_entropy"]) <= (
        original["fusion_top1"],
        -original["fusion_cross_entropy"],
    )
    write_json(
        output / "selection.json",
        {
            "epoch_zero_fallback_preferred": fallback,
            "deployed_policy_run": str(parent if fallback else output / "trained_best"),
            "benchmark_runs": [str(output / "trained_best"), str(output / "last")],
            "selection_data": "validation_only",
            "test_and_icdar_not_opened_during_training": True,
        },
    )
    write_json(
        output / "holdout_plan.json",
        {
            "selection_frozen": True,
            "manifest_sha256": prior["manifest_sha256"],
            "runs": {
                name: {
                    "path": str(output / name),
                    "checkpoint_sha256": file_digest(output / name / "checkpoint.pt"),
                    "metadata_sha256": file_digest(
                        output / name / "checkpoint_metadata.json"
                    ),
                }
                for name in ("trained_best", "last")
            },
            "splits": ["test", "icdar"],
            "test_based_selection": False,
        },
    )
    for split, dataset in datasets.items():
        dataset.store.close()
        for array in caches[split].values():
            array._mmap.close()
    report(
        "completed",
        total_epochs=args.total_epochs,
        best_trained_epoch=best_epoch,
        baseline_top1=scores["baseline_top1"],
        fusion_top1=scores["fusion_top1"],
        delta_pp=scores["delta_pp"],
        epoch_zero_fallback_preferred=fallback,
    )
    return scores
