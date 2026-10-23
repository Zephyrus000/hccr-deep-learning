"""Write a pilot or residual experiment configuration without launching training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from hccr.artifacts.checkpoints import file_digest
from hccr.data.manifest import read_manifest
from hccr.models.artifacts import load_model_from_run, read_checkpoint_metadata
from hccr.training.proposal import pilot_budget, resolve_residual_mask, state_digest


def _near_budget_rank(classes: int, dimension: int, rank: int, k: int) -> int:
    if not 1 <= k <= classes:
        raise ValueError("near-budget residual class count must fit the vocabulary")
    result = math.ceil(
        (rank * (classes + dimension) + k * dimension + dimension + 2)
        / (classes + dimension)
    )
    if result > min(classes, dimension):
        raise ValueError("near-budget rank exceeds the classifier dimensions")
    return result


def prepare_classifier_stage(
    stage: str,
    source_run: Path,
    manifest: Path,
    *,
    mask: Path | None = None,
    rank: int = 64,
    seed: int | None = None,
    device: str = "auto",
    dense_pilot_run: Path | None = None,
    near_budget_k: int | None = None,
    near_budget_pilot_run: Path | None = None,
) -> dict:
    """Resolve architecture and checkpoint identity before writing runnable YAML."""
    if stage == "pilots" and any(
        value is not None for value in (mask, dense_pilot_run, near_budget_pilot_run)
    ):
        raise ValueError("pilot generation does not accept residual-stage artifacts")
    if stage == "residuals" and near_budget_k is not None:
        raise ValueError("near-budget-k is only used when generating pilots")
    source_run, manifest = Path(source_run).resolve(), Path(manifest).resolve()
    metadata = read_checkpoint_metadata(source_run)
    model = metadata["model"]
    if metadata["manifest_digest"] != file_digest(manifest):
        raise ValueError("source run and stage manifest differ")
    resolved_seed = metadata["training"]["seed"] if seed is None else seed
    if resolved_seed != metadata["training"]["seed"]:
        raise ValueError("paired stages must keep the source pilot seed")
    base = {
        key: model[key]
        for key in (
            "width",
            "backbone_output_channels",
            "embedding_dim",
            "stage_depths",
            "stem_stride",
            "reparameterize_depthwise",
            "dropout",
            "classification_head",
            "logit_scale",
            "angular_margin",
            "gate_kind",
            "gate_use_gap",
            "gate_detach_gap",
            "normalization_eps",
        )
        if key in model
    }
    if model["name"] != "efficient_hccr":
        for key in (
            "width",
            "backbone_output_channels",
            "stage_depths",
            "stem_stride",
            "reparameterize_depthwise",
            "dropout",
        ):
            base.pop(key, None)
    base.update(
        {
            "model": model["name"],
            "num_classes": model["num_classes"],
            "image_size": metadata["preprocess"]["image_size"],
            "batch_size": metadata["training"]["batch_size"],
            "reference_batch_size": metadata["training"]["batch_size"],
            "optimizer_step_policy": "configured_epochs",
            "learning_rate_scaling": "none",
            "weight_decay": 1e-4,
            "device": device,
            "precision": "float32",
            "dataset_backend": "auto",
            "num_workers": 0,
            "require_official_data": True,
            "evaluation_policy": "validation_only",
            "initialize_from_run": str(source_run),
            "early_stopping_patience": 0,
            "reproducibility_mode": "strict",
            "scheduler": "cosine",
            "margin_warmup_ratio": 0.1,
            "train_rotation_degrees": 7.0,
            "train_translate_ratio": 0.05,
            "train_blur_probability": 0.0,
        }
    )
    if stage == "pilots":
        if not 1 <= rank <= min(model["num_classes"], model["embedding_dim"]):
            raise ValueError("pilot rank must fit the classifier dimensions")
        if (
            model.get("classifier_structure", "dense") != "dense"
            or model["classification_head"] == "softmax"
        ):
            raise ValueError("pilots require a normalized dense teacher")
        base.update(
            {
                "epochs": 10,
                "learning_rate": 3e-4,
                "freeze_backbone": True,
                "checkpoint_selection": "final",
            }
        )
        variants = [
            {"name": "dense_pilot", "args": {"classifier_structure": "dense"}},
            {
                "name": f"rank{rank}_pilot",
                "args": {"classifier_structure": "low_rank", "classifier_rank": rank},
            },
        ]
        if near_budget_k is not None:
            rank_prime = _near_budget_rank(
                int(model["num_classes"]),
                int(model["embedding_dim"]),
                rank,
                near_budget_k,
            )
            variants.append(
                {
                    "name": f"rank{rank_prime}_budget_pilot",
                    "args": {
                        "classifier_structure": "low_rank",
                        "classifier_rank": rank_prime,
                    },
                }
            )
    elif stage == "residuals":
        if mask is None or model.get("classifier_structure") != "low_rank":
            raise ValueError("residuals require a low-rank pilot and mask")
        allocation = resolve_residual_mask(mask, manifest, read_manifest(manifest))
        if allocation["provenance"]["low_rank"]["checkpoint_digest"] != file_digest(
            source_run / "checkpoint.pt"
        ):
            raise ValueError("mask belongs to a different low-rank pilot")
        base.update(
            {
                "epochs": 30,
                "learning_rate": 1e-4,
                "freeze_backbone": False,
                "checkpoint_selection": "best_validation",
                "classifier_rank": model["classifier_rank"],
                "residual_mask": str(mask.resolve()),
                "residual_regularization": 1e-4,
            }
        )
        variants = [
            {
                "name": "uniform_low_rank",
                "args": {
                    "classifier_structure": "low_rank",
                    "residual_mask": None,
                    "residual_regularization": 0.0,
                },
            },
            {
                "name": "static_residual",
                "args": {"classifier_structure": "static_residual"},
            },
            {
                "name": "dynamic_residual",
                "args": {"classifier_structure": "dynamic_residual"},
            },
        ]
    else:
        raise ValueError("stage must be pilots or residuals")
    if stage == "residuals" and dense_pilot_run is not None:
        dense_pilot_run = dense_pilot_run.resolve()
        dense = read_checkpoint_metadata(dense_pilot_run)
        if (
            dense.get("manifest_digest") != metadata["manifest_digest"]
            or file_digest(dense_pilot_run / "checkpoint.pt")
            != allocation["provenance"]["dense"]["checkpoint_digest"]
        ):
            raise ValueError("dense pilot does not match the paired allocation")
        common = {
            "initialize_from_run": str(dense_pilot_run),
            "residual_mask": None,
            "residual_regularization": 0.0,
        }
        variants.append(
            {
                "name": "dense_reference",
                "args": {**common, "classifier_structure": "dense"},
            }
        )
    if stage == "residuals" and near_budget_pilot_run is not None:
        near_budget_pilot_run = near_budget_pilot_run.resolve()
        near = read_checkpoint_metadata(near_budget_pilot_run)
        rank_prime = _near_budget_rank(
            int(model["num_classes"]),
            int(model["embedding_dim"]),
            int(model["classifier_rank"]),
            len(allocation["residual_class_indices"]),
        )
        if (
            near.get("manifest_digest") != metadata["manifest_digest"]
            or near["model"].get("classifier_structure") != "low_rank"
            or near["model"].get("classifier_rank") != rank_prime
            or near["model"].get("num_classes") != model["num_classes"]
            or near["model"].get("embedding_dim") != model["embedding_dim"]
            or pilot_budget(near)
            != allocation["provenance"]["dense"]["head_fit_budget"]
        ):
            raise ValueError("near-budget reference requires a matching fitted pilot")
        near_model = load_model_from_run(near_budget_pilot_run)
        if (
            state_digest(near_model.state_dict(), exclude_classifier=True)
            != allocation["provenance"]["dense"]["backbone_digest"]
        ):
            raise ValueError("near-budget pilot uses a different frozen backbone")
        variants.append(
            {
                "name": "uniform_near_budget",
                "args": {
                    "initialize_from_run": str(near_budget_pilot_run),
                    "residual_mask": None,
                    "residual_regularization": 0.0,
                    "classifier_structure": "low_rank",
                    "classifier_rank": rank_prime,
                },
            }
        )
    checkpoint_id = file_digest(source_run / "checkpoint.pt")[:8]
    if stage == "pilots":
        checkpoint_id += f"-rank{rank}"
    if mask is not None:
        checkpoint_id += "-" + file_digest(mask)[:8]
    if near_budget_k is not None:
        checkpoint_id += f"-budget{near_budget_k}"
    if dense_pilot_run is not None:
        checkpoint_id += "-dense" + file_digest(dense_pilot_run / "checkpoint.pt")[:8]
    if near_budget_pilot_run is not None:
        checkpoint_id += "-" + file_digest(near_budget_pilot_run / "checkpoint.pt")[:8]
    return {
        "experiment_id": f"official-{stage}-seed{resolved_seed}-{checkpoint_id}",
        "manifest": str(manifest),
        "output_dir": "experiments/official_classifier",
        "seeds": [resolved_seed],
        "full_class_num_classes": model["num_classes"],
        "profile_devices": [],
        "base_args": base,
        "variants": variants,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilots", "residuals"), required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--dense-pilot-run", type=Path)
    parser.add_argument("--near-budget-k", type=int)
    parser.add_argument("--near-budget-pilot-run", type=Path)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = prepare_classifier_stage(
        args.stage,
        args.source_run,
        args.manifest,
        mask=args.mask,
        rank=args.rank,
        seed=args.seed,
        device=args.device,
        dense_pilot_run=args.dense_pilot_run,
        near_budget_k=args.near_budget_k,
        near_budget_pilot_run=args.near_budget_pilot_run,
    )
    protected = {
        args.manifest.resolve(),
        (args.source_run / "checkpoint.pt").resolve(),
        (args.source_run / "checkpoint_metadata.json").resolve(),
    }
    if args.mask is not None:
        protected.add(args.mask.resolve())
    for run in (args.dense_pilot_run, args.near_budget_pilot_run):
        if run is not None:
            protected.update(
                (run / name).resolve()
                for name in ("checkpoint.pt", "checkpoint_metadata.json", "labels.json")
            )
    if args.output.resolve() in protected:
        raise ValueError("configuration output must not overwrite a source artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "configuration": str(args.output),
                "jobs": len(config["variants"]),
                "training_started": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
