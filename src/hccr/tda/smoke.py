"""Explicitly bounded synthetic workflow and real checkpoint inspection."""

from __future__ import annotations

import csv
from itertools import islice
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch
from PIL import Image, ImageDraw

from hccr.models import EfficientHCCRNet, load_model_from_run, read_checkpoint_metadata
from hccr.tda.cubical import cubical_features
from hccr.tda.experiment import PilotConfig, PlannedImages, Stage2Study
from hccr.tda.features import digest_file, write_json
from hccr.tda.projection import FrozenStage2Adapter, ScalarProjection


def create_synthetic_inputs(root: Path) -> tuple[Path, Path]:
    """Create a deterministic two-class checkpoint and 32 small labeled images."""
    torch.manual_seed(7)
    run = root / "baseline"
    run.mkdir(parents=True)
    model_settings = {
        "name": "efficient_hccr",
        "num_classes": 2,
        "width": 4,
        "stage_depths": [1, 1, 1],
        "embedding_dim": 16,
        "backbone_output_channels": 16,
        "dropout": 0.1,
        "classification_head": "cosface",
        "logit_scale": 32.0,
        "angular_margin": 0.1,
    }
    model = EfficientHCCRNet(
        **{key: value for key, value in model_settings.items() if key != "name"}
    )
    torch.save(model.state_dict(), run / "checkpoint.pt")
    write_json(
        run / "checkpoint_metadata.json",
        {
            "schema_version": 8,
            "model": model_settings,
            "preprocess": {"image_size": 16, "margin": 2},
            "evidence": "synthetic_random_weights_not_research",
        },
    )
    write_json(run / "labels.json", {"labels": {"0": "一", "1": "中"}})
    manifest = root / "data" / "processed" / "manifest.csv"
    manifest.parent.mkdir(parents=True)
    rows = []
    for split, count in (("train", 8), ("validation", 4), ("test", 2), ("icdar", 2)):
        for label in range(2):
            for index in range(count):
                name = f"{split}-{label}-{index}"
                image = Image.new("L", (16, 16), color=255)
                draw = ImageDraw.Draw(image)
                if label == 0:
                    draw.line((2, 7, 13, 7), fill=index * 3, width=2)
                else:
                    draw.rectangle((4, 3, 11, 12), outline=index * 3, width=2)
                    draw.line((7, 1, 7, 14), fill=0)
                image_path = root / "data" / "raw" / f"{name}.png"
                image_path.parent.mkdir(exist_ok=True)
                image.save(image_path)
                rows.append(
                    {
                        "sample_id": name,
                        "source_file": f"raw/{name}.png",
                        "writer_id": f"writer-{split}",
                        "unicode_label": "一" if label == 0 else "中",
                        "class_id": str(label),
                        "split": split,
                    }
                )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run, manifest


def synthetic_smoke(output: Path | None = None) -> dict:
    root = output or Path(mkdtemp(prefix="hccr-stage2-smoke-"))
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("smoke output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    run, manifest = create_synthetic_inputs(root)
    config = PilotConfig(
        str(run),
        str(manifest),
        str(root / "study"),
        max_classes=2,
        batch_size=4,
        projection_epochs=3,
        projection_learning_rate=0.01,
        calibration_samples=8,
        head_epochs=2,
        head_batch_size=8,
        device="cpu",
    )
    study = Stage2Study(config)
    study.fit_projection()
    selection = study.compare()
    heldout = study.evaluate(("test", "icdar"))
    report = {
        "evidence": "synthetic_smoke_only_not_research_accuracy",
        "output": str(root.resolve()),
        "projection_epochs": 3,
        "head_epochs": 2,
        "train_samples": 16,
        "validation_samples": 8,
        "test_samples": 4,
        "icdar_samples": 4,
        "variants": list(selection["variants"]),
        "heldout_splits": list(heldout["held_out"]),
        "projection_weights_updated": selection["projection_training"][
            "weights_updated"
        ],
        "research_accuracy_gain_established": False,
    }
    write_json(root / "smoke_report.json", report)
    return report


@torch.no_grad()
def real_smoke(config: PilotConfig, output: Path | None = None) -> dict:
    """Inspect four TRAIN images only; never extract a full dataset or train."""
    root = output or Path(mkdtemp(prefix="hccr-stage2-real-"))
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("real-smoke output must be a new or empty directory")
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu" if config.device == "auto" else config.device
    )
    run, manifest = Path(config.run_dir), Path(config.manifest)
    model = load_model_from_run(run, device).eval()
    metadata = read_checkpoint_metadata(run)
    from hccr.preprocessing import EvalPreprocessor

    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(
            islice(
                (row for row in csv.DictReader(handle) if row["split"] == "train"), 4
            )
        )
    if not rows:
        raise ValueError("real smoke requires at least one local TRAIN image")
    dataset = PlannedImages(
        manifest,
        rows,
        EvalPreprocessor(**metadata["preprocess"]),
        {row["unicode_label"]: 0 for row in rows},
        config.dataset_backend,
        digest_file(manifest),
    )
    images = torch.stack([dataset[index][0] for index in range(len(dataset))]).to(
        device
    )
    adapter = FrozenStage2Adapter(model).eval()
    embedding, stage, logits = adapter(images)
    original = model(images)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    projection = (
        ScalarProjection(stage.shape[1]).to(device).eval().requires_grad_(False)
    )
    maps = projection(stage)[:, 0].double().cpu().numpy()
    results = [cubical_features(array) for array in maps]
    np.save(root / "fixed_mean_maps.npy", maps)
    for index, result in enumerate(results):
        for dimension in (0, 1):
            np.save(
                root / f"sample-{index}-h{dimension}.npy", result.diagrams[dimension]
            )
    report = {
        "evidence": "bounded_real_shape_and_parity_check_not_accuracy_evaluation",
        "samples": len(rows),
        "baseline_run": str(run.resolve()),
        "checkpoint_sha256": digest_file(run / "checkpoint.pt"),
        "manifest_sha256": digest_file(manifest),
        "sample_ids": [row["sample_id"] for row in rows],
        "split": "train",
        "stage2_shape": list(stage.shape),
        "scalar_shape": [len(rows), 1, *maps.shape[1:]],
        "embedding_shape": list(embedding.shape),
        "baseline_logit_max_abs_error": float((logits - original).abs().max()),
        "projection": "fixed_channel_mean_not_learned",
        "calibration": "none_inspection_only",
        "ph_features": [result.features.tolist() for result in results],
        "output": str(root.resolve()),
    }
    write_json(root / "real_smoke_report.json", report)
    return report
