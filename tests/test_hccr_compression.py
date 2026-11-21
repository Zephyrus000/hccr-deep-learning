"""Checks for the budget-matched embedding-bottleneck control."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from hccr.artifacts.checkpoints import file_digest
from hccr.models import build_model
from hccr.training.proposal import initialize_training_model
from hccr.training.workflow import TrainingConfig, _model_metadata


def test_embedding_control_inherits_both_factors_at_equal_budget():
    def model(structure, dim):
        return build_model(
            "efficient_hccr",
            num_classes=3,
            width=4,
            stage_depths=(1, 1, 1),
            backbone_output_channels=8,
            embedding_dim=dim,
            reparameterize_depthwise=False,
            classifier_rank=2,
            classification_head="cosface",
            classifier_structure=structure,
        )

    source = model("low_rank", 8)
    target = model("dense", 2)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "manifest.csv"
        manifest.write_text("manifest", encoding="utf-8")
        labels = ["a", "b", "c"]
        (root / "labels.json").write_text(
            json.dumps({"labels": dict(enumerate(labels))}), encoding="utf-8"
        )
        config = TrainingConfig(
            manifest,
            root,
            3,
            width=4,
            stage_depths=(1, 1, 1),
            backbone_output_channels=8,
            embedding_dim=8,
            reparameterize_depthwise=False,
            classifier_structure="low_rank",
            classifier_rank=2,
            classification_head="cosface",
        )
        metadata = {
            "model": _model_metadata(source, config, 3),
            "manifest_digest": file_digest(manifest),
            "labels_digest": file_digest(root / "labels.json"),
        }
        (root / "checkpoint_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        torch.save(source.state_dict(), root / "checkpoint.pt")
        info = initialize_training_model(target, root, manifest, labels)
    assert info["mode"] == "low_rank_pilot_to_embedding_bottleneck"
    torch.testing.assert_close(
        target.embedding_projection.weight, source.classifier.basis
    )
    torch.testing.assert_close(target.classifier.weight, source.classifier.coefficients)
    assert sum(p.numel() for p in target.parameters()) == sum(
        p.numel() for p in source.parameters()
    )
