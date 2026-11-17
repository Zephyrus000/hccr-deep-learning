import json

import pytest
import torch
from scripts.prepare_classifier_stage import prepare_classifier_stage

from hccr.artifacts.checkpoints import file_digest
from hccr.cli import build_parser
from hccr.commands.train import config_from_arguments
from hccr.models import load_model_from_run
from hccr.training.workflow import (
    _build_training_model,
    _model_metadata,
    _validate_training_config,
)


@pytest.mark.parametrize("pooling,grid", [("gwoap", 1), ("gwap", 3)])
def test_weighted_pooling_training_and_reload(tmp_path, pooling, grid):
    args = build_parser().parse_args(
        [
            "train",
            "--manifest",
            "manifest.csv",
            "--num-classes",
            "3",
            "--width",
            "4",
            "--image-size",
            "17",
            "--pooling",
            pooling,
        ]
    )
    config = config_from_arguments(args)
    _validate_training_config(config)
    model = _build_training_model(config, 3).eval()
    features = torch.linspace(-1, 1, 2 * 16 * 3 * 3).reshape(2, 16, 3, 3)
    mean = features.mean(dim=(-2, -1), keepdim=True)
    torch.testing.assert_close(model.pool(features), mean)
    (delta,) = model.pool.parameters()
    assert delta.numel() == 16 * grid * grid

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.pool(features).square().sum().backward()
    assert delta.grad is not None and torch.count_nonzero(delta.grad)
    optimizer.step()
    assert torch.count_nonzero(delta)
    with torch.no_grad():
        delta.flatten()[0] = 0.25
        weights = 1 + delta.reshape(16, grid, grid)
        expected = (features * weights).mean(dim=(-2, -1), keepdim=True)
        torch.testing.assert_close(model.pool(features), expected)

        manifest = tmp_path / "manifest.csv"
        manifest.write_text("synthetic fixture", encoding="utf-8")
        metadata = {
            "model": _model_metadata(model, config, 3),
            "manifest_digest": file_digest(manifest),
            "preprocess": {"image_size": config.image_size},
            "training": {"seed": config.seed, "batch_size": config.batch_size},
        }
        (tmp_path / "checkpoint_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        torch.save(model.state_dict(), tmp_path / "checkpoint.pt")
        restored = load_model_from_run(tmp_path).eval()
        inputs = torch.randn(2, 1, 17, 17)
        torch.testing.assert_close(restored(inputs), model(inputs))

    pilot = prepare_classifier_stage("pilots", tmp_path, manifest, rank=2)
    assert pilot["base_args"]["pooling"] == pooling
    assert pilot["base_args"]["image_size"] == 17


def test_gwap_rejects_mismatched_feature_grid_and_baseline():
    args = build_parser().parse_args(
        [
            "train",
            "--manifest",
            "manifest.csv",
            "--num-classes",
            "3",
            "--pooling",
            "gwap",
        ]
    )
    config = config_from_arguments(args)
    model = _build_training_model(config, 3)
    with pytest.raises(ValueError, match="GWAP.*spatial"):
        model.pool(torch.ones(1, model.backbone_output_channels, 1, 1))

    args.model = "resnet18"
    with pytest.raises(ValueError, match="pooling.*efficient_hccr"):
        _validate_training_config(config_from_arguments(args))
