import json

import torch
from torch import nn

from hccr.models import build_model, load_model_from_run
from hccr.models.mscs_asa import AttentionSpatialAggregation, MultiScaleShuffle


def test_attention_spatial_aggregation_sums_gated_channels():
    aggregation = AttentionSpatialAggregation(16)
    with torch.no_grad():
        for parameter in aggregation.parameters():
            parameter.zero_()
    # sigmoid(0) gates each channel by 1/2 before summing its six pixels.
    expected = torch.arange(16, dtype=torch.float32).view(1, 16) * 3
    features = (
        torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1).expand(1, 16, 2, 3)
    )
    torch.testing.assert_close(aggregation(features), expected)


def test_mscs_asa_training_and_normal_checkpoint_roundtrip(tmp_path):
    model = build_model("mscs_asa", num_classes=3755)
    assert model.embedding_dim == 448
    assert sum(parameter.numel() for parameter in model.parameters()) == 6_003_015
    stages = []

    def record_shape(_module, _inputs, output):
        stages.append(tuple(output.shape[1:]))

    handles = [
        module.register_forward_hook(record_shape)
        for module in model.backbone
        if isinstance(module, MultiScaleShuffle)
    ]
    images = torch.rand(2, 1, 96, 96)
    logits = model.training_logits(images, torch.tensor([0, 3754]))
    assert logits.shape == (2, 3755)
    assert stages == [(64, 48, 48), (96, 24, 24), (192, 12, 12), (384, 6, 6)]
    nn.functional.cross_entropy(logits, torch.tensor([0, 3754])).backward()
    assert model.backbone[4].spatial[0].weight.grad.abs().sum() > 0
    for handle in handles:
        handle.remove()

    metadata = {"model": {"name": "mscs_asa", "num_classes": 3755}}
    (tmp_path / "checkpoint_metadata.json").write_text(json.dumps(metadata))
    torch.save(model.state_dict(), tmp_path / "checkpoint.pt")
    with torch.inference_mode():
        deployed = load_model_from_run(tmp_path).eval()
        torch.testing.assert_close(
            deployed(images), model.eval()(images), rtol=0, atol=0
        )
