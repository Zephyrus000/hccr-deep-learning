"""Behavior checks for supervision gradients, image topology, and deployment."""

import csv
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from hccr.data.lmdb_store import build_lmdb_image_store
from hccr.models.artifacts import load_model_from_run
from hccr.models.descriptor_supervision import DescriptorSupervision, SupervisionConfig
from hccr.models.efficient_hccr import EfficientHCCRNet
from hccr.training.descriptor_supervision import (
    DescriptorTrainingConfig,
    prepare_supervision_targets,
    read_supervision_config,
    run_descriptor_training,
)
from hccr.training.descriptor_targets import (
    TDA_TARGET_NAMES,
    TargetScaler,
    image_tda_targets,
)


def small_model(method):
    torch.manual_seed(7)
    cnn = EfficientHCCRNet(
        num_classes=3, width=2, stage_depths=(1, 1, 1), embedding_dim=320, dropout=0
    )
    return DescriptorSupervision(cnn, SupervisionConfig(method), image_size=16)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_hog_auxiliary_gradient_updates_stage2_without_classifier_or_target_gradient():
    model = small_model("hog").train()
    images = torch.rand(3, 1, 16, 16)
    targets = torch.tensor([0, 1, 2])
    before = model.cnn.features[1].block[0].weight.detach().clone()
    losses = model.training_losses(images, targets)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    losses["hog_loss"].backward()
    stage2 = model.cnn.features[1].block[0].weight
    assert stage2.grad is not None and stage2.grad.abs().sum() > 0
    assert model.cnn.classifier.weight.grad is None
    assert all(
        parameter.grad is None for parameter in model.cnn.features[2].parameters()
    )
    optimizer.step()
    assert not torch.equal(stage2, before)


def test_tda_auxiliary_updates_embedding_and_detaches_targets():
    model = small_model("tda").train()
    target = torch.randn(3, 12, requires_grad=True)
    losses = model.training_losses(
        torch.rand(3, 1, 16, 16),
        torch.tensor([0, 1, 2]),
        tda_targets=target,
        tda_active=torch.ones(12, dtype=torch.bool),
    )
    losses["tda_loss"].backward()
    assert model.cnn.embedding_projection.weight.grad.abs().sum() > 0
    assert model.cnn.stem[0].weight.grad.abs().sum() > 0
    assert model.cnn.classifier.weight.grad is None
    assert target.grad is None


def test_auxiliary_weighting_and_classification_path_are_exact():
    model = small_model("both").eval()
    images, targets = torch.rand(3, 1, 16, 16), torch.tensor([0, 1, 2])
    result = model.training_losses(
        images,
        targets,
        tda_targets=torch.ones(3, 12),
        tda_active=torch.ones(12, dtype=torch.bool),
        epoch=1,
    )
    torch.testing.assert_close(
        result["loss"],
        result["classification_loss"]
        + 0.05 * (result["hog_loss"] + result["tda_loss"]),
    )
    with patch.object(
        model.hog, "forward", side_effect=AssertionError("inference used HOG")
    ):
        torch.testing.assert_close(model(images), model.cnn(images), rtol=0, atol=0)
    assert not any("hog" in name or "tda" in name for name in model.cnn.state_dict())


def test_spatial_hog_layout_matches_descriptor_flatten_order():
    model = DescriptorSupervision(
        EfficientHCCRNet(num_classes=3, width=2, embedding_dim=320),
        SupervisionConfig("hog"),
        image_size=96,
    ).eval()
    # A 36-channel constant map becomes the 36 values of each spatial HOG block.
    with torch.no_grad():
        model.hog_head[0].weight.zero_()
        model.hog_head[0].bias.copy_(torch.arange(36))
    prediction = model.hog_head(torch.zeros(1, 4, 24, 24))
    flat = prediction.permute(0, 2, 3, 1).flatten(1)
    torch.testing.assert_close(flat, torch.arange(36).float().repeat(121)[None])
    assert flat.shape == (1, 4356)


def test_topology_targets_distinguish_a_hole_and_exclude_essential_counts():
    images = torch.ones(2, 1, 16, 16)
    images[0, 0, 3:13, 3:13] = 0
    images[1, 0, 3:13, 3:13] = 0
    images[1, 0, 6:10, 6:10] = 1
    targets = image_tda_targets(images)
    assert targets.shape == (2, 12)
    assert "essential" not in " ".join(TDA_TARGET_NAMES)
    assert targets[0, 6] == 0  # H1 finite count for a filled square.
    assert targets[1, 6] == 1  # H1 finite count for a dark ring.
    assert targets[1, 7] == 1  # Total H1 lifetime across intensities 0..1.
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        image_tda_targets(images + 1)


def test_scaler_uses_only_training_statistics_and_masks_constant_dimensions():
    train = np.zeros((2, 12), dtype=np.float32)
    train[:, 0] = [1, 3]
    scaler = TargetScaler.fit(train)
    assert scaler.mean[0] == 2
    assert scaler.std[0] == 1
    assert scaler.active.tolist() == [True] + [False] * 11
    validation = np.full((1, 12), 10, dtype=np.float32)
    scaled = scaler.transform(validation)
    torch.testing.assert_close(scaled, torch.tensor([[8.0] + [0.0] * 11]))
    assert scaler.mean[0] == 2
    with pytest.raises(ValueError, match="constant"):
        TargetScaler.fit(np.ones((2, 12), dtype=np.float32))


def make_run(root):
    source = root / "source"
    source.mkdir()
    labels = {"labels": {str(index): chr(0x4E00 + index) for index in range(3)}}
    for folder in (source, root):
        (folder / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    rows = []
    for index in range(8):
        filename = f"image-{index}.png"
        image = np.random.default_rng(index).integers(0, 256, (16, 16), dtype=np.uint8)
        Image.fromarray(image).save(root / filename)
        rows.append(
            {
                "sample_id": str(index),
                "source_file": filename,
                "writer_id": "train-writer" if index < 6 else "validation-writer",
                "unicode_label": chr(0x4E00 + index % 3),
                "class_id": str(index % 3),
                "split": "train" if index < 6 else "validation",
            }
        )
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    build_lmdb_image_store(manifest, rows, root, root / "images.lmdb")
    cnn = small_model("none").cnn
    metadata = {
        "model": {
            "name": "efficient_hccr",
            "num_classes": 3,
            "width": 2,
            "stage_depths": [1, 1, 1],
            "embedding_dim": 320,
            "dropout": 0,
        },
        "preprocess": {"image_size": 16, "margin": 2},
        "training": {"precision": {"resolved": "float32"}},
    }
    (source / "checkpoint_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    torch.save(cnn.state_dict(), source / "checkpoint.pt")
    return DescriptorTrainingConfig(
        source_run=str(source),
        manifest=str(manifest),
        output_dir=str(root / "run"),
        epochs=2,
        batch_size=2,
        num_workers=0,
        cpu_threads=1,
        device="cpu",
    )


@pytest.mark.parametrize("method", ["none", "hog", "tda"])
def test_real_lmdb_runner_exports_plain_cnn_and_resumes_without_extra_steps(method):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = replace(make_run(root), method=method, tda_cache=str(root / "cache"))
        if method == "tda":
            receipt = prepare_supervision_targets(config)
            assert receipt["train"]["samples"] == 6
            assert receipt["validation"]["samples"] == 2
        result = run_descriptor_training(config)
        assert result["status"] == "completed"
        assert result["test_status"] == result["icdar_status"] == "not_run"
        output = Path(config.output_dir)
        state = torch.load(output / "latest.pt", weights_only=True)
        assert [row["train"]["samples"] for row in state["history"]] == [6, 6]
        assert [row["train"]["optimizer_steps"] for row in state["history"]] == [3, 3]
        assert any(
            not torch.equal(
                value, small_model("none").cnn.state_dict()[key.removeprefix("cnn.")]
            )
            for key, value in state["model"].items()
            if key.startswith("cnn.features.")
        )
        deployed = load_model_from_run(output).eval()
        image = torch.rand(2, 1, 16, 16)
        stored = torch.load(output / "checkpoint.pt", weights_only=True)
        assert not any(key.startswith(("cnn.", "hog", "tda")) for key in stored)
        original_shape = load_model_from_run(Path(config.source_run)).eval()
        original_shape.load_state_dict(stored)
        torch.testing.assert_close(
            deployed(image), original_shape(image), rtol=0, atol=0
        )
        assert run_descriptor_training(config, resume=True) == result
        with pytest.raises(ValueError, match="resume configuration"):
            run_descriptor_training(replace(config, learning_rate=2e-4), resume=True)


def test_configuration_and_unprepared_cache_fail_before_training():
    config_path = Path("configs/experiment/descriptor_supervision.yaml")
    configs = [
        read_supervision_config(config_path, method)
        for method in ("none", "hog", "tda")
    ]
    assert {(c.epochs, c.batch_size, c.seed, c.learning_rate) for c in configs} == {
        (20, 128, 7, 1e-4)
    }
    assert len({c.output_dir for c in configs}) == 3
    with pytest.raises(ValueError, match="unaugmented"):
        replace(configs[2], augmentation="flip")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = replace(make_run(root), method="tda", tda_cache=str(root / "missing"))
        with pytest.raises(ValueError, match="incomplete"):
            run_descriptor_training(config)
        assert not Path(config.output_dir).exists()


def test_fp32_100_epoch_plan_resolves_distinct_seed_outputs_and_requires_teacher():
    path = Path("configs/experiment/descriptor_supervision_100ep.yaml")
    with pytest.raises(ValueError, match="native FP32 teacher"):
        read_supervision_config(path, "tda", seed=17)
    configs = [
        read_supervision_config(path, method, seed=seed, source_run=f"teacher-{seed}")
        for seed in (7, 17, 29)
        for method in ("none", "hog", "tda")
    ]
    assert len({config.output_dir for config in configs}) == 9
    for config in configs:
        assert config.source_run == f"teacher-{config.seed}"
        assert (config.epochs, config.batch_size, config.precision) == (
            100,
            256,
            "float32",
        )
        assert (config.training_cache, config.augmentation) == ("cuda", "gpu")


def test_training_rejects_teacher_that_has_not_finished_its_epoch_budget():
    with tempfile.TemporaryDirectory() as directory:
        config = replace(make_run(Path(directory)), require_completed_teacher=True)
        source = Path(config.source_run)
        (source / "config.json").write_text(json.dumps({"seed": 7, "epochs": 2}))
        (source / "curves.json").write_text(json.dumps({"epochs": [{"epoch": 1}]}))
        (source / "metrics.json").write_text(json.dumps({"validation": {}}))
        with pytest.raises(ValueError, match="completed matched-seed teacher"):
            run_descriptor_training(config)
        assert not Path(config.output_dir).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_cache_augments_classification_and_aligns_clean_tda_targets():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = replace(
            make_run(root),
            method="tda",
            tda_cache=str(root / "cache"),
            epochs=1,
            device="cuda",
            precision="float32",
            training_cache="cuda",
            training_cache_reserve_gib=0.1,
            augmentation="gpu",
        )
        prepare_supervision_targets(config)
        scaler = TargetScaler.fit(np.load(root / "cache/train/targets.npy"))
        classify, topology = (
            DescriptorSupervision.training_losses,
            DescriptorSupervision.clean_topology_loss,
        )
        seen = []
        clean_gradients = []

        def track_classification(model, images, labels, **kwargs):
            assert images.is_cuda and images.dtype == torch.float32
            assert kwargs["separate_tda_view"]
            seen.append(images.detach())
            return classify(model, images, labels, **kwargs)

        def track_topology(model, images, targets, active):
            assert images.is_cuda and targets.is_cuda
            expected = scaler.transform(image_tda_targets(images)).cuda()
            torch.testing.assert_close(targets, expected, atol=1e-5, rtol=1e-5)
            assert not torch.equal(images, seen[-1])
            running_stats = [
                (buffer, buffer.clone())
                for name, buffer in model.cnn.named_buffers()
                if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
            ]
            batch_norm = model.cnn.stem[1]

            def check_normalization(module, inputs, output):
                expected = torch.nn.functional.batch_norm(
                    inputs[0],
                    None,
                    None,
                    module.weight,
                    module.bias,
                    training=True,
                    eps=module.eps,
                )
                torch.testing.assert_close(output, expected)
                output.register_hook(lambda grad: clean_gradients.append(grad.norm()))

            with batch_norm.register_forward_hook(check_normalization):
                loss = topology(model, images, targets, active)
            for buffer, previous in running_stats:
                torch.testing.assert_close(buffer, previous, atol=0, rtol=0)
            return loss

        with (
            patch.object(
                DescriptorSupervision, "training_losses", track_classification
            ),
            patch.object(DescriptorSupervision, "clean_topology_loss", track_topology),
        ):
            run_descriptor_training(config)
        output = Path(config.output_dir)
        assert (
            json.loads((output / "training_cache.json").read_text())["placement"]
            == "cuda"
        )
        state = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        assert state["model"]["cnn.stem.1.num_batches_tracked"].item() == len(seen) == 3
        assert len(clean_gradients) == len(seen)
        assert all(norm > 0 for norm in clean_gradients)
        assert state["history"][0]["train"]["tda_loss"] > 0


def test_interrupted_epoch_resume_matches_uninterrupted_weights():
    from hccr.training.descriptor_supervision import train_supervision_epoch

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = replace(make_run(root), method="hog")
        run_descriptor_training(config)
        expected = torch.load(Path(config.output_dir) / "latest.pt", weights_only=True)
        interrupted = replace(config, output_dir=str(root / "interrupted"))

        def stop_before_epoch_two(*args):
            if args[5] == 2:
                raise RuntimeError("simulated interruption")
            return train_supervision_epoch(*args)

        with (
            patch(
                "hccr.training.descriptor_supervision.train_supervision_epoch",
                side_effect=stop_before_epoch_two,
            ),
            pytest.raises(RuntimeError, match="simulated interruption"),
        ):
            run_descriptor_training(interrupted)
        run_descriptor_training(interrupted, resume=True)
        actual = torch.load(
            Path(interrupted.output_dir) / "latest.pt", weights_only=True
        )
        assert actual["epoch"] == 2
        assert actual["history"] == expected["history"]
        assert actual["best_epoch"] == expected["best_epoch"]
        for name, value in actual["model"].items():
            torch.testing.assert_close(value, expected["model"][name], rtol=0, atol=0)
        for name, value in actual["selected_cnn"].items():
            torch.testing.assert_close(
                value, expected["selected_cnn"][name], rtol=0, atol=0
            )


def test_tda_cache_reuse_checks_image_target_provenance_and_file_content():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = replace(make_run(root), method="tda", tda_cache=str(root / "cache"))
        first = prepare_supervision_targets(config)
        assert prepare_supervision_targets(config) == first
        path = root / "cache/train/targets.npy"
        values = np.load(path, mmap_mode="r+")
        values[0, 0] += 1
        values.flush()
        del values
        with pytest.raises(ValueError, match="checksum changed"):
            run_descriptor_training(config)
        assert not Path(config.output_dir).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bfloat16_backward_keeps_auxiliary_and_cosface_arithmetic_float32():
    model = small_model("both").cuda().train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model.training_losses(
            torch.rand(3, 1, 16, 16, device="cuda"),
            torch.tensor([0, 1, 2], device="cuda"),
            tda_targets=torch.ones(3, 12, device="cuda"),
            tda_active=torch.ones(12, dtype=torch.bool, device="cuda"),
        )
    for name, value in losses.items():
        assert value.dtype == torch.float32, name
    losses["loss"].backward()
    assert model.cnn.features[1].block[0].weight.grad.abs().sum() > 0
    assert model.cnn.classifier.weight.grad.abs().sum() > 0
