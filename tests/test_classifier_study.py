from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts.run_classifier_study import (
    _study_lock,
    completed_pilots,
    prepare_study,
    teacher_identity,
)

from hccr.artifacts.checkpoints import file_digest


def test_direct_script_entrypoint():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/run_classifier_study.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "--teacher-run" in result.stdout
    assert "--through" in result.stdout


def make_teachers(root: Path) -> tuple[list[Path], Path]:
    manifest = root / "manifest.csv"
    manifest.write_text("official-manifest-fixture", encoding="utf-8")
    runs = []
    for seed in (7, 17, 29):
        run = root / f"teacher-{seed}"
        run.mkdir()
        metadata = {
            "manifest_digest": file_digest(manifest),
            "model": {
                "name": "efficient_hccr",
                "num_classes": 3755,
                "embedding_dim": 320,
                "classifier_structure": "dense",
                "classification_head": "cosface",
                "logit_scale": 32.0,
                "angular_margin": 0.1,
            },
            "preprocess": {"image_size": 96},
            "training": {
                "seed": seed,
                "batch_size": 256,
                "reference_batch_size": 256,
                "optimizer": "adamw",
                "base_learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "optimizer_step_policy": "configured_epochs",
                "resolved_epochs": 20,
                "total_optimizer_steps": 100,
                "completed_optimizer_steps": 95,
            },
            "selection": {"epoch": 19, "policy": "best_validation"},
        }
        (run / "checkpoint_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (run / "checkpoint.pt").write_bytes(f"weights-{seed}".encode())
        (run / "labels.json").write_text(
            json.dumps({"labels": {str(i): chr(0x4E00 + i) for i in range(3755)}}),
            encoding="utf-8",
        )
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "test": {"status": "not_run"},
                    "icdar": {"status": "not_run"},
                }
            ),
            encoding="utf-8",
        )
        (run / "curves.json").write_text(
            json.dumps(
                {
                    "epochs": [
                        {"epoch": e, "optimizer_step_end": 5 * e} for e in range(1, 21)
                    ]
                }
            ),
            encoding="utf-8",
        )
        runs.append(run)
    return runs, manifest


def test_plan_needs_no_gpu_or_training_and_preserves_selected_teacher(tmp_path):
    runs, manifest = make_teachers(tmp_path)
    output = tmp_path / "study"
    state = prepare_study(runs, manifest, output, rank=2, k=2)
    assert state["status"] == "planned"
    assert state["planned_jobs"] == {"pilots": 9, "final": 15}
    assert not (output / "pilot-queue").exists()
    for teacher, config_path in zip(
        state["settings"]["teachers"], state["pilot_configs"], strict=True
    ):
        assert teacher["selected_epoch"] == 19
        assert teacher["completed_epochs"] == 20
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        assert len(config["variants"]) == 3
        assert config["base_args"]["freeze_backbone"] is True
        assert config["base_args"]["precision"] == "float32"
        assert config["base_args"]["dataset_backend"] == "lmdb"
        assert config["base_args"]["evaluation_policy"] == "validation_only"
    resumed = prepare_study(runs, manifest, output, rank=2, k=2, resume=True)
    assert resumed["identity"] == state["identity"]
    with pytest.raises(FileExistsError):
        prepare_study(runs, manifest, output, rank=2, k=2)
    with pytest.raises(ValueError, match="inputs/settings changed"):
        prepare_study(runs, manifest, output, rank=1, k=2, resume=True)


def test_refuse_incomplete_run_or_opened_test_and_missing_seed(tmp_path):
    runs, manifest = make_teachers(tmp_path)
    with pytest.raises(ValueError, match="exactly teacher seeds"):
        prepare_study(runs[:2], manifest, tmp_path / "partial", rank=2, k=2)
    path = runs[0] / "curves.json"
    curves = json.loads(path.read_text(encoding="utf-8"))
    curves["epochs"].pop()
    path.write_text(json.dumps(curves), encoding="utf-8")
    with pytest.raises(ValueError, match="declared fitting budget"):
        teacher_identity(runs[0], file_digest(manifest))
    metrics = runs[1] / "metrics.json"
    metrics.write_text(
        json.dumps({"test": {"status": "completed"}, "icdar": {"status": "not_run"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unopened"):
        teacher_identity(runs[1], file_digest(manifest))


@pytest.mark.parametrize(
    ("field", "value"),
    [("base_learning_rate", 1.0), ("max_train_samples", 1), ("resolved_epochs", 1)],
)
def test_refuse_smoke_or_mismatched_teacher_recipe(tmp_path, field, value):
    runs, manifest = make_teachers(tmp_path)
    path = runs[0] / "checkpoint_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["training"][field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="locked 20-epoch full-fit"):
        teacher_identity(runs[0], file_digest(manifest))


def test_resume_refuses_changed_completed_output_and_concurrent_owner(tmp_path):
    runs, manifest = make_teachers(tmp_path)
    output = tmp_path / "study"
    prepare_study(runs, manifest, output, rank=2, k=2)
    report = output / "h1.json"
    report.write_text("verified", encoding="utf-8")
    status = output / "study.json"
    state = json.loads(status.read_text(encoding="utf-8"))
    state["artifacts"][str(report)] = file_digest(report)
    status.write_text(json.dumps(state), encoding="utf-8")
    report.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or disappeared"):
        prepare_study(runs, manifest, output, rank=2, k=2, resume=True)
    with _study_lock(output), pytest.raises(OSError), _study_lock(output):
        pytest.fail("Concurrent owner acquired lock")
    with _study_lock(output):
        pass


def test_pilot_resolution_uses_completed_records_and_rejects_missing_pair(tmp_path):
    queue = {"status": "completed", "jobs": []}
    for seed in (7, 17, 29):
        for role, rank in (("dense", 2), ("low_rank", 2), ("near_budget", 4)):
            folder = tmp_path / f"{seed}-{role}"
            run = folder / "run"
            run.mkdir(parents=True)
            structure = "dense" if role == "dense" else "low_rank"
            (run / "checkpoint_metadata.json").write_text(
                json.dumps(
                    {
                        "training": {"seed": seed},
                        "model": {
                            "classifier_structure": structure,
                            "classifier_rank": rank,
                        },
                    }
                ),
                encoding="utf-8",
            )
            child = folder / "status.json"
            child.write_text(
                json.dumps(
                    {"records": [{"status": "completed", "run_path": str(run)}]}
                ),
                encoding="utf-8",
            )
            queue["jobs"].append(
                {
                    "status": "completed",
                    "child_status": str(child),
                    "child": {"output_dir": str(folder)},
                }
            )
    result = completed_pilots(queue, 2)
    assert set(result[7]) == {"dense", "low_rank", "near_budget"}
    assert result[29]["near_budget"].name == "run"
    queue["jobs"].pop()
    with pytest.raises(ValueError, match="Missing"):
        completed_pilots(queue, 2)
