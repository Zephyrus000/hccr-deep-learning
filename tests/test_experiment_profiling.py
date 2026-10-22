from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch

from hccr.artifacts import write_json
from hccr.artifacts.checkpoints import file_digest
from hccr.benchmarking.profiling import _device_metadata
from hccr.experiments.runner import _profile_run
from hccr.experiments.spec import ExperimentJob, ExperimentSpec, ExperimentVariant


def make_run(root: Path):
    run = root / "run"
    run.mkdir()
    model = torch.nn.Linear(2, 2)
    torch.save(model.state_dict(), run / "checkpoint.pt")
    with torch.no_grad():
        model.weight.fill_(0.25)
    torch.save(model.state_dict(), run / "checkpoint_recalibrated.pt")
    write_json(
        run / "checkpoint_metadata.json",
        {
            "model": {"num_classes": 2},
            "preprocess": {"image_size": 4},
        },
    )
    write_json(
        run / "metrics.json",
        {
            "validation": {"selected_checkpoint": "checkpoint_recalibrated.pt"},
        },
    )
    profile = {
        "schema_version": 2,
        "measurement_stage": "selected_checkpoint",
        "checkpoint_name": "checkpoint_recalibrated",
        "checkpoint_sha256": file_digest(run / "checkpoint_recalibrated.pt"),
        "device": "cpu",
        "device_metadata": _device_metadata("cpu"),
        "full_class_projection": {"num_classes": 7186},
        "benchmark_protocol": {
            "warmup_iterations": 1,
            "timed_iterations": 2,
            "repetitions": 1,
        },
        "inference_benchmarks": [{"batch_size": 1, "latency_p95_ms": 1.0}],
    }
    write_json(run / "resource_profile.json", profile)
    spec = ExperimentSpec(
        "study",
        root / "manifest.csv",
        root,
        (7,),
        {},
        (ExperimentVariant("default", {}),),
    )
    job = ExperimentJob(
        "default/seed-7",
        "default",
        7,
        {
            "benchmark_warmup_iterations": 1,
            "benchmark_iterations": 2,
            "benchmark_repetitions": 1,
        },
        (),
    )
    return run, spec, job, profile


def test_matching_selected_checkpoint_profile_is_reused(tmp_path):
    run, spec, job, cached = make_run(tmp_path)
    with (
        patch("hccr.experiments.runner.load_model_from_run") as load,
        patch("hccr.experiments.runner.profile_model") as measure,
    ):
        result = _profile_run(spec, job, run, tmp_path / "sweep")
    load.assert_not_called()
    measure.assert_not_called()
    assert result["cpu"]["trained_head"] == cached
    assert not (tmp_path / "sweep").exists()


def test_changed_checkpoint_is_measured_once_without_synthetic_head(tmp_path):
    run, spec, job, _ = make_run(tmp_path)
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(0.75)
    torch.save(model.state_dict(), run / "checkpoint_recalibrated.pt")
    loaded = torch.nn.Linear(2, 2)
    measured = []

    def measure(model, *_args, **_kwargs):
        measured.append(model.weight.detach().clone())
        return {"schema_version": 2}

    with (
        patch(
            "hccr.experiments.runner.load_model_from_run", return_value=loaded
        ) as load,
        patch("hccr.experiments.runner.profile_model", side_effect=measure) as profile,
    ):
        result = _profile_run(spec, job, run, tmp_path / "sweep")
    load.assert_called_once_with(run, "cpu")
    profile.assert_called_once()
    assert torch.equal(measured[0], model.weight)
    assert "full_class_head" not in result["cpu"]
    saved = json.loads(
        (
            tmp_path
            / "sweep/profiles/default/seed-7/cpu/trained_head/resource_profile.json"
        ).read_text()
    )
    assert saved["checkpoint_sha256"] == file_digest(run / "checkpoint_recalibrated.pt")


def test_runtime_or_protocol_changes_invalidate_cached_profile(tmp_path):
    run, spec, job, cached = make_run(tmp_path)
    cached["device_metadata"]["host"] = "another-host"
    write_json(run / "resource_profile.json", cached)
    with (
        patch(
            "hccr.experiments.runner.load_model_from_run",
            return_value=torch.nn.Linear(2, 2),
        ),
        patch("hccr.experiments.runner.profile_model", return_value={}) as measure,
    ):
        _profile_run(spec, job, run, tmp_path / "sweep")
    measure.assert_called_once()
    cached["device_metadata"] = _device_metadata("cpu")
    cached["benchmark_protocol"]["timed_iterations"] = 999
    write_json(run / "resource_profile.json", cached)
    with (
        patch(
            "hccr.experiments.runner.load_model_from_run",
            return_value=torch.nn.Linear(2, 2),
        ),
        patch("hccr.experiments.runner.profile_model", return_value={}) as measure,
    ):
        _profile_run(spec, job, run, tmp_path / "sweep")
    measure.assert_called_once()
