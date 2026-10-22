from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from scripts.reprofile_checkpoints import profile_run, write_summary

from hccr.artifacts import write_json
from hccr.artifacts.checkpoints import save_checkpoint
from hccr.models import EfficientHCCRNet


def test_reprofile_reads_real_checkpoint_without_merging_source_tables(tmp_path: Path):
    source = tmp_path / "experiments"
    run = source / "run-1"
    output = tmp_path / "artifacts"
    model = EfficientHCCRNet(num_classes=2, width=4, stage_depths=(1, 1, 1))
    save_checkpoint(
        model,
        run,
        {
            "model": {
                "name": "efficient_hccr",
                "num_classes": 2,
                "width": 4,
                "stage_depths": [1, 1, 1],
            },
            "preprocess": {"image_size": 16},
        },
    )
    write_json(run / "config.json", {"seed": 7})
    legacy = "run_id,legacy_metric\nrun-1,42\n"
    (source / "experiment_summary.csv").write_text(legacy, encoding="utf-8")
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        row = profile_run(run, output, "cpu", 1, 1, 1, False)
        with patch("scripts.reprofile_checkpoints.profile_model") as measure:
            resumed = profile_run(run, output, "cpu", 1, 1, 1, False)
        measure.assert_not_called()
    finally:
        torch.set_num_threads(original_threads)
    assert resumed == row
    assert row["parameter_count"] > 0
    assert row["eager_batch1_p95_ms"] > 0
    assert not any("optimized" in name for name in row)
    write_summary(
        output,
        [row],
        [],
        Namespace(
            runs_root=source,
            warmup_iterations=1,
            benchmark_iterations=1,
            benchmark_repetitions=1,
        ),
        "cpu",
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["schema_version"] == 2
    assert (output / "summary.csv").is_file()
    assert not (output / "experiment_summary.csv").exists()
    assert (source / "experiment_summary.csv").read_text() == legacy
