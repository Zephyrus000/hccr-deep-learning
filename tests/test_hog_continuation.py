from __future__ import annotations

import csv
import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hccr.artifacts.checkpoints import file_digest
from hccr.data.lmdb_store import build_lmdb_image_store
from hccr.models.efficient_hccr import EfficientHCCRNet
from hccr.models.hog_fusion import load_hog_fusion_run
from hccr.training.hog_continuation import continue_hog_training
from hccr.training.hog_fusion import run_hog_fusion


def test_continuation_restores_optimizer_and_freezes_selection_before_holdouts():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        labels = {"labels": {str(i): chr(0x4E00 + i) for i in range(3755)}}
        for folder in (source, root):
            (folder / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        rows = []
        for index in range(6):
            filename = f"{index}.png"
            Image.fromarray(
                np.random.default_rng(index).integers(0, 256, (20, 20), dtype=np.uint8)
            ).save(root / filename)
            rows.append(
                {
                    "sample_id": str(index),
                    "source_file": filename,
                    "class_id": str(index % 2),
                    "unicode_label": chr(0x4E00 + index % 2),
                    "split": "train" if index < 4 else "validation",
                    "writer_id": "train-writer" if index < 4 else "validation-writer",
                }
            )
        manifest = root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        build_lmdb_image_store(manifest, rows, root, root / "images.lmdb")
        cnn = EfficientHCCRNet(
            num_classes=3755, width=2, stage_depths=(1, 1, 1), embedding_dim=320
        )
        torch.save(cnn.state_dict(), source / "checkpoint.pt")
        (source / "checkpoint_metadata.json").write_text(
            json.dumps(
                {
                    "model": {
                        "name": "efficient_hccr",
                        "num_classes": 3755,
                        "width": 2,
                        "stage_depths": [1, 1, 1],
                        "embedding_dim": 320,
                    },
                    "preprocess": {"image_size": 96, "margin": 4},
                    "training": {"precision": {"resolved": "float32"}},
                }
            ),
            encoding="utf-8",
        )
        pilot = root / "pilot"
        run_hog_fusion(
            Namespace(
                source_run=str(source),
                manifest=str(manifest),
                output_dir=str(pilot),
                resume=False,
                seed=7,
                device="cpu",
                cpu_threads=1,
                epochs=1,
                learning_rate=0.001,
                batch_size=2,
                extract_batch_size=2,
                num_workers=0,
                min_baseline_top1=0,
            )
        )
        original_hash = file_digest(pilot / "latest.pt")
        args = Namespace(
            parent_run=str(pilot),
            manifest=str(manifest),
            output_dir=str(root / "extended"),
            resume=False,
            total_epochs=2,
            device="cpu",
            cpu_threads=1,
        )
        result = continue_hog_training(args)
        assert result["total_epochs"] == 2
        assert result["frozen_state_unchanged"] is True
        assert file_digest(pilot / "latest.pt") == original_hash
        plan = json.loads((root / "extended/holdout_plan.json").read_text("utf-8"))
        assert plan["selection_frozen"] is True
        assert plan["test_based_selection"] is False
        last = load_hog_fusion_run(root / "extended/last")
        for name, value in last.cnn.state_dict().items():
            assert torch.equal(value, cnn.state_dict()[name])
        before_resume = file_digest(root / "extended/latest.pt")
        args.resume = True
        continue_hog_training(args)
        assert before_resume == file_digest(root / "extended/latest.pt")
