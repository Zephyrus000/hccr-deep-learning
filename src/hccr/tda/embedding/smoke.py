"""Small synthetic cache fixture and end-to-end CLI check, not an accuracy study."""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import torch

from hccr.tda.features import save_cache, write_json

from .study import EmbeddingConfig, EmbeddingStudy


def create_synthetic_caches(root: Path) -> None:
    rng = np.random.default_rng(17)
    mapping = {"中": 0, "文": 1, "字": 2}
    for split, per_class in (
        ("train", 8),
        ("validation", 4),
        ("test", 3),
        ("icdar", 3),
    ):
        targets = np.repeat(np.arange(3, dtype=np.int64), per_class)
        cnn = rng.normal(0, 0.025, (len(targets), 320)).astype(np.float32)
        cnn[np.arange(len(targets)), targets] += 2
        identity = {
            "schema_version": 1,
            "checkpoint": {"sha256": "synthetic-not-a-trained-checkpoint"},
            "manifest": {"sha256": "synthetic-split-plan"},
            "class_mapping": mapping,
            "preprocessing": {"image_size": 96, "margin": 4},
            "extraction_precision": "float32",
            "split": split,
            "sample_ids": [f"{split}-{i:04}" for i in range(len(targets))],
        }
        save_cache(root / split, {"cnn": cnn, "targets": targets}, identity)


def synthetic_smoke(output: Path | None = None) -> dict:
    root = Path(mkdtemp(prefix="hccr-embedding-ph-")) if output is None else output
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("synthetic smoke requires an empty output directory")
    create_synthetic_caches(root / "source")
    config = EmbeddingConfig(
        source_root=str(root / "source"),
        output_dir=str(root / "study"),
        max_classes=3,
        bank_per_class=4,
        neighbors=3,
        query_batch_size=4,
        reference_batch_size=5,
        head_epochs=1,
        head_batch_size=8,
        alpha_grid=(0.5, 1.0),
        knn_mix_grid=(0.0, 0.5, 1.0),
        device="cpu",
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with EmbeddingStudy(config) as study:
            selection = study.compare()
            heldout = study.evaluate(("test", "icdar"))
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "synthetic_only": True,
        "output_dir": str(root.resolve()),
        "bank": selection["bank"],
        "controls": list(selection["variants"]),
        "held_out": heldout["held_out"],
        "accuracy_claim": False,
    }
    write_json(root / "smoke.json", report)
    return report
