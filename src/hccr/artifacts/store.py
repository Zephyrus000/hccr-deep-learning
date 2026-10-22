"""Files owned by a single training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hccr.artifacts.checkpoints import (
    record_final_test_metrics,
    save_checkpoint,
    save_recalibrated_checkpoint,
)
from hccr.artifacts.provenance import initialize_run, write_curves
from hccr.artifacts.serialization import write_json


@dataclass(frozen=True)
class RunArtifacts:
    directory: Path

    def initialize(self, config: Any, metadata: dict[str, Any], run_id: str) -> str:
        return initialize_run(self.directory, config, metadata, run_id)

    def write_json(self, filename: str, payload: Any) -> None:
        write_json(self.directory / filename, payload)

    def write_curves(self, epochs: list[dict[str, float]]) -> None:
        write_curves(self.directory, epochs)

    def save_checkpoint(self, model, metadata: dict) -> None:
        save_checkpoint(model, self.directory, metadata)

    def save_recalibrated_checkpoint(
        self, model, selection_metrics: dict, recalibration: dict
    ) -> None:
        save_recalibrated_checkpoint(
            model, self.directory, selection_metrics, recalibration
        )

    def record_final_test_metrics(
        self, metrics: dict, checkpoint_name: str = "checkpoint", *, split: str = "test"
    ) -> None:
        record_final_test_metrics(self.directory, metrics, checkpoint_name, split=split)
