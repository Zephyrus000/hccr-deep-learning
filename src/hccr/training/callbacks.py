"""Training callbacks that keep stopping decisions explicit and testable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int | None
    min_delta: float = 0.0
    best_score: float = float("-inf")
    bad_epochs: int = 0

    def update(self, score: float) -> bool:
        """Record a validation score and return whether training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.patience is not None and self.bad_epochs >= self.patience
