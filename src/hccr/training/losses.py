"""Classification-loss construction for controlled training ablations."""

from torch import nn


def build_classification_loss(label_smoothing: float = 0.0) -> nn.Module:
    """Build cross entropy while keeping smoothing explicit in run metadata."""
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
