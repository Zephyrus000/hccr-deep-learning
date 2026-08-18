"""Artifact-backed top-k prediction."""

from __future__ import annotations

import torch


class Predictor:
    def __init__(self, model, labels: list[str], device: str = "cpu") -> None:
        self.model, self.labels, self.device = model.eval().to(device), labels, device

    @torch.inference_mode()
    def predict(self, image: torch.Tensor, top_k: int = 5) -> list[tuple[str, float]]:
        probabilities = self.model(image.unsqueeze(0).to(self.device)).softmax(dim=1)[0]
        scores, indices = probabilities.topk(min(top_k, len(self.labels)))
        return [
            (self.labels[index.item()], score.item())
            for score, index in zip(scores, indices, strict=True)
        ]
