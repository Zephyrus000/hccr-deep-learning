"""Bounded, detached observations of residual gates and optimizer updates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from hccr.models.heads import FactorizedClassifier


class GateValidationDiagnostics:
    """Observe a deterministic loader prefix without retaining its forward graph.

    The caller supplies a sequential, deterministically preprocessed validation
    loader. Statistics describe this panel, not the entire validation population.
    """

    def __init__(self, model: nn.Module, max_samples: int = 4096, *, enabled=True):
        head = getattr(model, "classifier", model)
        self.head = (
            head
            if enabled
            and isinstance(head, FactorizedClassifier)
            and head.residual_indices.numel()
            else None
        )
        self.max_samples = max(0, max_samples)
        self.samples = 0
        self.targets = None
        self.metadata = None
        self.sample_ids = []
        self.identity_complete = True
        self.handle = None
        self.batches = []

    def __enter__(self):
        if self.head is not None and self.max_samples:
            self.handle = self.head.register_forward_hook(self._observe)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.targets = None
        self.metadata = None

    def set_targets(self, targets, metadata=None):
        self.targets = targets.detach()
        self.metadata = metadata

    @torch.no_grad()
    def _observe(self, head, inputs, output):
        count = min(len(output), self.max_samples - self.samples)
        if count <= 0 or self.targets is None:
            return
        identities = (
            self.metadata.get("sample_id")
            if isinstance(self.metadata, Mapping)
            else self.metadata
            if isinstance(self.metadata, torch.Tensor)
            else None
        )
        if identities is None:
            self.identity_complete = False
        else:
            if isinstance(identities, torch.Tensor):
                identities = identities[:count].tolist()
            self.sample_ids.extend(str(value) for value in identities[:count])
        with torch.autocast(device_type=inputs[0].device.type, enabled=False):
            z = F.normalize(
                inputs[0][:count].detach().float(), dim=1, eps=head.normalization_eps
            )
            norms, base_squared, _, residual_squared = head._cached_norm_terms()
            base = ((z @ head.basis.float().T) @ head.coefficients.float().T) / norms
            alpha = head.gate_values(z, base, norms)
            zero_residual_scores = (
                base[:, head.residual_indices]
                / base_squared.clamp_min(head.normalization_eps**2).sqrt().float()
            ).clamp(-1 + 1e-7, 1 - 1e-7)
            score_change = (
                output[:count, head.residual_indices].float() / head.scale
                - zero_residual_scores
            ).abs()
            magnitude = alpha * residual_squared.float().sqrt()
        observed = torch.stack((alpha, magnitude, score_change)).detach().cpu()
        self.batches.append(
            (observed[0], self.targets[:count].cpu(), observed[1], observed[2])
        )
        self.samples += count

    def summary(self):
        if not self.batches:
            return None
        alpha, targets, magnitude, score_change = (
            torch.cat(values) for values in zip(*self.batches, strict=True)
        )
        indices = self.head.residual_indices.detach().cpu()
        target_mask = targets[:, None].eq(indices)
        return {
            "panel_policy": "first_rows_of_sequential_validation_loader",
            "max_samples": self.max_samples,
            "samples": self.samples,
            "panel_identity_sha256": (
                hashlib.sha256(json.dumps(self.sample_ids).encode()).hexdigest()
                if self.identity_complete
                else None
            ),
            "class_ids": indices.tolist(),
            "target": summarize_gate_values(
                alpha, target_mask, indices, magnitude, score_change
            ),
            "impostor": summarize_gate_values(
                alpha, ~target_mask, indices, magnitude, score_change
            ),
            "score_change_reference": "same_weights_with_selected_residuals_disabled",
            "quantile_policy": "exact_class_balanced_weighted_empirical_cdf",
            "variance_policy": "population_variance_with_observed_support",
        }


def summarize_gate_values(alpha, mask, class_ids, magnitude=None, score_change=None):
    """Give each supported selected class equal weight within one stratum."""
    alpha = alpha.detach().cpu().double()
    mask = mask.detach().cpu().bool()
    count = mask.sum(0)
    supported = count > 0
    divisor = count.clamp_min(1)
    mean = (alpha * mask).sum(0) / divisor
    second = (alpha.square() * mask).sum(0) / divisor
    variance = (second - mean.square()).clamp_min(0)
    low = ((alpha < 0.05) & mask).sum(0) / divisor
    high = ((alpha > 0.95) & mask).sum(0) / divisor
    slope = 4 * ((alpha * (1 - alpha)) * mask).sum(0) / divisor
    per_class = [
        {
            "class_id": int(class_id),
            "count": int(count[index]),
            "mean": float(mean[index]) if count[index] else None,
            "variance": float(variance[index]) if count[index] else None,
        }
        for index, class_id in enumerate(class_ids)
    ]
    result = {
        "count": int(count.sum()),
        "supported_classes": int(supported.sum()),
        "per_class": per_class,
    }
    if not supported.any():
        return result
    values, order = alpha[mask].sort()
    weights = divisor.double().reciprocal().expand_as(alpha)[mask][order]
    cumulative = weights.cumsum(0)
    probabilities = torch.tensor(
        [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=torch.float64
    )
    positions = torch.searchsorted(cumulative, probabilities * cumulative[-1])
    positions = positions.clamp_max(len(values) - 1)
    histogram = (
        torch.bincount((values * 10).long().clamp(0, 9), weights=weights, minlength=10)
        / weights.sum()
    )
    pooled_mean = mean[supported].mean()
    result.update(
        mean=float(pooled_mean),
        pooled_variance=float(
            (second[supported].mean() - pooled_mean.square()).clamp_min(0)
        ),
        within_class_variance=float(variance[supported].mean()),
        fraction_below_005=float(low[supported].mean()),
        fraction_above_095=float(high[supported].mean()),
        normalized_sigmoid_slope=float(slope[supported].mean()),
        histogram_edges=[index / 10 for index in range(11)],
        histogram_class_balanced_fractions=histogram.tolist(),
        quantiles={
            str(float(p)): float(v)
            for p, v in zip(probabilities, values[positions], strict=True)
        },
    )
    for name, observation in (
        ("effective_residual_magnitude_mean", magnitude),
        ("absolute_score_change_mean", score_change),
    ):
        if observation is not None:
            per_class_mean = (observation.double() * mask).sum(0) / divisor
            result[name] = float(per_class_mean[supported].mean())
    return result


class GateTrainingDiagnostics:
    """Observe total unscaled gradients and actual updates on the first steps."""

    def __init__(self, model: nn.Module, max_steps: int = 3):
        head = getattr(model, "classifier", model)
        self.parameters = (
            [
                parameter
                for name, parameter in head.named_parameters()
                if name.startswith("gate_") and parameter.requires_grad
            ]
            if isinstance(head, FactorizedClassifier)
            else []
        )
        self.max_steps = max_steps
        self.observations = []
        self.before = None

    def before_step(self, step):
        if not self.parameters or step > self.max_steps:
            return
        self.before = torch.cat([p.detach().flatten() for p in self.parameters]).clone()
        squares = [
            p.grad.detach().float().square().sum()
            for p in self.parameters
            if p.grad is not None
        ]
        norm = (
            torch.stack(squares).sum().sqrt() if squares else self.before.new_zeros(())
        )
        self.observations.append((step, norm, None, None))

    def after_step(self, stepped):
        if self.before is None:
            return
        after = torch.cat([p.detach().flatten() for p in self.parameters])
        step, gradient, _, _ = self.observations[-1]
        self.observations[-1] = (step, gradient, (after - self.before).norm(), stepped)
        self.before = None

    def summary(self):
        if not self.observations:
            return None
        norms = (
            torch.stack(
                [
                    torch.stack((gradient, update))
                    for _, gradient, update, _ in self.observations
                ]
            )
            .cpu()
            .tolist()
        )
        return {
            "gradient_source": "total_loss_after_unscale_including_regularization",
            "update_source": "actual_optimizer_step_including_history_and_decay",
            "max_steps_per_rank": self.max_steps,
            "parameter_count": sum(p.numel() for p in self.parameters),
            "observations": [
                {
                    "batch": step,
                    "gradient_norm": norms[index][0],
                    "update_norm": norms[index][1],
                    "optimizer_stepped": stepped,
                }
                for index, (step, _, _, stepped) in enumerate(self.observations)
            ],
        }
