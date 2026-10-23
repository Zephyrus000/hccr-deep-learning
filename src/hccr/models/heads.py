"""Classification heads shared by proposed and reference architectures."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _margin_logits(cosine, targets, kind, scale, margin, margin_multiplier):
    if not 0 <= margin_multiplier <= 1:
        raise ValueError("margin_multiplier must be between zero and one")
    if targets is None or margin_multiplier == 0 or kind == "normalized_softmax":
        return cosine * scale
    target = cosine.gather(1, targets[:, None])
    effective_margin = margin * margin_multiplier
    if kind == "cosface":
        changed = target - effective_margin
    else:
        angular = torch.cos(torch.acos(target) + effective_margin)
        threshold = math.cos(math.pi - effective_margin)
        tail = target - math.sin(math.pi - effective_margin) * effective_margin
        changed = torch.where(target > threshold, angular, tail)
    return cosine.scatter(1, targets[:, None], changed) * scale


class AngularMarginClassifier(nn.Linear):
    """Normalized classifier with a train-only CosFace or ArcFace margin."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        kind: str,
        scale: float,
        margin: float,
    ) -> None:
        super().__init__(embedding_dim, num_classes, bias=False)
        if kind not in {"cosface", "arcface", "normalized_softmax"}:
            raise ValueError("classification_head must be cosface or arcface")
        if scale <= 0:
            raise ValueError("logit_scale must be positive")
        if not 0 <= margin < torch.pi / 2:
            raise ValueError("angular_margin must be in [0, pi/2)")
        self.kind, self.scale, self.margin = kind, scale, margin
        # This is an inference-only cache, not model state.  Registering it as a
        # buffer makes DDP synchronize it even though its shape changes between
        # train and eval mode.  Rank-zero-only profiling can therefore leave
        # different buffer layouts across ranks before DDP construction.
        self._normalized_weight: torch.Tensor | None = None
        self._weight_cache_signature = None

    def train(self, mode: bool = True) -> AngularMarginClassifier:
        super().train(mode)
        if mode:
            self._normalized_weight = None
            self._weight_cache_signature = None
        else:
            self._normalized_weight = F.normalize(
                self.weight.detach().float(), dim=1
            ).contiguous()
            self._weight_cache_signature = (
                id(self.weight),
                self.weight._version,
                self.weight.device,
                self.weight.dtype,
            )
        return self

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor | None = None,
        margin_multiplier: float = 1.0,
    ) -> torch.Tensor:
        # Cosine normalization and ArcFace acos are sensitive to reduced
        # precision. Keep the head in FP32 while AMP accelerates the backbone.
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            cached_weight = self._normalized_weight
            cache_is_valid = (
                cached_weight is not None
                and self._weight_cache_signature
                == (
                    id(self.weight),
                    self.weight._version,
                    self.weight.device,
                    self.weight.dtype,
                )
            )
            if self.training or not cache_is_valid:
                normalized_weight = F.normalize(self.weight.float(), dim=1)
                if not self.training:
                    self._normalized_weight = normalized_weight.detach().contiguous()
                    self._weight_cache_signature = (
                        id(self.weight),
                        self.weight._version,
                        self.weight.device,
                        self.weight.dtype,
                    )
            else:
                assert cached_weight is not None
                normalized_weight = cached_weight.float()
            cosine = F.linear(
                F.normalize(embeddings.float(), dim=1), normalized_weight
            ).clamp(-1 + 1e-7, 1 - 1e-7)
            return _margin_logits(
                cosine, targets, self.kind, self.scale, self.margin, margin_multiplier
            )


class FactorizedClassifier(nn.Module):
    """Unit-base low-rank cosine head with optional selected-class residuals."""

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        kind: str,
        scale: float,
        margin: float,
        *,
        classifier_structure: str = "low_rank",
        classifier_rank: int = 64,
        residual_class_indices=(),
        gate_kind: str = "hadamard",
        gate_use_gap: bool = True,
        gate_detach_gap: bool = True,
        normalization_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or num_classes < 2:
            raise ValueError(
                "classifier requires positive width and at least two classes"
            )
        if kind not in {"cosface", "arcface", "normalized_softmax"}:
            raise ValueError("factorized classifiers require normalized logits")
        if classifier_structure not in {
            "low_rank",
            "static_residual",
            "dynamic_residual",
        }:
            raise ValueError("invalid classifier_structure")
        if not 1 <= classifier_rank <= min(embedding_dim, num_classes):
            raise ValueError("classifier_rank must fit both classifier dimensions")
        indices = list(residual_class_indices)
        if any(type(i) is not int or not 0 <= i < num_classes for i in indices):
            raise ValueError("residual indices must be integer class IDs in range")
        if len(set(indices)) != len(indices):
            raise ValueError("residual indices must be unique")
        if classifier_structure == "low_rank" and indices:
            raise ValueError("low_rank cannot contain residual classes")
        if classifier_structure != "low_rank" and not indices:
            raise ValueError("residual classifiers require selected class indices")
        if gate_kind not in {"hadamard", "cosine", "sample", "global", "class"}:
            raise ValueError("invalid gate_kind")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("logit_scale must be finite and positive")
        if not math.isfinite(margin) or not 0 <= margin < math.pi / 2:
            raise ValueError("angular_margin must be in [0, pi/2)")
        if not math.isfinite(normalization_eps) or normalization_eps <= 0:
            raise ValueError("normalization_eps must be finite and positive")
        self.in_features, self.out_features = embedding_dim, num_classes
        self.kind, self.scale, self.margin = kind, scale, margin
        self.classifier_structure, self.classifier_rank = (
            classifier_structure,
            classifier_rank,
        )
        self.gate_kind, self.gate_use_gap = gate_kind, gate_use_gap
        self.gate_detach_gap, self.normalization_eps = (
            gate_detach_gap,
            normalization_eps,
        )
        self.coefficients = nn.Parameter(torch.empty(num_classes, classifier_rank))
        self.basis = nn.Parameter(torch.empty(classifier_rank, embedding_dim))
        nn.init.orthogonal_(self.basis)
        nn.init.normal_(self.coefficients, std=classifier_rank**-0.5)
        self.register_buffer(
            "residual_indices", torch.tensor(indices, dtype=torch.long)
        )
        self.residuals = nn.Parameter(
            torch.zeros(len(indices), embedding_dim), requires_grad=bool(indices)
        )
        if classifier_structure == "dynamic_residual":
            size = embedding_dim if gate_kind in {"hadamard", "sample"} else 1
            if gate_kind not in {"global", "class"}:
                self.gate_weight = nn.Parameter(torch.zeros(size))
            bias_size = len(indices) if gate_kind == "class" else 1
            self.gate_bias = nn.Parameter(torch.zeros(bias_size))
            if gate_use_gap and gate_kind not in {"global", "class"}:
                self.gate_gamma = nn.Parameter(torch.zeros(()))
        self._deployment_cache = None
        self._gate_mean = None

    def __getstate__(self):
        state = super().__getstate__().copy()
        state["_gate_mean"] = None
        state["_deployment_cache"] = None
        return state

    def inference_macs(self, num_classes: int | None = None) -> int:
        """Matrix-product MACs only; scalar operations and cache building excluded."""
        if num_classes is not None and num_classes != self.out_features:
            raise ValueError("factorized MACs require the original class vocabulary")
        d, r, c, k = (
            self.in_features,
            self.classifier_rank,
            self.out_features,
            self.residual_indices.numel(),
        )
        count = r * (d + c) + k * d
        if self.classifier_structure == "dynamic_residual":
            if self.gate_kind == "hadamard":
                count += r * (d + k)
            elif self.gate_kind == "sample":
                count += d
        return count

    def _norm_terms(self):
        # Only scalar-per-class terms survive deployment cache construction.
        prototype = self.coefficients.float() @ self.basis.float()
        norms = prototype.norm(dim=1).clamp_min(self.normalization_eps)
        selected = prototype[self.residual_indices] / norms[self.residual_indices, None]
        residual = self.residuals.double()
        selected = selected.double()
        return (
            norms,
            selected.square().sum(1),
            (selected * residual).sum(1),
            residual.square().sum(1),
        )

    def _cached_norm_terms(self):
        if self.training or torch.is_grad_enabled():
            return self._norm_terms()
        tensors = (self.coefficients, self.basis, self.residuals, self.residual_indices)
        signature = tuple((id(t), t._version, t.device, t.dtype) for t in tensors)
        if self._deployment_cache is None or self._deployment_cache[0] != signature:
            self._deployment_cache = (signature, self._norm_terms())
        return self._deployment_cache[1]

    def train(self, mode=True):
        super().train(mode)
        self._deployment_cache = None
        self._gate_mean = None
        if not mode:
            with torch.no_grad():
                self._cached_norm_terms()
        return self

    def gate_values(self, normalized_embeddings, low_rank_scores, norms):
        batch, count = normalized_embeddings.shape[0], self.residual_indices.numel()
        if self.classifier_structure != "dynamic_residual":
            return normalized_embeddings.new_full((batch, count), 0.5)
        logits = self.gate_bias.float().expand(batch, count)
        if self.gate_kind == "hadamard":
            projected = (
                normalized_embeddings * self.gate_weight.float()
            ) @ self.basis.float().T
            logits = (
                logits
                + (projected @ self.coefficients[self.residual_indices].float().T)
                / norms[self.residual_indices]
            )
        elif self.gate_kind == "cosine":
            logits = (
                logits
                + self.gate_weight.float() * low_rank_scores[:, self.residual_indices]
            )
        elif self.gate_kind == "sample":
            logits = (
                logits + (normalized_embeddings @ self.gate_weight.float())[:, None]
            )
        if hasattr(self, "gate_gamma"):
            top = low_rank_scores.topk(2, dim=1).values
            gap = top[:, 0] - top[:, 1]
            if self.gate_detach_gap:
                gap = gap.detach()
            logits = logits + self.gate_gamma.float() * gap[:, None]
        return logits.sigmoid()

    def forward(self, embeddings, targets=None, margin_multiplier=1.0):
        if not 0 <= margin_multiplier <= 1:
            raise ValueError("margin_multiplier must be between zero and one")
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            self.last_forward_extra_macs = 0
            z = F.normalize(embeddings.float(), dim=1, eps=self.normalization_eps)
            norms, h, j, e = self._cached_norm_terms()
            cosine = ((z @ self.basis.float().T) @ self.coefficients.float().T) / norms
            self._gate_mean = None
            if self.residual_indices.numel():
                alpha = self.gate_values(z, cosine, norms)
                self._gate_mean = alpha.mean() if self.training else None
                numerator = cosine[:, self.residual_indices] + alpha * (
                    z @ self.residuals.float().T
                )
                # FP64 scalar norm arithmetic limits cancellation for opposing vectors.
                square_norm = (
                    h.double()
                    + 2 * alpha.double() * j.double()
                    + alpha.double().square() * e.double()
                )
                denominator = (
                    square_norm.clamp_min(self.normalization_eps**2).sqrt().float()
                )
                selected_scores = numerator / denominator
                unstable = square_norm < 1e-6 * (h + alpha.double().square() * e + 1)
                if unstable.any():
                    sample_ids, selected_ids = unstable.nonzero(as_tuple=True)
                    self.last_forward_extra_macs = (
                        self.out_features * self.classifier_rank * self.in_features
                        + sample_ids.numel() * self.in_features
                    )
                    class_ids = self.residual_indices[selected_ids]
                    prototype = self.coefficients.float() @ self.basis.float()
                    base = prototype[class_ids] / norms[class_ids, None]
                    vectors = (
                        base
                        + alpha[sample_ids, selected_ids, None]
                        * self.residuals[selected_ids].float()
                    )
                    exact = (
                        z[sample_ids]
                        * F.normalize(vectors, dim=1, eps=self.normalization_eps)
                    ).sum(1)
                    selected_scores = selected_scores.index_put(
                        (sample_ids, selected_ids), exact
                    )
                cosine = cosine.index_copy(1, self.residual_indices, selected_scores)
            cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)
            return _margin_logits(
                cosine, targets, self.kind, self.scale, self.margin, margin_multiplier
            )

    def regularization_loss(self, residual_coefficient=0.0, gate_coefficient=0.0):
        if residual_coefficient < 0 or gate_coefficient < 0:
            raise ValueError("regularization coefficients must be nonnegative")
        penalty = self.coefficients.new_zeros((), dtype=torch.float32)
        if self.residual_indices.numel():
            penalty = (
                residual_coefficient * self.residuals.float().square().sum(1).mean()
            )
            if gate_coefficient:
                if self._gate_mean is None:
                    raise RuntimeError(
                        "gate regularization requires a preceding forward"
                    )
                penalty = penalty + gate_coefficient * self._gate_mean
        self._gate_mean = None
        return penalty

    @torch.no_grad()
    def initialize_from_dense(self, weight, reconstruct_residual=False):
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError("dense weight shape disagrees with classifier")
        if not torch.isfinite(weight).all():
            raise ValueError("dense classifier contains nonfinite weights")
        normalized = F.normalize(
            weight.detach().to(self.basis).float(), dim=1, eps=self.normalization_eps
        )
        u, singular, vh = torch.linalg.svd(normalized, full_matrices=False)
        rank = self.classifier_rank
        self.coefficients.copy_(u[:, :rank] * singular[:rank])
        self.basis.copy_(vh[:rank])
        self.residuals.zero_()
        if reconstruct_residual and self.residual_indices.numel():
            p = self.coefficients[self.residual_indices].float() @ self.basis.float()
            q = F.normalize(p, dim=1, eps=self.normalization_eps)
            self.residuals.copy_(2 * (normalized[self.residual_indices] - q))
        for name, parameter in self.named_parameters():
            if name.startswith("gate_"):
                parameter.zero_()
        self._deployment_cache = None
        self._gate_mean = None


def build_classifier(
    embedding_dim,
    num_classes,
    classification_head,
    logit_scale,
    angular_margin,
    *,
    classifier_structure="dense",
    **options,
):
    if classifier_structure != "dense":
        return FactorizedClassifier(
            embedding_dim,
            num_classes,
            classification_head,
            logit_scale,
            angular_margin,
            classifier_structure=classifier_structure,
            **options,
        )
    if options.get("residual_class_indices"):
        raise ValueError("dense classifier cannot have selected residual classes")
    if classification_head == "softmax":
        return nn.Linear(embedding_dim, num_classes)
    return AngularMarginClassifier(
        embedding_dim, num_classes, classification_head, logit_scale, angular_margin
    )
