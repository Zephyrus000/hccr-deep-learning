"""Reference CNN baselines adapted to one-channel HCCR inputs.

The wrappers deliberately expose ``backbone`` and ``classifier`` as separate
modules.  This keeps resource profiling and full-class head projection
consistent with :class:`~hccr.models.efficient_hccr.EfficientHCCRNet`.
"""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import (
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
    shufflenet_v2_x1_0,
)

from hccr.models.heads import (
    AngularMarginClassifier,
    FactorizedClassifier,
    build_classifier,
)


class TorchvisionClassifierBaseline(nn.Module):
    """A torchvision feature extractor with a configurable shared head."""

    def __init__(
        self,
        *,
        name: str,
        backbone: nn.Module,
        embedding_projection: nn.Module,
        classifier: nn.Module,
        in_channels: int,
        classification_head: str,
        backbone_output_channels: int,
        embedding_dim: int,
        logit_scale: float,
        angular_margin: float,
    ) -> None:
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.embedding_projection = embedding_projection
        self.classifier = classifier
        self.effective_input_channels = in_channels
        self.classification_head = classification_head
        self.backbone_output_channels = backbone_output_channels
        self.embedding_dim = embedding_dim
        self.logit_scale = logit_scale
        self.angular_margin = angular_margin

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor | None = None,
        margin_multiplier: float = 1.0,
    ) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.effective_input_channels:
            raise ValueError(
                "baseline requires BCHW inputs with "
                f"{self.effective_input_channels} channel(s)"
            )
        embeddings = self.embedding_projection(self.backbone(inputs))
        if isinstance(self.classifier, (AngularMarginClassifier, FactorizedClassifier)):
            return self.classifier(embeddings, targets, margin_multiplier)
        return self.classifier(embeddings)

    def training_logits(
        self,
        inputs: Tensor,
        targets: Tensor,
        margin_multiplier: float = 1.0,
    ) -> Tensor:
        """Apply the shared margin path while remaining compatible with DDP."""
        return self(inputs, targets, margin_multiplier)


def build_resnet18(
    *,
    num_classes: int,
    in_channels: int = 1,
    classification_head: str = "softmax",
    embedding_dim: int | None = None,
    logit_scale: float = 32.0,
    angular_margin: float = 0.1,
    **head_options,
) -> nn.Module:
    """Build the standard ResNet-18 reference model without pretrained weights."""
    _validate_classifier_arguments(num_classes=num_classes, in_channels=in_channels)
    network = resnet18(weights=None, num_classes=num_classes)
    first_conv = network.conv1
    network.conv1 = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )
    classifier = network.fc
    network.fc = nn.Identity()
    return _wrap_baseline(
        name="resnet18",
        backbone=network,
        classifier=classifier,
        in_channels=in_channels,
        classification_head=classification_head,
        embedding_dim=embedding_dim,
        logit_scale=logit_scale,
        angular_margin=angular_margin,
        **head_options,
    )


def build_mobilenet_v3_small(
    *,
    num_classes: int,
    in_channels: int = 1,
    classification_head: str = "softmax",
    embedding_dim: int | None = None,
    logit_scale: float = 32.0,
    angular_margin: float = 0.1,
    **head_options,
) -> nn.Module:
    """Build the standard MobileNetV3-Small reference model without weights."""
    _validate_classifier_arguments(num_classes=num_classes, in_channels=in_channels)
    network = mobilenet_v3_small(weights=None, num_classes=num_classes)
    first_conv = network.features[0][0]
    network.features[0][0] = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )
    classifier = network.classifier
    network.classifier = nn.Identity()
    return _wrap_baseline(
        name="mobilenet_v3_small",
        backbone=network,
        classifier=classifier,
        in_channels=in_channels,
        classification_head=classification_head,
        embedding_dim=embedding_dim,
        logit_scale=logit_scale,
        angular_margin=angular_margin,
        **head_options,
    )


def build_shufflenet_v2_x1_0(
    *,
    num_classes: int,
    in_channels: int = 1,
    classification_head: str = "softmax",
    embedding_dim: int | None = None,
    logit_scale: float = 32.0,
    angular_margin: float = 0.1,
    **head_options,
) -> nn.Module:
    """Build the standard ShuffleNetV2 1.0x reference model without weights."""
    _validate_classifier_arguments(num_classes=num_classes, in_channels=in_channels)
    network = shufflenet_v2_x1_0(weights=None, num_classes=num_classes)
    first_conv = network.conv1[0]
    network.conv1[0] = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )
    classifier = network.fc
    network.fc = nn.Identity()
    return _wrap_baseline(
        name="shufflenet_v2_x1_0",
        backbone=network,
        classifier=classifier,
        in_channels=in_channels,
        classification_head=classification_head,
        embedding_dim=embedding_dim,
        logit_scale=logit_scale,
        angular_margin=angular_margin,
        **head_options,
    )


def build_efficientnet_b0(
    *,
    num_classes: int,
    in_channels: int = 1,
    classification_head: str = "softmax",
    embedding_dim: int | None = None,
    logit_scale: float = 32.0,
    angular_margin: float = 0.1,
    **head_options,
) -> nn.Module:
    """Build the standard EfficientNet-B0 reference model without weights."""
    _validate_classifier_arguments(num_classes=num_classes, in_channels=in_channels)
    network = efficientnet_b0(weights=None, num_classes=num_classes)
    first_conv = network.features[0][0]
    network.features[0][0] = nn.Conv2d(
        in_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )
    classifier = network.classifier
    network.classifier = nn.Identity()
    return _wrap_baseline(
        name="efficientnet_b0",
        backbone=network,
        classifier=classifier,
        in_channels=in_channels,
        classification_head=classification_head,
        embedding_dim=embedding_dim,
        logit_scale=logit_scale,
        angular_margin=angular_margin,
        **head_options,
    )


def build_baseline_model(name: str, **kwargs: object) -> nn.Module:
    """Build a supported reference classifier without pretrained weights."""
    if name == "resnet18":
        return build_resnet18(**kwargs)  # type: ignore[arg-type]
    if name == "mobilenet_v3_small":
        return build_mobilenet_v3_small(**kwargs)  # type: ignore[arg-type]
    if name == "shufflenet_v2_x1_0":
        return build_shufflenet_v2_x1_0(**kwargs)  # type: ignore[arg-type]
    if name == "efficientnet_b0":
        return build_efficientnet_b0(**kwargs)  # type: ignore[arg-type]
    if name in {"mobilenetv4_conv_small", "repvit_m0_9"}:
        return build_modern_baseline(name=name, **kwargs)  # type: ignore[arg-type]
    raise ValueError(
        "baseline model must be resnet18, mobilenet_v3_small, "
        "shufflenet_v2_x1_0, efficientnet_b0, mobilenetv4_conv_small, or repvit_m0_9"
    )


def build_modern_baseline(
    *,
    name: str,
    num_classes: int,
    in_channels: int = 1,
    classification_head: str = "softmax",
    embedding_dim: int | None = 320,
    logit_scale: float = 32.0,
    angular_margin: float = 0.1,
    **head_options,
) -> nn.Module:
    """Build an optional timm backbone with the common HCCR projection and head."""
    _validate_classifier_arguments(num_classes=num_classes, in_channels=in_channels)
    try:
        from timm import create_model
    except ImportError as error:
        raise ImportError(
            "Modern HCCR baselines require timm; install with "
            "python -m pip install -e '.[modern]'"
        ) from error
    backbone = create_model(
        name,
        pretrained=False,
        num_classes=0,
        in_chans=in_channels,
        global_pool="avg",
    )
    return _wrap_baseline(
        name=name,
        backbone=backbone,
        classifier=nn.Linear(backbone.num_features, num_classes),
        in_channels=in_channels,
        classification_head=classification_head,
        embedding_dim=320 if embedding_dim is None else embedding_dim,
        logit_scale=logit_scale,
        angular_margin=angular_margin,
        **head_options,
    )


def _validate_classifier_arguments(*, num_classes: int, in_channels: int) -> None:
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if in_channels != 1:
        raise ValueError("HCCR baseline models expect one grayscale input channel")


def _wrap_baseline(
    *,
    name: str,
    backbone: nn.Module,
    classifier: nn.Module,
    in_channels: int,
    classification_head: str,
    embedding_dim: int | None,
    logit_scale: float,
    angular_margin: float,
    **head_options,
) -> TorchvisionClassifierBaseline:
    if classification_head not in {
        "softmax",
        "cosface",
        "arcface",
        "normalized_softmax",
    }:
        raise ValueError("classification_head must be cosface, arcface, or softmax")
    if embedding_dim is not None and embedding_dim < 1:
        raise ValueError("embedding_dim must be positive when set")
    final_linear = _final_linear(classifier)
    backbone_output_channels = _classifier_input_dim(classifier)
    use_common_projection = embedding_dim is not None
    resolved_embedding_dim = embedding_dim or backbone_output_channels
    if use_common_projection:
        embedding_projection = nn.Linear(
            backbone_output_channels,
            resolved_embedding_dim,
            bias=False,
        )
        resolved_classifier = build_classifier(
            resolved_embedding_dim,
            final_linear.out_features,
            classification_head,
            logit_scale,
            angular_margin,
            **head_options,
        )
    elif (
        classification_head == "softmax"
        and head_options.get("classifier_structure", "dense") == "dense"
    ):
        embedding_projection = nn.Identity()
        resolved_classifier = classifier
        if head_options.get("residual_class_indices"):
            raise ValueError("dense classifier cannot have selected residual classes")
    else:
        embedding_projection = _classifier_prefix(classifier)
        resolved_classifier = build_classifier(
            final_linear.in_features,
            final_linear.out_features,
            classification_head,
            logit_scale,
            angular_margin,
            **head_options,
        )
    model = TorchvisionClassifierBaseline(
        name=name,
        backbone=backbone,
        embedding_projection=embedding_projection,
        classifier=resolved_classifier,
        in_channels=in_channels,
        classification_head=classification_head,
        backbone_output_channels=backbone_output_channels,
        embedding_dim=(
            resolved_embedding_dim
            if use_common_projection
            else final_linear.in_features
        ),
        logit_scale=logit_scale,
        angular_margin=angular_margin,
    )
    defaults = dict(
        classifier_structure="dense",
        classifier_rank=64,
        residual_class_indices=(),
        gate_kind="hadamard",
        gate_use_gap=True,
        gate_detach_gap=True,
        normalization_eps=1e-6,
    )
    for key, value in (defaults | head_options).items():
        setattr(model, key, value)
    return model


def _final_linear(classifier: nn.Module) -> nn.Linear:
    if isinstance(classifier, nn.Linear):
        return classifier
    if isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
        return classifier[-1]
    raise TypeError("baseline classifier must end in a Linear layer")


def _classifier_input_dim(classifier: nn.Module) -> int:
    if isinstance(classifier, nn.Linear):
        return classifier.in_features
    if isinstance(classifier, nn.Sequential):
        first_linear = next(
            (module for module in classifier if isinstance(module, nn.Linear)), None
        )
        if first_linear is not None:
            return first_linear.in_features
    raise TypeError("baseline classifier must contain a Linear layer")


def _classifier_prefix(classifier: nn.Module) -> nn.Module:
    if isinstance(classifier, nn.Linear):
        return nn.Identity()
    assert isinstance(classifier, nn.Sequential)
    prefix = tuple(classifier.children())[:-1]
    return nn.Sequential(*prefix) if prefix else nn.Identity()
