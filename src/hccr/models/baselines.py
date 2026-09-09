"""Reference CNN baselines adapted to one-channel HCCR inputs.

The wrappers deliberately expose ``backbone`` and ``classifier`` as separate
modules.  This keeps resource profiling and full-class head projection
consistent with :class:`~hccr.models.efficient_hccr.EfficientHCCRNet`.
"""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import mobilenet_v3_small, resnet18


class TorchvisionClassifierBaseline(nn.Module):
    """A torchvision feature extractor with an explicit softmax classifier."""

    classification_head = "softmax"

    def __init__(
        self,
        *,
        name: str,
        backbone: nn.Module,
        classifier: nn.Module,
        in_channels: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.backbone = backbone
        self.classifier = classifier
        self.effective_input_channels = in_channels

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.effective_input_channels:
            raise ValueError(
                "baseline requires BCHW inputs with "
                f"{self.effective_input_channels} channel(s)"
            )
        return self.classifier(self.backbone(inputs))


def build_resnet18(*, num_classes: int, in_channels: int = 1) -> nn.Module:
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
    return TorchvisionClassifierBaseline(
        name="resnet18",
        backbone=network,
        classifier=classifier,
        in_channels=in_channels,
    )


def build_mobilenet_v3_small(*, num_classes: int, in_channels: int = 1) -> nn.Module:
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
    return TorchvisionClassifierBaseline(
        name="mobilenet_v3_small",
        backbone=network,
        classifier=classifier,
        in_channels=in_channels,
    )


def build_baseline_model(name: str, **kwargs: object) -> nn.Module:
    """Build a supported torchvision reference classifier."""
    if name == "resnet18":
        return build_resnet18(**kwargs)  # type: ignore[arg-type]
    if name == "mobilenet_v3_small":
        return build_mobilenet_v3_small(**kwargs)  # type: ignore[arg-type]
    raise ValueError("baseline model must be resnet18 or mobilenet_v3_small")


def _validate_classifier_arguments(*, num_classes: int, in_channels: int) -> None:
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if in_channels != 1:
        raise ValueError("HCCR baseline models expect one grayscale input channel")
