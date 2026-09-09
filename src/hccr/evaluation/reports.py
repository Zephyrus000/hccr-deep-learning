"""Static Matplotlib/Seaborn artifacts for experiment analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def save_learning_curves(
    output_dir: Path,
    epochs: list[dict[str, float]],
    recalibrated: dict[str, float] | None = None,
    evaluation_name: str = "validation",
) -> Path:
    output = output_dir / "learning_curves.png"
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(
        [item["epoch"] for item in epochs],
        [item["train_loss"] for item in epochs],
        label="train loss",
    )
    loss_axis.set(xlabel="epoch", ylabel="loss", title="Training loss")
    loss_axis.legend()
    evaluation_epochs = [
        item
        for item in epochs
        if item.get("evaluation_performed", True) and item.get("top1") is not None
    ]
    if evaluation_epochs:
        accuracy_axis.plot(
            [item["epoch"] for item in evaluation_epochs],
            [item["top1"] for item in evaluation_epochs],
            marker="o",
            label=f"{evaluation_name} top-1",
        )
    if recalibrated is not None:
        accuracy_axis.scatter(
            [recalibrated["epoch"]],
            [recalibrated["top1"]],
            marker="*",
            s=120,
            label=f"BN-recalibrated {evaluation_name} top-1",
            zorder=3,
        )
    accuracy_axis.set(
        xlabel="epoch",
        ylabel="accuracy",
        title=f"{evaluation_name.title()} accuracy",
        ylim=(0, 1),
    )
    if accuracy_axis.get_legend_handles_labels()[0]:
        accuracy_axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def save_confusion_heatmap(output_dir: Path, matrix, labels: list[str]) -> Path:
    """Save a selected-class confusion matrix; do not render all 7k classes."""
    output = output_dir / "confusion_matrix.png"
    figure, axis = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, cmap="mako", ax=axis)
    axis.set(xlabel="predicted", ylabel="target", title="Selected confusion matrix")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def save_confidence_distribution(
    output_dir: Path, correct: list[float], incorrect: list[float]
) -> Path:
    output = output_dir / "confidence_distribution.png"
    figure, axis = plt.subplots(figsize=(8, 4))
    if correct:
        sns.histplot(
            correct,
            label="correct",
            stat="density",
            element="step",
            fill=False,
            ax=axis,
        )
    if incorrect:
        sns.histplot(
            incorrect,
            label="incorrect",
            stat="density",
            element="step",
            fill=False,
            ax=axis,
        )
    axis.set(xlabel="confidence", title="Prediction confidence")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output
