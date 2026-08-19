"""Visual QA gallery for preprocessing output."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path


def save_gallery(
    images: list,
    transform: Callable,
    output: Path,
    labels: Sequence[str] | None = None,
    columns: int = 4,
) -> None:
    """Save original/transformed image pairs as a contact sheet."""
    from PIL import Image, ImageDraw

    pairs = [(image.convert("L"), transform(image)) for image in images]
    if not pairs:
        raise ValueError("gallery requires at least one image")
    if labels is not None and len(labels) != len(pairs):
        raise ValueError("labels must match the number of gallery images")
    width, height = pairs[0][1].size
    rows = (len(pairs) + columns - 1) // columns
    caption_height = 16
    pair_height = height * 2 + caption_height * 2
    gallery = Image.new("RGB", (columns * width, rows * pair_height), "white")
    draw = ImageDraw.Draw(gallery)
    for index, (original, processed) in enumerate(pairs):
        original = original.copy()
        original.thumbnail((width, height), Image.Resampling.LANCZOS)
        x = (index % columns) * width
        y = (index // columns) * pair_height
        gallery.paste(original, (x + (width - original.width) // 2, y))
        label = labels[index] if labels is not None else f"sample {index + 1}"
        draw.text((x + 2, y + height), f"{label} | original", fill="black")
        gallery.paste(processed, (x, y + height + caption_height))
        draw.text(
            (x + 2, y + height * 2 + caption_height),
            f"{label} | processed",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(output)
