"""Visual QA gallery for preprocessing output."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def save_gallery(
    images: list, transform: Callable, output: Path, columns: int = 4
) -> None:
    """Save original/transformed image pairs as a contact sheet."""
    from PIL import Image

    pairs = [(image.convert("L"), transform(image)) for image in images]
    if not pairs:
        raise ValueError("gallery requires at least one image")
    width, height = pairs[0][1].size
    rows = (len(pairs) + columns - 1) // columns
    gallery = Image.new("L", (columns * width, rows * height * 2), 255)
    for index, (original, processed) in enumerate(pairs):
        original.thumbnail((width, height), Image.Resampling.LANCZOS)
        x = (index % columns) * width
        y = (index // columns) * height * 2
        gallery.paste(original, (x, y))
        gallery.paste(processed, (x, y + height))
    output.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(output)
