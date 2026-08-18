"""Run-scoped console and file logging."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hccr")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logging(logger)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logging(logger: logging.Logger) -> None:
    """Flush and close run handlers so Windows can release artifact files."""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
