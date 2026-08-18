"""Run-scoped console and file logging."""

from __future__ import annotations

import logging
from pathlib import Path


class _ExcludeProgressLogs(logging.Filter):
    """Keep batch logs in the file without interrupting an active tqdm bar."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith("hccr.train")


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hccr")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    close_logging(logger)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.addFilter(_ExcludeProgressLogs())
    for handler in (
        console_handler,
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
