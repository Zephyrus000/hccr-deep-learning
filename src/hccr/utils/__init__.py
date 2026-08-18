"""Shared helpers with no dependency on CLI orchestration."""

from hccr.utils.device import resolve_device
from hccr.utils.logging import close_logging, configure_logging

__all__ = ["close_logging", "configure_logging", "resolve_device"]
