"""Lightweight logging setup shared across scripts and the API.

Cloud Run and most GCP services capture stdout/stderr into Cloud Logging, so a
plain stream handler with a structured-ish format is enough. Call
``configure_logging`` once at process start (entry points) and use
``get_logger(__name__)`` everywhere else.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    log_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
