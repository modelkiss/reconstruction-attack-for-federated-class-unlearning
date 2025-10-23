"""
src package initializer.

Provides lightweight package metadata and a few convenience helpers that are
safe to import early (don't import heavy libs at module import time).

Typical usage in scripts/modules:
    from src import setup_logging, get_default_device, seed_all, __version__
"""

from __future__ import annotations

__all__ = [
    "setup_logging",
    "get_default_device",
    "seed_all",
    "__version__",
]

# package version: update when releasing / tagging
__version__ = "0.1.0"

import logging
import os
import random
import json
from typing import Optional


def setup_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> logging.Logger:
    """
    Initialize and return a project-wide logger.

    Args:
        level: logging level (e.g., logging.INFO, logging.DEBUG)
        fmt: optional format string for log messages; if None a sensible default is used.

    Returns:
        logging.Logger instance named "src".
    """
    logger = logging.getLogger("src")
    if logger.handlers:
        # already configured
        return logger

    if fmt is None:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(handler)
    logger.setLevel(level)
    # avoid propagation to root logger twice
    logger.propagate = False
    return logger


def get_default_device() -> str:
    """
    Return the default device string: "cuda" if available and usable, otherwise "cpu".
    We import torch lazily so importing src does not require torch to be installed.

    Returns:
        "cuda" or "cpu"
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def seed_all(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and (if available) PyTorch to improve reproducibility.

    Note: This is a convenience helper — modules that need stricter determinism
    should set torch.backends.cudnn.deterministic and other flags themselves.

    Args:
        seed: integer seed
    """
    import numpy as _np

    logger = logging.getLogger("src")
    logger.debug("Seeding RNGs with seed=%s", seed)

    random.seed(seed)
    _np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
            # Do NOT set deterministic flags here unconditionally; leave it to experiments
    except Exception:
        # torch not installed — that's fine for light-weight usage
        logger.debug("PyTorch not available while seeding; continuing.")


# convenience: create a module-level logger on import (but do not configure handlers aggressively)
_logger = logging.getLogger("src")
if not _logger.handlers:
    # default to INFO but do not attach handlers here; caller should call setup_logging()
    _logger.setLevel(logging.INFO)
