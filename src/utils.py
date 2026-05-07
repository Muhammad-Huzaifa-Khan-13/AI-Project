"""Utility helpers for logging, filesystem, and reproducibility."""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


def setup_logger(name: str = "preprocessing", level: int = logging.INFO) -> logging.Logger:
    """Create and return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_random_seed(seed: int) -> None:
    """Set deterministic seeds for reproducible local experiments."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Save dictionary payload as pretty JSON."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=True)
