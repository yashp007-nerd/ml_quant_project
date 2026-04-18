"""
utils.py
--------
Shared helpers: config loading, logging, reproducibility, plotting boilerplate.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# ─────────────────────────────── config ─────────────────────────────────────

def load_config(path: str | Path = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML config and return as a nested dict."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ─────────────────────────────── logging ────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger that writes to stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────────── reproducibility ────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────── device ─────────────────────────────────────

def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():          # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────── directory helpers ──────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────── normalisation ──────────────────────────────

def minmax_scale(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max scale array to [0, 1] column-wise (or 1-D)."""
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    return (arr - mn) / (mx - mn + eps)


def zscore_scale(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalise column-wise."""
    return (arr - arr.mean(axis=0)) / (arr.std(axis=0) + eps)


# ─────────────────────────────── label mapping ──────────────────────────────

CLASS_NAMES = {0: "SELL", 1: "BUY", 2: "HOLD"}
CLASS_COLORS = {0: "#e74c3c", 1: "#2ecc71", 2: "#95a5a6"}


def label_to_name(label: int) -> str:
    return CLASS_NAMES.get(label, "UNKNOWN")


# ─────────────────────────────── metric helpers ─────────────────────────────

def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient (multi-class)."""
    from sklearn.metrics import matthews_corrcoef
    return float(matthews_corrcoef(y_true, y_pred))


# ─────────────────────────────── checkpoint helpers ─────────────────────────

def save_checkpoint(
    state: Dict[str, Any],
    path: str | Path,
    logger: logging.Logger | None = None,
) -> None:
    ensure_dir(Path(path).parent)
    torch.save(state, path)
    if logger:
        logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    device = device or get_device()
    return torch.load(path, map_location=device, weights_only=False)
