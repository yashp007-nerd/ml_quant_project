"""
evaluate.py
-----------
Classification evaluation: accuracy, precision, recall, F1, MCC, AUC,
confusion matrix, ROC curve, and model benchmarking utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from scipy import stats

from .utils import CLASS_NAMES, ensure_dir, get_logger

logger = get_logger(__name__)

CLASS_LIST = [0, 1, 2]   # SELL, BUY, HOLD


# ─────────────────────────────── core metrics ────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Return a dict of all classification metrics.

    Parameters
    ----------
    y_true : ground-truth labels
    y_pred : predicted labels
    y_prob : (N, 3) probability matrix (for AUC)
    """
    metrics: Dict[str, float] = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(   y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(       y_true, y_pred, average="weighted", zero_division=0),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None and y_prob.shape[1] == 3:
        try:
            metrics["auc"] = roc_auc_score(
                label_binarize(y_true, classes=CLASS_LIST),
                y_prob,
                multi_class="ovr",
                average="weighted",
            )
        except Exception:
            metrics["auc"] = float("nan")
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "") -> None:
    if title:
        logger.info(f"──── {title} ────")
    for k, v in metrics.items():
        logger.info(f"  {k:>12} : {v:.4f}")


# ─────────────────────────────── classification report ───────────────────────

def full_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    target_names = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0,
    )


# ─────────────────────────────── confusion matrix ────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title:  str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LIST)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[CLASS_NAMES[c] for c in CLASS_LIST],
        yticklabels=[CLASS_NAMES[c] for c in CLASS_LIST],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"Confusion matrix saved → {save_path}")
    return fig


# ─────────────────────────────── ROC curve ───────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title:  str = "ROC Curve",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    y_bin = label_binarize(y_true, classes=CLASS_LIST)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#e74c3c", "#2ecc71", "#95a5a6"]

    for i, (cls, color) in enumerate(zip(CLASS_LIST, colors)):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{CLASS_NAMES[cls]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"ROC curve saved → {save_path}")
    return fig


# ─────────────────────────────── training curves ────────────────────────────

def plot_training_history(
    history: Dict[str, list],
    title:   str = "Training History",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linestyle="--")
    axes[0].set_title("Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   linestyle="--")
    axes[1].set_title("Accuracy", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ─────────────────────────────── benchmark comparison ────────────────────────

def benchmark_comparison(
    baseline_metrics: Dict[str, float],
    multimodal_metrics: Dict[str, float],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart comparing baseline vs multi-modal on key metrics."""
    keys   = ["accuracy", "precision", "recall", "f1", "mcc"]
    labels = [k.capitalize() for k in keys]
    base_v = [baseline_metrics.get(k, 0) for k in keys]
    mm_v   = [multimodal_metrics.get(k, 0) for k in keys]

    x   = np.arange(len(keys))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, base_v, w, label="LSTM Baseline", color="#3498db")
    b2 = ax.bar(x + w/2, mm_v,   w, label="Multi-Modal",   color="#e67e22")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Multi-Modal — Classification Performance", fontsize=13)
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"Benchmark comparison saved → {save_path}")
    return fig


# ─────────────────────────────── statistical validation ─────────────────────

def paired_t_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_name: str = "accuracy",
) -> Dict[str, float]:
    """
    Paired one-tailed t-test: H1 → model B significantly outperforms model A.

    Returns
    -------
    dict with t_stat, p_value, significant (bool at alpha=0.05)
    """
    t_stat, p_two_tailed = stats.ttest_rel(scores_b, scores_a)
    p_one_tailed = p_two_tailed / 2.0  # one-tailed
    result = {
        "metric":      metric_name,
        "t_statistic": float(t_stat),
        "p_value":     float(p_one_tailed),
        "significant": bool(p_one_tailed < 0.05 and t_stat > 0),
    }
    logger.info(
        f"[t-test] {metric_name}: t={t_stat:.4f}  p={p_one_tailed:.4f}  "
        f"significant={result['significant']}"
    )
    return result
