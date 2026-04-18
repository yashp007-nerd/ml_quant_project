"""
train.py
--------
Training pipeline for both the multi-modal and baseline models.
Includes:
  - PyTorch Dataset / DataLoader construction
  - Training loop with early stopping & LR scheduling
  - Per-epoch logging of loss and accuracy
  - Model checkpointing
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import ensure_dir, get_device, get_logger, save_checkpoint

logger = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  Dataset
# ═════════════════════════════════════════════════════════════════════════════

class StockDataset(Dataset):
    """
    Unified dataset for multi-modal and baseline models.

    Parameters
    ----------
    sequences  : (N, T, F) float32  — historical feature sequences
    labels     : (N,)      int64    — class labels
    images     : (N, C, H, W) float32  — optional normalised images [0,1]
    hog_feats  : (N, D_hog) float32    — optional HOG + candle features
    """

    def __init__(
        self,
        sequences:  np.ndarray,
        labels:     np.ndarray,
        images:     Optional[np.ndarray] = None,
        hog_feats:  Optional[np.ndarray] = None,
    ):
        self.sequences  = torch.from_numpy(sequences)
        self.labels     = torch.from_numpy(labels)
        self.images     = torch.from_numpy(images)    if images    is not None else None
        self.hog_feats  = torch.from_numpy(hog_feats) if hog_feats is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "seq":   self.sequences[idx],
            "label": self.labels[idx],
        }
        if self.images    is not None: item["img"] = self.images[idx]
        if self.hog_feats is not None: item["hog"] = self.hog_feats[idx]
        return item


def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Return a sampler that up-samples minority classes."""
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
    weights = np.array([class_weights[l] for l in labels], dtype=np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


def build_loaders(
    train_ds: StockDataset,
    val_ds:   StockDataset,
    batch_size: int = 32,
    balance: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    sampler = make_balanced_sampler(train_ds.labels.numpy()) if balance else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


# ═════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = -np.inf
        self.stop       = False

    def __call__(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def _step_multimodal(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for multi-modal model; returns (logits, labels)."""
    seq   = batch["seq"].to(device)
    label = batch["label"].to(device)
    img   = batch["img"].to(device)
    hog   = batch["hog"].to(device)
    return model(img, hog, seq), label


def _step_baseline(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    seq   = batch["seq"].to(device)
    label = batch["label"].to(device)
    return model(seq), label


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_multimodal: bool,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """Return (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    step_fn = _step_multimodal if is_multimodal else _step_baseline

    for batch in loader:
        logits, labels = step_fn(batch, model, device)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_multimodal: bool,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    step_fn = _step_multimodal if is_multimodal else _step_baseline

    for batch in loader:
        logits, labels = step_fn(batch, model, device)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def train_model(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    cfg:           dict,
    is_multimodal: bool = True,
    ckpt_path:     Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Full training loop.

    Returns
    -------
    history : dict with keys train_loss, val_loss, train_acc, val_acc
    """
    t_cfg   = cfg.get("training", {})
    epochs  = t_cfg.get("epochs", 10)
    lr      = t_cfg.get("learning_rate", 1e-3)
    wd      = t_cfg.get("weight_decay", 1e-4)
    pat     = t_cfg.get("early_stopping_patience", 4)

    device    = get_device()
    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    stopper   = EarlyStopping(patience=pat)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, is_multimodal
        )
        vl_loss, vl_acc, _, _ = evaluate_epoch(
            model, val_loader, criterion, device, is_multimodal
        )
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        logger.info(
            f"Epoch {epoch:>3}/{epochs}  "
            f"train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
            f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.6f}  "
            f"[{elapsed:.1f}s]"
        )

        if vl_acc > best_val_acc and ckpt_path:
            best_val_acc = vl_acc
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_acc": vl_acc, "config": cfg},
                ckpt_path,
                logger,
            )

        if stopper(vl_acc):
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    return history


# ═════════════════════════════════════════════════════════════════════════════
#  Prediction helpers
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(
    model:         nn.Module,
    loader:        DataLoader,
    device:        torch.device,
    is_multimodal: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, probabilities) arrays."""
    model.eval()
    model.to(device)
    all_preds, all_probs = [], []
    step_fn = _step_multimodal if is_multimodal else _step_baseline

    for batch in loader:
        logits, _ = step_fn(batch, model, device)
        probs = torch.softmax(logits, dim=-1)
        all_preds.extend(probs.argmax(dim=-1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_probs)
