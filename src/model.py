"""
model.py
--------
Multi-modal architecture: CNN image encoder + LSTM-attention temporal encoder
+ late fusion → 3-class classifier (BUY / SELL / HOLD).

Also defines the LSTM-only baseline used for benchmarking.

Architecture overview (paper approximation):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Image (C×H×W)  ──►  CNN Encoder  ──►  img_emb  (D_img)           │
  │                                                                     │
  │  HOG+Candle feats  ──►  Linear Proj  ──►  hog_emb  (D_hog)        │
  │                                                                     │
  │  Hist features (T×F)  ──►  LSTM  ──►  Attention  ──►  seq_emb (D) │
  │                                                                     │
  │  [img_emb ‖ hog_emb ‖ seq_emb]  ──►  GRN ──►  Softmax (3 classes)│
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════════════════
#  Building blocks
# ═════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightCNNEncoder(nn.Module):
    """
    Lightweight 4-layer CNN.  Input: (B, 3, 64, 64) → Output: (B, embed_dim)
    Suitable for CPU / Colab free tier.
    """
    def __init__(self, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,  32, pool=True),   # → 32×32
            ConvBlock(32, 64, pool=True),   # → 16×16
            ConvBlock(64, 128, pool=True),  # → 8×8
            ConvBlock(128, 256, pool=True), # → 4×4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # → 256×1×1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


class ScaledDotProductAttention(nn.Module):
    """Single-head scaled dot-product attention over a sequence."""
    def __init__(self, hidden: int):
        super().__init__()
        self.scale = math.sqrt(hidden)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, H)
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale   # (B, T, T)
        weights = torch.softmax(scores, dim=-1)
        out = torch.bmm(weights, V)   # (B, T, H)
        return out[:, -1, :]          # return last time-step representation


class LSTMAttentionEncoder(nn.Module):
    """
    Bi-LSTM + self-attention temporal encoder.
    Approximates the TFT's variable-selection + multi-head attention blocks.
    Input: (B, T, F)  →  Output: (B, hidden)
    """
    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        n_layers: int = 2,
        attn_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, F)
        lstm_out, _ = self.lstm(x)                         # (B, T, H)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # (B, T, H)
        attn_out = self.norm(attn_out + lstm_out)          # residual + LN
        return self.drop(attn_out[:, -1, :])               # last time-step


class GatedResidualNetwork(nn.Module):
    """
    GRN as used in TFT (Eq. 15-17 of the paper).
    Z1 = ReLU(W1·x + b1)
    Z2 = W2·Z1 + b2
    out = LayerNorm(Z2 + x)
    """
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.w1   = nn.Linear(dim, dim)
        self.w2   = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = F.relu(self.w1(x))
        z2 = self.w2(z1)
        gate = torch.sigmoid(self.gate(x))
        return self.norm(self.drop(gate * z2) + x)


# ═════════════════════════════════════════════════════════════════════════════
#  Multi-Modal Model
# ═════════════════════════════════════════════════════════════════════════════

class MultiModalStockClassifier(nn.Module):
    """
    Full multi-modal model:
        CNN(image) + Linear(HOG) + LSTM-Attn(hist_seq) → late fusion → GRN → softmax
    """
    def __init__(
        self,
        num_ts_features: int,
        num_hog_features: int,
        img_embed_dim:  int   = 128,
        hog_proj_dim:   int   = 64,
        lstm_hidden:    int   = 128,
        lstm_layers:    int   = 2,
        attn_heads:     int   = 4,
        dropout:        float = 0.3,
        num_classes:    int   = 3,
    ):
        super().__init__()

        # ── Image branch ──────────────────────────────────────────────────────
        self.img_encoder = LightCNNEncoder(embed_dim=img_embed_dim, dropout=dropout)

        # ── HOG branch ────────────────────────────────────────────────────────
        self.hog_proj = nn.Sequential(
            nn.Linear(num_hog_features, hog_proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Temporal branch ───────────────────────────────────────────────────
        self.ts_encoder = LSTMAttentionEncoder(
            input_dim=num_ts_features,
            hidden=lstm_hidden,
            n_layers=lstm_layers,
            attn_heads=attn_heads,
            dropout=dropout,
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        fused_dim = img_embed_dim + hog_proj_dim + lstm_hidden
        self.grn  = GatedResidualNetwork(fused_dim, dropout=dropout)

        # ── Classifier head (Eq. 18-20) ───────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(
        self,
        img: torch.Tensor,       # (B, 3, H, W)
        hog: torch.Tensor,       # (B, hog_dim)
        seq: torch.Tensor,       # (B, T, F)
    ) -> torch.Tensor:           # (B, num_classes)
        img_emb = self.img_encoder(img)
        hog_emb = self.hog_proj(hog)
        seq_emb = self.ts_encoder(seq)

        fused  = torch.cat([img_emb, hog_emb, seq_emb], dim=-1)
        fused  = self.grn(fused)
        return self.head(fused)


# ═════════════════════════════════════════════════════════════════════════════
#  Baseline LSTM-only Model
# ═════════════════════════════════════════════════════════════════════════════

class BaselineLSTMClassifier(nn.Module):
    """
    Baseline: LSTM + attention on historical features only (no images, no HOG).
    Used for benchmarking against the multi-modal model.
    """
    def __init__(
        self,
        num_ts_features: int,
        lstm_hidden:  int   = 128,
        lstm_layers:  int   = 2,
        attn_heads:   int   = 4,
        dropout:      float = 0.3,
        num_classes:  int   = 3,
    ):
        super().__init__()
        self.encoder = LSTMAttentionEncoder(
            input_dim=num_ts_features,
            hidden=lstm_hidden,
            n_layers=lstm_layers,
            attn_heads=attn_heads,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, num_classes),
        )

    def forward(self, seq: torch.Tensor, **kwargs) -> torch.Tensor:
        emb = self.encoder(seq)
        return self.head(emb)


# ─────────────────────────────── factory ─────────────────────────────────────

def build_multimodal_model(
    num_ts_features: int,
    num_hog_features: int,
    cfg: dict,
) -> MultiModalStockClassifier:
    m = cfg.get("model", {})
    return MultiModalStockClassifier(
        num_ts_features=num_ts_features,
        num_hog_features=num_hog_features,
        img_embed_dim=m.get("image_embed_dim", 128),
        hog_proj_dim=m.get("hog_proj_dim", 64),
        lstm_hidden=m.get("lstm_hidden", 128),
        lstm_layers=m.get("lstm_layers", 2),
        attn_heads=m.get("attn_heads", 4),
        dropout=m.get("dropout", 0.3),
        num_classes=m.get("num_classes", 3),
    )


def build_baseline_model(num_ts_features: int, cfg: dict) -> BaselineLSTMClassifier:
    m = cfg.get("model", {})
    return BaselineLSTMClassifier(
        num_ts_features=num_ts_features,
        lstm_hidden=m.get("lstm_hidden", 128),
        lstm_layers=m.get("lstm_layers", 2),
        attn_heads=m.get("attn_heads", 4),
        dropout=m.get("dropout", 0.3),
        num_classes=m.get("num_classes", 3),
    )
