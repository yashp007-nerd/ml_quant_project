"""
feature_engineering.py
-----------------------
Compute technical indicators (RSI, MACD, EMA, Bollinger Bands, ATR, etc.)
and assemble the final numerical feature matrix for the LSTM/TFT branch.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from .utils import get_logger, zscore_scale

logger = get_logger(__name__)


# ─────────────────────────────── indicator computation ──────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a rich set of technical indicators to *df* (in-place copy).

    Indicators added:
      - EMA (10, 20, 50 day)
      - RSI (14)
      - MACD line & signal
      - Bollinger Bands (upper, lower, % width)
      - ATR (14)
      - On-Balance Volume (OBV)
      - Returns (1-day, 5-day)
    Falls back to manual numpy implementation if `ta` is not installed.
    """
    out = df.copy()
    close = out["Close"]
    high  = out["High"]
    low   = out["Low"]
    vol   = out["Volume"]

    if _TA_AVAILABLE:
        # ── EMA ──────────────────────────────────────────────────────────────
        out["ema_10"] = ta.trend.EMAIndicator(close, window=10).ema_indicator()
        out["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
        out["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()

        # ── RSI ──────────────────────────────────────────────────────────────
        out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        # ── MACD ─────────────────────────────────────────────────────────────
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        out["macd"]        = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macd_hist"]   = macd.macd_diff()

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        out["bb_upper"] = bb.bollinger_hband()
        out["bb_lower"] = bb.bollinger_lband()
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / close

        # ── ATR ───────────────────────────────────────────────────────────────
        out["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

        # ── OBV ───────────────────────────────────────────────────────────────
        out["obv"] = ta.volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()

    else:
        logger.warning("`ta` library not found – using manual indicator calculations.")
        # ── Manual EMA ────────────────────────────────────────────────────────
        for w in [10, 20, 50]:
            out[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()

        # ── Manual RSI ────────────────────────────────────────────────────────
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        out["rsi_14"] = 100 - 100 / (1 + rs)

        # ── Manual MACD ───────────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        out["macd"]        = ema12 - ema26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_hist"]   = out["macd"] - out["macd_signal"]

        # ── Manual BB ─────────────────────────────────────────────────────────
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        out["bb_upper"] = sma20 + 2 * std20
        out["bb_lower"] = sma20 - 2 * std20
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / close

        # ── Manual ATR ────────────────────────────────────────────────────────
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr_14"] = tr.rolling(14).mean()

        # ── OBV ──────────────────────────────────────────────────────────────
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + vol.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - vol.iloc[i])
            else:
                obv.append(obv[-1])
        out["obv"] = obv

    # ── Price returns ─────────────────────────────────────────────────────────
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)

    # ── Normalised OHLCV ─────────────────────────────────────────────────────
    out["norm_open"]   = out["Open"]   / close
    out["norm_high"]   = out["High"]   / close
    out["norm_low"]    = out["Low"]    / close
    out["norm_volume"] = np.log1p(out["Volume"])

    return out


# ─────────────────────────────── feature list ───────────────────────────────

FEATURE_COLS: List[str] = [
    "norm_open", "norm_high", "norm_low", "norm_volume",
    "ret_1d", "ret_5d",
    "ema_10", "ema_20", "ema_50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "atr_14",
    "obv",
]


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    normalise: bool = True,
) -> np.ndarray:
    """
    Return the numerical feature matrix as a float32 ndarray.

    Parameters
    ----------
    df           : DataFrame with indicators already added.
    feature_cols : Which columns to use (defaults to FEATURE_COLS).
    normalise    : Z-score normalise each column independently.
    """
    cols = feature_cols or FEATURE_COLS
    # Keep only columns that exist in df
    cols = [c for c in cols if c in df.columns]
    mat = df[cols].values.astype(np.float32)
    if normalise:
        mat = zscore_scale(mat)
    return mat


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) where X[i] = features[i : i+window]  (shape: window × n_feat)
    and y[i] = labels[i + window - 1].

    Only rows where labels are not NaN are kept.
    """
    X, y = [], []
    for i in range(len(features) - window + 1):
        lbl = labels[i + window - 1]
        if np.isnan(lbl):
            continue
        X.append(features[i : i + window])
        y.append(int(lbl))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
