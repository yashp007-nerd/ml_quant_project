"""
data_loader.py
--------------
Download OHLCV data via yfinance, apply time-series splits, and expose
a clean DataFrame interface for downstream modules.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit

from .utils import ensure_dir, get_logger

logger = get_logger(__name__)

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ─────────────────────────────── download ───────────────────────────────────

def download_ticker(
    ticker: str,
    start: str,
    end: str,
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """
    Download daily OHLCV for *ticker* between *start* and *end*.
    Results are cached as pickle files to avoid re-downloading.
    """
    ensure_dir(cache_dir)
    key = hashlib.md5(f"{ticker}_{start}_{end}".encode()).hexdigest()[:10]
    cache_file = Path(cache_dir) / f"{ticker.replace('^', '')}_{key}.pkl"

    if cache_file.exists():
        logger.info(f"[cache] loading {ticker}")
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    logger.info(f"[yfinance] downloading {ticker} {start} → {end}")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        logger.warning(f"Empty data returned for {ticker}. Skipping.")
        return pd.DataFrame()

    # Flatten MultiIndex columns that yfinance sometimes produces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[REQUIRED_COLS].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    with open(cache_file, "wb") as fh:
        pickle.dump(df, fh)

    logger.info(f"  {ticker}: {len(df)} rows  ({df.index[0].date()} – {df.index[-1].date()})")
    return df


def download_all(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: str | Path = "data/cache",
) -> Dict[str, pd.DataFrame]:
    """Download all tickers; return {ticker: DataFrame}."""
    result: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = download_ticker(t, start, end, cache_dir)
        if not df.empty:
            result[t] = df
    logger.info(f"Loaded {len(result)}/{len(tickers)} tickers successfully.")
    return result


# ─────────────────────────────── labelling ──────────────────────────────────

def label_price_movement(
    df: pd.DataFrame,
    n: int = 1,
    threshold: float = 0.005,
) -> pd.Series:
    """
    Label each day based on n-day-ahead return:
        BUY  (1)  →  return > +threshold
        SELL (0)  →  return < -threshold
        HOLD (2)  →  |return| ≤ threshold

    Follows Eq. (21)-(22) from the paper.
    """
    future_close = df["Close"].shift(-n)
    ret = (future_close - df["Close"]) / df["Close"]

    label = pd.Series(2, index=df.index, name=f"label_{n}d")  # default HOLD
    label[ret > threshold] = 1   # BUY
    label[ret < -threshold] = 0  # SELL
    return label


# ─────────────────────────────── time-series split ──────────────────────────

def time_series_split(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Return a list of (train_df, test_df) tuples using sklearn's
    TimeSeriesSplit — NO data leakage.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(len(df))
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_idx, test_idx in tss.split(indices):
        splits.append((df.iloc[train_idx].copy(), df.iloc[test_idx].copy()))
    return splits


# ─────────────────────────────── summary stats ──────────────────────────────

def summarise(df: pd.DataFrame, ticker: str = "") -> None:
    """Log basic summary statistics."""
    prefix = f"[{ticker}] " if ticker else ""
    logger.info(
        f"{prefix}rows={len(df)}  "
        f"start={df.index[0].date()}  end={df.index[-1].date()}  "
        f"close_mean={df['Close'].mean():.2f}  "
        f"close_std={df['Close'].std():.2f}"
    )
