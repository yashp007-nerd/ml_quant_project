"""
image_generator.py
------------------
Generate candlestick chart images from OHLCV data using mplfinance.
Images are cached to disk to avoid regeneration on subsequent runs.
"""

from __future__ import annotations

import gc
import io
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .utils import ensure_dir, get_logger

logger = get_logger(__name__)

try:
    import mplfinance as mpf
    _MPF_AVAILABLE = True
except ImportError:
    _MPF_AVAILABLE = False
    logger.warning("mplfinance not installed – image generation will be skipped.")


# ─────────────────────────────── core renderer ──────────────────────────────

def render_candlestick(
    window_df: pd.DataFrame,
    img_size: int = 64,
    dpi: int = 80,
    style: str = "classic",
) -> Optional[np.ndarray]:
    """
    Render a single candlestick chart from *window_df* (OHLCV, N rows).

    Returns
    -------
    np.ndarray  shape (img_size, img_size, 3), dtype uint8
    None        if mplfinance is unavailable or rendering fails.
    """
    if not _MPF_AVAILABLE:
        return None

    # mplfinance requires a DatetimeIndex
    if not isinstance(window_df.index, pd.DatetimeIndex):
        window_df = window_df.copy()
        window_df.index = pd.to_datetime(window_df.index)

    fig_px = img_size / dpi          # figure size in inches
    buf = io.BytesIO()
    try:
        fig, _ = mpf.plot(
            window_df,
            type="candle",
            style=style,
            volume=False,
            axisoff=True,
            returnfig=True,
            figsize=(fig_px, fig_px),
            tight_layout=True,
        )
        fig.savefig(buf, format="jpeg", dpi=dpi, quality=40,
                    bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        img = Image.open(buf).convert("RGB").resize((img_size, img_size),
                                                     Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        fig.clf()
        import matplotlib.pyplot as plt
        plt.close(fig)
        del fig
        gc.collect()
        return arr
    except Exception as exc:
        logger.debug(f"render_candlestick failed: {exc}")
        return None


# ─────────────────────────────── batch generation ───────────────────────────

def generate_images_for_ticker(
    df: pd.DataFrame,
    window: int,
    img_size: int = 64,
    dpi: int = 80,
    style: str = "classic",
    cache_dir: Optional[str | Path] = None,
    ticker: str = "",
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate one image per valid rolling window of *window* days.

    Parameters
    ----------
    df        : DataFrame with at least Open/High/Low/Close columns.
    window    : Number of candles per chart.
    cache_dir : If given, cache results to disk.

    Returns
    -------
    images  : list of ndarray, each (img_size, img_size, 3)
    indices : list of integer row indices corresponding to each image
              (the *last* row of the window).
    """
    if cache_dir is not None:
        ensure_dir(cache_dir)
        fname = Path(cache_dir) / f"{ticker.replace('^','')}_w{window}_{img_size}px.pkl"
        if fname.exists():
            logger.info(f"[cache] loading images {fname}")
            with open(fname, "rb") as fh:
                return pickle.load(fh)

    images: List[np.ndarray] = []
    indices: List[int] = []

    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]

    logger.info(
        f"Generating {len(df) - window + 1} images "
        f"(ticker={ticker or 'df'}, window={window}, size={img_size})"
    )

    for i in range(len(df) - window + 1):
        chunk = df.iloc[i : i + window][ohlc_cols]
        img = render_candlestick(chunk, img_size=img_size, dpi=dpi, style=style)
        if img is None:
            # Fallback: blank (zeros) image — model will down-weight these
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        images.append(img)
        indices.append(i + window - 1)

        if (i + 1) % 500 == 0:
            gc.collect()
            logger.info(f"  {i + 1}/{len(df) - window + 1} images done")

    if cache_dir is not None:
        with open(fname, "wb") as fh:
            pickle.dump((images, indices), fh)
        logger.info(f"[cache] saved → {fname}")

    return images, indices


# ─────────────────────────────── tensor conversion ──────────────────────────

def images_to_tensor_array(
    images: List[np.ndarray],
    img_size: int = 64,
) -> np.ndarray:
    """
    Stack list of (H, W, C) uint8 images into float32 array
    of shape (N, C, H, W) normalised to [0, 1].
    """
    arr = np.stack(images, axis=0).astype(np.float32) / 255.0
    # (N, H, W, C) → (N, C, H, W)
    return np.transpose(arr, (0, 3, 1, 2))
