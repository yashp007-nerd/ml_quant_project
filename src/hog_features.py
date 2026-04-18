"""
hog_features.py
---------------
Extract HOG (Histogram of Oriented Gradients) descriptors from candlestick
images and augment them with hand-crafted candlestick attributes:
  candle colour (bullish/bearish), body size, upper wick, lower wick.

Implements Eq. (7) from the paper:
    ft = [c, sb, su, sl, hc]  ∈  R^k
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from skimage.feature import hog
    from skimage.color import rgb2gray
    _SKIMAGE_AVAILABLE = True
except ImportError:
    _SKIMAGE_AVAILABLE = False

from .utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────── HOG extraction ─────────────────────────────

def extract_hog(
    image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """
    Compute HOG descriptor for a single (H, W, 3) or (H, W) uint8/float image.

    Returns 1-D float32 feature vector.
    Falls back to a random vector of the expected length if skimage is absent.
    """
    if not _SKIMAGE_AVAILABLE:
        # estimate output length and return zeros
        img_size = image.shape[0]
        cells = img_size // pixels_per_cell[0]
        blocks = max(1, cells - cells_per_block[0] + 1)
        length = blocks * blocks * cells_per_block[0] * cells_per_block[1] * orientations
        return np.zeros(length, dtype=np.float32)

    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        gray = image.astype(np.float64)

    feat = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        feature_vector=True,
    )
    return feat.astype(np.float32)


# ─────────────────────────────── candlestick attributes ─────────────────────

def extract_candle_features(window_df: pd.DataFrame) -> np.ndarray:
    """
    Extract hand-crafted features from the *last* candle in the window:
      - c   : colour  (1 = bullish / green, 0 = bearish / red)
      - sb  : body size  (|close - open| / price_range)
      - su  : upper wick (high - max(open, close)) / price_range
      - sl  : lower wick (min(open, close) - low)  / price_range

    Returns 1-D float32 array of length 4.
    """
    row = window_df.iloc[-1]
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])

    price_range = max(h - l, 1e-8)
    colour = 1.0 if c >= o else 0.0
    body   = abs(c - o) / price_range
    upper  = (h - max(o, c)) / price_range
    lower  = (min(o, c) - l) / price_range

    return np.array([colour, body, upper, lower], dtype=np.float32)


# ─────────────────────────────── combined descriptor ────────────────────────

def extract_full_descriptor(
    image: np.ndarray,
    window_df: pd.DataFrame,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
) -> np.ndarray:
    """
    ft = [c, sb, su, sl, hc]   (Eq. 7 of the paper)

    Concatenate candle attributes with HOG vector.
    """
    hog_feat    = extract_hog(image, orientations, pixels_per_cell, cells_per_block)
    candle_feat = extract_candle_features(window_df)
    return np.concatenate([candle_feat, hog_feat])


# ─────────────────────────────── batch extraction ────────────────────────────

def extract_hog_batch(
    images: List[np.ndarray],
    window_dfs: Optional[List[pd.DataFrame]] = None,
    orientations: int = 9,
    pixels_per_cell: Tuple[int, int] = (16, 16),
    cells_per_block: Tuple[int, int] = (2, 2),
    include_candle_feats: bool = True,
) -> np.ndarray:
    """
    Extract features for a list of images.

    Parameters
    ----------
    images              : list of (H, W, 3) uint8 images
    window_dfs          : parallel list of DataFrames (one per image window);
                          required when include_candle_feats=True
    include_candle_feats: whether to prepend the 4-dim candle attributes

    Returns
    -------
    np.ndarray  (N, feature_dim)  float32
    """
    results = []
    for i, img in enumerate(images):
        if include_candle_feats and window_dfs is not None:
            feat = extract_full_descriptor(img, window_dfs[i],
                                           orientations, pixels_per_cell,
                                           cells_per_block)
        else:
            feat = extract_hog(img, orientations, pixels_per_cell,
                                cells_per_block)
        results.append(feat)

        if (i + 1) % 500 == 0:
            logger.info(f"  HOG: {i + 1}/{len(images)} done")

    arr = np.stack(results, axis=0)
    logger.info(f"HOG batch done: shape={arr.shape}")
    return arr
