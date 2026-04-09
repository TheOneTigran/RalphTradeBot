"""
normalizer.py — Price window preparation for DTW matching.

Pipeline:
  1. Compute typical price: (High + Low + Close) / 3
  2. Apply Savitzky-Golay filter to remove noise while preserving extrema
  3. Min-Max normalize to [0.0, 1.0]

IMPORTANT: The smoothed+normalized output is ONLY passed to the DTW engine.
The rule_validator always receives the raw OHLCV data (original High/Low wicks).
"""

import numpy as np
from scipy.signal import savgol_filter
from .config import SAVGOL_POLYORDER


def compute_typical_price(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """Compute typical price (HLC/3) for a window."""
    return (high + low + close) / 3.0


def smooth_window(prices: np.ndarray, polyorder: int = SAVGOL_POLYORDER) -> np.ndarray:
    """
    Apply Savitzky-Golay filter.
    window_length must be odd and > polyorder; we pick the largest odd number <= len/3
    to get meaningful smoothing without over-flattening.
    Falls back to the raw prices if the window is too small for filtering.
    """
    # Ensure strictly 1D — savgol_filter will raise ValueError on 2D input
    prices = np.asarray(prices, dtype=float).ravel()
    n = len(prices)
    # window_length: odd, at least polyorder+2, at most n (if n is odd) or n-1
    wl = max(polyorder + 2, n // 4 * 2 + 1)  # nearest odd >= n//4
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, n if n % 2 == 1 else n - 1)
    if wl <= polyorder:
        return prices.copy()
    return savgol_filter(prices, window_length=wl, polyorder=polyorder)


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Min-Max normalize to [0.0, 1.0].
    Returns a flat constant array of 0.5 if max == min (avoid ZeroDivisionError).
    """
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - mn) / (mx - mn)


def prepare_window(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Full preparation pipeline:
      typical price → SavGol smooth → Min-Max normalize → shape (n,)
    Returns the normalized float array ready for DTW.
    """
    typical = compute_typical_price(high, low, close)
    smoothed = smooth_window(typical)
    normalized = minmax_normalize(smoothed)
    return normalized


def sliding_windows(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window_size: int,
):
    """
    Generator yielding (start_idx, norm_window, raw_high_slice, raw_low_slice)
    for every sliding window of `window_size` over the full price series.

    Stride is always 1: the market does not adapt to our step size.
    Missing even one candle offset could mean missing the start of a clean impulse.
    """
    n = len(close)
    for start in range(n - window_size + 1):
        end = start + window_size
        h_slice = high[start:end]
        l_slice = low[start:end]
        c_slice = close[start:end]
        norm = prepare_window(h_slice, l_slice, c_slice)
        yield start, norm, h_slice, l_slice
