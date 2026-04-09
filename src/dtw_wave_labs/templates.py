"""
templates.py — Programmatic generator of ideal Elliott Wave template vectors.

Produces 8 normalized profiles:
  4 bullish (IMPULSE, ZIGZAG, FLAT, TRIANGLE) built via Fibonacci-proportioned
  linear interpolation, and 4 bearish counterparts obtained by inversion:
  bearish = 1.0 - bullish.

Each template is a numpy array of shape (TEMPLATE_RESOLUTION,)
with all values in [0.0, 1.0].
"""

import numpy as np
from .config import TEMPLATE_RESOLUTION


def _interpolate(key_x: list[float], key_y: list[float], n: int) -> np.ndarray:
    """
    Build a smooth piecewise-linear vector of length `n` from key points.
    key_x: relative positions in [0, 1]
    key_y: normalized price values in [0, 1]
    """
    x = np.linspace(0.0, 1.0, n)
    return np.interp(x, key_x, key_y)


def _make_bullish_impulse(n: int) -> np.ndarray:
    """
    Classic 5-wave bullish impulse (Fibonacci proportions):
      0: start at 0.0
      1: wave-1 peak  ~0.382
      2: wave-2 low   ~0.146  (38.2% retrace of wave 1)
      3: wave-3 peak  ~1.000  (extended, strongest wave)
      4: wave-4 low   ~0.618  (does NOT overlap wave-1 territory)
      5: wave-5 peak  ~0.910  (slightly below wave-3)
    """
    key_x = [0.0,   0.18,  0.35,  0.60,  0.78,  1.00]
    key_y = [0.000, 0.382, 0.146, 1.000, 0.618, 0.910]
    return _interpolate(key_x, key_y, n)


def _make_bullish_zigzag(n: int) -> np.ndarray:
    """
    ABC corrective zigzag (bearish correction after uptrend, ends lower):
      A: drops from top to ~0.35
      B: bounces to ~0.65  (< 100% of A)
      C: falls below A to ~0.0
    Viewed as a downward correction, we normalize so start=1.0 end≈0:
    """
    # Zigzag moves DOWN first (corrective after a bull run)
    key_x = [0.0,   0.40,  0.65,  1.00]
    key_y = [1.000, 0.350, 0.650, 0.000]
    return _interpolate(key_x, key_y, n)


def _make_bullish_flat(n: int) -> np.ndarray:
    """
    Flat corrective pattern (B ≈ A, C ≈ A length):
      A: drops to ~0.20
      B: rebounds to ~0.95 (≥ 90% of A's start)
      C: falls to ~0.10  (approximately equal to A)
    """
    key_x = [0.0,   0.35,  0.70,  1.00]
    key_y = [1.000, 0.200, 0.950, 0.100]
    return _interpolate(key_x, key_y, n)


def _make_bullish_triangle(n: int) -> np.ndarray:
    """
    Contracting triangle ABCDE — each successive extreme is less extreme:
      A: fall to 0.20
      B: rise  to 0.75
      C: fall  to 0.35
      D: rise  to 0.65
      E: fall  to 0.45  (converging)
    """
    key_x = [0.0,   0.20,  0.40,  0.60,  0.80,  1.00]
    key_y = [0.900, 0.200, 0.750, 0.350, 0.650, 0.450]
    return _interpolate(key_x, key_y, n)


def get_all_templates() -> dict[str, np.ndarray]:
    """
    Return all 8 template arrays keyed by name.
    Bearish variants are simple inversions: 1.0 - bullish_arr.
    """
    n = TEMPLATE_RESOLUTION

    bullish_impulse  = _make_bullish_impulse(n)
    bullish_zigzag   = _make_bullish_zigzag(n)
    bullish_flat     = _make_bullish_flat(n)
    bullish_triangle = _make_bullish_triangle(n)

    templates = {
        "BULLISH_IMPULSE":  bullish_impulse,
        "BULLISH_ZIGZAG":   bullish_zigzag,
        "BULLISH_FLAT":     bullish_flat,
        "BULLISH_TRIANGLE": bullish_triangle,
        "BEARISH_IMPULSE":  1.0 - bullish_impulse,
        "BEARISH_ZIGZAG":   1.0 - bullish_zigzag,
        "BEARISH_FLAT":     1.0 - bullish_flat,
        "BEARISH_TRIANGLE": 1.0 - bullish_triangle,
    }
    return templates
