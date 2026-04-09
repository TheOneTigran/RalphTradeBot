"""
rule_validator.py — Hard 89WAVES rule enforcement for DTW-found patterns.

The validator receives RAW (unsmoothed) High/Low pivot prices from the
original OHLCV data and applies strict Elliott Wave rules.

Input pivots dict:  {wave_index: price}
  IMPULSE:  {0: start, 1: w1_peak, 2: w2_low, 3: w3_peak, 4: w4_low, 5: w5_peak}
  ZIGZAG:   {0: A_start, 1: A_end_B_start, 2: B_end_C_start, 3: C_end}
  FLAT:     {0: A_start, 1: A_end_B_start, 2: B_end_C_start, 3: C_end}
  TRIANGLE: {0: A_high, 1: A_low, 2: B_high, 3: B_low, 4: C_high, 5: C_low}

Returns: (bool, str) — (passed, reason_if_failed)
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wave_len(p_start: float, p_end: float) -> float:
    return abs(p_end - p_start)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern validators
# ─────────────────────────────────────────────────────────────────────────────

def _validate_impulse(pivots: dict[int, float], is_bullish: bool) -> tuple[bool, str]:
    """
    Bullish impulse pivots: {0: start_low, 1: w1_high, 2: w2_low,
                              3: w3_high, 4: w4_low, 5: w5_high}
    Rules (applied to lows/highs per direction):
      1. Wave 2 does not retrace below Wave 0 start.
      2. Wave 4 does not overlap Wave 1 territory (low[4] > high[1]).
      3. Wave 3 is not the shortest of waves 1, 3, 5.
    """
    try:
        p0 = pivots[0]
        p1 = pivots[1]
        p2 = pivots[2]
        p3 = pivots[3]
        p4 = pivots[4]
        p5 = pivots[5]
    except KeyError as e:
        return False, f"Missing pivot key: {e}"

    if is_bullish:
        # Rule 1: Wave 2 low must be above Wave 0 low
        if p2 <= p0:
            return False, f"Wave-2 ({p2:.2f}) retraced below Wave-0 start ({p0:.2f})"
        # Rule 2: Wave 4 low must be above Wave 1 high
        if p4 <= p1:
            return False, f"Wave-4 ({p4:.2f}) overlapped Wave-1 high ({p1:.2f})"
    else:
        # Bearish: highs and lows are inverted
        if p2 >= p0:
            return False, f"Bear Wave-2 ({p2:.2f}) retraced above Wave-0 start ({p0:.2f})"
        if p4 >= p1:
            return False, f"Bear Wave-4 ({p4:.2f}) overlapped Wave-1 low ({p1:.2f})"

    # Rule 3: Wave 3 is not the shortest (both directions same logic on lengths)
    len1 = _wave_len(p0, p1)
    len3 = _wave_len(p2, p3)
    len5 = _wave_len(p4, p5)

    if len3 < len1 and len3 < len5:
        return False, (
            f"Wave-3 ({len3:.2f}) is shortest of w1={len1:.2f}, w3={len3:.2f}, w5={len5:.2f}"
        )

    return True, "OK"


def _validate_zigzag(pivots: dict[int, float], is_bullish: bool) -> tuple[bool, str]:
    """
    Zigzag corrective pattern — ABC.

    Bullish zigzag (upward correction after downtrend):
      Pivots: {0: A_start_low, 1: A_end_B_start_high, 2: B_end_C_start_low, 3: C_end_high}
      Rules: B must not retrace fully back to A start (B < A_start);
             C must extend above A_end.

    Bearish zigzag (downward correction after uptrend):
      Pivots: {0: A_start_high, 1: A_end_B_start_low, 2: B_end_C_start_high, 3: C_end_low}
      Rules: B_high must stay below A_start_high (< 100% retrace);
             C must extend below A_end (new low).
    """
    try:
        p_a_start = pivots[0]
        p_a_end   = pivots[1]
        p_b_end   = pivots[2]
        p_c_end   = pivots[3]
    except KeyError as e:
        return False, f"Missing pivot key: {e}"

    if is_bullish:
        # Bullish: A goes up, B retraces down, C pushes to new high
        # B must not reach A_start (full retrace would be bearish continuation)
        if p_b_end <= p_a_start:
            return False, f"Bull ZigZag B ({p_b_end:.2f}) fully retraced to A start ({p_a_start:.2f})"
        # C must exceed A_end (establish new high)
        if p_c_end <= p_a_end:
            return False, f"Bull ZigZag C ({p_c_end:.2f}) does not exceed A's peak ({p_a_end:.2f})"
    else:
        # Bearish: A falls, B bounces, C falls to new low
        # B bounce must stay below A_start high (< 100% retrace)
        if p_b_end >= p_a_start:
            return False, f"Bear ZigZag B ({p_b_end:.2f}) >= A start ({p_a_start:.2f}) — exceeds 100% retrace"
        # C must push below A_end (establish new low)
        if p_c_end >= p_a_end:
            return False, f"Bear ZigZag C ({p_c_end:.2f}) does not extend below A trough ({p_a_end:.2f})"

    return True, "OK"


def _validate_flat(pivots: dict[int, float], is_bullish: bool) -> tuple[bool, str]:
    """
    Flat correction:
      Pivots: {0: A_start, 1: A_end, 2: B_end, 3: C_end}

    Rules:
      1. B retracement >= 90% of A's range
      2. C length ≈ A length (within 30%)
    """
    try:
        p_a_start = pivots[0]
        p_a_end   = pivots[1]
        p_b_end   = pivots[2]
        p_c_end   = pivots[3]
    except KeyError as e:
        return False, f"Missing pivot key: {e}"

    len_a = _wave_len(p_a_start, p_a_end)
    len_b = _wave_len(p_a_end, p_b_end)
    len_c = _wave_len(p_b_end, p_c_end)

    if len_a < 1e-9:
        return False, "Wave A has zero length"

    # Rule 1: B >= 90% of A
    b_retrace_ratio = len_b / len_a
    if b_retrace_ratio < 0.90:
        return False, f"Flat B retrace {b_retrace_ratio:.2%} < 90% of A"

    # Rule 2: C length within ±30% of A
    c_ratio = len_c / len_a
    if not (0.70 <= c_ratio <= 1.30):
        return False, f"Flat C/A ratio {c_ratio:.2%} outside 70–130%"

    return True, "OK"


def _validate_triangle(pivots: dict[int, float], is_bullish: bool) -> tuple[bool, str]:
    """
    Contracting triangle ABCDE:
      Pivots: {0: A_apex, 1: lower_A, 2: B_apex, 3: lower_B, 4: C_apex, 5: lower_C}
      (highs and lows alternating)

    Rule: Each successive extreme is less extreme than the previous one.
      |A_apex - lower_A| > |B_apex - lower_B| > ...
    """
    try:
        amplitudes = []
        for i in range(0, len(pivots) - 1, 2):
            hi = pivots[i]
            lo = pivots[i + 1]
            amplitudes.append(abs(hi - lo))
    except KeyError as e:
        return False, f"Missing pivot key: {e}"

    for i in range(len(amplitudes) - 1):
        if amplitudes[i + 1] >= amplitudes[i]:
            return False, (
                f"Triangle not contracting at leg {i}→{i+1}: "
                f"{amplitudes[i]:.2f} → {amplitudes[i+1]:.2f}"
            )

    return True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate(pattern_name: str, pivots: dict[int, float]) -> tuple[bool, str]:
    """
    Validate a DTW-matched pattern against hard 89WAVES rules.

    Args:
        pattern_name: e.g. "BULLISH_IMPULSE", "BEARISH_ZIGZAG", ...
        pivots:       dict of {wave_index: raw_price}  (High/Low wicks)

    Returns:
        (True, "OK") if valid, (False, reason_string) if rejected.
    """
    is_bullish = pattern_name.startswith("BULLISH")
    base = pattern_name.replace("BULLISH_", "").replace("BEARISH_", "")

    dispatch = {
        "IMPULSE":  _validate_impulse,
        "ZIGZAG":   _validate_zigzag,
        "FLAT":     _validate_flat,
        "TRIANGLE": _validate_triangle,
    }

    validator_fn = dispatch.get(base)
    if validator_fn is None:
        return False, f"Unknown pattern type: {base}"

    return validator_fn(pivots, is_bullish)
