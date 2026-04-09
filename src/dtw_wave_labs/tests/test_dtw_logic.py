"""
test_dtw_logic.py — Unit tests for DTW Wave Labs core math.

Tests cover:
  - normalizer.py: min/max contract, zero-division safety, SavGol smoke-test
  - templates.py:  all 8 profiles have correct length and value range
  - dtw_matcher.py: bullish impulse synthetic signal → score < threshold;
                    bearish signal against BULLISH_IMPULSE → score > threshold
  - rule_validator.py: clean 5-wave impulse → True;
                       wave-4 overlap violation → False

Run from project root:
    .venv\\Scripts\\python.exe -m pytest src/dtw_wave_labs/tests/test_dtw_logic.py -v
"""

import os
import sys
import unittest

import numpy as np

# Ensure the project root is importable when pytest is run from root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dtw_wave_labs.config import TEMPLATE_RESOLUTION, DTW_DISTANCE_THRESHOLD
from src.dtw_wave_labs.templates import get_all_templates
from src.dtw_wave_labs.normalizer import minmax_normalize, prepare_window
from src.dtw_wave_labs.dtw_matcher import find_best_match
from src.dtw_wave_labs.rule_validator import validate


# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizer(unittest.TestCase):

    def test_minmax_range(self):
        arr = np.array([10.0, 20.0, 15.0, 5.0, 25.0])
        norm = minmax_normalize(arr)
        self.assertAlmostEqual(norm.min(), 0.0, places=9)
        self.assertAlmostEqual(norm.max(), 1.0, places=9)

    def test_constant_array_no_zerodiv(self):
        arr = np.full(50, 42.0)
        norm = minmax_normalize(arr)
        self.assertTrue(np.all(np.isfinite(norm)), "No NaN/Inf expected")
        self.assertAlmostEqual(norm.mean(), 0.5, places=9)

    def test_prepare_window_output_range(self):
        n = 55
        rng = np.random.default_rng(42)
        h = rng.uniform(100, 110, n)
        l = h - rng.uniform(1, 3, n)
        c = (h + l) / 2
        result = prepare_window(h, l, c)
        self.assertEqual(len(result), n)
        self.assertGreaterEqual(result.min(), 0.0 - 1e-9)
        self.assertLessEqual(result.max(), 1.0 + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────

class TestTemplates(unittest.TestCase):

    def setUp(self):
        self.templates = get_all_templates()

    def test_count(self):
        self.assertEqual(len(self.templates), 8, "Expected exactly 8 templates")

    def test_names(self):
        expected = {
            "BULLISH_IMPULSE", "BULLISH_ZIGZAG", "BULLISH_FLAT", "BULLISH_TRIANGLE",
            "BEARISH_IMPULSE", "BEARISH_ZIGZAG", "BEARISH_FLAT", "BEARISH_TRIANGLE",
        }
        self.assertEqual(set(self.templates.keys()), expected)

    def test_length(self):
        for name, arr in self.templates.items():
            self.assertEqual(
                len(arr), TEMPLATE_RESOLUTION,
                f"{name} has length {len(arr)}, expected {TEMPLATE_RESOLUTION}"
            )

    def test_value_range(self):
        for name, arr in self.templates.items():
            self.assertGreaterEqual(
                arr.min(), -1e-9, f"{name} has values below 0"
            )
            self.assertLessEqual(
                arr.max(), 1.0 + 1e-9, f"{name} has values above 1"
            )

    def test_bearish_is_inversion(self):
        bullish = self.templates["BULLISH_IMPULSE"]
        bearish = self.templates["BEARISH_IMPULSE"]
        np.testing.assert_allclose(bearish, 1.0 - bullish, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────

class TestDTWMatcher(unittest.TestCase):

    def setUp(self):
        self.templates = get_all_templates()

    def _synthetic_impulse_signal(self, n: int = 89, bullish: bool = True) -> tuple:
        """
        Build a clean 5-wave synthetic OHLCV snippet that strongly resembles
        the BULLISH_IMPULSE template (after normalization).
        """
        template = self.templates["BULLISH_IMPULSE"]
        # Resample template to length n
        x_template = np.linspace(0, 1, len(template))
        x_window   = np.linspace(0, 1, n)
        close = np.interp(x_window, x_template, template)
        if not bullish:
            close = 1.0 - close
        # Scale to realistic BTC prices
        close = close * 5000 + 60000
        high  = close + np.random.default_rng(0).uniform(50, 150, n)
        low   = close - np.random.default_rng(1).uniform(50, 150, n)
        return high, low, close

    def test_bullish_signal_matches_bullish_template(self):
        high, low, close = self._synthetic_impulse_signal(89, bullish=True)
        h_slice = high
        l_slice = low
        norm = prepare_window(high, low, close)
        results = find_best_match(norm, h_slice, l_slice, self.templates)

        # BULLISH_IMPULSE must appear in results below threshold
        pattern_names = [r["pattern"] for r in results]
        self.assertIn(
            "BULLISH_IMPULSE", pattern_names,
            f"Expected BULLISH_IMPULSE in matches, got: {pattern_names}"
        )
        bi_score = next(r["dtw_score"] for r in results if r["pattern"] == "BULLISH_IMPULSE")
        self.assertLess(bi_score, DTW_DISTANCE_THRESHOLD)

    def test_bearish_signal_does_not_match_bullish_template(self):
        """A bearish signal should NOT have a low DTW score on BULLISH_IMPULSE."""
        high, low, close = self._synthetic_impulse_signal(89, bullish=False)
        norm = prepare_window(high, low, close)
        bullish_only = {"BULLISH_IMPULSE": self.templates["BULLISH_IMPULSE"]}
        results = find_best_match(norm, high, low, bullish_only)
        # Should be empty (score above threshold)
        self.assertEqual(len(results), 0, "Bearish signal must NOT match bullish template")


# ─────────────────────────────────────────────────────────────────────────────

class TestRuleValidator(unittest.TestCase):

    def _make_clean_bullish_impulse_pivots(self):
        """Pivots that satisfy all 3 impulse rules."""
        return {
            0: 60000.0,   # wave 0 start (low)
            1: 62000.0,   # wave 1 peak  (high)
            2: 61100.0,   # wave 2 low   > wave 0 ✓
            3: 65000.0,   # wave 3 peak  (longest) ✓
            4: 63500.0,   # wave 4 low   > wave 1 high ✓
            5: 64800.0,   # wave 5 peak
        }

    def test_clean_impulse_passes(self):
        pivots = self._make_clean_bullish_impulse_pivots()
        passed, reason = validate("BULLISH_IMPULSE", pivots)
        self.assertTrue(passed, f"Should pass but got: {reason}")

    def test_wave4_overlap_fails(self):
        """Wave 4 low dips into wave 1 territory → must fail."""
        pivots = self._make_clean_bullish_impulse_pivots()
        pivots[4] = 61500.0  # below wave-1 high of 62000 → INVALID
        passed, reason = validate("BULLISH_IMPULSE", pivots)
        self.assertFalse(passed, "Wave-4 overlap with Wave-1 should fail")
        self.assertIn("Wave-4", reason)

    def test_wave2_retrace_fails(self):
        """Wave 2 retracing below wave 0 start → must fail."""
        pivots = self._make_clean_bullish_impulse_pivots()
        pivots[2] = 59000.0  # below wave 0 at 60000 → INVALID
        passed, reason = validate("BULLISH_IMPULSE", pivots)
        self.assertFalse(passed, "Wave-2 below wave-0 should fail")

    def test_wave3_shortest_fails(self):
        """Wave 3 being the shortest of 1, 3, 5 → must fail."""
        pivots = {
            0: 60000.0,
            1: 65000.0,   # wave 1: len = 5000
            2: 62000.0,
            3: 63500.0,   # wave 3: len = 1500  ← shortest → INVALID
            4: 62500.0,
            5: 65800.0,   # wave 5: len = 3300
        }
        passed, reason = validate("BULLISH_IMPULSE", pivots)
        self.assertFalse(passed, "Wave-3 as shortest should fail")

    def test_zigzag_valid(self):
        pivots = {
            0: 65000.0,   # A start (top)
            1: 60000.0,   # A end / B start (bottom)
            2: 63000.0,   # B end (< 65000, does not exceed A) ✓
            3: 58000.0,   # C end (< 60000, extends below A) ✓
        }
        passed, reason = validate("BEARISH_ZIGZAG", pivots)
        self.assertTrue(passed, f"Should pass but got: {reason}")

    def test_flat_b_ratio_fails(self):
        pivots = {
            0: 65000.0,
            1: 60000.0,   # A len = 5000
            2: 61500.0,   # B len = 1500 → 30% of A < 90% → FAIL
            3: 59000.0,
        }
        passed, reason = validate("BEARISH_FLAT", pivots)
        self.assertFalse(passed, "Flat B retracement < 90% should fail")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
