"""
pipeline.py — Main orchestrator for the DTW Wave Labs module.

Flow per window size:
  1. Slide window over full OHLCV data
  2. Normalize (typical price + SavGol + MinMax) → DTW match
  3. For each match below threshold: extract pivots from DTW path
  4. Validate with rule_validator (raw High/Low)
  5. Collect all results, then apply NMS to remove > 50% overlapping duplicates

Returns: list of PatternResult dicts (JSON-serializable)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

from .config import WINDOW_SIZES, NMS_OVERLAP_THRESHOLD
from .normalizer import sliding_windows
from .dtw_matcher import find_best_match
from .rule_validator import validate
from .templates import get_all_templates


@dataclass
class PatternResult:
    start_index: int
    end_index: int
    pattern: str
    dtw_score: float
    passed_validation: bool
    validation_reason: str
    pivots: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert int keys in pivots to strings for JSON-compliance
        d["pivots"] = {str(k): round(v, 4) for k, v in self.pivots.items()}
        return d


# ─────────────────────────────────────────────────────────────────────────────
# NMS — Non-Maximum Suppression
# ─────────────────────────────────────────────────────────────────────────────

def _overlap_fraction(a: PatternResult, b: PatternResult) -> float:
    """Intersection-over-Union of two index ranges."""
    inter_start = max(a.start_index, b.start_index)
    inter_end   = min(a.end_index,   b.end_index)
    if inter_end <= inter_start:
        return 0.0
    intersection = inter_end - inter_start
    union = max(a.end_index, b.end_index) - min(a.start_index, b.start_index)
    return intersection / union if union > 0 else 0.0


def _apply_nms(results: list[PatternResult]) -> list[PatternResult]:
    """
    Suppress duplicate detections of the same market movement.

    Two results are considered duplicates if:
      - Same pattern TYPE (e.g. both BULLISH_IMPULSE)
      - Temporal overlap (IoU) > NMS_OVERLAP_THRESHOLD

    This correctly deduplicates the same impulse found by
    windows of size 55, 89, AND 144 — they all cover the same
    price action and should produce only one winner (lowest dtw_score).
    """
    results = sorted(results, key=lambda r: r.dtw_score)
    kept: list[PatternResult] = []

    for candidate in results:
        suppressed = False
        for winner in kept:
            # Cross-window NMS: same pattern type, significant temporal overlap
            if winner.pattern != candidate.pattern:
                continue
            if _overlap_fraction(candidate, winner) > NMS_OVERLAP_THRESHOLD:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window_sizes: list[int] | None = None,
    verbose: bool = False,
) -> tuple[list[PatternResult], float]:
    """
    Run the full DTW wave detection pipeline.

    Args:
        high, low, close: numpy arrays of raw OHLCV price series
        window_sizes:     override default WINDOW_SIZES from config
        verbose:          print progress info

    Returns:
        (results, elapsed_seconds)
          results — list of PatternResult (after NMS)
          elapsed_seconds — wall-clock time
    """
    if window_sizes is None:
        window_sizes = WINDOW_SIZES

    templates = get_all_templates()
    raw_results: list[PatternResult] = []

    t0 = time.perf_counter()

    for w_size in window_sizes:
        if len(close) < w_size:
            continue

        if verbose:
            print(f"  [pipeline] window_size={w_size} …", flush=True)

        for start, norm_window, h_slice, l_slice in sliding_windows(high, low, close, w_size):
            end = start + w_size - 1

            matches = find_best_match(norm_window, h_slice, l_slice, templates, window_size=w_size)

            for match in matches:
                pattern  = match["pattern"]
                score    = match["dtw_score"]
                pivots   = match["pivots"]

                passed, reason = validate(pattern, pivots)

                raw_results.append(
                    PatternResult(
                        start_index=start,
                        end_index=end,
                        pattern=pattern,
                        dtw_score=score,
                        passed_validation=passed,
                        validation_reason=reason,
                        pivots=pivots,
                    )
                )

    # Apply NMS only to validated (passing) results per pattern type
    # Keep ALL failing results intact for visualization
    passing  = [r for r in raw_results if r.passed_validation]
    failing  = [r for r in raw_results if not r.passed_validation]

    passing_nms = _apply_nms(passing)
    # Also de-duplicate failing results (still useful for debug viz)
    failing_nms = _apply_nms(failing)

    final = sorted(passing_nms + failing_nms, key=lambda r: r.start_index)
    elapsed = time.perf_counter() - t0

    if verbose:
        print(
            f"  [pipeline] Done. Found {len(passing_nms)} valid + "
            f"{len(failing_nms)} failed patterns in {elapsed:.3f}s"
        )

    return final, elapsed
