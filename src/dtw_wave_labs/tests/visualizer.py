"""
visualizer.py — Generate an interactive Plotly HTML report proving the DTW
Wave Labs pipeline works on REAL market data.

Prerequisites:
    1. Run download_fixture.py once to fetch btc_1000.csv
    2. pip install plotly

Usage (from project root):
    .venv\\Scripts\\python.exe src/dtw_wave_labs/tests/visualizer.py

Output:
    src/dtw_wave_labs/tests/dtw_report.html
    (also prints elapsed time to stdout)
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dtw_wave_labs.pipeline import run_pipeline, PatternResult

# ── Constants ─────────────────────────────────────────────────────────────────
FIXTURE_PATH = os.path.join(_HERE, "fixtures", "btc_1000.csv")
OUTPUT_HTML  = os.path.join(_HERE, "dtw_report.html")

# Color scheme
COLOR_VALID   = "rgba(0, 220, 100, 0.90)"    # green — passed validator
COLOR_INVALID = "rgba(255, 60, 60, 0.70)"    # red   — DTW ok but validator failed
DASH_VALID    = "solid"
DASH_INVALID  = "dot"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_fixture() -> pd.DataFrame:
    if not os.path.exists(FIXTURE_PATH):
        print(
            f"[ERROR] Fixture not found: {FIXTURE_PATH}\n"
            "Run download_fixture.py first:\n"
            "  .venv\\Scripts\\python.exe "
            "src/dtw_wave_labs/tests/download_fixture.py"
        )
        sys.exit(1)

    df = pd.read_csv(FIXTURE_PATH, parse_dates=["datetime"])
    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] CSV missing columns: {missing}")
        sys.exit(1)

    print(f"Loaded {len(df)} candles from {os.path.basename(FIXTURE_PATH)}")
    return df


# ── Visualization helpers ─────────────────────────────────────────────────────

def _pattern_label(r: PatternResult) -> str:
    status = "✓" if r.passed_validation else "✗"
    return f"{status} {r.pattern} | DTW={r.dtw_score:.4f} | [{r.start_index}–{r.end_index}]"


def _build_pattern_trace(
    r: PatternResult,
    df: pd.DataFrame,
) -> go.Scatter:
    """
    Draw a line connecting all pivot points of the pattern on the real chart.
    X axis: datetime values at pivot candle indices.
    Y axis: pivot prices (raw High/Low extracted by pivot_extractor).
    """
    n_pivots = len(r.pivots)
    if n_pivots < 2:
        return None

    # Absolute candle indices of pivots — roughly evenly spaced in window
    window_len = r.end_index - r.start_index + 1
    frac_positions = [i / (n_pivots - 1) for i in range(n_pivots)]
    abs_indices = [
        min(r.start_index + int(f * window_len), r.end_index)
        for f in frac_positions
    ]

    x_vals = [df["datetime"].iloc[idx] for idx in abs_indices]
    y_vals = [r.pivots[k] for k in sorted(r.pivots.keys())]

    color = COLOR_VALID if r.passed_validation else COLOR_INVALID
    dash  = DASH_VALID  if r.passed_validation else DASH_INVALID

    label = _pattern_label(r)

    return go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines+markers",
        name=label,
        line=dict(color=color, width=2, dash=dash),
        marker=dict(size=7, color=color, symbol="circle"),
        hovertemplate=(
            f"<b>{r.pattern}</b><br>"
            f"DTW score: {r.dtw_score:.4f}<br>"
            f"Validated: {r.passed_validation}<br>"
            "%{y:.2f}<extra></extra>"
        ),
        legendgroup=r.pattern,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DTW Wave Labs — Visualizer")
    print("=" * 60)

    df = load_fixture()

    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    print("Running pipeline …")
    t0 = time.perf_counter()
    results, elapsed = run_pipeline(high, low, close, verbose=True)
    total_time = time.perf_counter() - t0

    n_valid   = sum(1 for r in results if r.passed_validation)
    n_invalid = sum(1 for r in results if not r.passed_validation)

    print(f"\nResults: {n_valid} valid | {n_invalid} rejected | {total_time:.3f}s total")
    print(
        "⚡ Performance:",
        "OK ✓" if total_time <= 3.0 else f"SLOW ✗ ({total_time:.1f}s > 3s target)"
    )

    # ── Build Plotly figure ────────────────────────────────────────────────
    fig = go.Figure()

    # Candlestick base chart
    fig.add_trace(
        go.Candlestick(
            x=df["datetime"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTCUSDT 1H",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
            showlegend=True,
        )
    )

    # Pattern overlays
    shown_labels = set()
    traces_added = 0
    for r in results:
        trace = _build_pattern_trace(r, df)
        if trace is None:
            continue
        # Avoid 100s of legend entries for the same pattern type
        leg_key = (r.pattern, r.passed_validation)
        if leg_key in shown_labels:
            trace.showlegend = False
        else:
            shown_labels.add(leg_key)
        fig.add_trace(trace)
        traces_added += 1

    # Layout
    fig.update_layout(
        title=dict(
            text=(
                f"<b>DTW Elliott Wave Labs — BTCUSDT 1H</b><br>"
                f"<sup>{n_valid} valid patterns (green) | "
                f"{n_invalid} rejected (red dashed) | "
                f"Pipeline: {total_time:.2f}s</sup>"
            ),
            font=dict(size=18),
            x=0.5,
        ),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis=dict(title="Price (USDT)"),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            bgcolor="rgba(20,20,30,0.85)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=11, color="white"),
        ),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="Inter, Roboto, sans-serif"),
        margin=dict(r=320, t=100, b=60),
        hovermode="x unified",
        height=750,
    )

    # Summary annotation
    fig.add_annotation(
        text=(
            f"<b>Summary</b><br>"
            f"Candles: {len(df)}<br>"
            f"Windows: 21,55,89,144,233<br>"
            f"Valid: {n_valid} | Rejected: {n_invalid}<br>"
            f"Time: {total_time:.3f}s"
        ),
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        bgcolor="rgba(20,20,30,0.80)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        font=dict(size=11, color="#8b949e"),
        align="left",
    )

    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"\n✅ Report saved → {OUTPUT_HTML}")
    print("   Open it in your browser to inspect the patterns.")


if __name__ == "__main__":
    main()
