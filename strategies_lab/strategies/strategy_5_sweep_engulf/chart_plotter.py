"""
chart_plotter.py — Модуль визуализации сделок стратегии 5 в эталонном качестве TradingView (Production Bot).

Особенности:
- Свечи чистые, БЕЗ фиолетовых рамок и значков на свече поглощения (как просил пользователь).
- Точная привязка точек волн Эллиотта (W0, W1, W2, W3, W4, W5) по точным индексам wave_indices из elliott_detector.
- Отрисовка линий дивергенции W3 vs W5 на графике цены и на панели Wilder RSI.
- Отрисовка W0 Origin Divergence (аквамариновая линия W0-DIV), если она есть.
- Линии уровней сделки:
  * 50% Limit Entry (золотая штрихпунктирная линия)
  * Stop-Loss (красная штриховая линия строго по кончику тени)
  * Take-Profit (зеленая сплошная линия строго 1:3 Net R)
- Метки входа и выхода (TP / SL / Timeout).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as mticker

from strategies_lab.strategies.strategy_5_sweep_engulf.config import CHARTS_DIR
from strategies_lab.strategies.strategy_5_sweep_engulf.patterns import SweepEngulfPattern
from strategies_lab.strategies.base_strategy import TradeSignal, TradeResult
from signal_scanner import calculate_rsi_wilder


def plot_sweep_engulf_trade(
    signal: TradeSignal,
    result: TradeResult,
    pattern: SweepEngulfPattern,
    full_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    lookback_before_w0: int = 20,
    lookahead_after_exit: int = 12,
) -> Path:
    """
    Строит график сделки профессионального уровня (TradingView dark theme).
    """
    w0_bar = signal.w0_bar
    w5_bar = signal.w5_bar
    direction = signal.direction
    sym = signal.symbol
    tf = signal.interval

    # Начало окна — чуть раньше старта импульса (W0 или предваряющей дивергенции)
    orig_b = signal.details.get("origin_div_bar") if signal.details else None
    earliest_bar = min(w0_bar, orig_b) if orig_b is not None else w0_bar
    start_bar = max(0, earliest_bar - lookback_before_w0)

    # Конец окна — после выхода из сделки
    exit_bar_approx = w5_bar + (pattern.s_rel_idx if pattern.s_rel_idx >= 0 else 0) + 1 + result.bars_held
    end_bar = min(len(full_df), exit_bar_approx + lookahead_after_exit + 5)
    df_window = full_df.iloc[start_bar:end_bar].copy().reset_index(drop=True)
    n_bars = len(df_window)

    if n_bars < 5:
        return Path("")

    # Цветовая палитра TradingView Dark
    bg_dark = "#131722"
    panel_dark = "#1E222D"
    grid_clr = "#2A2E39"
    text_clr = "#D1D4DC"
    bull_clr = "#089981"
    bear_clr = "#F23645"
    limit_clr = "#FFD700"     # Золотой для 50% лимита
    sl_clr = "#FF5252"        # Красный для стопа
    tp_clr = "#00E676"        # Ярко-зеленый для 1:3 тейка
    wave_clr = "#00E5FF" if direction == "SHORT" else "#FF4081"
    div_clr = "#FFD700"
    w0_div_clr = "#00E5FF"

    fig, (ax_main, ax_rsi) = plt.subplots(
        2, 1, figsize=(15, 8.5), dpi=110,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
        facecolor=bg_dark,
    )

    for ax in (ax_main, ax_rsi):
        ax.set_facecolor(bg_dark)
        ax.grid(True, color=grid_clr, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(colors=text_clr, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(grid_clr)

    # ── 1. Свечи (чистые, без мешающих рамок) ──
    opens = df_window['open'].values
    highs = df_window['high'].values
    lows = df_window['low'].values
    closes = df_window['close'].values

    body_width = 0.65
    wick_width = 1.2

    for i in range(n_bars):
        c_open, c_high, c_low, c_close = opens[i], highs[i], lows[i], closes[i]
        c_color = bull_clr if c_close >= c_open else bear_clr

        # Тень
        ax_main.plot([i, i], [c_low, c_high], color=c_color, linewidth=wick_width, zorder=2)
        # Тело
        bottom = min(c_open, c_close)
        height = abs(c_close - c_open)
        if height == 0:
            height = (c_high - c_low) * 0.03
        rect = patches.Rectangle(
            (i - body_width / 2.0, bottom),
            body_width, height,
            facecolor=c_color, edgecolor=c_color, linewidth=0.8, zorder=3,
        )
        ax_main.add_patch(rect)

    # ── 2. Точная отрисовка волн Эллиотта по wave_indices ──
    wi = signal.wave_indices
    wp = signal.wave_points
    wave_pts = []

    for w_name in ["W0", "W1", "W2", "W3", "W4", "W5"]:
        if wi and w_name in wi and wp and w_name in wp:
            abs_b = wi[w_name]
            rel_b = abs_b - start_bar
            if 0 <= rel_b < n_bars:
                wave_pts.append((rel_b, wp[w_name], w_name))

    if len(wave_pts) >= 2:
        xs = [p[0] for p in wave_pts]
        ys = [p[1] for p in wave_pts]
        ax_main.plot(xs, ys, color=wave_clr, linewidth=2.2, linestyle="-", alpha=0.85, zorder=6)

        price_span = max(highs) - min(lows)
        for rel_b, py, w_label in wave_pts:
            ax_main.scatter(rel_b, py, color=wave_clr, s=55, zorder=7, edgecolors="#FFFFFF", linewidth=1.0)
            is_peak = (direction == "SHORT" and w_label in ("W1", "W3", "W5")) or (direction == "LONG" and w_label in ("W0", "W2", "W4"))
            offset_y = price_span * 0.035 * (1 if is_peak else -1)
            ax_main.text(
                rel_b, py + offset_y, w_label,
                color="#FFFFFF", fontsize=10, fontweight="bold",
                ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="circle,pad=0.22", facecolor="#1A1A2E", edgecolor=wave_clr, linewidth=1.2),
            )

    # ── 3. Расчёт локальных X-координат свечи S, входа и выхода ──
    s_abs = signal.w5_bar if pattern.s_rel_idx == -1 else signal.w5_bar + 1 + pattern.s_rel_idx
    s_rel = s_abs - start_bar

    # Поиск точного бара входа по entry_time
    entry_rel = -1
    for idx_w in range(n_bars):
        if result.entry_time and full_df.index[start_bar + idx_w] == result.entry_time:
            entry_rel = idx_w
            break
    if entry_rel == -1:
        entry_rel = s_rel + 1

    # Поиск точного бара выхода по exit_time
    exit_rel = -1
    for idx_w in range(n_bars):
        if result.exit_time and full_df.index[start_bar + idx_w] == result.exit_time:
            exit_rel = idx_w
            break
    if exit_rel == -1:
        exit_rel = min(n_bars - 1, entry_rel + max(0, result.bars_held - 1))

    # ── 4. Уровни сделки: 50% Limit, SL, TP 1:3 ──
    entry_p = result.entry_price
    sl_p = result.initial_sl
    tp_p = result.entry_price - (result.initial_sl - result.entry_price) * 3.0 if direction == "SHORT" else result.entry_price + (result.entry_price - result.initial_sl) * 3.0

    line_start = max(0, s_rel)
    line_end = min(n_bars - 1, exit_rel + 4)

    # 50% Limit Entry
    ax_main.hlines(entry_p, line_start, line_end, colors=limit_clr, linestyles="-.", linewidth=1.6, zorder=5, label=f"Вход (50% S): ${entry_p:,.2f}")
    # Stop-Loss (строго конец тени)
    ax_main.hlines(sl_p, line_start, line_end, colors=sl_clr, linestyles="--", linewidth=1.6, zorder=5, label=f"Stop-Loss: ${sl_p:,.2f}")
    # Take-Profit 1:3 Net
    ax_main.hlines(tp_p, line_start, line_end, colors=tp_clr, linestyles="-", linewidth=1.8, zorder=5, label=f"Take-Profit 1:3 (Net): ${tp_p:,.2f}")

    # Метка исполнения лимита
    if 0 <= entry_rel < n_bars:
        ax_main.scatter(entry_rel, entry_p, color=limit_clr, s=95, marker="D", zorder=9, edgecolors="#000000", linewidth=1.2)
        offset_dir = 1 if direction == "LONG" else -1
        ax_main.annotate(
            "ВХОД 50%", xy=(entry_rel, entry_p),
            xytext=(entry_rel, entry_p + offset_dir * (highs.max() - lows.min()) * 0.04),
            arrowprops=dict(arrowstyle="->", color=limit_clr, lw=1.2),
            color=limit_clr, fontsize=8.5, fontweight="bold", ha="center", zorder=10,
        )

    # Метка выхода
    res_color = tp_clr if result.pnl_r > 0 else sl_clr
    res_badge = f"{result.exit_reason.upper()} ({result.pnl_r:+0.2f} R / ${result.net_pnl_usd:+.2f})"
    if 0 <= exit_rel < n_bars:
        ax_main.scatter(exit_rel, result.exit_price, color=res_color, s=110, marker="X", zorder=9, edgecolors="#FFFFFF")
        offset_exit = -0.05 if direction == "SHORT" and result.exit_reason == "tp" else 0.05
        ax_main.annotate(
            res_badge, xy=(exit_rel, result.exit_price),
            xytext=(exit_rel, result.exit_price + offset_exit * (highs.max() - lows.min())),
            arrowprops=dict(arrowstyle="->", color=res_color, lw=1.2),
            color="#FFFFFF", fontsize=9, fontweight="bold", ha="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=res_color, alpha=0.9, edgecolor="#FFFFFF"),
        )

    # Линия дивергенции W3 vs W5 на цене
    w3_abs = signal.w3_bar
    w5_abs = signal.w5_bar
    w3_x = w3_abs - start_bar
    w5_x = w5_abs - start_bar
    if 0 <= w3_x < n_bars and 0 <= w5_x < n_bars and wp and "W3" in wp and "W5" in wp:
        p_w3 = wp["W3"]
        p_w5 = wp["W5"]
        ax_main.plot([w3_x, w5_x], [p_w3, p_w5], color=div_clr, linewidth=1.8, linestyle=":", zorder=7)
        mid_x = (w3_x + w5_x) / 2.0
        mid_y = (p_w3 + p_w5) / 2.0
        offset_d = (highs.max() - lows.min()) * 0.04 * (1 if direction == "SHORT" else -1)
        ax_main.annotate(
            "DIV", xy=(mid_x, mid_y), xytext=(mid_x, mid_y + offset_d),
            fontsize=8.5, fontweight="bold", color=div_clr, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1A1A2E", edgecolor=div_clr, alpha=0.85, linewidth=1.0),
            zorder=10,
        )

    # Линия W0 Origin Divergence на цене
    if signal.details and signal.details.get("origin_div") and orig_b is not None:
        orig_x = orig_b - start_bar
        w0_x = w0_bar - start_bar
        if 0 <= orig_x < n_bars and 0 <= w0_x < n_bars and wp and "W0" in wp:
            p_orig = signal.details.get("origin_div_price", lows[orig_x] if direction == "LONG" else highs[orig_x])
            p_w0 = wp["W0"]
            ax_main.plot([orig_x, w0_x], [p_orig, p_w0], color=w0_div_clr, linewidth=1.6, linestyle="--", zorder=7)
            mid_ox = (orig_x + w0_x) / 2.0
            mid_oy = (p_orig + p_w0) / 2.0
            ax_main.annotate(
                "W0-DIV", xy=(mid_ox, mid_oy),
                xytext=(mid_ox, mid_oy + (1 if direction == "LONG" else -1) * (highs.max() - lows.min()) * 0.04),
                fontsize=8, fontweight="bold", color=w0_div_clr, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1A1A2E", edgecolor=w0_div_clr, alpha=0.85, linewidth=1.0),
                zorder=10,
            )

    ax_main.set_xlim(-1, n_bars + 1)
    p_span = max(highs) - min(lows)
    ax_main.set_ylim(min(lows) - p_span * 0.08, max(highs) + p_span * 0.10)
    ax_main.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax_main.legend(loc="upper left", facecolor=panel_dark, edgecolor=grid_clr, labelcolor=text_clr, fontsize=8.5)

    status_icon = "[WIN 1:3]" if result.pnl_r > 0 else "[STOP-LOSS]"
    title_text = (
        f"{status_icon} | {sym} {tf} | {direction} | Исход: {result.exit_reason.upper()} | "
        f"Net PnL: {result.pnl_r:+0.2f} R (${result.net_pnl_usd:+.2f}) | Удержание: {result.bars_held} свечей"
    )
    ax_main.set_title(title_text, color="#FFFFFF", fontsize=11.5, fontweight="bold", pad=10)

    # ── 5. Панель RSI ──
    rsi_full = calculate_rsi_wilder(full_df['close'].values.astype(np.float64), period=7 if tf in ("5m", "15m") else 14)
    rsi_window = rsi_full[start_bar:end_bar]

    ax_rsi.plot(range(len(rsi_window)), rsi_window, color="#7B61FF", linewidth=1.6, label="Wilder RSI")
    ax_rsi.axhline(70, color="#FF6B6B", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_rsi.axhline(30, color="#51CF66", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_rsi.axhline(50, color=grid_clr, linestyle=":", linewidth=0.8, alpha=0.6)
    ax_rsi.fill_between(range(len(rsi_window)), 30, 70, color="#7B61FF", alpha=0.06)

    # Линия дивергенции W3 vs W5 на RSI
    if 0 <= w3_x < len(rsi_window) and 0 <= w5_x < len(rsi_window):
        r_w3 = rsi_window[w3_x]
        r_w5 = rsi_window[w5_x]
        ax_rsi.plot([w3_x, w5_x], [r_w3, r_w5], color=div_clr, linewidth=2.0, linestyle="-", marker="o", markersize=4, zorder=8)
        mid_rx = (w3_x + w5_x) / 2.0
        mid_ry = (r_w3 + r_w5) / 2.0
        ax_rsi.annotate(
            "DIV", xy=(mid_rx, mid_ry), xytext=(mid_rx, mid_ry + 8),
            fontsize=8, fontweight="bold", color=div_clr, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1A1A2E", edgecolor=div_clr, alpha=0.85, linewidth=1),
            zorder=10,
        )

    # Линия W0-DIV на RSI
    if signal.details and signal.details.get("origin_div") and orig_b is not None:
        orig_rx = orig_b - start_bar
        w0_rx = w0_bar - start_bar
        r_orig = signal.details.get("origin_div_rsi")
        if 0 <= orig_rx < len(rsi_window) and 0 <= w0_rx < len(rsi_window) and r_orig is not None:
            r_w0 = rsi_window[w0_rx]
            ax_rsi.plot([orig_rx, w0_rx], [r_orig, r_w0], color=w0_div_clr, linewidth=1.8, linestyle="--", marker="o", markersize=3.5, zorder=8)
            mid_orx = (orig_rx + w0_rx) / 2.0
            mid_ory = (r_orig + r_w0) / 2.0
            ax_rsi.annotate(
                "W0-DIV", xy=(mid_orx, mid_ory), xytext=(mid_orx, mid_ory - 8),
                fontsize=8, fontweight="bold", color=w0_div_clr, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1A1A2E", edgecolor=w0_div_clr, alpha=0.85, linewidth=1),
                zorder=10,
            )

    ax_rsi.set_xlim(-1, n_bars + 1)
    ax_rsi.set_ylim(10, 90)
    ax_rsi.set_ylabel("RSI", color=text_clr, fontsize=9)

    plt.tight_layout()

    out_file = save_path or (CHARTS_DIR / f"{sym}_{tf}_{direction}_{result.signal_id}_{result.exit_reason}.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, facecolor=bg_dark, edgecolor="none", dpi=110)
    plt.close(fig)
    return out_file
