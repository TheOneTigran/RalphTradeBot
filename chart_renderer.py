"""
chart_renderer.py — Рендер свечного графика + RSI в PNG.

Стиль TradingView (тёмная тема). Используется для генерации
скриншотов, которые отправляются в Vision AI модель.
"""
from __future__ import annotations

import io
import gc
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless, без GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BG_DARK, GRID_CLR, TEXT_CLR,
    CANDLE_UP, CANDLE_DN, RSI_CLR,
    RSI_OB_CLR, RSI_OS_CLR, SIGNAL_LINE_CLR,
    PIVOT_HIGH_CLR, PIVOT_LOW_CLR,
    CHART_WIDTH_PX, CHART_HEIGHT_PX, CHART_DPI,
)


def _smart_price_fmt(x: float, _pos=None) -> str:
    """Формат цены для Y-оси (как в TradingView)."""
    abs_x = abs(x)
    if abs_x >= 1000:
        return f"{x:,.2f}"
    elif abs_x >= 1:
        return f"{x:,.4f}"
    elif abs_x >= 0.01:
        return f"{x:.6f}"
    elif abs_x >= 0.0001:
        return f"{x:.8f}"
    else:
        return f"{x:.10f}"


def _hex_rgba(hex_color: str, alpha: float) -> tuple:
    """#RRGGBB → (r, g, b, a) normalized 0-1."""
    h = hex_color.lstrip("#")
    r, g, b = [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return (r, g, b, alpha)


# Цвета для волн Эллиотта
WAVE_LINE_BULLISH = '#00E5FF'    # Яркий голубой (импульс вверх)
WAVE_LINE_BEARISH = '#FF4081'    # Яркий розовый/малиновый (импульс вниз)
WAVE_LABEL_BG = '#1A1A2E'        # Фон кружка метки
WAVE_FONT_SIZE = 14


def _find_bar_for_price(
    highs: np.ndarray,
    lows: np.ndarray,
    target_price: float,
    search_start: int,
    search_end: int,
    is_extreme_high: bool,
) -> int:
    """
    Находит бар, ближайший к указанной цене (по high или low).
    
    Args:
        highs, lows: Массивы OHLC
        target_price: Целевая цена (от Vision AI)
        search_start, search_end: Диапазон поиска (индексы в окне)
        is_extreme_high: True = ищем экстремум high, False = low
        
    Returns:
        Индекс бара в окне
    """
    best_idx = search_start
    best_dist = float('inf')
    
    for i in range(max(0, search_start), min(len(highs), search_end)):
        if is_extreme_high:
            dist = abs(highs[i] - target_price)
        else:
            dist = abs(lows[i] - target_price)
        
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    
    return best_idx


def _draw_elliott_waves(
    ax, df_window, wave_points: dict, wave_direction: str,
    x: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    start_idx: int, n_bars: int,
    wave_indices: Optional[dict] = None,
    algo_score: Optional[float] = None,
):
    """
    Рисует 5-волновой импульс Эллиотта на графике цены.
    
    Линии W0→W1→W2→W3→W4→W5 с крупными номерами волн.
    
    Args:
        ax: matplotlib Axes (ценовой график)
        df_window: DataFrame окна
        wave_points: {"W0": price, "W1": price, ..., "W5": price}
        wave_direction: "bullish" или "bearish"
        x: массив X-координат
        highs, lows: массивы high/low цен окна
        start_idx: начальный индекс окна в общем DataFrame
        n_bars: кол-во баров в окне
        wave_indices: точные индексы баров {"W0": bar_idx, ..., "W5": bar_idx} (если известны)
        algo_score: балл алгоритмической оценки (для отображения)
    """
    # Проверяем что есть хотя бы 3 точки
    labels = ['W0', 'W1', 'W2', 'W3', 'W4', 'W5']
    valid_points = {}
    
    for label in labels:
        val = wave_points.get(label)
        if val is not None:
            try:
                valid_points[label] = float(val)
            except (ValueError, TypeError):
                continue
    
    if len(valid_points) < 3:
        return  # Слишком мало точек для отрисовки
    
    # Определяем цвет линий
    is_bullish = wave_direction == "bullish"
    line_color = WAVE_LINE_BULLISH if is_bullish else WAVE_LINE_BEARISH
    
    # Маппим цены на позиции баров
    points_xy = []
    ordered_labels = [l for l in labels if l in valid_points]
    
    for i, label in enumerate(ordered_labels):
        price = valid_points[label]
        
        # Если передан точный индекс бара — используем его напрямую!
        if wave_indices is not None and label in wave_indices:
            abs_bar = wave_indices[label]
            bar_idx = abs_bar - start_idx
            if 0 <= bar_idx < n_bars:
                points_xy.append((bar_idx, price, label))
                continue
        
        # Fallback: поиск по цене
        if i == 0:
            s_start = 0
        else:
            s_start = max(0, points_xy[-1][0] - 2) if points_xy else 0
        
        s_end = n_bars
        wave_num = int(label[1]) if label[1].isdigit() else 0
        
        if is_bullish:
            is_high = wave_num % 2 == 1
        else:
            is_high = wave_num % 2 == 0
        
        bar_idx = _find_bar_for_price(highs, lows, price, s_start, s_end, is_high)
        if is_high:
            y_price = highs[bar_idx]
        else:
            y_price = lows[bar_idx]
        
        points_xy.append((bar_idx, y_price, label))
    
    if len(points_xy) < 2:
        return
    
    # ── Рисуем линии между точками ──
    for i in range(len(points_xy) - 1):
        x1, y1, _ = points_xy[i]
        x2, y2, _ = points_xy[i + 1]
        
        ax.plot(
            [x1, x2], [y1, y2],
            color=line_color, linewidth=2.5, alpha=0.85,
            linestyle='-', zorder=10,
        )
    
    # ── Рисуем номера волн в кружках ──
    price_range = highs.max() - lows.min()
    offset = price_range * 0.025  # Отступ метки от точки
    
    for bar_idx, y_price, label in points_xy:
        wave_num = label  # "W0", "W1", ...
        display_num = label[1]  # "0", "1", ...
        num_val = int(display_num) if display_num.isdigit() else 0
        
        if is_bullish:
            label_above = num_val % 2 == 1
        else:
            label_above = num_val % 2 == 0
        
        y_label = y_price + offset if label_above else y_price - offset
        
        # Кружок с номером
        ax.annotate(
            display_num,
            xy=(bar_idx, y_price),
            xytext=(bar_idx, y_label),
            fontsize=WAVE_FONT_SIZE,
            fontweight='bold',
            color=line_color,
            ha='center', va='center',
            bbox=dict(
                boxstyle='circle,pad=0.3',
                facecolor=WAVE_LABEL_BG,
                edgecolor=line_color,
                linewidth=1.5,
                alpha=0.9,
            ),
            arrowprops=dict(
                arrowstyle='-',
                color=line_color,
                alpha=0.5,
                linewidth=0.8,
            ),
            zorder=15,
        )
    
    # Добавляем метку "Elliott Impulse" в углу
    direction_text = "BULLISH" if is_bullish else "BEARISH"
    score_text = f" (Algo Score: {algo_score:.0f})" if algo_score is not None else ""
    ax.text(
        0.02, 0.96, f"Elliott {direction_text} Impulse{score_text}",
        transform=ax.transAxes,
        fontsize=10, fontweight='bold',
        color=line_color, alpha=0.9,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=WAVE_LABEL_BG, 
                  edgecolor=line_color, alpha=0.8),
        zorder=20,
    )


def render_signal_chart(
    df: pd.DataFrame,
    rsi_values: np.ndarray,
    signal_idx: int,
    direction: str,
    symbol: str,
    interval: str,
    rsi_ob: float = 70.0,
    rsi_os: float = 30.0,
    swept_price: Optional[float] = None,
    pivot_levels: Optional[list] = None,
    lookback: int = 200,
    wave_points: Optional[dict] = None,
    wave_direction: Optional[str] = None,
    wave_indices: Optional[dict] = None,
    algo_score: Optional[float] = None,
    # Параметры дивергенции (для отрисовки линий на графике)
    swept_bar_idx: Optional[int] = None,     # Бар-индекс pivot (в общем DataFrame)
    swept_rsi: Optional[float] = None,       # RSI на pivot
    signal_rsi: Optional[float] = None,      # RSI на сигнале
    # Параметры предваряющей дивергенции W0 (W0 Origin Divergence)
    origin_div: bool = False,
    orig_prev_bar: Optional[int] = None,
    orig_prev_price: Optional[float] = None,
    orig_prev_rsi: Optional[float] = None,
) -> bytes:
    """
    Рендерит свечной график с RSI для отправки в Vision AI.
    
    Args:
        df: DataFrame с колонками ['open', 'high', 'low', 'close'] (и опционально 'volume')
        rsi_values: Массив RSI значений, совпадающий по длине с df
        signal_idx: Индекс бара сигнала в df
        direction: "LONG" или "SHORT"
        symbol: Тикер (для заголовка)
        interval: Таймфрейм (для заголовка)
        rsi_ob: Уровень перекупленности RSI
        rsi_os: Уровень перепроданности RSI
        swept_price: Уровень ликвидности, который был снят
        pivot_levels: Список (price, type) где type = "high" или "low"
        lookback: Сколько свечей до сигнала показывать
        
    Returns:
        PNG-изображение в bytes
    """
    # Определяем окно отображения
    start_idx = max(0, signal_idx - lookback)
    # Показываем ещё ~10 свечей после сигнала для контекста
    end_idx = min(len(df), signal_idx + 10)
    
    df_window = df.iloc[start_idx:end_idx].copy()
    rsi_window = rsi_values[start_idx:end_idx]
    
    n_bars = len(df_window)
    x = np.arange(n_bars)
    
    opens = df_window['open'].values
    highs = df_window['high'].values
    lows = df_window['low'].values
    closes = df_window['close'].values
    
    has_volume = 'volume' in df_window.columns
    
    # ── Создание фигуры ──
    if has_volume:
        fig, (ax_price, ax_vol, ax_rsi) = plt.subplots(
            3, 1, figsize=(CHART_WIDTH_PX / CHART_DPI, CHART_HEIGHT_PX / CHART_DPI),
            dpi=CHART_DPI, gridspec_kw={'height_ratios': [5, 1, 2]},
            facecolor=BG_DARK
        )
    else:
        fig, (ax_price, ax_rsi) = plt.subplots(
            2, 1, figsize=(CHART_WIDTH_PX / CHART_DPI, CHART_HEIGHT_PX / CHART_DPI),
            dpi=CHART_DPI, gridspec_kw={'height_ratios': [5, 2]},
            facecolor=BG_DARK
        )
        ax_vol = None
    
    fig.subplots_adjust(hspace=0.05, left=0.06, right=0.94, top=0.93, bottom=0.06)
    
    # ── Рисуем свечи ──
    ax_price.set_facecolor(BG_DARK)
    
    for i in range(n_bars):
        color = CANDLE_UP if closes[i] >= opens[i] else CANDLE_DN
        
        # Тень (фитиль)
        ax_price.plot(
            [x[i], x[i]], [lows[i], highs[i]],
            color=color, linewidth=0.8, solid_capstyle='round'
        )
        
        # Тело свечи
        body_bottom = min(opens[i], closes[i])
        body_height = abs(closes[i] - opens[i])
        if body_height < (highs[i] - lows[i]) * 0.01:
            body_height = (highs[i] - lows[i]) * 0.01  # минимальная видимость
        
        rect = plt.Rectangle(
            (x[i] - 0.35, body_bottom), 0.7, body_height,
            facecolor=color, edgecolor=color, linewidth=0.5
        )
        ax_price.add_patch(rect)
    
    # ── Вертикальная линия сигнала ──
    signal_x = signal_idx - start_idx
    if 0 <= signal_x < n_bars:
        ax_price.axvline(
            x=signal_x, color=SIGNAL_LINE_CLR, linewidth=2,
            linestyle='--', alpha=0.9, zorder=10
        )
        
        # Метка сигнала
        sig_label = f"{'▼ SHORT' if direction == 'SHORT' else '▲ LONG'}"
        sig_y = highs.max() if direction == "SHORT" else lows.min()
        ax_price.annotate(
            sig_label,
            xy=(signal_x, sig_y),
            fontsize=10, fontweight='bold',
            color=SIGNAL_LINE_CLR,
            ha='center', va='bottom' if direction == "SHORT" else 'top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_DARK, edgecolor=SIGNAL_LINE_CLR, alpha=0.9),
            zorder=11
        )
        
        # Линия на RSI тоже
        ax_rsi.axvline(x=signal_x, color=SIGNAL_LINE_CLR, linewidth=1.5, linestyle='--', alpha=0.6)
    
    # ── Swept уровень ликвидности ──
    if swept_price is not None:
        ax_price.axhline(
            y=swept_price, color=SIGNAL_LINE_CLR, linewidth=1,
            linestyle=':', alpha=0.5
        )
        ax_price.text(
            n_bars - 1, swept_price, f" Swept: {_smart_price_fmt(swept_price)}",
            fontsize=7, color=SIGNAL_LINE_CLR, va='center', alpha=0.7
        )
    
    # ── Pivot уровни ──
    if pivot_levels:
        for price, ptype in pivot_levels[:8]:  # максимум 8 уровней
            clr = PIVOT_HIGH_CLR if ptype == "high" else PIVOT_LOW_CLR
            ax_price.axhline(y=price, color=clr, linewidth=0.6, linestyle=':', alpha=0.35)
    
    # ── Настройка оси цены ──
    price_range = highs.max() - lows.min()
    ax_price.set_ylim(lows.min() - price_range * 0.03, highs.max() + price_range * 0.08)
    ax_price.set_xlim(-1, n_bars + 1)
    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(_smart_price_fmt))
    ax_price.grid(True, color=GRID_CLR, linewidth=0.3, alpha=0.5)
    ax_price.tick_params(colors=TEXT_CLR, labelsize=7)
    ax_price.set_xticklabels([])
    
    for spine in ax_price.spines.values():
        spine.set_color(GRID_CLR)
    
    # Заголовок
    ax_price.set_title(
        f"{symbol}  •  {interval}  •  Elliott Impulse Check  •  Signal: {direction}",
        fontsize=11, fontweight='bold', color=TEXT_CLR, pad=8
    )
    
    # ── Объём (если есть) ──
    if ax_vol is not None and has_volume:
        volumes = df_window['volume'].values
        vol_colors = [CANDLE_UP if closes[i] >= opens[i] else CANDLE_DN for i in range(n_bars)]
        ax_vol.bar(x, volumes, width=0.7, color=vol_colors, alpha=0.5)
        ax_vol.set_facecolor(BG_DARK)
        ax_vol.set_xlim(-1, n_bars + 1)
        ax_vol.set_xticklabels([])
        ax_vol.tick_params(colors=TEXT_CLR, labelsize=6)
        ax_vol.grid(True, color=GRID_CLR, linewidth=0.3, alpha=0.3)
        for spine in ax_vol.spines.values():
            spine.set_color(GRID_CLR)
        
        if 0 <= signal_x < n_bars:
            ax_vol.axvline(x=signal_x, color=SIGNAL_LINE_CLR, linewidth=1, linestyle='--', alpha=0.4)
    
    # ── RSI ──
    ax_rsi.set_facecolor(BG_DARK)
    ax_rsi.plot(x, rsi_window, color=RSI_CLR, linewidth=1.2, alpha=0.9)
    ax_rsi.axhline(y=rsi_ob, color=RSI_OB_CLR, linewidth=0.8, linestyle='--', alpha=0.6)
    ax_rsi.axhline(y=rsi_os, color=RSI_OS_CLR, linewidth=0.8, linestyle='--', alpha=0.6)
    ax_rsi.axhline(y=50, color=GRID_CLR, linewidth=0.5, linestyle='-', alpha=0.3)
    
    # Заливка зон перекупленности/перепроданности
    ax_rsi.fill_between(x, rsi_ob, 100, alpha=0.08, color=RSI_OB_CLR)
    ax_rsi.fill_between(x, 0, rsi_os, alpha=0.08, color=RSI_OS_CLR)
    
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_xlim(-1, n_bars + 1)
    ax_rsi.set_ylabel("RSI", fontsize=8, color=TEXT_CLR)
    ax_rsi.tick_params(colors=TEXT_CLR, labelsize=7)
    ax_rsi.grid(True, color=GRID_CLR, linewidth=0.3, alpha=0.3)
    
    # Метки OB/OS
    ax_rsi.text(n_bars, rsi_ob, f" {rsi_ob:.0f}", fontsize=6, color=RSI_OB_CLR, va='center')
    ax_rsi.text(n_bars, rsi_os, f" {rsi_os:.0f}", fontsize=6, color=RSI_OS_CLR, va='center')
    
    for spine in ax_rsi.spines.values():
        spine.set_color(GRID_CLR)
    
    # ── Отрисовка линии дивергенции (если есть данные) ──
    if swept_bar_idx is not None and swept_rsi is not None and signal_rsi is not None:
        # Позиция swept pivot в окне
        swept_x_in_window = swept_bar_idx - start_idx
        
        if 0 <= swept_x_in_window < n_bars and 0 <= signal_x < n_bars:
            # Цвет дивергенции
            div_color = '#FFEB3B'  # Яркий жёлтый
            
            # Ценовые экстремумы для линии дивергенции
            if direction == "LONG":
                # Bearish div: цена делает новый low, RSI выше → рисуем по lows
                price_1 = lows[swept_x_in_window]
                price_2 = lows[signal_x]
            else:
                # Bullish div: цена делает новый high, RSI ниже → рисуем по highs
                price_1 = highs[swept_x_in_window]
                price_2 = highs[signal_x]
            
            # ── Линия дивергенции на графике ЦЕНЫ ──
            ax_price.plot(
                [swept_x_in_window, signal_x], [price_1, price_2],
                color=div_color, linewidth=2.0, linestyle='-', alpha=0.85, zorder=8,
            )
            # Точки на концах
            ax_price.scatter(
                [swept_x_in_window, signal_x], [price_1, price_2],
                color=div_color, s=40, zorder=9, edgecolors='white', linewidth=0.5,
            )
            # Метка "DIV" на цене
            mid_x_price = (swept_x_in_window + signal_x) / 2
            mid_y_price = (price_1 + price_2) / 2
            price_range = highs.max() - lows.min()
            offset_dir = -1 if direction == "LONG" else 1
            ax_price.annotate(
                "DIV",
                xy=(mid_x_price, mid_y_price),
                xytext=(mid_x_price, mid_y_price + offset_dir * price_range * 0.04),
                fontsize=9, fontweight='bold', color=div_color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A1A2E', 
                          edgecolor=div_color, alpha=0.85, linewidth=1.5),
                arrowprops=dict(arrowstyle='-', color=div_color, alpha=0.5),
                zorder=12,
            )
            
            # ── Линия дивергенции на графике RSI ──
            ax_rsi.plot(
                [swept_x_in_window, signal_x], [swept_rsi, signal_rsi],
                color=div_color, linewidth=2.0, linestyle='-', alpha=0.85, zorder=8,
            )
            ax_rsi.scatter(
                [swept_x_in_window, signal_x], [swept_rsi, signal_rsi],
                color=div_color, s=30, zorder=9, edgecolors='white', linewidth=0.5,
            )
            # Метка "DIV" на RSI
            mid_x_rsi = (swept_x_in_window + signal_x) / 2
            mid_y_rsi = (swept_rsi + signal_rsi) / 2
            ax_rsi.annotate(
                "DIV",
                xy=(mid_x_rsi, mid_y_rsi),
                xytext=(mid_x_rsi, mid_y_rsi + 8),
                fontsize=8, fontweight='bold', color=div_color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#1A1A2E', 
                          edgecolor=div_color, alpha=0.85, linewidth=1),
                arrowprops=dict(arrowstyle='-', color=div_color, alpha=0.4),
                zorder=12,
            )
    
    # ── Линия предваряющей дивергенции W0 (W0 Origin Divergence) ──
    if origin_div and orig_prev_bar is not None and wave_indices and "W0" in wave_indices:
        w0_abs_bar = wave_indices["W0"]
        prev_x_win = orig_prev_bar - start_idx
        w0_x_win = w0_abs_bar - start_idx
        if 0 <= prev_x_win < n_bars and 0 <= w0_x_win < n_bars:
            p_prev = orig_prev_price if orig_prev_price is not None else (lows[prev_x_win] if direction == "LONG" else highs[prev_x_win])
            p_w0 = wave_points.get("W0", lows[w0_x_win] if direction == "LONG" else highs[w0_x_win]) if wave_points else (lows[w0_x_win] if direction == "LONG" else highs[w0_x_win])
            
            # W0-DIV на графике цены
            ax_price.plot(
                [prev_x_win, w0_x_win], [p_prev, p_w0],
                color='#00E5FF', linewidth=2.0, linestyle='--', alpha=0.9, zorder=8,
            )
            ax_price.scatter(
                [prev_x_win, w0_x_win], [p_prev, p_w0],
                color='#00E5FF', s=35, zorder=9, edgecolors='white', linewidth=0.5,
            )
            mid_pw0_x = (prev_x_win + w0_x_win) / 2
            mid_pw0_y = (p_prev + p_w0) / 2
            offset_w0 = 1 if direction == "LONG" else -1
            ax_price.annotate(
                "W0-DIV",
                xy=(mid_pw0_x, mid_pw0_y),
                xytext=(mid_pw0_x, mid_pw0_y + offset_w0 * (highs.max() - lows.min()) * 0.04),
                fontsize=8, fontweight='bold', color='#00E5FF',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#1A1A2E',
                          edgecolor='#00E5FF', alpha=0.85, linewidth=1.2),
                arrowprops=dict(arrowstyle='-', color='#00E5FF', alpha=0.5),
                zorder=12,
            )
            
            # W0-DIV на графике RSI
            if orig_prev_rsi is not None and 0 <= w0_abs_bar < len(rsi_values):
                r_w0 = rsi_values[w0_abs_bar]
                ax_rsi.plot(
                    [prev_x_win, w0_x_win], [orig_prev_rsi, r_w0],
                    color='#00E5FF', linewidth=1.8, linestyle='--', alpha=0.9, zorder=8,
                )
                ax_rsi.scatter(
                    [prev_x_win, w0_x_win], [orig_prev_rsi, r_w0],
                    color='#00E5FF', s=25, zorder=9, edgecolors='white', linewidth=0.5,
                )
                mid_rw0_x = (prev_x_win + w0_x_win) / 2
                mid_rw0_y = (orig_prev_rsi + r_w0) / 2
                ax_rsi.annotate(
                    "W0-DIV",
                    xy=(mid_rw0_x, mid_rw0_y),
                    xytext=(mid_rw0_x, mid_rw0_y + 7),
                    fontsize=7, fontweight='bold', color='#00E5FF',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1A1A2E',
                              edgecolor='#00E5FF', alpha=0.85, linewidth=1),
                    arrowprops=dict(arrowstyle='-', color='#00E5FF', alpha=0.4),
                    zorder=12,
                )

    # ── Отрисовка волн Эллиотта (если есть wave_points) ──
    if wave_points is not None:
        _draw_elliott_waves(
            ax_price, df_window, wave_points, wave_direction,
            x, highs, lows, start_idx, n_bars,
            wave_indices=wave_indices,
            algo_score=algo_score,
        )
    
    # ── Экспорт в PNG bytes ──
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=CHART_DPI, facecolor=BG_DARK, edgecolor='none')
    plt.close(fig)
    gc.collect()
    
    buf.seek(0)
    return buf.read()


def annotate_chart_with_analysis(
    chart_png: bytes,
    analysis: dict,
    signal_direction: str,
    signal_price: float,
    trade_result: str = "",
    trade_pnl: float = 0.0,
) -> bytes:
    """
    Добавляет панель с результатами разволновки Vision AI поверх графика.
    
    Создаёт составное изображение: оригинальный график + информационная панель
    справа с детальной разволновкой, score, нарушениями правил и т.д.
    
    Args:
        chart_png: Оригинальный PNG графика
        analysis: Dict от evaluate_elliott_impulse с полями:
            score, reason, wave_direction, waves_identified, rule_violations
        signal_direction: "LONG" или "SHORT"
        signal_price: Цена сигнала
        trade_result: "win" / "loss" / ""
        trade_pnl: P&L в процентах
        
    Returns:
        Составной PNG с аннотацией
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Загружаем оригинальный график
    chart_img = Image.open(io.BytesIO(chart_png))
    chart_w, chart_h = chart_img.size
    
    # Размеры панели
    panel_w = 380
    total_w = chart_w + panel_w
    
    # Создаём составное изображение
    composite = Image.new('RGB', (total_w, chart_h), color=(19, 23, 34))  # BG_DARK
    composite.paste(chart_img, (0, 0))
    
    draw = ImageDraw.Draw(composite)
    
    # Шрифт с полной поддержкой кириллицы (Windows: Arial / Segoe UI / Consolas)
    font_title = None
    for fname in ["arial.ttf", "segoeui.ttf", "consola.ttf", "calibri.ttf"]:
        try:
            font_title = ImageFont.truetype(fname, 16)
            font_body = ImageFont.truetype(fname, 12)
            font_small = ImageFont.truetype(fname, 10)
            break
        except (OSError, IOError):
            continue
    if font_title is None:
        font_title = ImageFont.load_default()
        font_body = font_title
        font_small = font_title
    
    # Цвета
    score = analysis.get("score", 0)
    passed = analysis.get("passed", False)
    
    if score >= 85:
        score_color = (38, 166, 154)     # зелёный
        status_text = "TEXTBOOK IMPULSE"
    elif score >= 70:
        score_color = (76, 175, 80)      # светло-зелёный
        status_text = "GOOD IMPULSE"
    elif score >= 50:
        score_color = (255, 183, 77)     # жёлтый
        status_text = "AMBIGUOUS"
    elif score >= 30:
        score_color = (255, 152, 0)      # оранжевый
        status_text = "UNLIKELY"
    else:
        score_color = (239, 83, 80)      # красный
        status_text = "NO IMPULSE"
    
    text_color = (209, 212, 220)      # TEXT_CLR
    dim_color = (120, 120, 140)
    border_color = (54, 58, 69)       # GRID_CLR
    
    x0 = chart_w + 12
    y = 15
    
    # ── Рамка панели ──
    draw.rectangle(
        [(chart_w + 2, 0), (total_w - 1, chart_h - 1)],
        outline=border_color, width=2
    )
    
    # ── Заголовок ──
    draw.text((x0, y), "ELLIOTT WAVE ANALYSIS", fill=text_color, font=font_title)
    y += 25
    draw.line([(x0, y), (total_w - 12, y)], fill=border_color, width=1)
    y += 10
    
    # ── Score ──
    score_bar_w = 200
    score_fill_w = int(score_bar_w * score / 100)
    
    draw.text((x0, y), f"Score: {score}/100", fill=score_color, font=font_title)
    y += 22
    
    # Score bar
    draw.rectangle([(x0, y), (x0 + score_bar_w, y + 12)], outline=border_color)
    if score_fill_w > 0:
        draw.rectangle([(x0 + 1, y + 1), (x0 + score_fill_w, y + 11)], fill=score_color)
    y += 18
    
    draw.text((x0, y), status_text, fill=score_color, font=font_body)
    y += 22
    
    draw.line([(x0, y), (total_w - 12, y)], fill=border_color, width=1)
    y += 10
    
    # ── Wave Direction ──
    wave_dir = analysis.get("wave_direction", "unclear")
    dir_color = (38, 166, 154) if wave_dir == "bullish" else (239, 83, 80) if wave_dir == "bearish" else dim_color
    draw.text((x0, y), f"Wave Direction:", fill=dim_color, font=font_small)
    y += 15
    draw.text((x0 + 8, y), wave_dir.upper(), fill=dir_color, font=font_body)
    y += 22
    
    # ── Signal Info ──
    draw.text((x0, y), f"Signal: {signal_direction}", fill=text_color, font=font_body)
    y += 18
    draw.text((x0, y), f"Price: {signal_price:.2f}", fill=dim_color, font=font_small)
    y += 18
    
    if trade_result:
        tr_color = (38, 166, 154) if trade_result == "win" else (239, 83, 80)
        draw.text((x0, y), f"Trade: {trade_result.upper()} ({trade_pnl:+.2f}%)", fill=tr_color, font=font_body)
        y += 22

    # ── W0 Status & Origin Divergence ──
    w0_rsi = analysis.get("w0_rsi")
    w0_status = analysis.get("w0_status")
    origin_div = analysis.get("origin_div", False)
    w3_mom = analysis.get("w3_momentum_peak", False)

    if w0_rsi is not None:
        if w0_status in ("OVERSOLD", "OVERBOUGHT"):
            stat_clr = (38, 166, 154)
        elif w0_status in ("COOL", "HOT"):
            stat_clr = (100, 181, 246)
        else:
            stat_clr = (239, 83, 80)
        draw.text((x0, y), f"W0 Origin: RSI {w0_rsi:.1f} ({w0_status})", fill=stat_clr, font=font_small)
        y += 16

    if origin_div:
        draw.text((x0, y), "[+] W0 ORIGIN DIVERGENCE", fill=(255, 215, 0), font=font_small)
        y += 16

    if w3_mom:
        draw.text((x0, y), "[+] W3 MOMENTUM PEAK", fill=(38, 166, 154), font=font_small)
        y += 16

    draw.line([(x0, y), (total_w - 12, y)], fill=border_color, width=1)
    y += 10
    
    # ── Разволновка (ключевая часть) ──
    waves = analysis.get("waves_identified", "")
    if waves:
        draw.text((x0, y), "WAVE IDENTIFICATION:", fill=text_color, font=font_body)
        y += 20
        
        # Разбиваем на строки (волны часто через запятую)
        wave_parts = waves.replace(", ", "\n").replace(",", "\n").split("\n")
        for wp in wave_parts:
            wp = wp.strip()
            if not wp:
                continue
            
            # Подсветка номеров волн
            wave_color = text_color
            if wp.startswith("W3") or wp.startswith("Wave 3"):
                wave_color = (38, 166, 154)   # W3 = самая важная
            elif wp.startswith("W5") or wp.startswith("Wave 5"):
                wave_color = (255, 183, 77)   # W5 = завершающая
            elif wp.startswith("W1") or wp.startswith("Wave 1"):
                wave_color = (100, 181, 246)  # W1 = начальная
            
            # Обрезаем если слишком длинная
            display_text = wp[:45] + "..." if len(wp) > 45 else wp
            draw.text((x0 + 4, y), display_text, fill=wave_color, font=font_small)
            y += 14
            
            if y > chart_h - 120:
                draw.text((x0 + 4, y), "...", fill=dim_color, font=font_small)
                y += 14
                break
    else:
        draw.text((x0, y), "WAVES: not identified", fill=dim_color, font=font_body)
        y += 20
    
    y += 5
    draw.line([(x0, y), (total_w - 12, y)], fill=border_color, width=1)
    y += 10
    
    # ── Нарушения правил ──
    violations = analysis.get("rule_violations", "")
    if violations and violations.lower() != "none":
        draw.text((x0, y), "RULE VIOLATIONS:", fill=(239, 83, 80), font=font_body)
        y += 18
        
        v_parts = violations.replace(", ", "\n").replace(",", "\n").split("\n")
        for vp in v_parts:
            vp = vp.strip()
            if not vp:
                continue
            display_text = vp[:45] + "..." if len(vp) > 45 else vp
            draw.text((x0 + 4, y), f"! {display_text}", fill=(255, 152, 0), font=font_small)
            y += 14
            if y > chart_h - 60:
                break
    else:
        draw.text((x0, y), "RULE VIOLATIONS: none", fill=(38, 166, 154), font=font_small)
        y += 18
    
    y += 5
    draw.line([(x0, y), (total_w - 12, y)], fill=border_color, width=1)
    y += 10
    
    # ── Reason ──
    reason = analysis.get("reason", "")
    if reason:
        draw.text((x0, y), "AI REASONING:", fill=dim_color, font=font_small)
        y += 16
        
        # Word wrap
        words = reason.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if len(test_line) > 40:
                draw.text((x0 + 4, y), line, fill=text_color, font=font_small)
                y += 13
                line = word
                if y > chart_h - 30:
                    break
            else:
                line = test_line
        if line and y < chart_h - 20:
            draw.text((x0 + 4, y), line, fill=text_color, font=font_small)
            y += 13
    
    # ── Provider ──
    provider = analysis.get("provider", "")
    if provider:
        draw.text((x0, chart_h - 20), f"via {provider}", fill=dim_color, font=font_small)
    
    # Экспорт
    buf = io.BytesIO()
    composite.save(buf, format='PNG')
    buf.seek(0)
    return buf.read()

