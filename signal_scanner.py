"""
signal_scanner.py — Сканер дивергенций DivergenceRadarBot для RalphTradeBot.

1. Поиск swing highs/lows через scipy.signal.argrelextrema.
2. Bearish Regular (SHORT):
   - Higher High по цене + Lower High по RSI в зоне перекупленности.
   - Чистота цены: между p1 и p2 нет баров выше p2.
   - Чистота RSI: между пивотами RSI не проваливается в OS (< rsi_os).
3. Bullish Regular (LONG):
   - Lower Low по цене + Higher Low по RSI в зоне перепроданности.
   - Чистота цены: между p1 и p2 нет баров ниже p2.
   - Чистота RSI: между пивотами RSI не выстреливает в OB (> rsi_ob).
4. Автономный расчет Wilder's RSI на чистом NumPy (без numba).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


def calculate_rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Вычисляет классический Wilder's RSI на чистом NumPy."""
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n <= period:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, len(deltas)):
        curr_gain = gains[i]
        curr_loss = losses[i]
        avg_gain = (avg_gain * (period - 1) + curr_gain) / period
        avg_loss = (avg_loss * (period - 1) + curr_loss) / period

        idx = i + 1
        if avg_loss == 0:
            rsi[idx] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[idx] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@dataclass
class SignalEvent:
    """Один обнаруженный сигнал (sweep + RSI divergence)."""
    bar_index: int                  # Индекс бара сигнала в DataFrame (P2 / W5)
    direction: str                  # "LONG" или "SHORT"
    signal_price: float             # Цена закрытия на баре сигнала
    rsi_value: float                # RSI на баре сигнала
    swept_level_price: float        # Цена первого pivot-уровня (P1 / W3)
    swept_level_rsi: float          # RSI на момент формирования P1
    swept_level_bar: int = 0        # Индекс бара P1 в DataFrame
    
    # Для бэктеста
    entry_price: float = 0.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    exit_price: float = 0.0
    trade_result: str = ""         # "win" / "loss"
    trade_pnl_pct: float = 0.0     # P&L в %
    
    # Контекст для рендера
    active_pivot_highs: List[Tuple[float, float]] = field(default_factory=list)
    active_pivot_lows: List[Tuple[float, float]] = field(default_factory=list)


def scan_signals(
    df: pd.DataFrame,
    left_bars: int = 5,
    right_bars: int = 3,
    rsi_len: int = 7,
    rsi_ob: float = 70.0,
    rsi_os: float = 30.0,
    tp_pct: float = 1.5,
    sl_pct: float = 3.0,
    pivot_order: int = 3,
    min_pivot_distance: int = 6,
    max_pivot_distance: int = 45,
    min_rsi_divergence: float = 3.0,
    rsi_zone_tolerance: float = 5.0,
) -> List[SignalEvent]:
    """Сканирует исторические данные и находит регулярные RSI-дивергенции."""
    open_p = df['open'].values.astype(np.float64)
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    n = len(df)
    
    if n < 2 * pivot_order + rsi_len + 10:
        return []
    
    rsi = calculate_rsi_wilder(close, rsi_len)
    
    tp_frac = tp_pct / 100.0 if tp_pct > 1.0 else tp_pct
    sl_frac = sl_pct / 100.0 if sl_pct > 1.0 else sl_pct
    
    high_pivots = argrelextrema(high, np.greater_equal, order=pivot_order)[0]
    low_pivots = argrelextrema(low, np.less_equal, order=pivot_order)[0]
    
    candidate_signals: List[Tuple[int, str, int, float, float, float, float]] = []
    
    # --- 1. Поиск медвежьих дивергенций (SHORT) ---
    ob_threshold = rsi_ob - rsi_zone_tolerance
    for i in range(1, len(high_pivots)):
        p1 = int(high_pivots[i - 1])
        p2 = int(high_pivots[i])
        dist = p2 - p1
        
        if dist < min_pivot_distance or dist > max_pivot_distance:
            continue
        
        if high[p2] <= high[p1]:
            continue
        
        r1, r2 = float(rsi[p1]), float(rsi[p2])
        if r2 >= r1 or (r1 - r2) < min_rsi_divergence:
            continue
        
        if r1 < ob_threshold or r2 < ob_threshold:
            continue
        
        # Фильтр чистоты цены
        between_highs = high[p1 + 1 : p2]
        if len(between_highs) > 0 and np.max(between_highs) > high[p2]:
            continue
        
        # Фильтр чистоты RSI
        between_rsi = rsi[p1 + 1 : p2]
        if len(between_rsi) > 0 and np.min(between_rsi) < rsi_os:
            continue
        
        candidate_signals.append((p2, "SHORT", p1, float(high[p1]), r1, r2, float(high[p2])))
    
    # --- 2. Поиск бычьих дивергенций (LONG) ---
    os_threshold = rsi_os + rsi_zone_tolerance
    for i in range(1, len(low_pivots)):
        p1 = int(low_pivots[i - 1])
        p2 = int(low_pivots[i])
        dist = p2 - p1
        
        if dist < min_pivot_distance or dist > max_pivot_distance:
            continue
        
        if low[p2] >= low[p1]:
            continue
        
        r1, r2 = float(rsi[p1]), float(rsi[p2])
        if r2 <= r1 or (r2 - r1) < min_rsi_divergence:
            continue
        
        if r1 > os_threshold or r2 > os_threshold:
            continue
        
        # Фильтр чистоты цены
        between_lows = low[p1 + 1 : p2]
        if len(between_lows) > 0 and np.min(between_lows) < low[p2]:
            continue
        
        # Фильтр чистоты RSI
        between_rsi = rsi[p1 + 1 : p2]
        if len(between_rsi) > 0 and np.max(between_rsi) > rsi_ob:
            continue
        
        candidate_signals.append((p2, "LONG", p1, float(low[p1]), r1, r2, float(low[p2])))
    
    candidate_signals.sort(key=lambda s: s[0])
    
    signals: List[SignalEvent] = []
    last_exit_bar = -1
    
    for p2, direction, p1, p1_price, p1_rsi, p2_rsi, p2_price in candidate_signals:
        if p2 <= last_exit_bar:
            continue
        
        if p2 + 1 >= n:
            continue
        
        entry = open_p[p2 + 1]
        
        if direction == "SHORT":
            tp_price = entry * (1.0 - tp_frac)
            sl_price = entry * (1.0 + sl_frac)
        else:
            tp_price = entry * (1.0 + tp_frac)
            sl_price = entry * (1.0 - sl_frac)
        
        recent_phs = [(float(high[b]), float(rsi[b])) for b in high_pivots if b <= p2][-5:]
        recent_pls = [(float(low[b]), float(rsi[b])) for b in low_pivots if b <= p2][-5:]
        
        signal = SignalEvent(
            bar_index=p2,
            direction=direction,
            signal_price=float(close[p2]),
            rsi_value=p2_rsi,
            swept_level_price=p1_price,
            swept_level_rsi=p1_rsi,
            swept_level_bar=p1,
            entry_price=entry,
            tp_price=tp_price,
            sl_price=sl_price,
            active_pivot_highs=recent_phs,
            active_pivot_lows=recent_pls,
        )
        
        exit_found = False
        for bar in range(p2 + 1, n):
            if direction == "SHORT":
                if high[bar] >= sl_price:
                    signal.exit_price = max(sl_price, open_p[bar])
                    signal.trade_result = "loss"
                    signal.trade_pnl_pct = (entry - signal.exit_price) / entry * 100.0
                    last_exit_bar = bar
                    exit_found = True
                    break
                elif low[bar] <= tp_price:
                    signal.exit_price = min(tp_price, open_p[bar])
                    signal.trade_result = "win"
                    signal.trade_pnl_pct = (entry - signal.exit_price) / entry * 100.0
                    last_exit_bar = bar
                    exit_found = True
                    break
            else:
                if low[bar] <= sl_price:
                    signal.exit_price = min(sl_price, open_p[bar])
                    signal.trade_result = "loss"
                    signal.trade_pnl_pct = (signal.exit_price - entry) / entry * 100.0
                    last_exit_bar = bar
                    exit_found = True
                    break
                elif high[bar] >= tp_price:
                    signal.exit_price = max(tp_price, open_p[bar])
                    signal.trade_result = "win"
                    signal.trade_pnl_pct = (signal.exit_price - entry) / entry * 100.0
                    last_exit_bar = bar
                    exit_found = True
                    break
        
        if not exit_found:
            signal.exit_price = float(close[-1])
            if direction == "SHORT":
                signal.trade_pnl_pct = (entry - signal.exit_price) / entry * 100.0
            else:
                signal.trade_pnl_pct = (signal.exit_price - entry) / entry * 100.0
            signal.trade_result = "win" if signal.trade_pnl_pct > 0 else "loss"
            last_exit_bar = n - 1
        
        signals.append(signal)
    
    return signals


def get_rsi_array(df: pd.DataFrame, rsi_len: int) -> np.ndarray:
    close = df['close'].values.astype(np.float64)
    return calculate_rsi_wilder(close, rsi_len)
