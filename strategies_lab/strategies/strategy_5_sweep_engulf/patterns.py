"""
patterns.py — Алгоритм выявления паттерна свечного свипа, поглощения тела и заполнения лимита.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np


@dataclass
class SweepEngulfPattern:
    """Результат детекции свечи свипа и поглощения."""
    found: bool
    s_rel_idx: int = -1             # Индекс свечи S: -1 = сама свеча W5, 0..N = индекс в future_candles
    s_time: Any = None              # Timestamp свечи S
    s_high: float = 0.0
    s_low: float = 0.0
    s_open: float = 0.0
    s_close: float = 0.0
    s_range: float = 0.0
    swept_high: float = 0.0
    swept_low: float = 0.0
    limit_entry_price: float = 0.0  # 50% диапазона свечи S
    sl_price: float = 0.0           # Строго конец тени свечи S (без буфера)
    tp_price: float = 0.0           # Тейк 1:3 R:R
    risk_dist: float = 0.0          # |SL - Entry|
    reward_dist: float = 0.0        # |TP - Entry| = 3 * risk_dist
    risk_pct: float = 0.0           # Риск в % от точки входа
    sweep_type: str = ""            # "w5_is_sweep_engulf" / "post_w5_sweep_engulf"
    reason: str = ""


@dataclass
class FillCheckResult:
    """Результат проверки заполнения лимитного ордера на последующих барах."""
    is_filled: bool
    fill_rel_idx: int = -1          # Индекс бара в future_candles, где лимит был исполнен
    fill_time: Any = None
    fill_price: float = 0.0
    immediate_sl: bool = False      # Задет ли SL на том же баре (консервативный худший исход)
    invalidation_reason: str = ""   # Если отменён до входа


def calculate_net_1_to_3_tp(
    entry_price: float,
    sl_price: float,
    direction: str,
    fee_rate: float = 0.0005,
    target_net_r: float = 3.0,
) -> Tuple[float, float, float]:
    """
    Рассчитывает точный уровень Take-Profit так, чтобы чистая прибыль
    ПОСЛЕ вычета всех биржевых комиссий (вход + выход) составляла строго target_net_r (1:3).
    """
    risk_dist = abs(entry_price - sl_price)
    risk_frac = (risk_dist / entry_price) if entry_price > 0 else 0.01
    reward_frac = (target_net_r * risk_frac + 2.0 * fee_rate) / (1.0 - fee_rate)
    reward_dist = entry_price * reward_frac
    if direction == "SHORT":
        tp_price = entry_price - reward_dist
    else:
        tp_price = entry_price + reward_dist
    return tp_price, risk_dist, reward_dist


def detect_sweep_engulf_candle(
    w5_candle: pd.Series,
    future_candles: pd.DataFrame,
    direction: str,
    prev_candle: Optional[pd.Series] = None,
    max_sweep_lookahead: int = 3,
    retrace_ratio: float = 0.50,
    fee_rate: float = 0.0005,
    target_net_r: float = 3.0,
) -> SweepEngulfPattern:
    """
    Ищет свечу свипа экстремума с поглощением тела:
    
    Случай 1 (Сама свеча W5 является свечой S):
      На пике импульса W5 пробивает предыдущую свечу тенью и поглощает её тело.
      Вход на 50% свечи W5 в течение следующих 1..3 свечей.
      
    Случай 2 (Свеча в future_candles пробивает W5):
      В пределах max_sweep_lookahead свечей после W5 формируется свеча S,
      которая пробивает экстремум W5 тенью и поглощает тело W5.
    """
    w5_high = float(w5_candle['high'])
    w5_low = float(w5_candle['low'])
    w5_open = float(w5_candle['open'])
    w5_close = float(w5_candle['close'])
    w5_time = w5_candle.name if hasattr(w5_candle, "name") else None

    # ═════════════════════════════════════════════════════════════════════════
    # 1. Проверяем, является ли сама свеча W5 свечой свипа и поглощения (Случай 1)
    # ═════════════════════════════════════════════════════════════════════════
    if prev_candle is not None:
        p_high = float(prev_candle['high'])
        p_low = float(prev_candle['low'])
        p_open = float(prev_candle['open'])
        p_close = float(prev_candle['close'])
        p_body_top = max(p_open, p_close)
        p_body_bottom = min(p_open, p_close)

        w5_range = w5_high - w5_low
        if w5_range > 0:
            if direction == "SHORT":
                # SHORT: W5 High > prev High, W5 закрывается медвежьей и ниже тела prev
                is_sweep = w5_high > p_high
                is_bearish = w5_close < w5_open
                engulfs = w5_close < p_body_bottom
                if is_sweep and is_bearish and engulfs:
                    limit_entry = w5_low + retrace_ratio * w5_range
                    sl_p = w5_high
                    if sl_p > limit_entry:
                        tp_p, risk, reward = calculate_net_1_to_3_tp(limit_entry, sl_p, "SHORT", fee_rate, target_net_r)
                        return SweepEngulfPattern(
                            found=True,
                            s_rel_idx=-1,  # Свеча S — это W5
                            s_time=w5_time,
                            s_high=w5_high,
                            s_low=w5_low,
                            s_open=w5_open,
                            s_close=w5_close,
                            s_range=w5_range,
                            swept_high=p_high,
                            swept_low=p_low,
                            limit_entry_price=limit_entry,
                            sl_price=sl_p,
                            tp_price=tp_p,
                            risk_dist=risk,
                            reward_dist=reward,
                            risk_pct=(risk / limit_entry) * 100.0,
                            sweep_type="w5_is_sweep_engulf",
                            reason="w5_swept_and_engulfed_prev_candle",
                        )
            else:
                # LONG: W5 Low < prev Low, W5 закрывается бычьей и выше тела prev
                is_sweep = w5_low < p_low
                is_bullish = w5_close > w5_open
                engulfs = w5_close > p_body_top
                if is_sweep and is_bullish and engulfs:
                    limit_entry = w5_low + retrace_ratio * w5_range
                    sl_p = w5_low
                    if limit_entry > sl_p:
                        tp_p, risk, reward = calculate_net_1_to_3_tp(limit_entry, sl_p, "LONG", fee_rate, target_net_r)
                        return SweepEngulfPattern(
                            found=True,
                            s_rel_idx=-1,  # Свеча S — это W5
                            s_time=w5_time,
                            s_high=w5_high,
                            s_low=w5_low,
                            s_open=w5_open,
                            s_close=w5_close,
                            s_range=w5_range,
                            swept_high=p_high,
                            swept_low=p_low,
                            limit_entry_price=limit_entry,
                            sl_price=sl_p,
                            tp_price=tp_p,
                            risk_dist=risk,
                            reward_dist=reward,
                            risk_pct=(risk / limit_entry) * 100.0,
                            sweep_type="w5_is_sweep_engulf",
                            reason="w5_swept_and_engulfed_prev_candle",
                        )

    # ═════════════════════════════════════════════════════════════════════════
    # 2. Проверяем, появилась ли свеча свипа в future_candles (Случай 2)
    # ═════════════════════════════════════════════════════════════════════════
    n_available = len(future_candles)
    if n_available == 0:
        return SweepEngulfPattern(found=False, reason="no_future_candles")

    w5_body_top = max(w5_open, w5_close)
    w5_body_bottom = min(w5_open, w5_close)
    check_limit = min(n_available, max_sweep_lookahead)

    for i in range(check_limit):
        c_high = float(future_candles['high'].iloc[i])
        c_low = float(future_candles['low'].iloc[i])
        c_open = float(future_candles['open'].iloc[i])
        c_close = float(future_candles['close'].iloc[i])
        c_range = c_high - c_low

        if c_range <= 0:
            continue

        if direction == "SHORT":
            prior_highs = [w5_high] + [float(future_candles['high'].iloc[k]) for k in range(i)]
            highest_prior = max(prior_highs)

            is_sweep = c_high > highest_prior
            is_bearish = c_close < c_open
            engulfs = c_close < w5_body_bottom

            if is_sweep and is_bearish and engulfs:
                limit_entry = c_low + retrace_ratio * c_range
                sl_price = c_high
                if sl_price <= limit_entry:
                    continue
                tp_price, risk_dist, reward_dist = calculate_net_1_to_3_tp(limit_entry, sl_price, "SHORT", fee_rate, target_net_r)

                return SweepEngulfPattern(
                    found=True,
                    s_rel_idx=i,
                    s_time=future_candles.index[i],
                    s_high=c_high,
                    s_low=c_low,
                    s_open=c_open,
                    s_close=c_close,
                    s_range=c_range,
                    swept_high=highest_prior,
                    swept_low=w5_low,
                    limit_entry_price=limit_entry,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    risk_dist=risk_dist,
                    reward_dist=reward_dist,
                    risk_pct=(risk_dist / limit_entry) * 100.0 if limit_entry > 0 else 0.0,
                    sweep_type="post_w5_sweep_engulf",
                    reason=f"post_w5_candle_{i+1}_swept_w5",
                )

        else:  # LONG
            prior_lows = [w5_low] + [float(future_candles['low'].iloc[k]) for k in range(i)]
            lowest_prior = min(prior_lows)

            is_sweep = c_low < lowest_prior
            is_bullish = c_close > c_open
            engulfs = c_close > w5_body_top

            if is_sweep and is_bullish and engulfs:
                limit_entry = c_low + retrace_ratio * c_range
                sl_price = c_low
                if limit_entry <= sl_price:
                    continue
                tp_price, risk_dist, reward_dist = calculate_net_1_to_3_tp(limit_entry, sl_price, "LONG", fee_rate, target_net_r)

                return SweepEngulfPattern(
                    found=True,
                    s_rel_idx=i,
                    s_time=future_candles.index[i],
                    s_high=c_high,
                    s_low=c_low,
                    s_open=c_open,
                    s_close=c_close,
                    s_range=c_range,
                    swept_high=w5_high,
                    swept_low=lowest_prior,
                    limit_entry_price=limit_entry,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    risk_dist=risk_dist,
                    reward_dist=reward_dist,
                    risk_pct=(risk_dist / limit_entry) * 100.0 if limit_entry > 0 else 0.0,
                    sweep_type="post_w5_sweep_engulf",
                    reason=f"post_w5_candle_{i+1}_swept_w5",
                )

    return SweepEngulfPattern(found=False, reason="no_sweep_engulf_found")


def check_limit_fill(
    pattern: SweepEngulfPattern,
    future_candles: pd.DataFrame,
    direction: str,
    fill_window_bars: int = 3,
) -> FillCheckResult:
    """
    Проверяет, дошла ли цена до 50% свечи S в пределах fill_window_bars.
    
    Если s_rel_idx == -1 (свеча S — это W5):
      проверка начинается со свечи 0 в future_candles (W5+1).
    Если s_rel_idx >= 0 (свеча S в future_candles):
      проверка начинается со свечи s_rel_idx + 1 в future_candles.
    """
    if not pattern.found:
        return FillCheckResult(is_filled=False, invalidation_reason="pattern_not_found")

    s_idx = pattern.s_rel_idx
    n_candles = len(future_candles)

    start_bar = 0 if s_idx == -1 else s_idx + 1
    end_bar = min(n_candles, start_bar + fill_window_bars)

    if start_bar >= n_candles:
        return FillCheckResult(is_filled=False, invalidation_reason="no_candles_after_s")

    limit_p = pattern.limit_entry_price
    sl_p = pattern.sl_price

    for b in range(start_bar, end_bar):
        c_high = float(future_candles['high'].iloc[b])
        c_low = float(future_candles['low'].iloc[b])
        c_open = float(future_candles['open'].iloc[b])
        c_time = future_candles.index[b]
        bar_offset = (b + 1) if s_idx == -1 else (b - s_idx)

        if direction == "SHORT":
            # 1. Инвалидация гэпом выше стопа
            if c_open >= sl_p:
                return FillCheckResult(
                    is_filled=False,
                    invalidation_reason=f"gapped_above_sl_at_bar_{bar_offset}",
                )

            # 2. Если на этой свече задет стоп
            if c_high >= sl_p:
                return FillCheckResult(
                    is_filled=True,
                    fill_rel_idx=b,
                    fill_time=c_time,
                    fill_price=limit_p,
                    immediate_sl=True,
                    invalidation_reason="sl_hit_on_fill_candle",
                )

            # 3. Нормальное касание лимита
            if c_high >= limit_p:
                return FillCheckResult(
                    is_filled=True,
                    fill_rel_idx=b,
                    fill_time=c_time,
                    fill_price=limit_p,
                    immediate_sl=False,
                )

        else:  # LONG
            if c_open <= sl_p:
                return FillCheckResult(
                    is_filled=False,
                    invalidation_reason=f"gapped_below_sl_at_bar_{bar_offset}",
                )

            if c_low <= sl_p:
                return FillCheckResult(
                    is_filled=True,
                    fill_rel_idx=b,
                    fill_time=c_time,
                    fill_price=limit_p,
                    immediate_sl=True,
                    invalidation_reason="sl_hit_on_fill_candle",
                )

            if c_low <= limit_p:
                return FillCheckResult(
                    is_filled=True,
                    fill_rel_idx=b,
                    fill_time=c_time,
                    fill_price=limit_p,
                    immediate_sl=False,
                )

    return FillCheckResult(
        is_filled=False,
        invalidation_reason=f"limit_unfilled_after_{fill_window_bars}_bars",
    )
