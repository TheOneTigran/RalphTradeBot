"""
strategy.py — Стратегия 3: Statistical Volatility Re-entry (Возврат в канал волатильности ATR/Keltner).

Математическая модель:
1. Волна W5 представляет собой статистический выброс за пределы нормального распределения волатильности.
2. Рассчитывается динамический канал Кельтнера:
   - Базовая линия: EMA(Close, 20)
   - Диапазон: ATR(14)
   - Верхняя граница: EMA + mult * ATR
   - Нижняя граница: EMA - mult * ATR
3. Условие подтверждения:
   - В момент пика W5 цена находилась за пределами границы (выброс волатильности).
   - Вход осуществляется строго тогда, когда закрытие свечи возвращается внутрь полосы:
     * SHORT: Close < Upper_Band (затухание аномального восходящего импульса).
     * LONG: Close > Lower_Band (затухание аномального нисходящего импульса).
4. Защита от бесконечного тренда:
   - Если свеча уходит дальше экстремума W5 за порог стоп-лосса — сигнал аннулируется.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from strategies_lab.strategies.base_strategy import BaseStrategy, TradeSignal, EntryDecision, TradeResult
from strategies_lab.strategies.baseline.strategy import BaselineStrategy


def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _calc_ema(series: np.ndarray, period: int = 20) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    ema = np.zeros_like(series)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
    return ema


class KeltnerReentryStrategy(BaseStrategy):
    """Стратегия возврата цены внутрь полос волатильности Кельтнера/ATR."""

    def __init__(
        self,
        atr_period: int = 14,
        ema_period: int = 20,
        atr_multiplier: float = 2.0,
        max_wait_bars: int = 5,
        buffer_pct: float = 0.20,
        min_rr: float = 1.5,
        be_offset_pct: float = 0.1,
    ):
        super().__init__(
            name="Strategy_3_Keltner_Reentry",
            description=f"Вход при возврате цены внутрь канала Кельтнера (EMA20 ± {atr_multiplier}*ATR14)."
        )
        self.atr_period = atr_period
        self.ema_period = ema_period
        self.atr_multiplier = atr_multiplier
        self.max_wait_bars = max_wait_bars
        self.buffer_pct = buffer_pct
        self.min_rr = min_rr
        self.be_offset_pct = be_offset_pct
        self._baseline = BaselineStrategy(buffer_pct=buffer_pct, min_rr=min_rr, be_offset_pct=be_offset_pct)

    def evaluate_entry(self, signal: TradeSignal, future_candles: pd.DataFrame) -> EntryDecision:
        w0 = signal.w0_price
        w5 = signal.w5_price
        w5_bar = signal.w5_bar
        direction = signal.direction
        df = signal.raw_df
        impulse_range = abs(w5 - w0)

        if impulse_range <= 0 or len(future_candles) == 0:
            return EntryDecision(should_enter=False, reason="no_data")

        # Расчёт ATR и EMA на объединённом срезе
        combined = pd.concat([df.iloc[:w5_bar + 1], future_candles])
        h = combined['high'].values.astype(np.float64)
        l = combined['low'].values.astype(np.float64)
        c = combined['close'].values.astype(np.float64)

        if len(c) < self.ema_period + 5:
            return EntryDecision(should_enter=False, reason="insufficient_history_for_bands")

        atr = _calc_atr(h, l, c, period=self.atr_period)
        ema = _calc_ema(c, period=self.ema_period)

        upper_bands = ema + self.atr_multiplier * atr
        lower_bands = ema - self.atr_multiplier * atr

        start_future_idx = w5_bar + 1
        max_check = min(len(future_candles), self.max_wait_bars)
        triggered = False
        trigger_idx = 0
        entry_price = 0.0

        if direction == "SHORT":
            sl_price = w5 * (1.0 + self.buffer_pct / 100.0)

            for i in range(max_check):
                curr_idx = start_future_idx + i
                if curr_idx >= len(c):
                    break

                c_high = h[curr_idx]
                c_close = c[curr_idx]
                upper = upper_bands[curr_idx]

                if c_high >= sl_price:
                    return EntryDecision(should_enter=False, reason="w5_extended_invalidated")

                # Триггер: закрытие вернулось ниже верхней границы
                if c_close < upper:
                    triggered = True
                    trigger_idx = i
                    entry_price = c_close
                    break

            if not triggered:
                return EntryDecision(should_enter=False, reason="no_keltner_reentry")

            risk = sl_price - entry_price
            if risk <= 0:
                return EntryDecision(should_enter=False, reason="negative_risk")

            tp1 = w5 - impulse_range * 0.236
            tp2 = w5 - impulse_range * 0.382
            tp3 = w5 - impulse_range * 0.500
            tp4 = w5 - impulse_range * 0.618

            avg_tp = 0.25 * tp1 + 0.35 * tp2 + 0.25 * tp3 + 0.15 * tp4
            reward = entry_price - avg_tp

        else:  # LONG
            sl_price = w5 * (1.0 - self.buffer_pct / 100.0)

            for i in range(max_check):
                curr_idx = start_future_idx + i
                if curr_idx >= len(c):
                    break

                c_low = l[curr_idx]
                c_close = c[curr_idx]
                lower = lower_bands[curr_idx]

                if c_low <= sl_price:
                    return EntryDecision(should_enter=False, reason="w5_extended_invalidated")

                # Триггер: закрытие вернулось выше нижней границы
                if c_close > lower:
                    triggered = True
                    trigger_idx = i
                    entry_price = c_close
                    break

            if not triggered:
                return EntryDecision(should_enter=False, reason="no_keltner_reentry")

            risk = entry_price - sl_price
            if risk <= 0:
                return EntryDecision(should_enter=False, reason="negative_risk")

            tp1 = w5 + impulse_range * 0.236
            tp2 = w5 + impulse_range * 0.382
            tp3 = w5 + impulse_range * 0.500
            tp4 = w5 + impulse_range * 0.618

            avg_tp = 0.25 * tp1 + 0.35 * tp2 + 0.25 * tp3 + 0.15 * tp4
            reward = avg_tp - entry_price

        rr = reward / risk if risk > 0 else 0.0
        if rr < self.min_rr:
            return EntryDecision(should_enter=False, reason=f"rr_too_low_{rr:.2f}")

        tp_levels = [
            {"name": "TP1", "price": tp1, "share": 0.25},
            {"name": "TP2", "price": tp2, "share": 0.35},
            {"name": "TP3", "price": tp3, "share": 0.25},
            {"name": "TP4", "price": tp4, "share": 0.15},
        ]

        return EntryDecision(
            should_enter=True,
            entry_bar=w5_bar + trigger_idx,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_levels=tp_levels,
            reason=f"keltner_reentry_bar_{trigger_idx}",
            extra_data={"rr": rr, "trigger_idx": trigger_idx}
        )

    def simulate_execution(
        self,
        decision: EntryDecision,
        direction: str,
        execution_candles: pd.DataFrame,
    ) -> TradeResult:
        trigger_offset = decision.extra_data.get("trigger_idx", 0)
        remaining_candles = execution_candles.iloc[trigger_offset:]
        if len(remaining_candles) == 0:
            remaining_candles = execution_candles

        return self._baseline.simulate_execution(decision, direction, remaining_candles)
