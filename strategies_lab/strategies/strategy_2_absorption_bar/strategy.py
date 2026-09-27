"""
strategy.py — Стратегия 2: Absorption & Rejection Bar (Свечное подтверждение исчерпания импульса).

Математическая модель:
1. Экстремум W5 фиксируется, но вход откладывается на 1-3 бара для проверки реакции ликвидности:
2. Триггер подтверждения (любое из двух условий):
   А. Outside / Engulfing Bar (Поглощение):
      - SHORT: свеча открывается/тестирует хай и закрывается ниже тела или минимума свечи W5.
      - LONG: свеча тестирует лой и закрывается выше тела или максимума свечи W5.
   Б. Rejection Wick (Пинбар с тенью отбоя):
      - SHORT: верхняя тень составляет >= 50% от всего диапазона (High - Low), а закрытие в нижней трети.
      - LONG: нижняя тень составляет >= 50% от всего диапазона (High - Low), а закрытие в верхней трети.
3. Инвалидация:
   - Если следующая свеча безоткатно пробивает экстремум W5 без формирования разворотного паттерна — сделка отменяется (защита от Wave Extension).
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


class AbsorptionBarStrategy(BaseStrategy):
    """Стратегия подтверждения по свечному поглощению и теням отбоя."""

    def __init__(
        self,
        max_look_bars: int = 3,
        min_wick_ratio: float = 0.50,
        buffer_pct: float = 0.20,
        min_rr: float = 1.5,
        be_offset_pct: float = 0.1,
    ):
        super().__init__(
            name="Strategy_2_Absorption_Bar",
            description="Вход только при подтверждении поглощением (Outside Bar) или пинбаром с тенью отбоя >= 50%."
        )
        self.max_look_bars = max_look_bars
        self.min_wick_ratio = min_wick_ratio
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

        w5_candle = df.iloc[w5_bar]
        w5_high = float(w5_candle['high'])
        w5_low = float(w5_candle['low'])
        w5_open = float(w5_candle['open'])
        w5_close = float(w5_candle['close'])

        max_check = min(len(future_candles), self.max_look_bars)
        triggered = False
        trigger_idx = 0
        entry_price = 0.0
        pattern_name = ""

        if direction == "SHORT":
            sl_price = w5 * (1.0 + self.buffer_pct / 100.0)

            for i in range(max_check):
                c_high = float(future_candles['high'].iloc[i])
                c_low = float(future_candles['low'].iloc[i])
                c_open = float(future_candles['open'].iloc[i])
                c_close = float(future_candles['close'].iloc[i])
                c_range = c_high - c_low

                # Инвалидация при пробое хая
                if c_high >= sl_price:
                    return EntryDecision(should_enter=False, reason="w5_extended_invalidated")

                # Паттерн 1: Медвежье поглощение (Outside Bar)
                if c_close < c_open and c_close <= w5_low:
                    triggered = True
                    trigger_idx = i
                    entry_price = c_close
                    pattern_name = "outside_engulfing"
                    break

                # Паттерн 2: Медвежий пинбар (Rejection Upper Wick)
                if c_range > 0:
                    upper_wick = c_high - max(c_open, c_close)
                    wick_ratio = upper_wick / c_range
                    if wick_ratio >= self.min_wick_ratio and c_close < (c_low + 0.40 * c_range):
                        triggered = True
                        trigger_idx = i
                        entry_price = c_close
                        pattern_name = f"rejection_wick_{wick_ratio:.2f}"
                        break

            if not triggered:
                return EntryDecision(should_enter=False, reason="no_absorption_pattern")

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
                c_high = float(future_candles['high'].iloc[i])
                c_low = float(future_candles['low'].iloc[i])
                c_open = float(future_candles['open'].iloc[i])
                c_close = float(future_candles['close'].iloc[i])
                c_range = c_high - c_low

                if c_low <= sl_price:
                    return EntryDecision(should_enter=False, reason="w5_extended_invalidated")

                # Паттерн 1: Бычье поглощение
                if c_close > c_open and c_close >= w5_high:
                    triggered = True
                    trigger_idx = i
                    entry_price = c_close
                    pattern_name = "outside_engulfing"
                    break

                # Паттерн 2: Бычий пинбар (Rejection Lower Wick)
                if c_range > 0:
                    lower_wick = min(c_open, c_close) - c_low
                    wick_ratio = lower_wick / c_range
                    if wick_ratio >= self.min_wick_ratio and c_close > (c_high - 0.40 * c_range):
                        triggered = True
                        trigger_idx = i
                        entry_price = c_close
                        pattern_name = f"rejection_wick_{wick_ratio:.2f}"
                        break

            if not triggered:
                return EntryDecision(should_enter=False, reason="no_absorption_pattern")

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
            reason=f"absorption_{pattern_name}_bar_{trigger_idx}",
            extra_data={"rr": rr, "pattern": pattern_name, "trigger_idx": trigger_idx}
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
