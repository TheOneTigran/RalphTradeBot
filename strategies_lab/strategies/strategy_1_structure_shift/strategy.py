"""
strategy.py — Стратегия 1: Multi-Timeframe Market Structure Shift (15m -> 1m).

Архитектура Multi-TF:
1. Сигнал (зона W5, RSI-дивергенция) детектируется на 15m графике.
2. В момент закрытия 15m-свечи алгоритм переключается на 1-минутные данные (df_1m).
3. На 1m-таймфрейме в окне последних 10-15 минут перед пиком W5 ищется локальный микро-свинг:
   - Для SHORT: Swing Low микроструктуры (последний локальный минимум перед пиком).
   - Для LONG: Swing High микроструктуры (последний локальный максимум перед дном).
4. Условие триггера:
   - Первая 1m-свеча, закрывшаяся ЗА уровнем микро-свинга (Close < Swing Low для SHORT / Close > Swing High для LONG).
5. Защита от удлинения (Wave Extension Invalidation):
   - Если 1m-свеча пробивает экстремум W5 (+ буфер 0.15%) ДО слома структуры — сигнал отменяется.
6. Метрика Slippage Tax:
   - Вычисляется % проскальзывания цены входа относительно идеального пика W5 к дистанции до TP1.
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


class MarketStructureShiftStrategy(BaseStrategy):
    """Стратегия Multi-Timeframe подтверждения по слому микроструктуры (15m -> 1m)."""

    def __init__(
        self,
        max_wait_1m_bars: int = 30,  # Максимальное ожидание слома на 1m (30 минут)
        lookback_1m_bars: int = 15,  # Глубина поиска микро-свинга на 1m (15 минут)
        buffer_pct: float = 0.15,
        min_rr: float = 1.3,
        be_offset_pct: float = 0.1,
    ):
        super().__init__(
            name="Strategy_1_MultiTF_MSS",
            description="Multi-TF слом микроструктуры: 15m импульс подтверждается пробоем свинга на 1m котировках."
        )
        self.max_wait_1m_bars = max_wait_1m_bars
        self.lookback_1m_bars = lookback_1m_bars
        self.buffer_pct = buffer_pct
        self.min_rr = min_rr
        self.be_offset_pct = be_offset_pct
        self._baseline = BaselineStrategy(buffer_pct=buffer_pct, min_rr=min_rr, be_offset_pct=be_offset_pct)

    def evaluate_entry(self, signal: TradeSignal, future_candles: pd.DataFrame) -> EntryDecision:
        w0 = signal.w0_price
        w5 = signal.w5_price
        w5_time = signal.w5_time
        direction = signal.direction
        impulse_range = abs(w5 - w0)
        df_1m = signal.df_1m

        if impulse_range <= 0 or len(future_candles) == 0:
            return EntryDecision(should_enter=False, reason="no_data")

        # Целевой TP1 для расчета Slippage Tax
        tp1_target = w5 - impulse_range * 0.236 if direction == "SHORT" else w5 + impulse_range * 0.236
        dist_to_tp1 = abs(tp1_target - w5)

        # -------------------------------------------------------------
        # ВАРИАНТ А: Доступны минутные данные (Multi-TF режим)
        # -------------------------------------------------------------
        if df_1m is not None and isinstance(df_1m.index, pd.DatetimeIndex) and w5_time in df_1m.index:
            w5_idx = df_1m.index.get_loc(w5_time)
            if isinstance(w5_idx, slice):
                w5_idx = w5_idx.start

            # Окно поиска свинга на 1m (последние 15 минут до пика W5)
            start_lookback = max(0, w5_idx - self.lookback_1m_bars)
            slice_1m_prior = df_1m.iloc[start_lookback : w5_idx + 1]

            # Будущие минутные свечи для проверки триггера (до 30 минут)
            slice_1m_future = df_1m.iloc[w5_idx + 1 : w5_idx + 1 + self.max_wait_1m_bars]

            if len(slice_1m_prior) >= 3 and len(slice_1m_future) >= 1:
                if direction == "SHORT":
                    # Локальный минимум микроструктуры на 1m
                    swing_level = float(slice_1m_prior['low'].min())
                    sl_price = w5 * (1.0 + self.buffer_pct / 100.0)

                    triggered = False
                    entry_price = 0.0
                    entry_time = None
                    trigger_1m_idx = 0

                    for m in range(len(slice_1m_future)):
                        m_high = float(slice_1m_future['high'].iloc[m])
                        m_close = float(slice_1m_future['close'].iloc[m])

                        # Инвалидация: цена перебила пик W5
                        if m_high >= sl_price:
                            return EntryDecision(should_enter=False, reason="w5_extended_invalidated_1m")

                        # Слом структуры на 1m: свеча закрылась ниже микро-свинга
                        if m_close < swing_level:
                            triggered = True
                            entry_price = m_close
                            entry_time = slice_1m_future.index[m]
                            trigger_1m_idx = m
                            break

                    if not triggered:
                        return EntryDecision(should_enter=False, reason="mss_1m_timeout")

                    risk = sl_price - entry_price
                    if risk <= 0:
                        return EntryDecision(should_enter=False, reason="negative_risk")

                    tp1 = w5 - impulse_range * 0.236
                    tp2 = w5 - impulse_range * 0.382
                    tp3 = w5 - impulse_range * 0.500
                    tp4 = w5 - impulse_range * 0.618

                    avg_tp = 0.25 * tp1 + 0.35 * tp2 + 0.25 * tp3 + 0.15 * tp4
                    reward = entry_price - avg_tp

                    # Расчет налога на проскальзывание (Slippage Tax)
                    slippage = abs(entry_price - w5)
                    slippage_tax = (slippage / dist_to_tp1 * 100.0) if dist_to_tp1 > 0 else 0.0

                else:  # LONG
                    swing_level = float(slice_1m_prior['high'].max())
                    sl_price = w5 * (1.0 - self.buffer_pct / 100.0)

                    triggered = False
                    entry_price = 0.0
                    entry_time = None
                    trigger_1m_idx = 0

                    for m in range(len(slice_1m_future)):
                        m_low = float(slice_1m_future['low'].iloc[m])
                        m_close = float(slice_1m_future['close'].iloc[m])

                        if m_low <= sl_price:
                            return EntryDecision(should_enter=False, reason="w5_extended_invalidated_1m")

                        if m_close > swing_level:
                            triggered = True
                            entry_price = m_close
                            entry_time = slice_1m_future.index[m]
                            trigger_1m_idx = m
                            break

                    if not triggered:
                        return EntryDecision(should_enter=False, reason="mss_1m_timeout")

                    risk = entry_price - sl_price
                    if risk <= 0:
                        return EntryDecision(should_enter=False, reason="negative_risk")

                    tp1 = w5 + impulse_range * 0.236
                    tp2 = w5 + impulse_range * 0.382
                    tp3 = w5 + impulse_range * 0.500
                    tp4 = w5 + impulse_range * 0.618

                    avg_tp = 0.25 * tp1 + 0.35 * tp2 + 0.25 * tp3 + 0.15 * tp4
                    reward = avg_tp - entry_price

                    slippage = abs(entry_price - w5)
                    slippage_tax = (slippage / dist_to_tp1 * 100.0) if dist_to_tp1 > 0 else 0.0

                rr = reward / risk if risk > 0 else 0.0
                if rr < self.min_rr:
                    return EntryDecision(should_enter=False, reason=f"rr_too_low_{rr:.2f}")

                tp_levels = [
                    {"name": "TP1", "price": tp1, "share": 0.25},
                    {"name": "TP2", "price": tp2, "share": 0.35},
                    {"name": "TP3", "price": tp3, "share": 0.25},
                    {"name": "TP4", "price": tp4, "share": 0.15},
                ]

                # Смещение в 15m свечах (15 минут = 1 свеча)
                trigger_15m_offset = max(0, trigger_1m_idx // 15)

                return EntryDecision(
                    should_enter=True,
                    entry_bar=signal.w5_bar + trigger_15m_offset,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_levels=tp_levels,
                    reason=f"mss_1m_breakout_m{trigger_1m_idx}",
                    slippage_tax_pct=slippage_tax,
                    extra_data={"rr": rr, "trigger_idx": trigger_15m_offset, "trigger_1m_idx": trigger_1m_idx}
                )

        # -------------------------------------------------------------
        # ВАРИАНТ Б: Фоллбэк на свечи текущего ТФ (если 1m недоступен)
        # -------------------------------------------------------------
        return EntryDecision(should_enter=False, reason="df_1m_not_found")

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
