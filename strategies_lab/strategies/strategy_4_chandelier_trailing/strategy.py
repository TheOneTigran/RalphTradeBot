"""
strategy.py — Стратегия 4: Chandelier Volatility Trailing (Динамическое сопровождение разворота).

Математическая модель:
1. Вход: по свечному подтверждению исчерпания W5 (Absorption Bar).
2. Фиксация прибыли:
   - TP1 (23.6% Фибо): фиксация 30% объема.
   - TP2 (38.2% Фибо): фиксация 30% объема, перенос базового SL в безубыток (+0.1%).
3. Динамический трейлинг оставшихся 40% позиции:
   - Вместо жесткого закрытия на 61.8% Фибоначчи оставшаяся часть позиции удерживается
     по динамическому скользящему стопу Chandelier Exit (на основе ATR):
     * LONG: Trail_SL = max(Trail_SL_{t-1}, High_t - 2.5 * ATR_14)
     * SHORT: Trail_SL = min(Trail_SL_{t-1}, Low_t + 2.5 * ATR_14)
4. Преимущество:
   - Позволяет забирать глобальные развороты тренда (когда импульс откатывает на 100%+ своей длины)
     и существенно увеличивать среднее математическое ожидание в R.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from strategies_lab.strategies.base_strategy import BaseStrategy, TradeSignal, EntryDecision, TradeResult
from strategies_lab.strategies.strategy_2_absorption_bar.strategy import AbsorptionBarStrategy
from strategies_lab.strategies.strategy_3_keltner_reentry.strategy import _calc_atr


class ChandelierTrailingStrategy(BaseStrategy):
    """Стратегия динамического трейлинга Chandelier Exit."""

    def __init__(
        self,
        atr_period: int = 14,
        trail_multiplier: float = 2.5,
        buffer_pct: float = 0.20,
        min_rr: float = 1.5,
        be_offset_pct: float = 0.1,
    ):
        super().__init__(
            name="Strategy_4_Chandelier_Trailing",
            description=f"Частичный ТП (30%+30%) + динамический волатильный трейлинг 40% остатка (Chandelier {trail_multiplier}*ATR)."
        )
        self.atr_period = atr_period
        self.trail_multiplier = trail_multiplier
        self.buffer_pct = buffer_pct
        self.min_rr = min_rr
        self.be_offset_pct = be_offset_pct
        self._entry_filter = AbsorptionBarStrategy(buffer_pct=buffer_pct, min_rr=min_rr, be_offset_pct=be_offset_pct)

    def evaluate_entry(self, signal: TradeSignal, future_candles: pd.DataFrame) -> EntryDecision:
        # Используем фильтр поглощения для надежного входа
        decision = self._entry_filter.evaluate_entry(signal, future_candles)
        if not decision.should_enter:
            return decision

        # Модифицируем структуру TP: 30% на TP1, 30% на TP2, 40% на Chandelier Trailing
        decision.tp_levels = [
            {"name": "TP1", "price": decision.tp_levels[0]["price"], "share": 0.30},
            {"name": "TP2", "price": decision.tp_levels[1]["price"], "share": 0.30},
        ]
        decision.reason = f"absorption_entry_with_chandelier_trail"
        return decision

    def simulate_execution(
        self,
        decision: EntryDecision,
        direction: str,
        execution_candles: pd.DataFrame,
    ) -> TradeResult:
        trigger_offset = decision.extra_data.get("trigger_idx", 0)
        candles = execution_candles.iloc[trigger_offset:]
        if len(candles) == 0:
            candles = execution_candles

        entry = decision.entry_price
        initial_sl = decision.sl_price
        current_sl = initial_sl
        tp_levels = decision.tp_levels

        rem_share = 1.0
        accumulated_pnl = 0.0
        tp_hits = []
        tp2_reached = False

        highs = candles['high'].values.astype(np.float64)
        lows = candles['low'].values.astype(np.float64)
        opens = candles['open'].values.astype(np.float64)
        closes = candles['close'].values.astype(np.float64)
        times = candles.index
        n = len(candles)

        atr = _calc_atr(highs, lows, closes, period=self.atr_period) if n >= self.atr_period else np.full(n, (highs[0] - lows[0]) or 1.0)

        be_price = entry * (1.0 - self.be_offset_pct / 100.0) if direction == "SHORT" else entry * (1.0 + self.be_offset_pct / 100.0)

        mfe = 0.0
        mae = 0.0
        exit_bar = n - 1
        exit_price = closes[-1] if n > 0 else entry
        exit_reason = "timeout"
        risk_dist = abs(entry - initial_sl)

        trail_sl = initial_sl

        for b in range(n):
            h, l, o, c = highs[b], lows[b], opens[b], closes[b]
            cur_atr = atr[b] if b < len(atr) and atr[b] > 0 else (h - l)

            if direction == "SHORT":
                fav = (entry - l) / entry * 100.0
                adv = (h - entry) / entry * 100.0
            else:
                fav = (h - entry) / entry * 100.0
                adv = (entry - l) / entry * 100.0
            mfe = max(mfe, fav)
            mae = max(mae, adv)

            if direction == "SHORT":
                # Обновление Chandelier Trailing после взятия TP2
                if tp2_reached:
                    chandelier_candidate = l + self.trail_multiplier * cur_atr
                    trail_sl = min(trail_sl, chandelier_candidate)
                    current_sl = min(be_price, trail_sl)

                # Проверка стоп-лосса (фиксированного или трейлинга)
                if h >= current_sl:
                    exit_bar = b
                    exit_price = max(current_sl, o)
                    loss_pnl = (entry - exit_price) / entry * 100.0
                    accumulated_pnl += rem_share * loss_pnl
                    exit_reason = "trailing" if tp2_reached else "sl"
                    rem_share = 0.0
                    break

                # Тейк-профиты 1 и 2
                for tp in tp_levels:
                    if tp["name"] not in tp_hits and l <= tp["price"]:
                        tp_hits.append(tp["name"])
                        gain = (entry - tp["price"]) / entry * 100.0
                        accumulated_pnl += tp["share"] * gain
                        rem_share -= tp["share"]
                        if tp["name"] == "TP2":
                            tp2_reached = True
                            trail_sl = be_price
                            current_sl = be_price

            else:  # LONG
                if tp2_reached:
                    chandelier_candidate = h - self.trail_multiplier * cur_atr
                    trail_sl = max(trail_sl, chandelier_candidate)
                    current_sl = max(be_price, trail_sl)

                if l <= current_sl:
                    exit_bar = b
                    exit_price = min(current_sl, o)
                    loss_pnl = (exit_price - entry) / entry * 100.0
                    accumulated_pnl += rem_share * loss_pnl
                    exit_reason = "trailing" if tp2_reached else "sl"
                    rem_share = 0.0
                    break

                for tp in tp_levels:
                    if tp["name"] not in tp_hits and h >= tp["price"]:
                        tp_hits.append(tp["name"])
                        gain = (tp["price"] - entry) / entry * 100.0
                        accumulated_pnl += tp["share"] * gain
                        rem_share -= tp["share"]
                        if tp["name"] == "TP2":
                            tp2_reached = True
                            trail_sl = be_price
                            current_sl = be_price

            if rem_share <= 0:
                break

        if rem_share > 0:
            final_pnl = (entry - exit_price) / entry * 100.0 if direction == "SHORT" else (exit_price - entry) / entry * 100.0
            accumulated_pnl += rem_share * final_pnl

        initial_risk_pct = (risk_dist / entry) * 100.0 if entry > 0 else 1.0
        
        risk_budget_usd = 10.0
        pos_size_usd = risk_budget_usd / (initial_risk_pct / 100.0) if initial_risk_pct > 0 else 1000.0
        
        entry_fee = pos_size_usd * 0.001
        gross_pnl_usd = pos_size_usd * (accumulated_pnl / 100.0)
        exit_fee = max(0.0, pos_size_usd + gross_pnl_usd) * 0.001
        total_fee_usd = entry_fee + exit_fee
        
        net_pnl_usd = gross_pnl_usd - total_fee_usd
        net_pnl_r = net_pnl_usd / risk_budget_usd

        mfe_r = mfe / initial_risk_pct if initial_risk_pct > 0 else 0.0
        mae_r = mae / initial_risk_pct if initial_risk_pct > 0 else 0.0

        entry_time = times[0] if len(times) > 0 else None
        exit_time = times[exit_bar] if exit_bar < len(times) else None

        return TradeResult(
            signal_id="",
            symbol="",
            interval="",
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry,
            exit_price=exit_price,
            initial_sl=initial_sl,
            final_pnl_pct=round(accumulated_pnl, 2),
            pnl_r=round(net_pnl_r, 2),
            exit_reason=exit_reason,
            bars_held=exit_bar + 1,
            mfe_pct=round(mfe, 2),
            mae_pct=round(mae, 2),
            mfe_r=round(mfe_r, 2),
            mae_r=round(mae_r, 2),
            position_size_usd=round(pos_size_usd, 2),
            fee_usd=round(total_fee_usd, 2),
            pnl_usd=round(gross_pnl_usd, 2),
            net_pnl_usd=round(net_pnl_usd, 2),
            slippage_tax_pct=round(decision.slippage_tax_pct, 2),
            tp_hits=tp_hits,
        )
