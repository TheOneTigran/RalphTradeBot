"""
strategy.py — Baseline Strategy (Текущая эталонная модель RalphTradeBot).

Логика:
1. Вход: немедленный на закрытии бара W5.
2. Стоп-лосс: за экстремумом W5 с буфером 0.3%.
3. Тейк-профиты: уровни коррекции Фибоначчи от всего импульса (W0 -> W5):
   - TP1: 23.6% (закрытие 25%)
   - TP2: 38.2% (закрытие 35%) -> перенос SL в безубыток (+0.1%)
   - TP3: 50.0% (закрытие 25%)
   - TP4: 61.8% (закрытие 15%)
4. Минимальный R:R: 1.5:1.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from strategies_lab.strategies.base_strategy import BaseStrategy, TradeSignal, EntryDecision, TradeResult


class BaselineStrategy(BaseStrategy):
    """Текущая эталонная стратегия RalphTradeBot v2.2."""

    def __init__(self, buffer_pct: float = 0.3, min_rr: float = 1.5, be_offset_pct: float = 0.1):
        super().__init__(
            name="Baseline_Immediate_W5",
            description="Вход сразу на закрытии W5, 4 ТП по Фибоначчи, безубыток после TP2."
        )
        self.buffer_pct = buffer_pct
        self.min_rr = min_rr
        self.be_offset_pct = be_offset_pct

    def evaluate_entry(self, signal: TradeSignal, future_candles: pd.DataFrame) -> EntryDecision:
        w0 = signal.w0_price
        w5 = signal.w5_price
        direction = signal.direction
        impulse_range = abs(w5 - w0)

        if impulse_range <= 0 or w5 <= 0:
            return EntryDecision(should_enter=False, reason="invalid_range")

        # Вход по закрытию W5 (или open следующей свечи)
        entry_price = float(future_candles['open'].iloc[0]) if len(future_candles) > 0 else float(signal.raw_df['close'].iloc[signal.w5_bar])

        if direction == "SHORT":
            sl_price = w5 * (1.0 + self.buffer_pct / 100.0)
            risk = sl_price - entry_price
            if risk <= 0:
                return EntryDecision(should_enter=False, reason="negative_risk")

            tp1 = w5 - impulse_range * 0.236
            tp2 = w5 - impulse_range * 0.382
            tp3 = w5 - impulse_range * 0.500
            tp4 = w5 - impulse_range * 0.618

            # Средневзвешенный TP
            avg_tp = 0.25 * tp1 + 0.35 * tp2 + 0.25 * tp3 + 0.15 * tp4
            reward = entry_price - avg_tp
        else:  # LONG
            sl_price = w5 * (1.0 - self.buffer_pct / 100.0)
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
            entry_bar=signal.w5_bar,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_levels=tp_levels,
            reason="baseline_w5_close",
            extra_data={"rr": rr, "w5": w5, "w0": w0}
        )

    def simulate_execution(
        self,
        decision: EntryDecision,
        direction: str,
        execution_candles: pd.DataFrame,
    ) -> TradeResult:
        entry = decision.entry_price
        initial_sl = decision.sl_price
        current_sl = initial_sl
        tp_levels = decision.tp_levels

        rem_share = 1.0
        accumulated_pnl = 0.0
        tp_hits = []
        tp2_reached = False

        highs = execution_candles['high'].values
        lows = execution_candles['low'].values
        opens = execution_candles['open'].values
        closes = execution_candles['close'].values
        times = execution_candles.index
        n = len(execution_candles)

        be_price = entry * (1.0 - self.be_offset_pct / 100.0) if direction == "SHORT" else entry * (1.0 + self.be_offset_pct / 100.0)

        mfe = 0.0
        mae = 0.0
        exit_bar = n - 1
        exit_price = closes[-1] if n > 0 else entry
        exit_reason = "timeout"

        risk_dist = abs(entry - initial_sl)

        for b in range(n):
            h, l, o, c = highs[b], lows[b], opens[b], closes[b]

            # MFE / MAE
            if direction == "SHORT":
                fav = (entry - l) / entry * 100.0
                adv = (h - entry) / entry * 100.0
            else:
                fav = (h - entry) / entry * 100.0
                adv = (entry - l) / entry * 100.0
            mfe = max(mfe, fav)
            mae = max(mae, adv)

            if direction == "SHORT":
                # Стоп-лосс
                if h >= current_sl:
                    exit_bar = b
                    exit_price = max(current_sl, o)
                    loss_pnl = (entry - exit_price) / entry * 100.0
                    accumulated_pnl += rem_share * loss_pnl
                    exit_reason = "be" if tp2_reached else "sl"
                    rem_share = 0.0
                    break

                # Тейк-профиты
                for tp in tp_levels:
                    if tp["name"] not in tp_hits and l <= tp["price"]:
                        tp_hits.append(tp["name"])
                        gain = (entry - tp["price"]) / entry * 100.0
                        accumulated_pnl += tp["share"] * gain
                        rem_share -= tp["share"]
                        if tp["name"] == "TP2":
                            tp2_reached = True
                            current_sl = be_price
                        if tp["name"] == "TP4":
                            exit_bar = b
                            exit_price = tp["price"]
                            exit_reason = "tp_full"
                            rem_share = 0.0
                            break
                if rem_share <= 0:
                    break

            else:  # LONG
                if l <= current_sl:
                    exit_bar = b
                    exit_price = min(current_sl, o)
                    loss_pnl = (exit_price - entry) / entry * 100.0
                    accumulated_pnl += rem_share * loss_pnl
                    exit_reason = "be" if tp2_reached else "sl"
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
                            current_sl = be_price
                        if tp["name"] == "TP4":
                            exit_bar = b
                            exit_price = tp["price"]
                            exit_reason = "tp_full"
                            rem_share = 0.0
                            break
                if rem_share <= 0:
                    break

        if rem_share > 0:
            final_pnl = (entry - exit_price) / entry * 100.0 if direction == "SHORT" else (exit_price - entry) / entry * 100.0
            accumulated_pnl += rem_share * final_pnl

        initial_risk_pct = (risk_dist / entry) * 100.0 if entry > 0 else 1.0
        
        # Финансовое моделирование: Депозит $1000, Риск 1% ($10), Комиссия 0.1% вход + 0.1% выход
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
