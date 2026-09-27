"""
exit_models.py — Изолированные модели выхода при ФИКСИРОВАННОМ входе Baseline (W5 Close).

Позволяет объективно оценить влияние механики закрытия позиций без искажения выборки фильтрами входа:
- Model A: Baseline Static Fibonacci (25% / 35% + БУ / 25% / 15%)
- Model B: Fail-Fast Time-Stop (ранний выход через N свечей, если нет +0.5R прибыли)
- Model C: Chandelier Volatility Trailing (50% на TP1 + 50% скользящий трейлинг 2.2*ATR)
- Model D: Dual Target 50/50 (50% на TP1 + 50% на TP2 с переносом в БУ)
- Model E: Quant Master Hybrid (Fail-Fast Time-Stop + частичные ТП + Chandelier Runner)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies_lab.strategies.base_strategy import BaseStrategy, TradeSignal, EntryDecision, TradeResult
from strategies_lab.strategies.baseline.strategy import BaselineStrategy
from strategies_lab.strategies.strategy_3_keltner_reentry.strategy import _calc_atr


class BaselineExitModel(BaselineStrategy):
    """Модель А: Классическая 4-ступенчатая фиксация по Фибоначчи с БУ после TP2."""
    def __init__(self):
        super().__init__()
        self.name = "Exit_Model_A_Baseline_Fibo"
        self.description = "Эталонная фиксация Фибоначчи (25/35/25/15) + перенос в БУ после TP2."


class FailFastTimeStopExitModel(BaseStrategy):
    """
    Модель B: Fail-Fast Time-Stop.
    Если за N свечей (по умолчанию 3 свечи) цена не достигла хотя бы +0.4R прибыли,
    сделка принудительно закрывается по рынку для отсечения зависших/убыточных позиций.
    """
    def __init__(self, time_stop_bars: int = 3, min_profit_r_to_stay: float = 0.4):
        super().__init__(
            name="Exit_Model_B_FailFast_TimeStop",
            description=f"Fail-Fast: принудительный выход на баре {time_stop_bars}, если прибыль < +{min_profit_r_to_stay}R."
        )
        self.time_stop_bars = time_stop_bars
        self.min_profit_r_to_stay = min_profit_r_to_stay
        self._baseline = BaselineStrategy()

    def evaluate_entry(self, signal: TradeSignal, future_candles: pd.DataFrame) -> EntryDecision:
        # Вход строго по Baseline (W5 close)
        return self._baseline.evaluate_entry(signal, future_candles)

    def simulate_execution(self, decision: EntryDecision, direction: str, execution_candles: pd.DataFrame) -> TradeResult:
        entry = decision.entry_price
        initial_sl = decision.sl_price
        current_sl = initial_sl
        tp_levels = decision.tp_levels

        rem_share = 1.0
        accumulated_pnl = 0.0
        tp_hits = []
        tp2_reached = False

        highs = execution_candles['high'].values.astype(np.float64)
        lows = execution_candles['low'].values.astype(np.float64)
        opens = execution_candles['open'].values.astype(np.float64)
        closes = execution_candles['close'].values.astype(np.float64)
        times = execution_candles.index
        n = len(execution_candles)

        be_price = entry * 0.999 if direction == "SHORT" else entry * 1.001
        risk_dist = abs(entry - initial_sl)
        initial_risk_pct = (risk_dist / entry) * 100.0 if entry > 0 else 1.0

        mfe = 0.0
        mae = 0.0
        exit_bar = n - 1
        exit_price = closes[-1] if n > 0 else entry
        exit_reason = "timeout"

        for b in range(n):
            h, l, o, c = highs[b], lows[b], opens[b], closes[b]

            if direction == "SHORT":
                fav = (entry - l) / entry * 100.0
                adv = (h - entry) / entry * 100.0
            else:
                fav = (h - entry) / entry * 100.0
                adv = (entry - l) / entry * 100.0
            mfe = max(mfe, fav)
            mae = max(mae, adv)

            # 1. Проверка Fail-Fast на баре time_stop_bars
            if b == self.time_stop_bars and not tp_hits:
                current_profit_pct = (entry - c) / entry * 100.0 if direction == "SHORT" else (c - entry) / entry * 100.0
                current_r = current_profit_pct / initial_risk_pct if initial_risk_pct > 0 else 0.0
                if current_r < self.min_profit_r_to_stay:
                    # Ранний выход
                    exit_bar = b
                    exit_price = c
                    exit_reason = "fail_fast"
                    loss_pnl = current_profit_pct
                    accumulated_pnl += rem_share * loss_pnl
                    rem_share = 0.0
                    break

            if direction == "SHORT":
                if h >= current_sl:
                    exit_bar = b
                    exit_price = max(current_sl, o)
                    accumulated_pnl += rem_share * ((entry - exit_price) / entry * 100.0)
                    exit_reason = "be" if tp2_reached else "sl"
                    rem_share = 0.0
                    break

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
            else:  # LONG
                if l <= current_sl:
                    exit_bar = b
                    exit_price = min(current_sl, o)
                    accumulated_pnl += rem_share * ((exit_price - entry) / entry * 100.0)
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

        # Финансовый расчет
        risk_budget_usd = 10.0
        pos_size_usd = risk_budget_usd / (initial_risk_pct / 100.0) if initial_risk_pct > 0 else 1000.0
        entry_fee = pos_size_usd * 0.001
        gross_pnl_usd = pos_size_usd * (accumulated_pnl / 100.0)
        exit_fee = max(0.0, pos_size_usd + gross_pnl_usd) * 0.001
        total_fee_usd = entry_fee + exit_fee
        net_pnl_usd = gross_pnl_usd - total_fee_usd
        net_pnl_r = net_pnl_usd / risk_budget_usd

        return TradeResult(
            signal_id="", symbol="", interval="", direction=direction,
            entry_time=times[0] if len(times) > 0 else None,
            exit_time=times[exit_bar] if exit_bar < len(times) else None,
            entry_price=entry, exit_price=exit_price, initial_sl=initial_sl,
            final_pnl_pct=round(accumulated_pnl, 2),
            pnl_r=round(net_pnl_r, 2),
            exit_reason=exit_reason, bars_held=exit_bar + 1,
            mfe_pct=round(mfe, 2), mae_pct=round(mae, 2),
            mfe_r=round(mfe / initial_risk_pct, 2) if initial_risk_pct > 0 else 0.0,
            mae_r=round(mae / initial_risk_pct, 2) if initial_risk_pct > 0 else 0.0,
            position_size_usd=round(pos_size_usd, 2),
            fee_usd=round(total_fee_usd, 2),
            pnl_usd=round(gross_pnl_usd, 2),
            net_pnl_usd=round(net_pnl_usd, 2),
            tp_hits=tp_hits,
        )


class PureChandelierExitModel(BaseStrategy):
    """
    Модель C: Baseline Entry + 50% на TP1 (23.6%) + БУ, остальные 50% по Chandelier Trailing (2.2*ATR).
    """
    def __init__(self, atr_multiplier: float = 2.2):
        super().__init__(
            name="Exit_Model_C_Chandelier_Trailing",
            description=f"Baseline Entry + 50% на TP1 + БУ, остальные 50% по Chandelier Trailing ({atr_multiplier}*ATR)."
        )
        self.atr_multiplier = atr_multiplier
        self._baseline = BaselineStrategy()

    def evaluate_entry(self, signal: TradeSignal, future_candles: pd.DataFrame) -> EntryDecision:
        decision = self._baseline.evaluate_entry(signal, future_candles)
        if decision.should_enter and decision.tp_levels:
            decision.tp_levels = [{"name": "TP1", "price": decision.tp_levels[0]["price"], "share": 0.50}]
        return decision

    def simulate_execution(self, decision: EntryDecision, direction: str, execution_candles: pd.DataFrame) -> TradeResult:
        entry = decision.entry_price
        initial_sl = decision.sl_price
        current_sl = initial_sl
        tp_levels = decision.tp_levels

        rem_share = 1.0
        accumulated_pnl = 0.0
        tp1_hit = False

        highs = execution_candles['high'].values.astype(np.float64)
        lows = execution_candles['low'].values.astype(np.float64)
        opens = execution_candles['open'].values.astype(np.float64)
        closes = execution_candles['close'].values.astype(np.float64)
        times = execution_candles.index
        n = len(execution_candles)

        atr = _calc_atr(highs, lows, closes, period=14) if n >= 14 else np.full(n, (highs[0] - lows[0]) or 1.0)
        be_price = entry * 0.999 if direction == "SHORT" else entry * 1.001
        risk_dist = abs(entry - initial_sl)
        initial_risk_pct = (risk_dist / entry) * 100.0 if entry > 0 else 1.0

        mfe = 0.0
        mae = 0.0
        exit_bar = n - 1
        exit_price = closes[-1] if n > 0 else entry
        exit_reason = "timeout"
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
                if tp1_hit:
                    chandelier = l + self.atr_multiplier * cur_atr
                    trail_sl = min(trail_sl, chandelier)
                    current_sl = min(be_price, trail_sl)

                if h >= current_sl:
                    exit_bar = b
                    exit_price = max(current_sl, o)
                    accumulated_pnl += rem_share * ((entry - exit_price) / entry * 100.0)
                    exit_reason = "trailing" if tp1_hit else "sl"
                    rem_share = 0.0
                    break

                if not tp1_hit and l <= tp_levels[0]["price"]:
                    tp1_hit = True
                    gain = (entry - tp_levels[0]["price"]) / entry * 100.0
                    accumulated_pnl += tp_levels[0]["share"] * gain
                    rem_share -= tp_levels[0]["share"]
                    trail_sl = be_price
                    current_sl = be_price

            else:  # LONG
                if tp1_hit:
                    chandelier = h - self.atr_multiplier * cur_atr
                    trail_sl = max(trail_sl, chandelier)
                    current_sl = max(be_price, trail_sl)

                if l <= current_sl:
                    exit_bar = b
                    exit_price = min(current_sl, o)
                    accumulated_pnl += rem_share * ((exit_price - entry) / entry * 100.0)
                    exit_reason = "trailing" if tp1_hit else "sl"
                    rem_share = 0.0
                    break

                if not tp1_hit and h >= tp_levels[0]["price"]:
                    tp1_hit = True
                    gain = (tp_levels[0]["price"] - entry) / entry * 100.0
                    accumulated_pnl += tp_levels[0]["share"] * gain
                    rem_share -= tp_levels[0]["share"]
                    trail_sl = be_price
                    current_sl = be_price

            if rem_share <= 0:
                break

        if rem_share > 0:
            final_pnl = (entry - exit_price) / entry * 100.0 if direction == "SHORT" else (exit_price - entry) / entry * 100.0
            accumulated_pnl += rem_share * final_pnl

        risk_budget_usd = 10.0
        pos_size_usd = risk_budget_usd / (initial_risk_pct / 100.0) if initial_risk_pct > 0 else 1000.0
        entry_fee = pos_size_usd * 0.001
        gross_pnl_usd = pos_size_usd * (accumulated_pnl / 100.0)
        exit_fee = max(0.0, pos_size_usd + gross_pnl_usd) * 0.001
        total_fee_usd = entry_fee + exit_fee
        net_pnl_usd = gross_pnl_usd - total_fee_usd
        net_pnl_r = net_pnl_usd / risk_budget_usd

        return TradeResult(
            signal_id="", symbol="", interval="", direction=direction,
            entry_time=times[0] if len(times) > 0 else None,
            exit_time=times[exit_bar] if exit_bar < len(times) else None,
            entry_price=entry, exit_price=exit_price, initial_sl=initial_sl,
            final_pnl_pct=round(accumulated_pnl, 2),
            pnl_r=round(net_pnl_r, 2),
            exit_reason=exit_reason, bars_held=exit_bar + 1,
            mfe_pct=round(mfe, 2), mae_pct=round(mae, 2),
            mfe_r=round(mfe / initial_risk_pct, 2) if initial_risk_pct > 0 else 0.0,
            mae_r=round(mae / initial_risk_pct, 2) if initial_risk_pct > 0 else 0.0,
            position_size_usd=round(pos_size_usd, 2),
            fee_usd=round(total_fee_usd, 2),
            pnl_usd=round(gross_pnl_usd, 2),
            net_pnl_usd=round(net_pnl_usd, 2),
            tp_hits=["TP1"] if tp1_hit else [],
        )


class DualTargetQuickLockExitModel(BaseStrategy):
    """
    Модель D: Conservative Quick Lock (50% на TP1 23.6%, 50% на TP2 38.2% с безубытком после TP1).
    """
    def __init__(self):
        super().__init__(
            name="Exit_Model_D_DualTarget_50_50",
            description="Быстрая фиксация: 50% на TP1 (23.6%) + БУ, 50% на TP2 (38.2%)."
        )
        self._baseline = BaselineStrategy()

    def evaluate_entry(self, signal: TradeSignal, future_candles: pd.DataFrame) -> EntryDecision:
        decision = self._baseline.evaluate_entry(signal, future_candles)
        if decision.should_enter and len(decision.tp_levels) >= 2:
            decision.tp_levels = [
                {"name": "TP1", "price": decision.tp_levels[0]["price"], "share": 0.50},
                {"name": "TP2", "price": decision.tp_levels[1]["price"], "share": 0.50},
            ]
        return decision

    def simulate_execution(self, decision: EntryDecision, direction: str, execution_candles: pd.DataFrame) -> TradeResult:
        entry = decision.entry_price
        initial_sl = decision.sl_price
        current_sl = initial_sl
        tp_levels = decision.tp_levels

        rem_share = 1.0
        accumulated_pnl = 0.0
        tp1_hit = False

        highs = execution_candles['high'].values.astype(np.float64)
        lows = execution_candles['low'].values.astype(np.float64)
        opens = execution_candles['open'].values.astype(np.float64)
        closes = execution_candles['close'].values.astype(np.float64)
        times = execution_candles.index
        n = len(execution_candles)

        be_price = entry * 0.999 if direction == "SHORT" else entry * 1.001
        risk_dist = abs(entry - initial_sl)
        initial_risk_pct = (risk_dist / entry) * 100.0 if entry > 0 else 1.0

        mfe = 0.0
        mae = 0.0
        exit_bar = n - 1
        exit_price = closes[-1] if n > 0 else entry
        exit_reason = "timeout"
        tp_hits = []

        for b in range(n):
            h, l, o, c = highs[b], lows[b], opens[b], closes[b]

            if direction == "SHORT":
                fav = (entry - l) / entry * 100.0
                adv = (h - entry) / entry * 100.0
            else:
                fav = (h - entry) / entry * 100.0
                adv = (entry - l) / entry * 100.0
            mfe = max(mfe, fav)
            mae = max(mae, adv)

            if direction == "SHORT":
                if h >= current_sl:
                    exit_bar = b
                    exit_price = max(current_sl, o)
                    accumulated_pnl += rem_share * ((entry - exit_price) / entry * 100.0)
                    exit_reason = "be" if tp1_hit else "sl"
                    rem_share = 0.0
                    break

                if not tp1_hit and l <= tp_levels[0]["price"]:
                    tp1_hit = True
                    tp_hits.append("TP1")
                    gain = (entry - tp_levels[0]["price"]) / entry * 100.0
                    accumulated_pnl += 0.50 * gain
                    rem_share -= 0.50
                    current_sl = be_price

                if tp1_hit and l <= tp_levels[1]["price"]:
                    tp_hits.append("TP2")
                    gain = (entry - tp_levels[1]["price"]) / entry * 100.0
                    accumulated_pnl += 0.50 * gain
                    rem_share = 0.0
                    exit_bar = b
                    exit_price = tp_levels[1]["price"]
                    exit_reason = "tp_full"
                    break

            else:  # LONG
                if l <= current_sl:
                    exit_bar = b
                    exit_price = min(current_sl, o)
                    accumulated_pnl += rem_share * ((exit_price - entry) / entry * 100.0)
                    exit_reason = "be" if tp1_hit else "sl"
                    rem_share = 0.0
                    break

                if not tp1_hit and h >= tp_levels[0]["price"]:
                    tp1_hit = True
                    tp_hits.append("TP1")
                    gain = (tp_levels[0]["price"] - entry) / entry * 100.0
                    accumulated_pnl += 0.50 * gain
                    rem_share -= 0.50
                    current_sl = be_price

                if tp1_hit and h >= tp_levels[1]["price"]:
                    tp_hits.append("TP2")
                    gain = (tp_levels[1]["price"] - entry) / entry * 100.0
                    accumulated_pnl += 0.50 * gain
                    rem_share = 0.0
                    exit_bar = b
                    exit_price = tp_levels[1]["price"]
                    exit_reason = "tp_full"
                    break

            if rem_share <= 0:
                break

        if rem_share > 0:
            final_pnl = (entry - exit_price) / entry * 100.0 if direction == "SHORT" else (exit_price - entry) / entry * 100.0
            accumulated_pnl += rem_share * final_pnl

        risk_budget_usd = 10.0
        pos_size_usd = risk_budget_usd / (initial_risk_pct / 100.0) if initial_risk_pct > 0 else 1000.0
        entry_fee = pos_size_usd * 0.001
        gross_pnl_usd = pos_size_usd * (accumulated_pnl / 100.0)
        exit_fee = max(0.0, pos_size_usd + gross_pnl_usd) * 0.001
        total_fee_usd = entry_fee + exit_fee
        net_pnl_usd = gross_pnl_usd - total_fee_usd
        net_pnl_r = net_pnl_usd / risk_budget_usd

        return TradeResult(
            signal_id="", symbol="", interval="", direction=direction,
            entry_time=times[0] if len(times) > 0 else None,
            exit_time=times[exit_bar] if exit_bar < len(times) else None,
            entry_price=entry, exit_price=exit_price, initial_sl=initial_sl,
            final_pnl_pct=round(accumulated_pnl, 2),
            pnl_r=round(net_pnl_r, 2),
            exit_reason=exit_reason, bars_held=exit_bar + 1,
            mfe_pct=round(mfe, 2), mae_pct=round(mae, 2),
            mfe_r=round(mfe / initial_risk_pct, 2) if initial_risk_pct > 0 else 0.0,
            mae_r=round(mae / initial_risk_pct, 2) if initial_risk_pct > 0 else 0.0,
            position_size_usd=round(pos_size_usd, 2),
            fee_usd=round(total_fee_usd, 2),
            pnl_usd=round(gross_pnl_usd, 2),
            net_pnl_usd=round(net_pnl_usd, 2),
            tp_hits=tp_hits,
        )


class QuantMasterHybridExitModel(BaseStrategy):
    """
    Модель E: Quant Master Hybrid.
    1. Baseline Entry на W5 close.
    2. Fail-Fast Time-Stop на баре 3: если прибыль < +0.35R, выход по рынку.
    3. TP1 (23.6% Фибо): фиксация 40% объема + БУ.
    4. TP2 (38.2% Фибо): фиксация 30% объема.
    5. Остаток 30%: Chandelier Trailing (2.2 * ATR) для максимизации MFE.
    """
    def __init__(self, time_stop_bars: int = 3, min_profit_r_to_stay: float = 0.35, atr_multiplier: float = 2.2):
        super().__init__(
            name="Exit_Model_E_Quant_Master_Hybrid",
            description=f"Quant Master: Fail-Fast (бар {time_stop_bars}) + 40% TP1 + 30% TP2 + 30% Chandelier Runner."
        )
        self.time_stop_bars = time_stop_bars
        self.min_profit_r_to_stay = min_profit_r_to_stay
        self.atr_multiplier = atr_multiplier
        self._baseline = BaselineStrategy()

    def evaluate_entry(self, signal: TradeSignal, future_candles: pd.DataFrame) -> EntryDecision:
        decision = self._baseline.evaluate_entry(signal, future_candles)
        if decision.should_enter and len(decision.tp_levels) >= 2:
            decision.tp_levels = [
                {"name": "TP1", "price": decision.tp_levels[0]["price"], "share": 0.40},
                {"name": "TP2", "price": decision.tp_levels[1]["price"], "share": 0.30},
            ]
        return decision

    def simulate_execution(self, decision: EntryDecision, direction: str, execution_candles: pd.DataFrame) -> TradeResult:
        entry = decision.entry_price
        initial_sl = decision.sl_price
        current_sl = initial_sl
        tp_levels = decision.tp_levels

        rem_share = 1.0
        accumulated_pnl = 0.0
        tp1_hit = False
        tp2_hit = False

        highs = execution_candles['high'].values.astype(np.float64)
        lows = execution_candles['low'].values.astype(np.float64)
        opens = execution_candles['open'].values.astype(np.float64)
        closes = execution_candles['close'].values.astype(np.float64)
        times = execution_candles.index
        n = len(execution_candles)

        atr = _calc_atr(highs, lows, closes, period=14) if n >= 14 else np.full(n, (highs[0] - lows[0]) or 1.0)
        be_price = entry * 0.999 if direction == "SHORT" else entry * 1.001
        risk_dist = abs(entry - initial_sl)
        initial_risk_pct = (risk_dist / entry) * 100.0 if entry > 0 else 1.0

        mfe = 0.0
        mae = 0.0
        exit_bar = n - 1
        exit_price = closes[-1] if n > 0 else entry
        exit_reason = "timeout"
        tp_hits = []
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

            # Fail-Fast проверка на баре 3
            if b == self.time_stop_bars and not tp_hits:
                current_profit_pct = (entry - c) / entry * 100.0 if direction == "SHORT" else (c - entry) / entry * 100.0
                current_r = current_profit_pct / initial_risk_pct if initial_risk_pct > 0 else 0.0
                if current_r < self.min_profit_r_to_stay:
                    exit_bar = b
                    exit_price = c
                    exit_reason = "fail_fast"
                    accumulated_pnl += rem_share * current_profit_pct
                    rem_share = 0.0
                    break

            if direction == "SHORT":
                # Chandelier trailing после TP2
                if tp2_hit:
                    chandelier = l + self.atr_multiplier * cur_atr
                    trail_sl = min(trail_sl, chandelier)
                    current_sl = min(be_price, trail_sl)

                if h >= current_sl:
                    exit_bar = b
                    exit_price = max(current_sl, o)
                    accumulated_pnl += rem_share * ((entry - exit_price) / entry * 100.0)
                    exit_reason = "trailing" if tp2_hit else ("be" if tp1_hit else "sl")
                    rem_share = 0.0
                    break

                if not tp1_hit and l <= tp_levels[0]["price"]:
                    tp1_hit = True
                    tp_hits.append("TP1")
                    accumulated_pnl += 0.40 * ((entry - tp_levels[0]["price"]) / entry * 100.0)
                    rem_share -= 0.40
                    current_sl = be_price

                if tp1_hit and not tp2_hit and l <= tp_levels[1]["price"]:
                    tp2_hit = True
                    tp_hits.append("TP2")
                    accumulated_pnl += 0.30 * ((entry - tp_levels[1]["price"]) / entry * 100.0)
                    rem_share -= 0.30
                    trail_sl = be_price

            else:  # LONG
                if tp2_hit:
                    chandelier = h - self.atr_multiplier * cur_atr
                    trail_sl = max(trail_sl, chandelier)
                    current_sl = max(be_price, trail_sl)

                if l <= current_sl:
                    exit_bar = b
                    exit_price = min(current_sl, o)
                    accumulated_pnl += rem_share * ((exit_price - entry) / entry * 100.0)
                    exit_reason = "trailing" if tp2_hit else ("be" if tp1_hit else "sl")
                    rem_share = 0.0
                    break

                if not tp1_hit and h >= tp_levels[0]["price"]:
                    tp1_hit = True
                    tp_hits.append("TP1")
                    accumulated_pnl += 0.40 * ((tp_levels[0]["price"] - entry) / entry * 100.0)
                    rem_share -= 0.40
                    current_sl = be_price

                if tp1_hit and not tp2_hit and h >= tp_levels[1]["price"]:
                    tp2_hit = True
                    tp_hits.append("TP2")
                    accumulated_pnl += 0.30 * ((tp_levels[1]["price"] - entry) / entry * 100.0)
                    rem_share -= 0.30
                    trail_sl = be_price

            if rem_share <= 0:
                break

        if rem_share > 0:
            final_pnl = (entry - exit_price) / entry * 100.0 if direction == "SHORT" else (exit_price - entry) / entry * 100.0
            accumulated_pnl += rem_share * final_pnl

        risk_budget_usd = 10.0
        pos_size_usd = risk_budget_usd / (initial_risk_pct / 100.0) if initial_risk_pct > 0 else 1000.0
        entry_fee = pos_size_usd * 0.001
        gross_pnl_usd = pos_size_usd * (accumulated_pnl / 100.0)
        exit_fee = max(0.0, pos_size_usd + gross_pnl_usd) * 0.001
        total_fee_usd = entry_fee + exit_fee
        net_pnl_usd = gross_pnl_usd - total_fee_usd
        net_pnl_r = net_pnl_usd / risk_budget_usd

        return TradeResult(
            signal_id="", symbol="", interval="", direction=direction,
            entry_time=times[0] if len(times) > 0 else None,
            exit_time=times[exit_bar] if exit_bar < len(times) else None,
            entry_price=entry, exit_price=exit_price, initial_sl=initial_sl,
            final_pnl_pct=round(accumulated_pnl, 2),
            pnl_r=round(net_pnl_r, 2),
            exit_reason=exit_reason, bars_held=exit_bar + 1,
            mfe_pct=round(mfe, 2), mae_pct=round(mae, 2),
            mfe_r=round(mfe / initial_risk_pct, 2) if initial_risk_pct > 0 else 0.0,
            mae_r=round(mae / initial_risk_pct, 2) if initial_risk_pct > 0 else 0.0,
            position_size_usd=round(pos_size_usd, 2),
            fee_usd=round(total_fee_usd, 2),
            pnl_usd=round(gross_pnl_usd, 2),
            net_pnl_usd=round(net_pnl_usd, 2),
            tp_hits=tp_hits,
        )
