"""
strategy.py — Реализация BaseStrategy для Стратегии 5 (Sweep & Engulf + 50% Limit Retracement).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from strategies_lab.strategies.base_strategy import (
    BaseStrategy,
    TradeSignal,
    EntryDecision,
    TradeResult,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.patterns import (
    detect_sweep_engulf_candle,
    check_limit_fill,
    SweepEngulfPattern,
    FillCheckResult,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.config import (
    MAX_SWEEP_LOOKAHEAD,
    RETRACE_RATIO,
    RR_TARGET,
    DEFAULT_FILL_WINDOW,
    FEE_RATE,
    ACCOUNT_DEPOSIT_USD,
    RISK_PER_TRADE_USD,
    TIMEOUT_BARS,
)


class SweepEngulfRetracementStrategy(BaseStrategy):
    """
    Стратегия входа на свечном свипе с поглощением тела экстремума W5
    и лимитным ордером на 50% отката свечи S с соотношением R:R 1:3.
    """

    def __init__(
        self,
        name: str = "Strategy_5_Sweep_Engulf",
        max_sweep_lookahead: int = MAX_SWEEP_LOOKAHEAD,
        retrace_ratio: float = RETRACE_RATIO,
        fill_window_bars: int = DEFAULT_FILL_WINDOW,
        rr_target: float = RR_TARGET,
        fee_rate: float = FEE_RATE,
        timeout_bars: int = TIMEOUT_BARS,
    ):
        desc = (
            f"Свип W5 + поглощение тела, лимит 50% свечи S (окно {fill_window_bars} бар), "
            f"SL по концу тени, TP 1:{int(rr_target)}"
        )
        super().__init__(name=name, description=desc)
        self.max_sweep_lookahead = max_sweep_lookahead
        self.retrace_ratio = retrace_ratio
        self.fill_window_bars = fill_window_bars
        self.rr_target = rr_target
        self.fee_rate = fee_rate
        self.timeout_bars = timeout_bars

    def evaluate_entry(
        self,
        signal: TradeSignal,
        future_candles: pd.DataFrame,
    ) -> EntryDecision:
        if len(future_candles) == 0:
            return EntryDecision(should_enter=False, reason="no_future_candles")

        w5_candle = signal.raw_df.iloc[-1]
        prev_candle = signal.raw_df.iloc[-2] if len(signal.raw_df) >= 2 else None
        direction = signal.direction

        # 1. Поиск свечи свипа и поглощения тела W5
        pattern = detect_sweep_engulf_candle(
            w5_candle=w5_candle,
            future_candles=future_candles,
            direction=direction,
            prev_candle=prev_candle,
            max_sweep_lookahead=self.max_sweep_lookahead,
            retrace_ratio=self.retrace_ratio,
        )

        if not pattern.found:
            return EntryDecision(
                should_enter=False,
                reason=pattern.reason,
                extra_data={"pattern_found": False},
            )

        # 2. Проверка исполнения лимитного ордера на 50% свечи S
        fill_res = check_limit_fill(
            pattern=pattern,
            future_candles=future_candles,
            direction=direction,
            fill_window_bars=self.fill_window_bars,
        )

        if not fill_res.is_filled:
            return EntryDecision(
                should_enter=False,
                reason=fill_res.invalidation_reason,
                extra_data={
                    "pattern_found": True,
                    "pattern": pattern,
                    "fill_res": fill_res,
                },
            )

        # Сигнал полностью подтвержден и лимит исполнен!
        abs_entry_bar = signal.w5_bar + 1 + fill_res.fill_rel_idx
        tp_levels = [
            {"name": "TP_1_3", "price": pattern.tp_price, "share": 1.0}
        ]

        return EntryDecision(
            should_enter=True,
            entry_bar=abs_entry_bar,
            entry_time=fill_res.fill_time,
            entry_price=pattern.limit_entry_price,
            sl_price=pattern.sl_price,
            tp_levels=tp_levels,
            reason=f"sweep_engulf_filled_bar_{fill_res.fill_rel_idx - pattern.s_rel_idx}",
            extra_data={
                "pattern": pattern,
                "fill_res": fill_res,
                "s_rel_idx": pattern.s_rel_idx,
                "fill_rel_idx": fill_res.fill_rel_idx,
                "immediate_sl": fill_res.immediate_sl,
            },
        )

    def simulate_execution(
        self,
        decision: EntryDecision,
        direction: str,
        execution_candles: pd.DataFrame,
    ) -> TradeResult:
        entry = decision.entry_price
        initial_sl = decision.sl_price
        tp_price = decision.tp_levels[0]["price"] if decision.tp_levels else entry
        extra = decision.extra_data
        pattern: SweepEngulfPattern = extra.get("pattern")
        fill_rel_idx: int = extra.get("fill_rel_idx", 0)
        immediate_sl: bool = extra.get("immediate_sl", False)

        risk_dist = abs(entry - initial_sl)
        initial_risk_pct = (risk_dist / entry) * 100.0 if entry > 0 else 1.0

        n = len(execution_candles)
        times = execution_candles.index
        highs = execution_candles['high'].values
        lows = execution_candles['low'].values
        opens = execution_candles['open'].values
        closes = execution_candles['close'].values

        mfe = 0.0
        mae = 0.0

        # Если на баре заполнения сразу был задет SL (консервативный худший исход)
        if immediate_sl:
            exit_bar = fill_rel_idx
            exit_price = initial_sl
            exit_reason = "sl"
            bars_held = 1
            if direction == "SHORT":
                accumulated_pnl = (entry - exit_price) / entry * 100.0
                mae = (highs[fill_rel_idx] - entry) / entry * 100.0
            else:
                accumulated_pnl = (exit_price - entry) / entry * 100.0
                mae = (entry - lows[fill_rel_idx]) / entry * 100.0
            mfe = 0.0
        else:
            exit_bar = n - 1
            exit_price = closes[-1] if n > 0 else entry
            exit_reason = "timeout"
            accumulated_pnl = 0.0

            # Итерируем со свечи заполнения лимита
            for b in range(fill_rel_idx, n):
                h, l, o, c = highs[b], lows[b], opens[b], closes[b]
                bars_since_fill = b - fill_rel_idx

                # MFE / MAE
                if direction == "SHORT":
                    fav = (entry - l) / entry * 100.0
                    adv = (h - entry) / entry * 100.0
                else:
                    fav = (h - entry) / entry * 100.0
                    adv = (entry - l) / entry * 100.0
                mfe = max(mfe, fav)
                mae = max(mae, adv)

                # Проверка Stop-Loss и Take-Profit
                if direction == "SHORT":
                    # Сначала консервативно проверяем SL
                    if h >= initial_sl:
                        exit_bar = b
                        exit_price = initial_sl
                        accumulated_pnl = (entry - exit_price) / entry * 100.0
                        exit_reason = "sl"
                        break
                    elif l <= tp_price:
                        exit_bar = b
                        exit_price = tp_price
                        accumulated_pnl = (entry - exit_price) / entry * 100.0
                        exit_reason = "tp"
                        break
                else:  # LONG
                    if l <= initial_sl:
                        exit_bar = b
                        exit_price = initial_sl
                        accumulated_pnl = (exit_price - entry) / entry * 100.0
                        exit_reason = "sl"
                        break
                    elif h >= tp_price:
                        exit_bar = b
                        exit_price = tp_price
                        accumulated_pnl = (exit_price - entry) / entry * 100.0
                        exit_reason = "tp"
                        break

                # Таймаут удержания позиции
                if bars_since_fill >= self.timeout_bars:
                    exit_bar = b
                    exit_price = c
                    accumulated_pnl = (entry - c) / entry * 100.0 if direction == "SHORT" else (c - entry) / entry * 100.0
                    exit_reason = "timeout"
                    break

            bars_held = exit_bar - fill_rel_idx + 1

        # ── Финансовое моделирование ──
        # Риск $10 на сделку (1% от $1,000)
        risk_budget_usd = RISK_PER_TRADE_USD
        pos_size_usd = risk_budget_usd / (initial_risk_pct / 100.0) if initial_risk_pct > 0 else ACCOUNT_DEPOSIT_USD

        entry_fee = pos_size_usd * self.fee_rate
        gross_pnl_usd = pos_size_usd * (accumulated_pnl / 100.0)
        exit_fee = max(0.0, pos_size_usd + gross_pnl_usd) * self.fee_rate
        total_fee_usd = entry_fee + exit_fee

        net_pnl_usd = gross_pnl_usd - total_fee_usd
        net_pnl_r = net_pnl_usd / risk_budget_usd

        mfe_r = mfe / initial_risk_pct if initial_risk_pct > 0 else 0.0
        mae_r = mae / initial_risk_pct if initial_risk_pct > 0 else 0.0

        entry_time = times[fill_rel_idx] if fill_rel_idx < len(times) else decision.entry_time
        exit_time = times[exit_bar] if exit_bar < len(times) else None

        tp_hits = ["TP_1_3"] if exit_reason == "tp" else []

        return TradeResult(
            signal_id="",
            symbol="",
            interval="",
            direction=direction,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=round(entry, 6),
            exit_price=round(exit_price, 6),
            initial_sl=round(initial_sl, 6),
            final_pnl_pct=round(accumulated_pnl, 2),
            pnl_r=round(net_pnl_r, 2),
            exit_reason=exit_reason,
            bars_held=bars_held,
            mfe_pct=round(mfe, 2),
            mae_pct=round(mae, 2),
            mfe_r=round(mfe_r, 2),
            mae_r=round(mae_r, 2),
            position_size_usd=round(pos_size_usd, 2),
            fee_usd=round(total_fee_usd, 2),
            pnl_usd=round(gross_pnl_usd, 2),
            net_pnl_usd=round(net_pnl_usd, 2),
            slippage_tax_pct=0.0,
            tp_hits=tp_hits,
        )
