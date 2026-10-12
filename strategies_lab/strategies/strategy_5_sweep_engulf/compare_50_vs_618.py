"""
compare_50_vs_618.py — Сравнительный аудит двух ключевых конфигураций Стратегии 5:

Вариант 1 (Вход 50% от свечи поглощения):
  - 1A: Стоп в БУ на +2.0R, Тейк строго 3.0R.
  - 1B: Стоп в БУ на +2.0R, Тейк 3.0R ИЛИ выход по обратному сигналу (поглощение / RSI разворот).

Вариант 2 (Вход 61.8% от свечи поглощения — "Золотое сечение"):
  - 2A: Стоп в БУ на +2.0R, Тейк строго 3.0R.
  - 2B: Стоп в БУ на +2.0R, Тейк 3.0R ИЛИ выход по обратному сигналу (поглощение / RSI разворот).

Анализируются:
1. Точное количество спасённых стопов (сделки, взявшие 2R, закрывшиеся в БУ, которые иначе выбили бы полный SL).
2. Статистика по всем 13 инструментам: частота сделок, профит, винрейт (поиск лучших пар).
3. Помесячная динамика для каждого варианта: лучший месяц, худший месяц, винрейт по месяцам.
4. Моделирование сложного процента (пересчёт депозита каждый месяц с риском 1.0% от баланса).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from strategies_lab.strategies.strategy_5_sweep_engulf.advanced_research import (
    collect_raw_patterns,
    RawPatternSignal,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.patterns import calculate_net_1_to_3_tp
from strategies_lab.strategies.strategy_5_sweep_engulf.config import (
    FEE_RATE,
    ACCOUNT_DEPOSIT_USD,
    RISK_PER_TRADE_USD,
    RESULTS_DIR,
)

logger = logging.getLogger("Compare50vs618")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)


@dataclass
class TradeExecutionResult:
    sig_id: str
    regime: str
    symbol: str
    direction: str
    entry_time: Any
    entry_price: float
    initial_sl: float
    exit_time: Any
    exit_price: float
    exit_reason: str          # "tp", "be", "sl", "reverse_signal", "timeout"
    net_pnl_usd: float
    net_pnl_r: float
    gross_pnl_usd: float
    fee_usd: float
    mfe_r: float
    mae_r: float
    bars_held: int
    hit_2r: bool
    saved_from_sl: bool       # Была бы сделка полным стопом, если бы не БУ?


def simulate_strategy_setup(
    patterns: List[RawPatternSignal],
    retrace_ratio: float = 0.50,
    fill_window_bars: int = 2,
    target_r: float = 3.0,
    use_reverse_signal_exit: bool = False,
    fee_rate: float = FEE_RATE,
) -> List[TradeExecutionResult]:
    """
    Симулирует сделки для конкретного уровня отката и правил выхода.
    """
    results: List[TradeExecutionResult] = []

    for p in patterns:
        rng = p.s_range
        if rng <= 0 or len(p.future_candles) == 0:
            continue

        direction = p.direction
        if direction == "SHORT":
            entry_p = p.s_low + retrace_ratio * rng
            sl_p = p.s_high
        else:
            entry_p = p.s_high - retrace_ratio * rng
            sl_p = p.s_low

        risk_dist = abs(entry_p - sl_p)
        if risk_dist <= 0 or entry_p <= 0:
            continue

        risk_pct = (risk_dist / entry_p) * 100.0
        pos_usd = RISK_PER_TRADE_USD / (risk_pct / 100.0)

        highs = p.future_candles['high'].values
        lows = p.future_candles['low'].values
        opens = p.future_candles['open'].values
        closes = p.future_candles['close'].values
        times = p.future_candles.index
        rsi_arr = p.rsi_future
        n_bars = len(highs)

        # 1. Проверяем исполнение лимита в окне fill_window_bars
        fill_b = -1
        immediate_sl = False

        for b in range(min(fill_window_bars, n_bars)):
            h, l, o = highs[b], lows[b], opens[b]
            if direction == "SHORT":
                if o >= sl_p:
                    break  # гэп
                if h >= sl_p:
                    fill_b = b
                    immediate_sl = True
                    break
                if h >= entry_p:
                    fill_b = b
                    break
            else:
                if o <= sl_p:
                    break  # гэп
                if l <= sl_p:
                    fill_b = b
                    immediate_sl = True
                    break
                if l <= entry_p:
                    fill_b = b
                    break

        if fill_b == -1:
            continue  # Не заполнился лимит

        # 2. Уровни для сделки
        be_trigger_p, _, _ = calculate_net_1_to_3_tp(entry_p, sl_p, direction, fee_rate, target_net_r=2.0)
        target_tp_p, _, _ = calculate_net_1_to_3_tp(entry_p, sl_p, direction, fee_rate, target_net_r=target_r)
        be_price = entry_p * (1.0 - 2.0 * fee_rate) if direction == "SHORT" else entry_p * (1.0 + 2.0 * fee_rate)

        mfe_val = 0.0
        mae_val = 0.0

        if immediate_sl:
            exit_b = fill_b
            exit_p = sl_p
            exit_reason = "sl"
            hit_2r = False
            saved_from_sl = False
        else:
            current_sl = sl_p
            hit_2r = False
            exit_b = n_bars - 1
            exit_p = closes[-1]
            exit_reason = "timeout"
            saved_from_sl = False

            for b in range(fill_b, n_bars):
                h, l, o, c = highs[b], lows[b], opens[b], closes[b]
                fav = (entry_p - l) if direction == "SHORT" else (h - entry_p)
                adv = (h - entry_p) if direction == "SHORT" else (entry_p - l)
                mfe_val = max(mfe_val, fav)
                mae_val = max(mae_val, adv)

                # До взятия 2R
                if not hit_2r:
                    # Проверка начального стопа
                    if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                        exit_b = b
                        exit_p = current_sl
                        exit_reason = "sl"
                        break
                    # Проверка касания 2.0R -> переносим стоп в БУ!
                    if (direction == "SHORT" and l <= be_trigger_p) or (direction == "LONG" and h >= be_trigger_p):
                        hit_2r = True
                        current_sl = be_price

                # После взятия 2R (стоп в БУ)
                else:
                    # 1. Проверяем тейк 3.0R
                    if (direction == "SHORT" and l <= target_tp_p) or (direction == "LONG" and h >= target_tp_p):
                        exit_b = b
                        exit_p = target_tp_p
                        exit_reason = "tp"
                        break

                    # 2. Если включен выход по обратному сигналу
                    if use_reverse_signal_exit and b > fill_b:
                        prev_b = b - 1
                        p_op, p_cl = opens[prev_b], closes[prev_b]
                        p_top = max(p_op, p_cl)
                        p_bot = min(p_op, p_cl)

                        is_opp_engulf = False
                        if direction == "SHORT":
                            if c > o and c > p_top:  # Бычье поглощение
                                is_opp_engulf = True
                        else:
                            if c < o and c < p_bot:  # Медвежье поглощение
                                is_opp_engulf = True

                        is_rsi_rev = False
                        if b < len(rsi_arr):
                            r_val = rsi_arr[b]
                            if direction == "SHORT" and r_val <= 30.0:
                                is_rsi_rev = True
                            elif direction == "LONG" and r_val >= 70.0:
                                is_rsi_rev = True

                        if is_opp_engulf or is_rsi_rev:
                            exit_b = b
                            exit_p = c
                            exit_reason = "reverse_signal"
                            break

                    # 3. Проверяем выбивание БУ
                    if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                        exit_b = b
                        exit_p = current_sl
                        exit_reason = "be"
                        # Проверяем: пошла ли цена дальше к исходному SL?
                        # Если на последующих барах цена пробила исходный SL — мы СПАСЛИ сделку!
                        for rem_b in range(b, min(b + 30, n_bars)):
                            if (direction == "SHORT" and highs[rem_b] >= sl_p) or (direction == "LONG" and lows[rem_b] <= sl_p):
                                saved_from_sl = True
                                break
                        break

        pct = (entry_p - exit_p) / entry_p * 100.0 if direction == "SHORT" else (exit_p - entry_p) / entry_p * 100.0
        gross_usd = pos_usd * (pct / 100.0)
        tot_fee = (pos_usd + max(0.0, pos_usd + gross_usd)) * fee_rate
        net_usd = gross_usd - tot_fee
        net_r = net_usd / RISK_PER_TRADE_USD

        results.append(TradeExecutionResult(
            sig_id=p.sig_id,
            regime=p.regime,
            symbol=p.symbol,
            direction=p.direction,
            entry_time=times[fill_b],
            entry_price=round(entry_p, 6),
            initial_sl=round(sl_p, 6),
            exit_time=times[exit_b],
            exit_price=round(exit_p, 6),
            exit_reason=exit_reason,
            net_pnl_usd=round(net_usd, 2),
            net_pnl_r=round(net_r, 2),
            gross_pnl_usd=round(gross_usd, 2),
            fee_usd=round(tot_fee, 2),
            mfe_r=round(mfe_val / risk_dist, 2),
            mae_r=round(mae_val / risk_dist, 2),
            bars_held=exit_b - fill_b + 1,
            hit_2r=hit_2r,
            saved_from_sl=saved_from_sl,
        ))

    return results


def analyze_trades_dataset(trades: List[TradeExecutionResult]) -> Dict[str, Any]:
    """Считает сводные метрики по пулу сделок."""
    n = len(trades)
    if n == 0:
        return {}

    wins = [t for t in trades if t.net_pnl_usd > 0.5]
    bes = [t for t in trades if abs(t.net_pnl_usd) <= 0.5]
    losses = [t for t in trades if t.net_pnl_usd < -0.5]
    saved = [t for t in trades if t.saved_from_sl]

    gw = sum(t.net_pnl_usd for t in wins)
    gl = abs(sum(t.net_pnl_usd for t in losses))
    pf = (gw / gl) if gl > 0 else 99.0
    net_pnl = sum(t.net_pnl_usd for t in trades)
    avg_r = np.mean([t.net_pnl_r for t in trades])

    eq = np.cumsum([t.net_pnl_usd for t in trades])
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    max_dd = (np.max(dd) / ACCOUNT_DEPOSIT_USD * 100.0) if len(dd) > 0 else 0.0

    return {
        "total_trades": n,
        "wins": len(wins),
        "bes": len(bes),
        "losses": len(losses),
        "saved_from_sl": len(saved),
        "win_rate_pct": round(len(wins) / n * 100.0, 1),
        "be_rate_pct": round(len(bes) / n * 100.0, 1),
        "loss_rate_pct": round(len(losses) / n * 100.0, 1),
        "profit_factor": round(pf, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "net_ev_r": round(avg_r, 2),
        "max_dd_pct": round(max_dd, 1),
    }


def compute_instrument_breakdown(trades: List[TradeExecutionResult]) -> pd.DataFrame:
    """Группирует сделки по монетам и считает активность, винрейт и PnL."""
    rows = []
    df = pd.DataFrame([t.__dict__ for t in trades])
    for sym, grp in df.groupby("symbol"):
        n = len(grp)
        wins = len(grp[grp['net_pnl_usd'] > 0.5])
        bes = len(grp[grp['net_pnl_usd'].between(-0.5, 0.5)])
        losses = len(grp[grp['net_pnl_usd'] < -0.5])
        wr = (wins / n * 100.0) if n > 0 else 0.0
        pnl = grp['net_pnl_usd'].sum()
        gw = grp[grp['net_pnl_usd'] > 0.5]['net_pnl_usd'].sum()
        gl = abs(grp[grp['net_pnl_usd'] < -0.5]['net_pnl_usd'].sum())
        pf = (gw / gl) if gl > 0 else 99.0

        rows.append({
            "symbol": sym,
            "trades": n,
            "wins": wins,
            "bes": bes,
            "losses": losses,
            "win_rate_pct": round(wr, 1),
            "net_pnl_usd": round(pnl, 2),
            "profit_factor": round(pf, 2),
        })

    res = pd.DataFrame(rows)
    res.sort_values(by="trades", ascending=False, inplace=True)
    return res


def compute_monthly_compound(
    trades: List[TradeExecutionResult],
    initial_deposit: float = 1000.0,
    risk_rate: float = 0.01,
) -> pd.DataFrame:
    """Считает помесячную разбивку и сложный процент."""
    df = pd.DataFrame([t.__dict__ for t in trades])
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df.sort_values('entry_time', inplace=True)
    df['month'] = df['entry_time'].dt.to_period('M')

    monthly_rows = []
    compounded_dep = initial_deposit

    for m_period, grp in df.groupby('month'):
        n = len(grp)
        wins = len(grp[grp['net_pnl_usd'] > 0.5])
        bes = len(grp[grp['net_pnl_usd'].between(-0.5, 0.5)])
        losses = len(grp[grp['net_pnl_usd'] < -0.5])
        wr = (wins / n * 100.0) if n > 0 else 0.0
        sum_r = grp['net_pnl_r'].sum()

        fix_pnl = grp['net_pnl_usd'].sum()
        fix_roi = (fix_pnl / initial_deposit) * 100.0

        risk_usd = compounded_dep * risk_rate
        comp_pnl = sum_r * risk_usd
        start_dep = compounded_dep
        compounded_dep += comp_pnl
        comp_roi = (comp_pnl / start_dep) * 100.0 if start_dep > 0 else 0.0

        monthly_rows.append({
            "month": str(m_period),
            "trades": n,
            "wins": wins,
            "bes": bes,
            "losses": losses,
            "win_rate_pct": round(wr, 1),
            "sum_r": round(sum_r, 2),
            "fixed_pnl_usd": round(fix_pnl, 2),
            "fixed_roi_pct": round(fix_roi, 1),
            "compound_start_usd": round(start_dep, 2),
            "compound_pnl_usd": round(comp_pnl, 2),
            "compound_end_usd": round(compounded_dep, 2),
            "compound_roi_pct": round(comp_roi, 1),
        })

    return pd.DataFrame(monthly_rows)
