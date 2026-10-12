"""
advanced_research.py — Глубокое исследование и оптимизация Стратегии 5:
1. Оптимизация глубины отката входа: 38.2% vs 50.0% vs 61.8% vs 68.0% vs 75.0%
2. Тестирование БУ на 2R и удержание до 3R, 4R, 5R, 10R
3. Тройное деление позиции (33% @ 2R + БУ, 33% @ 4R, 34% @ 8R / Runner)
4. Помесячная статистика: сделок в месяц, винрейт, прибыль в $
5. Моделирование сложного процента (пересчёт депозита каждый месяц с риском 1%).
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

from signal_scanner import calculate_rsi_wilder
from strategies_lab.engine import load_all_market_data, detect_all_signals_in_history
from strategies_lab.strategies.strategy_5_sweep_engulf.patterns import (
    calculate_net_1_to_3_tp,
    SweepEngulfPattern,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.config import (
    FEE_RATE,
    ACCOUNT_DEPOSIT_USD,
    RISK_PER_TRADE_USD,
    MIN_ELLIOTT_SCORE,
    RESULTS_DIR,
)

logger = logging.getLogger("AdvancedResearch")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)

DATASETS_BASE = Path("c:/Users/user/Desktop/данные для бектестов")

HISTORICAL_REGIMES = [
    ("1_Bull_Run_2020_2021", "🚀 Бычий рынок (2020-2021)"),
    ("2_Crypto_Winter_2022", "🥶 Крипто-зима / Крах FTX (2022)"),
    ("4_ETF_Rally_2023_2024", "🏢 Институциональный рост ETF (2023-2024)"),
    ("8_Mega_Bull_Run_2025", "🏔️ Мега-буллран (2025)"),
    ("9_Macro_Bear_Market_2026", "🩸 Медвежий макро-тренд (2025-2026)"),
]

TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT",
    "SUIUSDT", "AAVEUSDT", "1000PEPEUSDT",
]


@dataclass
class RawPatternSignal:
    """Обнаруженный свечной паттерн до применения конкретного уровня входа."""
    sig_id: str
    regime: str
    symbol: str
    direction: str
    w5_bar: int
    s_rel: int
    s_high: float
    s_low: float
    s_open: float
    s_close: float
    s_range: float
    future_candles: pd.DataFrame
    rsi_future: np.ndarray


def collect_raw_patterns() -> List[RawPatternSignal]:
    """Сканирует все эпохи и находит все свечи свипа и поглощения."""
    all_patterns = []
    pid = 0

    for reg_dir, reg_name in HISTORICAL_REGIMES:
        p_dir = DATASETS_BASE / reg_dir
        if not p_dir.exists():
            continue
        logger.info(f"⏳ Сбор паттернов: {reg_name}...")
        mdata = load_all_market_data(data_dir=p_dir, symbols=TOP_SYMBOLS, interval="15m")
        sigs = detect_all_signals_in_history(mdata, interval="15m", min_score=MIN_ELLIOTT_SCORE)

        for sig in sigs:
            full_df = mdata[sig.symbol]["main"]
            w5_bar = sig.w5_bar
            future_slice = full_df.iloc[w5_bar + 1 : w5_bar + 1 + 15]
            if len(future_slice) == 0:
                continue

            # Проверка Случай 1 (W5)
            s_cand = None
            s_rel = -99
            if len(sig.raw_df) >= 2:
                prev_c = sig.raw_df.iloc[-2]
                w5_c = sig.raw_df.iloc[-1]
                w5_h, w5_l = float(w5_c['high']), float(w5_c['low'])
                w5_o, w5_c_p = float(w5_c['open']), float(w5_c['close'])
                p_h, p_l = float(prev_c['high']), float(prev_c['low'])
                p_o, p_c_p = float(prev_c['open']), float(prev_c['close'])
                p_bt, p_bb = max(p_o, p_c_p), min(p_o, p_c_p)

                if sig.direction == "SHORT":
                    if w5_h > p_h and w5_c_p < w5_o and w5_c_p < p_bb:
                        s_cand = w5_c
                        s_rel = -1
                else:
                    if w5_l < p_l and w5_c_p > w5_o and w5_c_p > p_bt:
                        s_cand = w5_c
                        s_rel = -1

            # Проверка Случай 2 (после W5)
            if s_cand is None:
                w5_c = sig.raw_df.iloc[-1]
                w5_h, w5_l = float(w5_c['high']), float(w5_c['low'])
                for i in range(min(3, len(future_slice))):
                    c = future_slice.iloc[i]
                    ch, cl, co, cc = float(c['high']), float(c['low']), float(c['open']), float(c['close'])
                    if sig.direction == "SHORT":
                        if ch > w5_h and cc < co and cc < float(w5_c['close']):
                            s_cand = c
                            s_rel = i
                            break
                    else:
                        if cl < w5_l and cc > co and cc > float(w5_c['close']):
                            s_cand = c
                            s_rel = i
                            break

            if s_cand is not None:
                pid += 1
                abs_s = w5_bar if s_rel == -1 else w5_bar + 1 + s_rel
                exec_candles = full_df.iloc[abs_s + 1 : abs_s + 1 + 150].copy()
                rsi_period = 7
                rsi_full = calculate_rsi_wilder(full_df['close'].values.astype(np.float64), period=rsi_period)
                rsi_exec = rsi_full[abs_s + 1 : abs_s + 1 + 150]

                sh = float(s_cand['high'])
                sl = float(s_cand['low'])
                so = float(s_cand['open'])
                sc = float(s_cand['close'])
                s_range = sh - sl

                all_patterns.append(RawPatternSignal(
                    sig_id=f"P{pid:04d}_{sig.symbol}_{sig.direction}_{abs_s}",
                    regime=reg_dir,
                    symbol=sig.symbol,
                    direction=sig.direction,
                    w5_bar=w5_bar,
                    s_rel=s_rel,
                    s_high=sh,
                    s_low=sl,
                    s_open=so,
                    s_close=sc,
                    s_range=s_range,
                    future_candles=exec_candles,
                    rsi_future=rsi_exec,
                ))

    logger.info(f"🏆 Всего найдено свечных паттернов S: {len(all_patterns)}")
    return all_patterns


def evaluate_retrace_level(
    patterns: List[RawPatternSignal],
    retrace_ratio: float,
    fill_window_bars: int = 2,
    target_r: float = 3.0,
    fee_rate: float = FEE_RATE,
) -> Dict[str, Any]:
    """
    Тестирует процент отката свечи S (например 0.382, 0.50, 0.618, 0.68).
    - Для SHORT: откат вверх от Low: entry = Low + retrace_ratio * Range, SL = High.
    - Для LONG: откат вниз от High: entry = High - retrace_ratio * Range, SL = Low.
    """
    filled_trades = []

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

        # Проверка лимита в пределах fill_window_bars
        highs = p.future_candles['high'].values
        lows = p.future_candles['low'].values
        opens = p.future_candles['open'].values
        times = p.future_candles.index
        n_bars = len(highs)

        fill_bar = -1
        immediate_sl = False

        for b in range(min(fill_window_bars, n_bars)):
            h, l, o = highs[b], lows[b], opens[b]
            if direction == "SHORT":
                if o >= sl_p:
                    break  # гэп
                if h >= sl_p:
                    fill_bar = b
                    immediate_sl = True
                    break
                if h >= entry_p:
                    fill_bar = b
                    break
            else:
                if o <= sl_p:
                    break  # гэп
                if l <= sl_p:
                    fill_bar = b
                    immediate_sl = True
                    break
                if l <= entry_p:
                    fill_bar = b
                    break

        if fill_bar == -1:
            continue  # Лимит не исполнен

        # Сделка исполнена! Симулируем фиксированный тейк target_r
        tp_price, _, _ = calculate_net_1_to_3_tp(entry_p, sl_p, direction, fee_rate, target_r)
        pos_usd = RISK_PER_TRADE_USD / (risk_pct / 100.0)

        # Выход
        exit_p = entry_p
        exit_reason = "timeout"
        exit_bar = n_bars - 1

        if immediate_sl:
            exit_p = sl_p
            exit_reason = "sl"
            exit_bar = fill_bar
        else:
            for b in range(fill_bar, n_bars):
                h, l = highs[b], lows[b]
                if direction == "SHORT":
                    if h >= sl_p:
                        exit_p = sl_p
                        exit_reason = "sl"
                        exit_bar = b
                        break
                    elif l <= tp_price:
                        exit_p = tp_price
                        exit_reason = "tp"
                        exit_bar = b
                        break
                else:
                    if l <= sl_p:
                        exit_p = sl_p
                        exit_reason = "sl"
                        exit_bar = b
                        break
                    elif h >= tp_price:
                        exit_p = tp_price
                        exit_reason = "tp"
                        exit_bar = b
                        break

        pct = (entry_p - exit_p) / entry_p * 100.0 if direction == "SHORT" else (exit_p - entry_p) / entry_p * 100.0
        gross_usd = pos_usd * (pct / 100.0)
        tot_fee = (pos_usd + max(0.0, pos_usd + gross_usd)) * fee_rate
        net_usd = gross_usd - tot_fee
        net_r = net_usd / RISK_PER_TRADE_USD

        filled_trades.append({
            "sig_id": p.sig_id,
            "regime": p.regime,
            "entry_time": times[fill_bar],
            "risk_pct": risk_pct,
            "net_pnl_usd": net_usd,
            "net_pnl_r": net_r,
            "exit_reason": exit_reason,
            "fee_usd": tot_fee,
        })

    # Сводная статистика
    tot = len(patterns)
    filled_cnt = len(filled_trades)
    fill_rate = (filled_cnt / tot * 100.0) if tot > 0 else 0.0

    wins = [t for t in filled_trades if t['net_pnl_usd'] > 0]
    losses = [t for t in filled_trades if t['net_pnl_usd'] < 0]
    wr = (len(wins) / filled_cnt * 100.0) if filled_cnt > 0 else 0.0
    gw = sum(t['net_pnl_usd'] for t in wins)
    gl = abs(sum(t['net_pnl_usd'] for t in losses))
    pf = (gw / gl) if gl > 0 else 0.0
    net_pnl = sum(t['net_pnl_usd'] for t in filled_trades)
    avg_r = np.mean([t['net_pnl_r'] for t in filled_trades]) if filled_cnt > 0 else 0.0
    avg_risk_pct = np.mean([t['risk_pct'] for t in filled_trades]) if filled_cnt > 0 else 0.0

    eq = np.cumsum([t['net_pnl_usd'] for t in filled_trades])
    peak = np.maximum.accumulate(eq) if len(eq) > 0 else np.array([0])
    dd = (peak - eq) if len(eq) > 0 else np.array([0])
    max_dd = (np.max(dd) / ACCOUNT_DEPOSIT_USD * 100.0) if len(dd) > 0 else 0.0

    return {
        "retrace_pct": round(retrace_ratio * 100.0, 1),
        "total_patterns": tot,
        "filled_trades": filled_cnt,
        "fill_rate_pct": round(fill_rate, 1),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "net_ev_r": round(avg_r, 2),
        "avg_risk_pct": round(avg_risk_pct, 2),
        "max_dd_pct": round(max_dd, 1),
        "trades": filled_trades,
    }


def simulate_breakeven_and_runners(
    patterns: List[RawPatternSignal],
    retrace_ratio: float = 0.50,
    fill_window_bars: int = 2,
    fee_rate: float = FEE_RATE,
) -> Dict[str, Any]:
    """
    Симулирует стратегии:
    1. Перенос стопа в БУ при +2.0R, удержание до 3R, 4R, 5R, 10R.
    2. Тройное деление: 33% @ 2R + БУ, 33% @ 4R, 34% @ 8R (или Runner).
    """
    # Сначала исполняем 50% лимит
    eval_base = evaluate_retrace_level(patterns, retrace_ratio=retrace_ratio, fill_window_bars=fill_window_bars)
    raw_filled_sigs = {t['sig_id']: t for t in eval_base['trades']}

    # Готовим список сделок для сложных сценариев
    runner_configs = [
        ("БУ на 2R -> Цель 3.0R", 3.0),
        ("БУ на 2R -> Цель 4.0R", 4.0),
        ("БУ на 2R -> Цель 5.0R", 5.0),
        ("БУ на 2R -> Цель 10.0R", 10.0),
    ]

    runner_results = {}

    for name, target_r in runner_configs:
        outcomes = []
        for p in patterns:
            if p.sig_id not in raw_filled_sigs:
                continue

            direction = p.direction
            rng = p.s_range
            entry_p = p.s_low + retrace_ratio * rng if direction == "SHORT" else p.s_high - retrace_ratio * rng
            sl_p = p.s_high if direction == "SHORT" else p.s_low
            risk_dist = abs(entry_p - sl_p)
            risk_pct = (risk_dist / entry_p) * 100.0
            pos_usd = RISK_PER_TRADE_USD / (risk_pct / 100.0)

            be_trigger_p, _, _ = calculate_net_1_to_3_tp(entry_p, sl_p, direction, fee_rate, target_net_r=2.0)
            target_tp_p, _, _ = calculate_net_1_to_3_tp(entry_p, sl_p, direction, fee_rate, target_net_r=target_r)
            be_price = entry_p * (1.0 - 2.0 * fee_rate) if direction == "SHORT" else entry_p * (1.0 + 2.0 * fee_rate)

            highs = p.future_candles['high'].values
            lows = p.future_candles['low'].values
            closes = p.future_candles['close'].values
            n_bars = len(highs)

            # Найдём бар входа
            fill_b = 0
            for b in range(min(fill_window_bars, n_bars)):
                if (direction == "SHORT" and highs[b] >= entry_p) or (direction == "LONG" and lows[b] <= entry_p):
                    fill_b = b
                    break

            current_sl = sl_p
            be_hit = False
            exit_p = entry_p
            exit_reason = "timeout"

            for b in range(fill_b, n_bars):
                h, l = highs[b], lows[b]
                if not be_hit:
                    # Проверяем начальный SL
                    if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                        exit_p = current_sl
                        exit_reason = "sl"
                        break
                    # Проверяем триггер 2R для переноса в БУ
                    if (direction == "SHORT" and l <= be_trigger_p) or (direction == "LONG" and h >= be_trigger_p):
                        be_hit = True
                        current_sl = be_price
                else:
                    # Проверяем выбивание БУ
                    if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                        exit_p = current_sl
                        exit_reason = "be"
                        break
                    # Проверяем финальный TP
                    if (direction == "SHORT" and l <= target_tp_p) or (direction == "LONG" and h >= target_tp_p):
                        exit_p = target_tp_p
                        exit_reason = "tp"
                        break

            pct = (entry_p - exit_p) / entry_p * 100.0 if direction == "SHORT" else (exit_p - entry_p) / entry_p * 100.0
            gross_usd = pos_usd * (pct / 100.0)
            tot_fee = (pos_usd + max(0.0, pos_usd + gross_usd)) * fee_rate
            net_usd = gross_usd - tot_fee
            net_r = net_usd / RISK_PER_TRADE_USD

            outcomes.append({
                "net_usd": net_usd,
                "net_r": net_r,
                "exit_reason": exit_reason,
            })

        wins = [o for o in outcomes if o['net_usd'] > 0.5]
        bes = [o for o in outcomes if abs(o['net_usd']) <= 0.5]
        losses = [o for o in outcomes if o['net_usd'] < -0.5]
        n_tot = len(outcomes)
        wr = (len(wins) / n_tot * 100.0) if n_tot > 0 else 0.0
        be_pct = (len(bes) / n_tot * 100.0) if n_tot > 0 else 0.0
        loss_pct = (len(losses) / n_tot * 100.0) if n_tot > 0 else 0.0

        gw = sum(o['net_usd'] for o in wins)
        gl = abs(sum(o['net_usd'] for o in losses))
        pf = (gw / gl) if gl > 0 else 0.0
        pnl = sum(o['net_usd'] for o in outcomes)
        ev_r = np.mean([o['net_r'] for o in outcomes]) if n_tot > 0 else 0.0

        eq = np.cumsum([o['net_usd'] for o in outcomes])
        peak = np.maximum.accumulate(eq) if len(eq) > 0 else np.array([0])
        dd = (peak - eq) if len(eq) > 0 else np.array([0])
        max_dd = (np.max(dd) / ACCOUNT_DEPOSIT_USD * 100.0) if len(dd) > 0 else 0.0

        runner_results[name] = {
            "total_trades": n_tot,
            "wins": len(wins),
            "bes": len(bes),
            "losses": len(losses),
            "win_rate_pct": round(wr, 1),
            "be_rate_pct": round(be_pct, 1),
            "loss_rate_pct": round(loss_pct, 1),
            "profit_factor": round(pf, 2),
            "net_pnl_usd": round(pnl, 2),
            "net_ev_r": round(ev_r, 2),
            "max_dd_pct": round(max_dd, 1),
        }

    # ── Тройное деление: 33% @ 2R (БУ), 33% @ 4R, 34% @ 8R ──
    triple_outcomes = []
    for p in patterns:
        if p.sig_id not in raw_filled_sigs:
            continue

        direction = p.direction
        rng = p.s_range
        entry_p = p.s_low + retrace_ratio * rng if direction == "SHORT" else p.s_high - retrace_ratio * rng
        sl_p = p.s_high if direction == "SHORT" else p.s_low
        risk_dist = abs(entry_p - sl_p)
        risk_pct = (risk_dist / entry_p) * 100.0
        pos_usd = RISK_PER_TRADE_USD / (risk_pct / 100.0)

        tp1_p, _, _ = calculate_net_1_to_3_tp(entry_p, sl_p, direction, fee_rate, target_net_r=2.0)
        tp2_p, _, _ = calculate_net_1_to_3_tp(entry_p, sl_p, direction, fee_rate, target_net_r=4.0)
        tp3_p, _, _ = calculate_net_1_to_3_tp(entry_p, sl_p, direction, fee_rate, target_net_r=8.0)
        be_p = entry_p * (1.0 - 2.0 * fee_rate) if direction == "SHORT" else entry_p * (1.0 + 2.0 * fee_rate)

        highs = p.future_candles['high'].values
        lows = p.future_candles['low'].values
        closes = p.future_candles['close'].values
        n_bars = len(highs)

        fill_b = 0
        for b in range(min(fill_window_bars, n_bars)):
            if (direction == "SHORT" and highs[b] >= entry_p) or (direction == "LONG" and lows[b] <= entry_p):
                fill_b = b
                break

        tp1_hit, tp2_hit, tp3_hit = False, False, False
        p1_exit_p, p2_exit_p, p3_exit_p = entry_p, entry_p, entry_p
        current_sl = sl_p

        for b in range(fill_b, n_bars):
            h, l = highs[b], lows[b]

            if not tp1_hit:
                if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                    p1_exit_p = p2_exit_p = p3_exit_p = current_sl
                    break
                if (direction == "SHORT" and l <= tp1_p) or (direction == "LONG" and h >= tp1_p):
                    tp1_hit = True
                    p1_exit_p = tp1_p
                    current_sl = be_p  # Переносим стоп в БУ на оставшиеся 67%!

            if tp1_hit and not tp2_hit:
                if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                    p2_exit_p = p3_exit_p = current_sl
                    break
                if (direction == "SHORT" and l <= tp2_p) or (direction == "LONG" and h >= tp2_p):
                    tp2_hit = True
                    p2_exit_p = tp2_p
                    current_sl = tp1_p  # Подтягиваем трейлинг стоп на +2R!

            if tp2_hit and not tp3_hit:
                if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                    p3_exit_p = current_sl
                    break
                if (direction == "SHORT" and l <= tp3_p) or (direction == "LONG" and h >= tp3_p):
                    tp3_hit = True
                    p3_exit_p = tp3_p
                    break

        # Взвешенный PnL (33% p1, 33% p2, 34% p3)
        w1, w2, w3 = 0.33, 0.33, 0.34
        calc_pct = lambda ex: (entry_p - ex)/entry_p*100.0 if direction == "SHORT" else (ex - entry_p)/entry_p*100.0
        p1_gross = (pos_usd * w1) * (calc_pct(p1_exit_p) / 100.0)
        p2_gross = (pos_usd * w2) * (calc_pct(p2_exit_p) / 100.0)
        p3_gross = (pos_usd * w3) * (calc_pct(p3_exit_p) / 100.0)
        tot_gross = p1_gross + p2_gross + p3_gross
        tot_fee = (pos_usd + max(0.0, pos_usd + tot_gross)) * fee_rate
        net_usd = tot_gross - tot_fee
        triple_outcomes.append({"net_usd": net_usd, "net_r": net_usd / RISK_PER_TRADE_USD})

    t_wins = [o for o in triple_outcomes if o['net_usd'] > 0.5]
    t_bes = [o for o in triple_outcomes if abs(o['net_usd']) <= 0.5]
    t_losses = [o for o in triple_outcomes if o['net_usd'] < -0.5]
    t_tot = len(triple_outcomes)
    t_gw = sum(o['net_usd'] for o in t_wins)
    t_gl = abs(sum(o['net_usd'] for o in t_losses))
    t_pf = (t_gw / t_gl) if t_gl > 0 else 0.0

    eq_t = np.cumsum([o['net_usd'] for o in triple_outcomes])
    peak_t = np.maximum.accumulate(eq_t) if len(eq_t) > 0 else np.array([0])
    dd_t = (peak_t - eq_t) if len(eq_t) > 0 else np.array([0])
    max_dd_t = (np.max(dd_t) / ACCOUNT_DEPOSIT_USD * 100.0) if len(dd_t) > 0 else 0.0

    runner_results["Тройной сплит (33%@2R+БУ, 33%@4R, 34%@8R)"] = {
        "total_trades": t_tot,
        "wins": len(t_wins),
        "bes": len(t_bes),
        "losses": len(t_losses),
        "win_rate_pct": round(len(t_wins) / t_tot * 100.0, 1) if t_tot > 0 else 0.0,
        "be_rate_pct": round(len(t_bes) / t_tot * 100.0, 1) if t_tot > 0 else 0.0,
        "loss_rate_pct": round(len(t_losses) / t_tot * 100.0, 1) if t_tot > 0 else 0.0,
        "profit_factor": round(t_pf, 2),
        "net_pnl_usd": round(sum(o['net_usd'] for o in triple_outcomes), 2),
        "net_ev_r": round(np.mean([o['net_r'] for o in triple_outcomes]), 2) if t_tot > 0 else 0.0,
        "max_dd_pct": round(max_dd_t, 1),
    }

    return runner_results


def run_monthly_and_compound_simulation(
    trades: List[Dict[str, Any]],
    initial_deposit: float = 1000.0,
    risk_pct_monthly: float = 0.01,
) -> pd.DataFrame:
    """
    Группирует сделки по календарным месяцам и рассчитывает:
    1. Результаты без сложного процента (депозит $1,000, $10 на сделку).
    2. Результаты СО СЛОЖНЫМ ПРОЦЕНТОМ (депозит увеличивается на прибыль прошлого месяца).
    """
    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df.sort_values('entry_time', inplace=True)
    df['month'] = df['entry_time'].dt.to_period('M')

    monthly_rows = []
    compounded_deposit = initial_deposit

    for month_period, grp in df.groupby('month'):
        n_trades = len(grp)
        wins = len(grp[grp['net_pnl_usd'] > 0])
        losses = len(grp[grp['net_pnl_usd'] <= 0])
        wr = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

        # Сумма R за месяц
        sum_r = grp['net_pnl_r'].sum()
        fixed_pnl_usd = grp['net_pnl_usd'].sum()
        fixed_roi_pct = (fixed_pnl_usd / initial_deposit) * 100.0

        # Сложный процент: риск на сделку в этом месяце = 1% от compounded_deposit
        risk_per_trade_compound = compounded_deposit * risk_pct_monthly
        compound_pnl_usd = sum_r * risk_per_trade_compound
        dep_start_month = compounded_deposit
        compounded_deposit += compound_pnl_usd
        compound_roi_pct = (compound_pnl_usd / dep_start_month) * 100.0

        monthly_rows.append({
            "month": str(month_period),
            "trades": n_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wr, 1),
            "sum_r": round(sum_r, 2),
            "fixed_pnl_usd": round(fixed_pnl_usd, 2),
            "fixed_roi_pct": round(fixed_roi_pct, 1),
            "compound_start_usd": round(dep_start_month, 2),
            "compound_pnl_usd": round(compound_pnl_usd, 2),
            "compound_end_usd": round(compounded_deposit, 2),
            "compound_roi_pct": round(compound_roi_pct, 1),
        })

    return pd.DataFrame(monthly_rows)
