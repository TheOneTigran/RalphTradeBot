"""
prod_replication_audit.py — Полная репликация продакшен-движка RalphTradeBot (без ИИ)
с проверкой стратегии со свипом, поглощением и входом от 50% свечи S.

Параметры:
- Логика детекции импульса и дивергенции: 100% точная копия live_scanner.py + elliott_detector.py (Score >= 65).
- Выявление свечи со свипом и поглощением (Candle S): сигнал на закрытии свечи S.
- Лимитный ордер: строго 50% диапазона свечи S (окно 1-2 свечи).
- Стоп-лосс: строго кончик тени свечи S (без буфера).
- БУ: при достижении +2.0R Net перенос в Безубыток (с запасом на комиссии 0.05% in + 0.05% out).
- Тейк-профит: строго +3.0R Net.
- Данные: 2025–2026 непрерывные (15 монет, 15m ТФ).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from elliott_detector import detect_elliott_impulse, ElliottImpulseResult
from strategies_lab.strategies.strategy_5_sweep_engulf.patterns import (
    detect_sweep_engulf_candle,
    check_limit_fill,
    calculate_net_1_to_3_tp,
    SweepEngulfPattern,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.chart_plotter import plot_sweep_engulf_trade
from strategies_lab.strategies.base_strategy import TradeSignal, TradeResult

logger = logging.getLogger("ProdReplication")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)

DATA_DIR = Path(r"C:\Users\user\Desktop\данные для бектестов\2025_2026_Full_Top_Coins")
ARTIFACTS_DIR = Path(r"C:\Users\user\.gemini\antigravity-ide\brain\2249d323-d907-4b4e-ae7d-74c8bae47d0a")

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'NEARUSDT',
    'SUIUSDT', 'AAVEUSDT', 'APTUSDT', 'DOTUSDT', 'RENDERUSDT',
]

FEE_RATE = 0.0005  # 0.05% taker / maker
MIN_ALGO_SCORE = 65.0


@dataclass
class ProdCandidate:
    symbol: str
    direction: str
    p1: int
    p2: int
    p1_price: float
    p2_price: float
    p1_rsi: float
    p2_rsi: float
    algo_res: ElliottImpulseResult


def detect_prod_candidates(
    df: pd.DataFrame,
    symbol: str,
    interval: str = "15m",
) -> List[ProdCandidate]:
    """
    Точная копия логики scan_live_pair из live_scanner.py (без AI).
    Использует rolling 120-bar sliding window либо расчет экстремумов с order=3.
    """
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    open_p = df['open'].values.astype(np.float64)
    n = len(df)

    rsi_len = 7 if interval in ("5m", "15m") else 14
    rsi = calculate_rsi_wilder(close, period=rsi_len)

    rsi_ob = 70.0
    rsi_os = 30.0
    pivot_order = 3
    min_pivot_dist = 4
    max_pivot_dist = 45

    high_pivots = argrelextrema(high, np.greater_equal, order=pivot_order)[0]
    low_pivots = argrelextrema(low, np.less_equal, order=pivot_order)[0]

    candidates: List[ProdCandidate] = []
    seen_keys = set()

    # 1. Bearish Signals (SHORT)
    if len(high_pivots) >= 2:
        for i in range(1, len(high_pivots)):
            p1 = int(high_pivots[i - 1])
            p2 = int(high_pivots[i])

            if p2 >= n - 15:
                continue

            dist = p2 - p1
            if dist < min_pivot_dist or dist > max_pivot_dist:
                continue
            if high[p2] <= high[p1]:
                continue

            r1, r2 = float(rsi[p1]), float(rsi[p2])
            if r2 >= r1 or (r1 - r2) < 2.5:
                continue
            if r1 < 60.0 or r2 < 45.0:
                continue

            # Purity
            if len(high[p1 + 1 : p2]) > 0 and np.max(high[p1 + 1 : p2]) > high[p2]:
                continue
            if len(rsi[p1 + 1 : p2]) > 0 and np.min(rsi[p1 + 1 : p2]) < rsi_os:
                continue

            # Elliott Impulse Verification
            # В продакшене detect_elliott_impulse оценивает свечи до p2
            algo_res = detect_elliott_impulse(
                high=high[:p2 + 1],
                low=low[:p2 + 1],
                close=close[:p2 + 1],
                open_p=open_p[:p2 + 1],
                w3_bar=p1,
                w5_bar=p2,
                direction="SHORT",
                rsi=rsi[:p2 + 1],
            )

            if algo_res.is_valid and algo_res.score >= MIN_ALGO_SCORE:
                sig_key = f"{symbol}_SHORT_{p2}"
                if sig_key not in seen_keys:
                    seen_keys.add(sig_key)
                    candidates.append(
                        ProdCandidate(
                            symbol=symbol,
                            direction="SHORT",
                            p1=p1,
                            p2=p2,
                            p1_price=float(high[p1]),
                            p2_price=float(high[p2]),
                            p1_rsi=r1,
                            p2_rsi=r2,
                            algo_res=algo_res,
                        )
                    )

    # 2. Bullish Signals (LONG)
    if len(low_pivots) >= 2:
        for i in range(1, len(low_pivots)):
            p1 = int(low_pivots[i - 1])
            p2 = int(low_pivots[i])

            if p2 >= n - 15:
                continue

            dist = p2 - p1
            if dist < min_pivot_dist or dist > max_pivot_dist:
                continue
            if low[p2] >= low[p1]:
                continue

            r1, r2 = float(rsi[p1]), float(rsi[p2])
            if r2 <= r1 or (r2 - r1) < 2.5:
                continue
            if r1 > 40.0 or r2 > 55.0:
                continue

            # Purity
            if len(low[p1 + 1 : p2]) > 0 and np.min(low[p1 + 1 : p2]) < low[p2]:
                continue
            if len(rsi[p1 + 1 : p2]) > 0 and np.max(rsi[p1 + 1 : p2]) > rsi_ob:
                continue

            algo_res = detect_elliott_impulse(
                high=high[:p2 + 1],
                low=low[:p2 + 1],
                close=close[:p2 + 1],
                open_p=open_p[:p2 + 1],
                w3_bar=p1,
                w5_bar=p2,
                direction="LONG",
                rsi=rsi[:p2 + 1],
            )

            if algo_res.is_valid and algo_res.score >= MIN_ALGO_SCORE:
                sig_key = f"{symbol}_LONG_{p2}"
                if sig_key not in seen_keys:
                    seen_keys.add(sig_key)
                    candidates.append(
                        ProdCandidate(
                            symbol=symbol,
                            direction="LONG",
                            p1=p1,
                            p2=p2,
                            p1_price=float(low[p1]),
                            p2_price=float(low[p2]),
                            p1_rsi=r1,
                            p2_rsi=r2,
                            algo_res=algo_res,
                        )
                    )

    return candidates


@dataclass
class ExecutedTrade:
    trade_id: int
    symbol: str
    direction: str
    signal_bar: int
    s_bar: int
    entry_bar: int
    entry_time: pd.Timestamp
    entry_price: float
    initial_sl: float
    tp_price: float
    be_price: float
    exit_bar: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str          # "tp", "be", "sl", "timeout"
    net_pnl_usd: float
    net_pnl_r: float
    gross_pnl_usd: float
    fee_usd: float
    bars_held: int
    mfe_r: float
    mae_r: float
    hit_2r: bool
    saved_from_sl: bool
    candidate: ProdCandidate
    pattern: SweepEngulfPattern


def run_prod_audit() -> Tuple[List[ExecutedTrade], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    logger.info("Загрузка 15m данных 2025-2026...")
    mdata: Dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        p = DATA_DIR / f"{sym}_15m.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'open_time' in df.columns:
                df.index = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            elif 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'], utc=True)
            else:
                df.index = pd.to_datetime(df.iloc[:, 0], utc=True)
        mdata[sym] = df[['open', 'high', 'low', 'close', 'volume']]

    logger.info(f"Загружено монет: {len(mdata)}")

    all_candidates: List[ProdCandidate] = []
    for sym, df in mdata.items():
        cands = detect_prod_candidates(df, sym)
        all_candidates.extend(cands)
        logger.info(f"  {sym}: найдено {len(cands)} импульсов Эллиотта (Score >= 65, Prod criteria)")

    logger.info(f"Всего импульсов по всем монетам: {len(all_candidates)}")

    executed_trades: List[ExecutedTrade] = []
    trade_seq = 0
    s_count = 0
    filled_count = 0

    account_depo = 1000.0
    risk_usd = 10.0  # 1% от $1,000

    for cand in all_candidates:
        df = mdata[cand.symbol]
        p2 = cand.p2
        future_slice = df.iloc[p2 + 1 : p2 + 1 + 15]
        if len(future_slice) == 0:
            continue

        prev_c = df.iloc[p2 - 1] if p2 >= 1 else None
        pat = detect_sweep_engulf_candle(
            w5_candle=df.iloc[p2],
            future_candles=future_slice,
            direction=cand.direction,
            prev_candle=prev_c,
            retrace_ratio=0.50,
            fee_rate=FEE_RATE,
            target_net_r=3.0,
        )

        if not pat.found:
            continue
        s_count += 1

        # Проверка лимита на 50% свечи S строго в окне 1-2 бара
        fill_res = check_limit_fill(
            pattern=pat,
            future_candles=future_slice,
            direction=cand.direction,
            fill_window_bars=2,
        )

        if not fill_res.is_filled:
            continue
        filled_count += 1

        trade_seq += 1
        abs_fill_bar = p2 + 1 + fill_res.fill_rel_idx
        exec_candles = df.iloc[abs_fill_bar : abs_fill_bar + 160].copy()

        entry_p = pat.limit_entry_price
        sl_p = pat.sl_price
        direction = cand.direction
        risk_dist = abs(entry_p - sl_p)
        risk_pct = (risk_dist / entry_p) * 100.0 if entry_p > 0 else 1.0

        if risk_dist <= 0 or risk_pct < 0.15:
            continue

        pos_size_usd = risk_usd / (risk_dist / entry_p)
        qty = pos_size_usd / entry_p
        open_fee = pos_size_usd * FEE_RATE

        tp_p = pat.tp_price

        # Точный уровень безубытка (покрывающий обе комиссии: вход + выход)
        if direction == "SHORT":
            be_price = entry_p * (1.0 - 2.0 * FEE_RATE)
            trigger_2r = entry_p - 2.0 * risk_dist
        else:
            be_price = entry_p * (1.0 + 2.0 * FEE_RATE)
            trigger_2r = entry_p + 2.0 * risk_dist

        cur_sl = sl_p
        be_active = False
        hit_2r = False
        saved_from_sl = False

        exit_reason = "timeout"
        exit_p = entry_p
        exit_bar = len(exec_candles) - 1
        bars_held = len(exec_candles)

        max_fav = 0.0
        max_adv = 0.0

        for b_idx in range(len(exec_candles)):
            b_high = float(exec_candles['high'].iloc[b_idx])
            b_low = float(exec_candles['low'].iloc[b_idx])
            b_close = float(exec_candles['close'].iloc[b_idx])

            if direction == "SHORT":
                fav = entry_p - b_low
                adv = b_high - entry_p
                if fav > max_fav:
                    max_fav = fav
                if adv > max_adv:
                    max_adv = adv

                # Проверка +2R
                if not be_active and b_low <= trigger_2r:
                    be_active = True
                    hit_2r = True
                    cur_sl = be_price

                # Проверка TP (3R)
                if b_low <= tp_p:
                    exit_reason = "tp"
                    exit_p = tp_p
                    exit_bar = b_idx
                    bars_held = b_idx + 1
                    break

                # Проверка SL / БУ
                if b_high >= cur_sl:
                    if be_active:
                        exit_reason = "be"
                        exit_p = be_price
                        # Спасена ли от стопа? (если бы не БУ, ушла бы в SL)
                        if b_high >= sl_p:
                            saved_from_sl = True
                    else:
                        exit_reason = "sl"
                        exit_p = sl_p
                    exit_bar = b_idx
                    bars_held = b_idx + 1
                    break

            else:  # LONG
                fav = b_high - entry_p
                adv = entry_p - b_low
                if fav > max_fav:
                    max_fav = fav
                if adv > max_adv:
                    max_adv = adv

                if not be_active and b_high >= trigger_2r:
                    be_active = True
                    hit_2r = True
                    cur_sl = be_price

                if b_high >= tp_p:
                    exit_reason = "tp"
                    exit_p = tp_p
                    exit_bar = b_idx
                    bars_held = b_idx + 1
                    break

                if b_low <= cur_sl:
                    if be_active:
                        exit_reason = "be"
                        exit_p = be_price
                        if b_low <= sl_p:
                            saved_from_sl = True
                    else:
                        exit_reason = "sl"
                        exit_p = sl_p
                    exit_bar = b_idx
                    bars_held = b_idx + 1
                    break

        if exit_reason == "timeout":
            exit_p = float(exec_candles['close'].iloc[-1])
            exit_bar = len(exec_candles) - 1
            bars_held = len(exec_candles)

        exit_val = qty * exit_p
        close_fee = exit_val * FEE_RATE
        total_fee = open_fee + close_fee

        if direction == "SHORT":
            gross_pnl = qty * (entry_p - exit_p)
        else:
            gross_pnl = qty * (exit_p - entry_p)

        net_pnl = gross_pnl - total_fee
        net_r = net_pnl / risk_usd
        mfe_r = max_fav / risk_dist if risk_dist > 0 else 0.0
        mae_r = max_adv / risk_dist if risk_dist > 0 else 0.0

        s_bar_idx = p2 if pat.s_rel_idx == -1 else (p2 + 1 + pat.s_rel_idx)

        executed_trades.append(
            ExecutedTrade(
                trade_id=trade_seq,
                symbol=cand.symbol,
                direction=cand.direction,
                signal_bar=p2,
                s_bar=s_bar_idx,
                entry_bar=abs_fill_bar,
                entry_time=df.index[abs_fill_bar],
                entry_price=entry_p,
                initial_sl=sl_p,
                tp_price=tp_p,
                be_price=be_price,
                exit_bar=abs_fill_bar + exit_bar,
                exit_time=df.index[min(len(df) - 1, abs_fill_bar + exit_bar)],
                exit_price=exit_p,
                exit_reason=exit_reason,
                net_pnl_usd=net_pnl,
                net_pnl_r=net_r,
                gross_pnl_usd=gross_pnl,
                fee_usd=total_fee,
                bars_held=bars_held,
                mfe_r=mfe_r,
                mae_r=mae_r,
                hit_2r=hit_2r,
                saved_from_sl=saved_from_sl,
                candidate=cand,
                pattern=pat,
            )
        )

    logger.info(f"Сформировано паттернов S: {s_count}")
    logger.info(f"Исполнено лимитных входов: {len(executed_trades)} (Fill rate: {len(executed_trades)/s_count*100:.1f}%)")

    # Формируем сводные таблицы
    trade_dicts = []
    for t in executed_trades:
        trade_dicts.append({
            "id": t.trade_id,
            "symbol": t.symbol,
            "direction": t.direction,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "initial_sl": t.initial_sl,
            "tp_price": t.tp_price,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "net_pnl_usd": round(t.net_pnl_usd, 2),
            "net_pnl_r": round(t.net_pnl_r, 2),
            "fee_usd": round(t.fee_usd, 2),
            "bars_held": t.bars_held,
            "mfe_r": round(t.mfe_r, 2),
            "mae_r": round(t.mae_r, 2),
            "hit_2r": t.hit_2r,
            "saved_from_sl": t.saved_from_sl,
            "algo_score": t.candidate.algo_res.score,
        })
    df_trades = pd.DataFrame(trade_dicts)

    # Помесячная статистика
    df_trades['month'] = pd.to_datetime(df_trades['entry_time']).dt.to_period('M').astype(str)
    months_list = sorted(df_trades['month'].unique())

    monthly_rows = []
    compound_depo = 1000.0

    for m in months_list:
        m_df = df_trades[df_trades['month'] == m]
        n_trades = len(m_df)
        wins = len(m_df[m_df['exit_reason'] == "tp"])
        bes = len(m_df[m_df['exit_reason'] == "be"])
        sls = len(m_df[m_df['exit_reason'] == "sl"])
        to = len(m_df[m_df['exit_reason'] == "timeout"])
        wr = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

        r_sum = m_df['net_pnl_r'].sum()
        fix_pnl = m_df['net_pnl_usd'].sum()

        # Сложный процент: риск 1% от текущего депозита на начало месяца
        m_risk_unit = compound_depo * 0.01
        m_compound_pnl = r_sum * m_risk_unit
        compound_depo += m_compound_pnl
        growth_pct = (r_sum * 1.0)  # т.к. 1% риск

        monthly_rows.append({
            "month": m,
            "trades": n_trades,
            "wins": wins,
            "be": bes,
            "losses": sls,
            "timeout": to,
            "win_rate": round(wr, 1),
            "net_r": round(r_sum, 2),
            "fixed_pnl_usd": round(fix_pnl, 2),
            "compound_depo": round(compound_depo, 2),
            "compound_pnl": round(m_compound_pnl, 2),
            "growth_pct": round(growth_pct, 1),
        })
    df_monthly = pd.DataFrame(monthly_rows)

    # Статистика по монетам
    coin_rows = []
    for sym in SYMBOLS:
        c_df = df_trades[df_trades['symbol'] == sym]
        if len(c_df) == 0:
            continue
        n_t = len(c_df)
        w = len(c_df[c_df['exit_reason'] == "tp"])
        b = len(c_df[c_df['exit_reason'] == "be"])
        l = len(c_df[c_df['exit_reason'] == "sl"])
        wr = (w / n_t * 100.0) if n_t > 0 else 0.0
        pnl = c_df['net_pnl_usd'].sum()
        r_tot = c_df['net_pnl_r'].sum()
        gross_w = c_df[c_df['net_pnl_usd'] > 0]['net_pnl_usd'].sum()
        gross_l = abs(c_df[c_df['net_pnl_usd'] < 0]['net_pnl_usd'].sum())
        pf = (gross_w / gross_l) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0)

        coin_rows.append({
            "symbol": sym,
            "trades": n_t,
            "wins": w,
            "be": b,
            "losses": l,
            "win_rate": round(wr, 1),
            "profit_factor": round(pf, 2),
            "net_r": round(r_tot, 2),
            "net_pnl_usd": round(pnl, 2),
        })
    df_coins = pd.DataFrame(coin_rows).sort_values("net_pnl_usd", ascending=False)

    # KPIs
    total_trades = len(df_trades)
    wins_total = len(df_trades[df_trades['exit_reason'] == "tp"])
    be_total = len(df_trades[df_trades['exit_reason'] == "be"])
    sl_total = len(df_trades[df_trades['exit_reason'] == "sl"])
    saved_sl_total = df_trades['saved_from_sl'].sum()

    total_net_pnl = df_trades['net_pnl_usd'].sum()
    total_net_r = df_trades['net_pnl_r'].sum()
    total_fees = df_trades['fee_usd'].sum()
    gw = df_trades[df_trades['net_pnl_usd'] > 0]['net_pnl_usd'].sum()
    gl = abs(df_trades[df_trades['net_pnl_usd'] < 0]['net_pnl_usd'].sum())
    overall_pf = gw / gl if gl > 0 else 99.0

    # Max Drawdown
    cum_pnl = df_trades['net_pnl_usd'].cumsum()
    peak = cum_pnl.cummax()
    dd = peak - cum_pnl
    max_dd_usd = dd.max() if len(dd) > 0 else 0.0
    max_dd_pct = (max_dd_usd / 1000.0) * 100.0

    kpis = {
        "total_trades": total_trades,
        "s_candles_count": s_count,
        "fill_rate_pct": round(total_trades / s_count * 100.0, 1) if s_count > 0 else 0.0,
        "wins": wins_total,
        "be": be_total,
        "losses": sl_total,
        "saved_from_sl": int(saved_sl_total),
        "win_rate_pct": round(wins_total / total_trades * 100.0, 1) if total_trades > 0 else 0.0,
        "be_rate_pct": round(be_total / total_trades * 100.0, 1) if total_trades > 0 else 0.0,
        "loss_rate_pct": round(sl_total / total_trades * 100.0, 1) if total_trades > 0 else 0.0,
        "profit_factor": round(overall_pf, 2),
        "total_net_r": round(total_net_r, 2),
        "net_pnl_fixed_usd": round(total_net_pnl, 2),
        "compound_final_usd": round(compound_depo, 2),
        "compound_multiplier": round(compound_depo / 1000.0, 2),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "total_fees_paid": round(total_fees, 2),
        "net_ev_per_trade_r": round(total_net_r / total_trades, 2) if total_trades > 0 else 0.0,
    }

    return executed_trades, df_trades, df_monthly, df_coins, kpis, mdata


if __name__ == "__main__":
    trades, df_trades, df_monthly, df_coins, kpis, mdata = run_prod_audit()
    print("=== АУДИТ ПОЛНОЙ РЕПЛИКАЦИИ ПРОДАКШЕНА (БЕЗ ИИ) ===")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
