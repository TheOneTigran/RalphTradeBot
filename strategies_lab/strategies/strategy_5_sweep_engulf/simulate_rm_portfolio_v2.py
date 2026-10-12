"""
simulate_rm_portfolio_v2.py — Высокоточный и быстрый симулятор портфеля с риск-менеджментом:
- Кулдаун 6 свечей на одну монету.
- Максимум 3 или 4 одновременно открытые позиции на весь портфель.
- Строгий месячный лимит просадки 5% (с фиксацией блокировки до конца месяца).
- Устранено дублирование сигналов (строго 1 сигнал на свечу).
- Быстрое исполнение через numpy-массивы.
"""
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"c:\Users\user\Desktop\RalphTradeBot")
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from elliott_detector import detect_elliott_impulse

DATA_DIR = Path(r"C:\Users\user\Desktop\данные для бектестов\2025_2026_Full_Top_Coins")
RESULTS_DIR = BASE_DIR / "strategies_lab" / "strategies" / "strategy_5_sweep_engulf" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = RESULTS_DIR / "raw_signals_dedup_cache.pkl"

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'NEARUSDT',
    'SUIUSDT', 'AAVEUSDT', 'APTUSDT', 'DOTUSDT', 'RENDERUSDT',
]
FEE = 0.0005


@dataclass
class PendingOrder:
    symbol: str
    sym_idx: int
    direction: int  # +1 LONG, -1 SHORT
    signal_bar: int
    expire_bar: int
    entry_p: float
    sl_p: float
    tp_p: float
    be_p: float
    trig_2r: float
    risk_frac: float


@dataclass
class ActivePosition:
    trade_id: int
    symbol: str
    sym_idx: int
    direction: int
    entry_bar: int
    entry_p: float
    sl_p: float
    cur_sl: float
    tp_p: float
    be_p: float
    trig_2r: float
    risk_frac: float
    be_active: bool = False
    bars_held: int = 0


@dataclass
class ClosedTrade:
    trade_id: int
    symbol: str
    direction: str
    entry_bar: int
    exit_bar: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_p: float
    exit_p: float
    sl_p: float
    tp_p: float
    exit_reason: str
    net_r: float
    pnl_pct: float
    bars_held: int
    month: str


def load_all_market_data():
    mdata = {}
    for sym in SYMBOLS:
        p = DATA_DIR / f"{sym}_15m.csv"
        if not p.exists(): continue
        df = pd.read_csv(p)
        if 'open_time' in df.columns:
            df.index = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], utc=True)
        else:
            df.index = pd.to_datetime(df.iloc[:, 0], utc=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
        mdata[sym] = df
    return mdata


def extract_deduplicated_signals(mdata: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    if CACHE_FILE.exists():
        print(f"Загрузка сигналов из кэша: {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
            
    print("Генерация и дедупликация сигналов Model 2...")
    all_signals_dict = {}  # key: (sym, t) -> signal dict (keeps highest score)
    
    for sym in SYMBOLS:
        df = mdata[sym]
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_p = df['open'].values
        times = df.index
        n = len(df)
        
        rsi = calculate_rsi_wilder(close, period=7)
        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]
        
        # 1. SHORT
        for p1 in high_pivots:
            r1 = float(rsi[p1])
            if r1 < 60.0: continue
            
            for t in range(p1 + 5, min(n - 40, p1 + 35)):
                if high[t] <= high[p1]: continue
                if high[t] < np.max(high[p1+1 : t+1]): continue
                
                is_bearish = close[t] < open_p[t]
                prev_body_bottom = min(open_p[t-1], close[t-1])
                is_engulf = close[t] < prev_body_bottom
                if not (is_bearish and is_engulf): continue
                
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < 0.40: continue
                
                r_t = float(rsi[t])
                if r_t >= r1 - 2.5 or r_t < 45.0: continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:t + 1], low=low[:t + 1], close=close[:t + 1], open_p=open_p[:t + 1],
                    w3_bar=p1, w5_bar=t, direction="SHORT", rsi=rsi[:t + 1],
                )
                if not algo_res.is_valid or algo_res.score < 68: continue
                
                limit_entry = low[t] + 0.50 * c_range
                sl_p = high[t]
                risk_dist = sl_p - limit_entry
                if risk_dist <= 0: continue
                
                risk_frac = risk_dist / limit_entry
                reward_frac = (3.0 * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp_p = limit_entry - limit_entry * reward_frac
                trig_2r = limit_entry - 2.0 * risk_dist
                be_p = limit_entry * (1.0 - 2.0 * FEE)
                
                key = (sym, t)
                sig_data = {
                    "symbol": sym,
                    "direction": -1,  # SHORT
                    "dir_str": "SHORT",
                    "signal_time": times[t],
                    "signal_bar_idx": t,
                    "entry_p": limit_entry,
                    "sl_p": sl_p,
                    "tp_p": tp_p,
                    "be_p": be_p,
                    "trig_2r": trig_2r,
                    "risk_frac": risk_frac,
                    "score": algo_res.score,
                }
                if key not in all_signals_dict or algo_res.score > all_signals_dict[key]["score"]:
                    all_signals_dict[key] = sig_data
                break
        
        # 2. LONG
        for p1 in low_pivots:
            r1 = float(rsi[p1])
            if r1 > 40.0: continue
            
            for t in range(p1 + 5, min(n - 40, p1 + 35)):
                if low[t] >= low[p1]: continue
                if low[t] > np.min(low[p1+1 : t+1]): continue
                
                is_bullish = close[t] > open_p[t]
                prev_body_top = max(open_p[t-1], close[t-1])
                is_engulf = close[t] > prev_body_top
                if not (is_bullish and is_engulf): continue
                
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < 0.40: continue
                
                r_t = float(rsi[t])
                if r_t <= r1 + 2.5 or r_t > 55.0: continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:t + 1], low=low[:t + 1], close=close[:t + 1], open_p=open_p[:t + 1],
                    w3_bar=p1, w5_bar=t, direction="LONG", rsi=rsi[:t + 1],
                )
                if not algo_res.is_valid or algo_res.score < 68: continue
                
                limit_entry = low[t] + 0.50 * c_range
                sl_p = low[t]
                risk_dist = limit_entry - sl_p
                if risk_dist <= 0: continue
                
                risk_frac = risk_dist / limit_entry
                reward_frac = (3.0 * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp_p = limit_entry + limit_entry * reward_frac
                trig_2r = limit_entry + 2.0 * risk_dist
                be_p = limit_entry * (1.0 + 2.0 * FEE)
                
                key = (sym, t)
                sig_data = {
                    "symbol": sym,
                    "direction": 1,  # LONG
                    "dir_str": "LONG",
                    "signal_time": times[t],
                    "signal_bar_idx": t,
                    "entry_p": limit_entry,
                    "sl_p": sl_p,
                    "tp_p": tp_p,
                    "be_p": be_p,
                    "trig_2r": trig_2r,
                    "risk_frac": risk_frac,
                    "score": algo_res.score,
                }
                if key not in all_signals_dict or algo_res.score > all_signals_dict[key]["score"]:
                    all_signals_dict[key] = sig_data
                break
                
    signals_list = sorted(all_signals_dict.values(), key=lambda s: s["signal_bar_idx"])
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(signals_list, f)
    print(f"Сохранено уникальных сигналов в кэш: {len(signals_list)}")
    return signals_list


def run_fast_simulation(
    mdata: Dict[str, pd.DataFrame],
    signals: List[Dict[str, Any]],
    max_concurrent: int = 3,
    cooldown_bars: int = 6,
    risk_per_trade_pct: float = 1.0,
    monthly_dd_limit_pct: float = 5.0,  # 5.0%
):
    """
    Быстрая векторная симуляция.
    """
    sym_list = list(mdata.keys())
    sym_to_idx = {s: i for i, s in enumerate(sym_list)}
    
    first_df = mdata[sym_list[0]]
    timeline = first_df.index
    n_bars = len(timeline)
    n_syms = len(sym_list)
    
    # 2D матрицы цен [bar, sym]
    high_mat = np.zeros((n_bars, n_syms), dtype=np.float64)
    low_mat = np.zeros((n_bars, n_syms), dtype=np.float64)
    close_mat = np.zeros((n_bars, n_syms), dtype=np.float64)
    
    for s_idx, s in enumerate(sym_list):
        high_mat[:, s_idx] = mdata[s]['high'].values
        low_mat[:, s_idx] = mdata[s]['low'].values
        close_mat[:, s_idx] = mdata[s]['close'].values
        
    # Карта месяцев [bar] -> str
    months_arr = [t.strftime("%Y-%m") for t in timeline]
    
    # Сигналы по номеру бара
    sigs_by_bar = {}
    for s in signals:
        b = s["signal_bar_idx"]
        if b not in sigs_by_bar:
            sigs_by_bar[b] = []
        sigs_by_bar[b].append(s)
        
    active_positions: List[ActivePosition] = []
    pending_orders: List[PendingOrder] = []
    closed_trades: List[ClosedTrade] = []
    
    last_exit_bar = np.full(n_syms, -999, dtype=np.int32)
    
    cur_month = ""
    month_start_depo = 1000.0
    current_depo = 1000.0
    month_realized_pnl_pct = 0.0
    month_locked = False
    
    skipped_max_concurrent = 0
    skipped_cooldown = 0
    skipped_monthly_dd = 0
    trade_id_seq = 0
    
    for bar in range(n_bars):
        m_str = months_arr[bar]
        if m_str != cur_month:
            cur_month = m_str
            month_start_depo = current_depo
            month_realized_pnl_pct = 0.0
            month_locked = False
            
        # 1. ОБНОВЛЕНИЕ АКТИВНЫХ ПОЗИЦИЙ
        still_active = []
        for pos in active_positions:
            s_i = pos.sym_idx
            bh = high_mat[bar, s_i]
            bl = low_mat[bar, s_i]
            pos.bars_held += 1
            
            # Консервативная проверка: SL всегда проверяется первым
            if pos.direction == -1:  # SHORT
                hit_sl = bh >= pos.cur_sl
                hit_tp = bl <= pos.tp_p
                hit_2r = bl <= pos.trig_2r
            else:  # LONG
                hit_sl = bl <= pos.cur_sl
                hit_tp = bh >= pos.tp_p
                hit_2r = bh >= pos.trig_2r
                
            if hit_sl:
                reason = "be" if pos.be_active else "sl"
                exit_p = pos.cur_sl
                net_r = 0.0 if pos.be_active else -(pos.risk_frac + 2.0 * FEE) / pos.risk_frac
                trade_pnl_pct = net_r * risk_per_trade_pct
                
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id,
                    symbol=pos.symbol,
                    direction="SHORT" if pos.direction == -1 else "LONG",
                    entry_bar=pos.entry_bar,
                    exit_bar=bar,
                    entry_time=timeline[pos.entry_bar],
                    exit_time=timeline[bar],
                    entry_p=pos.entry_p,
                    exit_p=exit_p,
                    sl_p=pos.sl_p,
                    tp_p=pos.tp_p,
                    exit_reason=reason,
                    net_r=net_r,
                    pnl_pct=trade_pnl_pct,
                    bars_held=pos.bars_held,
                    month=m_str,
                ))
                last_exit_bar[s_i] = bar
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                
                if monthly_dd_limit_pct is not None and month_realized_pnl_pct <= -monthly_dd_limit_pct:
                    month_locked = True
                    # Отменяем все отложенные лимитки при срабатывании стопа месяца
                    pending_orders.clear()
                continue
                
            if hit_tp:
                trade_pnl_pct = 3.0 * risk_per_trade_pct
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id,
                    symbol=pos.symbol,
                    direction="SHORT" if pos.direction == -1 else "LONG",
                    entry_bar=pos.entry_bar,
                    exit_bar=bar,
                    entry_time=timeline[pos.entry_bar],
                    exit_time=timeline[bar],
                    entry_p=pos.entry_p,
                    exit_p=pos.tp_p,
                    sl_p=pos.sl_p,
                    tp_p=pos.tp_p,
                    exit_reason="tp",
                    net_r=3.0,
                    pnl_pct=trade_pnl_pct,
                    bars_held=pos.bars_held,
                    month=m_str,
                ))
                last_exit_bar[s_i] = bar
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                continue
                
            if hit_2r and not pos.be_active:
                pos.be_active = True
                pos.cur_sl = pos.be_p
                
            if pos.bars_held >= 120:  # Таймаут 30 часов
                bc = close_mat[bar, s_i]
                if pos.direction == -1:
                    raw_r = (pos.entry_p - bc) / (pos.sl_p - pos.entry_p)
                else:
                    raw_r = (bc - pos.entry_p) / (pos.entry_p - pos.sl_p)
                net_r = max(-1.0, raw_r)
                trade_pnl_pct = net_r * risk_per_trade_pct
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id,
                    symbol=pos.symbol,
                    direction="SHORT" if pos.direction == -1 else "LONG",
                    entry_bar=pos.entry_bar,
                    exit_bar=bar,
                    entry_time=timeline[pos.entry_bar],
                    exit_time=timeline[bar],
                    entry_p=pos.entry_p,
                    exit_p=bc,
                    sl_p=pos.sl_p,
                    tp_p=pos.tp_p,
                    exit_reason="timeout",
                    net_r=net_r,
                    pnl_pct=trade_pnl_pct,
                    bars_held=pos.bars_held,
                    month=m_str,
                ))
                last_exit_bar[s_i] = bar
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                if monthly_dd_limit_pct is not None and month_realized_pnl_pct <= -monthly_dd_limit_pct:
                    month_locked = True
                    pending_orders.clear()
                continue
                
            still_active.append(pos)
        active_positions = still_active
        
        # 2. ПРОВЕРКА ОТЛОЖЕННЫХ ЛИМИТОВ
        still_pending = []
        for po in pending_orders:
            if month_locked:
                skipped_monthly_dd += 1
                continue
                
            s_i = po.sym_idx
            bh = high_mat[bar, s_i]
            bl = low_mat[bar, s_i]
            
            # Отмена если стоп задет до входа
            if (po.direction == -1 and bh > po.sl_p) or (po.direction == 1 and bl < po.sl_p):
                continue
                
            # Проверка исполнения
            filled = (bh >= po.entry_p) if po.direction == -1 else (bl <= po.entry_p)
            if filled:
                if len(active_positions) < max_concurrent:
                    trade_id_seq += 1
                    active_positions.append(ActivePosition(
                        trade_id=trade_id_seq,
                        symbol=po.symbol,
                        sym_idx=po.sym_idx,
                        direction=po.direction,
                        entry_bar=bar,
                        entry_p=po.entry_p,
                        sl_p=po.sl_p,
                        cur_sl=po.sl_p,
                        tp_p=po.tp_p,
                        be_p=po.be_p,
                        trig_2r=po.trig_2r,
                        risk_frac=po.risk_frac,
                    ))
                else:
                    skipped_max_concurrent += 1
                continue
                
            if bar < po.expire_bar:
                still_pending.append(po)
        pending_orders = still_pending
        
        # 3. ПОСТУПЛЕНИЕ НОВЫХ СИГНАЛОВ
        if bar in sigs_by_bar:
            for sig in sigs_by_bar[bar]:
                sym = sig["symbol"]
                s_i = sym_to_idx[sym]
                
                # Фильтр месяца
                if month_locked:
                    skipped_monthly_dd += 1
                    continue
                    
                # Фильтр кулдауна 6 свечей
                if (bar - last_exit_bar[s_i]) < cooldown_bars:
                    skipped_cooldown += 1
                    continue
                    
                # Фильтр уже активной позиции по монете
                if any(p.sym_idx == s_i for p in active_positions):
                    skipped_max_concurrent += 1
                    continue
                    
                # Фильтр уже выставленного лимита по монете
                if any(p.sym_idx == s_i for p in pending_orders):
                    continue
                    
                # Выставляем лимит на 2 свечи вперед
                pending_orders.append(PendingOrder(
                    symbol=sym,
                    sym_idx=s_i,
                    direction=sig["direction"],
                    signal_bar=bar,
                    expire_bar=bar + 2,
                    entry_p=sig["entry_p"],
                    sl_p=sig["sl_p"],
                    tp_p=sig["tp_p"],
                    be_p=sig["be_p"],
                    trig_2r=sig["trig_2r"],
                    risk_frac=sig["risk_frac"],
                ))
                
    return closed_trades, {
        "skipped_max_concurrent": skipped_max_concurrent,
        "skipped_cooldown": skipped_cooldown,
        "skipped_monthly_dd": skipped_monthly_dd,
        "final_depo": current_depo,
    }


def summarize_simulation(closed_trades: List[ClosedTrade], stats_meta: dict, label: str):
    if not closed_trades:
        print(f"[{label}] 0 сделок")
        return None, None
        
    df_t = pd.DataFrame([{
        "id": t.trade_id,
        "symbol": t.symbol,
        "direction": t.direction,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "exit_reason": t.exit_reason,
        "net_r": t.net_r,
        "pnl_pct": t.pnl_pct,
        "bars_held": t.bars_held,
        "month": t.month,
    } for t in closed_trades])
    
    n_trades = len(df_t)
    wins = len(df_t[df_t['exit_reason'] == "tp"])
    bes = len(df_t[df_t['exit_reason'] == "be"])
    sls = len(df_t[df_t['exit_reason'] == "sl"])
    tos = len(df_t[df_t['exit_reason'] == "timeout"])
    
    tot_r = df_t['net_r'].sum()
    gw = df_t[df_t['net_r'] > 0]['net_r'].sum()
    gl = abs(df_t[df_t['net_r'] < 0]['net_r'].sum())
    pf = gw / gl if gl > 0 else 99.0
    
    # Помесячная статистика
    months = sorted(df_t['month'].unique())
    m_rows = []
    
    cum_depo = 1000.0
    for m in months:
        sub = df_t[df_t['month'] == m]
        m_t = len(sub)
        mw = len(sub[sub['exit_reason'] == "tp"])
        mb = len(sub[sub['exit_reason'] == "be"])
        ms = len(sub[sub['exit_reason'] == "sl"])
        mr = sub['net_r'].sum()
        
        # Точный расчет баланса месяца
        start_m_depo = cum_depo
        for pnl_p in sub['pnl_pct'].values:
            cum_depo *= (1.0 + pnl_p / 100.0)
        m_growth_pct = ((cum_depo - start_m_depo) / start_m_depo) * 100.0
        
        m_rows.append({
            "month": m,
            "trades": m_t,
            "wins": mw,
            "be": mb,
            "losses": ms,
            "win_rate": round(mw / m_t * 100.0, 1) if m_t > 0 else 0.0,
            "net_r": round(mr, 2),
            "month_growth_pct": round(m_growth_pct, 2),
            "end_depo": round(cum_depo, 2),
            "hit_stop": "СТОП 5%" if m_growth_pct <= -4.95 else "ОК",
        })
    df_m = pd.DataFrame(m_rows)
    
    # Максимальная просадка депозита
    depo_curve = [1000.0]
    for p in df_t['pnl_pct'].values:
        depo_curve.append(depo_curve[-1] * (1.0 + p / 100.0))
    s_depo = pd.Series(depo_curve)
    peak = s_depo.cummax()
    dd_pct = (peak - s_depo) / peak * 100.0
    max_dd = dd_pct.max()
    
    n_stop_months = len(df_m[df_m['hit_stop'] == "СТОП 5%"])
    positive_months = len(df_m[df_m['month_growth_pct'] > 0])
    
    print(f"\n=========================================================================")
    print(f"   {label}")
    print(f"=========================================================================")
    print(f"Всего исполнено сделок: {n_trades} ({n_trades/22:.1f} сделок/мес)")
    print(f"Тейки (+3R): {wins} ({wins/n_trades*100:.1f}%), БУ (0R): {bes} ({bes/n_trades*100:.1f}%), Стопы: {sls} ({sls/n_trades*100:.1f}%)")
    print(f"Profit Factor: {pf:.2f} | Суммарный Net R: {tot_r:+.2f} R | EV: {tot_r/n_trades:+.3f} R")
    print(f"Итоговый депозит: ${cum_depo:,.2f} ({cum_depo/1000.0:+.2f}x)")
    print(f"Макс. просадка депозита за 22 месяца: {max_dd:.2f}%")
    print(f"Месяцев в плюсе: {positive_months} из 22 | Месяцев сработавшего стопа 5%: {n_stop_months}")
    print(f"Пропущено из-за лимита позиций: {stats_meta['skipped_max_concurrent']}")
    print(f"Пропущено из-за кулдауна 6 свечей: {stats_meta['skipped_cooldown']}")
    print(f"Пропущено из-за стопа месяца: {stats_meta['skipped_monthly_dd']}")
    
    return df_t, df_m


if __name__ == "__main__":
    print("Загрузка данных...")
    mdata = load_all_market_data()
    signals = extract_deduplicated_signals(mdata)
    print(f"Всего уникальных сигналов Model 2: {len(signals)}")
    
    configs = [
        # (max_c, cooldown, risk_per_trade, monthly_dd_limit, label)
        (3, 6, 1.0, 5.0, "Вариант A1: Max 3 позиции | Риск 1.0% | Лимит 5% в месяц"),
        (4, 6, 1.0, 5.0, "Вариант A2: Max 4 позиции | Риск 1.0% | Лимит 5% в месяц"),
        (3, 6, 0.50, 5.0, "Вариант B1: Max 3 позиции | Риск 0.50% | Лимит 5% в месяц"),
        (4, 6, 0.50, 5.0, "Вариант B2: Max 4 позиции | Риск 0.50% | Лимит 5% в месяц"),
        (3, 6, 0.35, 5.0, "Вариант C1: Max 3 позиции | Риск 0.35% | Лимит 5% в месяц"),
        (4, 6, 0.35, 5.0, "Вариант C2: Max 4 позиции | Риск 0.35% | Лимит 5% в месяц"),
        (3, 6, 1.0, None, "Вариант D1 (Без блокировки): Max 3 позиции | Риск 1.0%"),
        (4, 6, 1.0, None, "Вариант D2 (Без блокировки): Max 4 позиции | Риск 1.0%"),
    ]
    
    results_summary = []
    for max_c, cd, r_trade, m_limit, lbl in configs:
        trades, stats = run_fast_simulation(
            mdata, signals, max_concurrent=max_c, cooldown_bars=cd,
            risk_per_trade_pct=r_trade, monthly_dd_limit_pct=m_limit
        )
        df_t, df_m = summarize_simulation(trades, stats, lbl)
        if df_t is not None:
            results_summary.append({
                "label": lbl,
                "max_c": max_c,
                "risk_pct": r_trade,
                "m_limit": m_limit if m_limit else "Нет",
                "trades": len(df_t),
                "pf": round((df_t[df_t['net_r'] > 0]['net_r'].sum()) / abs(df_t[df_t['net_r'] < 0]['net_r'].sum()), 2),
                "net_r": round(df_t['net_r'].sum(), 1),
                "final_depo": round(stats['final_depo'], 2),
                "growth_x": round(stats['final_depo'] / 1000.0, 2),
                "pos_months": len(df_m[df_m['month_growth_pct'] > 0]),
                "stop_months": len(df_m[df_m['hit_stop'] == "СТОП 5%"]),
            })
            
    df_res = pd.DataFrame(results_summary)
    df_res.to_csv(RESULTS_DIR / "comparison_matrix_rm.csv", index=False)
    print("\n================== ИТОГОВАЯ МАТРИЦА СРАВНЕНИЯ ==================")
    print(df_res[["label", "trades", "pf", "net_r", "final_depo", "growth_x", "pos_months", "stop_months"]].to_string())
