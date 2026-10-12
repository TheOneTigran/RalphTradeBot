"""
run_portfolio_rm_simulation.py — Портфельная хронологическая симуляция риск-менеджмента:
1. Кулдаун 6 свечей на одну монету.
2. Ограничение: максимум 3–4 одновременно открытые сделки на весь портфель из 15 монет.
3. Месячный лимит просадки: максимум -5.0% (-5R) в календарный месяц (при достижении -5% торговля останавливается до следующего месяца).
4. Правила сделок: вход лимит 50% (окно 1-2 свечи), SL за кончик тени, БУ при +2R Net, TP при +3R Net, комиссии 0.05% in + 0.05% out.
5. Непрерывные данные 2025–2026 гг. (22 месяца, 15 монет, 15m ТФ).
"""
import sys
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

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'NEARUSDT',
    'SUIUSDT', 'AAVEUSDT', 'APTUSDT', 'DOTUSDT', 'RENDERUSDT',
]

FEE = 0.0005


@dataclass
class PendingOrder:
    symbol: str
    direction: str
    signal_time: pd.Timestamp
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
    direction: str
    entry_time: pd.Timestamp
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
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_p: float
    exit_p: float
    sl_p: float
    tp_p: float
    exit_reason: str  # "tp", "be", "sl", "timeout"
    net_r: float
    net_pnl_usd: float
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


def detect_all_raw_signals(mdata: Dict[str, pd.DataFrame]):
    """
    Находит все сигналы Model 2 (W5 = свеча поглощения S в реальном времени, 0% lookahead).
    """
    signals = []
    
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
                
                signals.append({
                    "symbol": sym,
                    "direction": "SHORT",
                    "signal_time": times[t],
                    "signal_bar_idx": t,
                    "entry_p": limit_entry,
                    "sl_p": sl_p,
                    "tp_p": tp_p,
                    "be_p": be_p,
                    "trig_2r": trig_2r,
                    "risk_frac": risk_frac,
                })
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
                
                signals.append({
                    "symbol": sym,
                    "direction": "LONG",
                    "signal_time": times[t],
                    "signal_bar_idx": t,
                    "entry_p": limit_entry,
                    "sl_p": sl_p,
                    "tp_p": tp_p,
                    "be_p": be_p,
                    "trig_2r": trig_2r,
                    "risk_frac": risk_frac,
                })
                break
                
    signals.sort(key=lambda s: s["signal_time"])
    return signals


def run_portfolio_simulation(
    mdata: Dict[str, pd.DataFrame],
    signals: List[Dict[str, Any]],
    max_concurrent: int = 3,
    cooldown_bars: int = 6,
    monthly_dd_limit_pct: float = 5.0,
):
    """
    Хронологическая симуляция по всем монетам с общим временем,
    проверкой параллельных позиций, кулдауна и месячного стоп-лосса.
    """
    # Создаем объединенную временную шкалу 15m
    sample_df = next(iter(mdata.values()))
    timeline = sample_df.index
    
    # Индексируем сигналы по времени
    sigs_by_time: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for s in signals:
        t = s["signal_time"]
        if t not in sigs_by_time:
            sigs_by_time[t] = []
        sigs_by_time[t].append(s)
        
    active_positions: List[ActivePosition] = []
    pending_orders: List[PendingOrder] = []
    closed_trades: List[ClosedTrade] = []
    
    last_trade_bar: Dict[str, int] = {}
    
    # Месячный трекинг риска
    cur_month = ""
    month_net_r = 0.0
    month_locked = False
    
    skipped_max_concurrent = 0
    skipped_cooldown = 0
    skipped_monthly_dd = 0
    
    trade_id_seq = 0
    
    for bar_idx, cur_time in enumerate(timeline):
        m_str = cur_time.strftime("%Y-%m")
        if m_str != cur_month:
            cur_month = m_str
            month_net_r = 0.0
            month_locked = False
            
        # 1. ОБНОВЛЕНИЕ АКТИВНЫХ ПОЗИЦИЙ НА БАРЕ cur_time
        still_active = []
        for pos in active_positions:
            df = mdata[pos.symbol]
            if cur_time not in df.index:
                still_active.append(pos)
                continue
            
            row = df.loc[cur_time]
            bh, bl = float(row['high']), float(row['low'])
            pos.bars_held += 1
            
            # Консервативно: сначала стоп
            if pos.direction == "SHORT":
                hit_sl = bh >= pos.cur_sl
                hit_tp = bl <= pos.tp_p
                hit_2r = bl <= pos.trig_2r
            else:
                hit_sl = bl <= pos.cur_sl
                hit_tp = bh >= pos.tp_p
                hit_2r = bh >= pos.trig_2r
                
            if hit_sl:
                reason = "be" if pos.be_active else "sl"
                exit_p = pos.cur_sl
                net_r = 0.0 if pos.be_active else -(pos.risk_frac + 2.0 * FEE) / pos.risk_frac
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    entry_time=pos.entry_time,
                    exit_time=cur_time,
                    entry_p=pos.entry_p,
                    exit_p=exit_p,
                    sl_p=pos.sl_p,
                    tp_p=pos.tp_p,
                    exit_reason=reason,
                    net_r=net_r,
                    net_pnl_usd=net_r * 10.0,
                    bars_held=pos.bars_held,
                    month=m_str,
                ))
                month_net_r += net_r
                if month_net_r <= -monthly_dd_limit_pct:
                    month_locked = True
                continue
                
            if hit_tp:
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    entry_time=pos.entry_time,
                    exit_time=cur_time,
                    entry_p=pos.entry_p,
                    exit_p=pos.tp_p,
                    sl_p=pos.sl_p,
                    tp_p=pos.tp_p,
                    exit_reason="tp",
                    net_r=3.0,
                    net_pnl_usd=30.0,
                    bars_held=pos.bars_held,
                    month=m_str,
                ))
                month_net_r += 3.0
                continue
                
            if hit_2r and not pos.be_active:
                pos.be_active = True
                pos.cur_sl = pos.be_p
                
            if pos.bars_held >= 120:  # таймаут 30 часов
                closed_trades.append(ClosedTrade(
                    trade_id=pos.trade_id,
                    symbol=pos.symbol,
                    direction=pos.direction,
                    entry_time=pos.entry_time,
                    exit_time=cur_time,
                    entry_p=pos.entry_p,
                    exit_p=float(row['close']),
                    sl_p=pos.sl_p,
                    tp_p=pos.tp_p,
                    exit_reason="timeout",
                    net_r=0.0,
                    net_pnl_usd=0.0,
                    bars_held=pos.bars_held,
                    month=m_str,
                ))
                continue
                
            still_active.append(pos)
            
        active_positions = still_active
        
        # 2. ПРОВЕРКА ОТЛОЖЕННЫХ ЛИМИТНЫХ ОРДЕРОВ
        still_pending = []
        for po in pending_orders:
            df = mdata[po.symbol]
            if cur_time not in df.index:
                if bar_idx < po.expire_bar:
                    still_pending.append(po)
                continue
                
            row = df.loc[cur_time]
            bh, bl = float(row['high']), float(row['low'])
            
            # Проверка отмены по стопу до входа
            if (po.direction == "SHORT" and bh > po.sl_p) or (po.direction == "LONG" and bl < po.sl_p):
                continue  # отменён
                
            # Проверка исполнения лимита
            filled = False
            if po.direction == "SHORT" and bh >= po.entry_p:
                filled = True
            elif po.direction == "LONG" and bl <= po.entry_p:
                filled = True
                
            if filled:
                # Проверяем, есть ли свободный слот из max_concurrent
                if len(active_positions) < max_concurrent and not month_locked:
                    trade_id_seq += 1
                    active_positions.append(ActivePosition(
                        trade_id=trade_id_seq,
                        symbol=po.symbol,
                        direction=po.direction,
                        entry_time=cur_time,
                        entry_p=po.entry_p,
                        sl_p=po.sl_p,
                        cur_sl=po.sl_p,
                        tp_p=po.tp_p,
                        be_p=po.be_p,
                        trig_2r=po.trig_2r,
                        risk_frac=po.risk_frac,
                    ))
                    last_trade_bar[po.symbol] = bar_idx
                else:
                    if month_locked:
                        skipped_monthly_dd += 1
                    else:
                        skipped_max_concurrent += 1
                continue
                
            if bar_idx < po.expire_bar:
                still_pending.append(po)
                
        pending_orders = still_pending
        
        # 3. ПОСТУПЛЕНИЕ НОВЫХ СИГНАЛОВ НА ТЕКУЩЕМ БАРЕ
        if cur_time in sigs_by_time:
            for sig in sigs_by_time[cur_time]:
                sym = sig["symbol"]
                
                # Фильтр 1: Замок по месячной просадке 5%
                if month_locked:
                    skipped_monthly_dd += 1
                    continue
                    
                # Фильтр 2: Кулдаун 6 свечей на одну монету
                last_b = last_trade_bar.get(sym, -999)
                if (bar_idx - last_b) < cooldown_bars:
                    skipped_cooldown += 1
                    continue
                    
                # Фильтр 3: Не открывать новый лимит по монете, если уже есть активная позиция
                if any(p.symbol == sym for p in active_positions):
                    skipped_max_concurrent += 1
                    continue
                    
                # Выставляем лимит на 2 свечи (cur_bar + 1, cur_bar + 2)
                pending_orders.append(PendingOrder(
                    symbol=sym,
                    direction=sig["direction"],
                    signal_time=cur_time,
                    expire_bar=bar_idx + 2,
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
    }


def analyze_results(closed_trades: List[ClosedTrade], max_conc: int, stats_meta: dict):
    if not closed_trades:
        print(f"Max Concurrent {max_conc}: 0 сделок")
        return
        
    df_t = pd.DataFrame([{
        "id": t.trade_id,
        "symbol": t.symbol,
        "direction": t.direction,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "exit_reason": t.exit_reason,
        "net_r": t.net_r,
        "net_pnl_usd": t.net_pnl_usd,
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
    
    # Помесячная таблица
    months = sorted(df_t['month'].unique())
    m_rows = []
    comp_depo = 1000.0
    
    for m in months:
        sub = df_t[df_t['month'] == m]
        m_t = len(sub)
        mw = len(sub[sub['exit_reason'] == "tp"])
        mb = len(sub[sub['exit_reason'] == "be"])
        ms = len(sub[sub['exit_reason'] == "sl"])
        mr = sub['net_r'].sum()
        m_fix = mr * 10.0
        
        m_pnl_comp = mr * (comp_depo * 0.01)
        comp_depo += m_pnl_comp
        growth_pct = mr * 1.0
        
        m_rows.append({
            "month": m,
            "trades": m_t,
            "wins": mw,
            "be": mb,
            "losses": ms,
            "win_rate": round(mw / m_t * 100.0, 1) if m_t > 0 else 0.0,
            "net_r": round(mr, 2),
            "fixed_pnl": round(m_fix, 2),
            "compound_depo": round(comp_depo, 2),
            "growth_pct": round(growth_pct, 1),
            "hit_5pct_limit": "ДA (-5% Stop)" if mr <= -5.0 else "Нет",
        })
        
    df_m = pd.DataFrame(m_rows)
    
    # Максимальная просадка
    cum_r = df_t['net_r'].cumsum()
    peak = cum_r.cummax()
    dd_r = peak - cum_r
    max_dd_r = dd_r.max()
    max_dd_pct = max_dd_r * 1.0  # при 1% риске
    
    # По монетам
    coin_rows = []
    for sym in SYMBOLS:
        sub = df_t[df_t['symbol'] == sym]
        if len(sub) == 0: continue
        ct = len(sub)
        cw = len(sub[sub['exit_reason'] == "tp"])
        cb = len(sub[sub['exit_reason'] == "be"])
        cs = len(sub[sub['exit_reason'] == "sl"])
        cr = sub['net_r'].sum()
        c_gw = sub[sub['net_r'] > 0]['net_r'].sum()
        c_gl = abs(sub[sub['net_r'] < 0]['net_r'].sum())
        cpf = c_gw / c_gl if c_gl > 0 else 99.0
        coin_rows.append({
            "symbol": sym,
            "trades": ct,
            "wins": cw,
            "be": cb,
            "losses": cs,
            "win_rate": round(cw / ct * 100.0, 1),
            "profit_factor": round(cpf, 2),
            "net_r": round(cr, 2),
            "net_usd": round(cr * 10.0, 2),
        })
    df_c = pd.DataFrame(coin_rows).sort_values("net_r", ascending=False)
    
    # Сохраняем в CSV
    df_t.to_csv(RESULTS_DIR / f"portfolio_trades_max{max_conc}.csv", index=False)
    df_m.to_csv(RESULTS_DIR / f"portfolio_monthly_max{max_conc}.csv", index=False)
    df_c.to_csv(RESULTS_DIR / f"portfolio_coins_max{max_conc}.csv", index=False)
    
    print(f"\n=======================================================")
    print(f"   ПОРТФЕЛЬНЫЙ ОТЧЁТ (MAX CONCURRENT = {max_conc}, COOLDOWN = 6)")
    print(f"=======================================================")
    print(f"Всего исполнено сделок: {n_trades} (в среднем {n_trades/22:.1f} сделок/мес)")
    print(f"Пропущено из-за лимита {max_conc} позиций: {stats_meta['skipped_max_concurrent']}")
    print(f"Пропущено из-за кулдауна 6 свечей: {stats_meta['skipped_cooldown']}")
    print(f"Пропущено из-за месячного стопа 5%: {stats_meta['skipped_monthly_dd']}")
    print(f"Тейки (+3R Net): {wins} ({wins/n_trades*100:.1f}%)")
    print(f"Безубытки (0R): {bes} ({bes/n_trades*100:.1f}%)")
    print(f"Стоп-лоссы (-1R..-1.25R): {sls} ({sls/n_trades*100:.1f}%)")
    print(f"Таймауты: {tos}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Суммарный Net R: {tot_r:+.2f} R")
    print(f"EV на сделку: {tot_r/n_trades:+.3f} R")
    print(f"Макс. просадка портфеля: {max_dd_r:.2f} R ({max_dd_pct:.1f}%)")
    print(f"Итоговый депозит со сложным %: ${comp_depo:,.2f} (x{comp_depo/1000.0:.2f})")
    
    return df_t, df_m, df_c


if __name__ == "__main__":
    print("Загрузка данных...")
    mdata = load_all_market_data()
    print("Поиск всех сигналов Model 2 в истории...")
    all_signals = detect_all_raw_signals(mdata)
    print(f"Всего выявлено сигналов Model 2: {len(all_signals)}")
    
    for max_c in [3, 4]:
        trades, stats = run_portfolio_simulation(
            mdata, all_signals, max_concurrent=max_c, cooldown_bars=6, monthly_dd_limit_pct=5.0
        )
        analyze_results(trades, max_c, stats)
