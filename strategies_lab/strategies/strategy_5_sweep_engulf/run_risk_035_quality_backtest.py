"""
run_risk_035_quality_backtest.py — Детальный бектест с риском 0.35%:
Сравнение "сырых" настроек (TP 3.0R, BE 2.0R) и "качественных" настроек (TP 2.5R, BE 1.5R, Score >= 72, Vol > 1.1x)
с ограничением 3-4 параллельных сделок и лимитом -5% в месяц.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

BASE_DIR = Path(r"c:\Users\user\Desktop\RalphTradeBot")
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from elliott_detector import detect_elliott_impulse
from simulate_rm_portfolio_v2 import load_all_market_data, SYMBOLS, FEE, RESULTS_DIR


def extract_signals_set(mdata, target_r=2.5, be_r=1.5, min_score=72, min_range=0.45, use_vol=True):
    signals = []
    for sym in SYMBOLS:
        df = mdata[sym]
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_p = df['open'].values
        vol = df['volume'].values
        n = len(df)
        times = df.index
        
        rsi = calculate_rsi_wilder(close, period=7)
        vol_ma = pd.Series(vol).rolling(15).mean().values if use_vol else None
        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]
        
        # SHORT
        for p1 in high_pivots:
            if rsi[p1] < 60.0: continue
            for t in range(p1 + 5, min(n - 40, p1 + 35)):
                if high[t] <= high[p1] or high[t] < np.max(high[p1+1 : t+1]): continue
                if not (close[t] < open_p[t] and close[t] < min(open_p[t-1], close[t-1])): continue
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < min_range: continue
                if rsi[t] >= rsi[p1] - 2.5 or rsi[t] < 45.0: continue
                if use_vol and vol[t] < 1.1 * vol_ma[t]: continue
                algo_res = detect_elliott_impulse(
                    high=high[:t+1], low=low[:t+1], close=close[:t+1], open_p=open_p[:t+1],
                    w3_bar=p1, w5_bar=t, direction='SHORT', rsi=rsi[:t+1]
                )
                if not algo_res.is_valid or algo_res.score < min_score: continue
                
                limit_entry = low[t] + 0.50 * c_range
                sl_p = high[t]
                risk_dist = sl_p - limit_entry
                if risk_dist <= 0: continue
                
                risk_frac = risk_dist / limit_entry
                reward_frac = (target_r * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp_p = limit_entry - limit_entry * reward_frac
                trig_be = limit_entry - be_r * risk_dist
                be_p = limit_entry * (1.0 - 2.0 * FEE)
                
                signals.append({
                    "symbol": sym, "direction": "SHORT", "signal_time": times[t],
                    "signal_bar_idx": t, "entry_p": limit_entry, "sl_p": sl_p,
                    "tp_p": tp_p, "be_p": be_p, "trig_be": trig_be,
                    "target_r": target_r, "risk_frac": risk_frac,
                })
                break
                
        # LONG
        for p1 in low_pivots:
            if rsi[p1] > 40.0: continue
            for t in range(p1 + 5, min(n - 40, p1 + 35)):
                if low[t] >= low[p1] or low[t] > np.min(low[p1+1 : t+1]): continue
                if not (close[t] > open_p[t] and close[t] > max(open_p[t-1], close[t-1])): continue
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < min_range: continue
                if rsi[t] <= rsi[p1] + 2.5 or rsi[t] > 55.0: continue
                if use_vol and vol[t] < 1.1 * vol_ma[t]: continue
                algo_res = detect_elliott_impulse(
                    high=high[:t+1], low=low[:t+1], close=close[:t+1], open_p=open_p[:t+1],
                    w3_bar=p1, w5_bar=t, direction='LONG', rsi=rsi[:t+1]
                )
                if not algo_res.is_valid or algo_res.score < min_score: continue
                
                limit_entry = low[t] + 0.50 * c_range
                sl_p = low[t]
                risk_dist = limit_entry - sl_p
                if risk_dist <= 0: continue
                
                risk_frac = risk_dist / limit_entry
                reward_frac = (target_r * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp_p = limit_entry + limit_entry * reward_frac
                trig_be = limit_entry + be_r * risk_dist
                be_p = limit_entry * (1.0 + 2.0 * FEE)
                
                signals.append({
                    "symbol": sym, "direction": "LONG", "signal_time": times[t],
                    "signal_bar_idx": t, "entry_p": limit_entry, "sl_p": sl_p,
                    "tp_p": tp_p, "be_p": be_p, "trig_be": trig_be,
                    "target_r": target_r, "risk_frac": risk_frac,
                })
                break
                
    signals.sort(key=lambda s: s["signal_time"])
    return signals


def simulate_portfolio(mdata, signals, max_concurrent=3, cooldown_bars=6, risk_pct=0.35, monthly_limit_pct=5.0):
    timeline = next(iter(mdata.values())).index
    sigs_by_time = {}
    for s in signals:
        sigs_by_time.setdefault(s["signal_time"], []).append(s)
        
    active_positions = []
    pending_orders = []
    closed_trades = []
    last_trade_bar = {}
    
    cur_month = ""
    current_depo = 1000.0
    month_start_depo = 1000.0
    month_realized_pnl_pct = 0.0
    month_locked = False
    
    skipped_max_concurrent = 0
    skipped_cooldown = 0
    skipped_monthly_dd = 0
    trade_id_seq = 0
    
    for bar_idx, cur_time in enumerate(timeline):
        m_str = cur_time.strftime("%Y-%m")
        if m_str != cur_month:
            cur_month = m_str
            month_start_depo = current_depo
            month_realized_pnl_pct = 0.0
            month_locked = False
            
        # 1. Active positions
        still_active = []
        for pos in active_positions:
            df = mdata[pos["symbol"]]
            if cur_time not in df.index:
                still_active.append(pos); continue
            row = df.loc[cur_time]
            bh, bl = float(row['high']), float(row['low'])
            pos["bars_held"] += 1
            
            if pos["direction"] == "SHORT":
                hit_sl = bh >= pos["cur_sl"]
                hit_tp = bl <= pos["tp_p"]
                hit_be = bl <= pos["trig_be"]
            else:
                hit_sl = bl <= pos["cur_sl"]
                hit_tp = bh >= pos["tp_p"]
                hit_be = bh >= pos["trig_be"]
                
            if hit_sl:
                reason = "be" if pos["be_active"] else "sl"
                net_r = 0.0 if pos["be_active"] else -(pos["risk_frac"] + 2.0 * FEE) / pos["risk_frac"]
                trade_pnl_pct = net_r * risk_pct
                closed_trades.append({
                    "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                    "entry_time": pos["entry_time"], "exit_time": cur_time,
                    "reason": reason, "net_r": net_r, "pnl_pct": trade_pnl_pct,
                    "bars_held": pos["bars_held"], "month": m_str,
                })
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                if monthly_limit_pct is not None and month_realized_pnl_pct <= -monthly_limit_pct:
                    month_locked = True
                    pending_orders.clear()
                continue
                
            if hit_tp:
                trade_pnl_pct = pos["target_r"] * risk_pct
                closed_trades.append({
                    "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                    "entry_time": pos["entry_time"], "exit_time": cur_time,
                    "reason": "tp", "net_r": pos["target_r"], "pnl_pct": trade_pnl_pct,
                    "bars_held": pos["bars_held"], "month": m_str,
                })
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                continue
                
            if hit_be and not pos["be_active"]:
                pos["be_active"] = True
                pos["cur_sl"] = pos["be_p"]
                
            if pos["bars_held"] >= 120:
                closed_trades.append({
                    "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                    "entry_time": pos["entry_time"], "exit_time": cur_time,
                    "reason": "timeout", "net_r": 0.0, "pnl_pct": 0.0,
                    "bars_held": pos["bars_held"], "month": m_str,
                })
                continue
                
            still_active.append(pos)
        active_positions = still_active
        
        # 2. Pending orders
        still_pending = []
        for po in pending_orders:
            if month_locked:
                skipped_monthly_dd += 1; continue
            df = mdata[po["symbol"]]
            if cur_time not in df.index:
                if bar_idx < po["expire_bar"]: still_pending.append(po)
                continue
            row = df.loc[cur_time]
            bh, bl = float(row['high']), float(row['low'])
            if (po["direction"] == "SHORT" and bh > po["sl_p"]) or (po["direction"] == "LONG" and bl < po["sl_p"]):
                continue
            filled = (bh >= po["entry_p"]) if po["direction"] == "SHORT" else (bl <= po["entry_p"])
            if filled:
                if len(active_positions) < max_concurrent:
                    trade_id_seq += 1
                    active_positions.append({
                        "id": trade_id_seq, "symbol": po["symbol"], "direction": po["direction"],
                        "entry_time": cur_time, "entry_p": po["entry_p"], "sl_p": po["sl_p"],
                        "cur_sl": po["sl_p"], "tp_p": po["tp_p"], "be_p": po["be_p"],
                        "trig_be": po["trig_be"], "target_r": po["target_r"],
                        "risk_frac": po["risk_frac"], "be_active": False, "bars_held": 0,
                    })
                    last_trade_bar[po["symbol"]] = bar_idx
                else:
                    skipped_max_concurrent += 1
                continue
            if bar_idx < po["expire_bar"]:
                still_pending.append(po)
        pending_orders = still_pending
        
        # 3. New signals
        if cur_time in sigs_by_time:
            for sig in sigs_by_time[cur_time]:
                sym = sig["symbol"]
                if month_locked:
                    skipped_monthly_dd += 1; continue
                last_b = last_trade_bar.get(sym, -999)
                if (bar_idx - last_b) < cooldown_bars:
                    skipped_cooldown += 1; continue
                if any(p["symbol"] == sym for p in active_positions):
                    skipped_max_concurrent += 1; continue
                if any(p["symbol"] == sym for p in pending_orders):
                    continue
                pending_orders.append({
                    "symbol": sym, "direction": sig["direction"], "signal_time": cur_time,
                    "expire_bar": bar_idx + 2, "entry_p": sig["entry_p"], "sl_p": sig["sl_p"],
                    "tp_p": sig["tp_p"], "be_p": sig["be_p"], "trig_be": sig["trig_be"],
                    "target_r": sig["target_r"], "risk_frac": sig["risk_frac"],
                })
                
    return closed_trades, {
        "skipped_max_concurrent": skipped_max_concurrent,
        "skipped_cooldown": skipped_cooldown,
        "skipped_monthly_dd": skipped_monthly_dd,
        "final_depo": current_depo,
    }


def analyze_metrics(trades, label):
    if not trades:
        print(f"[{label}] 0 сделок")
        return None
    df = pd.DataFrame(trades)
    n_t = len(df)
    w = len(df[df['reason'] == "tp"])
    b = len(df[df['reason'] == "be"])
    l = len(df[df['reason'] == "sl"])
    tot_r = df['net_r'].sum()
    gw = df[df['net_r'] > 0]['net_r'].sum()
    gl = abs(df[df['net_r'] < 0]['net_r'].sum())
    pf = gw / gl if gl > 0 else 99.0
    
    # Расчет кривой депозита и просадки
    depo = 1000.0
    curve = [1000.0]
    for p in df['pnl_pct']:
        depo *= (1.0 + p / 100.0)
        curve.append(depo)
    s = pd.Series(curve)
    max_dd = ((s.cummax() - s) / s.cummax()).max() * 100.0
    
    # Помесячная таблица
    months = sorted(df['month'].unique())
    m_pnl = []
    d = 1000.0
    for m in months:
        sub = df[df['month'] == m]
        start_d = d
        for p in sub['pnl_pct']:
            d *= (1.0 + p / 100.0)
        pct = (d - start_d) / start_d * 100.0
        m_pnl.append((m, len(sub), sub['net_r'].sum(), pct, d))
    df_m = pd.DataFrame(m_pnl, columns=["month", "trades", "net_r", "month_pct", "end_depo"])
    pos_m = len(df_m[df_m['month_pct'] > 0])
    stop_m = len(df_m[df_m['month_pct'] <= -4.95])
    
    print(f"\n>>> {label} <<<")
    print(f"Сделок: {n_t} ({n_t/22:.1f}/мес) | TP: {w} ({w/n_t*100:.1f}%) | BE: {b} ({b/n_t*100:.1f}%) | SL: {l} ({l/n_t*100:.1f}%)")
    print(f"Profit Factor: {pf:.2f} | Net R: {tot_r:+.2f} R | EV: {tot_r/n_t:+.3f} R")
    print(f"Депозит: ${depo:,.2f} ({depo/1000.0:+.2f}x) | Макс. просадка: {max_dd:.1f}%")
    print(f"Месяцев в плюсе: {pos_m} из {len(months)} | Месяцев стопа 5%: {stop_m}")
    
    return {
        "label": label, "trades": n_t, "tp_pct": round(w/n_t*100, 1),
        "be_pct": round(b/n_t*100, 1), "sl_pct": round(l/n_t*100, 1),
        "pf": round(pf, 2), "net_r": round(tot_r, 1), "depo": round(depo, 2),
        "growth_x": round(depo/1000.0, 2), "max_dd_pct": round(max_dd, 1),
        "pos_m": pos_m, "stop_m": stop_m
    }


if __name__ == "__main__":
    print("Загрузка данных...")
    mdata = load_all_market_data()
    
    print("\n1. Генерация качественных сигналов (TP=2.5R, BE=1.5R, Score>=72, Vol>1.1x, Range>=0.45%)...")
    q_signals = extract_signals_set(mdata, target_r=2.5, be_r=1.5, min_score=72, min_range=0.45, use_vol=True)
    print(f"Найдено качественных сигналов: {len(q_signals)}")
    
    print("\n2. Генерация базовых сигналов (TP=3.0R, BE=2.0R, Score>=68, Vol=False, Range>=0.40%)...")
    raw_signals = extract_signals_set(mdata, target_r=3.0, be_r=2.0, min_score=68, min_range=0.40, use_vol=False)
    print(f"Найдено базовых сигналов: {len(raw_signals)}")
    
    configs = [
        # (signals, max_c, risk_pct, monthly_limit, label)
        (q_signals, 3, 0.35, 5.0,  "Качественные | Max 3 | Риск 0.35% | ЛИМИТ 5%"),
        (q_signals, 4, 0.35, 5.0,  "Качественные | Max 4 | Риск 0.35% | ЛИМИТ 5%"),
        (q_signals, 3, 0.35, None, "Качественные | Max 3 | Риск 0.35% | БЕЗ лимита"),
        (q_signals, 4, 0.35, None, "Качественные | Max 4 | Риск 0.35% | БЕЗ лимита"),
        (raw_signals, 3, 0.35, 5.0,  "Базовые (сырые) | Max 3 | Риск 0.35% | ЛИМИТ 5%"),
        (raw_signals, 4, 0.35, 5.0,  "Базовые (сырые) | Max 4 | Риск 0.35% | ЛИМИТ 5%"),
        (raw_signals, 3, 0.35, None, "Базовые (сырые) | Max 3 | Риск 0.35% | БЕЗ лимита"),
        (raw_signals, 4, 0.35, None, "Базовые (сырые) | Max 4 | Риск 0.35% | БЕЗ лимита"),
    ]
    
    summary = []
    for sigs, max_c, r_pct, m_lim, lbl in configs:
        trades, meta = simulate_portfolio(mdata, sigs, max_concurrent=max_c, cooldown_bars=6, risk_pct=r_pct, monthly_limit_pct=m_lim)
        res = analyze_metrics(trades, lbl)
        if res: summary.append(res)
        
    df_s = pd.DataFrame(summary)
    df_s.to_csv(RESULTS_DIR / "quality_vs_raw_035_comparison.csv", index=False)
    print("\n" + "="*95)
    print("                     ИТОГОВАЯ ТАБЛИЦА: РИСК 0.35% (КАЧЕСТВЕННЫЕ vs БАЗОВЫЕ)")
    print("="*95)
    print(df_s[["label", "trades", "pf", "net_r", "depo", "growth_x", "max_dd_pct", "pos_m", "stop_m"]].to_string())
