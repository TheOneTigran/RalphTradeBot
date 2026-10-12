"""
test_refined_impulse_engine.py — Проверка доработанного алгоритма детекции импульсов:
1. Защита от "вылета RSI в космос" (RSI Purity Gatekeeper).
2. Защита от раздувания 5-й волны (Time Ratio <= 2.0, Max Distance <= 18 свечей, Price L5 <= 1.4 * L3).
3. Правила выхода: TP = 3.0R Net, БУ строго при +1.5R Net.
4. Портфельное тестирование: риск 0.35%, макс 3 и 4 сделки, кулдаун 6 свечей, лимит -5% в месяц.
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


def extract_refined_signals(
    mdata,
    target_r=3.0,
    be_r=1.5,
    min_score=70,
    min_range=0.45,
    max_w3_w5_dist=18,
    max_time_ratio=2.0,
    max_l5_l3_ratio=1.4,
    use_vol=True,
):
    signals = []
    rejected_rsi_blowout = 0
    rejected_time_bloat = 0
    rejected_price_bloat = 0
    
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
        
        # 1. SHORT (Медвежий разворот на W5)
        for p1 in high_pivots:
            if rsi[p1] < 60.0: continue
            
            # W5 должна созревать органично (от 4 до max_w3_w5_dist свечей)
            for t in range(p1 + 4, min(n - 40, p1 + max_w3_w5_dist + 1)):
                if high[t] <= high[p1] or high[t] < np.max(high[p1+1 : t+1]): continue
                if not (close[t] < open_p[t] and close[t] < min(open_p[t-1], close[t-1])): continue
                
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < min_range: continue
                if rsi[t] >= rsi[p1] - 2.5 or rsi[t] < 45.0: continue
                
                # ── КРИТИЧЕСКИЙ ФИЛЬТР 1: Чистота RSI (никаких вылетов в космос между W3 и W5!) ──
                max_mid_rsi = np.max(rsi[p1 + 1 : t])
                if max_mid_rsi > rsi[p1] + 1.0:
                    rejected_rsi_blowout += 1
                    continue
                    
                if use_vol and vol[t] < 1.1 * vol_ma[t]: continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:t+1], low=low[:t+1], close=close[:t+1], open_p=open_p[:t+1],
                    w3_bar=p1, w5_bar=t, direction='SHORT', rsi=rsi[:t+1]
                )
                if not algo_res.is_valid or algo_res.score < min_score: continue
                
                w_idx = algo_res.wave_indices
                w0_b, w3_b, w5_b = w_idx['W0'], w_idx['W3'], w_idx['W5']
                dur_03 = max(2, w3_b - w0_b)
                dur_35 = w5_b - w3_b
                
                # ── КРИТИЧЕСКИЙ ФИЛЬТР 2: Пропорциональность времени (W5 не раздута относительно W0-W3) ──
                if (dur_35 / dur_03) > max_time_ratio:
                    rejected_time_bloat += 1
                    continue
                    
                # ── КРИТИЧЕСКИЙ ФИЛЬТР 3: Пропорциональность амплитуды (L5 не больше L3) ──
                l3 = algo_res.details.get('l3', 0.0)
                l5 = algo_res.details.get('l5', 0.0)
                if l3 > 0 and (l5 / l3) > max_l5_l3_ratio:
                    rejected_price_bloat += 1
                    continue
                    
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
                    "score": algo_res.score, "dur_ratio": dur_35 / dur_03,
                })
                break
                
        # 2. LONG (Бычий разворот на W5)
        for p1 in low_pivots:
            if rsi[p1] > 40.0: continue
            
            for t in range(p1 + 4, min(n - 40, p1 + max_w3_w5_dist + 1)):
                if low[t] >= low[p1] or low[t] > np.min(low[p1+1 : t+1]): continue
                if not (close[t] > open_p[t] and close[t] > max(open_p[t-1], close[t-1])): continue
                
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < min_range: continue
                if rsi[t] <= rsi[p1] + 2.5 or rsi[t] > 55.0: continue
                
                # ── КРИТИЧЕСКИЙ ФИЛЬТР 1: Чистота RSI (никаких провалов глубже W3!) ──
                min_mid_rsi = np.min(rsi[p1 + 1 : t])
                if min_mid_rsi < rsi[p1] - 1.0:
                    rejected_rsi_blowout += 1
                    continue
                    
                if use_vol and vol[t] < 1.1 * vol_ma[t]: continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:t+1], low=low[:t+1], close=close[:t+1], open_p=open_p[:t+1],
                    w3_bar=p1, w5_bar=t, direction='LONG', rsi=rsi[:t+1]
                )
                if not algo_res.is_valid or algo_res.score < min_score: continue
                
                w_idx = algo_res.wave_indices
                w0_b, w3_b, w5_b = w_idx['W0'], w_idx['W3'], w_idx['W5']
                dur_03 = max(2, w3_b - w0_b)
                dur_35 = w5_b - w3_b
                
                if (dur_35 / dur_03) > max_time_ratio:
                    rejected_time_bloat += 1
                    continue
                    
                l3 = algo_res.details.get('l3', 0.0)
                l5 = algo_res.details.get('l5', 0.0)
                if l3 > 0 and (l5 / l3) > max_l5_l3_ratio:
                    rejected_price_bloat += 1
                    continue
                    
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
                    "score": algo_res.score, "dur_ratio": dur_35 / dur_03,
                })
                break
                
    signals.sort(key=lambda s: s["signal_time"])
    print(f"Отсеяно из-за вылета RSI выше W3: {rejected_rsi_blowout}")
    print(f"Отсеяно из-за раздувания времени W5 (dur > 2.0 * dur_03): {rejected_time_bloat}")
    print(f"Отсеяно из-за чрезмерной длины W5 (L5 > 1.4 * L3): {rejected_price_bloat}")
    return signals


def simulate(mdata, signals, max_concurrent=3, cooldown_bars=6, risk_pct=0.35, monthly_limit_pct=5.0):
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
    trade_id_seq = 0
    
    for bar_idx, cur_time in enumerate(timeline):
        m_str = cur_time.strftime("%Y-%m")
        if m_str != cur_month:
            cur_month = m_str
            month_start_depo = current_depo
            month_realized_pnl_pct = 0.0
            month_locked = False
            
        # 1. Update Active Positions
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
            if month_locked: continue
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
                continue
            if bar_idx < po["expire_bar"]:
                still_pending.append(po)
        pending_orders = still_pending
        
        # 3. New signals
        if cur_time in sigs_by_time:
            for sig in sigs_by_time[cur_time]:
                sym = sig["symbol"]
                if month_locked: continue
                last_b = last_trade_bar.get(sym, -999)
                if (bar_idx - last_b) < cooldown_bars: continue
                if any(p["symbol"] == sym for p in active_positions): continue
                if any(p["symbol"] == sym for p in pending_orders): continue
                pending_orders.append({
                    "symbol": sym, "direction": sig["direction"], "signal_time": cur_time,
                    "expire_bar": bar_idx + 2, "entry_p": sig["entry_p"], "sl_p": sig["sl_p"],
                    "tp_p": sig["tp_p"], "be_p": sig["be_p"], "trig_be": sig["trig_be"],
                    "target_r": sig["target_r"], "risk_frac": sig["risk_frac"],
                })
                
    return closed_trades


def summarize(trades, label):
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
    
    depo = 1000.0
    curve = [1000.0]
    for p in df['pnl_pct']:
        depo *= (1.0 + p / 100.0)
        curve.append(depo)
    s = pd.Series(curve)
    max_dd = ((s.cummax() - s) / s.cummax()).max() * 100.0
    
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
    print(f"Сделок: {n_t} ({n_t/22:.1f}/мес) | TP (3R): {w} ({w/n_t*100:.1f}%) | BE (1.5R): {b} ({b/n_t*100:.1f}%) | SL: {l} ({l/n_t*100:.1f}%)")
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
    
    print("\nГенерация сигналов с органическим импульсом и чистой дивергенцией...")
    refined_sigs = extract_refined_signals(
        mdata,
        target_r=3.0,          # ТЕЙК 3.0R
        be_r=1.5,              # БЕЗУБЫТОК 1.5R
        min_score=70,          # Порог качества Эллиотта
        min_range=0.45,        # Мин размер свечи 0.45%
        max_w3_w5_dist=18,     # Максимум 18 свечей между W3 и W5
        max_time_ratio=2.0,    # Время W5 не больше 2.0x времени W0-W3
        max_l5_l3_ratio=1.4,   # Размах W5 не больше 1.4x размаха W3
        use_vol=True,          # Фильтр объема
    )
    print(f"Итого отобрано идеальных сигналов: {len(refined_sigs)}")
    
    # Проверяем, отсеяна ли та самая сделка sl_5_near_short (2025-01-03 15:00)
    near_bad = [s for s in refined_sigs if s["symbol"] == "NEARUSDT" and s["signal_time"].strftime("%Y-%m-%d %H") == "2025-01-03 15"]
    print(f"\n[ПРОВЕРКА] Попала ли проблемная сделка NEAR (sl_5_near_short) в новый набор сигналов? -> {'ДА' if near_bad else 'НЕТ, УСПЕШНО ОТСЕЯНА!'}")
    
    configs = [
        (3, 0.35, 5.0,  "Органический импульс | Max 3 сделки | Риск 0.35% | Лимит -5%"),
        (4, 0.35, 5.0,  "Органический импульс | Max 4 сделки | Риск 0.35% | Лимит -5%"),
        (3, 0.35, None, "Органический импульс | Max 3 сделки | Риск 0.35% | БЕЗ лимита"),
        (4, 0.35, None, "Органический импульс | Max 4 сделки | Риск 0.35% | БЕЗ лимита"),
    ]
    
    results = []
    for max_c, r_pct, m_lim, lbl in configs:
        t_res = simulate(mdata, refined_sigs, max_concurrent=max_c, cooldown_bars=6, risk_pct=r_pct, monthly_limit_pct=m_lim)
        res = summarize(t_res, lbl)
        if res: results.append(res)
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(RESULTS_DIR / "refined_impulse_results_035.csv", index=False)
    print("\n" + "="*95)
    print("           РЕЗУЛЬТАТЫ ДОРАБОТАННОГО АЛГОРИТМА (TP=3.0R, BE=1.5R, РИСК=0.35%)")
    print("="*95)
    print(df_res[["label", "trades", "tp_pct", "be_pct", "sl_pct", "pf", "net_r", "depo", "growth_x", "max_dd_pct", "pos_m", "stop_m"]].to_string())
