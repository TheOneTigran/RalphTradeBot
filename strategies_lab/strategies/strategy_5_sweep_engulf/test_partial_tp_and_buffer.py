"""
test_partial_tp_and_buffer.py — Тестирование стратегии с:
1. Буфером на стоп-лосс 0.15% (защита от микро-свипов).
2. Частичным тейком 30% на +1.5R Net + перенос остатка 70% в безубыток.
3. Полным тейком оставшихся 70% на +3.0R Net.
4. Доработанным алгоритмом органического импульса (RSI purity, W5 limits).
5. Тестирование при риске 1.0% и 0.35%, Max 3 и Max 4 параллельных сделок, кулдаун 6 свечей.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

BASE_DIR = Path(r"c:\Users\user\Desktop\RalphTradeBot")
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from elliott_detector import detect_elliott_impulse
from simulate_rm_portfolio_v2 import load_all_market_data, SYMBOLS, FEE, RESULTS_DIR


def extract_signals_with_buffer(
    mdata,
    buffer_pct=0.0015,  # 0.15% буфер
    min_score=70,
    min_range=0.45,
    max_dist=18,
):
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
        vol_ma = pd.Series(vol).rolling(15).mean().values
        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]
        
        # 1. SHORT
        for p1 in high_pivots:
            if rsi[p1] < 60.0: continue
            for t in range(p1 + 4, min(n - 40, p1 + max_dist + 1)):
                if high[t] <= high[p1] or high[t] < np.max(high[p1+1 : t+1]): continue
                if not (close[t] < open_p[t] and close[t] < min(open_p[t-1], close[t-1])): continue
                
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < min_range: continue
                if rsi[t] >= rsi[p1] - 2.5 or rsi[t] < 45.0: continue
                
                # Фильтр чистоты RSI (никаких вылетов выше W3)
                if np.max(rsi[p1+1 : t]) > rsi[p1] + 1.0: continue
                if vol[t] < 1.1 * vol_ma[t]: continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:t+1], low=low[:t+1], close=close[:t+1], open_p=open_p[:t+1],
                    w3_bar=p1, w5_bar=t, direction='SHORT', rsi=rsi[:t+1]
                )
                if not algo_res.is_valid or algo_res.score < min_score: continue
                
                w_idx = algo_res.wave_indices
                dur_03 = max(2, w_idx['W3'] - w_idx['W0'])
                dur_35 = w_idx['W5'] - w_idx['W3']
                if (dur_35 / dur_03) > 2.0: continue
                
                l3 = algo_res.details.get('l3', 0.0)
                l5 = algo_res.details.get('l5', 0.0)
                if l3 > 0 and (l5 / l3) > 1.4: continue
                
                limit_entry = low[t] + 0.50 * c_range
                
                # СТОП-ЛОСС С БУФЕРОМ 0.15%
                sl_p = high[t] + buffer_pct * limit_entry
                risk_dist = sl_p - limit_entry
                if risk_dist <= 0: continue
                
                risk_frac = risk_dist / limit_entry
                
                # Цели с запасом на комиссии (0.05% вход + 0.05% выход)
                # TP1 (+1.5R Net)
                reward_frac_15 = (1.5 * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp1_p = limit_entry - limit_entry * reward_frac_15
                
                # TP2 (+3.0R Net)
                reward_frac_30 = (3.0 * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp2_p = limit_entry - limit_entry * reward_frac_30
                
                be_p = limit_entry * (1.0 - 2.0 * FEE)
                
                signals.append({
                    "symbol": sym, "direction": "SHORT", "signal_time": times[t],
                    "signal_bar_idx": t, "entry_p": limit_entry, "sl_p": sl_p,
                    "tp1_p": tp1_p, "tp2_p": tp2_p, "be_p": be_p,
                    "risk_frac": risk_frac, "score": algo_res.score,
                })
                break
                
        # 2. LONG
        for p1 in low_pivots:
            if rsi[p1] > 40.0: continue
            for t in range(p1 + 4, min(n - 40, p1 + max_dist + 1)):
                if low[t] >= low[p1] or low[t] > np.min(low[p1+1 : t+1]): continue
                if not (close[t] > open_p[t] and close[t] > max(open_p[t-1], close[t-1])): continue
                
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < min_range: continue
                if rsi[t] <= rsi[p1] + 2.5 or rsi[t] > 55.0: continue
                
                if np.min(rsi[p1+1 : t]) < rsi[p1] - 1.0: continue
                if vol[t] < 1.1 * vol_ma[t]: continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:t+1], low=low[:t+1], close=close[:t+1], open_p=open_p[:t+1],
                    w3_bar=p1, w5_bar=t, direction='LONG', rsi=rsi[:t+1]
                )
                if not algo_res.is_valid or algo_res.score < min_score: continue
                
                w_idx = algo_res.wave_indices
                dur_03 = max(2, w_idx['W3'] - w_idx['W0'])
                dur_35 = w_idx['W5'] - w_idx['W3']
                if (dur_35 / dur_03) > 2.0: continue
                
                l3 = algo_res.details.get('l3', 0.0)
                l5 = algo_res.details.get('l5', 0.0)
                if l3 > 0 and (l5 / l3) > 1.4: continue
                
                limit_entry = low[t] + 0.50 * c_range
                
                # СТОП-ЛОСС С БУФЕРОМ 0.15%
                sl_p = low[t] - buffer_pct * limit_entry
                risk_dist = limit_entry - sl_p
                if risk_dist <= 0: continue
                
                risk_frac = risk_dist / limit_entry
                
                # TP1 (+1.5R Net)
                reward_frac_15 = (1.5 * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp1_p = limit_entry + limit_entry * reward_frac_15
                
                # TP2 (+3.0R Net)
                reward_frac_30 = (3.0 * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp2_p = limit_entry + limit_entry * reward_frac_30
                
                be_p = limit_entry * (1.0 + 2.0 * FEE)
                
                signals.append({
                    "symbol": sym, "direction": "LONG", "signal_time": times[t],
                    "signal_bar_idx": t, "entry_p": limit_entry, "sl_p": sl_p,
                    "tp1_p": tp1_p, "tp2_p": tp2_p, "be_p": be_p,
                    "risk_frac": risk_frac, "score": algo_res.score,
                })
                break
                
    signals.sort(key=lambda s: s["signal_time"])
    return signals


def simulate_partial_portfolio(
    mdata,
    signals,
    max_concurrent=4,
    cooldown_bars=6,
    risk_pct=1.0,
    monthly_limit_pct=5.0,
    partial_tp1_ratio=0.30,  # 30% на TP1 (+1.5R)
):
    """
    Симуляция с частичной фиксацией:
    - 30% закрывается при +1.5R Net
    - Оставшиеся 70% переводятся в безубыток
    - 70% закрываются при +3.0R Net либо по безубытку (0.0R)
    """
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
            
        # 1. Обновление активных позиций
        still_active = []
        for pos in active_positions:
            df = mdata[pos["symbol"]]
            if cur_time not in df.index:
                still_active.append(pos); continue
            row = df.loc[cur_time]
            bh, bl = float(row['high']), float(row['low'])
            pos["bars_held"] += 1
            
            # Проверка уровней
            if pos["direction"] == "SHORT":
                hit_sl = bh >= pos["cur_sl"]
                hit_tp1 = bl <= pos["tp1_p"]
                hit_tp2 = bl <= pos["tp2_p"]
            else:
                hit_sl = bl <= pos["cur_sl"]
                hit_tp1 = bh >= pos["tp1_p"]
                hit_tp2 = bh >= pos["tp2_p"]
                
            if not pos["tp1_hit"]:
                # ── Фаза 1: До достижения TP1 (+1.5R) позиция защищена начальным стопом с буфером ──
                if hit_sl:
                    # Стоп-лосс сработал первым (консервативный расчет)
                    net_r = -(pos["risk_frac"] + 2.0 * FEE) / pos["risk_frac"]
                    trade_pnl_pct = net_r * risk_pct
                    closed_trades.append({
                        "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                        "entry_time": pos["entry_time"], "exit_time": cur_time,
                        "reason": "sl", "net_r": net_r, "pnl_pct": trade_pnl_pct,
                        "bars_held": pos["bars_held"], "month": m_str,
                    })
                    current_depo *= (1.0 + trade_pnl_pct / 100.0)
                    month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                    if monthly_limit_pct is not None and month_realized_pnl_pct <= -monthly_limit_pct:
                        month_locked = True
                        pending_orders.clear()
                    continue
                    
                if hit_tp2:
                    # Мощный импульс пробил сразу TP1 и TP2
                    net_r = partial_tp1_ratio * 1.5 + (1.0 - partial_tp1_ratio) * 3.0
                    trade_pnl_pct = net_r * risk_pct
                    closed_trades.append({
                        "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                        "entry_time": pos["entry_time"], "exit_time": cur_time,
                        "reason": "tp_full", "net_r": net_r, "pnl_pct": trade_pnl_pct,
                        "bars_held": pos["bars_held"], "month": m_str,
                    })
                    current_depo *= (1.0 + trade_pnl_pct / 100.0)
                    month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                    continue
                    
                if hit_tp1:
                    # Достигнут TP1 (+1.5R): фиксируем 30%, переносим стоп остатка 70% в безубыток
                    pos["tp1_hit"] = True
                    pos["cur_sl"] = pos["be_p"]
            else:
                # ── Фаза 2: 30% уже зафиксировано на TP1, остаток 70% на безубытке BE ──
                hit_be = (bh >= pos["be_p"]) if pos["direction"] == "SHORT" else (bl <= pos["be_p"])
                if hit_be:
                    # Сработал безубыток для 70%: 30%@1.5R + 70%@0.0R = +0.45R Net
                    net_r = partial_tp1_ratio * 1.5
                    trade_pnl_pct = net_r * risk_pct
                    closed_trades.append({
                        "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                        "entry_time": pos["entry_time"], "exit_time": cur_time,
                        "reason": "tp1_be", "net_r": net_r, "pnl_pct": trade_pnl_pct,
                        "bars_held": pos["bars_held"], "month": m_str,
                    })
                    current_depo *= (1.0 + trade_pnl_pct / 100.0)
                    month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                    continue
                    
                if hit_tp2:
                    # Достигнут полный TP2 (+3.0R): 30%@1.5R + 70%@3.0R = +2.55R Net
                    net_r = partial_tp1_ratio * 1.5 + (1.0 - partial_tp1_ratio) * 3.0
                    trade_pnl_pct = net_r * risk_pct
                    closed_trades.append({
                        "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                        "entry_time": pos["entry_time"], "exit_time": cur_time,
                        "reason": "tp_full", "net_r": net_r, "pnl_pct": trade_pnl_pct,
                        "bars_held": pos["bars_held"], "month": m_str,
                    })
                    current_depo *= (1.0 + trade_pnl_pct / 100.0)
                    month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                    continue
                
            # ── 1.4 Таймаут ──
            if pos["bars_held"] >= 120:
                net_r = (partial_tp1_ratio * 1.5) if pos["tp1_hit"] else 0.0
                trade_pnl_pct = net_r * risk_pct
                closed_trades.append({
                    "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                    "entry_time": pos["entry_time"], "exit_time": cur_time,
                    "reason": "timeout", "net_r": net_r, "pnl_pct": trade_pnl_pct,
                    "bars_held": pos["bars_held"], "month": m_str,
                })
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                continue
                
            still_active.append(pos)
        active_positions = still_active
        
        # 2. Проверка отложенных лимиток
        still_pending = []
        for po in pending_orders:
            if month_locked: continue
            df = mdata[po["symbol"]]
            if cur_time not in df.index:
                if bar_idx < po["expire_bar"]: still_pending.append(po)
                continue
            row = df.loc[cur_time]
            bh, bl = float(row['high']), float(row['low'])
            
            # Отмена если стоп (с буфером) задет до входа
            if (po["direction"] == "SHORT" and bh > po["sl_p"]) or (po["direction"] == "LONG" and bl < po["sl_p"]):
                continue
                
            filled = (bh >= po["entry_p"]) if po["direction"] == "SHORT" else (bl <= po["entry_p"])
            if filled:
                if len(active_positions) < max_concurrent:
                    trade_id_seq += 1
                    active_positions.append({
                        "id": trade_id_seq, "symbol": po["symbol"], "direction": po["direction"],
                        "entry_time": cur_time, "entry_p": po["entry_p"], "sl_p": po["sl_p"],
                        "cur_sl": po["sl_p"], "tp1_p": po["tp1_p"], "tp2_p": po["tp2_p"],
                        "be_p": po["be_p"], "risk_frac": po["risk_frac"],
                        "tp1_hit": False, "bars_held": 0,
                    })
                    last_trade_bar[po["symbol"]] = bar_idx
                continue
            if bar_idx < po["expire_bar"]:
                still_pending.append(po)
        pending_orders = still_pending
        
        # 3. Новые сигналы
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
                    "tp1_p": sig["tp1_p"], "tp2_p": sig["tp2_p"], "be_p": sig["be_p"],
                    "risk_frac": sig["risk_frac"],
                })
                
    return closed_trades


def analyze(trades, label):
    if not trades:
        print(f"[{label}] 0 сделок")
        return None
    df = pd.DataFrame(trades)
    n_t = len(df)
    
    tp_full = len(df[df['reason'] == "tp_full"])
    tp1_be = len(df[df['reason'] == "tp1_be"])
    sls = len(df[df['reason'] == "sl"])
    
    # Сделки в плюсе = tp_full (+2.55R) + tp1_be (+0.45R)
    profitable_trades = tp_full + tp1_be
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
    print(f"Сделок: {n_t} ({n_t/22:.1f}/мес) | Полных TP (+2.55R): {tp_full} ({tp_full/n_t*100:.1f}%) | Частичных БУ (+0.45R): {tp1_be} ({tp1_be/n_t*100:.1f}%) | Стопов (-1R): {sls} ({sls/n_t*100:.1f}%)")
    print(f"Доля ПРИБЫЛЬНЫХ сделок (TP2 + TP1_BE): {profitable_trades}/{n_t} ({profitable_trades/n_t*100:.1f}%)")
    print(f"Profit Factor: {pf:.2f} | Net R: {tot_r:+.2f} R | EV: {tot_r/n_t:+.3f} R")
    print(f"Депозит: ${depo:,.2f} ({depo/1000.0:+.2f}x) | Макс. просадка: {max_dd:.1f}%")
    print(f"Месяцев в плюсе: {pos_m} из {len(months)} | Месяцев стопа 5%: {stop_m}")
    
    return {
        "label": label, "trades": n_t,
        "tp_full_pct": round(tp_full/n_t*100, 1),
        "tp1_be_pct": round(tp1_be/n_t*100, 1),
        "sl_pct": round(sls/n_t*100, 1),
        "win_rate_total": round(profitable_trades/n_t*100, 1),
        "pf": round(pf, 2), "net_r": round(tot_r, 1),
        "depo": round(depo, 2), "growth_x": round(depo/1000.0, 2),
        "max_dd_pct": round(max_dd, 1), "pos_m": pos_m, "stop_m": stop_m
    }


if __name__ == "__main__":
    print("1. Загрузка данных и генерация сигналов с буфером 0.15%...")
    mdata = load_all_market_data()
    signals_buf = extract_signals_with_buffer(mdata, buffer_pct=0.0015, min_score=70)
    print(f"Найдено качественных сигналов: {len(signals_buf)}")
    
    configs = [
        # max_c, risk_pct, monthly_limit, label
        (4, 1.0, 5.0,  "Риск 1.0% | Max 4 сделки | Частичный тейк (30%@1.5R + 70%@3R) + Буфер 0.15% | ЛИМИТ 5%"),
        (4, 1.0, None, "Риск 1.0% | Max 4 сделки | Частичный тейк (30%@1.5R + 70%@3R) + Буфер 0.15% | БЕЗ лимита"),
        (3, 1.0, 5.0,  "Риск 1.0% | Max 3 сделки | Частичный тейк (30%@1.5R + 70%@3R) + Буфер 0.15% | ЛИМИТ 5%"),
        (3, 1.0, None, "Риск 1.0% | Max 3 сделки | Частичный тейк (30%@1.5R + 70%@3R) + Буфер 0.15% | БЕЗ лимита"),
        (4, 0.35, 5.0, "Риск 0.35% | Max 4 сделки | Частичный тейк (30%@1.5R + 70%@3R) + Буфер 0.15% | ЛИМИТ 5%"),
        (4, 0.35, None,"Риск 0.35% | Max 4 сделки | Частичный тейк (30%@1.5R + 70%@3R) + Буфер 0.15% | БЕЗ лимита"),
        (3, 0.35, 5.0, "Риск 0.35% | Max 3 сделки | Частичный тейк (30%@1.5R + 70%@3R) + Буфер 0.15% | ЛИМИТ 5%"),
        (3, 0.35, None,"Риск 0.35% | Max 3 сделки | Частичный тейк (30%@1.5R + 70%@3R) + Буфер 0.15% | БЕЗ лимита"),
    ]
    
    summary = []
    for max_c, r_pct, m_lim, lbl in configs:
        trades = simulate_partial_portfolio(
            mdata, signals_buf, max_concurrent=max_c, cooldown_bars=6,
            risk_pct=r_pct, monthly_limit_pct=m_lim, partial_tp1_ratio=0.30
        )
        res = analyze(trades, lbl)
        if res: summary.append(res)
        
    df_s = pd.DataFrame(summary)
    df_s.to_csv(RESULTS_DIR / "partial_tp_and_buffer_results.csv", index=False)
    print("\n" + "="*110)
    print("      ИТОГОВАЯ СВОДНАЯ ТАБЛИЦА: ЧАСТИЧНЫЙ ТЕЙК (30%@1.5R + 70%@3R) + БУФЕР СТОПА 0.15%")
    print("="*110)
    print(df_s[["label", "trades", "win_rate_total", "tp_full_pct", "tp1_be_pct", "sl_pct", "pf", "net_r", "depo", "growth_x", "max_dd_pct", "pos_m", "stop_m"]].to_string())
