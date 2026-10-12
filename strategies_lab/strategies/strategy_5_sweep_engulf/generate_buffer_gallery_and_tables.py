"""
generate_buffer_gallery_and_tables.py — Генерация помесячных таблиц и графиков для модели с:
1. Буфером 0.15% на стоп-лосс.
2. Фиксацией 30% на +1.5R Net, перенос остатка 70% в безубыток.
3. Добитием 70% на +3.0R Net (+2.55R суммарно) или выходом в безубыток (+0.45R суммарно).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import argrelextrema

BASE_DIR = Path(r"c:\Users\user\Desktop\RalphTradeBot")
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from elliott_detector import detect_elliott_impulse
from simulate_rm_portfolio_v2 import load_all_market_data, SYMBOLS, FEE, RESULTS_DIR
from test_partial_tp_and_buffer import simulate_partial_portfolio

ARTIFACTS_DIR = Path(r"C:\Users\user\.gemini\antigravity-ide\brain\2249d323-d907-4b4e-ae7d-74c8bae47d0a")


def get_signals(mdata, buffer_pct=0.0015):
    signals = []
    for sym in SYMBOLS:
        df = mdata[sym]
        high, low, close, open_p, vol = df['high'].values, df['low'].values, df['close'].values, df['open'].values, df['volume'].values
        n, times = len(df), df.index
        rsi = calculate_rsi_wilder(close, period=7)
        vol_ma = pd.Series(vol).rolling(15).mean().values
        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]
        
        # SHORT
        for p1 in high_pivots:
            if rsi[p1] < 60.0: continue
            for t in range(p1 + 4, min(n - 40, p1 + 19)):
                if high[t] <= high[p1] or high[t] < np.max(high[p1+1 : t+1]): continue
                if not (close[t] < open_p[t] and close[t] < min(open_p[t-1], close[t-1])): continue
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < 0.45: continue
                if rsi[t] >= rsi[p1] - 2.5 or rsi[t] < 45.0: continue
                if np.max(rsi[p1+1 : t]) > rsi[p1] + 1.0: continue
                if vol[t] < 1.1 * vol_ma[t]: continue
                algo_res = detect_elliott_impulse(high=high[:t+1], low=low[:t+1], close=close[:t+1], open_p=open_p[:t+1], w3_bar=p1, w5_bar=t, direction='SHORT', rsi=rsi[:t+1])
                if not algo_res.is_valid or algo_res.score < 70: continue
                w_idx = algo_res.wave_indices
                dur_03 = max(2, w_idx['W3'] - w_idx['W0'])
                dur_35 = w_idx['W5'] - w_idx['W3']
                if (dur_35 / dur_03) > 2.0: continue
                l3 = algo_res.details.get('l3', 0.0)
                l5 = algo_res.details.get('l5', 0.0)
                if l3 > 0 and (l5 / l3) > 1.4: continue
                
                limit_entry = low[t] + 0.50 * c_range
                base_risk_dist = high[t] - limit_entry
                sl_p = high[t] + buffer_pct * limit_entry
                
                actual_risk_frac = (sl_p - limit_entry) / limit_entry
                tp_risk_frac = base_risk_dist / limit_entry
                reward_frac_15 = (1.5 * tp_risk_frac + 2.0 * FEE) / (1.0 - FEE)
                reward_frac_30 = (3.0 * tp_risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp1_p = limit_entry - limit_entry * reward_frac_15
                tp2_p = limit_entry - limit_entry * reward_frac_30
                be_p = limit_entry * (1.0 - 2.0 * FEE)
                
                signals.append({
                    'symbol': sym, 'direction': 'SHORT', 'signal_time': times[t],
                    'signal_bar_idx': t, 'entry_p': limit_entry, 'sl_p': sl_p,
                    'orig_sl_p': high[t],
                    'tp1_p': tp1_p, 'tp2_p': tp2_p, 'be_p': be_p,
                    'risk_frac': actual_risk_frac, 'score': algo_res.score,
                    'algo_res': algo_res, 'candle_s_idx': t,
                })
                break
                
        # LONG
        for p1 in low_pivots:
            if rsi[p1] > 40.0: continue
            for t in range(p1 + 4, min(n - 40, p1 + 19)):
                if low[t] >= low[p1] or low[t] > np.min(low[p1+1 : t+1]): continue
                if not (close[t] > open_p[t] and close[t] > max(open_p[t-1], close[t-1])): continue
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < 0.45: continue
                if rsi[t] <= rsi[p1] + 2.5 or rsi[t] > 55.0: continue
                if np.min(rsi[p1+1 : t]) < rsi[p1] - 1.0: continue
                if vol[t] < 1.1 * vol_ma[t]: continue
                algo_res = detect_elliott_impulse(high=high[:t+1], low=low[:t+1], close=close[:t+1], open_p=open_p[:t+1], w3_bar=p1, w5_bar=t, direction='LONG', rsi=rsi[:t+1])
                if not algo_res.is_valid or algo_res.score < 70: continue
                w_idx = algo_res.wave_indices
                dur_03 = max(2, w_idx['W3'] - w_idx['W0'])
                dur_35 = w_idx['W5'] - w_idx['W3']
                if (dur_35 / dur_03) > 2.0: continue
                l3 = algo_res.details.get('l3', 0.0)
                l5 = algo_res.details.get('l5', 0.0)
                if l3 > 0 and (l5 / l3) > 1.4: continue
                
                limit_entry = low[t] + 0.50 * c_range
                base_risk_dist = limit_entry - low[t]
                sl_p = low[t] - buffer_pct * limit_entry
                
                actual_risk_frac = (limit_entry - sl_p) / limit_entry
                tp_risk_frac = base_risk_dist / limit_entry
                reward_frac_15 = (1.5 * tp_risk_frac + 2.0 * FEE) / (1.0 - FEE)
                reward_frac_30 = (3.0 * tp_risk_frac + 2.0 * FEE) / (1.0 - FEE)
                tp1_p = limit_entry + limit_entry * reward_frac_15
                tp2_p = limit_entry + limit_entry * reward_frac_30
                be_p = limit_entry * (1.0 + 2.0 * FEE)
                
                signals.append({
                    'symbol': sym, 'direction': 'LONG', 'signal_time': times[t],
                    'signal_bar_idx': t, 'entry_p': limit_entry, 'sl_p': sl_p,
                    'orig_sl_p': low[t],
                    'tp1_p': tp1_p, 'tp2_p': tp2_p, 'be_p': be_p,
                    'risk_frac': actual_risk_frac, 'score': algo_res.score,
                    'algo_res': algo_res, 'candle_s_idx': t,
                })
                break
    signals.sort(key=lambda s: s['signal_time'])
    return signals


def simulate_with_meta(mdata, signals, max_concurrent=4, cooldown_bars=6, risk_pct=1.0, monthly_limit_pct=5.0):
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
                hit_tp1 = bl <= pos["tp1_p"]
                hit_tp2 = bl <= pos["tp2_p"]
                # Проверяем, спасла ли позицию подушка буфера (был ли пробит orig_sl, но НЕ пробит cur_sl)
                if bh > pos["orig_sl_p"] and not hit_sl:
                    pos["saved_by_buffer"] = True
            else:
                hit_sl = bl <= pos["cur_sl"]
                hit_tp1 = bh >= pos["tp1_p"]
                hit_tp2 = bh >= pos["tp2_p"]
                if bl < pos["orig_sl_p"] and not hit_sl:
                    pos["saved_by_buffer"] = True
                
            if not pos["tp1_hit"]:
                if hit_sl:
                    net_r = -(pos["risk_frac"] + 2.0 * FEE) / pos["risk_frac"]
                    trade_pnl_pct = net_r * risk_pct
                    closed_trades.append({
                        **pos, "exit_time": cur_time, "reason": "sl", "net_r": net_r,
                        "pnl_pct": trade_pnl_pct, "month": m_str, "exit_p": pos["cur_sl"]
                    })
                    current_depo *= (1.0 + trade_pnl_pct / 100.0)
                    month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                    if monthly_limit_pct is not None and month_realized_pnl_pct <= -monthly_limit_pct:
                        month_locked = True
                        pending_orders.clear()
                    continue
                if hit_tp2:
                    net_r = 0.30 * 1.5 + 0.70 * 3.0
                    trade_pnl_pct = net_r * risk_pct
                    closed_trades.append({
                        **pos, "exit_time": cur_time, "reason": "tp_full", "net_r": net_r,
                        "pnl_pct": trade_pnl_pct, "month": m_str, "exit_p": pos["tp2_p"]
                    })
                    current_depo *= (1.0 + trade_pnl_pct / 100.0)
                    month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                    continue
                if hit_tp1:
                    pos["tp1_hit"] = True
                    pos["cur_sl"] = pos["be_p"]
            else:
                hit_be = (bh >= pos["be_p"]) if pos["direction"] == "SHORT" else (bl <= pos["be_p"])
                if hit_be:
                    net_r = 0.30 * 1.5
                    trade_pnl_pct = net_r * risk_pct
                    closed_trades.append({
                        **pos, "exit_time": cur_time, "reason": "tp1_be", "net_r": net_r,
                        "pnl_pct": trade_pnl_pct, "month": m_str, "exit_p": pos["be_p"]
                    })
                    current_depo *= (1.0 + trade_pnl_pct / 100.0)
                    month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                    continue
                if hit_tp2:
                    net_r = 0.30 * 1.5 + 0.70 * 3.0
                    trade_pnl_pct = net_r * risk_pct
                    closed_trades.append({
                        **pos, "exit_time": cur_time, "reason": "tp_full", "net_r": net_r,
                        "pnl_pct": trade_pnl_pct, "month": m_str, "exit_p": pos["tp2_p"]
                    })
                    current_depo *= (1.0 + trade_pnl_pct / 100.0)
                    month_realized_pnl_pct = ((current_depo - month_start_depo) / month_start_depo) * 100.0
                    continue
                    
            if pos["bars_held"] >= 120:
                net_r = (0.30 * 1.5) if pos["tp1_hit"] else 0.0
                trade_pnl_pct = net_r * risk_pct
                closed_trades.append({
                    **pos, "exit_time": cur_time, "reason": "timeout", "net_r": net_r,
                    "pnl_pct": trade_pnl_pct, "month": m_str, "exit_p": pos["entry_p"]
                })
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                continue
            still_active.append(pos)
        active_positions = still_active
        
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
                        **po, "id": trade_id_seq, "entry_time": cur_time,
                        "cur_sl": po["sl_p"], "tp1_hit": False, "saved_by_buffer": False,
                        "bars_held": 0,
                    })
                    last_trade_bar[po["symbol"]] = bar_idx
                continue
            if bar_idx < po["expire_bar"]:
                still_pending.append(po)
        pending_orders = still_pending
        
        if cur_time in sigs_by_time:
            for sig in sigs_by_time[cur_time]:
                sym = sig["symbol"]
                if month_locked: continue
                last_b = last_trade_bar.get(sym, -999)
                if (bar_idx - last_b) < cooldown_bars: continue
                if any(p["symbol"] == sym for p in active_positions): continue
                if any(p["symbol"] == sym for p in pending_orders): continue
                pending_orders.append({
                    **sig, "expire_bar": bar_idx + 2,
                })
    return closed_trades


def render_buffer_chart(trade, mdata, out_path, title_note=""):
    sym = trade["symbol"]
    direction = trade["direction"]
    reason = trade["reason"]
    algo_res = trade["algo_res"]
    
    df = mdata[sym]
    high, low, close, open_p, times = df['high'].values, df['low'].values, df['close'].values, df['open'].values, df.index
    rsi = calculate_rsi_wilder(close, period=7)
    
    w_idx = algo_res.wave_indices
    w0_b, w5_b = w_idx["W0"], w_idx["W5"]
    exit_t = trade["exit_time"]
    exit_b = df.index.get_loc(exit_t) if exit_t in df.index else w5_b + trade["bars_held"] + 1
    
    start_b = max(0, w0_b - 8)
    end_b = min(len(df), exit_b + 12)
    
    s_h, s_l, s_c, s_o, s_rsi = high[start_b:end_b], low[start_b:end_b], close[start_b:end_b], open_p[start_b:end_b], rsi[start_b:end_b]
    n_bars = len(s_h)
    
    fig, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(14, 8), dpi=130, gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08})
    bg_color, grid_color, text_color = "#131722", "#202636", "#d1d4dc"
    up_c, down_c = "#089981", "#f23645"
    
    fig.patch.set_facecolor(bg_color)
    for ax in (ax_p, ax_r):
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle=":", alpha=0.6)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values(): spine.set_color(grid_color)
        
    for i in range(n_bars):
        c, o, h, l = s_c[i], s_o[i], s_h[i], s_l[i]
        col = up_c if c >= o else down_c
        ax_p.plot([i, i], [l, h], color=col, linewidth=1.1, zorder=2)
        b_bottom, b_height = min(o, c), max(abs(c - o), (h - l) * 0.02)
        ax_p.add_patch(patches.Rectangle((i - 0.3, b_bottom), 0.6, b_height, facecolor=col, edgecolor=col, linewidth=0.8, zorder=3))
        
    # Волны
    w_coords = []
    for w_name in ["W0", "W1", "W2", "W3", "W4", "W5"]:
        b = w_idx[w_name] - start_b
        p = s_l[b] if (direction == "SHORT" and w_name in ["W0", "W2", "W4"]) or (direction == "LONG" and w_name in ["W1", "W3", "W5"]) else s_h[b]
        w_coords.append((b, p))
        offset = (s_h[b] - s_l[b]) * 0.75
        va = "bottom" if p == s_h[b] else "top"
        y_text = p + offset if va == "bottom" else p - offset
        ax_p.text(b, y_text, w_name, color="#f59e0b", fontsize=10, fontweight="bold", ha="center", va=va, zorder=5)
        
    w_xs, w_ys = zip(*w_coords)
    ax_p.plot(w_xs, w_ys, color="#f59e0b", linestyle="--", linewidth=1.5, alpha=0.85, zorder=4)
    
    # Линии сделки
    ent_b = df.index.get_loc(trade["entry_time"]) - start_b
    ex_b = exit_b - start_b
    ax_p.plot([ent_b, ex_b], [trade["entry_p"], trade["entry_p"]], color="#facc15", linestyle="-.", linewidth=1.8, label=f"50% Entry: {trade['entry_p']:.4f}")
    ax_p.plot([ent_b, ex_b], [trade["orig_sl_p"], trade["orig_sl_p"]], color="#ea580c", linestyle=":", linewidth=1.4, alpha=0.7, label=f"Wick Tip SL: {trade['orig_sl_p']:.4f}")
    ax_p.plot([ent_b, ex_b], [trade["sl_p"], trade["sl_p"]], color="#ef4444", linestyle="--", linewidth=1.8, label=f"Buffered SL (+0.15%): {trade['sl_p']:.4f}")
    ax_p.plot([ent_b, ex_b], [trade["tp1_p"], trade["tp1_p"]], color="#38bdf8", linestyle="--", linewidth=1.8, label=f"TP1 (+1.5R, 30%): {trade['tp1_p']:.4f}")
    ax_p.plot([ent_b, ex_b], [trade["tp2_p"], trade["tp2_p"]], color="#10b981", linestyle="-", linewidth=2.0, label=f"TP2 (+3.0R, 70%): {trade['tp2_p']:.4f}")
    
    # Метка исхода
    res_label = f"FULL TP: +2.55R Net" if reason == "tp_full" else (f"PARTIAL BE: +0.45R Net" if reason == "tp1_be" else "STOP-LOSS: -1.0R Net")
    res_color = "#10b981" if reason == "tp_full" else ("#38bdf8" if reason == "tp1_be" else "#ef4444")
    ax_p.scatter([ex_b], [trade["exit_p"]], color=res_color, s=120, zorder=6, edgecolor="#ffffff", linewidth=1.5)
    ax_p.annotate(f"{res_label}\n({trade['bars_held']} баров)", xy=(ex_b, trade["exit_p"]), xytext=(ex_b + 1, trade["exit_p"]),
                  color="#ffffff", fontsize=9, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=res_color, edgecolor="none", alpha=0.9), zorder=6)
                  
    ax_p.legend(loc="upper left", facecolor="#1e222d", edgecolor="#2a2e39", labelcolor=text_color, fontsize=8)
    ax_p.set_title(f"{sym} {direction} | {title_note} | {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}", color="#ffffff", fontsize=12, fontweight="bold", pad=10)
    
    # RSI
    ax_r.plot(s_rsi, color="#8b5cf6", linewidth=1.5)
    ax_r.axhline(70, color="#f43f5e", linestyle=":", alpha=0.5)
    ax_r.axhline(50, color="#64748b", linestyle=":", alpha=0.4)
    ax_r.axhline(30, color="#10b981", linestyle=":", alpha=0.5)
    ax_r.set_ylabel("Wilder RSI (7)", color=text_color, fontsize=9)
    ax_r.set_ylim(10, 90)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, facecolor=bg_color)
    plt.close()
    print(f"График сохранен: {out_path.name}")


if __name__ == "__main__":
    print("Загрузка данных и симуляция с метаданными...")
    mdata = load_all_market_data()
    signals = get_signals(mdata, buffer_pct=0.0015)
    
    trades_1pct = simulate_with_meta(mdata, signals, max_concurrent=4, cooldown_bars=6, risk_pct=1.0, monthly_limit_pct=5.0)
    df_t = pd.DataFrame(trades_1pct)
    df_t.to_csv(RESULTS_DIR / "trades_buffer_015_partial_1pct.csv", index=False)
    
    # 1. Помесячная таблица
    months = sorted(df_t['month'].unique())
    m_rows = []
    depo = 1000.0
    for m in months:
        sub = df_t[df_t['month'] == m]
        start_d = depo
        for p in sub['pnl_pct']: depo *= (1.0 + p / 100.0)
        pct = (depo - start_d) / start_d * 100.0
        tp2 = len(sub[sub['reason'] == "tp_full"])
        be = len(sub[sub['reason'] == "tp1_be"])
        sl = len(sub[sub['reason'] == "sl"])
        m_rows.append({"month": m, "trades": len(sub), "tp2": tp2, "be": be, "sl": sl, "net_r": round(sub['net_r'].sum(), 2), "pnl_pct": round(pct, 2), "end_depo": round(depo, 2)})
    df_m = pd.DataFrame(m_rows)
    df_m.to_csv(RESULTS_DIR / "monthly_buffer_015_partial_1pct.csv", index=False)
    
    print("\nПомесячные результаты (Риск 1.0% | Max 4 | Лимит 5%):")
    print(df_m.to_string(index=False))
    
    # 2. Ищем спасенные буфером сделки
    saved_tps = [t for t in trades_1pct if t.get("saved_by_buffer") and t["reason"] == "tp_full"]
    print(f"\nНайдено сделок, СПАСЕННЫХ буфером 0.15% и дошедших до полного TP2: {len(saved_tps)}")
    
    if saved_tps:
        sample_saved = saved_tps[0]
        render_buffer_chart(sample_saved, mdata, ARTIFACTS_DIR / "buffer_saved_tp_sol.png", title_note="СПАСЕНО БУФЕРОМ 0.15% -> ПОЛНЫЙ ТЕЙК +2.55R")
        
    be_trades = [t for t in trades_1pct if t["reason"] == "tp1_be"]
    if be_trades:
        sample_be = be_trades[0]
        render_buffer_chart(sample_be, mdata, ARTIFACTS_DIR / "buffer_partial_be_eth.png", title_note="ЧАСТИЧНЫЙ ТЕЙК 30% НА +1.5R + БЕЗУБЫТОК 70% (+0.45R)")
