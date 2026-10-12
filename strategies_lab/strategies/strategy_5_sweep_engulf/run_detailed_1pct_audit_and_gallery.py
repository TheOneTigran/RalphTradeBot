"""
run_detailed_1pct_audit_and_gallery.py — Помесячный анализ просадок при риске 1% и генерация галереи графиков.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import argrelextrema

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"c:\Users\user\Desktop\RalphTradeBot")
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from elliott_detector import detect_elliott_impulse
from simulate_rm_portfolio_v2 import load_all_market_data, SYMBOLS, FEE, RESULTS_DIR

ARTIFACTS_DIR = Path(r"C:\Users\user\.gemini\antigravity-ide\brain\2249d323-d907-4b4e-ae7d-74c8bae47d0a")


def extract_refined_signals(mdata, target_r=3.0, be_r=1.5, min_score=70, min_range=0.45, max_dist=18):
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
        
        # SHORT
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
                    "score": algo_res.score, "algo_res": algo_res, "p1": p1, "t": t,
                })
                break
                
        # LONG
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
                    "score": algo_res.score, "algo_res": algo_res, "p1": p1, "t": t,
                })
                break
                
    signals.sort(key=lambda s: s["signal_time"])
    return signals


def simulate_portfolio_detailed(mdata, signals, max_concurrent=4, cooldown_bars=6, risk_pct=1.0):
    timeline = next(iter(mdata.values())).index
    sigs_by_time = {}
    for s in signals:
        sigs_by_time.setdefault(s["signal_time"], []).append(s)
        
    active_positions = []
    pending_orders = []
    closed_trades = []
    last_trade_bar = {}
    current_depo = 1000.0
    trade_id_seq = 0
    
    for bar_idx, cur_time in enumerate(timeline):
        m_str = cur_time.strftime("%Y-%m")
        
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
                    "entry_p": pos["entry_p"], "sl_p": pos["sl_p"], "tp_p": pos["tp_p"],
                    "trig_be": pos["trig_be"], "exit_p": pos["cur_sl"],
                    "reason": reason, "net_r": net_r, "pnl_pct": trade_pnl_pct,
                    "bars_held": pos["bars_held"], "month": m_str,
                    "sig_meta": pos["sig_meta"],
                })
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                continue
                
            if hit_tp:
                trade_pnl_pct = pos["target_r"] * risk_pct
                closed_trades.append({
                    "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                    "entry_time": pos["entry_time"], "exit_time": cur_time,
                    "entry_p": pos["entry_p"], "sl_p": pos["sl_p"], "tp_p": pos["tp_p"],
                    "trig_be": pos["trig_be"], "exit_p": pos["tp_p"],
                    "reason": "tp", "net_r": pos["target_r"], "pnl_pct": trade_pnl_pct,
                    "bars_held": pos["bars_held"], "month": m_str,
                    "sig_meta": pos["sig_meta"],
                })
                current_depo *= (1.0 + trade_pnl_pct / 100.0)
                continue
                
            if hit_be and not pos["be_active"]:
                pos["be_active"] = True
                pos["cur_sl"] = pos["be_p"]
                
            if pos["bars_held"] >= 120:
                closed_trades.append({
                    "id": pos["id"], "symbol": pos["symbol"], "direction": pos["direction"],
                    "entry_time": pos["entry_time"], "exit_time": cur_time,
                    "entry_p": pos["entry_p"], "sl_p": pos["sl_p"], "tp_p": pos["tp_p"],
                    "trig_be": pos["trig_be"], "exit_p": float(row['close']),
                    "reason": "timeout", "net_r": 0.0, "pnl_pct": 0.0,
                    "bars_held": pos["bars_held"], "month": m_str,
                    "sig_meta": pos["sig_meta"],
                })
                continue
            still_active.append(pos)
        active_positions = still_active
        
        # 2. Pending orders
        still_pending = []
        for po in pending_orders:
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
                        "sig_meta": po["sig_meta"],
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
                last_b = last_trade_bar.get(sym, -999)
                if (bar_idx - last_b) < cooldown_bars: continue
                if any(p["symbol"] == sym for p in active_positions): continue
                if any(p["symbol"] == sym for p in pending_orders): continue
                pending_orders.append({
                    "symbol": sym, "direction": sig["direction"], "signal_time": cur_time,
                    "expire_bar": bar_idx + 2, "entry_p": sig["entry_p"], "sl_p": sig["sl_p"],
                    "tp_p": sig["tp_p"], "be_p": sig["be_p"], "trig_be": sig["trig_be"],
                    "target_r": sig["target_r"], "risk_frac": sig["risk_frac"],
                    "sig_meta": sig,
                })
                
    return closed_trades


def build_monthly_drawdown_table(closed_trades):
    df = pd.DataFrame(closed_trades)
    months = sorted(df['month'].unique())
    rows = []
    
    cum_depo = 1000.0
    for m in months:
        sub = df[df['month'] == m].copy()
        n_t = len(sub)
        w = len(sub[sub['reason'] == "tp"])
        b = len(sub[sub['reason'] == "be"])
        l = len(sub[sub['reason'] == "sl"])
        net_r = sub['net_r'].sum()
        
        start_d = cum_depo
        intramonth_curve = [start_d]
        for p in sub['pnl_pct']:
            cum_depo *= (1.0 + p / 100.0)
            intramonth_curve.append(cum_depo)
            
        s = pd.Series(intramonth_curve)
        peak = s.cummax()
        intra_dd = ((peak - s) / peak * 100.0).max()
        m_growth = ((cum_depo - start_d) / start_d) * 100.0
        
        rows.append({
            "month": m,
            "trades": n_t,
            "tp": w,
            "be": b,
            "sl": l,
            "win_rate": round(w / n_t * 100.0, 1) if n_t > 0 else 0.0,
            "net_r": round(net_r, 2),
            "month_pnl_pct": round(m_growth, 2),
            "max_intra_dd_pct": round(intra_dd, 2),
            "end_depo": round(cum_depo, 2),
        })
        
    df_m = pd.DataFrame(rows)
    return df, df_m


def render_tradingview_chart(trade_row, mdata, out_path):
    sym = trade_row["symbol"]
    direction = trade_row["direction"]
    reason = trade_row["reason"]
    sig_meta = trade_row["sig_meta"]
    algo_res = sig_meta["algo_res"]
    
    df = mdata[sym]
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_p = df['open'].values
    times = df.index
    
    rsi = calculate_rsi_wilder(close, period=7)
    
    w_idx = algo_res.wave_indices
    w0_b = w_idx["W0"]
    w5_b = w_idx["W5"]
    
    # Срез для графика: от W0 - 10 свечей до выхода + 12 свечей
    exit_t = trade_row["exit_time"]
    exit_b = df.index.get_loc(exit_t) if exit_t in df.index else w5_b + trade_row["bars_held"] + 1
    
    start_b = max(0, w0_b - 10)
    end_b = min(len(df), exit_b + 12)
    
    slice_high = high[start_b:end_b]
    slice_low = low[start_b:end_b]
    slice_close = close[start_b:end_b]
    slice_open = open_p[start_b:end_b]
    slice_rsi = rsi[start_b:end_b]
    n_slice = len(slice_high)
    
    x_w0 = w0_b - start_b
    x_w1 = w_idx["W1"] - start_b
    x_w2 = w_idx["W2"] - start_b
    x_w3 = w_idx["W3"] - start_b
    x_w4 = w_idx["W4"] - start_b
    x_w5 = w5_b - start_b
    x_exit = min(n_slice - 1, exit_b - start_b)
    
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, figsize=(14, 8), dpi=130,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08}
    )
    
    bg_color = "#131722"
    grid_color = "#202636"
    text_color = "#d1d4dc"
    up_color = "#089981"
    down_color = "#f23645"
    
    fig.patch.set_facecolor(bg_color)
    for ax in (ax_price, ax_rsi):
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle=":", alpha=0.6)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
            
    # Рисуем свечи
    bar_width = 0.62
    for i in range(n_slice):
        o, c, h, l = slice_open[i], slice_close[i], slice_high[i], slice_low[i]
        c_color = up_color if c >= o else down_color
        ax_price.plot([i, i], [l, h], color=c_color, linewidth=1.2, zorder=2)
        bottom = min(o, c)
        height = max(abs(c - o), (slice_high.max() - slice_low.min()) * 0.001)
        rect = patches.Rectangle(
            (i - bar_width / 2.0, bottom), bar_width, height,
            facecolor=c_color, edgecolor=c_color, linewidth=0.5, zorder=3
        )
        ax_price.add_patch(rect)
        
    # Соединяем волны W0..W5
    wave_x = [x_w0, x_w1, x_w2, x_w3, x_w4, x_w5]
    wave_y = [
        algo_res.wave_points["W0"], algo_res.wave_points["W1"],
        algo_res.wave_points["W2"], algo_res.wave_points["W3"],
        algo_res.wave_points["W4"], algo_res.wave_points["W5"],
    ]
    ax_price.plot(wave_x, wave_y, color="#00d4ff", linewidth=2.0, alpha=0.9, zorder=4)
    
    for wx, wy, w_lbl in zip(wave_x, wave_y, ["W0", "W1", "W2", "W3", "W4", "W5"]):
        ax_price.plot(wx, wy, "o", color="#00d4ff", markersize=6, zorder=5)
        offset_y = (slice_high.max() - slice_low.min()) * 0.03
        va_val = "bottom" if wy >= np.median(wave_y) else "top"
        txt_y = wy + offset_y if va_val == "bottom" else wy - offset_y
        ax_price.text(
            wx, txt_y, w_lbl, color="#ffffff", fontweight="bold", fontsize=9,
            ha="center", va=va_val,
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="#005b82", edgecolor="#00d4ff", lw=1.2),
            zorder=6
        )
        
    # Уровни ордера
    entry_p = trade_row["entry_p"]
    sl_p = trade_row["sl_p"]
    tp_p = trade_row["tp_p"]
    trig_be = trade_row["trig_be"]
    
    # Линии
    x_span = range(x_w5, x_exit + 4)
    ax_price.hlines(entry_p, x_w5, x_exit + 3, colors="#ffd700", linestyles="-.", linewidth=1.6, label="Вход 50% (Limit)")
    ax_price.hlines(sl_p, x_w5, x_exit + 3, colors="#ff4d4d", linestyles="--", linewidth=1.6, label="Стоп-Лосс")
    ax_price.hlines(tp_p, x_w5, x_exit + 3, colors="#00ff88", linestyles="-", linewidth=1.8, label="Тейк-Профит 3.0R")
    ax_price.hlines(trig_be, x_w5, x_exit + 3, colors="#00bfff", linestyles=":", linewidth=1.4, label="Триггер БУ (+1.5R)")
    
    # Точка входа и выхода
    ax_price.plot(x_w5 + 1, entry_p, "D", color="#ffd700", markersize=7, zorder=7)
    
    exit_icon = "X" if reason == "sl" else ("*" if reason == "tp" else "o")
    exit_col = "#ff4d4d" if reason == "sl" else ("#00ff88" if reason == "tp" else "#00bfff")
    ax_price.plot(x_exit, trade_row["exit_p"], exit_icon, color=exit_col, markersize=9, markeredgewidth=2, zorder=8)
    
    # Заголовок
    status_map = {"tp": "ТЕЙК-ПРОФИТ (+3.0R)", "be": "БЕЗУБЫТОК (0.0R)", "sl": "СТОП-ЛОСС (-1.0R)"}
    status_str = status_map.get(reason, reason.upper())
    title_str = (
        f"{sym} 15m · {direction} · {status_str} | "
        f"Вход: {entry_p:.4f} | SL: {sl_p:.4f} | TP: {tp_p:.4f} | "
        f"Удержание: {trade_row['bars_held']} свечей"
    )
    ax_price.set_title(title_str, color=text_color, fontsize=12, fontweight="bold", pad=12)
    ax_price.legend(loc="upper left", facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color, fontsize=8)
    
    # ── ПАНЕЛЬ RSI ──
    ax_rsi.plot(range(n_slice), slice_rsi, color="#a080ff", linewidth=1.6)
    ax_rsi.axhline(70, color="#f23645", linestyle=":", alpha=0.5)
    ax_rsi.axhline(30, color="#089981", linestyle=":", alpha=0.5)
    ax_rsi.set_ylim(15, 85)
    ax_rsi.set_ylabel("RSI(7)", color=text_color, fontsize=9)
    
    # Линия дивергенции на RSI
    r_w3 = slice_rsi[x_w3]
    r_w5 = slice_rsi[x_w5]
    ax_rsi.plot([x_w3, x_w5], [r_w3, r_w5], color="#ffd700", linestyle="--", linewidth=1.8, marker="o", markersize=5)
    
    plt.tight_layout()
    plt.savefig(out_path, facecolor=bg_color, edgecolor="none", dpi=130)
    plt.close()


if __name__ == "__main__":
    print("Загрузка данных и генерация...")
    mdata = load_all_market_data()
    signals = extract_refined_signals(mdata, target_r=3.0, be_r=1.5, min_score=70)
    print(f"Всего отобрано органических сигналов: {len(signals)}")
    
    print("\nСимуляция портфеля: Риск 1.0% | Max 4 сделки...")
    closed_trades = simulate_portfolio_detailed(mdata, signals, max_concurrent=4, cooldown_bars=6, risk_pct=1.0)
    df_trades, df_monthly = build_monthly_drawdown_table(closed_trades)
    
    df_trades.to_csv(RESULTS_DIR / "refined_trades_1pct_detailed.csv", index=False)
    df_monthly.to_csv(RESULTS_DIR / "refined_monthly_1pct_drawdowns.csv", index=False)
    
    print("\n" + "="*85)
    print("               ПОМЕСЯЧНАЯ СТАТИСТИКА И ПРОСАДКИ (РИСК 1.0%)")
    print("="*85)
    print(df_monthly.to_string())
    
    # Генерация галереи графиков
    print("\nГенерация 8 графиков TradingView Dark (Тейки, Безубытки, Стопы)...")
    
    # Выбираем разнородные примеры
    tp_samples = [t for t in closed_trades if t["reason"] == "tp" and t["symbol"] in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "APTUSDT")][:3]
    be_samples = [t for t in closed_trades if t["reason"] == "be" and t["symbol"] in ("NEARUSDT", "LINKUSDT", "AVAXUSDT")][:2]
    sl_samples = [t for t in closed_trades if t["reason"] == "sl" and t["symbol"] in ("BTCUSDT", "SOLUSDT", "DOGEUSDT")][:3]
    
    selected_gallery = tp_samples + be_samples + sl_samples
    chart_files = []
    
    for idx, tr in enumerate(selected_gallery, start=1):
        s_name = f"refined_gallery_{idx}_{tr['reason']}_{tr['symbol'].lower()}.png"
        out_p = ARTIFACTS_DIR / s_name
        render_tradingview_chart(tr, mdata, out_p)
        chart_files.append((s_name, tr))
        print(f"  [+] Сгенерирован график {idx}: {s_name} ({tr['symbol']} · {tr['direction']} · {tr['reason'].upper()})")
