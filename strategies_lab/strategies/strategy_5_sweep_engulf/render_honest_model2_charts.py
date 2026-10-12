"""
render_honest_model2_charts.py — Рендер чистых графиков реального времени (Model 2):
Демонстрация того, как бот на закрытии свечи W5 (Candle S) фиксирует импульс,
начинает отслеживать откат к 50%, входит лимиткой и забирает тейк.
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

DATA_DIR = Path(r"C:\Users\user\Desktop\данные для бектестов\2025_2026_Full_Top_Coins")
ARTIFACTS_DIR = Path(r"C:\Users\user\.gemini\antigravity-ide\brain\2249d323-d907-4b4e-ae7d-74c8bae47d0a")

FEE = 0.0005


def load_one_symbol(sym: str) -> pd.DataFrame:
    p = DATA_DIR / f"{sym}_15m.csv"
    df = pd.read_csv(p)
    if 'open_time' in df.columns:
        df.index = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    elif 'timestamp' in df.columns:
        df.index = pd.to_datetime(df['timestamp'], utc=True)
    else:
        df.index = pd.to_datetime(df.iloc[:, 0], utc=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def find_prime_trades():
    # Ищем красивые примеры на BTCUSDT и SOLUSDT
    examples = []
    
    for sym in ["BTCUSDT", "SOLUSDT", "ETHUSDT", "APTUSDT"]:
        df = load_one_symbol(sym)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_p = df['open'].values
        n = len(df)
        
        rsi = calculate_rsi_wilder(close, period=7)
        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]
        
        # SHORT
        for p1 in high_pivots:
            r1 = float(rsi[p1])
            if r1 < 62.0: continue
            
            for t in range(p1 + 5, min(n - 40, p1 + 35)):
                if high[t] <= high[p1]: continue
                if high[t] < np.max(high[p1+1 : t+1]): continue
                
                is_bearish = close[t] < open_p[t]
                prev_body_bottom = min(open_p[t-1], close[t-1])
                is_engulf = close[t] < prev_body_bottom
                if not (is_bearish and is_engulf): continue
                
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < 0.45: continue
                
                r_t = float(rsi[t])
                if r_t >= r1 - 3.0 or r_t < 45.0: continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:t + 1], low=low[:t + 1], close=close[:t + 1], open_p=open_p[:t + 1],
                    w3_bar=p1, w5_bar=t, direction="SHORT", rsi=rsi[:t + 1],
                )
                if not algo_res.is_valid or algo_res.score < 70: continue
                
                limit_entry = low[t] + 0.50 * c_range
                sl_p = high[t]
                
                # Заполнение
                fill_bar = None
                for fb in (t + 1, t + 2):
                    if fb >= n: break
                    if high[fb] > sl_p: break
                    if high[fb] >= limit_entry:
                        fill_bar = fb
                        break
                
                if fill_bar is None: continue
                
                # Симуляция
                risk_dist = sl_p - limit_entry
                reward_frac = (3.0 * (risk_dist / limit_entry) + 2.0 * FEE) / (1.0 - FEE)
                tp_p = limit_entry - limit_entry * reward_frac
                trig_2r = limit_entry - 2.0 * risk_dist
                be_p = limit_entry * (1.0 - 2.0 * FEE)
                
                cur_sl = sl_p
                be_active = False
                res = "timeout"
                exit_bar = fill_bar
                
                for b in range(fill_bar, min(n, fill_bar + 100)):
                    bh, bl = high[b], low[b]
                    if bh >= cur_sl:
                        res = "be" if be_active else "sl"
                        exit_bar = b
                        break
                    if bl <= tp_p:
                        res = "tp"
                        exit_bar = b
                        break
                    if bl <= trig_2r and not be_active:
                        be_active = True
                        cur_sl = be_p
                
                if res == "tp":
                    examples.append({
                        "sym": sym,
                        "dir": "SHORT",
                        "p1": p1,
                        "w5_bar": t,
                        "fill_bar": fill_bar,
                        "exit_bar": exit_bar,
                        "res": res,
                        "entry_p": limit_entry,
                        "sl_p": sl_p,
                        "tp_p": tp_p,
                        "algo_res": algo_res,
                        "df": df,
                        "rsi": rsi,
                    })
                    break
        
        # LONG
        for p1 in low_pivots:
            r1 = float(rsi[p1])
            if r1 > 38.0: continue
            
            for t in range(p1 + 5, min(n - 40, p1 + 35)):
                if low[t] >= low[p1]: continue
                if low[t] > np.min(low[p1+1 : t+1]): continue
                
                is_bullish = close[t] > open_p[t]
                prev_body_top = max(open_p[t-1], close[t-1])
                is_engulf = close[t] > prev_body_top
                if not (is_bullish and is_engulf): continue
                
                c_range = high[t] - low[t]
                if (c_range / close[t]) * 100.0 < 0.45: continue
                
                r_t = float(rsi[t])
                if r_t <= r1 + 3.0 or r_t > 55.0: continue
                
                algo_res = detect_elliott_impulse(
                    high=high[:t + 1], low=low[:t + 1], close=close[:t + 1], open_p=open_p[:t + 1],
                    w3_bar=p1, w5_bar=t, direction="LONG", rsi=rsi[:t + 1],
                )
                if not algo_res.is_valid or algo_res.score < 70: continue
                
                limit_entry = low[t] + 0.50 * c_range
                sl_p = low[t]
                
                fill_bar = None
                for fb in (t + 1, t + 2):
                    if fb >= n: break
                    if low[fb] < sl_p: break
                    if low[fb] <= limit_entry:
                        fill_bar = fb
                        break
                
                if fill_bar is None: continue
                
                risk_dist = limit_entry - sl_p
                reward_frac = (3.0 * (risk_dist / limit_entry) + 2.0 * FEE) / (1.0 - FEE)
                tp_p = limit_entry + limit_entry * reward_frac
                trig_2r = limit_entry + 2.0 * risk_dist
                be_p = limit_entry * (1.0 + 2.0 * FEE)
                
                cur_sl = sl_p
                be_active = False
                res = "timeout"
                exit_bar = fill_bar
                
                for b in range(fill_bar, min(n, fill_bar + 100)):
                    bh, bl = high[b], low[b]
                    if bl <= cur_sl:
                        res = "be" if be_active else "sl"
                        exit_bar = b
                        break
                    if bh >= tp_p:
                        res = "tp"
                        exit_bar = b
                        break
                    if bh >= trig_2r and not be_active:
                        be_active = True
                        cur_sl = be_p
                
                if res == "tp":
                    examples.append({
                        "sym": sym,
                        "dir": "LONG",
                        "p1": p1,
                        "w5_bar": t,
                        "fill_bar": fill_bar,
                        "exit_bar": exit_bar,
                        "res": res,
                        "entry_p": limit_entry,
                        "sl_p": sl_p,
                        "tp_p": tp_p,
                        "algo_res": algo_res,
                        "df": df,
                        "rsi": rsi,
                    })
                    break
        
        if len(examples) >= 4:
            break
            
    return examples


def plot_clean_model2_trade(ex: dict, out_name: str, subtitle: str):
    df = ex["df"]
    rsi = ex["rsi"]
    algo_res = ex["algo_res"]
    wi = algo_res.wave_indices
    wp = algo_res.wave_points
    direction = ex["dir"]
    sym = ex["sym"]
    
    w0_b = wi.get("W0", ex["p1"] - 20)
    w5_b = ex["w5_bar"]
    exit_b = ex["exit_bar"]
    
    # Окно графика: чуть до W0 и чуть после выхода
    start_b = max(0, w0_b - 12)
    end_b = min(len(df), exit_b + 12)
    
    sub_df = df.iloc[start_b:end_b].copy().reset_index(drop=True)
    n_bars = len(sub_df)
    
    bg_dark = "#131722"
    grid_clr = "#252936"
    text_clr = "#D1D4DC"
    bull_clr = "#089981"
    bear_clr = "#F23645"
    wave_clr = "#00E5FF" if direction == "SHORT" else "#FF4081"
    gold_clr = "#FFD700"
    tp_clr = "#00E676"
    sl_clr = "#FF5252"
    
    fig, (ax_main, ax_rsi) = plt.subplots(
        2, 1, figsize=(15, 8.5), dpi=120,
        gridspec_kw={"height_ratios": [3.3, 1.0], "hspace": 0.08},
        facecolor=bg_dark,
    )
    
    for ax in (ax_main, ax_rsi):
        ax.set_facecolor(bg_dark)
        ax.grid(True, color=grid_clr, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.tick_params(colors=text_clr, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(grid_clr)
            
    opens = sub_df['open'].values
    highs = sub_df['high'].values
    lows = sub_df['low'].values
    closes = sub_df['close'].values
    
    # 1. ЧИСТЫЕ СВЕЧИ (без рамок, без лишнего мусора)
    for i in range(n_bars):
        c_o, c_h, c_l, c_c = opens[i], highs[i], lows[i], closes[i]
        c_col = bull_clr if c_c >= c_o else bear_clr
        ax_main.plot([i, i], [c_l, c_h], color=c_col, linewidth=1.1, zorder=2)
        bottom = min(c_o, c_c)
        h_rect = abs(c_c - c_o)
        if h_rect == 0:
            h_rect = (c_h - c_l) * 0.03
        rect = patches.Rectangle(
            (i - 0.32, bottom), 0.64, h_rect,
            facecolor=c_col, edgecolor=c_col, linewidth=0.6, zorder=3
        )
        ax_main.add_patch(rect)
        
    # 2. Волны Эллиотта (тонкая аккуратная линия W0..W5)
    wave_pts = []
    for w_name in ["W0", "W1", "W2", "W3", "W4", "W5"]:
        if w_name in wi and w_name in wp:
            abs_idx = wi[w_name]
            rel_idx = abs_idx - start_b
            if 0 <= rel_idx < n_bars:
                wave_pts.append((rel_idx, wp[w_name], w_name))
                
    if len(wave_pts) >= 2:
        xs = [p[0] for p in wave_pts]
        ys = [p[1] for p in wave_pts]
        ax_main.plot(xs, ys, color=wave_clr, linewidth=1.6, linestyle="-", alpha=0.8, zorder=4)
        
        span_y = highs.max() - lows.min()
        for rel_idx, p_y, w_name in wave_pts:
            ax_main.scatter(rel_idx, p_y, color=wave_clr, s=40, zorder=5, edgecolors="#FFFFFF", linewidth=0.8)
            is_peak = (direction == "SHORT" and w_name in ("W1", "W3", "W5")) or (direction == "LONG" and w_name in ("W0", "W2", "W4"))
            offset = span_y * 0.035 * (1 if is_peak else -1)
            ax_main.text(
                rel_idx, p_y + offset, w_name,
                color="#FFFFFF", fontsize=9, fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="circle,pad=0.2", facecolor="#131722", edgecolor=wave_clr, linewidth=1.0)
            )

    # 3. ЗОНА МОНИТОРИНГА: показываем аккуратную тонкую линию пробоя W3
    w3_rel = wi["W3"] - start_b
    w4_rel = wi.get("W4", ex["p1"] + 2) - start_b
    s_rel = w5_b - start_b
    
    # Зона ожидания разворота: когда W4 завершилась и цена вышла на перехай W3
    if 0 <= w4_rel < n_bars and 0 <= s_rel < n_bars:
        ax_main.axvspan(w4_rel, s_rel, color="#FFD700", alpha=0.04, zorder=1)
        mid_x = (w4_rel + s_rel) / 2.0
        y_top = highs.max() - span_y * 0.05
        ax_main.text(
            mid_x, y_top, "Зона формирования W5 / ожидание поглощения",
            color="#FFD700", fontsize=8.5, ha="center", style="italic", alpha=0.85, zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E222D", edgecolor="#FFD700", linewidth=0.6, alpha=0.8)
        )

    # 4. СВЕЧА ПОГЛОЩЕНИЯ (Candle S) на W5
    # Подсвечиваем свечу S ненавязчивой стрелкой сверху/снизу
    s_h = highs[s_rel]
    s_l = lows[s_rel]
    arrow_dir = 1 if direction == "SHORT" else -1
    ax_main.annotate(
        "⚡ Свеча S (Поглощение)",
        xy=(s_rel, s_h if direction == "SHORT" else s_l),
        xytext=(s_rel, (s_h if direction == "SHORT" else s_l) + arrow_dir * span_y * 0.06),
        arrowprops=dict(arrowstyle="->", color="#00E5FF", lw=1.2),
        color="#00E5FF", fontsize=9, fontweight="bold", ha="center", zorder=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#1E222D", edgecolor="#00E5FF", linewidth=1.0)
    )

    # 5. Уровни сделки (Лимит 50%, Стоп, Тейк)
    entry_rel = ex["fill_bar"] - start_b
    exit_rel = ex["exit_bar"] - start_b
    
    ax_main.hlines(ex["entry_p"], s_rel, min(n_bars - 1, exit_rel + 4), colors=gold_clr, linestyles="-.", linewidth=1.4, zorder=5)
    ax_main.hlines(ex["sl_p"], s_rel, min(n_bars - 1, exit_rel + 4), colors=sl_clr, linestyles="--", linewidth=1.4, zorder=5)
    ax_main.hlines(ex["tp_p"], s_rel, min(n_bars - 1, exit_rel + 4), colors=tp_clr, linestyles="-", linewidth=1.6, zorder=5)
    
    # Метка входа на 50%
    if 0 <= entry_rel < n_bars:
        ax_main.scatter(entry_rel, ex["entry_p"], color=gold_clr, s=80, marker="D", zorder=9, edgecolors="#000000", linewidth=1.0)
        ax_main.text(
            entry_rel, ex["entry_p"] - arrow_dir * span_y * 0.035,
            f"Вход 50% (${ex['entry_p']:,.2f})",
            color=gold_clr, fontsize=8.5, fontweight="bold", ha="center", zorder=10
        )
        
    # Метка тейка
    if 0 <= exit_rel < n_bars:
        ax_main.scatter(exit_rel, ex["tp_p"], color=tp_clr, s=90, marker="X", zorder=9, edgecolors="#FFFFFF", linewidth=1.0)
        ax_main.text(
            exit_rel, ex["tp_p"] + arrow_dir * span_y * 0.035,
            f"TAKE-PROFIT +3.0R (${ex['tp_p']:,.2f})",
            color=tp_clr, fontsize=9, fontweight="bold", ha="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#131722", edgecolor=tp_clr, linewidth=1.0)
        )
        
    # 6. Панель RSI с дивергенцией W3 vs W5
    sub_rsi = rsi[start_b:end_b]
    ax_rsi.plot(range(len(sub_rsi)), sub_rsi, color="#7B61FF", linewidth=1.3, zorder=3)
    ax_rsi.axhline(70, color="#FF6B6B", linestyle=":", linewidth=0.8)
    ax_rsi.axhline(30, color="#51CF66", linestyle=":", linewidth=0.8)
    ax_rsi.set_ylim(15, 85)
    ax_rsi.set_ylabel("RSI (7)", color=text_clr, fontsize=9)
    
    # Линия дивергенции на RSI
    if 0 <= w3_rel < len(sub_rsi) and 0 <= s_rel < len(sub_rsi):
        r_w3 = sub_rsi[w3_rel]
        r_w5 = sub_rsi[s_rel]
        ax_rsi.plot([w3_rel, s_rel], [r_w3, r_w5], color=gold_clr, linewidth=1.6, linestyle="--", zorder=4)
        ax_rsi.scatter([w3_rel, s_rel], [r_w3, r_w5], color=gold_clr, s=35, zorder=5)
        ax_rsi.text((w3_rel + s_rel)/2.0, (r_w3 + r_w5)/2.0 + 4, "RSI DIV", color=gold_clr, fontsize=8, ha="center", fontweight="bold")

    stop_pct = abs(ex["entry_p"] - ex["sl_p"]) / ex["entry_p"] * 100.0
    ax_main.set_title(
        f"RalphTradeBot · {sym} 15m · {direction} (Model 2 Real-Time) | {subtitle}\n"
        f"Вход (50% S): ${ex['entry_p']:,.2f} │ SL: ${ex['sl_p']:,.2f} ({stop_pct:.2f}%) │ TP (3R Net): ${ex['tp_p']:,.2f}",
        color="#FFFFFF", fontsize=11, fontweight="bold", pad=12
    )
    
    out_path = ARTIFACTS_DIR / out_name
    plt.savefig(out_path, bbox_inches="tight", facecolor=bg_dark)
    plt.close()
    print(f"График сохранен: {out_path}")


def main():
    print("Поиск эталонных сделок Model 2...")
    examples = find_prime_trades()
    print(f"Найдено примеров: {len(examples)}")
    
    short_ex = next((e for e in examples if e["dir"] == "SHORT"), None)
    long_ex = next((e for e in examples if e["dir"] == "LONG"), None)
    
    if short_ex:
        plot_clean_model2_trade(
            short_ex,
            "model2_realtime_short_btc.png",
            "SHORT по закрытию поглощения на W5 (без заглядывания в будущее)"
        )
        
    if long_ex:
        plot_clean_model2_trade(
            long_ex,
            "model2_realtime_long_eth.png",
            "LONG по закрытию поглощения на W5 (без заглядывания в будущее)"
        )


if __name__ == "__main__":
    main()
