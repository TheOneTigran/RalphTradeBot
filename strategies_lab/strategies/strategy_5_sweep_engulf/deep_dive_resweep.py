"""
deep_dive_resweep.py — Глубокий анализ механики пересвипа предположительной 5-й волны:
1. Сохранение RSI-дивергенции на свече пересвипа: RSI(S) < RSI(W3).
2. Фильтр чистоты поглощения: относительный размер тела, объем.
3. Проверка параметров выхода: Take-Profit (1.5R, 2.0R, 3.0R), БУ (1.0R, 1.5R, 2.0R, или без БУ).
4. Оценка ретрейсмента входа: 38.2%, 50%, 61.8%, по рынку на закрытии S.
"""
import sys
from pathlib import Path
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
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'NEARUSDT',
    'SUIUSDT', 'AAVEUSDT', 'APTUSDT', 'DOTUSDT', 'RENDERUSDT',
]

FEE = 0.0005


def run_deep_dive():
    mdata = {}
    for sym in SYMBOLS:
        p = DATA_DIR / f"{sym}_15m.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if 'open_time' in df.columns:
            df.index = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        elif 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'], utc=True)
        else:
            df.index = pd.to_datetime(df.iloc[:, 0], utc=True)
        mdata[sym] = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"Загружено {len(mdata)} монет.")

    raw_candidates = []

    for sym in SYMBOLS:
        df = mdata[sym]
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        close = df['close'].values.astype(np.float64)
        open_p = df['open'].values.astype(np.float64)
        n = len(df)

        rsi = calculate_rsi_wilder(close, period=7)

        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]

        # SHORT
        if len(high_pivots) >= 2:
            for i in range(1, len(high_pivots)):
                p1 = int(high_pivots[i - 1])
                p2 = int(high_pivots[i])

                conf_bar = p2 + 3
                if conf_bar >= n - 20:
                    continue

                dist = p2 - p1
                if dist < 4 or dist > 45:
                    continue
                if high[p2] <= high[p1]:
                    continue

                r1, r2 = float(rsi[p1]), float(rsi[p2])
                if r2 >= r1 or (r1 - r2) < 2.5:
                    continue
                if r1 < 60.0 or r2 < 45.0:
                    continue

                algo_res = detect_elliott_impulse(
                    high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                    w3_bar=p1, w5_bar=p2, direction="SHORT", rsi=rsi[:p2 + 1],
                )
                if not algo_res.is_valid or algo_res.score < 65:
                    continue

                w5_high = high[p2]
                w4_p = algo_res.wave_points.get("W4", low[p2])

                # Мониторим свечи t в окне до 12 баров
                for t in range(conf_bar, min(n - 20, p2 + 13)):
                    if low[t] < w4_p:
                        break  # пробит W4 — отмена

                    if high[t] > w5_high:
                        prev_body_bottom = min(open_p[t - 1], close[t - 1])
                        is_bearish = close[t] < open_p[t]
                        is_engulf = close[t] < prev_body_bottom

                        if is_bearish and is_engulf:
                            # Проверяем, сохраняется ли RSI дивергенция на свече S!
                            s_rsi = float(rsi[t])
                            div_preserved = s_rsi < r1 - 2.0

                            raw_candidates.append({
                                "symbol": sym,
                                "direction": "SHORT",
                                "w3_bar": p1,
                                "w5_bar": p2,
                                "sweep_bar": t,
                                "s_high": high[t],
                                "s_low": low[t],
                                "s_open": open_p[t],
                                "s_close": close[t],
                                "w3_rsi": r1,
                                "s_rsi": s_rsi,
                                "div_preserved": div_preserved,
                            })
                            break
                        else:
                            w5_high = high[t]

        # LONG
        if len(low_pivots) >= 2:
            for i in range(1, len(low_pivots)):
                p1 = int(low_pivots[i - 1])
                p2 = int(low_pivots[i])

                conf_bar = p2 + 3
                if conf_bar >= n - 20:
                    continue

                dist = p2 - p1
                if dist < 4 or dist > 45:
                    continue
                if low[p2] >= low[p1]:
                    continue

                r1, r2 = float(rsi[p1]), float(rsi[p2])
                if r2 <= r1 or (r2 - r1) < 2.5:
                    continue
                if r1 > 40.0 or r2 > 55.0:
                    continue

                algo_res = detect_elliott_impulse(
                    high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                    w3_bar=p1, w5_bar=p2, direction="LONG", rsi=rsi[:p2 + 1],
                )
                if not algo_res.is_valid or algo_res.score < 65:
                    continue

                w5_low = low[p2]
                w4_p = algo_res.wave_points.get("W4", high[p2])

                for t in range(conf_bar, min(n - 20, p2 + 13)):
                    if high[t] > w4_p:
                        break

                    if low[t] < w5_low:
                        prev_body_top = max(open_p[t - 1], close[t - 1])
                        is_bullish = close[t] > open_p[t]
                        is_engulf = close[t] > prev_body_top

                        if is_bullish and is_engulf:
                            s_rsi = float(rsi[t])
                            div_preserved = s_rsi > r1 + 2.0

                            raw_candidates.append({
                                "symbol": sym,
                                "direction": "LONG",
                                "w3_bar": p1,
                                "w5_bar": p2,
                                "sweep_bar": t,
                                "s_high": high[t],
                                "s_low": low[t],
                                "s_open": open_p[t],
                                "s_close": close[t],
                                "w3_rsi": r1,
                                "s_rsi": s_rsi,
                                "div_preserved": div_preserved,
                            })
                            break
                        else:
                            w5_low = low[t]

    print(f"Всего выявлено сетапов пересвипа: {len(raw_candidates)}")
    div_kept_count = sum(1 for c in raw_candidates if c["div_preserved"])
    print(f"Из них с сохранением RSI-дивергенции на свече S: {div_kept_count} ({div_kept_count/len(raw_candidates)*100:.1f}%)")

    # Теперь симулируем разные вариации:
    # 1. Все сетапы vs Только с сохраненной дивергенцией
    # 2. Вход 50% vs Вход по рынку на закрытии S (0% откат) vs 38.2%
    # 3. Тейк 2R vs 3R
    # 4. БУ на 1.5R vs 2R vs Без БУ
    
    for div_only in [False, True]:
        for entry_mode in ["limit_50", "market_close", "limit_382"]:
            for target_r in [2.0, 3.0]:
                for be_trigger_r in [1.5, 2.0, 999.0]:
                    trades_res = []
                    
                    for c in raw_candidates:
                        if div_only and not c["div_preserved"]:
                            continue
                        
                        df = mdata[c["symbol"]]
                        n = len(df)
                        high = df['high'].values
                        low = df['low'].values
                        sb = c["sweep_bar"]
                        d = c["direction"]
                        
                        s_h, s_l = c["s_high"], c["s_low"]
                        s_range = s_h - s_l
                        if s_range <= 0:
                            continue
                        
                        if entry_mode == "market_close":
                            entry_p = c["s_close"]
                            fill_bar = sb + 1
                        elif entry_mode == "limit_50":
                            entry_p = s_l + 0.50 * s_range
                            fill_bar = None
                            for fb in (sb + 1, sb + 2):
                                if fb >= n: break
                                if d == "SHORT" and high[fb] >= entry_p:
                                    fill_bar = fb; break
                                if d == "LONG" and low[fb] <= entry_p:
                                    fill_bar = fb; break
                        else:  # limit_382
                            retrace = 0.382 if d == "SHORT" else 0.618  # 38.2% отката от хая или от лоя
                            entry_p = s_l + 0.618 * s_range if d == "SHORT" else s_l + 0.382 * s_range
                            fill_bar = None
                            for fb in (sb + 1, sb + 2):
                                if fb >= n: break
                                if d == "SHORT" and high[fb] >= entry_p:
                                    fill_bar = fb; break
                                if d == "LONG" and low[fb] <= entry_p:
                                    fill_bar = fb; break
                        
                        if fill_bar is None or fill_bar >= n:
                            continue
                        
                        sl_p = s_h if d == "SHORT" else s_l
                        risk_dist = abs(entry_p - sl_p)
                        if risk_dist <= 0:
                            continue
                        
                        risk_frac = risk_dist / entry_p
                        reward_frac = (target_r * risk_frac + 2.0 * FEE) / (1.0 - FEE)
                        tp_p = entry_p - entry_p * reward_frac if d == "SHORT" else entry_p + entry_p * reward_frac
                        
                        trig_be = entry_p - be_trigger_r * risk_dist if d == "SHORT" else entry_p + be_trigger_r * risk_dist
                        be_price = entry_p * (1.0 - 2.0 * FEE) if d == "SHORT" else entry_p * (1.0 + 2.0 * FEE)
                        
                        be_active = False
                        cur_sl = sl_p
                        res = "timeout"
                        
                        for b in range(fill_bar, min(n, fill_bar + 160)):
                            bh, bl = high[b], low[b]
                            
                            # Консервативно: стоп первым
                            if (d == "SHORT" and bh >= cur_sl) or (d == "LONG" and bl <= cur_sl):
                                res = "be" if be_active else "sl"
                                break
                            
                            if (d == "SHORT" and bl <= tp_p) or (d == "LONG" and bh >= tp_p):
                                res = "tp"
                                break
                            
                            if not be_active and be_trigger_r < 900:
                                if (d == "SHORT" and bl <= trig_be) or (d == "LONG" and bh >= trig_be):
                                    be_active = True
                                    cur_sl = be_price
                        
                        if res == "tp":
                            r_val = target_r
                        elif res == "be":
                            r_val = 0.0
                        elif res == "sl":
                            r_val = -(risk_frac + 2.0 * FEE) / risk_frac
                        else:
                            r_val = 0.0
                        
                        trades_res.append(r_val)
                    
                    if len(trades_res) >= 20:
                        n_t = len(trades_res)
                        w = sum(1 for r in trades_res if r > 0)
                        tot_r = sum(trades_res)
                        gw = sum(r for r in trades_res if r > 0)
                        gl = abs(sum(r for r in trades_res if r < 0))
                        pf = (gw / gl) if gl > 0 else 99.0
                        print(f"[DivOnly={div_only} | {entry_mode:<12} | TP={target_r}R | BE={be_trigger_r}R] -> N={n_t:<3}, WR={w/n_t*100:4.1f}%, PF={pf:4.2f}, SumR={tot_r:+6.1f}R, EV={tot_r/n_t:+5.2f}R")

if __name__ == "__main__":
    run_deep_dive()
