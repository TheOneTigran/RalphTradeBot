"""
test_parallel_detect.py — Тест параллельного сканирования импульсов Эллиотта.
"""
import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))

from signal_scanner import calculate_rsi_wilder
from scipy.signal import argrelextrema
from elliott_detector import detect_elliott_impulse


def scan_single_symbol(args):
    sym, p_15m, min_score = args
    df = pd.read_pickle(p_15m)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
    high = df['high'].values.astype(np.float64)
    low = df['low'].values.astype(np.float64)
    close = df['close'].values.astype(np.float64)
    open_p = df['open'].values.astype(np.float64)
    n = len(df)
    if n < 100:
        return sym, []
        
    rsi = calculate_rsi_wilder(close, period=7)
    high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
    low_pivots = argrelextrema(low, np.less_equal, order=3)[0]
    
    signals = []
    
    # SHORT
    if len(high_pivots) >= 2:
        for i in range(1, len(high_pivots)):
            p1, p2 = int(high_pivots[i - 1]), int(high_pivots[i])
            dist = p2 - p1
            if dist < 6 or dist > 45 or p2 >= n - 15:
                continue
            if high[p2] <= high[p1]:
                continue
            r1, r2 = float(rsi[p1]), float(rsi[p2])
            if r2 >= r1 or (r1 - r2) < 2.5 or r1 < 64.0 or r2 < 64.0:
                continue
            if len(high[p1 + 1 : p2]) > 0 and np.max(high[p1 + 1 : p2]) > high[p2]:
                continue
            res = detect_elliott_impulse(
                high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                w3_bar=p1, w5_bar=p2, direction="SHORT", rsi=rsi[:p2 + 1]
            )
            if res.is_valid and res.score >= min_score:
                w0_bar = res.wave_indices.get("W0", max(0, p1 - 20))
                signals.append({
                    "symbol": sym, "direction": "SHORT", "w5_bar": p2, "w5_time": df.index[p2],
                    "w5_price": float(high[p2]), "w3_bar": p1, "w3_price": float(high[p1]),
                    "w0_bar": w0_bar, "w0_price": res.wave_points.get("W0", float(low[w0_bar])),
                    "algo_score": res.score, "wave_points": res.wave_points
                })
                
    # LONG
    if len(low_pivots) >= 2:
        for i in range(1, len(low_pivots)):
            p1, p2 = int(low_pivots[i - 1]), int(low_pivots[i])
            dist = p2 - p1
            if dist < 6 or dist > 45 or p2 >= n - 15:
                continue
            if low[p2] >= low[p1]:
                continue
            r1, r2 = float(rsi[p1]), float(rsi[p2])
            if r2 <= r1 or (r2 - r1) < 2.5 or r1 > 36.0 or r2 > 36.0:
                continue
            if len(low[p1 + 1 : p2]) > 0 and np.min(low[p1 + 1 : p2]) < low[p2]:
                continue
            res = detect_elliott_impulse(
                high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                w3_bar=p1, w5_bar=p2, direction="LONG", rsi=rsi[:p2 + 1]
            )
            if res.is_valid and res.score >= min_score:
                w0_bar = res.wave_indices.get("W0", max(0, p1 - 20))
                signals.append({
                    "symbol": sym, "direction": "LONG", "w5_bar": p2, "w5_time": df.index[p2],
                    "w5_price": float(low[p2]), "w3_bar": p1, "w3_price": float(low[p1]),
                    "w0_bar": w0_bar, "w0_price": res.wave_points.get("W0", float(high[w0_bar])),
                    "algo_score": res.score, "wave_points": res.wave_points
                })
                
    return sym, signals


if __name__ == "__main__":
    folder = r"C:\Users\user\Desktop\данные для бектестов\1_Bull_Run_2020_2021"
    files_15m = [f for f in os.listdir(folder) if f.endswith("_15m.pkl")]
    tasks = [(f.split("_")[0], os.path.join(folder, f), 80.0) for f in files_15m]
    
    t0 = time.time()
    total_signals = 0
    with ProcessPoolExecutor() as executor:
        for sym, sigs in executor.map(scan_single_symbol, tasks):
            total_signals += len(sigs)
            print(f"Parallel: {sym:12} -> {len(sigs):4} impulses")
            
    t1 = time.time()
    print("-" * 50)
    print(f"Done! Found {total_signals} impulses in {t1 - t0:.2f} seconds!")
