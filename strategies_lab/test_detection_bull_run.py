"""
test_detection_bull_run.py — Проверка вызова extract_all_elliott_signals на 1_Bull_Run_2020_2021.
"""
import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies_lab.engine import detect_all_signals_in_history

folder = r"C:\Users\user\Desktop\данные для бектестов\1_Bull_Run_2020_2021"
files_15m = [f for f in os.listdir(folder) if f.endswith("_15m.pkl")]
symbols = [f.split("_")[0] for f in files_15m]

market_data = {}
for sym in symbols:
    p_15m = os.path.join(folder, f"{sym}_15m.pkl")
    p_3m = os.path.join(folder, f"{sym}_3m.pkl")
    
    df_15m = pd.read_pickle(p_15m)
    if "timestamp" in df_15m.columns:
        df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
        df_15m.set_index("timestamp", inplace=True)
    
    df_3m = None
    if os.path.exists(p_3m):
        df_3m = pd.read_pickle(p_3m)
        if "timestamp" in df_3m.columns:
            df_3m["timestamp"] = pd.to_datetime(df_3m["timestamp"])
            df_3m.set_index("timestamp", inplace=True)
            
    market_data[sym] = {
        "main": df_15m,
        "1m": df_3m,  # передаем 3m как микроструктуру
    }

signals = detect_all_signals_in_history(market_data, interval="15m", min_score=80.0)
print(f"Total Elliott Wave impulses found in 1_Bull_Run_2020_2021: {len(signals)}")
for s in signals[:5]:
    print(f"Signal: {s.symbol} {s.direction} at {s.w5_time}, price={s.w5_price:.2f}, score={s.algo_score:.1f}")
