"""
test_o1_loader.py — Проверка загрузки данных и построения In-Memory Hash-Map O(1).
"""
import os
import time
import pandas as pd
import numpy as np

folder = r"C:\Users\user\Desktop\данные для бектестов\1_Bull_Run_2020_2021"
t0 = time.time()

files_15m = [f for f in os.listdir(folder) if f.endswith("_15m.pkl")]
symbols = [f.split("_")[0] for f in files_15m]

print(f"Loading {len(symbols)} symbols from {folder}...")

data = {}
for sym in symbols:
    p_15m = os.path.join(folder, f"{sym}_15m.pkl")
    p_3m = os.path.join(folder, f"{sym}_3m.pkl")
    
    df_15m = pd.read_pickle(p_15m)
    if "timestamp" in df_15m.columns:
        df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
        df_15m.set_index("timestamp", inplace=True)
    
    # Hash-map для 3m свечей
    candles_3m_dict = {}
    if os.path.exists(p_3m):
        df_3m = pd.read_pickle(p_3m)
        if "timestamp" in df_3m.columns:
            df_3m["timestamp"] = pd.to_datetime(df_3m["timestamp"])
        
        bins = df_3m["timestamp"].dt.floor("15min").values
        ts = df_3m["timestamp"].values
        opens = df_3m["open"].values
        highs = df_3m["high"].values
        lows = df_3m["low"].values
        closes = df_3m["close"].values
        
        for i in range(len(df_3m)):
            k = bins[i]
            if k not in candles_3m_dict:
                candles_3m_dict[k] = []
            candles_3m_dict[k].append({
                "ts": ts[i], "open": opens[i], "high": highs[i], "low": lows[i], "close": closes[i]
            })
            
    data[sym] = {
        "df_15m": df_15m,
        "dict_3m": candles_3m_dict,
    }

t1 = time.time()
print(f"Loaded {len(data)} symbols with O(1) 3m Hash-Maps in {t1 - t0:.2f} seconds!")
for sym in symbols[:3]:
    sample_key = next(iter(data[sym]["dict_3m"].keys()))
    print(f"Sample {sym} at {sample_key}: {len(data[sym]['dict_3m'][sample_key])} 3m sub-candles")
