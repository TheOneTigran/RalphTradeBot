"""
test_dataset_structure.py — Проверка формата данных во всех 9 папках стресс-тестов.
"""
import os
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
base_dir = r"C:\Users\user\Desktop\данные для бектестов"

folders = [
    "1_Bull_Run_2020_2021",
    "2_Crypto_Winter_2022",
    "3_Boring_Flat_2023",
    "4_ETF_Rally_2023_2024",
    "5_Corona_Crash_2020",
    "6_Summer_Accumulation_2021",
    "7_Post_Halving_Crash_2024",
    "8_Mega_Bull_Run_2025",
    "9_Macro_Bear_Market_2026",
]

print(f"{'Folder':30} | {'Pairs':6} | {'TFs':12} | {'Rows (15m)':10} | {'Date Range'}")
print("-" * 90)

for fld in folders:
    fld_path = os.path.join(base_dir, fld)
    if not os.path.exists(fld_path):
        print(f"Missing: {fld}")
        continue
    
    files = [f for f in os.listdir(fld_path) if f.endswith(".pkl")]
    symbols = set()
    tfs = set()
    sample_15m = None
    
    for f in files:
        parts = f.replace(".pkl", "").split("_")
        if len(parts) >= 2:
            sym = parts[0]
            tf = parts[1]
            symbols.add(sym)
            tfs.add(tf)
            if tf == "15m" and sample_15m is None:
                sample_15m = os.path.join(fld_path, f)
                
    date_str = "N/A"
    rows_str = "N/A"
    if sample_15m:
        df = pd.read_pickle(sample_15m)
        rows_str = str(len(df))
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            date_str = f"{ts.iloc[0].strftime('%Y-%m-%d')} .. {ts.iloc[-1].strftime('%Y-%m-%d')}"
        elif isinstance(df.index, pd.DatetimeIndex):
            date_str = f"{df.index[0].strftime('%Y-%m-%d')} .. {df.index[-1].strftime('%Y-%m-%d')}"
            
    print(f"{fld:30} | {len(symbols):6} | {str(tfs):12} | {rows_str:10} | {date_str}")
