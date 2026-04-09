"""
download_fixture.py — One-time script to fetch 1000 BTCUSDT 1H candles
from Bybit via ccxt and save them as tests/fixtures/btc_1000.csv.

Run once from project root:
    .venv\Scripts\python.exe src/dtw_wave_labs/tests/download_fixture.py
"""

import os
import sys

# Ensure project root is on sys.path when run directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import ccxt
import pandas as pd

FIXTURE_DIR = os.path.join(_HERE, "fixtures")
FIXTURE_FILE = os.path.join(FIXTURE_DIR, "btc_1000.csv")
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 1000


def download():
    os.makedirs(FIXTURE_DIR, exist_ok=True)

    print(f"Connecting to Bybit …")
    exchange = ccxt.bybit({"enableRateLimit": True})

    print(f"Fetching {LIMIT} × {TIMEFRAME} candles for {SYMBOL} …")
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["datetime", "timestamp", "open", "high", "low", "close", "volume"]]
    df.to_csv(FIXTURE_FILE, index=False)

    print(f"Saved {len(df)} rows → {FIXTURE_FILE}")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    download()
