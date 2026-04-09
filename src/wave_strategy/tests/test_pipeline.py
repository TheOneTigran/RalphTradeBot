import pandas as pd
from src.wave_strategy.pipeline import run_wave_strategy

def main():
    # Load the 1000 BTC candles we already downloaded
    df = pd.read_csv("src/dtw_wave_labs/tests/fixtures/btc_1000.csv")
    
    # We will pretend the same 1000 candles are 4h, 1h, and 15m just to test the integration.
    # In a real environment, they would be different data frames.
    candles = df.to_dict('records')
    
    candles_by_tf = {
        "4h": candles,
        "1h": candles,
        "15m": candles
    }
    
    print("Testing Wave Strategy E2E Flow...")
    trades = run_wave_strategy(
        symbol="BTC/USDT",
        candles_by_tf=candles_by_tf,
        macro_tfs=["4h", "1h"],
        micro_tfs=["15m"],
        min_rr=1.5,
        verbose=True
    )
    
    print(f"\nFound {len(trades)} Trade Plans:")
    for t in trades:
        print(t)

if __name__ == "__main__":
    main()
