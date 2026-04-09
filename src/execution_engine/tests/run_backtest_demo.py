import os
import pandas as pd
from src.wave_strategy.pipeline import run_wave_strategy
from src.execution_engine.backtester import run_backtest

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(root_dir, "src", "execution_engine", "tests", "data")
    
    path_1h = os.path.join(data_dir, "btc_1h.csv")
    path_15m = os.path.join(data_dir, "btc_15m.csv")
    path_1m = os.path.join(data_dir, "btc_1m.csv")
    
    if not (os.path.exists(path_1h) and os.path.exists(path_15m) and os.path.exists(path_1m)):
        print(f"Файлы реальных данных не найдены в {data_dir}. Запускаем скачивание...")
        from .download_real_data import main as download_main
        download_main()
        
    df_1h = pd.read_csv(path_1h)
    df_15m = pd.read_csv(path_15m)
    df_1m = pd.read_csv(path_1m)
    
    candles_by_tf = {
        "1h": df_1h.to_dict('records'),
        "15m": df_15m.to_dict('records')
    }
    
    print("1. Запуск Сигнального Сканера (DTW) - O(1) Run")
    trade_plans = run_wave_strategy(
        symbol="BTC/USDT",
        candles_by_tf=candles_by_tf,
        macro_tfs=["1h"],     
        micro_tfs=["15m"],    
        verbose=True
    )
    
    print(f"Найдено {len(trade_plans)} предварительных планов.")
    
    print("\n2. Запуск Событийно-Ориентированного Бектестера на настоящих 1-минутных физических тиках")
    csv_out = os.path.join(root_dir, "backtest_results.csv")
    stats = run_backtest(
        trade_plans=trade_plans,
        candles_1m=df_1m.to_dict('records'),
        csv_output_path=csv_out
    )
    
    print(f"\nCSV сохранен в {csv_out}")

if __name__ == "__main__":
    main()
