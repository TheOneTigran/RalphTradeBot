"""
download_real_data.py — Загрузчик исторических данных (ccxt).
Выкачивает синхронизированные свечи 1H и 1m для реального бэктеста, не нарушая физику гранулярности.
"""
import os
import ccxt
import time
import pandas as pd

def fetch_paginated(exchange, symbol, timeframe, since=None, total_limit=10000):
    all_candles = []
    print(f"[{timeframe}] Начинаю скачивание {total_limit} свечей для {symbol}...")
    
    while len(all_candles) < total_limit:
        try:
            limit = min(1000, total_limit - len(all_candles)) # Bybit max limit is 1000
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            
            if not ohlcv:
                break
                
            all_candles.extend(ohlcv)
            # Следующий запрос начнем со времени последней свечи + 1мс
            since = ohlcv[-1][0] + 1
            
            # Rate limit protection
            time.sleep(exchange.rateLimit / 1000.0)
            print(f"  Скачано {len(all_candles)}/{total_limit}...")
            
        except Exception as e:
            print(f"Ошибка при скачивании: {e}")
            break
            
    # Конвертируем в DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    return df

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    target_dir = os.path.join(root_dir, "src", "execution_engine", "tests", "data")
    os.makedirs(target_dir, exist_ok=True)
    
    exchange = ccxt.bybit({'enableRateLimit': True})
    symbol = 'BTC/USDT'
    
    # 1 месяц истории:
    # 1 месяц = 30 дней * 24 часа = 720 свечей по 1H
    # 1 месяц = 30 * 24 * 60 = 43200 свечей по 1m
    
    # Считаем timestamp (30 дней назад от текущего момента)
    now = exchange.milliseconds()
    thirty_days_ms = 30 * 24 * 60 * 60 * 1000
    since_ms = now - thirty_days_ms
    
    print("\n--- Скачивание 1H (Макро-Контекст) ---")
    df_1h = fetch_paginated(exchange, symbol, '1h', since=since_ms, total_limit=720)
    path_1h = os.path.join(target_dir, "btc_1h.csv")
    df_1h.to_csv(path_1h, index=False)
    print(f"Сохранено: {path_1h}")
    
    print("\n--- Скачивание 15m (Микро-Контекст) ---")
    df_15m = fetch_paginated(exchange, symbol, '15m', since=since_ms, total_limit=2880)
    path_15m = os.path.join(target_dir, "btc_15m.csv")
    df_15m.to_csv(path_15m, index=False)
    print(f"Сохранено: {path_15m}")
    
    print("\n--- Скачивание 1m (Физика Бэктестера) ---")
    df_1m = fetch_paginated(exchange, symbol, '1m', since=since_ms, total_limit=43200)
    path_1m = os.path.join(target_dir, "btc_1m.csv")
    df_1m.to_csv(path_1m, index=False)
    print(f"Сохранено: {path_1m}")
    
    print("\nГотово! Теперь обновите пути в run_backtest_demo.py к этим файлам.")

if __name__ == "__main__":
    main()
