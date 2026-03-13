"""
data_manager.py — Модуль для работы с историческими данными (загрузка и кэширование).
"""
import os
import pandas as pd
import logging
import time
from datetime import datetime, timezone
import ccxt
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

HISTORY_DIR = "data/history"

def get_history_path(symbol: str, timeframe: str) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    clean_symbol = symbol.replace("/", "").replace(":", "")
    return os.path.join(HISTORY_DIR, f"{clean_symbol}_{timeframe}.csv")

def save_history(df: pd.DataFrame, symbol: str, timeframe: str):
    path = get_history_path(symbol, timeframe)
    df.to_csv(path, index=False)
    logger.info(f"Сохранено {len(df)} свечей в {path}")

def load_history(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    path = get_history_path(symbol, timeframe)
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Убеждаемся, что ts — это int
        df['ts'] = df['ts'].astype(int)
        return df
    return None

def fetch_historical_ohlcv(
    symbol: str, 
    timeframe: str, 
    since_ms: int, 
    until_ms: Optional[int] = None
) -> pd.DataFrame:
    """
    Выкачивает историю свечей с использованием пагинации.
    """
    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "linear"}})
    all_ohlcv = []
    current_since = since_ms
    limit = 1000 
    
    target_until = until_ms or int(time.time() * 1000)
    
    logger.info(f"Начинаю загрузку истории {symbol} {timeframe} с {datetime.fromtimestamp(since_ms/1000, tz=timezone.utc)}")

    while current_since < target_until:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            if not ohlcv:
                # Если данных больше нет совсем
                break
            
            all_ohlcv.extend(ohlcv)
            
            last_ts = ohlcv[-1][0]
            if last_ts <= current_since:
                break
            current_since = last_ts + 1
            
            # Если получили меньше лимита, это может быть конец данных
            if len(ohlcv) < limit:
                # Но проверяем, не слишком ли мы далеко от цели
                if current_since >= target_until - (limit * 1000): # Примерный запас
                    break
            
            time.sleep(exchange.rateLimit / 1000)
            logger.info(f"  [{timeframe}] Загружено {len(all_ohlcv)} свечей... ({datetime.fromtimestamp(last_ts/1000, tz=timezone.utc)})")
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df.drop_duplicates(subset=['ts'], inplace=True)
    df.sort_values('ts', inplace=True)
    return df

def get_or_fetch_history(symbol: str, timeframe: str, since_ms: int, until_ms: Optional[int] = None) -> pd.DataFrame:
    """
    Интеллектуальная загрузка: проверяет кэш и докачивает если нужно.
    """
    df = load_history(symbol, timeframe)
    target_until = until_ms or int(time.time() * 1000)
    
    if df is not None and not df.empty:
        first_ts = df['ts'].iloc[0]
        last_ts = df['ts'].iloc[-1]
        
        # Если кэш полностью покрывает период
        if first_ts <= since_ms and last_ts >= target_until - (1800000): # 30 мин запас
            logger.info(f"Использую локальный кэш {symbol} {timeframe}")
            return df
        
        logger.info(f"Кэш {symbol} {timeframe} неполон (нужно {since_ms}-{target_until}, есть {first_ts}-{last_ts}). Обновляю...")

    # Выкачиваем за весь период
    df = fetch_historical_ohlcv(symbol, timeframe, since_ms, target_until)
    save_history(df, symbol, timeframe)
    return df

def fetch_all_needed_history(symbol: str, timeframes: List[str], since_ms: int, until_ms: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Выкачивает историю для всех ТФ.
    """
    results = {}
    for tf in timeframes:
        results[tf] = get_or_fetch_history(symbol, tf, since_ms, until_ms)
    return results
