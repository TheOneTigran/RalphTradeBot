"""
indicators.py — Технические индикаторы для стратегии.
"""
import pandas as pd
from typing import List, Dict

def get_rsi_dict(candles: List[Dict], period: int = 14) -> Dict[int, float]:
    """
    Рассчитывает RSI (по Уайлдеру) и возвращает словарь {timestamp: rsi_value}.
    Быстрый доступ по O(1) внутри ассемблера.
    """
    if not candles:
        return {}
        
    df = pd.DataFrame(candles)
    # Определяем нужную колонку времени
    ts_col = 'ts' if 'ts' in df.columns else 'timestamp'
    
    if len(df) <= period:
        return {}
        
    delta = df['close'].diff()
    
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Первая средняя (SMA)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # RMA (Smoothed) по Уайлдеру для последующих свечей
    for i in range(period, len(df)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    df['rsi'] = rsi
    # Убираем NaN
    valid_df = df.dropna(subset=['rsi'])
    
    return dict(zip(valid_df[ts_col], valid_df['rsi']))

def get_current_atr(candles: List[Dict], period: int = 14) -> float:
    """Оставляет совместимость со старым вызовом в pipeline."""
    if len(candles) < period + 1:
        return 0.0
    df = pd.DataFrame(candles)
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return atr

def get_ema_dict(candles: List[Dict], period: int = 200) -> Dict[int, float]:
    if not candles:
        return {}
    df = pd.DataFrame(candles)
    ts_col = 'ts' if 'ts' in df.columns else 'timestamp'
    if len(df) < period:
        return {}
    
    df['ema'] = df['close'].ewm(span=period, adjust=False).mean()
    valid_df = df.dropna(subset=['ema'])
    return dict(zip(valid_df[ts_col], valid_df['ema']))
