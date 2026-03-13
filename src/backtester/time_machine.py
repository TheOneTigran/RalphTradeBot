"""
time_machine.py — Логика симуляции времени для бэктестинга.
"""
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TimeMachine:
    def __init__(self, full_history: pd.DataFrame, candle_limit: int = 500):
        """
        Args:
            full_history: Полный DataFrame истории (сортированный по ts).
            candle_limit: Сколько свечей "в прошлое" отдавать боту для анализа.
        """
        self.history = full_history
        self.candle_limit = candle_limit

    def get_snapshot(self, target_ts: int) -> Optional[List[Dict]]:
        """
        Возвращает срез свечей, которые "закрылись" до target_ts.
        """
        # Фильтруем свечи, которые закончились до target_ts
        mask = self.history['ts'] < target_ts
        snapshot_df = self.history.loc[mask].tail(self.candle_limit)
        
        if len(snapshot_df) < self.candle_limit * 0.5:
            logger.warning(f"Недостаточно данных для таймстампа {target_ts}. Найдено {len(snapshot_df)} свечей.")
            return None
            
        # Превращаем в формат, который ожидает Pipeline бота
        # [{ 'ts', 'open', 'high', 'low', 'close', 'volume' }]
        return snapshot_df.to_dict('records')

    def get_future_candles(self, start_ts: int, limit: int = 100) -> List[Dict]:
        """
        Возвращает свечи, которые идут ПОСЛЕ start_ts (для проверки исполнения).
        """
        mask = self.history['ts'] >= start_ts
        future_df = self.history.loc[mask].head(limit)
        return future_df.to_dict('records')

    def get_all_timestamps(self, step: int = 1) -> List[int]:
        """
        Возвращает список таймстампов, на которых можно запускать анализ.
        step — шаг в количестве свечей.
        """
        # Не берем первые X свечей, так как нам нужна предыстория
        indices = range(self.candle_limit, len(self.history), step)
        return [self.history.iloc[i]['ts'] for i in indices]
