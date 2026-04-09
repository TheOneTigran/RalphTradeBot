"""
dtw_scanner.py — Рамка-адаптер над модулем dtw_wave_labs.

Принимает сырые свечи из бота, "скармливает" их dtw_wave_labs,
конвертирует результаты в изолированные объекты `DtwPattern`.
"""

import time
import logging
from typing import Dict, List

from src.dtw_wave_labs.pipeline import run_pipeline
from .models import DtwPattern

logger = logging.getLogger(__name__)


def scan_all_timeframes(
    candles_by_tf: Dict[str, List[Dict]],
    verbose: bool = False
) -> Dict[str, List[DtwPattern]]:
    """
    Прогоняет DTW-пайплайн по всем доступным таймфреймам.
    
    Args:
        candles_by_tf: Словарь {timeframe: [свечи]}, свеча = {'ts': int, 'high': float, ...}
        verbose: Логировать прогресс DTW пайплайна
        
    Returns:
        Словарь {timeframe: [DtwPattern, DtwPattern, ...]}
    """
    results: Dict[str, List[DtwPattern]] = {}
    
    t0 = time.perf_counter()
    total_found = 0
    
    for tf, candles in candles_by_tf.items():
        if len(candles) < 233:  # Минимальное окно 233
            logger.warning(f"  [dtw_scanner] Слишком мало свечей для ТФ {tf}: {len(candles)}")
            results[tf] = []
            continue
            
        timestamps = [int(c.get("ts", c.get("timestamp", 0))) for c in candles]
        
        # dtw_wave_labs требует массивов high, low, close
        import numpy as np
        high_arr  = np.array([c["high"] for c in candles], dtype=np.float64)
        low_arr   = np.array([c["low"] for c in candles], dtype=np.float64)
        close_arr = np.array([c["close"] for c in candles], dtype=np.float64)
        
        if verbose:
            print(f"  [dtw_scanner] Сканирование {tf} ({len(candles)} свечей)...")
            
        # Запуск изолированного DTW пайплайна
        dtw_matches, elapsed = run_pipeline(high_arr, low_arr, close_arr, verbose=verbose)
        
        # Конвертация PatternResult -> DtwPattern
        tf_patterns = []
        for match in dtw_matches:
            if not match.passed_validation:
                continue
                
            # Восстанавливаем время экстремумов
            pivots_ts = {}
            for wave_idx, price in match.pivots.items():
                # Найдём ближайшую свечу с такой ценой в диапазоне паттерна
                start = match.start_index
                end = match.end_index
                
                best_idx = start
                min_diff = float("inf")
                
                # Ищем где именно был этот Extreme Price (High или Low)
                is_high = True # Approximate, we just want timestamp.
                # Actually, pivots are exact values from raw_high or raw_low.
                for i in range(start, end + 1):
                    dh = abs(high_arr[i] - price)
                    dl = abs(low_arr[i] - price)
                    if dh < min_diff:
                        min_diff = dh
                        best_idx = i
                    if dl < min_diff:
                        min_diff = dl
                        best_idx = i
                
                pivots_ts[wave_idx] = timestamps[best_idx]
            
            p = DtwPattern(
                pattern_type=match.pattern,
                timeframe=tf,
                window_size=match.end_index - match.start_index + 1,
                start_ts=timestamps[match.start_index],
                end_ts=timestamps[match.end_index],
                dtw_score=match.dtw_score,
                pivots=match.pivots,
                pivots_ts=pivots_ts
            )
            tf_patterns.append(p)
            
        results[tf] = tf_patterns
        total_found += len(tf_patterns)
        
    t1 = time.perf_counter()
    logger.info(f"[dtw_scanner] Готово. Найдено {total_found} паттернов за {t1 - t0:.2f}с")
    
    return results
