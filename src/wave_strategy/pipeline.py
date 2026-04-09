"""
pipeline.py — Оркестратор математической торговой логики (End-to-End Flow).

Этот модуль объединяет dtw_scanner, fractal_assembler и math_trader,
образуя законченный конвейер, не зависящий от LLM.
"""

import logging
import math
from typing import Dict, List, Optional

from src.math_engine.indicators import atr
from .models import DtwTradePlan
from .dtw_scanner import scan_all_timeframes
from .fractal_assembler import assemble_setups
from .math_trader import calculate_trade_plan
from .indicators import get_rsi_dict, get_ema_dict

logger = logging.getLogger(__name__)


def get_current_atr(candles: List[Dict], period: int = 14) -> float:
    """Вычисляет актуальное значение ATR для массива свечей."""
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    
    atr_series = atr(highs, lows, closes, period=period)
    
    # Ищем последнее не-NaN значение
    for val in reversed(atr_series):
        if not math.isnan(val):
            return val
    return 0.0


def run_wave_strategy(
    symbol: str,
    candles_by_tf: Dict[str, List[Dict]],
    macro_tfs: List[str] = ["4h", "1h"],
    micro_tfs: List[str] = ["15m", "5m"],
    min_rr: float = 1.2,
    verbose: bool = False
) -> List[DtwTradePlan]:
    """
    Основной цикл программы (поиск торговых планов).
    """
    logger.info(f"=== Запуск Wave Strategy Pipeline ({symbol}) ===")
    
    # 1. Сканирование DTW
    all_patterns = scan_all_timeframes(candles_by_tf, verbose=verbose)
    
    # Заранее вычислим ATR и RSI
    atr_cache = {}
    rsi_cache = {}
    for tf in micro_tfs:
        if tf in candles_by_tf and candles_by_tf[tf]:
            atr_cache[tf] = get_current_atr(candles_by_tf[tf])
            rsi_cache[tf] = get_rsi_dict(candles_by_tf[tf])
        else:
            atr_cache[tf] = 0.0
            rsi_cache[tf] = {}
            
    # Вычислим EMA для макро-таймфреймов (HTF Trend Filter)
    ema_cache = {}
    for tf in macro_tfs:
        if tf in candles_by_tf and candles_by_tf[tf]:
            ema_cache[tf] = get_ema_dict(candles_by_tf[tf], period=200)
            
    # 2. Фрактальная сборка
    setups = assemble_setups(all_patterns, macro_tfs, micro_tfs, rsi_cache=rsi_cache, ema_cache=ema_cache)
    
    if not setups:
        logger.info("[pipeline] Нет согласованных фрактальных сетапов.")
        return []
        
    trades: List[DtwTradePlan] = []
            
    # 3. Расчёт ордеров
    for setup in setups:
        micro_tf = setup.micro_pattern.timeframe
        atr_micro = atr_cache.get(micro_tf, 0.0)
        
        if atr_micro == 0:
            logger.warning(f"  [pipeline] Не удалось вычислить ATR для {micro_tf}, пропускаем.")
            continue
            
        trade = calculate_trade_plan(
            symbol, setup, atr_micro
        )
        
        if trade:
            if trade.risk_reward_ratio >= min_rr:
                trades.append(trade)
                logger.info(f"  [+] План Одобрен: {trade.direction} Entry {trade.entry_price} (RR {trade.risk_reward_ratio})")
            else:
                logger.info(f"  [-] План Отклонён (RR {trade.risk_reward_ratio} < {min_rr}): {trade.direction}")
                
    logger.info(f"=== Завершено. Найдено торговых планов: {len(trades)} ===")
    return trades
