"""
fractal_assembler.py — Фильтрация и слияние паттернов.

Применяет строгую логику ВА (Волн Эллиотта) для отсева ложных паттернов
и поиска связок Макро-Тренд (1H, 4H) + Микро-Откат (15m).
"""

import logging
from typing import Dict, List

from .models import DtwPattern, AlignedSetup, Direction

logger = logging.getLogger(__name__)


def assemble_setups(
    all_patterns: Dict[str, List[DtwPattern]],
    macro_tfs: List[str] = ["4h", "1h"],
    micro_tfs: List[str] = ["15m", "5m"],
    rsi_cache: Dict[str, Dict[int, float]] = None,
    ema_cache: Dict[str, Dict[int, float]] = None
) -> List[AlignedSetup]:
    """
    Ищет математические пересечения (Alignment) между макро- и микро-таймфреймами.
    
    Условия:
    1. Макро-контекст: IMPULSE или DIAGONAL с высоким качеством (score < 0.08).
    2. Микро-триггер: Коррекция (ZIGZAG, FLAT, TRIANGLE) против макро-направления.
    3. Хронология: Микро-паттерн начался ПОСЛЕ пика Волны 3 макро-паттерна.
    4. Инвалидация: Микро-паттерн не пробил уровень вершины Волны 1 макро-импульса.
    """
    setups: List[AlignedSetup] = []
    
    # 1. Сбор макро-контекстов
    macro_patterns = []
    for tf in macro_tfs:
        for p in all_patterns.get(tf, []):
            if p.is_impulsive and p.dtw_score <= 0.08:
                macro_patterns.append(p)
                
    # 2. Сбор микро-триггеров
    micro_patterns = []
    for tf in micro_tfs:
        for p in all_patterns.get(tf, []):
            if p.is_corrective:
                micro_patterns.append(p)
                
    logger.info(f"[fractal_assembler] Макро-импульсов найдено: {len(macro_patterns)}, микро-коррекций: {len(micro_patterns)}")
    
    # 3. Alignment
    for macro in macro_patterns:
        # Для импульса [0=start, 1=W1, 2=W2, 3=W3, 4=W4, 5=W5]
        # Вершина Волны 3 - это индекс 3.
        # Вершина Волны 1 - это индекс 1.
        macro_w3_ts = macro.pivots_ts.get(3, 0)
        macro_w1_price = macro.pivots.get(1, 0.0)
        
        for micro in micro_patterns:
            # 3.1 Контр-тренд: направление микро должно быть противоположно макро
            if macro.direction == micro.direction:
                continue
                
            # 3.2 Time Bounds Alignment: Микро начался после пика макро Волны 3
            micro_start_ts = micro.pivots_ts.get(0, 0)
            if micro_start_ts < macro_w3_ts:
                continue
                
            macro_w2_price = macro.pivots.get(0, 0.0)
            macro_w3_price = macro.pivots.get(1, 0.0)
            micro_c_price = micro.pivots.get(3, 0.0)
            
            # 3.2.1 Fibonacci Retracement Filter (Глубина Коррекции Волны 4)
            w3_len = abs(macro_w3_price - macro_w2_price)
            if w3_len > 0:
                retracement = abs(macro_w3_price - micro_c_price) / w3_len
                if retracement < 0.236 or retracement > 0.618:
                    continue  # Игнорируем рыночный шум на хаях или чрезмерно глубокие краши
                
            # 3.3 Уровень инвалидации: коррекция Волны 4 не должна пробивать пик Волны 1
            
            invalidated = False
            if macro.direction == Direction.BULLISH:
                # В бычьем рынке коррекция не должна упасть ниже W1 Peak
                if micro_c_price <= macro_w1_price:
                    invalidated = True
            else:
                # В медвежьем рынке коррекция не должна подняться выше W1 Bottom
                if micro_c_price >= macro_w1_price:
                    invalidated = True
                    
            if invalidated:
                continue
                
            # 3.4 Индикаторная Валидация (RSI Divergence: Истинное Дно)
            if rsi_cache and micro.timeframe in rsi_cache:
                rsi_dict = rsi_cache[micro.timeframe]
                micro_a_ts = micro.pivots_ts.get(1, 0) # Вершина/Дно Волны А
                micro_c_ts = micro.pivots_ts.get(3, 0) # Вершина/Дно Волны С
                micro_a_price = micro.pivots.get(1, 0.0)
                
                rsi_a = rsi_dict.get(micro_a_ts)
                rsi_c = rsi_dict.get(micro_c_ts)
                
                if rsi_a is not None and rsi_c is not None:
                    if macro.direction == Direction.BULLISH:
                        # Дивергенция ИЛИ жесткая перепроданность
                        has_divergence = (micro_c_price < micro_a_price) and (rsi_c > rsi_a)
                        if not (has_divergence or rsi_c < 30):
                            continue # Продолжение тренда, дно не подтверждено
                    else:
                        has_divergence = (micro_c_price > micro_a_price) and (rsi_c < rsi_a)
                        if not (has_divergence or rsi_c > 70):
                            continue
                            
            # 3.5 HTF Trend Filter (EMA 200 Macro Защита от падающих ножей)
            if ema_cache and macro.timeframe in ema_cache:
                ema_dict = ema_cache[macro.timeframe]
                ema_val = None
                # Ищем последнюю известную EMA на закрытии свечи микро-паттерна
                for ts_macro in sorted(ema_dict.keys(), reverse=True):
                    if ts_macro <= micro.end_ts:
                        ema_val = ema_dict[ts_macro]
                        break
                        
                if ema_val is not None:
                    if macro.direction == Direction.BULLISH:
                        # Только лонги над EMA 200 (Восходящий макро-тренд)
                        if micro_c_price < ema_val:
                            continue
                    else:
                        # Только шорты под EMA 200
                        if micro_c_price > ema_val:
                            continue
                
            # Бинго! У нас есть сетап
            setups.append(
                AlignedSetup(
                    macro_pattern=macro,
                    micro_pattern=micro,
                    status="ALIGNED"
                )
            )
            
    logger.info(f"[fractal_assembler] Успешных слияний (AlignedSetups): {len(setups)}")
    return setups
