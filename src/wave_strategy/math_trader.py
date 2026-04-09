"""
math_trader.py — Генератор ордеров.

Превращает найденный `AlignedSetup` в рабочий торговый план, вычисляя цены
по строго математическим правилам, опираясь на ATR.
"""

from typing import Optional

from .models import AlignedSetup, DtwTradePlan, Direction

def calculate_trade_plan(
    symbol: str,
    setup: AlignedSetup,
    atr_micro: float,
    entry_atr_mult: float = 0.2, # Близко к C_Close
    tolerance_atr_mult: float = 1.5, # Динамическая рыночная зона входа повышенной лояльности
    sl_fib_mult: float = 1.272, # Фрактальный стоп-лосс (расширение)
    sl_atr_mult: float = 1.5, # Максимальный порог растяжения стопа
    tp_fib_mult: float = 1.0
) -> Optional[DtwTradePlan]:
    """
    Рассчитывает точные параметры сделки по опорным точкам DTW паттернов.
    Теперь реализует логику Снайперского Входа (подтвержденный отскок от дна коррекции C).
    """
    macro = setup.macro_pattern
    micro = setup.micro_pattern
    
    macro_w1_len = abs(macro.pivots.get(1, 0.0) - macro.pivots.get(0, 0.0))
    macro_origin = macro.pivots.get(0, 0.0)
    
    micro_b_price = micro.pivots.get(2, 0.0)
    micro_c_price = micro.pivots.get(3, 0.0)
    
    bc_vector = abs(micro_b_price - micro_c_price)
    
    # Расчёты для Бычьего сетапа (лонг на откате)
    if macro.direction == Direction.BULLISH:
        trade_dir = "LONG"
        
        # Снайперский Вход
        entry = micro_c_price + (entry_atr_mult * atr_micro)
        entry_tol = entry + (tolerance_atr_mult * atr_micro)
        
        # Цель откладываем строго от подтвержденного дна C
        tp = micro_c_price + (tp_fib_mult * macro_w1_len)
        
        # Стопы (Ограничитель растяжения: выбираем самый узкий стоп между Фрактальным и ATR)
        fractal_sl = micro_c_price - (bc_vector * sl_fib_mult)
        atr_sl = micro_c_price - (sl_atr_mult * atr_micro)
        sl_agg = max(fractal_sl, atr_sl)
        
        sl_cons = macro_origin
        
        # Финальная проверка логичности
        if not (sl_cons < sl_agg < entry < tp):
            return None
            
    # Расчёты для Медвежьего сетапа (шорт на отскоке)
    else:
        trade_dir = "SHORT"
        
        # Снайперский Вход 
        entry = micro_c_price - (entry_atr_mult * atr_micro)
        entry_tol = entry - (tolerance_atr_mult * atr_micro)
        
        # Цель откладываем от точки C
        tp = micro_c_price - (tp_fib_mult * macro_w1_len)
        
        # Стопы (Ограничитель растяжения)
        fractal_sl = micro_c_price + (bc_vector * sl_fib_mult)
        atr_sl = micro_c_price + (sl_atr_mult * atr_micro)
        sl_agg = min(fractal_sl, atr_sl)
        
        sl_cons = macro_origin
        
        # Финальная проверка логичности
        if not (tp < entry < sl_agg < sl_cons):
            return None
            
    # Расчет Risk-Reward
    risk = abs(entry - sl_agg)
    reward = abs(tp - entry)
    
    if risk == 0:
        return None
        
    rr_ratio = round(reward / risk, 2)
    
    # Расчет динамического TTL (Время жизни)
    mult = 60 if macro.timeframe == "1h" else 240 if macro.timeframe == "4h" else 15 if macro.timeframe == "15m" else 1
    ttl_minutes = int(macro.window_size * mult * 0.5)
    
    return DtwTradePlan(
        symbol=symbol,
        direction=trade_dir,
        point_c_price=round(micro_c_price, 4),
        entry_price=round(entry, 4),
        entry_tolerance=round(entry_tol, 4),
        sl_aggressive=round(sl_agg, 4),
        sl_conservative=round(sl_cons, 4),
        take_profit=round(tp, 4),
        creation_time=micro.end_ts,
        ttl_minutes=ttl_minutes,
        risk_reward_ratio=rr_ratio,
        macro_context=f"{macro.pattern_type} on {macro.timeframe} (win {macro.window_size}, score {macro.dtw_score:.3f})",
        micro_trigger=f"{micro.pattern_type} on {micro.timeframe} (win {micro.window_size})"
    )
