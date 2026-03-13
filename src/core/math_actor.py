"""
math_actor.py — Математический "Актёр", заменяющий LLM для поиска сделок по структурам ВА.
"""
import logging
from typing import List, Optional
from src.core.models import TradePlan, LLMContext, WaveCoordinate
from src.core.config import MIN_RR_RATIO
from src.math_engine.wave_analyzer import (
    _vectors_to_extrema, _check_impulse, _check_diagonal, _check_triangle,
    _check_zigzag, _check_flat, _check_wxy, _check_forming_123,
    WaveStructure, WavePoint
)

logger = logging.getLogger(__name__)

def get_math_trade_plan(context: LLMContext) -> TradePlan:
    """
    Анализирует все ТФ, находит паттерны и возвращает лучший торговый план.
    """
    all_valid_structures: List[tuple[str, WaveStructure]] = [] # (tf, structure)
    
    for tf_data in context.timeframes:
        tf = tf_data.timeframe
        vectors = tf_data.vectors
        if not vectors or len(vectors) < 3:
            continue
            
        extrema = _vectors_to_extrema(vectors)
        tf_structures: List[WaveStructure] = []
        
        # 1. Поиск паттернов на конкретном ТФ
        recent_6 = extrema[-15:] if len(extrema) > 15 else extrema
        recent_4 = extrema[-10:] if len(extrema) > 10 else extrema
        
        # --- 6-точечные паттерны ---
        if len(recent_6) >= 6:
            for i in range(len(recent_6) - 5):
                pts = recent_6[i:i+6]
                for func in [_check_impulse, _check_diagonal, _check_triangle]:
                    res = func(pts, tf) if func != _check_impulse else func(pts, tf, vectors)
                    if res: tf_structures.append(res)
        
        # --- 4-точечные паттерны ---
        if len(recent_4) >= 4:
            for i in range(len(recent_4) - 3):
                pts = recent_4[i:i+4]
                for func in [_check_zigzag, _check_flat, _check_wxy]:
                    res = func(pts, tf)
                    if res: tf_structures.append(res)
        
        # --- Формирующиеся 1-2-3 ---
        if len(extrema) >= 4:
            res = _check_forming_123(extrema[-4:], tf)
            if res: tf_structures.append(res)

        # Фильтруем только "свежие" паттерны
        recent_ts = {round(float(e.timestamp)) for e in extrema[-3:]}
        for s in tf_structures:
            if round(float(s.points[-1].timestamp)) in recent_ts:
                # Добавляем базовый бонус сложности
                bonus = 0
                if "Импульс" in s.pattern_type: bonus = 40
                elif "Диагональ" in s.pattern_type: bonus = 30
                elif "Треугольник" in s.pattern_type: bonus = 25
                elif "Зигзаг" in s.pattern_type: bonus = 10
                s.confidence += bonus
                all_valid_structures.append((tf, s))

    if not all_valid_structures:
        return TradePlan(
            wave_count_label="No Patterns",
            detailed_logic="Математический анализ не выявил паттернов со входом.",
            main_scenario="WAIT",
            trade_params={"direction": "WAIT"}
        )

    # 2. Кросс-ТФ подтверждение (Weighted Scoring)
    # Если на разных ТФ одинаковое направление — даем бонус
    direction_votes = {"LONG": 0.0, "SHORT": 0.0}
    for tf, s in all_valid_structures:
        dir_key = "LONG" if s.direction == "БЫЧИЙ" else "SHORT"
        weight = {"1d": 3.0, "4h": 2.0, "1h": 1.5, "15m": 1.0, "5m": 0.5}.get(tf, 1.0)
        direction_votes[dir_key] += (s.confidence / 100.0) * weight

    # Сортируем все найденные структуры по итоговому весу
    def score_structure(item):
        tf, s = item
        dir_key = "LONG" if s.direction == "БЫЧИЙ" else "SHORT"
        # Итоговый балл = уверенность + бонус за общее направление рынка
        return s.confidence + (direction_votes[dir_key] * 10)

    all_valid_structures.sort(key=score_structure, reverse=True)

    # 3. Выбор Main и Alternative
    best_tf, best_s = all_valid_structures[0]
    main_plan = _convert_structure_to_plan(best_s, context, best_tf)
    
    alt_desc = "Нет альтернатив"
    if len(all_valid_structures) > 1:
        alt_tf, alt_s = all_valid_structures[1]
        alt_desc = f"Альтернатива: {alt_s.pattern_type} {alt_s.direction} на {alt_tf} ({alt_s.confidence:.0f}%)"

    if not main_plan:
        # Если лучший не прошел R:R, пробуем второй и т.д.
        for i in range(1, len(all_valid_structures)):
            best_tf, best_s = all_valid_structures[i]
            main_plan = _convert_structure_to_plan(best_s, context, best_tf)
            if main_plan: break
            
    if not main_plan:
        return TradePlan(wave_count_label="RR Filtered", main_scenario="WAIT", trade_params={"direction": "WAIT"})

    # Добавляем информацию об альтернативах в логику
    all_seen = [f"{s.pattern_type}({tf})" for tf, s in all_valid_structures[:4]]
    main_plan.detailed_logic = (
        f"ГЛАВНЫЙ: {best_s.pattern_type} ({best_tf}). "
        f"Рассмотрено параллельно: {', '.join(all_seen)}. "
        f"{alt_desc}. {main_plan.detailed_logic}"
    )
    main_plan.alternative_scenario = alt_desc
    
    return main_plan

def _convert_structure_to_plan(s: WaveStructure, context: LLMContext, tf: str) -> Optional[TradePlan]:
    """Превращает математическую структуру в торговый план."""
    
    direction = "LONG" if s.direction == "БЫЧИЙ" else "SHORT"
    entry = s.points[-1].price # Вход по рынку в конце паттерна (упрощенно)
    sl = s.invalidation_price
    
    if not sl or sl == entry:
        return None
        
    # Цель: либо канал, либо Фибо (например 1.618 от предыдущей волны)
    tp = s.channel_target
    if not tp:
        # Используем MIN_RR_RATIO из конфига + небольшой запас (0.1), чтобы точно пройти валидацию
        mult = max(2.0, MIN_RR_RATIO + 0.1)
        dist = abs(entry - sl)
        tp = entry + dist * mult if direction == "LONG" else entry - dist * mult

    # Проверка R:R
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0 or (reward / risk) < MIN_RR_RATIO:
        return None
        
    # Формируем WaveCoordinate для плана
    waves = []
    for i in range(len(s.points) - 1):
        p_start = s.points[i]
        p_end = s.points[i+1]
        waves.append(WaveCoordinate(
            timeframe=tf,
            wave_name=p_end.label,
            start_price=p_start.price,
            end_price=p_end.price,
            start_time=p_start.timestamp,
            end_time=p_end.timestamp
        ))

    # Формируем человекочитаемый список точек для TradingView
    from datetime import datetime
    points_desc = []
    for p in s.points:
        dt_str = datetime.fromtimestamp(p.timestamp / 1000).strftime('%Y-%m-%d %H:%M')
        points_desc.append(f"{p.label}: {p.price:.2f} ({dt_str})")
    
    tv_coords = " | ".join(points_desc)

    return TradePlan(
        wave_count_label=f"{s.pattern_type} {s.direction}",
        detailed_logic=f"Математический сигнал: {s.pattern_type}. Детали: {s.details}. Координаты ТВ: {tv_coords}",
        main_scenario=f"{direction} от {entry} с целью {tp}",
        alternative_scenario=f"Отмена при пробое {sl}",
        main_scenario_probability=int(s.confidence),
        alternative_scenario_probability=100 - int(s.confidence),
        trigger_prices={
            "confirmation_level": f"{entry}",
            "invalid_level": f"{sl}"
        },
        trade_params={
            "direction": direction,
            "stop_loss": f"{sl}",
            "take_profit_levels": [f"{tp}"],
        },
        waves_breakdown=waves
    )
