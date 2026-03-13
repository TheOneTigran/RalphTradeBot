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
        
        # Используем экстремумы напрямую (уже профильтрованы в preprocessor)
        final_extrema = extrema
        recent_6 = final_extrema[-15:] if len(final_extrema) > 15 else final_extrema
        recent_4 = final_extrema[-20:] if len(final_extrema) > 10 else final_extrema
        
        tf_structures = []
        tf_atr = tf_data.current_atr or 0
        
        # Находим sub_tf_vectors для фрактальной проверки
        sub_tf_map = {"1d": "4h", "4h": "1h", "1h": "15m", "15m": "5m", "5m": None}
        sub_tf_name = sub_tf_map.get(tf)
        sub_tf_vectors = next((t.vectors for t in context.timeframes if t.timeframe == sub_tf_name), None) if sub_tf_name else None

        # --- 6-точечные паттерны ---
        if len(recent_6) >= 6:
            for i in range(len(recent_6) - 5):
                pts = recent_6[i:i+6]
                for func in [_check_impulse, _check_diagonal, _check_triangle]:
                    # Передаем векторы и суб-векторы для фракталов и объема
                    if func == _check_impulse:
                        res = func(pts, tf, vectors, sub_tf_vectors)
                    elif func == _check_diagonal:
                        res = func(pts, tf, vectors)
                    else:
                        res = func(pts, tf)
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
            res = _check_forming_123(extrema[-4:], tf, sub_tf_vectors)
            if res: tf_structures.append(res)

        # Фильтруем только "свежие" паттерны
        recent_ts = {round(float(e.timestamp)) for e in extrema[-20:]}
        # logger.info(f"    [DEBUG] TF:{tf} | Extrema total:{len(extrema)} | Structures found:{len(tf_structures)}")
        for s in tf_structures:
            last_ts = round(float(s.points[-1].timestamp))
            is_recent = last_ts in recent_ts
            logger.info(f"  [MATH] Найден паттерн {s.pattern_type} ({tf}) {s.direction}, конфиденс: {s.confidence:.1f} | Fresh:{is_recent}")
            if is_recent:
                # Добавляем базовый бонус сложности
                bonus = 0
                if "Импульс" in s.pattern_type: bonus = 40
                elif "Диагональ" in s.pattern_type: bonus = 30
                elif "Треугольник" in s.pattern_type: bonus = 25
                elif "Зигзаг" in s.pattern_type: bonus = 10
                s.confidence += bonus
                all_valid_structures.append((tf, s, tf_atr))

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
    for tf, s, atr in all_valid_structures:
        dir_key = "LONG" if s.direction == "БЫЧИЙ" else "SHORT"
        weight = {"1d": 3.0, "4h": 2.0, "1h": 1.5, "15m": 1.0, "5m": 0.5}.get(tf, 1.0)
        direction_votes[dir_key] += (s.confidence / 100.0) * weight

    # Сортируем все найденные структуры по итоговому весу
    def score_structure(item):
        tf, s, atr = item
        dir_key = "LONG" if s.direction == "БЫЧИЙ" else "SHORT"
        # Итоговый балл = уверенность + бонус за общее направление рынка
        return s.confidence + (direction_votes[dir_key] * 10)

    all_valid_structures.sort(key=score_structure, reverse=True)

    # 3. Выбор Main и Alternative
    current_idx = 0
    best_tf, best_s, best_atr = all_valid_structures[current_idx]
    main_plan = _convert_structure_to_plan(best_s, context, best_tf, best_atr)
    
    alt_desc = "Нет альтернатив"
    if len(all_valid_structures) > 1:
        alt_tf, alt_s, _alt_atr = all_valid_structures[1]
        alt_desc = f"Альтернатива: {alt_s.pattern_type} {alt_s.direction} на {alt_tf} ({alt_s.confidence:.0f}%)"

    if not main_plan:
        for i_alt in range(1, len(all_valid_structures)):
            best_tf, best_s, best_atr = all_valid_structures[i_alt]
            main_plan = _convert_structure_to_plan(best_s, context, best_tf, best_atr)
            if main_plan:
                current_idx = i_alt
                break
    if not main_plan:
        return TradePlan(wave_count_label="RR Filtered", main_scenario="WAIT", trade_params={"direction": "WAIT"})

    alt_desc = "Нет альтернатив"
    rem_idx = 0 if current_idx != 0 else 1
    if len(all_valid_structures) > 1:
        alt_tf, alt_s, _alt_atr = all_valid_structures[rem_idx]
        alt_desc = f"Альтернатива: {alt_s.pattern_type} {alt_s.direction} на {alt_tf} ({alt_s.confidence:.0f}%)"

    # Добавляем информацию об альтернативах в логику
    all_seen = [f"{s.pattern_type}({tf})" for tf, s, atr in all_valid_structures[:4]]
    main_plan.detailed_logic = (
        f"ГЛАВНЫЙ: {best_s.pattern_type} ({best_tf}). "
        f"Рассмотрено параллельно: {', '.join(all_seen)}. "
        f"{alt_desc}. {main_plan.detailed_logic}"
    )
    main_plan.alternative_scenario = alt_desc
    
    return main_plan

def _convert_structure_to_plan(s: WaveStructure, context: LLMContext, tf: str, atr: float = 0) -> Optional[TradePlan]:
    """Превращает математическую структуру в торговый план с подтверждением входа."""
    
    # ── Направление сделки ──
    is_reversal = s.pattern_type in ["Импульс", "Зигзаг", "Диагональ", "Плоскость"]
    
    if is_reversal:
        direction = "SHORT" if s.direction == "БЫЧИЙ" else "LONG"
    else:
        direction = "LONG" if s.direction == "БЫЧИЙ" else "SHORT"
    
    # ── ПУНКТ 2.А: Вход по подтверждению (ATR-учет) ──
    last_p = s.points[-1]
    prev_p = s.points[-2]
    
    offset = atr * 0.07 if atr > 0 else (prev_p.price * 0.0005)

    if is_reversal:
        sl = last_p.price
        if direction == "LONG":
            entry = prev_p.price + offset
        else:
            entry = prev_p.price - offset
    else:
        sl = s.invalidation_price
        if direction == "LONG":
            entry = last_p.price + offset
        else:
            entry = last_p.price - offset

    # Убедимся, что стоп не равен входу и находится с нужной стороны
    if not sl or sl == entry:
        logger.warning(f"  [PLAN] Отказ: SL == Entry ({sl})")
        return None
    if direction == "LONG" and sl >= entry:
        logger.warning(f"  [PLAN] Отказ: LONG SL({sl}) >= Entry({entry})")
        return None
    if direction == "SHORT" and sl <= entry:
        logger.warning(f"  [PLAN] Отказ: SHORT SL({sl}) <= Entry({entry})")
        return None
        
    # ── ПУНКТ 4: Геометрический Тейк-Профит ──
    # Ищем цели среди Фибо-уровней и Каналов
    targets = sorted(s.fibo_targets) if s.fibo_targets else []
    if s.channel_target: targets.append(s.channel_target)
    
    if not targets:
        # Если целей нет, используем Фибо-расширение 1.618 как стандарт
        dist = abs(s.points[-1].price - s.points[0].price)
        tp = entry + dist * 1.618 if direction == "LONG" else entry - dist * 1.618
    else:
        # ПУНКТ 2.Б: Выбираем БЛИЖАЙШУЮ цель, проходящую по R:R
        # Сортируем по расстоянию от точки входа
        targets.sort(key=lambda t: abs(t - entry))
        risk = abs(entry - sl)
        valid_targets = [t for t in targets if (abs(t - entry) / risk) >= MIN_RR_RATIO]
        
        if not valid_targets:
            return None
        tp = valid_targets[0] # Самая близкая, но валидная цель

    # Финальная проверка R:R
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk == 0 or (reward / risk) < MIN_RR_RATIO:
        logger.info(f"    [RR_FAIL] План не прошел валидацию: {direction} R:R={reward/risk:.2f} < {MIN_RR_RATIO}")
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
