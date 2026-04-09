"""
scoring.py — Нормы Волн Эллиотта (Guidelines).

Нормы (Guidelines) — это не строгие правила, а статистические закономерности.
Нарушение нормы не инвалидирует паттерн, но снижает его `confidence` (вероятность).
Соблюдение норм — повышает confidence.

Возвращает модификатор уверенности (delta).
"""
from __future__ import annotations

import math
from typing import List

from src.wave_engine.extremum_finder import Extremum


class WaveScoring:
    """Модуль начисления весов/уверенности на основе Норм (Guidelines)."""

    @staticmethod
    def score_impulse_guidelines(points: List[Extremum]) -> tuple[float, dict]:
        """
        Оценивает 6 точек полного импульса и возвращает бонус(штраф) и список причин.
        Returns: (delta_score, details_dict)
        """
        delta = 0.0
        details = {}
        
        n = len(points)
        if n < 6:
            return 0.0, {}

        p0, p1, p2, p3, p4, p5 = points[:6]
        
        len1 = abs(p1.price - p0.price)
        len2 = abs(p2.price - p1.price)
        len3 = abs(p3.price - p2.price)
        len4 = abs(p4.price - p3.price)
        # len5 = abs(p5.price - p4.price)

        t0, t1, t2, t3, t4 = p0.timestamp, p1.timestamp, p2.timestamp, p3.timestamp, p4.timestamp
        dt1 = t1 - t0
        dt2 = t2 - t1
        # dt3 = t3 - t2
        dt4 = t4 - t3

        if len1 == 0 or len3 == 0:
            return 0.0, {}

        # 1. Альтернация по глубине (Alternation of depth) - W2 vs W4
        ratio2 = len2 / len1
        ratio4 = len4 / len3
        
        # W2 обычно глубокая (50-61.8%), W4 мелкая (38.2%)
        is_w2_deep = ratio2 > 0.5
        is_w4_deep = ratio4 > 0.5
        
        if is_w2_deep != is_w4_deep:
            delta += 0.15
            details["alternation_depth"] = True
        else:
            delta -= 0.05
            details["alternation_depth"] = False

        # 2. Альтернация по времени (Alternation of time) - W2 vs W4
        if dt4 > 0 and dt2 > 0:
            time_ratio = max(dt4, dt2) / min(dt4, dt2)
            if time_ratio >= 1.5:
                delta += 0.10
                details["alternation_time"] = True
                
        # 2.1. Time Harmony (Синхронизация степеней волн)
        # Если W4 формируется в 5 раз быстрее чем W2, это почти наверняка субструктура внутри W3
        if dt2 > 0 and dt4 > 0:
            w4_w2_time_ratio = dt4 / dt2
            details["w4_w2_time_ratio"] = w4_w2_time_ratio
            if w4_w2_time_ratio < 0.2:
                delta -= 0.40  # Жесткий пенальти за "куцую" волну 4
                details["w4_too_fast_subwave"] = True
            elif w4_w2_time_ratio > 5.0:
                delta -= 0.40  # Жесткий пенальти, если W2 была микроскопической
                details["w2_too_fast_subwave"] = True
        
        # 3. Фибоначчи зоны W2
        if 0.5 <= ratio2 <= 0.618:
            delta += 0.15
            details["w2_fibo_zone"] = True
        elif ratio2 > 0.8:
            delta -= 0.10
            details["w2_too_deep"] = True

        # 4. Фибоначчи зоны W4
        if 0.382 <= ratio4 <= 0.5:
            delta += 0.15
            details["w4_fibo_zone"] = True
        
        # 5. Расширение W3 (W3 is usually extended)
        extension_ratio = len3 / len1
        if 1.618 <= extension_ratio <= 2.618:
            delta += 0.20
            details["w3_extended"] = True
        elif extension_ratio < 1.0:
            delta -= 0.10
            details["w3_short"] = True

        return delta, details

    @staticmethod
    def calc_fibo_distance(price: float, levels: List[float]) -> float:
        """Расстояние в процентах от цены до ближайшего Фибо-уровня."""
        if not levels:
            return 1.0
        min_dist = min([abs(price - lvl) / price for lvl in levels])
        return min_dist

    @staticmethod
    def score_zscore_volume(volume: float, avg_volume: float) -> float:
        if avg_volume == 0:
            return 0.0
        ratio = volume / avg_volume
        # Простой proxy для Z-score, если мы не храним полную стд:
        if ratio >= 2.5:
            return 0.2  # Сильный всплеск (тормозящий)
        if ratio >= 1.5:
            return 0.1
        return 0.0
