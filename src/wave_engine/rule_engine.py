"""
rule_engine.py — Абсолютные правила Волн Эллиотта (Elliott Wave Theory).

Правила (Rules) — это строгие законы, нарушение которых немедленно инвалидирует гипотезу.
Никаких компромиссов. Если правило нарушено, структура 100% не является тем, чем кажется.

Для каждого паттерна:
Вход: список точек (Extremum)
Выход: Tuple[bool, str] — (valid, invalidation_reason)
"""
from __future__ import annotations

from typing import List

from src.wave_engine.extremum_finder import Extremum


class ElliottRuleEngine:
    """Движок жестких правил для паттернов."""

    @staticmethod
    def validate_impulse(points: List[Extremum]) -> tuple[bool, str]:
        """
        Импульс (5 волн, 6 точек: 0, 1, 2, 3, 4, 5).
        
        Абсолютные правила (Elliott Wave Principle):
        1. Волна 2 никогда не уходит за начало Волны 1.
        2. Волна 3 никогда не бывает самой короткой из действующих (1, 3, 5).
        3. Волна 4 никогда не заходит на ценовую территорию Волны 1.
        4. (Дополнительно, но строго) Волна 3 должна пробивать вершину Волны 1.
        """
        n = len(points)
        if n < 3:
            return True, ""  # Невозможно валидировать, пока слишком мало точек

        p0 = points[0]
        p1 = points[1]
        bullish = p1.price > p0.price

        # Если есть точка 2 (конец волны 2)
        if n >= 3:
            p2 = points[2]
            # Правило 1: W2 не уходит за начало W1
            if bullish and p2.price <= p0.price:
                return False, "W2_BEYOND_W1_START"
            if not bullish and p2.price >= p0.price:
                return False, "W2_BEYOND_W1_START"

        # Если есть точка 3 (конец волны 3)
        if n >= 4:
            p3 = points[3]
            # Пик W3 должен лежать в направлении импульса
            if bullish and p3.price <= p1.price:
                return False, "W3_DOES_NOT_BREAK_W1"
            if not bullish and p3.price >= p1.price:
                return False, "W3_DOES_NOT_BREAK_W1"

        # Если есть точка 4 (конец волны 4)
        if n >= 5:
            p4 = points[4]
            # Правило 3: W4 не заходит на территорию W1 (перекрытие)
            if bullish and p4.price <= p1.price:
                return False, "W4_OVERLAPS_W1"
            if not bullish and p4.price >= p1.price:
                return False, "W4_OVERLAPS_W1"

            # W4 не должна уходить за начало W3 (W2)
            if bullish and p4.price <= p2.price:
                return False, "W4_BEYOND_W3_START"
            if not bullish and p4.price >= p2.price:
                return False, "W4_BEYOND_W3_START"

        # Если есть точка 5 (конец волны 5) - завершенный импульс
        if n >= 6:
            p5 = points[5]
            # Правило 2: W3 не может быть самой короткой
            len1 = abs(p1.price - p0.price)
            len3 = abs(p3.price - p2.price)
            len5 = abs(p5.price - p4.price)

            if len3 < len1 and len3 < len5:
                return False, "W3_SHORTEST_MOTIVE_WAVE"

            # W5 обычно пробивает вершину W3 (допускается усечение, но мы требуем пробоя для жестких импульсов)
            if bullish and p5.price <= p3.price:
                return False, "W5_DOES_NOT_BREAK_W3"
            if not bullish and p5.price >= p3.price:
                return False, "W5_DOES_NOT_BREAK_W3"

        return True, ""

    @staticmethod
    def validate_diagonal(points: List[Extremum]) -> tuple[bool, str]:
        """
        Начальный / Конечный Клин (Diagonal, 5 волн, 6 точек).
        
        Особенности:
        1. Волна 4 ОБЯЗАНА заходить в территорию Волны 1 (перекрытие).
        2. Формирует сходящийся или расходящийся клин.
        3. Волна 3 не самая короткая.
        """
        n = len(points)
        if n < 3:
            return True, ""

        p0 = points[0]
        p1 = points[1]
        bullish = p1.price > p0.price

        if n >= 3:
            p2 = points[2]
            if bullish and p2.price <= p0.price:
                return False, "W2_BEYOND_W1_START"
            if not bullish and p2.price >= p0.price:
                return False, "W2_BEYOND_W1_START"

        if n >= 4:
            p3 = points[3]
            if bullish and p3.price <= p1.price:
                return False, "W3_DOES_NOT_BREAK_W1"
            if not bullish and p3.price >= p1.price:
                return False, "W3_DOES_NOT_BREAK_W1"

        if n >= 5:
            p4 = points[4]
            # КЛЮЧЕВОЕ: Перекрытие в диагонали обязательно!
            if bullish and p4.price > p1.price:
                return False, "NO_OVERLAP_IN_DIAGONAL"
            if not bullish and p4.price < p1.price:
                return False, "NO_OVERLAP_IN_DIAGONAL"

            if bullish and p4.price <= p2.price:
                return False, "W4_BEYOND_W3_START"
            if not bullish and p4.price >= p2.price:
                return False, "W4_BEYOND_W3_START"

        if n >= 6:
            p5 = points[5]
            len1 = abs(p1.price - p0.price)
            len3 = abs(p3.price - p2.price)
            len5 = abs(p5.price - p4.price)

            if len3 < len1 and len3 < len5:
                return False, "W3_SHORTEST_MOTIVE_WAVE"

            if bullish and p5.price <= p3.price:
                return False, "W5_DOES_NOT_BREAK_W3"
            if not bullish and p5.price >= p3.price:
                return False, "W5_DOES_NOT_BREAK_W3"

            # Проверка сходящегося/расширяющегося типа
            is_contracting = (len1 > len3 > len5)
            # is_expanding = (len1 < len3 < len5) # Менее строгий
            is_expanding = (len3 > len1 and len5 > len3)
            
            if not (is_contracting or is_expanding):
                return False, "NOT_CONTRACTING_OR_EXPANDING_PROPERLY"

        return True, ""

    @staticmethod
    def validate_zigzag(points: List[Extremum]) -> tuple[bool, str]:
        """
        Зигзаг (A-B-C, 4 точки: 0, A, B, C).
        
        Правила:
        1. Волна B не должна заходить за начало A.
        2. Волна C должна пробить вершину A.
        """
        n = len(points)
        if n < 3:
            return True, ""

        p0 = points[0]
        pA = points[1]
        bullish = pA.price > p0.price

        if n >= 3:
            pB = points[2]
            # W_B не уходит за W_A start
            lenA = abs(pA.price - p0.price)
            lenB = abs(pB.price - pA.price)
            
            if lenB >= lenA:
                return False, "WB_RETRACES_100_WA"

        if n >= 4:
            pC = points[3]
            # W_C должна пробить A (в направлении тренда)
            if bullish and pC.price <= pA.price:
                return False, "WC_DOES_NOT_BREAK_WA"
            if not bullish and pC.price >= pA.price:
                return False, "WC_DOES_NOT_BREAK_WA"

        return True, ""

    @staticmethod
    def validate_flat(points: List[Extremum]) -> tuple[bool, str]:
        """
        Плоскость (A-B-C, 4 точки).
        
        Особенности:
        1. Волна B откатывает минимум на 90% от A. (По ТЗ >= 0.9)
        2. Волна C обычно соразмерна A, но не обязательно пробивает вершину.
        """
        n = len(points)
        if n < 3:
            return True, ""

        p0 = points[0]
        pA = points[1]

        if n >= 3:
            pB = points[2]
            lenA = abs(pA.price - p0.price)
            lenB = abs(pB.price - pA.price)

            if lenA == 0:
                return False, "ZERO_LENGTH"

            ratioB = lenB / lenA
            if ratioB < 0.90:
                return False, "WB_RETRACE_LESS_THAN_90"

        return True, ""
