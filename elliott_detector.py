"""
elliott_detector.py — Алгоритмический детектор 5-волнового импульса Эллиотта.

Модуль математически выявляет и верифицирует структуру волн W0-W1-W2-W3-W4-W5
перед точкой входа (сигналом дивергенции):
1. W3 и W5 берутся из точек подтверждённой RSI-дивергенции (P1 и P2).
2. W4 определяется как экстремум отката строго между W3 и W5.
3. W0, W1, W2 находятся перебором локальных свингов до W3 с проверкой:
   - RULE 1: Волна 2 никогда не уходит за начало Волны 1 (W0).
   - RULE 2: Волна 3 никогда не является самой короткой среди (W1, W3, W5).
   - RULE 3: Волна 4 никогда не заходит на ценовую территорию Волны 1 (Overlap = 0).
   - Дополнительно: подтверждение локальных экстремумов каждой волны.
4. Скоринг качества (0-100):
   - Доминирование W3 (самая длинная импульсная волна)
   - Пропорции Фибоначчи (W2 retrace 38-78%, W4 retrace 23-50%)
   - Чередование W2 и W4 по глубине и времени
   - Направленная чистота движения (Efficiency Ratio: net_move / gross_move)
   - Достаточная длительность в барах для каждой волны
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


@dataclass
class WavePoint:
    """Точка разволновки Эллиотта."""
    label: str          # "W0", "W1", "W2", "W3", "W4", "W5"
    bar_idx: int        # Индекс бара в общем DataFrame
    price: float        # Экстремум (high для вершин, low для впадин)


@dataclass
class ElliottImpulseResult:
    """Результат алгоритмического анализа 5-волнового импульса."""
    is_valid: bool                                # Выполнены ли все 3 строгих правила
    score: float                                  # Качество структуры от 0 до 100
    wave_direction: str                           # "bullish" (для SHORT) или "bearish" (для LONG)
    wave_points: Dict[str, float]                 # {"W0": price, ..., "W5": price}
    wave_indices: Dict[str, int]                  # {"W0": bar_idx, ..., "W5": bar_idx}
    rule_violations: List[str]                    # Список нарушений (если есть)
    quality_factors: Dict[str, float]             # Детальные баллы
    details: Dict[str, Any]                       # Длины, ретрейсменты, длительность
    reason: str                                   # Краткое резюме анализа


def _calculate_efficiency(close: np.ndarray, start_bar: int, end_bar: int) -> float:
    """
    Вычисляет коэффициент направленной эффективности (Efficiency Ratio):
    net_move / sum(|step_moves|).
    1.0 = идеальная прямая линия, 0.1 = хаотичный шум.
    """
    if end_bar <= start_bar:
        return 0.0
    net_move = abs(close[end_bar] - close[start_bar])
    gross_move = float(np.sum(np.abs(np.diff(close[start_bar : end_bar + 1]))))
    if gross_move < 1e-9:
        return 0.0
    return float(net_move / gross_move)


def _check_origin_divergence(
    high: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    w0_b: int,
    is_bullish: bool,
    lookback: int = 35,
) -> Tuple[bool, Optional[int], Optional[float], Optional[float]]:
    """
    Проверяет, сформировалась ли точка W0 на разворотной дивергенции относительно
    предшествующего свинга:
    - Для Bullish: W0 low <= prev low, но RSI(W0) > RSI(prev) + 2.0 (бычья дивергенция на старте).
    - Для Bearish: W0 high >= prev high, но RSI(W0) < RSI(prev) - 2.0 (медвежья дивергенция на старте).
    Возвращает: (найдено_ли, prev_bar_idx, prev_price, prev_rsi).
    """
    start = max(0, w0_b - lookback)
    end = max(0, w0_b - 2)
    if end <= start + 2:
        return False, None, None, None
    
    if is_bullish:
        p_lows = argrelextrema(low[start:end], np.less_equal, order=2)[0] + start
        if len(p_lows) == 0:
            p_lows = argrelextrema(low[start:end], np.less_equal, order=1)[0] + start
        w0_low = float(low[w0_b])
        w0_rsi = float(np.min(rsi[max(0, w0_b - 1) : min(len(rsi), w0_b + 2)]))
        for pb in reversed(p_lows):
            if low[pb] >= w0_low and rsi[pb] < w0_rsi - 2.0:
                return True, int(pb), float(low[pb]), float(rsi[pb])
    else:
        p_highs = argrelextrema(high[start:end], np.greater_equal, order=2)[0] + start
        if len(p_highs) == 0:
            p_highs = argrelextrema(high[start:end], np.greater_equal, order=1)[0] + start
        w0_high = float(high[w0_b])
        w0_rsi = float(np.max(rsi[max(0, w0_b - 1) : min(len(rsi), w0_b + 2)]))
        for pb in reversed(p_highs):
            if high[pb] <= w0_high and rsi[pb] > w0_rsi + 2.0:
                return True, int(pb), float(high[pb]), float(rsi[pb])
    return False, None, None, None
    
    if is_bullish:
        p_lows = argrelextrema(low[start:end], np.less_equal, order=2)[0] + start
        if len(p_lows) == 0:
            p_lows = argrelextrema(low[start:end], np.less_equal, order=1)[0] + start
        w0_low = low[w0_b]
        w0_rsi = float(np.min(rsi[max(0, w0_b - 1) : min(len(rsi), w0_b + 2)]))
        for pb in p_lows:
            if low[pb] >= w0_low and rsi[pb] < w0_rsi - 2.0:
                return True
    else:
        p_highs = argrelextrema(high[start:end], np.greater_equal, order=2)[0] + start
        if len(p_highs) == 0:
            p_highs = argrelextrema(high[start:end], np.greater_equal, order=1)[0] + start
        w0_high = high[w0_b]
        w0_rsi = float(np.max(rsi[max(0, w0_b - 1) : min(len(rsi), w0_b + 2)]))
        for pb in p_highs:
            if high[pb] <= w0_high and rsi[pb] > w0_rsi + 2.0:
                return True
    return False


def detect_elliott_impulse(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_p: np.ndarray,
    w3_bar: int,
    w5_bar: int,
    direction: str,
    max_lookback_bars: int = 100,
    min_wave_bars: int = 2,
    rsi: Optional[np.ndarray] = None,
) -> ElliottImpulseResult:
    """
    Находит и математически верифицирует лучший 5-волновой импульс Эллиотта
    перед точкой дивергенции.

    Args:
        high, low, close, open_p: Массивы цен
        w3_bar: Индекс бара P1 (первый пивот дивергенции, вершина W3)
        w5_bar: Индекс бара P2 (второй пивот дивергенции, вершина W5)
        direction: "SHORT" (ищем bullish импульс вверх) или "LONG" (ищем bearish импульс вниз)
        max_lookback_bars: Макс. число баров до W3 для поиска W0-W1-W2
        min_wave_bars: Минимальная длительность каждой волны в свечах

    Returns:
        ElliottImpulseResult с полной разволновкой, баллами и диагностикой.
    """
    is_bullish = (direction == "SHORT")
    wave_dir = "bullish" if is_bullish else "bearish"

    # Базовая валидация расстояния между пивотами дивергенции
    if w5_bar - w3_bar < 2:
        return ElliottImpulseResult(
            is_valid=False,
            score=0.0,
            wave_direction=wave_dir,
            wave_points={},
            wave_indices={},
            rule_violations=["Между W3 и W5 менее 2 баров — нет места для коррекции W4"],
            quality_factors={},
            details={},
            reason="Недостаточно баров между точками дивергенции для формирования W4",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Определение W3, W4, W5
    # ═══════════════════════════════════════════════════════════════════════
    if is_bullish:
        # Bullish импульс: W3 = high, W5 = high, W4 = min(low) между ними
        w3_price = float(high[w3_bar])
        w5_price = float(high[w5_bar])
        w4_rel = int(np.argmin(low[w3_bar + 1 : w5_bar]))
        w4_bar = w3_bar + 1 + w4_rel
        w4_price = float(low[w4_bar])

        # Проверка базовой структуры W3-W4-W5
        if w5_price <= w3_price:
            return ElliottImpulseResult(
                is_valid=False, score=0.0, wave_direction=wave_dir,
                wave_points={}, wave_indices={},
                rule_violations=["W5 не выше W3 в бычьем импульсе"],
                quality_factors={}, details={},
                reason="W5 не сделала higher high относительно W3",
            )
        if w4_price >= w3_price:
            return ElliottImpulseResult(
                is_valid=False, score=0.0, wave_direction=wave_dir,
                wave_points={}, wave_indices={},
                rule_violations=["W4 не скорректировалась ниже W3"],
                quality_factors={}, details={},
                reason="W4 не является коррекцией от W3",
            )
    else:
        # Bearish импульс: W3 = low, W5 = low, W4 = max(high) между ними
        w3_price = float(low[w3_bar])
        w5_price = float(low[w5_bar])
        w4_rel = int(np.argmax(high[w3_bar + 1 : w5_bar]))
        w4_bar = w3_bar + 1 + w4_rel
        w4_price = float(high[w4_bar])

        # Проверка базовой структуры W3-W4-W5
        if w5_price >= w3_price:
            return ElliottImpulseResult(
                is_valid=False, score=0.0, wave_direction=wave_dir,
                wave_points={}, wave_indices={},
                rule_violations=["W5 не ниже W3 в медвежьем импульсе"],
                quality_factors={}, details={},
                reason="W5 не сделала lower low относительно W3",
            )
        if w4_price <= w3_price:
            return ElliottImpulseResult(
                is_valid=False, score=0.0, wave_direction=wave_dir,
                wave_points={}, wave_indices={},
                rule_violations=["W4 не скорректировалась выше W3"],
                quality_factors={}, details={},
                reason="W4 не является коррекцией от W3",
            )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Поиск кандидатов W0, W1, W2 до W3
    # ═══════════════════════════════════════════════════════════════════════
    start_search = max(0, w3_bar - max_lookback_bars)
    if w3_bar - start_search < 6:
        return ElliottImpulseResult(
            is_valid=False, score=0.0, wave_direction=wave_dir,
            wave_points={}, wave_indices={},
            rule_violations=["Недостаточно истории до W3 для формирования W0, W1, W2"],
            quality_factors={}, details={},
            reason="Слишком мало свечей до W3",
        )

    # Находим локальные экстремумы для кандидатов
    window_high = high[start_search:w3_bar]
    window_low = low[start_search:w3_bar]

    # Свинги разного порядка (1 и 2) для надёжного покрытия
    p_highs_1 = set(argrelextrema(window_high, np.greater_equal, order=1)[0] + start_search)
    p_lows_1 = set(argrelextrema(window_low, np.less_equal, order=1)[0] + start_search)
    p_highs_2 = set(argrelextrema(window_high, np.greater_equal, order=2)[0] + start_search)
    p_lows_2 = set(argrelextrema(window_low, np.less_equal, order=2)[0] + start_search)

    cand_highs = sorted(list(p_highs_1 | p_highs_2))
    cand_lows = sorted(list(p_lows_1 | p_lows_2))

    valid_candidates: List[Dict[str, Any]] = []
    recorded_violations: List[str] = []

    if is_bullish:
        # BULLISH IMPULSE: W0(low) -> W1(high) -> W2(low) -> W3(high) -> W4(low) -> W5(high)
        for w2_b in cand_lows:
            if not (start_search < w2_b < w3_bar):
                continue
            w2_p = float(low[w2_b])

            # Базовые условия для W2
            if w2_p >= w3_price:
                continue
            if np.min(low[w2_b : w3_bar]) < w2_p:
                continue  # W2 должен быть истинным минимумом отката
            if np.max(high[w2_b : w3_bar]) > w3_price:
                continue

            for w1_b in cand_highs:
                if not (start_search < w1_b < w2_b):
                    continue
                w1_p = float(high[w1_b])

                # W1 выше W2, W3 выше W1
                if w1_p <= w2_p or w3_price <= w1_p:
                    continue

                # ── RULE 3: W4 НЕ перекрывает W1 ──
                if w4_price <= w1_p:
                    recorded_violations.append(f"Rule 3 violation: W4 low ({w4_price:.1f}) <= W1 high ({w1_p:.1f})")
                    continue
                # И во всём диапазоне W3..W5 цена не заходит за W1
                if np.min(low[w3_bar : w5_bar + 1]) <= w1_p:
                    recorded_violations.append(f"Rule 3 violation: pullback between W3-W5 overlaps W1 ({w1_p:.1f})")
                    continue

                # Чистота отката W1 -> W2
                if np.max(high[w1_b : w2_b]) > w1_p:
                    continue
                if np.min(low[w1_b : w2_b]) < w2_p:
                    continue

                for w0_b in cand_lows:
                    if not (start_search <= w0_b < w1_b):
                        continue
                    w0_p = float(low[w0_b])

                    # ── RULE 1: W2 НЕ заходит за начало W1 (W0) ──
                    if w2_p <= w0_p:
                        recorded_violations.append(f"Rule 1 violation: W2 low ({w2_p:.1f}) <= W0 low ({w0_p:.1f})")
                        continue
                    if w0_p >= w1_p:
                        continue
                    if np.min(low[w0_b : w1_b]) < w0_p:
                        continue
                    if np.max(high[w0_b : w1_b]) > w1_p:
                        continue

                    # Длины волн
                    l1 = w1_p - w0_p
                    l3 = w3_price - w2_p
                    l5 = w5_price - w4_price

                    if l1 <= 0 or l3 <= 0 or l5 <= 0:
                        continue

                    # ── RULE 2: W3 НЕ самая короткая волна ──
                    if l3 < min(l1, l5):
                        recorded_violations.append(f"Rule 2 violation: W3 length ({l3:.1f}) is shortest (L1={l1:.1f}, L5={l5:.1f})")
                        continue

                    # Минимальные требования к барам
                    dur_w1 = w1_b - w0_b
                    dur_w2 = w2_b - w1_b
                    dur_w3 = w3_bar - w2_b
                    dur_w4 = w4_bar - w3_bar
                    dur_w5 = w5_bar - w4_bar

                    if min(dur_w1, dur_w2, dur_w3, dur_w4, dur_w5) < 1:
                        continue
                    if dur_w1 < min_wave_bars and dur_w3 < min_wave_bars:
                        continue

                    valid_candidates.append({
                        "w0": (w0_b, w0_p),
                        "w1": (w1_b, w1_p),
                        "w2": (w2_b, w2_p),
                        "w3": (w3_bar, w3_price),
                        "w4": (w4_bar, w4_price),
                        "w5": (w5_bar, w5_price),
                        "l1": l1, "l3": l3, "l5": l5,
                        "dur_w1": dur_w1, "dur_w2": dur_w2, "dur_w3": dur_w3,
                        "dur_w4": dur_w4, "dur_w5": dur_w5,
                    })

    else:
        # BEARISH IMPULSE: W0(high) -> W1(low) -> W2(high) -> W3(low) -> W4(high) -> W5(low)
        for w2_b in cand_highs:
            if not (start_search < w2_b < w3_bar):
                continue
            w2_p = float(high[w2_b])

            if w2_p <= w3_price:
                continue
            if np.max(high[w2_b : w3_bar]) > w2_p:
                continue
            if np.min(low[w2_b : w3_bar]) < w3_price:
                continue

            for w1_b in cand_lows:
                if not (start_search < w1_b < w2_b):
                    continue
                w1_p = float(low[w1_b])

                if w1_p >= w2_p or w3_price >= w1_p:
                    continue

                # ── RULE 3: W4 НЕ перекрывает W1 ──
                if w4_price >= w1_p:
                    recorded_violations.append(f"Rule 3 violation: W4 high ({w4_price:.1f}) >= W1 low ({w1_p:.1f})")
                    continue
                if np.max(high[w3_bar : w5_bar + 1]) >= w1_p:
                    recorded_violations.append(f"Rule 3 violation: pullback between W3-W5 overlaps W1 ({w1_p:.1f})")
                    continue

                if np.min(low[w1_b : w2_b]) < w1_p:
                    continue
                if np.max(high[w1_b : w2_b]) > w2_p:
                    continue

                for w0_b in cand_highs:
                    if not (start_search <= w0_b < w1_b):
                        continue
                    w0_p = float(high[w0_b])

                    # ── RULE 1: W2 НЕ заходит за начало W1 (W0) ──
                    if w2_p >= w0_p:
                        recorded_violations.append(f"Rule 1 violation: W2 high ({w2_p:.1f}) >= W0 high ({w0_p:.1f})")
                        continue
                    if w0_p <= w1_p:
                        continue
                    if np.max(high[w0_b : w1_b]) > w0_p:
                        continue
                    if np.min(low[w0_b : w1_b]) < w1_p:
                        continue

                    l1 = w0_p - w1_p
                    l3 = w2_p - w3_price
                    l5 = w4_price - w5_price

                    if l1 <= 0 or l3 <= 0 or l5 <= 0:
                        continue

                    # ── RULE 2: W3 НЕ самая короткая волна ──
                    if l3 < min(l1, l5):
                        recorded_violations.append(f"Rule 2 violation: W3 length ({l3:.1f}) is shortest (L1={l1:.1f}, L5={l5:.1f})")
                        continue

                    dur_w1 = w1_b - w0_b
                    dur_w2 = w2_b - w1_b
                    dur_w3 = w3_bar - w2_b
                    dur_w4 = w4_bar - w3_bar
                    dur_w5 = w5_bar - w4_bar

                    if min(dur_w1, dur_w2, dur_w3, dur_w4, dur_w5) < 1:
                        continue
                    if dur_w1 < min_wave_bars and dur_w3 < min_wave_bars:
                        continue

                    valid_candidates.append({
                        "w0": (w0_b, w0_p),
                        "w1": (w1_b, w1_p),
                        "w2": (w2_b, w2_p),
                        "w3": (w3_bar, w3_price),
                        "w4": (w4_bar, w4_price),
                        "w5": (w5_bar, w5_price),
                        "l1": l1, "l3": l3, "l5": l5,
                        "dur_w1": dur_w1, "dur_w2": dur_w2, "dur_w3": dur_w3,
                        "dur_w4": dur_w4, "dur_w5": dur_w5,
                    })

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Если нет валидных кандидатов — отказ с диагностикой
    # ═══════════════════════════════════════════════════════════════════════
    if not valid_candidates:
        unique_violations = list(dict.fromkeys(recorded_violations))[:5]
        return ElliottImpulseResult(
            is_valid=False,
            score=0.0,
            wave_direction=wave_dir,
            wave_points={},
            wave_indices={},
            rule_violations=unique_violations or ["Ни одна комбинация свингов не удовлетворила все 3 правила Эллиотта"],
            quality_factors={},
            details={},
            reason="Структура не соответствует 5-волновому импульсу Эллиотта (нарушены базовые правила)",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Скоринг и выбор лучшего кандидата
    # ═══════════════════════════════════════════════════════════════════════
    scored_candidates = []

    for cand in valid_candidates:
        w0_b, w0_p = cand["w0"]
        w1_b, w1_p = cand["w1"]
        w2_b, w2_p = cand["w2"]
        w3_b, w3_p = cand["w3"]
        w4_b, w4_p = cand["w4"]
        w5_b, w5_p = cand["w5"]
        l1, l3, l5 = cand["l1"], cand["l3"], cand["l5"]

        # ── Базовый балл за соблюдение 3 строгих правил (30 pts) ──
        score_base = 30.0

        # ── 1. Доминирование W3 (до 25 pts) ──
        score_w3 = 0.0
        w3_longest = (l3 > l1 and l3 > l5)
        if w3_longest:
            score_w3 += 20.0
            if l3 >= 1.5 * l1:
                score_w3 += 5.0  # Классическое расширение 161.8%
        elif l3 >= l1:
            score_w3 += 10.0
        else:
            score_w3 += 5.0  # l3 >= l5 (минимум по правилу 2)

        # ── 2. Пропорции Фибоначчи (до 20 pts) ──
        score_fib = 0.0
        if is_bullish:
            retrace_w2 = (w1_p - w2_p) / l1
            retrace_w4 = (w3_p - w4_p) / l3
        else:
            retrace_w2 = (w2_p - w1_p) / l1
            retrace_w4 = (w4_p - w3_p) / l3

        # W2 retrace (идеально 50-61.8%, допустимо 38.2-78.6%)
        if 0.45 <= retrace_w2 <= 0.65:
            score_fib += 8.0
        elif 0.35 <= retrace_w2 <= 0.80:
            score_fib += 5.0
        else:
            score_fib += 2.0

        # W4 retrace (идеально 23.6-38.2%, допустимо 15-50%)
        if 0.20 <= retrace_w4 <= 0.40:
            score_fib += 8.0
        elif 0.12 <= retrace_w4 <= 0.52:
            score_fib += 5.0
        else:
            score_fib += 2.0

        # W5 vs W1 соотношение (0.618 - 1.618)
        ratio_5_1 = l5 / l1
        if 0.60 <= ratio_5_1 <= 1.65:
            score_fib += 4.0

        # ── 3. Чередование W2 и W4 (до 10 pts) ──
        score_alt = 0.0
        depth_diff = abs(retrace_w2 - retrace_w4)
        if depth_diff >= 0.15:
            score_alt += 6.0  # Одно глубокое, другое мелкое
        dur_w2 = cand["dur_w2"]
        dur_w4 = cand["dur_w4"]
        if (dur_w2 >= 1.5 * dur_w4) or (dur_w4 >= 1.5 * dur_w2):
            score_alt += 4.0  # Чередование по времени

        # ── 4. Чистота движения волн (Efficiency Ratio) (до 15 pts) ──
        eff_1 = _calculate_efficiency(close, w0_b, w1_b)
        eff_3 = _calculate_efficiency(close, w2_b, w3_b)
        eff_5 = _calculate_efficiency(close, w4_b, w5_b)
        avg_eff = (eff_1 + eff_3 + eff_5) / 3.0

        if avg_eff >= 0.55:
            score_eff = 15.0
        elif avg_eff >= 0.42:
            score_eff = 10.0
        elif avg_eff >= 0.30:
            score_eff = 5.0
        else:
            score_eff = 2.0

        # Временная пропорция: W3 не должна быть многократным боковым распилом
        dur_w1 = cand["dur_w1"]
        dur_w3 = cand["dur_w3"]
        if dur_w3 >= 6 * max(1, dur_w1) and eff_3 < 0.45:
            score_eff -= 15.0  # Штраф за искусственную растянутость во флэте

        # ── 5. RSI Анализ истощения W0 и моментума волн (до 25 pts) ──
        score_rsi = 0.0
        w0_rsi_val = None
        w0_status = "N/A"
        origin_div = False
        w3_momentum_peak = False
        orig_prev_b, orig_prev_p, orig_prev_r = None, None, None

        if rsi is not None and len(rsi) > w5_b:
            if is_bullish:
                w0_rsi_val = float(np.min(rsi[max(0, w0_b - 1) : min(len(rsi), w0_b + 2)]))
                if w0_rsi_val <= 32.0:
                    score_rsi += 12.0  # Глубокая перепроданность / капитуляция
                    w0_status = "OVERSOLD"
                elif w0_rsi_val <= 40.0:
                    score_rsi += 6.0   # Допустимое охлаждение
                    w0_status = "COOL"
                elif w0_rsi_val > 48.0:
                    score_rsi -= 12.0  # Штраф за зарождение во флэте
                    w0_status = "CHOP"

                # Предваряющая дивергенция на старте (W0 Origin Divergence)
                origin_div, orig_prev_b, orig_prev_p, orig_prev_r = _check_origin_divergence(high, low, rsi, w0_b, is_bullish=True)
                if origin_div:
                    score_rsi += 8.0

                # Закон моментума волны 3: RSI(W3) должен быть пиком
                rsi_w1 = float(np.max(rsi[max(0, w1_b - 1) : min(len(rsi), w1_b + 2)]))
                rsi_w3 = float(np.max(rsi[max(0, w3_b - 1) : min(len(rsi), w3_b + 2)]))
                rsi_w5 = float(np.max(rsi[max(0, w5_b - 1) : min(len(rsi), w5_b + 2)]))
                if rsi_w3 >= rsi_w1 and rsi_w3 >= rsi_w5:
                    w3_momentum_peak = True
                    score_rsi += 5.0
                elif rsi_w1 > rsi_w3 + 4.0:
                    score_rsi -= 6.0  # W1 сильнее W3 — признак коррекции ABC
            else:
                w0_rsi_val = float(np.max(rsi[max(0, w0_b - 1) : min(len(rsi), w0_b + 2)]))
                if w0_rsi_val >= 68.0:
                    score_rsi += 12.0  # Глубокая перекупленность
                    w0_status = "OVERBOUGHT"
                elif w0_rsi_val >= 60.0:
                    score_rsi += 6.0
                    w0_status = "HOT"
                elif w0_rsi_val < 52.0:
                    score_rsi -= 12.0  # Штраф за зарождение во флэте
                    w0_status = "CHOP"

                origin_div, orig_prev_b, orig_prev_p, orig_prev_r = _check_origin_divergence(high, low, rsi, w0_b, is_bullish=False)
                if origin_div:
                    score_rsi += 8.0

                rsi_w1 = float(np.min(rsi[max(0, w1_b - 1) : min(len(rsi), w1_b + 2)]))
                rsi_w3 = float(np.min(rsi[max(0, w3_b - 1) : min(len(rsi), w3_b + 2)]))
                rsi_w5 = float(np.min(rsi[max(0, w5_b - 1) : min(len(rsi), w5_b + 2)]))
                if rsi_w3 <= rsi_w1 and rsi_w3 <= rsi_w5:
                    w3_momentum_peak = True
                    score_rsi += 5.0
                elif rsi_w1 < rsi_w3 - 4.0:
                    score_rsi -= 6.0

        raw_score = score_base + score_w3 + score_fib + score_alt + score_eff + score_rsi
        total_score = max(5.0, min(100.0, raw_score))

        scored_candidates.append({
            "cand": cand,
            "score": total_score,
            "w3_longest": w3_longest,
            "w0_status": w0_status,
            "w0_rsi": w0_rsi_val,
            "origin_div": origin_div,
            "origin_div_bar": orig_prev_b,
            "origin_div_price": orig_prev_p,
            "origin_div_rsi": orig_prev_r,
            "w3_momentum_peak": w3_momentum_peak,
            "avg_eff": avg_eff,
            "retrace_w2": retrace_w2,
            "retrace_w4": retrace_w4,
            "ratio_5_1": ratio_5_1,
            "quality_factors": {
                "base_rules": score_base,
                "w3_dominance": score_w3,
                "fibonacci": score_fib,
                "alternation": score_alt,
                "cleanliness": score_eff,
                "rsi_momentum": score_rsi,
            }
        })

    # Сортируем: сначала те где W3 самая длинная, затем отфильтровываем CHOP на W0, затем по общему баллу
    scored_candidates.sort(
        key=lambda x: (x["w3_longest"], x["w0_status"] != "CHOP", x["score"]),
        reverse=True,
    )
    best = scored_candidates[0]
    best_cand = best["cand"]

    w_points = {
        "W0": best_cand["w0"][1],
        "W1": best_cand["w1"][1],
        "W2": best_cand["w2"][1],
        "W3": best_cand["w3"][1],
        "W4": best_cand["w4"][1],
        "W5": best_cand["w5"][1],
    }

    w_indices = {
        "W0": best_cand["w0"][0],
        "W1": best_cand["w1"][0],
        "W2": best_cand["w2"][0],
        "W3": best_cand["w3"][0],
        "W4": best_cand["w4"][0],
        "W5": best_cand["w5"][0],
    }

    details = {
        "l1": best_cand["l1"],
        "l3": best_cand["l3"],
        "l5": best_cand["l5"],
        "w3_longest": best["w3_longest"],
        "retrace_w2": best["retrace_w2"],
        "retrace_w4": best["retrace_w4"],
        "ratio_5_1": best["ratio_5_1"],
        "avg_efficiency": best["avg_eff"],
        "dur_w1": best_cand["dur_w1"],
        "dur_w2": best_cand["dur_w2"],
        "dur_w3": best_cand["dur_w3"],
        "dur_w4": best_cand["dur_w4"],
        "dur_w5": best_cand["dur_w5"],
        "total_bars": best_cand["w5"][0] - best_cand["w0"][0],
        "w0_rsi": best["w0_rsi"],
        "w0_status": best["w0_status"],
        "origin_div": best["origin_div"],
        "origin_div_bar": best["cand"].get("origin_div_bar") if "origin_div_bar" in best["cand"] else best.get("origin_div_bar"),
        "origin_div_price": best.get("origin_div_price"),
        "origin_div_rsi": best.get("origin_div_rsi"),
        "w3_momentum_peak": best["w3_momentum_peak"],
    }

    l1_s, l3_s, l5_s = best_cand["l1"], best_cand["l3"], best_cand["l5"]
    w3_status = "W3 самая длинная" if best["w3_longest"] else "W3 >= min(W1, W5)"
    
    w0_desc = ""
    if best["w0_rsi"] is not None:
        w0_desc = f"W0 RSI={best['w0_rsi']:.1f} ({best['w0_status']})"
        if best["origin_div"]:
            w0_desc += " [Origin DIV: OK]"
        w0_desc += ". "

    reason = (
        f"Валидный {wave_dir} импульс (Score: {best['score']:.0f}). "
        f"{w3_status} (L1={l1_s:.1f}, L3={l3_s:.1f}, L5={l5_s:.1f}). "
        f"{w0_desc}"
        f"Ретрейс W2={best['retrace_w2']:.1%}, W4={best['retrace_w4']:.1%}. "
        f"Overlap W4/W1 отсутствует. Чистота: {best['avg_eff']:.2f}."
    )

    return ElliottImpulseResult(
        is_valid=True,
        score=best["score"],
        wave_direction=wave_dir,
        wave_points=w_points,
        wave_indices=w_indices,
        rule_violations=[],
        quality_factors=best["quality_factors"],
        details=details,
        reason=reason,
    )
