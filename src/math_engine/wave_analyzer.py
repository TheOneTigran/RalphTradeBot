"""
wave_analyzer.py — Математический анализатор структур Волн Эллиотта V3.

Улучшения V3:
  1. Волна 5 ОБЯЗАНА пробивать конец волны 3 (строгое правило Эллиотта)
  2. Объёмный профиль (CVD) интегрирован в проверку импульса:
     - Максимальный кумулятивный объём ДОЛЖЕН быть в волне 3
     - Если максимум в волне 5 → снижаем уверенность (скорее C, не 5)
  3. Треугольник (Triangle A-B-C-D-E): сужающийся боковик из 5 субволн
  4. Параллельные Каналы Эллиотта (Channeling):
     - Канал по точкам 2-4, параллельная через 3 → цель W5
  5. Подготовка к Фрактальной Рекурсии: analyze_wave_structure принимает
     sub_tf_data для кросс-ТФ валидации (если внутри W3 нет 5-волнового
     импульса на младшем ТФ → снижаем уверенность)
"""

from __future__ import annotations
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import datetime
import logging
import math

from src.core.models import Vector

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Карта степеней волн (Degree Labels) — строгая нотация по таймфреймам
# ═════════════════════════════════════════════════════════════════════════════

DEGREE_LABELS: Dict[str, Dict[str, List[str]]] = {
    "1w": {
        "impulse":    ["Старт", "I", "II", "III", "IV", "V"],
        "diagonal":   ["Старт", "I", "II", "III", "IV", "V"],
        "correction": ["Старт", "A", "B", "C"],
        "wxy":        ["Старт", "W", "X", "Y"],
        "triangle":   ["Старт", "A", "B", "C", "D", "E"],
        "forming":    ["Старт", "I", "II", "III"],
    },
    "1d": {
        "impulse":    ["Старт", "(1)", "(2)", "(3)", "(4)", "(5)"],
        "diagonal":   ["Старт", "(1)", "(2)", "(3)", "(4)", "(5)"],
        "correction": ["Старт", "(A)", "(B)", "(C)"],
        "wxy":        ["Старт", "(W)", "(X)", "(Y)"],
        "triangle":   ["Старт", "(A)", "(B)", "(C)", "(D)", "(E)"],
        "forming":    ["Старт", "(1)", "(2)", "(3)"],
    },
    "4h": {
        "impulse":    ["Старт", "1", "2", "3", "4", "5"],
        "diagonal":   ["Старт", "1", "2", "3", "4", "5"],
        "correction": ["Старт", "A", "B", "C"],
        "wxy":        ["Старт", "W", "X", "Y"],
        "triangle":   ["Старт", "A", "B", "C", "D", "E"],
        "forming":    ["Старт", "1", "2", "3"],
    },
    "1h": {
        "impulse":    ["Старт", "[i]", "[ii]", "[iii]", "[iv]", "[v]"],
        "diagonal":   ["Старт", "[i]", "[ii]", "[iii]", "[iv]", "[v]"],
        "correction": ["Старт", "[a]", "[b]", "[c]"],
        "wxy":        ["Старт", "[w]", "[x]", "[y]"],
        "triangle":   ["Старт", "[a]", "[b]", "[c]", "[d]", "[e]"],
        "forming":    ["Старт", "[i]", "[ii]", "[iii]"],
    },
    "15m": {
        "impulse":    ["Старт", "(i)", "(ii)", "(iii)", "(iv)", "(v)"],
        "diagonal":   ["Старт", "(i)", "(ii)", "(iii)", "(iv)", "(v)"],
        "correction": ["Старт", "(a)", "(b)", "(c)"],
        "wxy":        ["Старт", "(w)", "(x)", "(y)"],
        "triangle":   ["Старт", "(a)", "(b)", "(c)", "(d)", "(e)"],
        "forming":    ["Старт", "(i)", "(ii)", "(iii)"],
    },
    "5m": {
        "impulse":    ["Старт", "i", "ii", "iii", "iv", "v"],
        "diagonal":   ["Старт", "i", "ii", "iii", "iv", "v"],
        "correction": ["Старт", "a", "b", "c"],
        "wxy":        ["Старт", "w", "x", "y"],
        "triangle":   ["Старт", "a", "b", "c", "d", "e"],
        "forming":    ["Старт", "i", "ii", "iii"],
    },
}


def _get_labels(timeframe: str, pattern_key: str) -> List[str]:
    """Возвращает метки волн с правильной нотацией степени для данного ТФ."""
    tf = timeframe.lower()
    if tf in DEGREE_LABELS and pattern_key in DEGREE_LABELS[tf]:
        return DEGREE_LABELS[tf][pattern_key]
    # Fallback: 4h-стиль
    fallback = DEGREE_LABELS.get("4h", {})
    return fallback.get(pattern_key, ["Старт", "1", "2", "3", "4", "5"])


# ═════════════════════════════════════════════════════════════════════════════
# Dataclasses для результатов
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class WavePoint:
    """Одна точка (экстремум) в волновой структуре."""
    label: str        # напр. "(1)", "A", "[iii]"
    price: float
    timestamp: int    # ms

    def fmt_price_ts(self) -> str:
        return f"{self.price:.2f} [ts:{self.timestamp}]"


@dataclass
class WaveStructure:
    """Одна найденная волновая структура."""
    pattern_type: str          # "Импульс", "Зигзаг", "Плоскость", "Диагональ", etc.
    direction: str             # "БЫЧИЙ" / "МЕДВЕЖИЙ"
    points: List[WavePoint]    # Все экстремумы
    confidence: float = 0.0    # Оценка "качества" паттерна (0-100)
    details: str = ""          # Пояснение почему именно этот паттерн
    channel_target: Optional[float] = None  # Целевая цена по каналу Эллиотта
    fibo_targets: List[float] = field(default_factory=list) # Список целей по расширениям Фибоначчи
    invalidation_price: Optional[float] = None  # Цена инвалидации разметки
    is_completed: bool = False  # True = структура завершена (не текущая)

    def summary(self) -> str:
        segments = []
        for i in range(len(self.points) - 1):
            p_start = self.points[i]
            p_end = self.points[i+1]
            segments.append(f"Волна {p_end.label}: {p_start.fmt_price_ts()} → {p_end.fmt_price_ts()}")
        
        pts_str = " | ".join(segments)
        ch = f" | Канал-цель: {self.channel_target:.2f}" if self.channel_target else ""
        fb = f" | Фибо-цели: {', '.join([f'{t:.2f}' for t in self.fibo_targets])}" if self.fibo_targets else ""
        inv = f" | ⛔Инвалидация: {self.invalidation_price:.2f}" if self.invalidation_price else ""
        status = "[ЗАВЕРШЁН]" if self.is_completed else "[АКТИВНЫЙ]"
        return f"{status} {self.direction} {self.pattern_type} [{self.confidence:.0f}%]: {pts_str}{ch}{fb}{inv}"


def _deduplicate_structures(structures: List[WaveStructure]) -> List[WaveStructure]:
    """Удаляет дубликаты паттернов, использующих одни и те же экстремумы."""
    unique: Dict[tuple, WaveStructure] = {}
    for s in structures:
        # Ключ: кортеж таймстампов всех точек паттерна
        ts_key = tuple(p.timestamp for p in s.points)
        if ts_key not in unique or s.confidence > unique[ts_key].confidence:
            unique[ts_key] = s
    return list(unique.values())


# ═════════════════════════════════════════════════════════════════════════════
# Вспомогательные: преобразование векторов в экстремумы
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Extremum:
    price: float
    timestamp: int
    is_high: bool  # True = локальный максимум


def _vectors_to_extrema(vectors: List[Vector]) -> List[Extremum]:
    """Преобразует последовательность векторов в список чередующихся экстремумов."""
    if not vectors:
        return []
    extrema = [
        Extremum(price=vectors[0].start_price, timestamp=vectors[0].start_time,
                 is_high=not vectors[0].is_bullish)
    ]
    for v in vectors:
        extrema.append(
            Extremum(price=v.end_price, timestamp=v.end_time, is_high=v.is_bullish)
        )
    return extrema


# ═════════════════════════════════════════════════════════════════════════════
# Объёмный анализ: CVD внутри волн
# ═════════════════════════════════════════════════════════════════════════════

def _get_cumulative_volume_for_wave(
    vectors: List[Vector], start_ts: int, end_ts: int
) -> float:
    """
    Суммирует абсолютное изменение цены × объём (proxy CVD) для векторов,
    попадающих во временной диапазон [start_ts, end_ts].
    """
    total = 0.0
    for v in vectors:
        # Вектор пересекается с интервалом волны?
        if v.end_time >= start_ts and v.start_time <= end_ts:
            # Используем atr_size_ratio как proxy размера движения
            size = abs(v.end_price - v.start_price)
            vol_weight = 2.0 if v.volume_anomaly else 1.0
            total += size * vol_weight
    return total


def _volume_validates_impulse(
    pts: List, vectors: List[Vector]
) -> tuple[bool, str]:
    """
    Проверяет объёмное правило: максимальный CVD-объём должен быть в волне 3.
    Если максимум в волне 5 — это скорее C расширенной плоскости.

    Returns:
        (is_valid, description)
    """
    if not vectors:
        return True, "Нет данных объёма"

    # Считаем proxy-CVD для каждой импульсной волны
    wave_volumes = {}
    wave_names = ["W1", "W2", "W3", "W4", "W5"]
    for i in range(5):
        start_ts = pts[i].timestamp
        end_ts = pts[i + 1].timestamp
        vol = _get_cumulative_volume_for_wave(vectors, start_ts, end_ts)
        wave_volumes[wave_names[i]] = vol

    w3_vol = wave_volumes.get("W3", 0)
    w5_vol = wave_volumes.get("W5", 0)
    w1_vol = wave_volumes.get("W1", 0)

    desc_parts = [f"{k}={v:.1f}" for k, v in wave_volumes.items()]
    desc = "Объём по волнам: " + ", ".join(desc_parts)

    # W3 должна иметь максимальный объём среди движущих волн (W1, W3, W5)
    motive_vols = [w1_vol, w3_vol, w5_vol]
    if max(motive_vols) == w3_vol and w3_vol > 0:
        return True, desc + " ✓ Макс. объём в W3"
    elif max(motive_vols) == w5_vol and w5_vol > w3_vol * 1.2:
        return False, desc + " ✗ Макс. объём в W5 (возможно волна C, не импульс)"
    else:
        return True, desc + " ~ Объём не противоречит"


# ═════════════════════════════════════════════════════════════════════════════
# Каналы Эллиотта (Channeling)
# ═════════════════════════════════════════════════════════════════════════════

def _calculate_elliott_channel(
    pts: List, bullish: bool
) -> Optional[float]:
    """
    Канал Эллиотта для прогноза цели волны 5:
    - Трендовая линия проводится через точки 2 и 4
    - Параллельная копия проходит через точку 3
    - Пересечение параллельной с вертикалью "после W4" → цель W5

    Для завершённого импульса вычисляем теоретическую цель:
    slope_24 = (p4 - p2) / (t4 - t2)
    target = p3 + slope_24 * (t5_expected - t3)

    Упрощённый расчёт: канал2-4, параллель через 1 или 3.
    Возвращает цену цели W5 по каналу.
    """
    p0, p1, p2, p3, p4 = pts[:5]

    t2 = p2.timestamp
    t4 = p4.timestamp
    t3 = p3.timestamp

    if t4 == t2:
        return None

    # Наклон линии 2-4
    slope = (p4.price - p2.price) / (t4 - t2)

    # Параллельная линия через точку 3 (или 1)
    # Цель W5: проецируем от W3 по наклону 2-4 на расстояние (t4-t3) вперёд
    # т.е. цель = p3_price + slope * (t_target - t3)
    # где t_target ≈ t4 + (t4 - t2) / 2  (среднее будущее)

    dt_24 = t4 - t2
    t_target = t4 + dt_24  # прогнозируем на одну "секцию" вперёд

    # Канальная цель: точка на параллельной через p1 (базовая линия)
    # p1_projected = p1.price + slope * (t_target - p1.timestamp)
    # Или через p3 (агрессивная цель):
    channel_target = p3.price + slope * (t_target - t3)

    # Альтернативный расчёт: симметричная проекция
    # Точка на линии 0-2 в момент t_target
    if t2 != p0.timestamp:
        slope_02 = (p2.price - p0.price) / (t2 - p0.timestamp)
        channel_alt = p0.price + slope_02 * (t_target - p0.timestamp)
    else:
        channel_alt = None

    # Среднее из двух каналов даёт робастную цель
    if channel_alt is not None:
        final_target = (channel_target + channel_alt) / 2
    else:
        final_target = channel_target

    return round(final_target, 2)


# ═════════════════════════════════════════════════════════════════════════════
# Фрактальная рекурсия: проверка внутренней структуры на младшем ТФ
# ═════════════════════════════════════════════════════════════════════════════

def _fractal_validate_wave(
    wave_start_ts: int,
    wave_end_ts: int,
    sub_tf_vectors: Optional[List[Vector]],
    expected_internal: str = "impulse",
) -> tuple[float, str]:
    """
    Проверяет, есть ли на младшем ТФ ожидаемая внутренняя структура
    внутри временного окна [wave_start_ts, wave_end_ts].

    Для импульсных волн (1, 3, 5) внутри должен быть 5-волновой паттерн.
    Для коррекционных (2, 4) — 3-волновой (ABC).

    Returns:
        (confidence_adjustment, description)
    """
    if not sub_tf_vectors:
        return 0.0, "Нет данных младшего ТФ для фрактальной проверки"

    # Фильтруем векторы младшего ТФ, попадающие в окно
    sub_vectors = [
        v for v in sub_tf_vectors
        if v.start_time >= wave_start_ts and v.end_time <= wave_end_ts
    ]

    if len(sub_vectors) < 3:
        return -5.0, f"Мало данных младшего ТФ внутри волны ({len(sub_vectors)} векторов)"

    sub_extrema = _vectors_to_extrema(sub_vectors)

    if expected_internal == "impulse":
        # Ищем 5-волновой импульс внутри
        has_correct = False
        has_wrong = False
        if len(sub_extrema) >= 6:
            for start_idx in range(len(sub_extrema) - 5):
                sub_pts = sub_extrema[start_idx: start_idx + 6]
                if _check_impulse_core(sub_pts):
                    has_correct = True; break
        
        # Проверяем "не-импульсную" природу (WXY/Zigzag) - КРИТИКА Пункт 5
        if not has_correct and len(sub_extrema) >= 4:
            for start_idx in range(len(sub_extrema) - 3):
                sub_pts = sub_extrema[start_idx: start_idx + 4]
                if _check_zigzag_core(sub_pts) or _check_wxy_core(sub_pts):
                    has_wrong = True; break

        if has_correct:
            return 15.0, "✓ Фрактал: подтвержден внутренний микро-импульс"
        if has_wrong:
            return -20.0, "✗ Фрактал: вместо импульса обнаружена тройка (WXY/Zigzag)"
        return -5.0, f"Фрактал неясен (всего {len(sub_extrema)} точек)"

    elif expected_internal == "correction":
        # Ищем 3-волновую коррекцию внутри
        if len(sub_extrema) >= 4:
            for start_idx in range(len(sub_extrema) - 3):
                sub_pts = sub_extrema[start_idx: start_idx + 4]
                if _check_zigzag_core(sub_pts) or _check_flat_core(sub_pts):
                    return 5.0, "✓ Фрактал: внутри найдена 3-волновая коррекция"
            return -5.0, "✗ Фрактал: внутри НЕ найдена 3-волновая коррекция"
        return 0.0, "Мало данных для проверки коррекционной структуры"

    return 0.0, ""


# ═════════════════════════════════════════════════════════════════════════════
# CORE-проверки (без создания объектов — для фрактальной рекурсии)
# ═════════════════════════════════════════════════════════════════════════════

def _check_impulse_core(pts: List[Extremum]) -> Optional[str]:
    """Быстрая проверка 6 точек на импульс, возвращает 'БЫЧИЙ/МЕДВЕЖИЙ Импульс' или None."""
    p0, p1, p2, p3, p4, p5 = pts
    bullish = p1.price > p0.price

    if bullish:
        if not (p1.price > p0.price and p2.price < p1.price and
                p3.price > p2.price and p4.price < p3.price and
                p5.price > p4.price):
            return None
        if p3.price <= p1.price: return None
    else:
        if not (p1.price < p0.price and p2.price > p1.price and
                p3.price < p2.price and p4.price > p3.price and
                p5.price < p4.price):
            return None
        if p3.price >= p1.price: return None

    len1 = abs(p1.price - p0.price)
    len3 = abs(p3.price - p2.price)
    len5 = abs(p5.price - p4.price)

    # W2 не за начало W1
    if bullish and p2.price <= p0.price: return None
    if not bullish and p2.price >= p0.price: return None
    # W3 не самая короткая
    if len3 <= len1 and len3 <= len5: return None
    # W4 не в зону W1
    if bullish and p4.price <= p1.price: return None
    if not bullish and p4.price >= p1.price: return None
    # W3 пробивает W1
    if bullish and p3.price <= p1.price: return None
    if not bullish and p3.price >= p1.price: return None
    # W5 пробивает W3
    if bullish and p5.price <= p3.price: return None
    if not bullish and p5.price >= p3.price: return None

    return f"{'БЫЧИЙ' if bullish else 'МЕДВЕЖИЙ'} Импульс"


def _check_zigzag_core(pts: List[Extremum]) -> bool:
    """Быстрая проверка 4 точек на зигзаг."""
    p0, pA, pB, pC = pts
    bullish = pA.price > p0.price
    if bullish:
        if not (pA.price > p0.price and pB.price < pA.price and pC.price > pB.price):
            return False
    else:
        if not (pA.price < p0.price and pB.price > pA.price and pC.price < pB.price):
            return False
    lenA = abs(pA.price - p0.price)
    lenB = abs(pB.price - pA.price)
    if lenB >= lenA: return False
    if bullish and pC.price <= pA.price: return False
    if not bullish and pC.price >= pA.price: return False
    return True


def _check_flat_core(pts: List[Extremum]) -> bool:
    """Быстрая проверка 4 точек на плоскость."""
    p0, pA, pB, pC = pts
    bullish = pA.price > p0.price
    if bullish:
        if not (pA.price > p0.price and pB.price < pA.price and pC.price > pB.price):
            return False
    else:
        if not (pA.price < p0.price and pB.price > pA.price and pC.price < pB.price):
            return False
    lenA = abs(pA.price - p0.price)
    lenB = abs(pB.price - pA.price)
    if lenA == 0: return False
    ratioB = lenB / lenA
    return 0.80 <= ratioB <= 1.382


# ═════════════════════════════════════════════════════════════════════════════
# Проверки паттернов (полные, с объектами WaveStructure)
# ═════════════════════════════════════════════════════════════════════════════


def _check_wxy_core(pts: List[Extremum]) -> bool:
    """Быстрая проверка 4 точек на W-X-Y."""
    p0, pW, pX, pY = pts
    bullish = pW.price > p0.price
    lenW = abs(pW.price - p0.price)
    lenX = abs(pX.price - pW.price)
    lenY = abs(pY.price - pX.price)
    if lenW == 0: return False
    if lenX >= lenW: return False
    if bullish and pY.price <= pW.price: return False
    if not bullish and pY.price >= pW.price: return False
    return True

def _check_impulse(
    pts: List[Extremum],
    timeframe: str,
    vectors: Optional[List[Vector]] = None,
    sub_tf_vectors: Optional[List[Vector]] = None,
) -> Optional[WaveStructure]:
    """
    Проверяет 6 точек (0-1-2-3-4-5) на соответствие импульсу.
    Правила Эллиотта:
      1. Чередование направлений
      2. Волна 2 НЕ заходит за начало 1
      3. Волна 3 НЕ самая короткая (среди 1, 3, 5)
      4. Волна 4 НЕ пересекает территорию волны 1
      5. Волна 3 пробивает конец волны 1
      6. ★ Волна 5 пробивает конец волны 3 (НОВОЕ)
    + Объёмная валидация CVD
    + Канал Эллиотта
    + Фрактальная рекурсия
    """
    p0, p1, p2, p3, p4, p5 = pts

    bullish = p1.price > p0.price

    # Чередование
    if bullish:
        if not (p1.price > p0.price and p2.price < p1.price and
                p3.price > p2.price and p4.price < p3.price and
                p5.price > p4.price):
            return None
        if p3.price <= p1.price: return None
    else:
        if not (p1.price < p0.price and p2.price > p1.price and
                p3.price < p2.price and p4.price > p3.price and
                p5.price < p4.price):
            return None
        if p3.price >= p1.price: return None

    len1 = abs(p1.price - p0.price)
    len2 = abs(p2.price - p1.price)
    len3 = abs(p3.price - p2.price)
    len4 = abs(p4.price - p3.price)
    len5 = abs(p5.price - p4.price)

    # Правило 1: Волна 2 не заходит за начало 1
    if bullish and p2.price <= p0.price:
        return None
    if not bullish and p2.price >= p0.price:
        return None

    # Правило 2: Волна 3 не самая короткая
    if len3 <= len1 and len3 <= len5:
        return None

    # Правило 3: Волна 4 не пересекает территорию волны 1
    if bullish and p4.price <= p1.price:
        return None
    if not bullish and p4.price >= p1.price:
        return None

    # Правило 4: Волна 3 пробивает конец волны 1
    if bullish and p3.price <= p1.price:
        return None
    if not bullish and p3.price >= p1.price:
        return None

    # ★ Правило 5 (НОВОЕ): Волна 5 ОБЯЗАНА пробить конец волны 3
    if bullish and p5.price <= p3.price:
        return None
    if not bullish and p5.price >= p3.price:
        return None

    # ── Оценка уверенности ────────────
    confidence = 50.0
    details_parts = []

    # Волна 3 длиннее 1.618 × Волна 1? (+20)
    if len3 >= 1.618 * len1:
        confidence += 20
    elif len3 >= 1.0 * len1:
        confidence += 10

    # Чередование волн 2 и 4 (разная глубина)? (+10)
    ratio2 = len2 / len1 if len1 > 0 else 0
    ratio4 = len4 / len3 if len3 > 0 else 0
    if abs(ratio2 - ratio4) > 0.15:
        confidence += 10

    # Волна 5 = 0.618-1.0 × Волна 1? (+10)
    if len1 > 0 and 0.5 <= len5 / len1 <= 1.382:
        confidence += 10

    if len1:
        details_parts.append(f"W1={len1:.2f}, W3={len3:.2f} ({len3/len1:.2f}×W1)")
    details_parts.append(f"W5={len5:.2f} (W5>W3 ✓)")
    details_parts.append(f"W2 откат={ratio2:.1%}, W4 откат={ratio4:.1%}")

    # ── Объёмная валидация (CVD) ──────
    if vectors:
        vol_ok, vol_desc = _volume_validates_impulse(pts, vectors)
        details_parts.append(vol_desc)
        if vol_ok:
            confidence += 5
        else:
            confidence -= 15  # Сильно снижаем — возможно это C, не 5

    # ── ПРАВИЛА (Rules) - Если нарушены, это не импульс ────
    # 1. Волна 2 не заходит за начало Волны 1
    if bullish and p2.price <= p0.price: return None
    if not bullish and p2.price >= p0.price: return None
    
    # 2. Волна 3 не должна быть самой короткой (среди 1, 3, 5)
    len1, len3, len5 = abs(p1.price-p0.price), abs(p3.price-p2.price), abs(p5.price-p4.price)
    if len3 < len1 and len3 < len5: return None
    
    # 3. Волна 4 не заходит в территорию Волны 1 (для чистого импульса)
    if bullish and p4.price <= p1.price: return None
    if not bullish and p4.price >= p1.price: return None
    
    # 4. Волна 3 пробивает пик Волны 1
    if bullish and p3.price <= p1.price: return None
    if not bullish and p3.price >= p1.price: return None

    # 5. Волна 5 пробивает пик Волны 3 (строгое правило V3)
    if bullish and p5.price <= p3.price: return None
    if not bullish and p5.price >= p3.price: return None

    # ── НОРМЫ (Guidelines / Confidence) ────
    confidence = 40.0 # База за прохождение Правил
    details_parts = []

    # Фибо-пропорции (Норма)
    ratio3_1 = len3 / len1 if len1 > 0 else 0
    if 1.618 <= ratio3_1 <= 2.618: confidence += 20; details_parts.append("W3 расширенная (Норма)")
    elif 1.0 <= ratio3_1 < 1.618: confidence += 10; details_parts.append("W3 стандартная")
    
    ratio2_1 = abs(p2.price-p1.price) / len1 if len1 > 0 else 0
    if 0.5 <= ratio2_1 <= 0.618: confidence += 10; details_parts.append("W2 глубокая коррекция (Норма)")
    
    # Чередование (Alternation) - Волна 2 (острая) vs Волна 4 (боковая)
    # Здесь упрощенно: если W2 глубокая (>0.5), а W4 мелкая (<0.382) -> Бонус
    ratio4_3 = abs(p4.price-p3.price) / len3 if len3 > 0 else 0
    if ratio2_1 > 0.5 and ratio4_3 < 0.4: confidence += 15; details_parts.append("Чередование W2/W4 (Норма)")

    # Канал Эллиотта
    channel_target = _calculate_elliott_channel(pts, bullish)
    if channel_target:
        confidence += 10
        details_parts.append(f"Канал Эллиотта → цель W5 достигнута")

    # Цели после завершения импульса (Коррекция)
    # После 5 волн ожидаем откат к 0.382, 0.5 или 0.618 всей длины (0-5)
    total_len = abs(p5.price - p0.price)
    fibo_targets = []
    for f in [0.382, 0.618, 1.0, 1.618]:
        target = p5.price - (total_len * f) if bullish else p5.price + (total_len * f)
        fibo_targets.append(round(target, 2))

    # Фрактальная рекурсия (W3)
    if sub_tf_vectors:
        frac_adj, frac_desc = _fractal_validate_wave(p2.timestamp, p3.timestamp, sub_tf_vectors, "impulse")
        confidence += frac_adj
        details_parts.append(frac_desc)

    direction = "БЫЧИЙ" if bullish else "МЕДВЕЖИЙ"
    labels = _get_labels(timeframe, "impulse")
    inv_price = p0.price # Для импульса инвалидация - начало W1
    
    wave_points = [
        WavePoint(label=labels[i], price=pts[i].price, timestamp=pts[i].timestamp)
        for i in range(6)
    ]

    return WaveStructure(
        pattern_type="Импульс",
        direction=direction,
        points=wave_points,
        confidence=min(confidence, 100),
        details=", ".join(d for d in details_parts if d),
        channel_target=channel_target,
        fibo_targets=fibo_targets,
        invalidation_price=round(inv_price, 2),
    )


def _check_diagonal(
    pts: List[Extremum],
    timeframe: str,
    vectors: Optional[List[Vector]] = None,
) -> Optional[WaveStructure]:
    """
    Проверяет 6 точек на Начальный/Конечный клин (diagonal).
    Ключевое отличие от импульса: Волна 4 ЗАХОДИТ в территорию волны 1.
    Линии 1-3 и 2-4 сходятся. Длины убывают.
    """
    p0, p1, p2, p3, p4, p5 = pts
    bullish = p1.price > p0.price

    if bullish:
        if not (p1.price > p0.price and p2.price < p1.price and
                p3.price > p2.price and p4.price < p3.price and
                p5.price > p4.price):
            return None
        if p3.price <= p1.price: return None
    else:
        if not (p1.price < p0.price and p2.price > p1.price and
                p3.price < p2.price and p4.price > p3.price and
                p5.price < p4.price):
            return None
        if p3.price >= p1.price: return None

    len1 = abs(p1.price - p0.price)
    len2 = abs(p2.price - p1.price)
    len3 = abs(p3.price - p2.price)
    len4 = abs(p4.price - p3.price)
    len5 = abs(p5.price - p4.price)

    # Волна 4 ДОЛЖНА зайти в зону волны 1 (ключ для diagonal)
    if bullish and p4.price > p1.price:
        return None
    if not bullish and p4.price < p1.price:
        return None

    # --- ПУНКТ 2: Сходящаяся vs Расширяющаяся Диагональ ---
    is_contracting = len1 > len3 > len5 * 0.8
    is_expanding = len1 < len3 < len5 * 1.2
    
    if not (is_contracting or is_expanding):
        return None
    
    if is_expanding:
        # В расширяющейся Волна 4 обычно глубже Волны 2
        if bullish and p4.price > p2.price: return None
        if not bullish and p4.price < p2.price: return None
        subtype = "Расширяющаяся"
    else:
        subtype = "Сходящаяся"

    confidence = 65.0
    details_parts = [f"{subtype} Диагональ: W1={len1:.0f}, W3={len3:.0f}, W5={len5:.0f}. W4 зашла в зону W1."]

    # Объёмная проверка для диагоналей
    if vectors:
        vol_ok, vol_desc = _volume_validates_impulse(pts, vectors)
        details_parts.append(vol_desc)
        if not vol_ok:
            confidence -= 10

    channel_target = _calculate_elliott_channel(pts, bullish)
    if channel_target:
        details_parts.append(f"Канал → {channel_target:.2f}")

    confidence = min(max(confidence, 10), 80)
    direction = "БЫЧИЙ" if bullish else "МЕДВЕЖИЙ"
    labels = _get_labels(timeframe, "diagonal")
    inv_price = p0.price

    wave_points = [
        WavePoint(label=labels[i], price=pts[i].price, timestamp=pts[i].timestamp)
        for i in range(6)
    ]

    return WaveStructure(
        pattern_type="Диагональ",
        direction=direction,
        points=wave_points,
        confidence=confidence,
        details=", ".join(details_parts),
        channel_target=channel_target,
        invalidation_price=round(inv_price, 2),
    )


def _check_zigzag(pts: List[Extremum], timeframe: str) -> Optional[WaveStructure]:
    """
    Проверяет 4 точки (Start-A-B-C) на Зигзаг (5-3-5).
    """
    p0, pA, pB, pC = pts
    bullish = pA.price > p0.price

    # ── ПРАВИЛА (Rules) ────
    # 1. Формально точки должны идти зигзагом
    if bullish:
        if not (pA.price > p0.price and pB.price < pA.price and pC.price > pB.price): return None
    else:
        if not (pA.price < p0.price and pB.price > pA.price and pC.price < pB.price): return None

    lenA, lenB, lenC = abs(pA.price - p0.price), abs(pB.price - pA.price), abs(pC.price - pB.price)

    # 2. Волна B не должна заходить за начало Волны A
    if lenB >= lenA: return None
    
    # 3. Волна C должна пробить пик Волны A (иначе это может быть треугольник)
    if bullish and pC.price <= pA.price: return None
    if not bullish and pC.price >= pA.price: return None

    # ── НОРМЫ (Guidelines / Confidence) ────
    confidence = 50.0
    details_parts = []
    
    ratioB = lenB / lenA if lenA > 0 else 0
    ratioC = lenC / lenA if lenA > 0 else 0

    # Норма для B: откат 38.2% - 61.8%
    if 0.382 <= ratioB <= 0.618: confidence += 15; details_parts.append("W-B нормальный откат")
    elif ratioB > 0.8: confidence -= 10; details_parts.append("W-B слишком глубокая")

    # Норма для C: равенство с A или расширение 1.618
    target_hit = False
    for target in [0.618, 1.0, 1.272, 1.618]:
        if abs(ratioC - target) < 0.1:
            confidence += 15
            details_parts.append(f"W-C достигла Фибо-{target}")
            target_hit = True
            break
    if not target_hit: confidence -= 5

    # Цели после завершения зигзага 
    # Если это коррекция, ожидаем возврат к началу импульса или 1.618 предыдущего движения.
    # Здесь упрощенно: цели на разворот (0.618 и 1.0 от всего ABC)
    total_len = abs(pC.price - p0.price)
    fibo_targets = []
    # Если зигзаг бычий (вниз-вверх-вниз?), мы ждем разворот в SHORT? 
    # Нет, bullish в коде значит pA > p0 (движение вверх). Т.е. зигзаг на РОСТ. 
    # Значит после C (пик) ждем откат.
    for f in [0.618, 1.0]:
        target = pC.price - (total_len * f) if bullish else pC.price + (total_len * f)
        fibo_targets.append(round(target, 2))

    direction = "БЫЧИЙ" if bullish else "МЕДВЕЖИЙ"
    labels = _get_labels(timeframe, "correction")
    inv_price = p0.price # Начало A

    wave_points = [
        WavePoint(label=labels[i], price=pts[i].price, timestamp=pts[i].timestamp)
        for i in range(4)
    ]

    return WaveStructure(
        pattern_type="Зигзаг",
        direction=direction,
        points=wave_points,
        confidence=min(confidence, 95),
        details=f"A={lenA:.2f}, B={ratioB:.1%} от A, C={ratioC:.1%} от A. " + ", ".join(details_parts),
        fibo_targets=fibo_targets,
        invalidation_price=round(inv_price, 2),
    )


def _check_flat(pts: List[Extremum], timeframe: str) -> Optional[WaveStructure]:
    """
    Проверяет 4 точки на Плоскость (Flat 3-3-5).
    """
    p0, pA, pB, pC = pts
    bullish = pA.price > p0.price

    if bullish:
        if not (pA.price > p0.price and pB.price < pA.price and pC.price > pB.price):
            return None
    else:
        if not (pA.price < p0.price and pB.price > pA.price and pC.price < pB.price):
            return None

    lenA = abs(pA.price - p0.price)
    lenB = abs(pB.price - pA.price)
    lenC = abs(pC.price - pB.price)

    if lenA == 0:
        return None

    ratioB = lenB / lenA
    # КРИТИКА: Плоскость обязана иметь откат B >= 90% от A
    if ratioB < 0.90 or ratioB > 1.382:
        return None
    if lenB > 0 and lenC < 0.618 * lenB:
        return None

    confidence = 50.0
    if 0.9 <= ratioB <= 1.1:
        confidence += 15
    if ratioB > 1.0:
        confidence += 5

    confidence = min(confidence, 90)
    direction = "БЫЧИЙ" if bullish else "МЕДВЕЖИЙ"
    subtype = "Расширенная" if ratioB > 1.05 else "Обычная"
    labels = _get_labels(timeframe, "correction")
    inv_price = p0.price if ratioB <= 1.0 else pB.price

    # Цели: возврат к началу импульса
    fibo_targets = [round(p0.price, 2)]

    wave_points = [
        WavePoint(label=labels[i], price=pts[i].price, timestamp=pts[i].timestamp)
        for i in range(4)
    ]

    return WaveStructure(
        pattern_type=f"Плоскость ({subtype})",
        direction=direction,
        points=wave_points,
        confidence=confidence,
        details=f"B={ratioB:.1%} от A, C={lenC:.2f}",
        fibo_targets=fibo_targets,
        invalidation_price=round(inv_price, 2),
    )


def _check_wxy(pts: List[Extremum], timeframe: str) -> Optional[WaveStructure]:
    """
    Проверяет 4 точки (Start-W-X-Y) на Двойной Зигзаг (W-X-Y).
    """
    p0, pW, pX, pY = pts
    bullish = pW.price > p0.price

    if bullish:
        if not (pW.price > p0.price and pX.price < pW.price and pY.price > pX.price):
            return None
    else:
        if not (pW.price < p0.price and pX.price > pW.price and pY.price < pX.price):
            return None

    lenW = abs(pW.price - p0.price)
    lenX = abs(pX.price - pW.price)
    lenY = abs(pY.price - pX.price)

    if lenX >= lenW:
        return None
    if bullish and pY.price <= pW.price:
        return None
    if not bullish and pY.price >= pW.price:
        return None

    confidence = 45.0
    ratioX = lenX / lenW if lenW > 0 else 0
    ratioY = lenY / lenW if lenW > 0 else 0

    if 0.5 <= ratioX <= 0.9:
        confidence += 10
    if 0.618 <= ratioY <= 1.618:
        confidence += 15
    if 0.9 <= ratioY <= 1.1:
        confidence += 10

    confidence = min(confidence, 80)
    # Цели после WXY (аналогично зигзагу)
    total_len = abs(pY.price - p0.price)
    fibo_targets = []
    for f in [0.618, 1.0]:
        target = pY.price - (total_len * f) if bullish else pY.price + (total_len * f)
        fibo_targets.append(round(target, 2))

    direction = "БЫЧИЙ" if bullish else "МЕДВЕЖИЙ"
    labels = _get_labels(timeframe, "wxy")
    inv_price = p0.price

    wave_points = [
        WavePoint(label=labels[i], price=pts[i].price, timestamp=pts[i].timestamp)
        for i in range(4)
    ]

    return WaveStructure(
        pattern_type="Двойной Зигзаг (W-X-Y)",
        direction=direction,
        points=wave_points,
        confidence=confidence,
        details=f"X={ratioX:.1%} от W, Y={ratioY:.1%} от W ({lenY:.2f})",
        fibo_targets=fibo_targets,
        invalidation_price=round(inv_price, 2),
    )


def _check_triangle(pts: List[Extremum], timeframe: str) -> Optional[WaveStructure]:
    """
    Проверяет 6 точек (Start-A-B-C-D-E) на Сходящийся Треугольник.
    Правила:
      1. 5 субволн (A-B-C-D-E) чередуются по направлению
      2. Каждая последующая волна КОРОЧЕ предыдущей (сужение)
      3. Линии A-C и B-D сходятся (верхняя и нижняя границы)
      4. В итоге цена "сжимается" → прорыв в направлении предшествующего тренда
    """
    p0, pA, pB, pC, pD, pE = pts

    # Определяем направление: первая субволна (A) задаёт "верхнюю" или "нижнюю" границу
    a_bull = pA.price > p0.price

    # Чередование: A, C, E в одну сторону; B, D — в другую
    if a_bull:
        if not (pA.price > p0.price and pB.price < pA.price and
                pC.price > pB.price and pD.price < pC.price and
                pE.price > pD.price):
            return None
    else:
        if not (pA.price < p0.price and pB.price > pA.price and
                pC.price < pB.price and pD.price > pC.price and
                pE.price < pD.price):
            return None

    lenA = abs(pA.price - p0.price)
    lenB = abs(pB.price - pA.price)
    lenC = abs(pC.price - pB.price)
    lenD = abs(pD.price - pC.price)
    lenE = abs(pE.price - pD.price)

    # Сужение: каждая волна должна быть меньше (или примерно равна) предыдущей одноимённой
    # A > C > E (нечётные) и B > D (чётные)
    if not (lenA > lenC * 0.75 and lenC > lenE * 0.65):
        return None
    if not (lenB > lenD * 0.75):
        return None

    # Сходимость линий: A-C должны сужаться, B-D тоже
    # Для бычьего: пики убывают (pA > pC > pE), впадины растут (p0 < pB < pD)
    # Для медвежьего: зеркально
    # Сходимость линий: пики должны снижаться, впадины расти (контракт)
    if a_bull:
        # Хаи снижаются (A > C > E), Лои растут (0 < B < D)
        if not (pA.price > pC.price > pE.price): return None
        if not (p0.price < pB.price < pD.price): return None
    else:
        # Лои растут (A < C < E), Хаи снижаются (0 > B > D)
        if not (pA.price < pC.price < pE.price): return None
        if not (p0.price > pB.price > pD.price): return None

    confidence = 55.0

    # Чем сильнее сужение, тем выше уверенность
    compression_ratio = lenE / lenA if lenA > 0 else 1
    if compression_ratio < 0.5:
        confidence += 15
    elif compression_ratio < 0.7:
        confidence += 10

    # Есть ли чёткое чередование глубин?
    if abs(lenA - lenC) > 0.05 * lenA and abs(lenB - lenD) > 0.05 * lenB:
        confidence += 5

    confidence = min(confidence, 85)

    # Прогноз прорыва: обычно в направлении предшествующего тренда
    # Целевой уровень = начало треугольника ± ширина A
    if a_bull:
        breakout_target = pE.price + lenA  # Бычий прорыв вверх
        direction = "БЫЧИЙ"
    else:
        breakout_target = pE.price - lenA  # Медвежий прорыв вниз
        direction = "МЕДВЕЖИЙ"

    labels = _get_labels(timeframe, "triangle")
    inv_price = p0.price
    wave_points = [
        WavePoint(label=labels[i], price=pts[i].price, timestamp=pts[i].timestamp)
        for i in range(6)
    ]

    return WaveStructure(
        pattern_type="Треугольник",
        direction=direction,
        points=wave_points,
        confidence=confidence,
        details=(
            f"Сужение: A={lenA:.2f}, C={lenC:.2f}, E={lenE:.2f}. "
            f"B={lenB:.2f}, D={lenD:.2f}. Компрессия={compression_ratio:.1%}. "
            f"Цель прорыва: {breakout_target:.2f}"
        ),
        channel_target=round(breakout_target, 2),
        invalidation_price=round(inv_price, 2),
    )


def _check_forming_123(
    pts: List[Extremum],
    timeframe: str,
    sub_tf_vectors: Optional[List[Vector]] = None,
) -> Optional[WaveStructure]:
    """
    Проверяет последние 4 точки (0-1-2-3) на формирующийся импульс.
    + Фрактальная рекурсия: если W1 содержит 5 волн внутри → подтверждение.
    """
    p0, p1, p2, p3 = pts
    bullish = p1.price > p0.price

    if bullish:
        if not (p1.price > p0.price and p2.price < p1.price and p3.price > p2.price):
            return None
    else:
        if not (p1.price < p0.price and p2.price > p1.price and p3.price < p2.price):
            return None

    len1 = abs(p1.price - p0.price)
    len2 = abs(p2.price - p1.price)
    len3 = abs(p3.price - p2.price)

    if bullish and p2.price <= p0.price:
        return None
    if not bullish and p2.price >= p0.price:
        return None

    if bullish and p3.price <= p1.price:
        return None
    if not bullish and p3.price >= p1.price:
        return None

    if len3 < 0.8 * len1:
        return None

    confidence = 45.0
    details_parts = []

    if len3 >= 1.618 * len1:
        confidence += 15
    elif len3 >= 1.0 * len1:
        confidence += 8

    details_parts.append(f"Сформированы W1-W2-W3. Ожидается коррекция W4. W3={len3/len1:.2f}×W1.")

    # Фрактальная проверка W1
    if sub_tf_vectors:
        frac_adj, frac_desc = _fractal_validate_wave(
            p0.timestamp, p1.timestamp, sub_tf_vectors, "impulse"
        )
        confidence += frac_adj
        details_parts.append(frac_desc)

    confidence = min(max(confidence, 10), 80)
    direction = "БЫЧИЙ" if bullish else "МЕДВЕЖИЙ"
    labels = _get_labels(timeframe, "forming")
    # Инвалидация: начало формирования (точка 0)
    inv_price = p0.price

    # Цели для продолжения тренда (W5)
    fibo_targets = []
    # Волна 4 (откат W3): 0.382
    target4 = p3.price - (len3 * 0.382) if bullish else p3.price + (len3 * 0.382)
    fibo_targets.append(round(target4, 2))
    # Волна 5: W1 + W3 (приблизительно)
    target5 = p3.price + (len1 * 0.618) if bullish else p3.price - (len1 * 0.618)
    fibo_targets.append(round(target5, 2))

    wave_points = [
        WavePoint(label=labels[i], price=pts[i].price, timestamp=pts[i].timestamp)
        for i in range(4)
    ]

    return WaveStructure(
        pattern_type="Forming_123",
        direction=direction,
        points=wave_points,
        confidence=confidence,
        details=" ".join(details_parts),
        fibo_targets=fibo_targets,
        invalidation_price=round(inv_price, 2),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Главная функция
# ═════════════════════════════════════════════════════════════════════════════

def analyze_wave_structure(
    vectors: List[Vector],
    timeframe: str = "",
    sub_tf_vectors: Optional[List[Vector]] = None,
) -> str:
    """
    Полный математический анализ волн для одного таймфрейма.

    Args:
        vectors:        Векторы текущего ТФ.
        timeframe:      Строка ТФ (напр. "1d").
        sub_tf_vectors: Векторы младшего ТФ для фрактальной валидации.
                        Например, для 1D передаём векторы 4H.

    Returns:
        Детальный многострочный отчёт для LLM.
    """
    if not vectors or len(vectors) < 3:
        return (
            f"[{timeframe}] Недостаточно векторов ({len(vectors) if vectors else 0}). "
            "Минимум 3 для определения структуры."
        )

    all_structures: List[WaveStructure] = []
    extrema = _vectors_to_extrema(vectors)

    if len(extrema) < 4:
        return (
            f"[{timeframe}] Недостаточно экстремумов ({len(extrema)}). "
            "Минимум 4 для идентификации паттерна."
        )

    # ── Relevance Window ─────────────────────────────────────────────────
    # Для 6-точечных паттернов: последние 15 экстремумов
    # Для 4-точечных: последние 10 экстремумов
    # Старые данные остаются в контексте, но не перегружают поиск
    WINDOW_6PT = 15
    WINDOW_4PT = 10
    recent_6 = extrema[-WINDOW_6PT:] if len(extrema) > WINDOW_6PT else extrema
    recent_4 = extrema[-WINDOW_4PT:] if len(extrema) > WINDOW_4PT else extrema

    # ── Поиск Импульсов (6 точек) ────────────────────────────────────────
    if len(recent_6) >= 6:
        for start_idx in range(len(recent_6) - 5):
            pts = recent_6[start_idx: start_idx + 6]
            result = _check_impulse(pts, timeframe, vectors, sub_tf_vectors)
            if result:
                all_structures.append(result)

    # ── Поиск Диагоналей (6 точек) ───────────────────────────────────────
    if len(recent_6) >= 6:
        for start_idx in range(len(recent_6) - 5):
            pts = recent_6[start_idx: start_idx + 6]
            result = _check_diagonal(pts, timeframe, vectors)
            if result:
                all_structures.append(result)

    # ── Поиск Треугольников (6 точек) ────────────────────────────────────
    if len(recent_6) >= 6:
        for start_idx in range(len(recent_6) - 5):
            pts = recent_6[start_idx: start_idx + 6]
            result = _check_triangle(pts, timeframe)
            if result:
                all_structures.append(result)

    # ── Поиск Зигзагов (4 точки) ────────────────────────────────────────
    if len(recent_4) >= 4:
        for start_idx in range(len(recent_4) - 3):
            pts = recent_4[start_idx: start_idx + 4]
            result = _check_zigzag(pts, timeframe)
            if result:
                all_structures.append(result)

    # ── Поиск Плоскостей (4 точки) ───────────────────────────────────────
    if len(recent_4) >= 4:
        for start_idx in range(len(recent_4) - 3):
            pts = recent_4[start_idx: start_idx + 4]
            result = _check_flat(pts, timeframe)
            if result:
                all_structures.append(result)

    # ── Поиск Двойных Зигзагов (W-X-Y) ──────────────────────────────────
    if len(recent_4) >= 4:
        for start_idx in range(len(recent_4) - 3):
            pts = recent_4[start_idx: start_idx + 4]
            result = _check_wxy(pts, timeframe)
            if result:
                all_structures.append(result)

    # ── Поиск формирующегося Импульса 1-2-3 ──────────────────────────────
    if len(extrema) >= 4:
        pts = extrema[-4:]
        result = _check_forming_123(pts, timeframe, sub_tf_vectors)
        if result:
            all_structures.append(result)

    # ── Дедупликация ──────────────────────────────────────────────────────
    all_structures = _deduplicate_structures(all_structures)

    # ── Маркировка: Completed vs Active ──────────────────────────────────
    # Активная = последняя точка входит в последние 3 экстремума
    last_3_ts = {e.timestamp for e in extrema[-3:]}
    for s in all_structures:
        s.is_completed = s.points[-1].timestamp not in last_3_ts

    # ── Формируем отчёт ──────────────────────────────────────────────────
    if not all_structures:
        last_vecs = vectors[-5:] if len(vectors) >= 5 else vectors
        moves = []
        for v in last_vecs:
            d = "↑" if v.is_bullish else "↓"
            moves.append(f"{d}{v.start_price:.2f}→{v.end_price:.2f}")
        return (
            f"[{timeframe}] Строгих паттернов не выявлено. "
            f"Последние {len(last_vecs)} движений: {', '.join(moves)}. "
            "Рынок в сложной фазе (без чёткой волновой структуры)."
        )

    # Разделяем на активные и завершённые
    active = [s for s in all_structures if not s.is_completed]
    completed = [s for s in all_structures if s.is_completed]

    # Сортируем каждую группу по уверенности
    active.sort(key=lambda s: s.confidence, reverse=True)
    completed.sort(key=lambda s: s.confidence, reverse=True)

    lines = [f"[{timeframe}] МАТЕМАТИЧЕСКИ ОПРЕДЕЛЁННЫЕ СТРУКТУРЫ:"]

    # ── АКТИВНЫЕ (текущие, торгуемые) ─────────────────────────────
    if active:
        best = active[0]
        lines.append(f"  ★ ГЛАВНЫЙ ПАТТЕРН (АКТИВНЫЙ): {best.summary()}")
        if best.details:
            lines.append(f"    Детали: {best.details}")
        if best.invalidation_price:
            lines.append(f"    ⛔ Инвалидация разметки: {best.invalidation_price:.2f}")

        # Формирующийся паттерн
        forming = next((s for s in active if s.pattern_type == "Forming_123" and s is not best), None)
        if forming:
            lines.append(f"  ◆ ФОРМИРУЮЩИЙСЯ: {forming.summary()}")
            if forming.invalidation_price:
                lines.append(f"    ⛔ Инвалидация: {forming.invalidation_price:.2f}")

        # Активные альтернативы
        alt_active = [s for s in active if s is not best and s is not forming][:2]
        for alt in alt_active:
            lines.append(f"  ○ Альтернатива (АКТИВНАЯ): {alt.summary()}")
    else:
        # Нет активных — берём лучший из завершённых
        if completed:
            lines.append(f"  ★ ГЛАВНЫЙ (последний завершённый): {completed[0].summary()}")

    # ── ЗАВЕРШЁННЫЕ (контекст для старших ТФ) ─────────────────────
    if completed:
        shown = completed[:2]
        lines.append(f"  ◇ ЗАВЕРШЁННЫЕ СТРУКТУРЫ ({len(completed)} всего, контекст):")
        for c in shown:
            lines.append(f"    - {c.summary()}")

    return "\n".join(lines)

