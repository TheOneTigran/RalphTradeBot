"""
math_preprocessor.py — Вычислительное ядро RalphTradeBot V2.

Пайплайн обработки данных для одного таймфрейма:
  1. Расчёт ATR, RSI, AO, объёмного SMA
  2. Обнаружение значимых экстремумов (ATR-Fractal фильтрация)
  3. Построение хронологических ценовых векторов с ATR-размером
  4. Расчёт уровней Фибоначчи от 3-х последних значимых свингов
  5. Поиск кластеров Фибоначчи (пересечения ± FIB_CLUSTER_TOLERANCE)
  6. Определение дивергенций RSI и AO на экстремумах
  7. Сборка объектов TimeframeData и LLMContext
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

from src.core.models import FibLevel, FibCluster, LLMContext, TimeframeData, Vector
from src.core.config import (
    VOLUME_ANOMALY_MULTIPLIER,
    VOLUME_ANOMALY_PERIOD,
    ATR_FRACTAL_SETTINGS,
    FIB_CLUSTER_TOLERANCE,
)
from src.core.exceptions import PreprocessingError
from src.math_engine.indicators import (
    atr,
    rsi,
    sma,
    volume_anomaly_mask,
    detect_rsi_divergence,
    awesome_oscillator,
    calculate_vpvr_poc,
    calculate_cvd,
)
from src.math_engine.wave_analyzer import analyze_wave_structure

logger = logging.getLogger(__name__)

# ─── Константы Фибоначчи ─────────────────────────────────────────────────────

# Коррекционные уровни (откат от предыдущего хода)
FIB_RETRACEMENT_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
# Расширения (продолжение за конец волны)
FIB_EXTENSION_RATIOS   = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
# Все коэффициенты для полного набора уровней
FIB_ALL_RATIOS = sorted(set(FIB_RETRACEMENT_RATIOS + FIB_EXTENSION_RATIOS))

# Допуск: попадание цены в уровень Фиб ± 4%
FIB_TOLERANCE = 0.04


# ─── АТR-Fractal: интеллектуальный поиск экстремумов ─────────────────────────

def _atr_fractal_pivots(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    fractal_n: int,
    atr_vals: list[float],
    atr_mult: float,
) -> tuple[list[int], list[int]]:
    """
    Находит значимые свинги методом Williams Fractals + ATR-фильтрации.

    Шаг 1: Фрактал Вильямса — экстремум отделён n барами с каждой стороны.
    Шаг 2: ATR-фильтр — вектор между двумя соседними экстремумами должен быть
            >= atr_mult × ATR. Это отсеивает мелкие коррекции и алго-шум.

    Args:
        highs, lows, closes: Ценовые ряды.
        fractal_n:           Половина окна фрактала (2 → окно 5 баров).
        atr_vals:            Массив ATR той же длины.
        atr_mult:            Минимальный размер вектора в ATR-единицах.

    Returns:
        (pivot_high_indices, pivot_low_indices)
    """
    ph_raw, pl_raw = [], []
    length = len(highs)

    # Шаг 1: чистые фракталы Вильямса
    for i in range(fractal_n, length - fractal_n):
        window_h = highs[i - fractal_n : i + fractal_n + 1]
        window_l = lows[i - fractal_n : i + fractal_n + 1]
        if highs[i] == max(window_h):
            ph_raw.append(i)
        if lows[i] == min(window_l):
            pl_raw.append(i)

    # Шаг 2: ATR-фильтрация пар экстремумов
    # Объединяем и сортируем хронологически, затем убираем слишком мелкие ходы
    all_pivots_raw = (
        [(i, highs[i], True) for i in ph_raw] +
        [(i, lows[i], False) for i in pl_raw]
    )
    all_pivots_raw.sort(key=lambda x: x[0])

    # Чередуем вершины и основания (удаляем подряд идущие одного типа и с ОДИНАКОВЫМ индексом)
    merged: list[tuple[int, float, bool]] = []
    for idx, price, is_high in all_pivots_raw:
        if merged:
            prev_idx, prev_price, prev_is_high = merged[-1]
            
            # 1. Если тот же индекс (одна свеча и хай, и лой) - берем более экстремальный
            if idx == prev_idx:
                if (is_high and price > prev_price) or (not is_high and price < prev_price):
                    merged[-1] = (idx, price, is_high)
                continue
            
            # 2. Если тот же тип (два хая или два лоя подряд) - берем более экстремальный
            if is_high == prev_is_high:
                if (is_high and price > prev_price) or (not is_high and price < prev_price):
                    merged[-1] = (idx, price, is_high)
                continue
                
        merged.append((idx, price, is_high))

    # ATR-фильтр: вектор между соседними должен быть >= atr_mult × ATR
    filtered: list[tuple[int, float, bool]] = []
    for i, (idx, price, is_high) in enumerate(merged):
        if i == 0:
            filtered.append((idx, price, is_high))
            continue
        prev_idx, prev_price, _ = filtered[-1]
        local_atr = atr_vals[idx] if not math.isnan(atr_vals[idx]) else 0
        swing_size = abs(price - prev_price)

        if local_atr > 0 and swing_size >= atr_mult * local_atr:
            filtered.append((idx, price, is_high))
        else:
            # Вектор слишком мелкий: оставляем только самый экстремальный из пары
            if (is_high and price > filtered[-1][1]) or (not is_high and price < filtered[-1][1]):
                filtered[-1] = (idx, price, is_high)

    ph_filtered = [idx for idx, _, is_high in filtered if is_high]
    pl_filtered = [idx for idx, _, is_high in filtered if not is_high]
    return ph_filtered, pl_filtered


def _merge_pivots(
    pivot_highs: list[int],
    pivot_lows: list[int],
    timestamps: list[int],
    highs: list[float],
    lows: list[float],
) -> list[dict]:
    """
    Объединяет свинговые максимумы и минимумы в единый хронологический список.
    Возвращает: [{'idx': int, 'ts': int, 'price': float, 'is_high': bool}]
    """
    pivots = []
    for i in pivot_highs:
        pivots.append({"idx": i, "ts": timestamps[i], "price": highs[i], "is_high": True})
    for i in pivot_lows:
        pivots.append({"idx": i, "ts": timestamps[i], "price": lows[i], "is_high": False})

    pivots.sort(key=lambda x: x["ts"])

    # Финальная зачистка чередования (на случай если ATR-фильтр пропустил дубли)
    cleaned: list[dict] = []
    for p in pivots:
        if cleaned and cleaned[-1]["is_high"] == p["is_high"]:
            if p["is_high"]:
                if p["price"] > cleaned[-1]["price"]:
                    cleaned[-1] = p
            else:
                if p["price"] < cleaned[-1]["price"]:
                    cleaned[-1] = p
        else:
            cleaned.append(p)

    return cleaned


# ─── Расчёт уровней Фибоначчи ────────────────────────────────────────────────

def calc_fib_levels(wave_start: float, wave_end: float) -> list[FibLevel]:
    """
    Рассчитывает уровни Фибоначчи (коррекции + расширения) для заданного хода.

    Args:
        wave_start: Начало значимого движения.
        wave_end:   Конец значимого движения.

    Returns:
        Список объектов FibLevel.
    """
    delta = wave_end - wave_start
    levels = []
    for ratio in FIB_ALL_RATIOS:
        price = wave_end - delta * ratio
        label = f"{ratio * 100:.1f}% Fib от {wave_start:.2f}→{wave_end:.2f}"
        levels.append(FibLevel(ratio=ratio, price=round(price, 6), label=label))
    return levels


def _nearest_fib_ratio(correction_pct: float) -> Optional[float]:
    """Возвращает ближайший Фибо-коэффициент с допуском FIB_TOLERANCE."""
    best = None
    best_diff = float("inf")
    for ratio in FIB_ALL_RATIOS:
        diff = abs(correction_pct - ratio)
        if diff < best_diff:
            best_diff = diff
            best = ratio
    return best if best_diff <= FIB_TOLERANCE else None


# ─── Поиск кластеров Фибоначчи ───────────────────────────────────────────────

def find_fib_clusters(
    pivots: list[dict],
    current_price: float,
    tolerance: float = FIB_CLUSTER_TOLERANCE,
    top_n_swings: int = 5,
) -> list[FibCluster]:
    """
    Находит кластеры — зоны, где уровни Фибоначчи от разных волновых свингов совпадают.

    Алгоритм:
      1. Берём последние top_n_swings пар экстремумов.
      2. Для каждой пары рассчитываем коррекционные И расширенные уровни.
      3. Группируем все уровни по близости (± tolerance).
      4. Группы с ≥2 уровнями — это кластеры.

    Args:
        pivots:        Список экстремумов в хронологическом порядке.
        current_price: Текущая цена для определения поддержки или сопротивления.
        tolerance:     Допуск совпадения цен (по умолчанию 0.5%).
        top_n_swings:  Сколько последних свингов анализировать.

    Returns:
        Список FibCluster, отсортированный по силе (убывание).
    """
    if len(pivots) < 2:
        return []

    # Берём последние N пар
    recent_pivots = pivots[-(top_n_swings + 1):]

    # Собираем все уровни (price, label)
    all_levels: list[tuple[float, str]] = []
    for i in range(1, len(recent_pivots)):
        p0 = recent_pivots[i - 1]
        p1 = recent_pivots[i]
        wave_type = "ret" if (p1["is_high"] != p0["is_high"]) else "ext"
        ratios = FIB_RETRACEMENT_RATIOS if wave_type == "ret" else FIB_EXTENSION_RATIOS

        delta = p1["price"] - p0["price"]
        for ratio in ratios:
            price = p1["price"] - delta * ratio
            label = (
                f"[{i}/{len(recent_pivots)-1}] {ratio*100:.1f}% "
                f"{'ретрейс' if wave_type == 'ret' else 'экст'} "
                f"{p0['price']:.2f}→{p1['price']:.2f}"
            )
            all_levels.append((round(price, 8), label))

    if not all_levels:
        return []

    # Кластеризация: жадный алгоритм
    all_levels.sort(key=lambda x: x[0])
    clusters: list[FibCluster] = []
    used = [False] * len(all_levels)

    for i, (price_i, label_i) in enumerate(all_levels):
        if used[i]:
            continue
        group_prices = [price_i]
        group_labels = [label_i]
        used[i] = True

        for j, (price_j, label_j) in enumerate(all_levels):
            if used[j] or i == j:
                continue
            if abs(price_j - price_i) / max(abs(price_i), 1e-9) <= tolerance:
                group_prices.append(price_j)
                group_labels.append(label_j)
                used[j] = True

        if len(group_prices) >= 2:
            center = round(sum(group_prices) / len(group_prices), 6)
            clusters.append(FibCluster(
                price=center,
                strength=len(group_prices),
                levels=group_labels,
                is_support=center < current_price,
                has_poc=False  # Будет обновлено позже при передаче векторов
            ))

    # Сортируем по силе (количество совпавших уровней)
    clusters.sort(key=lambda c: c.strength, reverse=True)
    return clusters


# ─── Построение векторов ─────────────────────────────────────────────────────

def _build_vectors(
    pivots: list[dict],
    candles: list[dict],
    rsi_series: list[float],
    ao_series: list[float],
    vol_anomaly: list[bool],
    rsi_divergence: list[bool],
    atr_vals: list[float],
    cvd_series: list[float],
) -> list[Vector]:
    """
    Строит список ценовых векторов из экстремумов.

    Для каждого вектора вычисляет:
    - Процентное изменение цены
    - Глубину Фибоначчи к предыдущему вектору
    - RSI и AO на конечной свече
    - Флаги дивергенций и аномалий объёма
    - Размер вектора в ATR-единицах

    Args:
        pivots:        Хронологически упорядоченные экстремумы.
        rsi_series:    Массив RSI.
        ao_series:     Массив Awesome Oscillator.
        vol_anomaly:   Маска аномального объёма.
        rsi_divergence: Маска дивергенции RSI.
        atr_vals:      Массив ATR.

    Returns:
        Список Vector.
    """
    vectors: list[Vector] = []

    for i in range(1, len(pivots)):
        prev = pivots[i - 1]
        curr = pivots[i]

        start_p = prev["price"]
        end_p = curr["price"]
        pct = (end_p - start_p) / start_p * 100

        # Коррекция Фибоначчи к предыдущему вектору
        fib_ratio: Optional[float] = None
        if i >= 2:
            prev_prev = pivots[i - 2]
            prev_move = abs(prev["price"] - prev_prev["price"])
            curr_move = abs(end_p - start_p)
            if prev_move and curr_move:
                retracement = curr_move / prev_move
                fib_ratio = _nearest_fib_ratio(retracement)

        end_idx = curr["idx"]
        start_idx = prev["idx"]
        
        # VPVR (Point of Control) для данного вектора
        poc = calculate_vpvr_poc(candles, start_idx, min(end_idx + 1, len(candles)))

        # ─── Liquidity Sweep & CVD Divergence ──────────────────────────────────
        is_sweep = False
        cvd_div = None
        
        if i >= 2:
            prev_prev = pivots[i - 2]
            prev_prev_idx = prev_prev["idx"]
            candle_close = candles[end_idx]["close"]
            
            if curr["is_high"]:
                # Бычий вектор (ищем медвежий свип или дивергенцию)
                if end_p > prev_prev["price"]:
                    # Цена обновила максимум. А где закрытие?
                    if candle_close < prev_prev["price"]:
                        is_sweep = True  # Прокололи хай ликвидностями и вернулись
                    # Проверяем CVD
                    if cvd_series[end_idx] < cvd_series[prev_prev_idx]:
                        cvd_div = "Bearish CVD Div"
            else:
                # Медвежий вектор (ищем бычий свип или дивергенцию)
                if end_p < prev_prev["price"]:
                    # Цена обновила минимум.
                    if candle_close > prev_prev["price"]:
                        is_sweep = True  # Прокололи лой
                    if cvd_series[end_idx] > cvd_series[prev_prev_idx]:
                        cvd_div = "Bullish CVD Div"

        # RSI
        rsi_val = rsi_series[end_idx] if end_idx < len(rsi_series) else float("nan")
        if math.isnan(rsi_val):
            rsi_val = 50.0

        # AO дивергенция: AO на конечной свече противостоит направлению вектора
        ao_val = ao_series[end_idx] if end_idx < len(ao_series) else float("nan")
        ao_prev_val = ao_series[end_idx - 1] if end_idx > 0 else float("nan")
        ao_div = False
        if not math.isnan(ao_val) and not math.isnan(ao_prev_val):
            # Бычья: цена на новом минимуме, AO выше предыдущего minima
            if not curr["is_high"] and ao_val > ao_prev_val:
                ao_div = True
            # Медвежья: цена на новом максимуме, AO ниже предыдущего maxima
            elif curr["is_high"] and ao_val < ao_prev_val:
                ao_div = True

        # ATR-размер вектора
        local_atr = atr_vals[end_idx] if end_idx < len(atr_vals) and not math.isnan(atr_vals[end_idx]) else None
        atr_ratio = round(abs(end_p - start_p) / local_atr, 2) if local_atr and local_atr > 0 else None

        vectors.append(
            Vector(
                start_price=round(start_p, 6),
                end_price=round(end_p, 6),
                start_time=prev["ts"],
                end_time=curr["ts"],
                price_change_percent=round(pct, 2),
                fib_retracement_of_prev=fib_ratio,
                rsi_at_end=round(rsi_val, 2),
                rsi_divergence=rsi_divergence[end_idx] if end_idx < len(rsi_divergence) else False,
                ao_divergence=ao_div,
                volume_anomaly=vol_anomaly[end_idx] if end_idx < len(vol_anomaly) else False,
                is_bullish=end_p > start_p,
                atr_size_ratio=atr_ratio,
                poc_price=round(poc, 6) if poc else None,
                is_liquidity_sweep=is_sweep,
                cvd_divergence=cvd_div,
            )
        )

    return vectors


# ─── Публичный API ────────────────────────────────────────────────────────────

def preprocess_timeframe(
    candles: list[dict],
    timeframe: str,
    orderbook_walls: Optional[dict] = None,
    sub_tf_vectors: Optional[list] = None,
) -> TimeframeData:
    """
    Полная обработка данных для одного таймфрейма с ATR-Fractal фильтрацией.

    Параметры ATR_FRACTAL_SETTINGS выбираются автоматически по таймфрейму.

    Args:
        candles:         Список свечей из data_fetcher.fetch_ohlcv().
        timeframe:       Строка таймфрейма (1w, 1d и т.д.).
        orderbook_walls: Результат data_fetcher.fetch_orderbook_walls() или None.

    Returns:
        Объект TimeframeData, готовый к помещению в LLMContext.

    Raises:
        PreprocessingError: Если данных недостаточно для обработки.
    """
    if len(candles) < 30:
        raise PreprocessingError(
            f"Слишком мало свечей для ТФ {timeframe}: {len(candles)} < 30"
        )

    # Настройки ATR-Fractal для данного ТФ (по умолчанию берём 1h настройки)
    tf_settings = ATR_FRACTAL_SETTINGS.get(timeframe, ATR_FRACTAL_SETTINGS.get("1h", {
        "fractal_n": 2, "atr_mult": 2.0, "atr_period": 14
    }))
    fractal_n  = tf_settings["fractal_n"]
    atr_mult   = tf_settings["atr_mult"]
    atr_period = tf_settings["atr_period"]

    timestamps = [c["ts"]     for c in candles]
    highs      = [c["high"]   for c in candles]
    lows       = [c["low"]    for c in candles]
    closes     = [c["close"]  for c in candles]
    vols       = [c["volume"] for c in candles]

    # ── Индикаторы ─────────────────────────────────────────────────────────
    rsi_series = rsi(closes, period=14)
    atr_series = atr(highs, lows, closes, period=atr_period)
    ao_series  = awesome_oscillator(highs, lows)
    vol_anom   = volume_anomaly_mask(vols, VOLUME_ANOMALY_PERIOD, VOLUME_ANOMALY_MULTIPLIER)
    rsi_div    = detect_rsi_divergence(closes, rsi_series, window=5)

    cvd_series = calculate_cvd(candles)

    # ── ATR-Fractal поиск экстремумов ──────────────────────────────────────
    ph_idx, pl_idx = _atr_fractal_pivots(
        highs, lows, closes,
        fractal_n=fractal_n,
        atr_vals=atr_series,
        atr_mult=atr_mult,
    )

    if len(ph_idx) + len(pl_idx) < 4:
        logger.warning(
            "Мало значимых экстремумов на ТФ %s (%d+%d). "
            "ATR-фильтр слишком строгий — снижаем atr_mult до %.1f",
            timeframe, len(ph_idx), len(pl_idx), atr_mult * 0.5,
        )
        # Авто-фолбэк: ослабляем фильтр если данных мало
        ph_idx, pl_idx = _atr_fractal_pivots(
            highs, lows, closes,
            fractal_n=fractal_n,
            atr_vals=atr_series,
            atr_mult=atr_mult * 0.5,
        )

    pivots = _merge_pivots(ph_idx, pl_idx, timestamps, highs, lows)

    # ── Векторы ────────────────────────────────────────────────────────────
    vectors = _build_vectors(
        pivots, candles, rsi_series, ao_series, vol_anom, rsi_div, atr_series, cvd_series
    )

    # ── Уровни Фибоначчи от 3 последних значимых свингов ──────────────────
    fib_levels: list[FibLevel] = []
    if len(pivots) >= 2:
        # Основной расчёт: последний значимый ход
        fib_levels = calc_fib_levels(pivots[-2]["price"], pivots[-1]["price"])

    # ── Поиск кластеров Фибоначчи ──────────────────────────────────────────
    current_price = closes[-1]
    fib_clusters = find_fib_clusters(
        pivots,
        current_price=current_price,
        tolerance=FIB_CLUSTER_TOLERANCE,
        top_n_swings=6,     # анализируем последние 6 свингов
    )
    
    # Сверка кластеров с Point of Control из векторов
    # Если кластер близко к POC какого-либо значимого вектора -> has_poc = True
    poc_nodes = [v.poc_price for v in vectors if v.poc_price]
    for cluster in fib_clusters:
        for poc in poc_nodes:
            if abs(cluster.price - poc) / poc <= FIB_CLUSTER_TOLERANCE:
                cluster.has_poc = True
                cluster.strength += 1  # Дополнительный балл силы за POC
                break

    # ── Финальные данные ───────────────────────────────────────────────────
    current_rsi = rsi_series[-1] if not math.isnan(rsi_series[-1]) else 50.0
    current_atr = atr_series[-1] if atr_series and not math.isnan(atr_series[-1]) else None

    walls = orderbook_walls or {}

    logger.info(
        "ТФ %s: %d свечей → %d векторов, %d Fib-кластеров, цена=%.2f, RSI=%.1f, ATR=%.2f",
        timeframe, len(candles), len(vectors), len(fib_clusters),
        current_price, current_rsi, current_atr or 0,
    )

    # ── Математический анализ структуры волн ────────────────────────────────
    # Фрактальная рекурсия: передаём векторы младшего ТФ для кросс-валидации
    wave_state = analyze_wave_structure(
        vectors,
        timeframe=timeframe,
        sub_tf_vectors=sub_tf_vectors,
    )

    return TimeframeData(
        timeframe=timeframe,
        vectors=vectors,
        fib_levels=fib_levels,
        fib_clusters=fib_clusters,
        current_price=round(current_price, 6),
        current_rsi=round(current_rsi, 2),
        current_atr=round(current_atr, 6) if current_atr else None,
        nearest_bid_wall=walls.get("bid_wall"),
        nearest_ask_wall=walls.get("ask_wall"),
        mathematical_wave_state=wave_state,
    )


def preprocess_all(
    symbol: str,
    all_candles: dict[str, list[dict]],
    timeframes: list[str],
    orderbook_walls: Optional[dict] = None,
) -> LLMContext:
    """
    Обрабатывает данные по всем таймфреймам от старшего к младшему.
    Реализует фрактальную рекурсию: сначала обрабатываются младшие ТФ,
    их векторы передаются в старший ТФ для кросс-валидации.

    Args:
        symbol:          Название пары.
        all_candles:     Словарь {timeframe: [candles]}.
        timeframes:      Список таймфреймов (старший первый).
        orderbook_walls: Стакан — добавляется только к самому младшему ТФ.

    Returns:
        LLMContext, готовый к сериализации для LLM.
    """
    # Обрабатываем от МЛАДШЕГО к СТАРШЕМУ, чтобы у каждого старшего
    # были доступны векторы младшего ТФ для фрактальной валидации.
    reversed_tfs = list(reversed(timeframes))
    tf_results: dict[str, TimeframeData] = {}

    for i, tf in enumerate(reversed_tfs):
        candles = all_candles.get(tf)
        if not candles:
            logger.warning("Нет свечей для ТФ %s, пропускаем.", tf)
            continue

        # Стакан добавляем только к самому младшему ТФ (первый в reversed)
        walls = orderbook_walls if i == 0 else None

        # Векторы младшего ТФ (предыдущий обработанный = i-1 в reversed)
        sub_tf_vecs = None
        if i > 0:
            prev_tf = reversed_tfs[i - 1]
            if prev_tf in tf_results:
                sub_tf_vecs = tf_results[prev_tf].vectors
                logger.info(
                    "Фрактальная рекурсия: %s получает %d векторов от %s",
                    tf, len(sub_tf_vecs), prev_tf,
                )

        try:
            tf_data = preprocess_timeframe(candles, tf, walls, sub_tf_vectors=sub_tf_vecs)
            tf_results[tf] = tf_data
        except PreprocessingError as e:
            logger.error("Ошибка обработки ТФ %s: %s", tf, e)

    # Собираем результат в правильном порядке (старший → младший)
    tf_data_list = [tf_results[tf] for tf in timeframes if tf in tf_results]

    return LLMContext(symbol=symbol, timeframes=tf_data_list)
