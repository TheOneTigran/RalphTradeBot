"""
extremum_finder.py — Адаптивный поиск экстремумов (пиков и впадин).

Три метода детекции:
  1. ATR-Fractal (из legacy math_preprocessor — проверенный)
  2. scipy.signal.find_peaks с динамическим prominence = f(ATR)
  3. CWT (Continuous Wavelet Transform) для мульти-scale разложения

Каждый метод возвращает List[Extremum].
Итоговый результат — консенсус (пересечение или взвешенное голосование).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data Classes
# ═════════════════════════════════════════════════════════════════════════════

class DetectionMethod(str, Enum):
    ATR_FRACTAL = "atr_fractal"
    FIND_PEAKS = "find_peaks"
    CWT = "cwt"
    CONSENSUS = "consensus"


@dataclass
class Extremum:
    """
    Одна точка экстремума (пик или впадина).
    
    Attributes:
        price: Цена экстремума
        timestamp: Unix timestamp (ms)
        index: Индекс в массиве свечей
        is_high: True = локальный максимум, False = локальный минимум
        method: Метод детекции
        confidence: Уверенность в экстремуме (0..1). 
                    1.0 = подтверждён множеством методов
    """
    price: float
    timestamp: int
    index: int
    is_high: bool
    method: DetectionMethod = DetectionMethod.ATR_FRACTAL
    confidence: float = 1.0

    def __repr__(self) -> str:
        kind = "HIGH" if self.is_high else "LOW"
        return f"Extremum({kind}, {self.price:.2f}, idx={self.index}, conf={self.confidence:.2f})"


# ═════════════════════════════════════════════════════════════════════════════
# ATR Calculation
# ═════════════════════════════════════════════════════════════════════════════

def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Вычисляет Average True Range (Wilder's smoothing)."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.zeros(n)
    atr[:period] = np.nan
    atr[period - 1] = np.mean(tr[:period])

    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


# ═════════════════════════════════════════════════════════════════════════════
# Method 1: ATR-Fractal (проверенный, из legacy)
# ═════════════════════════════════════════════════════════════════════════════

def _find_fractals_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray,
    fractal_n: int = 2,
    atr_mult: float = 1.0,
    atr_period: int = 14,
) -> List[Extremum]:
    """
    Фрактал Вильямса + ATR фильтр.
    
    Фрактальный High: high[i] > high[i-n..i-1] AND high[i] > high[i+1..i+n]
    Фрактальный Low:  low[i] < low[i-n..i-1] AND low[i] < low[i+1..i+n]
    
    ATR фильтр: отбрасываем маленькие свинги (< atr_mult * ATR).
    """
    n = len(close)
    if n < 2 * fractal_n + 1:
        return []

    atr = compute_atr(high, low, close, atr_period)
    raw_extrema: List[Extremum] = []

    for i in range(fractal_n, n - fractal_n):
        # Фрактальный High
        is_fractal_high = all(high[i] > high[i - j] for j in range(1, fractal_n + 1)) and \
                          all(high[i] > high[i + j] for j in range(1, fractal_n + 1))
        if is_fractal_high:
            raw_extrema.append(Extremum(
                price=float(high[i]),
                timestamp=int(timestamps[i]),
                index=i,
                is_high=True,
                method=DetectionMethod.ATR_FRACTAL,
            ))

        # Фрактальный Low
        is_fractal_low = all(low[i] < low[i - j] for j in range(1, fractal_n + 1)) and \
                         all(low[i] < low[i + j] for j in range(1, fractal_n + 1))
        if is_fractal_low:
            raw_extrema.append(Extremum(
                price=float(low[i]),
                timestamp=int(timestamps[i]),
                index=i,
                is_high=False,
                method=DetectionMethod.ATR_FRACTAL,
            ))

    # Сортируем по индексу
    raw_extrema.sort(key=lambda e: e.index)

    # ATR-фильтр: отбрасываем малые свинги
    if len(raw_extrema) < 2:
        return raw_extrema

    filtered: List[Extremum] = [raw_extrema[0]]
    for ext in raw_extrema[1:]:
        prev = filtered[-1]
        swing_size = abs(ext.price - prev.price)
        local_atr = atr[ext.index] if not np.isnan(atr[ext.index]) else atr[~np.isnan(atr)][-1]
        min_size = atr_mult * local_atr

        # Если направление одинаковое — берём более экстремальный
        if ext.is_high == prev.is_high:
            if ext.is_high and ext.price > prev.price:
                filtered[-1] = ext
            elif not ext.is_high and ext.price < prev.price:
                filtered[-1] = ext
            continue

        # Если свинг достаточно большой — добавляем
        if swing_size >= min_size:
            filtered.append(ext)

    return filtered


# ═════════════════════════════════════════════════════════════════════════════
# Method 2: scipy.signal.find_peaks
# ═════════════════════════════════════════════════════════════════════════════

def _find_peaks_scipy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray,
    atr_period: int = 14,
    prominence_mult: float = 1.0,
) -> List[Extremum]:
    """
    Поиск экстремумов через scipy.signal.find_peaks.
    
    Prominence (значимость) адаптивна: prominence = ATR * prominence_mult
    Это автоматически фильтрует шум на волатильных рынках.
    """
    atr = compute_atr(high, low, close, atr_period)
    median_atr = float(np.nanmedian(atr))
    prominence = median_atr * prominence_mult

    # Пики (highs)
    peak_indices, _ = find_peaks(high, prominence=prominence, distance=3)

    # Впадины (lows) — инвертируем
    valley_indices, _ = find_peaks(-low, prominence=prominence, distance=3)

    extrema: List[Extremum] = []

    for idx in peak_indices:
        extrema.append(Extremum(
            price=float(high[idx]),
            timestamp=int(timestamps[idx]),
            index=int(idx),
            is_high=True,
            method=DetectionMethod.FIND_PEAKS,
            confidence=0.9,
        ))

    for idx in valley_indices:
        extrema.append(Extremum(
            price=float(low[idx]),
            timestamp=int(timestamps[idx]),
            index=int(idx),
            is_high=False,
            method=DetectionMethod.FIND_PEAKS,
            confidence=0.9,
        ))

    extrema.sort(key=lambda e: e.index)
    return extrema


# ═════════════════════════════════════════════════════════════════════════════
# Method 3: Multi-Scale Find Peaks (Fallback for CWT)
# ═════════════════════════════════════════════════════════════════════════════

def _find_extrema_cwt(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray,
    widths: Optional[List[int]] = None,
    min_snr: float = 1.0,
) -> List[Extremum]:
    """
    Multi-scale экстремумы.
    Используем find_peaks с разными значениями distance/prominence для поиска пиков на разных масштабах.
    """
    if widths is None:
        widths = [3, 8, 20]
        
    extrema: List[Extremum] = []
    
    prominences = [0.5, 1.0, 2.0]

    for scale_idx, (w, prom) in enumerate(zip(widths, prominences)):
        # Считаем ATR
        atr = compute_atr(high, low, close, period=14)
        median_atr = float(np.nanmedian(atr))
        run_prom = median_atr * prom
        
        peak_idx, _ = find_peaks(high, distance=w, prominence=run_prom)
        for idx in peak_idx:
            extrema.append(Extremum(
                price=float(high[idx]),
                timestamp=int(timestamps[idx]),
                index=int(idx),
                is_high=True,
                method=DetectionMethod.CWT,
                confidence=0.7 + 0.1 * scale_idx,
            ))

        valley_idx, _ = find_peaks(-low, distance=w, prominence=run_prom)
        for idx in valley_idx:
            extrema.append(Extremum(
                price=float(low[idx]),
                timestamp=int(timestamps[idx]),
                index=int(idx),
                is_high=False,
                method=DetectionMethod.CWT,
                confidence=0.7 + 0.1 * scale_idx,
            ))

    extrema.sort(key=lambda e: (e.index, -e.confidence))
    deduped: List[Extremum] = []
    for ext in extrema:
        if deduped and abs(ext.index - deduped[-1].index) <= 2 and ext.is_high == deduped[-1].is_high:
            if ext.confidence > deduped[-1].confidence:
                deduped[-1] = ext
        else:
            deduped.append(ext)

    return deduped


# ═════════════════════════════════════════════════════════════════════════════
# ExtremumFinder: Consensus Engine
# ═════════════════════════════════════════════════════════════════════════════

class ExtremumFinder:
    """
    Фасад для поиска экстремумов с консенсусом между методами.
    
    Режимы:
      - single: использовать один метод (по умолчанию ATR_FRACTAL)
      - consensus: голосование 2-из-3 методов (более надёжно, медленнее)
    
    Пример:
        finder = ExtremumFinder(mode="consensus")
        extrema = finder.find(high, low, close, timestamps, settings)
    """

    def __init__(self, mode: str = "single", primary_method: DetectionMethod = DetectionMethod.ATR_FRACTAL):
        self.mode = mode
        self.primary_method = primary_method

    def find(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timestamps: np.ndarray,
        fractal_n: int = 2,
        atr_mult: float = 1.0,
        atr_period: int = 14,
    ) -> List[Extremum]:
        """
        Основной метод поиска экстремумов.
        
        Returns:
            Список Extremum, отсортированных по индексу, с чередованием high/low.
        """
        if self.mode == "single":
            if self.primary_method == DetectionMethod.ATR_FRACTAL:
                return _find_fractals_atr(high, low, close, timestamps, fractal_n, atr_mult, atr_period)
            elif self.primary_method == DetectionMethod.FIND_PEAKS:
                return _find_peaks_scipy(high, low, close, timestamps, atr_period, atr_mult)
            elif self.primary_method == DetectionMethod.CWT:
                return _find_extrema_cwt(high, low, close, timestamps)
            else:
                return _find_fractals_atr(high, low, close, timestamps, fractal_n, atr_mult, atr_period)
        else:
            return self._consensus(high, low, close, timestamps, fractal_n, atr_mult, atr_period)

    def _consensus(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timestamps: np.ndarray,
        fractal_n: int,
        atr_mult: float,
        atr_period: int,
    ) -> List[Extremum]:
        """
        Консенсус 2-из-3: экстремум подтверждён если минимум 2 метода согласны.
        
        Алгоритм:
          1. Запускаем все 3 метода
          2. Для каждого экстремума считаем, сколько методов его нашли (±2 бара)
          3. Confidence = votes / 3
          4. Фильтр: оставляем только с votes >= 2
          5. Обеспечиваем чередование high/low
        """
        results_atr = _find_fractals_atr(high, low, close, timestamps, fractal_n, atr_mult, atr_period)
        results_peaks = _find_peaks_scipy(high, low, close, timestamps, atr_period, atr_mult)
        results_cwt = _find_extrema_cwt(high, low, close, timestamps)

        all_methods = [results_atr, results_peaks, results_cwt]
        
        # Собираем все уникальные экстремумы (по индексу ± tolerance)
        tolerance = 2  # баров
        candidates: Dict[Tuple[int, bool], List[Extremum]] = {}

        for method_results in all_methods:
            for ext in method_results:
                key = (ext.index, ext.is_high)
                # Ищем ближайший существующий ключ
                matched = False
                for existing_key in list(candidates.keys()):
                    if existing_key[1] == ext.is_high and abs(existing_key[0] - ext.index) <= tolerance:
                        candidates[existing_key].append(ext)
                        matched = True
                        break
                if not matched:
                    candidates[key] = [ext]

        # Фильтруем: минимум 2 голоса
        confirmed: List[Extremum] = []
        for key, votes in candidates.items():
            n_methods = len(set(v.method for v in votes))
            if n_methods >= 2:
                # Берём экстремум с наибольшей confidence
                best = max(votes, key=lambda v: v.confidence)
                best.confidence = n_methods / 3.0
                best.method = DetectionMethod.CONSENSUS
                confirmed.append(best)

        confirmed.sort(key=lambda e: e.index)

        # Обеспечиваем чередование high/low
        return self._enforce_alternation(confirmed)

    @staticmethod
    def _enforce_alternation(extrema: List[Extremum]) -> List[Extremum]:
        """
        Обеспечивает строгое чередование HIGH-LOW-HIGH-LOW.
        При конфликте оставляем более экстремальный (выше high / ниже low).
        """
        if len(extrema) <= 1:
            return extrema

        result: List[Extremum] = [extrema[0]]
        for ext in extrema[1:]:
            if ext.is_high != result[-1].is_high:
                # Чередование соблюдается
                result.append(ext)
            else:
                # Конфликт: два подряд одного типа
                if ext.is_high:
                    # Оставляем более высокий
                    if ext.price > result[-1].price:
                        result[-1] = ext
                else:
                    # Оставляем более низкий
                    if ext.price < result[-1].price:
                        result[-1] = ext

        return result
