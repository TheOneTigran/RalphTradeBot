"""
indicators.py — Расчёт технических индикаторов на массивах OHLCV.

Использует только numpy, без внешних TA-библиотек — максимальная гибкость.
"""
from __future__ import annotations

import numpy as np


# ─── RSI ─────────────────────────────────────────────────────────────────────

def rsi(closes: list[float], period: int = 14) -> list[float]:
    """
    RSI(period) с расчётом по методу Wilder's smoothing (EMA-like).

    Args:
        closes: Список цен закрытия (хронологический порядок).
        period: Период RSI (по умолчанию 14).

    Returns:
        Список значений RSI (float). Длина = len(closes). Первые period — NaN.
    """
    prices = np.array(closes, dtype=float)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.full(len(prices), np.nan)
    avg_loss = np.full(len(prices), np.nan)

    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
    rsi_values = np.where(avg_loss != 0, 100.0 - (100.0 / (1.0 + rs)), 100.0)
    rsi_values[:period] = np.nan
    return rsi_values.tolist()


# ─── SMA ─────────────────────────────────────────────────────────────────────

def sma(values: list[float], period: int) -> list[float]:
    """
    Simple Moving Average (SMA) по скользящему окну.

    Args:
        values: Исходный массив числовых значений.
        period: Ширина окна.

    Returns:
        Список значений SMA. Первые (period-1) элементов — NaN.
    """
    arr = np.array(values, dtype=float)
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.mean(arr[i - period + 1 : i + 1])
    return result.tolist()


# ─── ATR ─────────────────────────────────────────────────────────────────────

def atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> list[float]:
    """
    Average True Range (ATR) — мера волатильности рынка.

    True Range[i] = max(high - low,
                        |high - prev_close|,
                        |low  - prev_close|)

    Реализация: Wilder's smoothing (как в оригинальном индикаторе Welles Wilder).

    Args:
        highs:  Максимумы свечей.
        lows:   Минимумы свечей.
        closes: Цены закрытия.
        period: Период ATR (по умолчанию 14).

    Returns:
        Список значений ATR. Первые period элементов — NaN.
    """
    h = np.array(highs, dtype=float)
    l = np.array(lows, dtype=float)
    c = np.array(closes, dtype=float)

    n = len(h)
    tr = np.full(n, np.nan)

    # Первый TR — просто high - low
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    atr_vals = np.full(n, np.nan)
    if n > period:
        # Seed: SMA от первых period значений TR
        atr_vals[period - 1] = np.mean(tr[:period])
        # Wilder's smoothing
        for i in range(period, n):
            atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period

    return atr_vals.tolist()


# ─── Аномальный объём ────────────────────────────────────────────────────────

def volume_anomaly_mask(
    volumes: list[float],
    sma_period: int = 20,
    multiplier: float = 2.5,
) -> list[bool]:
    """
    Маска аномально высокого объёма: volume[i] > multiplier × SMA(volume, period).

    Args:
        volumes:    Массив объёмов (хронологический).
        sma_period: Период SMA для нормализации.
        multiplier: Множитель порога (по умолчанию 2.5).

    Returns:
        Список булевых значений той же длины.
    """
    sma_vol = sma(volumes, sma_period)
    result = []
    for v, s in zip(volumes, sma_vol):
        if np.isnan(s) or s == 0:
            result.append(False)
        else:
            result.append(v > multiplier * s)
    return result


# ─── Дивергенция RSI ─────────────────────────────────────────────────────────

def detect_rsi_divergence(
    prices: list[float],
    rsi_values: list[float],
    window: int = 5,
) -> list[bool]:
    """
    Детектирует бычью и медвежью дивергенцию RSI на скользящем окне.

    Медвежья: цена обновляет максимум, RSI — нет.
    Бычья:    цена обновляет минимум, RSI — нет.

    Args:
        prices:     Массив цен (close).
        rsi_values: Массив RSI той же длины.
        window:     Размер сравниваемого окна (свечей назад).

    Returns:
        Список булевых значений: True — дивергенция на данной свече.
    """
    n = len(prices)
    divergence = [False] * n

    for i in range(window, n):
        prev_prices = prices[i - window : i]
        prev_rsi = rsi_values[i - window : i]

        if any(np.isnan(r) for r in prev_rsi) or np.isnan(rsi_values[i]):
            continue

        if prices[i] > max(prev_prices) and rsi_values[i] < max(prev_rsi):
            divergence[i] = True
        elif prices[i] < min(prev_prices) and rsi_values[i] > min(prev_rsi):
            divergence[i] = True

    return divergence


# ─── Awesome Oscillator ──────────────────────────────────────────────────────

def awesome_oscillator(
    highs: list[float],
    lows: list[float],
    fast: int = 5,
    slow: int = 34,
) -> list[float]:
    """
    Awesome Oscillator (AO) = SMA(median_price, fast) - SMA(median_price, slow).

    Медианная цена: (high + low) / 2.
    Используется для подтверждения разворота волны и фильтрации ложных свингов.

    Args:
        highs: Максимумы свечей.
        lows:  Минимумы свечей.
        fast:  Быстрый период SMA (по умолчанию 5).
        slow:  Медленный период SMA (по умолчанию 34).

    Returns:
        Список значений AO. Первые (slow-1) элементов — NaN.
    """
    median = [(h + l) / 2 for h, l in zip(highs, lows)]
    sma_fast = sma(median, fast)
    sma_slow = sma(median, slow)

    result = []
    for f, s in zip(sma_fast, sma_slow):
        if np.isnan(f) or np.isnan(s):
            result.append(float("nan"))
        else:
            result.append(f - s)
    return result


# ─── Volume Profile / VPVR ───────────────────────────────────────────────────

def calculate_vpvr_poc(
    candles: list[dict],
    start_idx: int,
    end_idx: int,
    bins: int = 50,
) -> float | None:
    """
    Рассчитывает Point of Control (POC) — цену с максимальным проторгованным объёмом 
    в заданном диапазоне (от start_idx до end_idx).
    
    Args:
        candles:   Список словарей OHLCV (ccxt format).
        start_idx: Индекс начала свинга.
        end_idx:   Индекс конца свинга.
        bins:      Количество ценовых уровней (корзин) для профиля.
        
    Returns:
        Цена POC или None при ошибке.
    """
    if start_idx < 0 or end_idx > len(candles) or start_idx >= end_idx:
        return None
        
    slice_candles = candles[start_idx:end_idx]
    if not slice_candles:
        return None
        
    prices = []
    vols = []
    
    for c in slice_candles:
        # В качестве цены сделок берём медианную цену Typical Price свечи
        tp = (c['high'] + c['low'] + c['close']) / 3.0
        prices.append(tp)
        vols.append(c['volume'])
        
    if not prices:
        return None
        
    min_p, max_p = min(prices), max(prices)
    if min_p == max_p:
        return min_p
        
    step = (max_p - min_p) / bins
    if step <= 0: return min_p
    
    profile = {}
    for p, v in zip(prices, vols):
        bin_idx = int((p - min_p) / step)
        if bin_idx == bins: 
            bin_idx -= 1
        bin_center = min_p + (bin_idx * step) + (step / 2.0)
        profile[bin_center] = profile.get(bin_center, 0.0) + v
        
    if not profile:
        return None
        
    # Ищем бин с максимальным объемом
    return max(profile.keys(), key=lambda k: profile[k])


# ─── Cumulative Volume Delta (CVD) ───────────────────────────────────────────

def calculate_cvd(candles: list[dict]) -> list[float]:
    """
    Рассчитывает Cumulative Volume Delta (CVD) — кумулятивную дельту объёма.
    Поскольку тиковых данных нет, дельта аппроксимируется по формуле:
    Delta = Volume * ((Close - Open) / (High - Low)).
    
    Args:
        candles: Список словарей OHLCV (ccxt format).
        
    Returns:
        Список значений CVD на каждой свече.
    """
    cvd = []
    cum_delta = 0.0
    for c in candles:
        h, l, o, cl, v = c['high'], c['low'], c['open'], c['close'], c['volume']
        range_hl = h - l
        if range_hl == 0:
            delta = 0.0
        else:
            delta = v * ((cl - o) / range_hl)
        cum_delta += delta
        cvd.append(cum_delta)
    return cvd
