"""
data_fetcher.py — Модуль взаимодействия с биржей Bybit через ccxt.

Обязанности:
  - Выкачка OHLCV-свечей по нескольким таймфреймам с пагинацией
  - Сбор данных Open Interest (история OI для анализа тренда)
  - Сбор позиций Funding Rate (сентимент лонг/шорт)
  - Расширенный анализ стакана: поиск крупных лимитных блоков
"""
from __future__ import annotations

import time
import logging
from typing import Optional

import ccxt
import numpy as np

from src.core.config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    CANDLE_LIMIT,
    TIMEFRAMES,
)
from src.core.models import OISnapshot
from src.core.exceptions import DataFetchError

logger = logging.getLogger(__name__)

# ─── Инициализация биржи ──────────────────────────────────────────────────────

def _create_exchange() -> ccxt.bybit:
    """Создаёт и возвращает экземпляр ccxt.bybit."""
    return ccxt.bybit(
        {
            "apiKey": BYBIT_API_KEY or None,
            "secret": BYBIT_API_SECRET or None,
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},   # USDT-perpetuals
        }
    )


# ─── OHLCV ───────────────────────────────────────────────────────────────────

_TF_MAP = {
    "1w": "1w", "1d": "1d", "4h": "4h",
    "1h": "1h", "15m": "15m", "5m": "5m",
}


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = CANDLE_LIMIT,
) -> list[dict]:
    """
    Выкачивает историю OHLCV с Bybit для заданного таймфрейма (с пагинацией).

    Args:
        symbol:    Торговая пара, например 'BTCUSDT'.
        timeframe: Таймфрейм (1w / 1d / 4h / 1h / 15m / 5m).
        limit:     Нужное количество свечей. Если limit > 1000, будет пагинация.

    Returns:
        Список словарей: [{ 'ts', 'open', 'high', 'low', 'close', 'volume' }]
    """
    exchange = _create_exchange()
    tf = _TF_MAP.get(timeframe)
    if tf is None:
        raise DataFetchError(f"Неизвестный таймфрейм: {timeframe}")

    all_raw = []
    # Bybit max limit per request is 1000
    batch_size = 1000
    
    # Calculate starting 'since' timestamp by estimating from the limit
    # This is a rough estimation, ccxt provides an easier way using end_time 
    # but ccxt Bybit implementation paginates smoothly moving forward from 'since'.
    # To fetch recent N candles best, we first fetch latest to get current time,
    # then calculate 'since' strictly based on timeframe seconds.
    tf_seconds = exchange.parse_timeframe(tf)
    
    try:
        logger.info("Загрузка %s свечей [%s] для %s...", limit, tf, symbol)
        latest = exchange.fetch_ohlcv(symbol, tf, limit=1)
        if not latest:
            return []
        
        last_ts = latest[0][0]
        # Start ts:
        start_ts = last_ts - (limit * tf_seconds * 1000)
        
        since = start_ts
        while len(all_raw) < limit:
            remaining = limit - len(all_raw)
            fetch_limit = min(batch_size, remaining)
            
            batch = exchange.fetch_ohlcv(symbol, tf, since=since, limit=fetch_limit)
            if not batch:
                break
                
            all_raw.extend(batch)
            since = batch[-1][0] + (tf_seconds * 1000)
            
            if len(batch) < fetch_limit:
                # No more data
                break
            
            time.sleep(0.2) # Rate limit respect
            
    except ccxt.NetworkError as e:
        raise DataFetchError(f"Ошибка сети при загрузке OHLCV: {e}") from e
    except ccxt.ExchangeError as e:
        raise DataFetchError(f"Ошибка биржи при загрузке OHLCV: {e}") from e

    # ccxt might give duplicates if 'since' overlaps, so deduplicate
    unique_candles = {c[0]: c for c in all_raw}
    sorted_raw = sorted(unique_candles.values(), key=lambda x: x[0])
    
    # Ensure we don't return more than requested
    sorted_raw = sorted_raw[-limit:]

    candles = [
        {
            "ts": c[0],
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        }
        for c in sorted_raw
    ]
    logger.info("Получено %d свечей [%s] для %s.", len(candles), tf, symbol)
    return candles


def fetch_all_timeframes(symbol: str) -> dict[str, list[dict]]:
    """
    Выкачивает OHLCV по всем настроенным таймфреймам для одной монеты.
    Возвращает словарь { timeframe: [candles] }.
    """
    result: dict[str, list[dict]] = {}
    for tf in TIMEFRAMES:
        result[tf] = fetch_ohlcv(symbol, tf, limit=CANDLE_LIMIT)
        time.sleep(0.3)
    return result


# ─── Open Interest ────────────────────────────────────────────────────────────

def fetch_open_interest(
    symbol: str,
    period: str = "1d",
    limit: int = 30,
) -> Optional[OISnapshot]:
    """
    Загружает историю Open Interest с Bybit V5 и вычисляет ключевые метрики.

    Метрики:
      - current_oi: текущий OI
      - oi_change_pct_24h: изменение OI за 24ч (сравниваем последние 2 точки)
      - oi_price_divergence: цена и OI движутся в разные стороны (сигнал разворота)
      - funding_rate: актуальная ставка финансирования

    Args:
        symbol: Торговая пара, например 'BTCUSDT'.
        period: Период истории OI ('5min', '15min', '1h', '4h', '1d').
        limit:  Количество точек истории (макс. 200).

    Returns:
        OISnapshot или None при ошибке.
    """
    exchange = _create_exchange()

    try:
        # Bybit V5: /v5/market/open-interest
        oi_history = exchange.fetch_open_interest_history(
            symbol=symbol,
            timeframe=period,
            limit=limit,
        )
    except Exception as e:
        logger.warning("Не удалось загрузить OI для %s: %s", symbol, e)
        return None

    if not oi_history or len(oi_history) < 2:
        return None

    # Извлекаем значения OI (ccxt возвращает список dicts с 'openInterestAmount')
    try:
        oi_values = [
            float(entry.get("openInterestAmount") or entry.get("openInterest") or 0)
            for entry in oi_history
        ]
        oi_values = [v for v in oi_values if v > 0]
        if len(oi_values) < 2:
            return None

        current_oi = oi_values[-1]
        prev_oi = oi_values[-2]
        oi_change_pct = round((current_oi - prev_oi) / prev_oi * 100, 3) if prev_oi else None

        # Тренд OI за весь период (первый vs последний)
        old_oi = oi_values[0]
        oi_trend_positive = current_oi > old_oi   # OI растёт = новые деньги

    except (KeyError, TypeError, ZeroDivisionError) as e:
        logger.warning("Ошибка обработки OI для %s: %s", symbol, e)
        return None

    # Funding Rate
    funding_rate: Optional[float] = None
    try:
        funding_data = exchange.fetch_funding_rate(symbol)
        if funding_data:
            funding_rate = round(float(funding_data.get("fundingRate", 0)) * 100, 6)
    except Exception:
        pass  # Funding rate не критичен, не прерываем

    logger.info(
        "OI [%s]: %.2f (Δ24h: %s%%), funding: %s%%",
        symbol, current_oi,
        f"{oi_change_pct:+.3f}" if oi_change_pct is not None else "N/A",
        f"{funding_rate:.4f}" if funding_rate is not None else "N/A",
    )

    return OISnapshot(
        current_oi=round(current_oi, 2),
        oi_change_pct_24h=oi_change_pct,
        oi_price_divergence=None,   # расчёт дивергенции делаем ниже в preprocessor
        funding_rate=funding_rate,
    )


# ─── Расширенный стакан ───────────────────────────────────────────────────────

def fetch_orderbook_walls(
    symbol: str,
    depth: int = 200,
    wall_multiplier: float = 5.0,
) -> dict:
    """
    Определяет крупные блоки ликвидности в стакане (bid/ask walls).

    Алгоритм:
      1. Загружает топ-depth уровней с каждой стороны.
      2. Считает средний объём на уровень.
      3. Уровень = «стена» если volume > wall_multiplier × среднего.
      4. Возвращает ВСЕ стены (не только ближайшую), отсортированные по близости к цене.

    Args:
        symbol:          Торговая пара.
        depth:           Глубина стакана (строк с каждой стороны).
        wall_multiplier: Множитель для аномально крупного ордера.

    Returns:
        {
          'bid_wall': float | None,          # ближайший крупный бид
          'ask_wall': float | None,          # ближайший крупный аск
          'bid_walls': list[dict],           # все стены покупателей
          'ask_walls': list[dict],           # все стены продавцов
          'mid_price': float | None,
        }
    """
    exchange = _create_exchange()
    try:
        ob = exchange.fetch_order_book(symbol, limit=depth)
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        logger.warning("Не удалось загрузить стакан для %s: %s", symbol, e)
        return {"bid_wall": None, "ask_wall": None, "bid_walls": [], "ask_walls": [], "mid_price": None}

    bids: list[list] = ob.get("bids", [])
    asks: list[list] = ob.get("asks", [])

    mid_price: Optional[float] = None
    if bids and asks:
        mid_price = (bids[0][0] + asks[0][0]) / 2

    def _find_walls(side: list[list], is_bid: bool) -> list[dict]:
        if not side:
            return []
        qtys = [row[1] for row in side]
        avg = float(np.mean(qtys)) if qtys else 0
        threshold = avg * wall_multiplier
        walls = []
        for price, qty in side:
            if qty >= threshold:
                walls.append({
                    "price": float(price),
                    "qty": float(qty),
                    "strength_x": round(qty / avg, 1) if avg > 0 else 0,
                    "side": "bid" if is_bid else "ask",
                })
        # Сортируем по близости к mid_price
        if mid_price:
            walls.sort(key=lambda w: abs(w["price"] - mid_price))
        return walls

    bid_walls = _find_walls(bids, is_bid=True)
    ask_walls = _find_walls(asks, is_bid=False)

    if bid_walls or ask_walls:
        logger.info(
            "Стакан [%s]: %d bid-стен, %d ask-стен, mid=%.4g",
            symbol, len(bid_walls), len(ask_walls), mid_price or 0,
        )

    return {
        "bid_wall": bid_walls[0]["price"] if bid_walls else None,
        "ask_wall": ask_walls[0]["price"] if ask_walls else None,
        "bid_walls": bid_walls[:5],    # топ-5 крупнейших стен бидов
        "ask_walls": ask_walls[:5],    # топ-5 крупнейших стен асков
        "mid_price": mid_price,
    }
