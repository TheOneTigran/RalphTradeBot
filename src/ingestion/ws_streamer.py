"""
ws_streamer.py — Отказоустойчивый WebSocket стример (CCXT PRO).

Архитектура: Two-Tier Event Model
═══════════════════════════════════
  FAST PATH (тики):
    watchTrades → ClusterBuilder (аккумуляция в окне текущей свечи)
    watchOrderBook → RedisCache (горячий кэш стакана)

  SLOW PATH (закрытие свечи):
    watchOHLCV → NewCandleEvent → [DAG recalculation]
    ClusterBuilder.flush() → ClusterUpdatedEvent
    LiquidityMapper.check() → LiquidityMapUpdatedEvent

Стример НЕ считает ничего. Он только:
  1. Поддерживает WS-соединения (мультиплексированные каналы)
  2. Реконнектится с exponential backoff при обрывах
  3. Инвалидирует OrderBook кэш при дисконнекте
  4. Пушит сырые данные в asyncio.Queue для downstream consumers
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from src.core.config import (
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    WS_MAX_RECONNECT_ATTEMPTS,
    WS_RECONNECT_DELAY,
)
from src.events.bus import get_event_bus
from src.events.models import (
    EventType,
    NewCandleEvent,
    NewTradeEvent,
    OrderBookSnapshotEvent,
)
from src.storage.redis_cache import get_cache

logger = logging.getLogger(__name__)


class WSStreamer:
    """
    Мультиплексированный WebSocket стример через CCXT PRO.

    Каналы:
      - OHLCV (watchOHLCV): детектирует закрытие свечи → NewCandleEvent
      - Trades (watchTrades): лента тиков → trade_queue для ClusterBuilder
      - OrderBook (watchOrderBook): снимки стакана → Redis + orderbook_queue

    Reconnect: exponential backoff (delay * 2^attempt), max attempts конфигурируем.
    При дисконнекте: OrderBook кэш инвалидируется через Redis TTL.
    """

    def __init__(
        self,
        symbol: str,
        timeframes: List[str] = None,
        trade_queue: Optional[asyncio.Queue] = None,
        orderbook_queue: Optional[asyncio.Queue] = None,
    ):
        self.symbol = symbol
        self.timeframes = timeframes or ["1m", "5m", "15m", "1h", "4h"]
        self.trade_queue = trade_queue or asyncio.Queue(maxsize=50_000)
        self.orderbook_queue = orderbook_queue or asyncio.Queue(maxsize=1_000)

        self._exchange = None
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Трекинг последних закрытых свечей (для детекции "нового close")
        self._last_candle_ts: Dict[str, int] = {}

        # Статистика
        self._stats = {
            "trades_received": 0,
            "candles_emitted": 0,
            "orderbook_updates": 0,
            "reconnects": 0,
            "errors": 0,
        }

    # ─── Lifecycle ────────────────────────────────────────────────────────

    async def _create_exchange(self):
        """Создает async CCXT PRO exchange."""
        try:
            import ccxt.pro as ccxtpro
        except ImportError:
            # Fallback: ccxt async (без WS)
            import ccxt.async_support as ccxtpro
            logger.warning("ccxt.pro not available, falling back to REST polling mode")

        self._exchange = ccxtpro.bybit({
            "apiKey": BYBIT_API_KEY or None,
            "secret": BYBIT_API_SECRET or None,
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })
        return self._exchange

    async def start(self) -> None:
        """Запуск всех WS каналов."""
        if self._running:
            return

        await self._create_exchange()
        self._running = True

        # Запускаем каждый канал в отдельной таске
        self._tasks.append(asyncio.create_task(self._run_channel("ohlcv")))
        self._tasks.append(asyncio.create_task(self._run_channel("trades")))
        self._tasks.append(asyncio.create_task(self._run_channel("orderbook")))

        logger.info(
            "WSStreamer started: %s | TFs: %s | Channels: ohlcv, trades, orderbook",
            self.symbol, self.timeframes,
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

        if self._exchange:
            try:
                await self._exchange.close()
            except Exception:
                pass
            self._exchange = None

        logger.info("WSStreamer stopped. Stats: %s", self._stats)

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats.copy()

    # ─── Channel Runner (with reconnect) ──────────────────────────────────

    async def _run_channel(self, channel: str) -> None:
        """
        Запуск канала с auto-reconnect и exponential backoff.
        Каждый канал работает в бесконечном цикле.
        """
        attempt = 0
        delay = WS_RECONNECT_DELAY

        while self._running:
            try:
                if channel == "ohlcv":
                    await self._watch_ohlcv()
                elif channel == "trades":
                    await self._watch_trades()
                elif channel == "orderbook":
                    await self._watch_orderbook()

                # Если вышли нормально — сбрасываем счётчик
                attempt = 0
                delay = WS_RECONNECT_DELAY

            except asyncio.CancelledError:
                break

            except Exception as e:
                attempt += 1
                self._stats["errors"] += 1
                self._stats["reconnects"] += 1

                if attempt > WS_MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "Channel '%s' exceeded max reconnect attempts (%d). Giving up.",
                        channel, WS_MAX_RECONNECT_ATTEMPTS,
                    )
                    break

                # Exponential backoff: 5s → 10s → 20s → 40s → ...
                wait = min(delay * (2 ** (attempt - 1)), 120.0)
                logger.warning(
                    "Channel '%s' error (attempt %d/%d): %s. Reconnecting in %.1fs...",
                    channel, attempt, WS_MAX_RECONNECT_ATTEMPTS, e, wait,
                )

                # Инвалидируем OrderBook кэш при дисконнекте
                if channel == "orderbook":
                    cache = get_cache()
                    cache.delete(f"ob:{self.symbol}")
                    logger.info("OrderBook cache invalidated for %s", self.symbol)

                await asyncio.sleep(wait)

                # Пересоздаём exchange после дисконнекта
                try:
                    if self._exchange:
                        await self._exchange.close()
                except Exception:
                    pass
                await self._create_exchange()

    # ─── OHLCV Watcher ────────────────────────────────────────────────────

    async def _watch_ohlcv(self) -> None:
        """
        Стримит OHLCV и детектирует закрытие свечи.
        
        CCXT PRO watchOHLCV возвращает массив свечей.
        Мы отслеживаем timestamp последней закрытой свечи.
        Когда ts меняется — значит предыдущая свеча закрылась.
        """
        bus = get_event_bus()

        while self._running:
            for tf in self.timeframes:
                try:
                    ohlcv_list = await self._exchange.watch_ohlcv(self.symbol, tf)
                except Exception as e:
                    logger.debug("watchOHLCV %s:%s error: %s", self.symbol, tf, e)
                    raise  # Пробросим для reconnect

                if not ohlcv_list:
                    continue

                # Последняя свеча в массиве — текущая (ещё формируется)
                # Предпоследняя (если есть) — только что закрылась
                for candle in ohlcv_list:
                    ts = int(candle[0])
                    key = f"{self.symbol}:{tf}"

                    if key in self._last_candle_ts and ts != self._last_candle_ts[key]:
                        # Новый timestamp → предыдущая свеча закрылась
                        prev = ohlcv_list[-2] if len(ohlcv_list) >= 2 else candle
                        event = NewCandleEvent(
                            symbol=self.symbol,
                            timeframe=tf,
                            ts=int(prev[0]),
                            open=float(prev[1]),
                            high=float(prev[2]),
                            low=float(prev[3]),
                            close=float(prev[4]),
                            volume=float(prev[5]),
                        )
                        await bus.publish(event)
                        self._stats["candles_emitted"] += 1

                    self._last_candle_ts[key] = ts

    # ─── Trades Watcher (Fast Path → ClusterBuilder queue) ────────────────

    async def _watch_trades(self) -> None:
        """
        Стримит ленту тиков (AggTrades).
        Каждый тик пушится в trade_queue для ClusterBuilder.
        Не публикует в EventBus напрямую (слишком высокая частота).
        """
        while self._running:
            try:
                trades = await self._exchange.watch_trades(self.symbol)
            except Exception as e:
                logger.debug("watchTrades %s error: %s", self.symbol, e)
                raise

            for trade in trades:
                self._stats["trades_received"] += 1

                trade_data = {
                    "trade_id": trade.get("id", 0),
                    "price": float(trade["price"]),
                    "quantity": float(trade["amount"]),
                    "is_buyer_maker": trade.get("side", "") == "sell",
                    "trade_time": int(trade["timestamp"]),
                }

                try:
                    self.trade_queue.put_nowait(trade_data)
                except asyncio.QueueFull:
                    # Дропаем самый старый тик, чтобы не блокировать
                    try:
                        self.trade_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    self.trade_queue.put_nowait(trade_data)

    # ─── OrderBook Watcher (Fast Path → Redis + queue) ────────────────────

    async def _watch_orderbook(self) -> None:
        """
        Стримит стакан.
        Пушит в Redis кэш (TTL=5s) и в orderbook_queue для LiquidityMapper.
        
        При дисконнекте Redis запись протухнет по TTL,
        что автоматически инвалидизирует стейт.
        """
        cache = get_cache()

        while self._running:
            try:
                ob = await self._exchange.watch_order_book(self.symbol, limit=50)
            except Exception as e:
                logger.debug("watchOrderBook %s error: %s", self.symbol, e)
                raise

            bids = ob.get("bids", [])[:25]
            asks = ob.get("asks", [])[:25]
            mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else None

            snapshot = {
                "bids": bids,
                "asks": asks,
                "mid_price": mid_price,
                "timestamp": int(time.time() * 1000),
            }

            # Горячий кэш (TTL=5s — автоинвалидация при дисконнекте)
            cache.cache_orderbook(self.symbol, snapshot)

            self._stats["orderbook_updates"] += 1

            # Пушим в очередь для LiquidityMapper (с дропом старых)
            try:
                self.orderbook_queue.put_nowait(snapshot)
            except asyncio.QueueFull:
                try:
                    self.orderbook_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.orderbook_queue.put_nowait(snapshot)
