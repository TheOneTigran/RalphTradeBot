"""
bus.py — Внутренний Event Bus на базе asyncio.

Паттерн Pub/Sub для Event-Driven Architecture.
Каждый модуль подписывается на нужные EventType и публикует свои события.

Контракт:
  - subscribe(event_type, callback) — подписка
  - publish(event) — публикация (async)
  - Callback получает BaseEvent (или подкласс)

Преимущества перед Kafka/RabbitMQ для MVP:
  - Нулевая латентность (in-process)
  - Нет внешних зависимостей
  - Строгая типизация через Pydantic
  - Легко мигрировать на ZeroMQ/Kafka позже (тот же контракт publish/subscribe)
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from src.events.models import BaseEvent, EventType

logger = logging.getLogger(__name__)

# Тип callback: sync или async функция, принимающая BaseEvent
EventCallback = Callable[[BaseEvent], Union[None, Awaitable[None]]]


class EventBus:
    """
    Асинхронная шина событий (Pub/Sub).
    
    Потокобезопасна в рамках одного event loop.
    Поддерживает как sync, так и async обработчики.
    
    Пример использования:
        bus = EventBus()
        
        async def on_candle(event: NewCandleEvent):
            print(f"New candle: {event.symbol} {event.close}")
        
        bus.subscribe(EventType.NEW_CANDLE, on_candle)
        await bus.publish(NewCandleEvent(symbol="BTCUSDT", ...))
    """

    def __init__(self, max_queue_size: int = 10_000):
        self._subscribers: Dict[EventType, List[EventCallback]] = defaultdict(list)
        self._queue: asyncio.Queue[BaseEvent] = asyncio.Queue(maxsize=max_queue_size)
        self._running: bool = False
        self._dispatch_task: Optional[asyncio.Task] = None
        self._event_count: int = 0
        self._error_count: int = 0

    # ─── Public API ───────────────────────────────────────────────────────

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """
        Подписка на определённый тип событий.
        
        Args:
            event_type: Тип события из EventType enum
            callback: Функция-обработчик (sync или async)
        """
        self._subscribers[event_type].append(callback)
        logger.debug(
            "Subscribed %s to %s (total: %d)",
            callback.__name__, event_type.value, len(self._subscribers[event_type])
        )

    def unsubscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Отписка от типа события."""
        try:
            self._subscribers[event_type].remove(callback)
        except ValueError:
            pass

    async def publish(self, event: BaseEvent) -> None:
        """
        Публикация события в шину.
        
        Событие попадает в очередь и обрабатывается асинхронно.
        Если шина не запущена — вызываем обработчики напрямую (синхронный режим).
        """
        if self._running:
            await self._queue.put(event)
        else:
            # Прямая доставка без очереди (для тестов и синхронного режима)
            await self._dispatch(event)

    async def publish_nowait(self, event: BaseEvent) -> bool:
        """
        Неблокирующая публикация. Возвращает False если очередь переполнена.
        """
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            logger.warning("Event bus queue full, dropping event: %s", event.event_type.value)
            self._error_count += 1
            return False

    async def start(self) -> None:
        """Запуск фонового цикла обработки событий."""
        if self._running:
            return
        self._running = True
        self._dispatch_task = asyncio.create_task(self._process_loop())
        logger.info("Event Bus started (queue_size=%d)", self._queue.maxsize)

    async def stop(self) -> None:
        """Остановка шины. Дожидается обработки оставшихся событий."""
        if not self._running:
            return
        self._running = False

        # Ждём пока очередь опустеет
        if not self._queue.empty():
            logger.info("Draining %d remaining events...", self._queue.qsize())
            await self._queue.join()

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
            self._dispatch_task = None

        logger.info(
            "Event Bus stopped. Total events: %d, errors: %d",
            self._event_count, self._error_count
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """Статистика шины."""
        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "total_events": self._event_count,
            "total_errors": self._error_count,
            "subscriber_counts": {
                et.value: len(cbs) for et, cbs in self._subscribers.items() if cbs
            },
        }

    # ─── Internal ─────────────────────────────────────────────────────────

    async def _process_loop(self) -> None:
        """Фоновый цикл: забираем из очереди и диспатчим."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Event Bus loop error: %s", e, exc_info=True)
                self._error_count += 1

    async def _dispatch(self, event: BaseEvent) -> None:
        """Вызов всех подписчиков для данного типа события."""
        self._event_count += 1
        callbacks = self._subscribers.get(event.event_type, [])

        if not callbacks:
            logger.debug("No subscribers for %s", event.event_type.value)
            return

        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
            except Exception as e:
                logger.error(
                    "Error in subscriber %s for %s: %s",
                    callback.__name__, event.event_type.value, e,
                    exc_info=True,
                )
                self._error_count += 1


# ═════════════════════════════════════════════════════════════════════════════
# Singleton для глобального доступа
# ═════════════════════════════════════════════════════════════════════════════

_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Возвращает глобальный EventBus (singleton)."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Сброс глобального EventBus (для тестов)."""
    global _global_bus
    _global_bus = None
