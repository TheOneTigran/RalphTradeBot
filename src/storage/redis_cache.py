"""
redis_cache.py — Тонкая обёртка над redis-py для кэширования in-memory стейта.

Кэшируемые данные:
  - Текущий стакан (orderbook snapshot) — TTL 5s
  - Последний скоринг гипотезы — TTL 60s
  - Активные гипотезы (top-N) — TTL 300s
  - Последний сигнал — TTL 3600s

Redis опционален: если сервер недоступен, все методы gracefully деградируют
(возвращают None / False), а система продолжает работать без кэша.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from src.core.config import REDIS_URL

logger = logging.getLogger(__name__)

# Пытаемся импортировать redis, но не ломаемся если его нет
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis-py не установлен. Кэширование отключено.")


# TTL-константы (секунды)
TTL_ORDERBOOK = 5
TTL_SCORING = 60
TTL_HYPOTHESES = 300
TTL_SIGNAL = 3600


class RedisCache:
    """
    Кэш-адаптер для Redis.
    Graceful degradation: если Redis недоступен — молча возвращает None.
    """

    def __init__(self, url: Optional[str] = None):
        self._url = url or REDIS_URL
        self._client: Optional[Any] = None
        self._connected = False

    def connect(self) -> bool:
        """Подключение к Redis. Возвращает True при успехе."""
        if not REDIS_AVAILABLE:
            return False

        try:
            self._client = redis.from_url(self._url, decode_responses=True)
            self._client.ping()
            self._connected = True
            logger.info("Redis connected: %s", self._url)
            return True
        except Exception as e:
            logger.warning("Redis недоступен (%s). Кэширование отключено.", e)
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ─── Generic Get/Set ──────────────────────────────────────────────────

    def set_json(self, key: str, data: Any, ttl: int = 60) -> bool:
        """Сохраняет JSON-сериализуемый объект с TTL."""
        if not self._connected:
            return False
        try:
            self._client.setex(key, ttl, json.dumps(data, default=str))
            return True
        except Exception as e:
            logger.debug("Redis set error: %s", e)
            return False

    def get_json(self, key: str) -> Optional[Any]:
        """Извлекает JSON-объект. Возвращает None если ключ не найден или Redis недоступен."""
        if not self._connected:
            return None
        try:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.debug("Redis get error: %s", e)
            return None

    def delete(self, key: str) -> bool:
        """Удаляет ключ."""
        if not self._connected:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception:
            return False

    # ─── Domain-Specific Methods ──────────────────────────────────────────

    def cache_orderbook(self, symbol: str, orderbook: Dict) -> bool:
        """Кэширование снимка стакана."""
        return self.set_json(f"ob:{symbol}", orderbook, TTL_ORDERBOOK)

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Получение кэшированного стакана."""
        return self.get_json(f"ob:{symbol}")

    def cache_scoring(self, hypothesis_id: str, scoring: Dict) -> bool:
        """Кэширование результата скоринга гипотезы."""
        return self.set_json(f"score:{hypothesis_id}", scoring, TTL_SCORING)

    def get_scoring(self, hypothesis_id: str) -> Optional[Dict]:
        """Получение кэшированного скоринга."""
        return self.get_json(f"score:{hypothesis_id}")

    def cache_active_hypotheses(self, symbol: str, hypotheses: list) -> bool:
        """Кэширование списка активных гипотез."""
        return self.set_json(f"hyp:{symbol}", hypotheses, TTL_HYPOTHESES)

    def get_active_hypotheses(self, symbol: str) -> Optional[list]:
        """Получение кэшированных гипотез."""
        return self.get_json(f"hyp:{symbol}")

    def cache_signal(self, symbol: str, signal: Dict) -> bool:
        """Кэширование последнего сигнала."""
        return self.set_json(f"signal:{symbol}", signal, TTL_SIGNAL)

    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Получение кэшированного сигнала."""
        return self.get_json(f"signal:{symbol}")

    def close(self) -> None:
        """Закрытие соединения."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._connected = False


# ═════════════════════════════════════════════════════════════════════════════
# Singleton
# ═════════════════════════════════════════════════════════════════════════════

_cache: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Возвращает глобальный RedisCache (singleton)."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
        _cache.connect()  # Если Redis недоступен — graceful degradation
    return _cache
