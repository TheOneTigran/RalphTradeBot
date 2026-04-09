"""
cluster_builder.py — Детектор "тормозящих объёмов" (Absorption Volume).

Архитектура: Two-Tier Event Model
══════════════════════════════════
  FAST PATH: Подписывается на trade_queue (тики от WSStreamer).
             Аккумулирует buy/sell объёмы по ценовым тикам внутри ТЕКУЩЕЙ свечи.
             Не эмитит событий — только копит стейт.

  SLOW PATH: При получении NewCandleEvent (закрытие свечи):
             1. flush() → ClusterUpdatedEvent с POC, VAH, VAL, Delta
             2. Считает Cluster_Volume_ZScore (для Feature Extractor)
             3. Детектирует "торможение" (высокий sell volume без слома цены)
             4. Сбрасывает аккумулятор под новую свечу

Контракт для ML:
  cluster_volume_zscore — Z-score объёма текущей свечи vs SMA(vol, 20)
  absorption_detected   — True если обнаружена лимитная стена (торможение)
  volume_delta_ratio    — (buy_vol - sell_vol) / total_vol
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from src.events.bus import get_event_bus
from src.events.models import ClusterUpdatedEvent, EventType, NewCandleEvent
from src.storage.duckdb_store import get_store

logger = logging.getLogger(__name__)


class ClusterBuilder:
    """
    Строит кластерный (Market Profile) анализ из ленты принтов.

    Внутренний стейт:
      _levels: Dict[float, {"buy_vol": float, "sell_vol": float}]
      Накапливается от open до close текущей свечи.

    При flush():
      - POC (Point of Control): цена с максимальным суммарным объёмом
      - VAH/VAL (Value Area High/Low): 70% объёма сконцентрировано тут
      - Delta: суммарный (buy - sell)
      - Absorption detection
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        tick_size: float = 0.01,
        volume_history_len: int = 20,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.tick_size = tick_size  # Гранулярность ценовых уровней (группировка)
        self._volume_history_len = volume_history_len

        # Аккумулятор текущей свечи
        self._levels: Dict[float, Dict[str, float]] = defaultdict(
            lambda: {"buy_vol": 0.0, "sell_vol": 0.0}
        )
        self._current_candle_ts: Optional[int] = None
        self._total_buy: float = 0.0
        self._total_sell: float = 0.0
        self._trade_count: int = 0

        # История объёмов для Z-score (rolling SMA)
        self._volume_history: List[float] = []

        # Running flag
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None

    # ─── Public API ───────────────────────────────────────────────────────

    async def start(self, trade_queue: asyncio.Queue) -> None:
        """Запуск consumer'а, слушающего trade_queue."""
        if self._running:
            return

        self._running = True

        # Подписываемся на NewCandleEvent для flush
        bus = get_event_bus()
        bus.subscribe(EventType.NEW_CANDLE, self._on_candle_close)

        # Запускаем consumer
        self._consumer_task = asyncio.create_task(self._consume_trades(trade_queue))
        logger.info("ClusterBuilder started: %s [%s]", self.symbol, self.timeframe)

    async def stop(self) -> None:
        """Остановка."""
        self._running = False
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

    def ingest_trade(self, trade: Dict[str, Any]) -> None:
        """
        Обработка одного тика (синхронный, для вызова из consumer loop).
        """
        price = trade["price"]
        qty = trade["quantity"]
        is_sell = trade.get("is_buyer_maker", False)

        # Округляем цену до tick_size для группировки
        rounded_price = self._round_price(price)

        if is_sell:
            self._levels[rounded_price]["sell_vol"] += qty
            self._total_sell += qty
        else:
            self._levels[rounded_price]["buy_vol"] += qty
            self._total_buy += qty

        self._trade_count += 1

    def flush(self, candle_ts: int) -> Dict[str, Any]:
        """
        Завершает кластер текущей свечи, возвращает профиль и сбрасывает стейт.

        Returns:
            Dict с ключами: poc_price, vah_price, val_price, total_volume,
            delta, cluster_volume_zscore, absorption_detected, volume_delta_ratio
        """
        if not self._levels:
            return self._empty_result()

        # Сортируем уровни по цене
        sorted_levels = sorted(self._levels.items(), key=lambda x: x[0])

        # Считаем POC (Point of Control) — цена с max объёмом
        poc_price = 0.0
        poc_volume = 0.0
        total_volume = 0.0
        level_volumes: List[tuple] = []  # (price, total_vol)

        for price, vol_dict in sorted_levels:
            lvl_vol = vol_dict["buy_vol"] + vol_dict["sell_vol"]
            total_volume += lvl_vol
            level_volumes.append((price, lvl_vol))
            if lvl_vol > poc_volume:
                poc_volume = lvl_vol
                poc_price = price

        # Value Area: 70% объёма вокруг POC
        vah_price, val_price = self._calculate_value_area(level_volumes, poc_price, total_volume)

        # Delta
        delta = self._total_buy - self._total_sell
        delta_ratio = delta / total_volume if total_volume > 0 else 0.0

        # Z-score объёма vs history
        self._volume_history.append(total_volume)
        if len(self._volume_history) > self._volume_history_len:
            self._volume_history = self._volume_history[-self._volume_history_len:]

        zscore = self._calc_zscore(total_volume)

        # Absorption Detection — "тормозящий объём"
        absorption = self._detect_absorption(sorted_levels)

        # Levels для сохранения
        levels_dict = {str(p): v["buy_vol"] + v["sell_vol"] for p, v in sorted_levels}

        result = {
            "candle_ts": candle_ts,
            "poc_price": poc_price,
            "vah_price": vah_price,
            "val_price": val_price,
            "total_volume": total_volume,
            "delta": delta,
            "delta_ratio": delta_ratio,
            "cluster_volume_zscore": zscore,
            "absorption_detected": absorption,
            "trade_count": self._trade_count,
            "levels": levels_dict,
        }

        # Сохраняем в DuckDB
        try:
            store = get_store()
            store.upsert_cluster_profile(
                symbol=self.symbol,
                timeframe=self.timeframe,
                candle_ts=candle_ts,
                poc_price=poc_price,
                vah_price=vah_price,
                val_price=val_price,
                total_volume=total_volume,
                delta=delta,
                levels=levels_dict,
            )
        except Exception as e:
            logger.warning("Failed to persist cluster profile: %s", e)

        # Сброс аккумулятора
        self._reset()

        return result

    def get_context_for_ml(self) -> Dict[str, float]:
        """
        Возвращает фичи для Feature Extractor (контракт для ML layer).
        Вызывается ПОСЛЕ flush() последней свечи.
        """
        latest = self._volume_history[-1] if self._volume_history else 0.0
        zscore = self._calc_zscore(latest)

        return {
            "cluster_volume_zscore": zscore,
            "volume_delta_ratio": 0.0,  # Обновится после следующего flush
        }

    # ─── Private ──────────────────────────────────────────────────────────

    async def _consume_trades(self, trade_queue: asyncio.Queue) -> None:
        """Бесконечный цикл: забираем тики из очереди и инжестим."""
        while self._running:
            try:
                trade = await asyncio.wait_for(trade_queue.get(), timeout=1.0)
                self.ingest_trade(trade)
                trade_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _on_candle_close(self, event: NewCandleEvent) -> None:
        """
        Callback при закрытии свечи.
        Только для нашего символа и таймфрейма.
        """
        if event.symbol != self.symbol or event.timeframe != self.timeframe:
            return

        result = self.flush(event.ts)

        # Эмитим ClusterUpdatedEvent (SLOW PATH)
        bus = get_event_bus()
        await bus.publish(ClusterUpdatedEvent(
            symbol=self.symbol,
            timeframe=self.timeframe,
            candle_ts=event.ts,
            poc_price=result["poc_price"],
            vah_price=result["vah_price"],
            val_price=result["val_price"],
            total_volume=result["total_volume"],
            delta=result["delta"],
            levels={str(k): v for k, v in result.get("levels", {}).items()},
        ))

        logger.info(
            "Cluster [%s %s]: POC=%.2f, Delta=%.2f, ZScore=%.2f, Absorption=%s",
            self.symbol, self.timeframe,
            result["poc_price"], result["delta"],
            result["cluster_volume_zscore"], result["absorption_detected"],
        )

    def _round_price(self, price: float) -> float:
        """Округление цены до tick_size."""
        if self.tick_size <= 0:
            return price
        return round(round(price / self.tick_size) * self.tick_size, 8)

    def _calculate_value_area(
        self,
        level_volumes: List[tuple],
        poc_price: float,
        total_volume: float,
    ) -> tuple[float, float]:
        """
        Value Area: зона, содержащая 70% всего объёма, расширяясь от POC.
        """
        if not level_volumes or total_volume == 0:
            return poc_price, poc_price

        target = total_volume * 0.70
        accumulated = 0.0

        # Находим индекс POC
        poc_idx = 0
        for i, (p, v) in enumerate(level_volumes):
            if p == poc_price:
                poc_idx = i
                accumulated = v
                break

        low_idx = poc_idx
        high_idx = poc_idx

        while accumulated < target and (low_idx > 0 or high_idx < len(level_volumes) - 1):
            # Берём строку сверху или снизу — где объём больше
            up_vol = level_volumes[high_idx + 1][1] if high_idx < len(level_volumes) - 1 else 0
            down_vol = level_volumes[low_idx - 1][1] if low_idx > 0 else 0

            if up_vol >= down_vol and high_idx < len(level_volumes) - 1:
                high_idx += 1
                accumulated += up_vol
            elif low_idx > 0:
                low_idx -= 1
                accumulated += down_vol
            else:
                break

        vah = level_volumes[high_idx][0]
        val = level_volumes[low_idx][0]
        return vah, val

    def _calc_zscore(self, current_volume: float) -> float:
        """Z-score текущего объёма vs скользящее окно."""
        if len(self._volume_history) < 3:
            return 0.0

        arr = np.array(self._volume_history)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return 0.0

        return float((current_volume - mean) / std)

    def _detect_absorption(self, sorted_levels: List[tuple]) -> bool:
        """
        Детекция "тормозящего объёма" (Absorption / Stopping Volume).

        Паттерн: в нижних N ценовых уровнях (лои свечи) — аномально высокий
        sell volume, но цена закрылась выше. Значит, лимитные ордера
        поглотили рыночные продажи.

        Аналогично для верхних уровней (хаи) — высокий buy volume,
        цена закрылась ниже.
        """
        if len(sorted_levels) < 5:
            return False

        n_edge = max(3, len(sorted_levels) // 5)  # Крайние 20% уровней

        # Нижние уровни: если sell_vol > buy_vol * 2 → поглощение продаж
        bottom_levels = sorted_levels[:n_edge]
        bottom_sell = sum(v["sell_vol"] for _, v in bottom_levels)
        bottom_buy = sum(v["buy_vol"] for _, v in bottom_levels)

        # Верхние уровни: если buy_vol > sell_vol * 2 → поглощение покупок
        top_levels = sorted_levels[-n_edge:]
        top_buy = sum(v["buy_vol"] for _, v in top_levels)
        top_sell = sum(v["sell_vol"] for _, v in top_levels)

        # Общий объём для threshold
        total = self._total_buy + self._total_sell
        if total == 0:
            return False

        # Absorption detected if edge zone has 30%+ of total volume AND
        # one side dominates 2:1
        edge_ratio = (bottom_sell + bottom_buy) / total
        if edge_ratio > 0.30 and bottom_sell > bottom_buy * 2:
            return True  # Bullish absorption (sell stopped at lows)

        edge_ratio_top = (top_buy + top_sell) / total
        if edge_ratio_top > 0.30 and top_buy > top_sell * 2:
            return True  # Bearish absorption (buy stopped at highs)

        return False

    def _reset(self) -> None:
        """Сброс аккумулятора."""
        self._levels.clear()
        self._total_buy = 0.0
        self._total_sell = 0.0
        self._trade_count = 0

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "candle_ts": 0,
            "poc_price": 0.0,
            "vah_price": 0.0,
            "val_price": 0.0,
            "total_volume": 0.0,
            "delta": 0.0,
            "delta_ratio": 0.0,
            "cluster_volume_zscore": 0.0,
            "absorption_detected": False,
            "trade_count": 0,
            "levels": {},
        }
