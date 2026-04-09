"""
liquidity_mapper.py — Детектор пулов ликвидности и свипов.

Рынок ходит от ликвидности к ликвидности.
Волны C и 5 часто заканчиваются свипом старых экстремумов.

Архитектура: Two-Tier Event Model
══════════════════════════════════
  SLOW PATH: Работает на закрытых свечах.
    1. Определяет Untouched Liquidity Pools (экстремумы, не тестированные
       ценой длительное время)
    2. Детектирует Sweep (прокол уровня + быстрый возврат)
    3. Эмитит LiquidityMapUpdatedEvent

  FAST PATH (опционально): 
    Может подписаться на OrderBook очередь для обнаружения лимитных стен
    около liquidity pools (подтверждение институционального интереса).

Контракт для ML (FeatureExtractor):
  liquidity_sweep: 1.0 если на текущем экстремуме произошёл свип
  nearest_pool_distance: расстояние до ближайшего нетронутого пула (% от цены)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks

from src.events.bus import get_event_bus
from src.events.models import EventType, LiquidityMapUpdatedEvent, NewCandleEvent
from src.wave_engine.extremum_finder import compute_atr

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data Models
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class LiquidityPool:
    """
    Пул ликвидности — исторический экстремум, за которым скопились стопы.
    
    price: уровень экстремума
    is_high: True = пул над ценой (стопы шортов), False = под ценой (стопы лонгов)
    formation_index: когда сформировался
    touched: был ли протестирован (но не свипнут)
    swept: был ли полностью снят (прокол + возврат)
    age_candles: сколько свечей назад сформировался (старше = сочнее)
    """
    price: float
    is_high: bool
    formation_index: int
    formation_ts: int = 0
    touched: bool = False
    swept: bool = False
    sweep_index: Optional[int] = None
    age_candles: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# Liquidity Mapper
# ═════════════════════════════════════════════════════════════════════════════

class LiquidityMapper:
    """
    Строит карту ликвидности и детектирует свипы.
    
    Алгоритм:
    1. find_peaks на high[] и -low[] → исторические экстремумы
    2. Каждый экстремум, не протестированный ценой N свечей = Untouched Pool
    3. Если цена прокалывает пул и закрывается обратно = Sweep
    4. Sweep на конечной волне (W5, WC) = сильнейший сигнал разворота
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        min_age_candles: int = 10,
        sweep_tolerance_pct: float = 0.001,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.min_age = min_age_candles       # Минимум свечей для "зрелого" пула
        self.sweep_tol = sweep_tolerance_pct  # Допуск при определении прокола (0.1%)

        self._pools: List[LiquidityPool] = []
        self._candle_count: int = 0

        # Кэш последних закрытых свечей для валидации свипа
        self._recent_candles: List[Dict[str, Any]] = []
        self._max_recent = 500

    # ─── Public API ───────────────────────────────────────────────────────

    def build_from_history(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timestamps: np.ndarray,
    ) -> List[LiquidityPool]:
        """
        Строит начальную карту ликвидности по историческим данным.
        Вызывается один раз при старте (или при подгрузке исторических свечей).
        """
        n = len(high)
        self._candle_count = n

        # ATR для адаптивного prominence
        atr = compute_atr(high, low, close, period=14)
        median_atr = float(np.nanmedian(atr))
        prominence = median_atr * 1.0

        # Пики (resistance pools)
        peak_idx, _ = find_peaks(high, prominence=prominence, distance=5)
        for idx in peak_idx:
            pool = LiquidityPool(
                price=float(high[idx]),
                is_high=True,
                formation_index=int(idx),
                formation_ts=int(timestamps[idx]),
                age_candles=n - int(idx),
            )
            self._pools.append(pool)

        # Впадины (support pools)
        valley_idx, _ = find_peaks(-low, prominence=prominence, distance=5)
        for idx in valley_idx:
            pool = LiquidityPool(
                price=float(low[idx]),
                is_high=False,
                formation_index=int(idx),
                formation_ts=int(timestamps[idx]),
                age_candles=n - int(idx),
            )
            self._pools.append(pool)

        # Проверяем какие пулы уже были протестированы/свипнуты
        self._mark_touched_and_swept(high, low, close)

        active = [p for p in self._pools if not p.swept]
        logger.info(
            "LiquidityMapper built for %s [%s]: %d total pools, %d active (untouched/touched)",
            self.symbol, self.timeframe, len(self._pools), len(active),
        )
        return self._pools

    def on_new_candle(self, candle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обработка новой закрытой свечи.
        Обновляет карту и детектирует свипы.

        Returns:
            Dict с контекстом для ML:
              liquidity_sweep: 1.0 если обнаружен свип, 0.0 иначе
              nearest_pool_distance: расстояние до ближайшего пула (%)
              sweep_direction: "bullish" / "bearish" / None
              pools_nearby: количество активных пулов в радиусе 1%
        """
        self._candle_count += 1

        # Кэш для истории
        self._recent_candles.append(candle)
        if len(self._recent_candles) > self._max_recent:
            self._recent_candles = self._recent_candles[-self._max_recent:]

        curr_high = candle["high"]
        curr_low = candle["low"]
        curr_close = candle["close"]
        curr_open = candle.get("open", curr_close)

        sweep_detected = False
        sweep_direction = None

        for pool in self._pools:
            if pool.swept:
                continue

            pool.age_candles = self._candle_count - pool.formation_index

            # === Sweep Detection ===
            # HIGH pool: цена проколола вверх и закрылась ниже = bearish sweep
            if pool.is_high and not pool.swept:
                if curr_high > pool.price * (1 + self.sweep_tol):
                    # Прокол сверху. Проверяем закрытие ниже → пинбар/поглощение
                    if curr_close < pool.price:
                        pool.swept = True
                        pool.sweep_index = self._candle_count
                        sweep_detected = True
                        sweep_direction = "bearish"
                        logger.info(
                            "SWEEP detected [%s]: HIGH pool %.2f pierced (high=%.2f, close=%.2f) → BEARISH reversal",
                            self.symbol, pool.price, curr_high, curr_close,
                        )
                    else:
                        # Прокол без возврата — пул тронут, но не свипнут (breakout)
                        pool.touched = True

            # LOW pool: цена проколола вниз и закрылась выше = bullish sweep
            if not pool.is_high and not pool.swept:
                if curr_low < pool.price * (1 - self.sweep_tol):
                    if curr_close > pool.price:
                        pool.swept = True
                        pool.sweep_index = self._candle_count
                        sweep_detected = True
                        sweep_direction = "bullish"
                        logger.info(
                            "SWEEP detected [%s]: LOW pool %.2f pierced (low=%.2f, close=%.2f) → BULLISH reversal",
                            self.symbol, pool.price, curr_low, curr_close,
                        )
                    else:
                        pool.touched = True

        # Возможно, текущая свеча сама стала новым экстремумом (новый пул)
        # Проверим в следующих свечах (нужна правая часть), пока просто трекаем

        # Расчет ML-фич
        active_pools = [p for p in self._pools if not p.swept and p.age_candles >= self.min_age]
        nearest_dist = self._nearest_pool_distance(curr_close, active_pools)
        pools_nearby = self._count_pools_nearby(curr_close, active_pools, radius_pct=0.01)

        return {
            "liquidity_sweep": 1.0 if sweep_detected else 0.0,
            "nearest_pool_distance": nearest_dist,
            "sweep_direction": sweep_direction,
            "pools_nearby": float(pools_nearby),
            "active_pools_count": float(len(active_pools)),
        }

    def get_active_pools(self) -> List[LiquidityPool]:
        """Возвращает все активные (не свипнутые) пулы."""
        return [p for p in self._pools if not p.swept]

    def get_context_for_ml(self, current_price: float) -> Dict[str, float]:
        """Контракт для FeatureExtractor."""
        active = self.get_active_pools()
        mature = [p for p in active if p.age_candles >= self.min_age]

        return {
            "liquidity_sweep": 0.0,  # Обновляется в on_new_candle
            "nearest_pool_distance": self._nearest_pool_distance(current_price, mature),
            "active_pools_count": float(len(mature)),
        }

    # ─── EventBus Integration ─────────────────────────────────────────────

    async def handle_new_candle(self, event: NewCandleEvent) -> None:
        """Callback для EventBus."""
        if event.symbol != self.symbol or event.timeframe != self.timeframe:
            return

        candle = {
            "ts": event.ts,
            "open": event.open,
            "high": event.high,
            "low": event.low,
            "close": event.close,
            "volume": event.volume,
        }

        result = self.on_new_candle(candle)

        # Эмитим LiquidityMapUpdatedEvent
        bus = get_event_bus()
        active = self.get_active_pools()
        zones = [
            {
                "price": p.price,
                "type": "resistance" if p.is_high else "support",
                "age_candles": p.age_candles,
                "touched": p.touched,
            }
            for p in active[:20]  # Top-20
        ]

        await bus.publish(LiquidityMapUpdatedEvent(
            symbol=self.symbol,
            liquidity_zones=zones,
        ))

    # ─── Private ──────────────────────────────────────────────────────────

    def _mark_touched_and_swept(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> None:
        """Ретроспективно помечает пулы, которые были тронуты/свипнуты в истории."""
        n = len(high)

        for pool in self._pools:
            form_idx = pool.formation_index

            # Проверяем все свечи ПОСЛЕ формирования пула
            for i in range(form_idx + 1, n):
                if pool.swept:
                    break

                if pool.is_high:
                    # HIGH pool: свеча проколола выше и закрылась ниже?
                    if high[i] > pool.price * (1 + self.sweep_tol):
                        if close[i] < pool.price:
                            pool.swept = True
                            pool.sweep_index = i
                        else:
                            pool.touched = True
                else:
                    # LOW pool: свеча проколола ниже и закрылась выше?
                    if low[i] < pool.price * (1 - self.sweep_tol):
                        if close[i] > pool.price:
                            pool.swept = True
                            pool.sweep_index = i
                        else:
                            pool.touched = True

    @staticmethod
    def _nearest_pool_distance(price: float, pools: List[LiquidityPool]) -> float:
        """Расстояние до ближайшего пула в % от цены."""
        if not pools or price == 0:
            return 1.0  # Далеко (нет пулов)

        min_dist = min(abs(p.price - price) / price for p in pools)
        return float(min_dist)

    @staticmethod
    def _count_pools_nearby(price: float, pools: List[LiquidityPool], radius_pct: float = 0.01) -> int:
        """Количество пулов в радиусе radius_pct от цены."""
        return sum(1 for p in pools if abs(p.price - price) / price <= radius_pct)
