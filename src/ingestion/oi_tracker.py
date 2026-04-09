"""
oi_tracker.py — Трекер Open Interest для фильтрации свипов.

Ключевая логика:
  Цена пробивает лой + OI ПАДАЕТ  → Ликвидации (стопы ритейла). Истинный Sweep. 
                                     Вероятность отскока РАСТЁТ.
  Цена пробивает лой + OI РАСТЁТ  → Новые шортисты (истинный пробой).
                                     Ловить ножи нельзя. Sweep ОТМЕНЯЕТСЯ.

Генерирует фичу: oi_divergence_flag для ML-слоя.
  +1.0 = OI падает при движении (подтверждает sweep/разворот)
  -1.0 = OI растёт при движении (подтверждает продолжение тренда)
   0.0 = нет данных / нейтрально
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.core.config import BYBIT_API_KEY, BYBIT_API_SECRET

logger = logging.getLogger(__name__)


class OITracker:
    """
    Периодически тянет Open Interest через REST (OI не всегда доступен по WS).
    Рассчитывает дельту OI для фильтрации свипов.

    Использует ccxt fetch_open_interest_history (Bybit V5).
    Polling каждые N минут (не нужна секундная точность для OI).
    """

    def __init__(self, symbol: str, poll_interval_seconds: float = 300):
        self.symbol = symbol
        self.poll_interval = poll_interval_seconds

        self._exchange = None
        self._oi_history: List[Dict[str, Any]] = []  # [{ts, oi_value}, ...]
        self._last_poll: float = 0
        self._current_oi: float = 0
        self._prev_oi: float = 0
        self._funding_rate: Optional[float] = None

    def _create_exchange(self):
        """Создаёт sync ccxt exchange для REST polling."""
        import ccxt
        self._exchange = ccxt.bybit({
            "apiKey": BYBIT_API_KEY or None,
            "secret": BYBIT_API_SECRET or None,
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })
        return self._exchange

    def poll(self) -> Dict[str, Any]:
        """
        Тянет текущий OI и Funding Rate.
        Вызывать периодически (таймер или перед каждым решением).

        Returns:
            {
                "current_oi": float,
                "oi_change_pct": float,     # % изменение за период
                "oi_delta_direction": str,  # "rising" | "falling" | "flat"
                "funding_rate": float,
                "funding_extreme": float,   # нормализованный 0..1
            }
        """
        if self._exchange is None:
            self._create_exchange()

        now = time.time()
        if now - self._last_poll < self.poll_interval and self._current_oi > 0:
            return self._cached_result()

        self._last_poll = now

        # Fetch OI
        try:
            oi_data = self._exchange.fetch_open_interest_history(
                symbol=self.symbol, timeframe="5m", limit=10
            )
            if oi_data and len(oi_data) >= 2:
                oi_values = [
                    float(entry.get("openInterestAmount") or entry.get("openInterest") or 0)
                    for entry in oi_data
                ]
                oi_values = [v for v in oi_values if v > 0]

                if len(oi_values) >= 2:
                    self._prev_oi = oi_values[-2]
                    self._current_oi = oi_values[-1]
                    self._oi_history.append({
                        "ts": int(now * 1000),
                        "oi": self._current_oi,
                    })
                    # Trim history
                    if len(self._oi_history) > 1000:
                        self._oi_history = self._oi_history[-500:]

        except Exception as e:
            logger.warning("OI fetch failed for %s: %s", self.symbol, e)

        # Fetch Funding Rate
        try:
            funding_data = self._exchange.fetch_funding_rate(self.symbol)
            if funding_data:
                self._funding_rate = float(funding_data.get("fundingRate", 0))
        except Exception:
            pass

        return self._cached_result()

    def evaluate_sweep(self, price_broke_low: bool, price_broke_high: bool) -> float:
        """
        Оценивает OI-дивергенцию для конкретного события свипа.

        Args:
            price_broke_low: цена пробила предыдущий лой
            price_broke_high: цена пробила предыдущий хай

        Returns:
            oi_divergence_flag:
              +1.0 = OI падает при проколе (подтверждает sweep → разворот)
              -1.0 = OI растёт при проколе (подтверждает пробой → продолжение)
               0.0 = нет данных
        """
        if self._current_oi == 0 or self._prev_oi == 0:
            return 0.0

        oi_change_pct = (self._current_oi - self._prev_oi) / self._prev_oi * 100

        if price_broke_low:
            # Пробой лоя
            if oi_change_pct < -0.5:
                # OI падает → ликвидации лонгов (стопы ритейла) → BULLISH sweep
                return 1.0
            elif oi_change_pct > 0.5:
                # OI растёт → новые шорты входят → истинный пробой вниз
                return -1.0

        if price_broke_high:
            # Пробой хая
            if oi_change_pct < -0.5:
                # OI падает → ликвидации шортов → BEARISH sweep
                return 1.0
            elif oi_change_pct > 0.5:
                # OI растёт → новые лонги → истинный пробой вверх
                return -1.0

        return 0.0

    def get_context_for_ml(self) -> Dict[str, float]:
        """Фичи для FeatureExtractor."""
        result = self._cached_result()
        return {
            "oi_divergence_flag": 0.0,  # Обновляется через evaluate_sweep()
            "funding_extreme": result.get("funding_extreme", 0.0),
            "oi_change_pct": result.get("oi_change_pct", 0.0),
        }

    def _cached_result(self) -> Dict[str, Any]:
        oi_change = 0.0
        direction = "flat"

        if self._prev_oi > 0:
            oi_change = (self._current_oi - self._prev_oi) / self._prev_oi * 100
            if oi_change > 0.5:
                direction = "rising"
            elif oi_change < -0.5:
                direction = "falling"

        # Funding extreme: нормализуем абсолютное значение
        # Funding 0.01% = нормально, 0.1% = экстремально
        funding_extreme = 0.0
        if self._funding_rate is not None:
            funding_extreme = min(1.0, abs(self._funding_rate) / 0.001)

        return {
            "current_oi": self._current_oi,
            "oi_change_pct": round(oi_change, 3),
            "oi_delta_direction": direction,
            "funding_rate": self._funding_rate,
            "funding_extreme": funding_extreme,
        }
