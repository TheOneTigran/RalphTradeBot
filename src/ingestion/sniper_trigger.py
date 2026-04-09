"""
sniper_trigger.py — Агрессивный вход внутри формирующейся свечи (Fast Path).

Решает проблему Late Entry: на H1/M15 ожидание закрытия бара после свипа
убивает Risk/Reward. Отскок — в первых минутах.

Архитектура:
═══════════════
  DAG (Slow Path) заранее вычисляет:
    - target_fibo_zone: [min, max] — зона ожидаемого входа
    - invalidation_level: цена, при которой гипотеза мертва

  SniperTrigger (Fast Path) мониторит тики и стреляет когда 3 условия
  выполняются ОДНОВРЕМЕННО:
    1. Цена вошла в Фибо-зону
    2. LiquidityMapper зафиксировал Sweep (прокол уровня)
    3. ClusterBuilder зафиксировал Absorption (тормозящий объём)

  При срабатывании → EarlyWarningSignal → ExecutionAgent начинает набор
  позиции лимитками в зоне прокола.

  Если свеча закрывается за invalidation_level → жёсткий стоп.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.events.bus import get_event_bus
from src.events.models import EventType, SignalGeneratedEvent

logger = logging.getLogger(__name__)


@dataclass
class SniperSetup:
    """Подготовленный сетап от DAG (Slow Path) для Sniper."""
    hypothesis_id: str
    symbol: str
    direction: str              # "LONG" | "SHORT"
    fibo_zone: List[float]      # [min_price, max_price]
    invalidation_level: float
    take_profit_targets: List[float]
    degree: str
    pattern_type: str
    confidence: float
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = 14400  # 4 часа по умолчанию


@dataclass
class SniperState:
    """Текущее состояние трёх триггеров для одного сетапа."""
    in_fibo_zone: bool = False
    liquidity_swept: bool = False
    absorption_detected: bool = False
    last_price: float = 0.0
    triggered: bool = False
    triggered_at: float = 0.0

    @property
    def all_conditions_met(self) -> bool:
        return self.in_fibo_zone and self.liquidity_swept and self.absorption_detected


class SniperTrigger:
    """
    Мониторит Fast Path потоки и стреляет Early Warning Signal
    при одновременном совпадении трёх триггеров.

    Пример использования:
        sniper = SniperTrigger()

        # DAG подготовил сетап после Slow Path анализа
        sniper.arm(SniperSetup(
            hypothesis_id="abc",
            symbol="BTCUSDT",
            direction="LONG",
            fibo_zone=[60000, 61000],
            invalidation_level=59500,
            ...
        ))

        # Fast Path: на каждом тике
        signal = sniper.on_tick(price=60500, absorption=True, sweep=True)
        if signal:
            # Execution Agent начинает набор позиции
    """

    def __init__(self):
        self._armed_setups: Dict[str, SniperSetup] = {}
        self._states: Dict[str, SniperState] = {}
        self._fired_count: int = 0

    def arm(self, setup: SniperSetup) -> None:
        """Подготавливает сетап для мониторинга."""
        self._armed_setups[setup.hypothesis_id] = setup
        self._states[setup.hypothesis_id] = SniperState()
        logger.info(
            "Sniper ARMED: %s %s %s zone=[%.2f, %.2f] inv=%.2f",
            setup.symbol, setup.direction, setup.pattern_type,
            setup.fibo_zone[0], setup.fibo_zone[1], setup.invalidation_level,
        )

    def disarm(self, hypothesis_id: str) -> None:
        """Снимает сетап с мониторинга."""
        self._armed_setups.pop(hypothesis_id, None)
        self._states.pop(hypothesis_id, None)

    def disarm_all(self) -> None:
        """Снимает все сетапы."""
        self._armed_setups.clear()
        self._states.clear()

    def on_tick(
        self,
        symbol: str,
        price: float,
        absorption: bool = False,
        sweep: bool = False,
    ) -> Optional[SignalGeneratedEvent]:
        """
        Вызывается на каждом тике (из Fast Path consumer).

        Args:
            symbol: Торговая пара
            price: Текущая цена
            absorption: ClusterBuilder detected absorption on current accumulation
            sweep: LiquidityMapper detected sweep of a pool

        Returns:
            SignalGeneratedEvent если все 3 триггера активны, иначе None
        """
        now = time.time()
        result = None

        for hyp_id, setup in list(self._armed_setups.items()):
            if setup.symbol != symbol:
                continue

            state = self._states[hyp_id]

            # TTL expired — disarm
            if now - setup.created_at > setup.ttl_seconds:
                logger.debug("Sniper setup %s expired (TTL)", hyp_id[:8])
                self.disarm(hyp_id)
                continue

            # Already triggered — skip
            if state.triggered:
                continue

            state.last_price = price

            # Check invalidation
            if setup.direction == "LONG" and price < setup.invalidation_level:
                logger.info("Sniper %s invalidated: price %.2f < inv %.2f",
                           hyp_id[:8], price, setup.invalidation_level)
                self.disarm(hyp_id)
                continue
            if setup.direction == "SHORT" and price > setup.invalidation_level:
                logger.info("Sniper %s invalidated: price %.2f > inv %.2f",
                           hyp_id[:8], price, setup.invalidation_level)
                self.disarm(hyp_id)
                continue

            # Trigger 1: Price in Fibo zone
            zone_min, zone_max = setup.fibo_zone[0], setup.fibo_zone[1]
            state.in_fibo_zone = zone_min <= price <= zone_max

            # Trigger 2: Liquidity Sweep (sticky within candle)
            if sweep:
                state.liquidity_swept = True

            # Trigger 3: Absorption (sticky within candle)
            if absorption:
                state.absorption_detected = True

            # Check all three
            if state.all_conditions_met and not state.triggered:
                state.triggered = True
                state.triggered_at = now
                self._fired_count += 1

                logger.info(
                    "SNIPER FIRED: %s %s @ %.2f (zone=[%.2f,%.2f], inv=%.2f)",
                    setup.symbol, setup.direction, price,
                    zone_min, zone_max, setup.invalidation_level,
                )

                result = SignalGeneratedEvent(
                    hypothesis_id=setup.hypothesis_id,
                    symbol=setup.symbol,
                    direction=setup.direction,
                    probability_score=setup.confidence,
                    entry_zone=setup.fibo_zone,
                    invalidation_stop=setup.invalidation_level,
                    take_profit_targets=setup.take_profit_targets,
                    trend_degree=setup.degree,
                    current_wave_hypothesis=setup.pattern_type,
                    confluence_triggers={
                        "sniper_trigger": True,
                        "in_fibo_zone": True,
                        "liquidity_swept": True,
                        "absorption_detected": True,
                        "entry_price": price,
                    },
                )

        return result

    def reset_candle_state(self, symbol: str) -> None:
        """
        Сброс sticky-состояний при открытии новой свечи.
        Sweep и Absorption — sticky внутри свечи, сбрасываются при новом баре.
        """
        for hyp_id, setup in self._armed_setups.items():
            if setup.symbol != symbol:
                continue
            state = self._states.get(hyp_id)
            if state and not state.triggered:
                state.liquidity_swept = False
                state.absorption_detected = False

    @property
    def armed_count(self) -> int:
        return len(self._armed_setups)

    @property
    def fired_count(self) -> int:
        return self._fired_count
