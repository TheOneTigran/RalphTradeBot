"""
signal_filter.py — Финальный фильтр сигналов перед экзекьюшеном.

Берет Top-гипотезу, фичи из Confluence и выносит окончательный вердикт (SignalGeneratedEvent)
"""
from typing import Any, Dict, Optional

from src.core.config import CONFIDENCE_THRESHOLD
from src.events.models import SignalGeneratedEvent
from src.wave_engine.hypothesis_dag import WaveHypothesis


class SignalFilter:
    """Принимает решение о публикации торгового сигнала."""

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = threshold

    def evaluate(
        self,
        symbol: str,
        hyp: WaveHypothesis, 
        features: Dict[str, float], 
        probability: float
    ) -> Optional[SignalGeneratedEvent]:
        """
        Проверяет, превышает ли P(setup) порог уверенности.
        Если да, формирует сигнальный ивент с расчетом SL/TP.
        """
        if probability < self.threshold:
            return None

        # Гипотеза прошла все фильтры и ML скоринг. Готовим сигнал на вход.
        
        # SL рассчитывается исходя из абсолютного правила (начало текущей волны)
        invalidation_level = 0.0
        if len(hyp.points) > 0:
            if hyp.points[0].is_high == hyp.is_bullish:
                invalidation_level = hyp.points[0].price
            elif len(hyp.points) > 1:
                invalidation_level = hyp.points[1].price

        # Приблизительные TP цели по фибоначчи
        last_price = hyp.points[-1].price if len(hyp.points) > 0 else 0.0
        swing = abs(hyp.points[-2].price - hyp.points[-1].price) if len(hyp.points) > 1 else 0.0
        
        target_1 = last_price + (swing * 0.618) if hyp.is_bullish else last_price - (swing * 0.618)
        target_2 = last_price + (swing * 1.618) if hyp.is_bullish else last_price - (swing * 1.618)

        # Зона входа: агрессивная (с текущих до 38.2% отката)
        entry_1 = last_price
        entry_2 = last_price - (swing * 0.382) if hyp.is_bullish else last_price + (swing * 0.382)

        return SignalGeneratedEvent(
            hypothesis_id=hyp.id,
            symbol=symbol,
            direction="LONG" if hyp.is_bullish else "SHORT",
            probability_score=probability,
            entry_zone=[min(entry_1, entry_2), max(entry_1, entry_2)],
            invalidation_stop=invalidation_level,
            take_profit_targets=[target_1, target_2],
            trend_degree=hyp.degree.value,
            current_wave_hypothesis=hyp.pattern_type.value,
            confluence_triggers=features
        )
