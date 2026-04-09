"""
models.py — Изолированные структуры данных для математического конвейера.

Не использует Pydantic и схемы из core/models.py, чтобы оставаться
независимым от LLM-платформы.
"""

from dataclasses import dataclass, field
from enum import Enum


class Direction(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class DtwPattern:
    """Одиночная находка DTW-сканера."""
    pattern_type: str            # например 'BULLISH_IMPULSE', 'BEARISH_ZIGZAG'
    timeframe: str               # '4h', '1h', '15m'
    window_size: int             # 21, 55, 89, 144, 233
    start_ts: int                # Unix timestamp начала паттерна (мс)
    end_ts: int                  # Unix timestamp окончания паттерна (мс)
    dtw_score: float             # Дистанция (чем меньше, тем точнее, обычно < 0.08)
    pivots: dict[int, float]     # Экстремумы волн: {0: price, 1: price, 2: price...}
    pivots_ts: dict[int, int]    # Таймстемпы экстремумов (для точной проверки времени)

    @property
    def direction(self) -> Direction:
        return Direction.BULLISH if "BULLISH" in self.pattern_type else Direction.BEARISH

    @property
    def is_impulsive(self) -> bool:
        return "IMPULSE" in self.pattern_type or "DIAGONAL" in self.pattern_type

    @property
    def is_corrective(self) -> bool:
        return "ZIGZAG" in self.pattern_type or "FLAT" in self.pattern_type or "TRIANGLE" in self.pattern_type


@dataclass
class AlignedSetup:
    """
    Математическое совмещение макро-контекста и микро-триггера.
    Макро задает тренд, Микро задает завершённый откат против тренда.
    """
    macro_pattern: DtwPattern
    micro_pattern: DtwPattern
    status: str = "ALIGNED"


@dataclass
class DtwTradePlan:
    """Точный торговый план, рассчитанный математическим трейдером."""
    symbol: str
    direction: str              # 'LONG' or 'SHORT'
    
    # Расчетные уровни
    point_c_price: float        # Абсолютное дно/вершина Волны C
    entry_price: float          # Точка входа (пробой Микро-B + ATR отступ)
    entry_tolerance: float      # Зона жадности (+0.5 ATR от точки входа)
    sl_aggressive: float        # Консервативный стоп-лосс (Микро-C + буфер)
    sl_conservative: float      # Жесткий стоп-лосс инвалидации (Макро-1)
    take_profit: float          # Цель (Расширение 1.618 Волны Макро-1, отложенное от Микро-C)
    
    # Временные метки
    creation_time: int          # Время создания плана (timestamp конца микро-паттерна)
    ttl_minutes: int            # Максимальное время ожидания отложенного ордера
    
    # Характеристики
    risk_reward_ratio: float    # RR Ratio (TP - Entry) / (Entry - SL_conservative)
    
    # Метаданные для логов
    macro_context: str          # Описание: "BULLISH_IMPULSE on 4h (window 89)"
    micro_trigger: str          # Описание: "BEARISH_ZIGZAG on 15m (window 55)"
