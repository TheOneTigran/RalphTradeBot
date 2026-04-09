"""
models.py — Pydantic-схемы для обмена данными внутри RalphTradeBot.

Схема передачи данных:
  math_preprocessor  →  LLMContext (JSON)  →  ai_prompt_builder  →  LLM
                                                                        ↓
  user              ←  TradePlan (JSON)   ←  trading_plan_generator
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any


class Vector(BaseModel):
    """
    Математическое представление однонаправленного ценового движения (свинга).
    Вектор задаётся двумя экстремумами, определёнными алгоритмом ATR-Fractal.
    """
    start_price: float = Field(..., description="Начальная цена вектора (основание свинга)")
    end_price: float = Field(..., description="Конечная цена вектора (экстремум свинга)")
    start_time: int = Field(..., description="Unix-timestamp начала вектора (мс)")
    end_time: int = Field(..., description="Unix-timestamp окончания вектора (мс)")

    price_change_percent: float = Field(
        ..., description="Процентное изменение цены от start до end (со знаком)"
    )
    fib_retracement_of_prev: Optional[float] = Field(
        None,
        description=(
            "Глубина коррекции данного вектора к предыдущему в долях Фибоначчи "
            "(0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.0). "
            "None — если нет предыдущего вектора или вектор не является коррекцией."
        ),
    )
    rsi_at_end: float = Field(
        ..., description="Значение RSI (14) на свече, которая формирует end_price"
    )
    rsi_divergence: bool = Field(
        False,
        description=(
            "True — если зафиксирована дивергенция RSI (цена обновляет экстремум, RSI — нет). "
            "Классический сигнал окончания волны 3 или 5."
        ),
    )
    ao_divergence: bool = Field(
        False,
        description=(
            "True — если Awesome Oscillator подтверждает дивергенцию на данном экстремуме. "
            "Дополнительный фильтр ложных разворотов."
        ),
    )
    volume_anomaly: bool = Field(
        False,
        description=(
            "True — если средний объём на данном векторе превышает "
            "2.5 × SMA(Volume, 20). Характерно для импульсных волн 1 и 3."
        ),
    )
    is_bullish: bool = Field(
        ..., description="True — вектор направлен вверх (end_price > start_price)"
    )
    atr_size_ratio: Optional[float] = Field(
        None,
        description=(
            "Размер вектора в единицах ATR: abs(end - start) / ATR. "
            "Позволяет LLM понять масштаб движения относительно текущей волатильности."
        ),
    )
    
    # Orderflow & Liquidity 
    poc_price: Optional[float] = Field(
        None,
        description="Point of Control (POC): Цена с максимальным проторгованным объёмом внутри свинга."
    )
    is_liquidity_sweep: bool = Field(
        False,
        description=(
            "True — если экстремум вектора лишь проколол предыдущий экстремум тенью (свип ликвидности), "
            "но закрытие свечи произошло ниже/выше него. Классический разворотный паттерн."
        )
    )
    cvd_divergence: Optional[str] = Field(
        None,
        description=(
            "Наличие дивергенции по Cumulative Volume Delta (CVD) на экстремуме. "
            "Например, 'Bearish CVD divergence' (цена сделала перехай, а покупки упали)."
        )
    )


class FibLevel(BaseModel):
    """Ключевой расчётный уровень Фибоначчи от последнего значимого колебания."""
    ratio: float = Field(..., description="Коэффициент Фибоначчи (0.382, 0.5, 0.618 и т.д.)")
    price: float = Field(..., description="Цена уровня")
    label: str = Field(..., description="Метка уровня, например: '61.8% Fib retracement'")


class FibCluster(BaseModel):
    """
    Кластер Фибоначчи — зона, где несколько независимых расчётов дают совпадающие уровни.
    Это сильные области поддержки/сопротивления, которые важнее одиночных уровней.
    """
    price: float = Field(..., description="Центральная цена кластера")
    strength: int = Field(..., description="Количество совпавших уровней (чем больше — тем сильнее)")
    levels: List[str] = Field(
        ...,
        description=(
            "Описания совпавших уровней, например: "
            "['1D 0.618 от 50000→80000', '4H 1.618 ext от 60000→75000']"
        ),
    )
    is_support: bool = Field(
        ...,
        description=(
            "True — кластер ниже текущей цены (поддержка), "
            "False — выше (сопротивление)"
        ),
    )
    has_poc: bool = Field(
        False,
        description=(
            "True — если кластер совпадает (intersects) с максимальным объёмным узлом (POC) "
            "одного из векторов. Железобетонный уровень."
        )
    )


class OISnapshot(BaseModel):
    """
    Снимок данных Open Interest (открытый интерес) для анализа рыночного сентимента.
    OI помогает разделять импульсные движения (рост OI) от аномалий (закрытие позиций).
    """
    current_oi: Optional[float] = Field(None, description="Текущее значение OI (контракты)")
    oi_change_pct_24h: Optional[float] = Field(
        None, description="Изменение OI за 24ч в процентах (+ = рост, - = снижение)"
    )
    oi_price_divergence: Optional[bool] = Field(
        None,
        description=(
            "True — расхождение OI и цены: рост цены + падение OI (закрытие шортов/слабость), "
            "или падение цены + рост OI (накопление лонгов/возможный разворот)"
        ),
    )
    funding_rate: Optional[float] = Field(
        None, description="Текущая ставка финансирования в % (показывает перекос лонг/шорт)"
    )


class TimeframeData(BaseModel):
    """
    Данные анализа по одному таймфрейму.
    Включает последовательность векторов (свингов), ключевые уровни и маркет-данные.
    """
    timeframe: str = Field(..., description="Таймфрейм: 1w, 1d, 4h, 1h, 15m, 5m")
    vectors: List[Vector] = Field(
        ...,
        description=(
            "Хронологическая последовательность свинговых векторов (от старого к новому). "
            "Каждый вектор — поочерёдно действующий или коррективный."
        ),
    )
    fib_levels: List[FibLevel] = Field(
        default_factory=list,
        description="Расчётные уровни Фибоначчи от последнего значимого колебания на данном ТФ",
    )
    fib_clusters: List[FibCluster] = Field(
        default_factory=list,
        description=(
            "Кластеры совпадающих уровней Фибоначчи — сильные зоны поддержки/сопротивления. "
            "Передавай приоритетно в описании уровней."
        ),
    )
    current_price: float = Field(..., description="Текущая цена (last close)")
    current_rsi: float = Field(..., description="Текущее значение RSI (14)")
    current_atr: Optional[float] = Field(
        None, description="Текущее значение ATR(14) — мера волатильности"
    )

    # Стакан / ликвидность
    nearest_bid_wall: Optional[float] = Field(
        None, description="Ближайший крупный бид (поддержка в стакане), если собирался"
    )
    nearest_ask_wall: Optional[float] = Field(
        None, description="Ближайший крупный аск (сопротивление в стакане), если собирался"
    )

    # Open Interest
    oi_data: Optional[OISnapshot] = Field(
        None, description="Данные Open Interest и funding rate, если доступны"
    )

    mathematical_wave_state: Optional[Any] = Field(
        None, description="Математически выведенное состояние Волн Эллиотта (от алгоритма)"
    )


class LLMContext(BaseModel):
    """
    Итоговый JSON-контекст, который math_preprocessor передаёт в ai_prompt_builder.
    Структура упорядочена от старшего ТФ к младшему — это критично для фрактального анализа.
    """
    symbol: str = Field(..., description="Торговая пара, например BTCUSDT")
    timeframes: List[TimeframeData] = Field(
        ...,
        description=(
            "Массив данных по таймфреймам. "
            "ОБЯЗАТЕЛЕН порядок: от старшего ТФ (1W) к младшему (5m)."
        ),
    )


# ─── Схема выходного торгового плана ─────────────────────────────────────────

class WaveCoordinate(BaseModel):
    """
    Координаты конкретной волны для отображения на графике.
    """
    timeframe: str = Field(..., description="Таймфрейм (например, 1D, 4H, 1H, 15m)")
    wave_name: str = Field(..., description="Имя волны в строгой нотации (например, (3), [ii], C)")
    start_time: int = Field(0, description="Unix-timestamp начала волны (мс)")
    end_time: int = Field(0, description="Unix-timestamp окончания волны (мс)")
    start_price: float = Field(..., description="Цена начала волны")
    end_price: float = Field(..., description="Цена окончания волны")

class TradePlan(BaseModel):
    """
    Структурированный торговый план, полученный от LLM (V2).
    Поддерживает Multi-Count: вероятностную оценку сценариев.
    """
    wave_count_label: str = Field(..., description="Текущая маркировка волн")
    detailed_logic: str = Field(..., description="Подробная логика счета волн")
    main_scenario: str = Field(..., description="Основной план")
    main_scenario_probability: Optional[int] = Field(
        None,
        description="Вероятность основного сценария в % (0-100). Сумма с alt = 100.",
    )
    alternative_scenario: str = Field(..., description="Альтернативный план")
    alternative_scenario_probability: Optional[int] = Field(
        None,
        description="Вероятность альтернативного сценария в % (0-100).",
    )

    trigger_prices: dict = Field(..., description="Цены подтверждения, входа и отмены")
    trade_params: dict = Field(..., description="Технические параметры сделки")
    
    waves_breakdown: List[WaveCoordinate] = Field(
        default_factory=list,
        description="Координаты всех упомянутых в анализе волн для отрисовки на графике"
    )

    # Поле для трекинга: был ли план скорректирован Критиком
    critic_validated: Optional[bool] = Field(
        None, description="True = план прошёл проверку Критика без критических ошибок"
    )
    critic_warnings: Optional[List[str]] = Field(
        None, description="Предупреждения от Критика (не критичные, но важные)"
    )


class CriticFeedback(BaseModel):
    """
    Результат проверки TradePlan от LLM-Критика (Self-Correction пайплайн).
    """
    is_valid: bool = Field(..., description="True = план корректен по правилам 89WAVES")
    critical_errors: List[str] = Field(
        default_factory=list,
        description="Критические нарушения правил Эллиотта — требуют пересмотра плана",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Предупреждения — не блокируют план, но важны для учёта",
    )
    corrected_wave_label: Optional[str] = Field(
        None, description="Скорректированная маркировка от Критика (если is_valid=False)"
    )
    confidence_boost: Optional[str] = Field(
        None, description="Пояснение валидности (если is_valid=True)"
    )
    hard_errors: Optional[List[str]] = Field(
        None, description="Программные ошибки (от Hard Validator), требующие исправления"
    )
