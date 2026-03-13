"""
exceptions.py — Пользовательские исключения RalphTradeBot.
"""


class WaveEngineError(Exception):
    """Базовый класс для всех ошибок RalphTradeBot."""


class DataFetchError(WaveEngineError):
    """Ошибка при получении данных с биржи."""


class PreprocessingError(WaveEngineError):
    """Ошибка при математической обработке данных."""


class LLMError(WaveEngineError):
    """Ошибка при взаимодействии с LLM."""


class PlanParsingError(WaveEngineError):
    """Ошибка при парсинге торгового плана из ответа LLM."""
