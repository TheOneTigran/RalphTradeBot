"""
Wave Strategy — Математический конвейер без LLM

Изолированный модуль для превращения DTW-паттернов в торговые планы.
"""

from .pipeline import run_wave_strategy

__all__ = ["run_wave_strategy"]
