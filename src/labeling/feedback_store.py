"""
feedback_store.py — API для работы с базой HITL разметок.

Обеспечивает сохранение решений эксперта (Accept/Reject/Correct) в DuckDB,
а также извлечение очереди гипотез для разметки.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.storage.duckdb_store import get_store

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Обертка над DuckDBStore для HITL-операций."""

    def __init__(self):
        self.store = get_store()

    def submit_label(
        self,
        symbol: str,
        timeframe: str,
        hypothesis_type: str,
        features: Dict[str, float],
        label: int,
        source: str,
        wave_points: List[Dict[str, Any]],
        notes: str = "",
    ) -> str:
        """
        Сохранить ответную реакцию эксперта в историю.
        
        Args:
            label: 1 = Accept (подходит для входа), 0 = Reject (мусор)
            source: 'algorithm' (оригинальная), 'human_corrected' (исправлено руками)
            features: json dictionary с фичами (для обучения ML)
            wave_points: координаты узлов (для отрисовки и валидации)
        """
        return self.store.insert_labeled_setup(
            symbol=symbol,
            timeframe=timeframe,
            hypothesis_type=hypothesis_type,
            features=features,
            label=label,
            source=source,
            wave_points=wave_points,
            notes=notes,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Статистика размеченного датасета."""
        counts = self.store.get_labeled_count()
        accept_count = counts.get("accept", 0)
        reject_count = counts.get("reject", 0)
        total = accept_count + reject_count
        
        return {
            "total_labels": total,
            "accepted": accept_count,
            "rejected": reject_count,
            "accept_rate": (accept_count / total * 100) if total > 0 else 0.0,
        }
        
    def get_dataset(self) -> List[Dict]:
        """Получить весь размеченный датасет для обучения ML."""
        return self.store.get_labeled_setups()
