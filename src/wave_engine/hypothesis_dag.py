"""
hypothesis_dag.py — Ядро Wave Engine. Directed Acyclic Graph (DAG) гипотез.

При каждом новом экстремуме DAG ветвится:
1. Продление существующих гипотез (добавление точки)
2. Зарождение новых гипотез из этой точки
3. Pruning: отсечение ветвей, нарушивших абсолютные правила (Rule Engine)
4. Scoring: оценка оставшихся ветвей (Scoring Engine)
"""
from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.wave_engine.extremum_finder import Extremum
from src.wave_engine.rule_engine import ElliottRuleEngine
from src.wave_engine.scoring import WaveScoring

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Models
# ═════════════════════════════════════════════════════════════════════════════

class PatternType(str, Enum):
    """Типы волновых паттернов по Эллиотту."""
    IMPULSE = "IMPULSE"        # 5 волн (1-2-3-4-5)
    DIAGONAL = "DIAGONAL"      # 5 волн клин
    ZIGZAG = "ZIGZAG"          # 3 волны (A-B-C) 5-3-5
    FLAT = "FLAT"              # 3 волны (A-B-C) 3-3-5
    TRIANGLE = "TRIANGLE"      # 5 волн сходящиеся (A-B-C-D-E)
    WXY = "WXY"                # Сложная коррекция


class WaveDegree(str, Enum):
    """Степени волн (таймфреймы)."""
    GRAND_SUPERCYCLE = "GRAND_SUPERCYCLE"   # Multi-year
    SUPERCYCLE = "SUPERCYCLE"               # 1w
    CYCLE = "CYCLE"                         # 1d
    PRIMARY = "PRIMARY"                     # 4h
    INTERMEDIATE = "INTERMEDIATE"           # 1h
    MINOR = "MINOR"                         # 15m
    MINUTE = "MINUTE"                       # 5m


@dataclass
class WaveHypothesis:
    """Один узел (состояние) в графе гипотез."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    pattern_type: PatternType = PatternType.IMPULSE
    degree: WaveDegree = WaveDegree.PRIMARY
    is_bullish: bool = True
    
    points: List[Extremum] = field(default_factory=list)
    confidence: float = 0.5  # Базовая 50%
    
    is_invalidated: bool = False
    invalidation_reason: str = ""
    is_completed: bool = False
    
    features: Dict[str, float] = field(default_factory=dict)
    
    def clone(self) -> WaveHypothesis:
        """Deep copy для ветвления."""
        new_hyp = copy.deepcopy(self)
        new_hyp.id = str(uuid.uuid4())
        new_hyp.parent_id = self.id
        return new_hyp

    def __repr__(self) -> str:
        status = "INV" if self.is_invalidated else ("DONE" if self.is_completed else "PEND")
        direction = "BULL" if self.is_bullish else "BEAR"
        p_len = len(self.points)
        return (f"Hypothesis({status} {direction} {self.pattern_type.value} "
                f"[{p_len} pts] Conf={self.confidence:.2f})")


# ═════════════════════════════════════════════════════════════════════════════
# DAG Engine
# ═════════════════════════════════════════════════════════════════════════════

class HypothesisDAG:
    """
    Граф волновых гипотез.
    Хранит активные ветви, занимается pruning'ом и роутингом оценок.
    """

    def __init__(self, degree_override: Optional[WaveDegree] = None):
        self.active_hypotheses: List[WaveHypothesis] = []
        self.completed_hypotheses: List[WaveHypothesis] = []
        self.invalidated_hypotheses: List[WaveHypothesis] = []
        self._degree = degree_override or WaveDegree.PRIMARY

    def get_top_hypotheses(self, limit: int = 3) -> List[WaveHypothesis]:
        """Возвращает лучшие гипотезы по уверенности."""
        valid = [h for h in self.active_hypotheses if not h.is_invalidated]
        return sorted(valid, key=lambda h: h.confidence, reverse=True)[:limit]

    def ingest_extremum(self, ext: Extremum) -> None:
        """
        Главный метод. Подаём новый экстремум в DAG.
        """
        logger.debug("Ingesting extremum: %s", ext)
        new_hypotheses: List[WaveHypothesis] = []

        # 1. Продление существующих гипотез
        for hyp in self.active_hypotheses:
            if hyp.is_completed or hyp.is_invalidated:
                continue

            # Проверяем на чередование направления:
            # Следующая точка должна быть противоположной (пик-впадина-пик)
            if len(hyp.points) > 0:
                last_ext = hyp.points[-1]
                if last_ext.is_high == ext.is_high:
                    # Направление то же. Два варианта:
                    # А) Это обновление текущего экстремума (перехай/перелой)
                    if (ext.is_high and ext.price > last_ext.price) or (not ext.is_high and ext.price < last_ext.price):
                        hyp.points[-1] = ext
                        new_hypotheses.append(hyp)
                    # Б) Игнорируем (это внутренний откат)
                    continue
            
            # Ветвление: создаём копию с добавленной новой точкой
            branch = hyp.clone()
            branch.points.append(ext)
            new_hypotheses.append(branch)

            # Родительская ветвь также остается (на случай если этот экстремум - шум, и мы ждем другой)
            new_hypotheses.append(hyp)

        # 2. Создание новых базовых гипотез из этой точки
        # (Каждая точка может быть началом нового паттерна)
        # Допустим, мы рассматриваем Импульс, Зигзаг и Плоскость
        
        # Bullish паттерны (начинаются с Low)
        if not ext.is_high:
            for p_type in [PatternType.IMPULSE, PatternType.DIAGONAL, PatternType.ZIGZAG, PatternType.FLAT]:
                new_hypotheses.append(WaveHypothesis(
                    pattern_type=p_type,
                    is_bullish=True,
                    points=[ext],
                    degree=self._degree
                ))
        
        # Bearish паттерны (начинаются с High)
        if ext.is_high:
            for p_type in [PatternType.IMPULSE, PatternType.DIAGONAL, PatternType.ZIGZAG, PatternType.FLAT]:
                new_hypotheses.append(WaveHypothesis(
                    pattern_type=p_type,
                    is_bullish=False,
                    points=[ext],
                    degree=self._degree
                ))

        # Очистка текущих перед загрузкой новых
        self.active_hypotheses.clear()
        
        # 3. Pruning и Scoring
        self._prune_and_score(new_hypotheses)

    def _prune_and_score(self, candidates: List[WaveHypothesis]) -> None:
        """
        Прогоняет гипотезы через Rule Engine (pruning) и Scoring Engine.
        Отсеянные переносит в invalidated_history.
        """
        # Дедупликация (две гипотезы с одинаковыми точками одного типа)
        unique_cands: Dict[str, WaveHypothesis] = {}
        for c in candidates:
            k = f"{c.pattern_type.value}_{c.is_bullish}_" + ",".join([str(p.index) for p in c.points])
            if k not in unique_cands or c.confidence > unique_cands[k].confidence:
                unique_cands[k] = c

        for hyp in unique_cands.values():
            if len(hyp.points) < 3:
                # Мало точек для инвалидации, пропускаем
                self.active_hypotheses.append(hyp)
                continue

            # Pruning
            is_valid, reason = self._validate(hyp)
            if not is_valid:
                hyp.is_invalidated = True
                hyp.invalidation_reason = reason
                self.invalidated_hypotheses.append(hyp)
                continue

            # Check completion
            self._check_completion(hyp)

            # Scoring
            hyp.confidence = min(0.99, max(0.01, 0.5 + self._score(hyp)))
            
            if hyp.is_completed:
                self.completed_hypotheses.append(hyp)
            else:
                self.active_hypotheses.append(hyp)
                
        # Ограничение размера DAG (оставляем топ-100 активных чтобы не взорвать память)
        self.active_hypotheses.sort(key=lambda h: (len(h.points), h.confidence), reverse=True)
        if len(self.active_hypotheses) > 100:
            self.active_hypotheses = self.active_hypotheses[:100]

    def _validate(self, hyp: WaveHypothesis) -> tuple[bool, str]:
        """Обращение к Rule Engine."""
        if hyp.pattern_type == PatternType.IMPULSE:
            return ElliottRuleEngine.validate_impulse(hyp.points)
        elif hyp.pattern_type == PatternType.DIAGONAL:
            return ElliottRuleEngine.validate_diagonal(hyp.points)
        elif hyp.pattern_type == PatternType.ZIGZAG:
            return ElliottRuleEngine.validate_zigzag(hyp.points)
        elif hyp.pattern_type == PatternType.FLAT:
            return ElliottRuleEngine.validate_flat(hyp.points)
        
        return True, ""

    def _score(self, hyp: WaveHypothesis) -> float:
        """Обращение к Scoring."""
        delta = 0.0
        if hyp.pattern_type == PatternType.IMPULSE:
            val, feats = WaveScoring.score_impulse_guidelines(hyp.points)
            delta += val
            hyp.features.update(feats)
            
            # Добавим фичу "количество сформированных точек" (больше = выше уверенность)
            delta += (len(hyp.points) - 2) * 0.05
            
        return delta

    def _check_completion(self, hyp: WaveHypothesis) -> None:
        """Помечает гипотезу как завершенную, если собрано достаточно точек."""
        if hyp.pattern_type in [PatternType.IMPULSE, PatternType.DIAGONAL] and len(hyp.points) == 6:
            hyp.is_completed = True
        elif hyp.pattern_type in [PatternType.ZIGZAG, PatternType.FLAT] and len(hyp.points) == 4:
            hyp.is_completed = True

    def clear(self) -> None:
        """Полная очистка DAG."""
        self.active_hypotheses.clear()
        self.invalidated_hypotheses.clear()
        self.completed_hypotheses.clear()
