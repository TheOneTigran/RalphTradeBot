"""
ml_scorer.py — Обертка над XGBoost/LightGBM для скоринга вероятностей.

Принимает features (dict), возвращает вероятность успеха (P(setup)).
Если модель ещё не обучена (холодный старт), использует эвристический fallback
(взвешенную сумму фичей).
"""
import logging
import os
from typing import Dict, Tuple

import joblib
import numpy as np

from src.core.config import ML_MODEL_PATH

logger = logging.getLogger(__name__)


class MLScorer:
    """XGBoost Probability Scorer с поддержкой Platt Scaling и Fallback режима."""

    def __init__(self, model_path: str = ML_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self) -> None:
        """Пытается загрузить обученную XGBoost модель."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
                logger.info("ML model loaded from %s", self.model_path)
            except Exception as e:
                logger.error("Failed to load ML model: %s", e)
                self.is_loaded = False
        else:
            logger.info("No ML model found at %s. Running in FALLBACK mode.", self.model_path)
            self.is_loaded = False

    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Предсказывает вероятность успешного сетапа (0.0 .. 1.0).
        """
        if self.is_loaded and self.model is not None:
            return self._predict_ml(features)
        else:
            return self._predict_fallback(features)

    def _predict_ml(self, features: Dict[str, float]) -> float:
        import pandas as pd
        
        # XGBoost ожидает 2D массив (или DataFrame), фичи должны быть в том же порядке
        # Для безопасности используем DataFrame, чтобы имена колонок матчились (если модель была обучена на DF)
        df = pd.DataFrame([features])
        
        try:
            # predict_proba возвращает массив [p_class0, p_class1]
            probs = self.model.predict_proba(df)[0]
            if len(probs) >= 2:
                p_success = float(probs[1]) 
            else:
                p_success = float(probs[0])
            
            return p_success
        except Exception as e:
            logger.error("ML Prediction error, falling back: %s", e)
            return self._predict_fallback(features)

    def _predict_fallback(self, features: Dict[str, float]) -> float:
        """
        Эвристическая оценка, если HITL-модель еще не собрана.
        Веса заданы по ТЗ (Rule=40%, Fibo=20%, Liquidity=20%, Volume=20%).
        """
        score = 0.0

        # 1. Wave Confidence (Rule Engine) [Max 0.40]
        # features['wave_confidence'] обычно 0.5 - 0.99
        w_conf = features.get("wave_confidence", 0.0)
        score += w_conf * 0.40

        # 2. Fibonacci Zone [Max 0.20]
        # Если fibo_dist_618 близко к нулю -> бонус
        fibo_dist = min(
            features.get("fibo_dist_382", 1.0),
            features.get("fibo_dist_500", 1.0),
            features.get("fibo_dist_618", 1.0)
        )
        # Чем ближе к 0, тем лучше. 1.0 -> 0 очков, 0.0 -> 0.20 очков
        fibo_score = max(0.0, 1.0 - (fibo_dist * 5.0)) * 0.20
        score += fibo_score

        # 3. Liquidity Sweep [Max 0.20]
        sweep = features.get("liquidity_sweep", 0.0)
        score += sweep * 0.20

        # 4. Cluster Volume Z-Score [Max 0.20]
        # Допустим z-score 2.0 дает максимум
        zscore = features.get("cluster_volume_zscore", 0.0)
        vol_score = min(2.0, zscore) / 2.0 * 0.20
        score += vol_score

        return min(0.99, max(0.01, score))
