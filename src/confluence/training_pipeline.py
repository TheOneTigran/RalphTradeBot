"""
training_pipeline.py — Автоматическое дообучение ML-модели на HITL-разметке.

Забирает labels из DuckDB (Accept=1, Reject=0), 
тренирует XGBoost Classifier с Walk-Forward кросс-валидацией
и сохраняет joblib модель.
"""
import logging
import os
from datetime import datetime
from typing import Dict, List

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.core.config import HITL_MIN_LABELS_FOR_TRAINING, ML_MODEL_PATH
from src.storage.duckdb_store import get_store

logger = logging.getLogger(__name__)


def prepare_dataset() -> pd.DataFrame:
    """Извлекает и форматирует датасет из БД DuckDB."""
    store = get_store()
    raw_data = store.get_labeled_setups()
    
    if not raw_data:
        return pd.DataFrame()

    rows = []
    for item in raw_data:
        row = item["features"].copy()
        row["label"] = item["label"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def train_model() -> bool:
    """
    Запускает пайплайн обучения, если хватает размеченных данных.
    Возвращает True если модель обновлена.
    """
    logger.info("Starting ML Training Pipeline...")
    df = prepare_dataset()
    
    if len(df) < HITL_MIN_LABELS_FOR_TRAINING:
        logger.warning(
            "Not enough data to train ML. Found %d, need %d. Keep labeling via HITL Dashboard.", 
            len(df), HITL_MIN_LABELS_FOR_TRAINING
        )
        return False

    logger.info("Found %d labeled samples. Training XGBoost...", len(df))

    # Выделяем целевую переменную
    y = df["label"]
    X = df.drop(columns=["label"])
    
    # XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )

    # Cross-validation для оценки
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    acc_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc_scores.append(accuracy_score(y_test, preds))
        try:
            auc_scores.append(roc_auc_score(y_test, probs))
        except ValueError:
            # Слишком мало данных одного класса в фолде
            pass

    mean_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0
    mean_auc = sum(auc_scores) / len(auc_scores) if auc_scores else 0
    
    logger.info(f"CV Validation Results - Accuracy: {mean_acc:.3f}, AUC-ROC: {mean_auc:.3f}")

    # Обучаем финальную модель на всех данных
    model.fit(X, y)

    # Сохраняем
    os.makedirs(os.path.dirname(ML_MODEL_PATH), exist_ok=True)
    joblib.dump(model, ML_MODEL_PATH)
    logger.info(f"Model saved to {ML_MODEL_PATH}")

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
