"""
feature_extractor.py — Подготовка фич для ML слоя.

Преобразует WaveHypothesis и контекстные данные (Кластеры, Ликвидность, Индикаторы)
в плоский вектор признаков (dict / pandas DataFrame) для скармливания в XGBoost.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from src.wave_engine.hypothesis_dag import WaveHypothesis

logger = logging.getLogger(__name__)

# Допустимые уровни коррекции Фибоначчи
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]


class FeatureExtractor:
    """Извлечение табличных фич из волновой разметки и внешнего рыночного контекста."""

    @staticmethod
    def extract_features(
        hyp: WaveHypothesis,
        market_context: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """
        Извлекает ~15 фич для XGBoost.
        
        Args:
            hyp: Волновая гипотеза
            market_context: Доп. данные (z-score объема, funding rate, RSI divergency)
        """
        if market_context is None:
            market_context = {}

        feats = {}

        # 1. Характеристики самой волны (из Scoring Engine)
        feats["wave_confidence"] = hyp.confidence
        feats["nodes_count"] = float(len(hyp.points))
        
        # Инкорпорируем фичи подсчитанные в WaveScoring (например alternation_depth)
        for k, v in hyp.features.items():
            if isinstance(v, bool):
                feats[f"hyp_{k}"] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                feats[f"hyp_{k}"] = float(v)

        # 2. Фибоначчи расстояния
        if len(hyp.points) >= 3:
            # Для простоты: дистанция последней точки от ближайшего ожидаемого Фибо
            # Например, завершение W4 (5 точек). W4 откатывается от W3.
            p0 = hyp.points[-3].price
            p1 = hyp.points[-2].price
            p2 = hyp.points[-1].price
            
            swing = abs(p1 - p0)
            retrace = abs(p2 - p1)
            ratio = (retrace / swing) if swing > 0 else 0.0
            
            # Расстояние до ключевых уровней
            feats["fibo_dist_382"] = abs(ratio - 0.382)
            feats["fibo_dist_500"] = abs(ratio - 0.500)
            feats["fibo_dist_618"] = abs(ratio - 0.618)
        else:
            feats["fibo_dist_382"] = 1.0
            feats["fibo_dist_500"] = 1.0
            feats["fibo_dist_618"] = 1.0

        # 3. Дополнительные технические фичи из контекста
        # (Они будут приходить через Event Bus от Ingestion Pipeline)
        
        # Z-Score аномального объема (тормозящий объем на кластере)
        feats["cluster_volume_zscore"] = float(market_context.get("cluster_volume_zscore", 0.0))
        
        # Снят ли исторический экстремум на этом пике?
        feats["liquidity_sweep"] = float(market_context.get("liquidity_sweep", 0.0))
        
        # RSI дивергенция
        feats["rsi_divergence"] = float(market_context.get("rsi_divergence", 0.0))
        
        # Дельта (объем покупок минус объем продаж) в последнем кластере
        feats["volume_delta_ratio"] = float(market_context.get("volume_delta_ratio", 0.0))

        # Funding rate extreme (если рынок сильно перегрет шортами/лонгами)
        feats["funding_extreme"] = float(market_context.get("funding_extreme", 0.0))
        
        # ATR нормализация (размер последнего свинга в ATR)
        feats["move_in_atr"] = float(market_context.get("move_in_atr", 1.0))

        # 4. Multi-Timeframe Sub-wave Validation (MTF)
        # Проверка структуры Волны 1 на младшем ТФ
        feats["mtf_w1_subwaves_valid"] = 1.0  # Default assumption
        if len(hyp.points) >= 2:
            try:
                from src.storage.duckdb_store import get_store
                from src.wave_engine.extremum_finder import ExtremumFinder
                import numpy as np

                store = get_store()
                w0_ts = hyp.points[0].timestamp
                w1_ts = hyp.points[1].timestamp
                
                # Fetch M15 between W0 and W1
                # Usually we subtract a little buffer before W0 and after W1.
                sub_candles = store.get_ohlcv(hyp.symbol if hasattr(hyp, 'symbol') else "BTCUSDT", "15m", since_ts=w0_ts, limit=500)
                # Filter strictly between W0 and W1
                sub_candles = [c for c in sub_candles if w0_ts <= c["ts"] <= w1_ts]
                
                if len(sub_candles) > 10:
                    h = np.array([c["high"] for c in sub_candles])
                    l = np.array([c["low"] for c in sub_candles])
                    c_arr = np.array([c["close"] for c in sub_candles])
                    t = np.array([c["ts"] for c in sub_candles])
                    
                    finder = ExtremumFinder(mode="single")
                    sub_extrema = finder.find(h, l, c_arr, t, fractal_n=2, atr_mult=1.0)
                    
                    # Пятиволновка требует 6 экстремумов (0,1,2,3,4,5)
                    # Если экстремумов меньше, это скорее тройка (A-B-C)
                    if len(sub_extrema) < 6:
                        feats["mtf_w1_subwaves_valid"] = 0.0
            except Exception as e:
                logger.debug("MTF Validation skipped/failed: %s", e)

        # Возвращаем плоский словарь флоатов
        return feats
