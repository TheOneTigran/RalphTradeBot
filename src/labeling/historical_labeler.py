"""
historical_labeler.py — Подготовка очереди гипотез для ручной разметки.

Прогоняет Wave Engine по историческим данным,
собирает Top-связки гипотез и помещает их в очередь (DuckDB)
для валидации экспертом через Streamlit интерфейс.
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

import numpy as np

from src.storage.duckdb_store import get_store
from src.wave_engine.extremum_finder import ExtremumFinder
from src.wave_engine.hypothesis_dag import HypothesisDAG

logger = logging.getLogger(__name__)

def setup_queue_table():
    """Создаёт таблицу для очереди разметки, если её нет."""
    store = get_store()
    store.conn.execute("""
        CREATE TABLE IF NOT EXISTS labeling_queue (
            id              VARCHAR PRIMARY KEY,
            symbol          VARCHAR NOT NULL,
            timeframe       VARCHAR NOT NULL,
            pattern_type    VARCHAR NOT NULL,
            score           DOUBLE NOT NULL,
            is_bullish      BOOLEAN NOT NULL,
            wave_points_json VARCHAR NOT NULL,
            features_json   VARCHAR NOT NULL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status          VARCHAR DEFAULT 'pending'  -- 'pending', 'labeled'
        );
    """)

def add_to_queue(symbol: str, timeframe: str, hyp: Any) -> None:
    """Добавляет гипотезу в очередь."""
    store = get_store()
    
    # Конвертируем точки в JSON
    pts = [{"index": p.index, "price": p.price, "timestamp": p.timestamp, "is_high": p.is_high} 
           for p in hyp.points]
           
    store.conn.execute(
        """
        INSERT OR IGNORE INTO labeling_queue 
        (id, symbol, timeframe, pattern_type, score, is_bullish, wave_points_json, features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            hyp.id, symbol, timeframe, hyp.pattern_type.value, hyp.confidence, 
            hyp.is_bullish, json.dumps(pts), json.dumps(hyp.features)
        ]
    )

def generate_historical_queue(
    symbol: str, 
    timeframe: str, 
    start_ts: Optional[int] = None,
    limit: int = 1000
) -> int:
    """
    Генератор очереди.
    Скачивает свечи (OHLCV из БД), находит экстремумы, прогоняет через DAG.
    Самые уверенные гипотезы складывает в DuckDB очередь `labeling_queue`.
    
    Возвращает количество добавленных.
    """
    setup_queue_table()
    store = get_store()
    
    candles = store.get_ohlcv(symbol=symbol, timeframe=timeframe, since_ts=start_ts, limit=limit)
    if not candles:
        logger.warning("No candles found for %s %s", symbol, timeframe)
        return 0
        
    high = np.array([c["high"] for c in candles])
    low = np.array([c["low"] for c in candles])
    close = np.array([c["close"] for c in candles])
    timestamps = np.array([c["ts"] for c in candles])
    
    # atr_mult=3.0 — берём только действительно значимые экстремумы (крупные свинги)
    # fractal_n=5 — требуем 5 баров подтверждения с каждой стороны
    finder = ExtremumFinder(mode='single')
    extrema = finder.find(high, low, close, timestamps, fractal_n=5, atr_mult=3.0)
    
    dag = HypothesisDAG()
    for ext in extrema:
        dag.ingest_extremum(ext)
        
    # Собираем ВСЕ завершенные гипотезы для создания хорошего исторического датасета
    candidates = dag.completed_hypotheses
    
    added = 0
    for h in candidates:
        if h.confidence >= 0.4 and len(h.points) >= 4:
            # --- Amplitude filter ---
            # W1 (от 0 до 1) должна покрывать минимум 3% движения на 1H BTC.
            # Меньше — это внутридневной шум, а не структурная волна.
            p0 = h.points[0].price
            p1 = h.points[1].price
            amplitude_pct = abs(p1 - p0) / p0
            if amplitude_pct < 0.03:
                continue

            # --- Full-span filter ---
            # Весь паттерн от начала до конца должен занимать > 3% ценового диапазона.
            all_prices = [p.price for p in h.points]
            full_range_pct = (max(all_prices) - min(all_prices)) / min(all_prices)
            if full_range_pct < 0.05:
                continue

            add_to_queue(symbol, timeframe, h)
            added += 1
            
    return added

def get_queue_items(limit: int = 1) -> List[Dict]:
    """Получить элементы из очереди для Streamlit."""
    setup_queue_table()
    store = get_store()
    rows = store.conn.execute(
        "SELECT * FROM labeling_queue WHERE status = 'pending' ORDER BY score DESC LIMIT ?",
        [limit]
    ).fetchall()
    
    cols = ["id", "symbol", "timeframe", "pattern_type", "score", "is_bullish", "wave_points_json", "features_json", "created_at", "status"]
    
    res = []
    for r in rows:
        d = dict(zip(cols, r))
        d["wave_points"] = json.loads(d["wave_points_json"])
        d["features"] = json.loads(d["features_json"])
        res.append(d)
        
    return res

def mark_labeled(hyp_id: str) -> None:
    """Пометить как обработанную."""
    store = get_store()
    store.conn.execute("UPDATE labeling_queue SET status = 'labeled' WHERE id = ?", [hyp_id])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = generate_historical_queue("BTCUSDT", "1h", limit=5000)
    print(f"Added {n} hypotheses to queue.")
