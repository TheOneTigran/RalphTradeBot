"""
duckdb_store.py — Адаптер DuckDB для хранения временных рядов и HITL-разметки.

Embedded columnar OLAP-база. Идеально для:
  - OHLCV свечи (сотни тысяч строк, быстрые агрегации)
  - AggTrades (тики)
  - Кластерные профили (Market Profile per candle)
  - HITL labeled_setups (обучающая выборка для ML)

Zero-config: один файл .duckdb, нет сервера.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb

from src.core.config import DUCKDB_PATH

logger = logging.getLogger(__name__)


class DuckDBStore:
    """
    Единый адаптер для всех операций с DuckDB.
    
    Таблицы:
      - ohlcv: свечные данные по символам и таймфреймам
      - agg_trades: лента принтов
      - cluster_profiles: Market Profile (POC, VAH, VAL, delta)
      - labeled_setups: HITL-разметка для обучения ML
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or DUCKDB_PATH
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    # ─── Connection Management ────────────────────────────────────────────

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Подключение к БД (создаёт файл если не существует)."""
        if self._conn is not None:
            return self._conn

        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        self._conn = duckdb.connect(self._db_path)
        self._create_tables()
        logger.info("DuckDB connected: %s", self._db_path)
        return self._conn

    def close(self) -> None:
        """Закрытие соединения."""
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            return self.connect()
        return self._conn

    # ─── Schema ───────────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        """Создание всех таблиц (idempotent)."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol      VARCHAR NOT NULL,
                timeframe   VARCHAR NOT NULL,
                ts          BIGINT NOT NULL,          -- Unix timestamp (ms)
                open        DOUBLE NOT NULL,
                high        DOUBLE NOT NULL,
                low         DOUBLE NOT NULL,
                close       DOUBLE NOT NULL,
                volume      DOUBLE NOT NULL,
                fetched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, ts)
            );
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS agg_trades (
                symbol          VARCHAR NOT NULL,
                trade_id        BIGINT NOT NULL,
                price           DOUBLE NOT NULL,
                quantity        DOUBLE NOT NULL,
                is_buyer_maker  BOOLEAN NOT NULL,
                trade_time      BIGINT NOT NULL,       -- Unix timestamp (ms)
                fetched_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, trade_id)
            );
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_profiles (
                symbol      VARCHAR NOT NULL,
                timeframe   VARCHAR NOT NULL,
                candle_ts   BIGINT NOT NULL,
                poc_price   DOUBLE NOT NULL,
                vah_price   DOUBLE NOT NULL,
                val_price   DOUBLE NOT NULL,
                total_volume DOUBLE NOT NULL,
                delta       DOUBLE NOT NULL,            -- buy_vol - sell_vol
                levels_json VARCHAR,                    -- JSON: {price: volume, ...}
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, candle_ts)
            );
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS labeled_setups (
                id              VARCHAR PRIMARY KEY,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                symbol          VARCHAR NOT NULL,
                timeframe       VARCHAR NOT NULL,
                hypothesis_type VARCHAR NOT NULL,       -- IMPULSE, ZIGZAG, FLAT, etc.
                features_json   VARCHAR NOT NULL,       -- JSON: {fibo_dist_382: 0.02, ...}
                label           INTEGER NOT NULL,       -- 1 = Accept, 0 = Reject
                source          VARCHAR NOT NULL,       -- 'algorithm' | 'human_corrected'
                wave_points_json VARCHAR NOT NULL,      -- JSON: [{label, price, ts}, ...]
                notes           VARCHAR,
                model_version   VARCHAR                 -- версия модели, на которой обучена
            );
        """)

        logger.debug("DuckDB tables created/verified")

    # ─── OHLCV ────────────────────────────────────────────────────────────

    def upsert_ohlcv(self, symbol: str, timeframe: str, candles: List[Dict]) -> int:
        """
        Вставка/обновление OHLCV свечей.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            candles: Список dict с ключами: ts, open, high, low, close, volume
            
        Returns:
            Количество вставленных записей
        """
        if not candles:
            return 0

        values = [
            (symbol, timeframe, c["ts"], c["open"], c["high"], c["low"], c["close"], c["volume"])
            for c in candles
        ]

        self.conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        return len(values)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ts: Optional[int] = None,
        until_ts: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Получение OHLCV свечей из БД."""
        query = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = ?"
        params: list = [symbol, timeframe]

        if since_ts is not None:
            query += " AND ts >= ?"
            params.append(since_ts)

        if until_ts is not None:
            query += " AND ts <= ?"
            params.append(until_ts)

        query += " ORDER BY ts ASC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [
            {"ts": r[0], "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
            for r in rows
        ]

    def get_ohlcv_count(self, symbol: str, timeframe: str) -> int:
        """Количество свечей в БД для пары/ТФ."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            [symbol, timeframe],
        ).fetchone()
        return result[0] if result else 0

    # ─── AggTrades ────────────────────────────────────────────────────────

    def insert_agg_trades(self, symbol: str, trades: List[Dict]) -> int:
        """Вставка агрегированных трейдов."""
        if not trades:
            return 0

        values = [
            (symbol, t["trade_id"], t["price"], t["quantity"], t["is_buyer_maker"], t["trade_time"])
            for t in trades
        ]

        self.conn.executemany(
            """
            INSERT OR IGNORE INTO agg_trades (symbol, trade_id, price, quantity, is_buyer_maker, trade_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        return len(values)

    def get_trades_in_range(self, symbol: str, start_ts: int, end_ts: int) -> List[Dict]:
        """Получение трейдов в временном диапазоне."""
        rows = self.conn.execute(
            """
            SELECT trade_id, price, quantity, is_buyer_maker, trade_time
            FROM agg_trades
            WHERE symbol = ? AND trade_time >= ? AND trade_time <= ?
            ORDER BY trade_time ASC
            """,
            [symbol, start_ts, end_ts],
        ).fetchall()
        return [
            {"trade_id": r[0], "price": r[1], "quantity": r[2], "is_buyer_maker": r[3], "trade_time": r[4]}
            for r in rows
        ]

    # ─── Cluster Profiles ─────────────────────────────────────────────────

    def upsert_cluster_profile(
        self,
        symbol: str,
        timeframe: str,
        candle_ts: int,
        poc_price: float,
        vah_price: float,
        val_price: float,
        total_volume: float,
        delta: float,
        levels: Optional[Dict[str, float]] = None,
    ) -> None:
        """Вставка/обновление кластерного профиля для свечи."""
        levels_json = json.dumps(levels) if levels else "{}"
        self.conn.execute(
            """
            INSERT OR REPLACE INTO cluster_profiles
            (symbol, timeframe, candle_ts, poc_price, vah_price, val_price, total_volume, delta, levels_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [symbol, timeframe, candle_ts, poc_price, vah_price, val_price, total_volume, delta, levels_json],
        )

    def get_cluster_profiles(self, symbol: str, timeframe: str, since_ts: Optional[int] = None) -> List[Dict]:
        """Получение кластерных профилей."""
        query = "SELECT * FROM cluster_profiles WHERE symbol = ? AND timeframe = ?"
        params: list = [symbol, timeframe]
        if since_ts:
            query += " AND candle_ts >= ?"
            params.append(since_ts)
        query += " ORDER BY candle_ts ASC"

        rows = self.conn.execute(query, params).fetchall()
        columns = ["symbol", "timeframe", "candle_ts", "poc_price", "vah_price", "val_price",
                    "total_volume", "delta", "levels_json", "created_at"]
        return [dict(zip(columns, r)) for r in rows]

    # ─── HITL Labeled Setups ──────────────────────────────────────────────

    def insert_labeled_setup(
        self,
        symbol: str,
        timeframe: str,
        hypothesis_type: str,
        features: Dict[str, float],
        label: int,
        source: str,
        wave_points: List[Dict],
        notes: Optional[str] = None,
    ) -> str:
        """
        Запись размеченного сетапа (HITL).
        
        Returns:
            ID записи (UUID)
        """
        record_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO labeled_setups
            (id, symbol, timeframe, hypothesis_type, features_json, label, source, wave_points_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record_id, symbol, timeframe, hypothesis_type,
                json.dumps(features), label, source,
                json.dumps(wave_points), notes,
            ],
        )
        logger.info("HITL label saved: %s %s %s label=%d source=%s", record_id[:8], symbol, hypothesis_type, label, source)
        return record_id

    def get_labeled_setups(
        self,
        symbol: Optional[str] = None,
        min_date: Optional[str] = None,
        label: Optional[int] = None,
    ) -> List[Dict]:
        """
        Получение размеченных сетапов для обучения ML.
        
        Returns:
            Список dict с распарсенными features и wave_points
        """
        query = "SELECT * FROM labeled_setups WHERE 1=1"
        params: list = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if min_date:
            query += " AND created_at >= ?"
            params.append(min_date)
        if label is not None:
            query += " AND label = ?"
            params.append(label)

        query += " ORDER BY created_at ASC"
        rows = self.conn.execute(query, params).fetchall()
        columns = [
            "id", "created_at", "symbol", "timeframe", "hypothesis_type",
            "features_json", "label", "source", "wave_points_json", "notes", "model_version",
        ]
        results = []
        for r in rows:
            d = dict(zip(columns, r))
            d["features"] = json.loads(d.pop("features_json"))
            d["wave_points"] = json.loads(d.pop("wave_points_json"))
            results.append(d)
        return results

    def get_labeled_count(self) -> Dict[str, int]:
        """Статистика разметки: количество Accept vs Reject."""
        rows = self.conn.execute(
            "SELECT label, COUNT(*) FROM labeled_setups GROUP BY label"
        ).fetchall()
        return {("accept" if r[0] == 1 else "reject"): r[1] for r in rows}

    # ─── Analytics ────────────────────────────────────────────────────────

    def query(self, sql: str, params: Optional[list] = None) -> List[Any]:
        """Произвольный SQL-запрос (для аналитики и бэктестов)."""
        if params:
            return self.conn.execute(sql, params).fetchall()
        return self.conn.execute(sql).fetchall()


# ═════════════════════════════════════════════════════════════════════════════
# Singleton
# ═════════════════════════════════════════════════════════════════════════════

_store: Optional[DuckDBStore] = None


def get_store() -> DuckDBStore:
    """Возвращает глобальный DuckDBStore (singleton)."""
    global _store
    if _store is None:
        _store = DuckDBStore()
        _store.connect()
    return _store
