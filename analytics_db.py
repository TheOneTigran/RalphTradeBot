"""
analytics_db.py — Модуль SQLite базы данных аналитики для RalphTradeBot.

Хранит:
1. `signals` — полную информацию о каждом обнаруженном 5-волновом импульсе:
   геометрия волн (W0-W5), RSI-дивергенция, длительность, оценки Algo и Vision,
   торговый план (Entry, SL, TP1-TP4, R:R).
2. `signal_outcomes` — отслеживание реальной отработки сигнала:
   достижение целей TP1-TP4, срабатывание SL, перенос в безубыток (+0.1%),
   максимальное благоприятное (MFE) и неблагоприятное (MAE) движение цены,
   число баров до закрытия и итоговый PnL.
3. `scanner_health` — диагностические метрики каждого цикла сканера.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import ANALYTICS_DB_PATH

logger = logging.getLogger("RalphAnalyticsDB")


def get_connection(db_path: Path | str = ANALYTICS_DB_PATH) -> sqlite3.Connection:
    """Возвращает оптимизированное соединение с SQLite (WAL-режим для многопоточности)."""
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextlib.contextmanager
def db_session(db_path: Path | str = ANALYTICS_DB_PATH):
    """Контекстный менеджер для безопасных транзакций."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _migrate_schema(conn: sqlite3.Connection):
    """Безопасная автомиграция схемы базы данных."""
    try:
        # Проверяем колонку telegram_msg_id в signals
        cols_sig = [r[1] for r in conn.execute("PRAGMA table_info(signals)").fetchall()]
        if "telegram_msg_id" not in cols_sig:
            conn.execute("ALTER TABLE signals ADD COLUMN telegram_msg_id INTEGER;")

        # Проверяем колонки в signal_outcomes
        cols_out = [r[1] for r in conn.execute("PRAGMA table_info(signal_outcomes)").fetchall()]
        add_cols = [
            ("pos_size_usd", "REAL DEFAULT 0.0"),
            ("pos_size_coins", "REAL DEFAULT 0.0"),
            ("risk_usd", "REAL DEFAULT 10.0"),
            ("net_pnl_usd", "REAL DEFAULT 0.0"),
            ("fail_fast_triggered", "INTEGER DEFAULT 0"),
            ("fail_fast_hit_at", "TEXT"),
            ("notified_tp1", "INTEGER DEFAULT 0"),
            ("notified_tp2", "INTEGER DEFAULT 0"),
            ("notified_closed", "INTEGER DEFAULT 0"),
            ("notified_fail_fast", "INTEGER DEFAULT 0"),
        ]
        for col_name, col_def in add_cols:
            if col_name not in cols_out:
                conn.execute(f"ALTER TABLE signal_outcomes ADD COLUMN {col_name} {col_def};")
    except Exception as e:
        logger.warning(f"Предупреждение при миграции схемы: {e}")


def init_db(db_path: Path | str = ANALYTICS_DB_PATH):
    """Инициализирует таблицы базы данных и индексы."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with db_session(db_path) as conn:
        conn.executescript(
            """
            -- Таблица 1: Сигналы и геометрия импульсов
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                direction TEXT NOT NULL,
                algo_score REAL NOT NULL,
                vision_score INTEGER NOT NULL,
                vision_provider TEXT,
                vision_reason TEXT,
                entry_price REAL NOT NULL,
                sl_price REAL NOT NULL,
                tp1_price REAL NOT NULL,
                tp2_price REAL NOT NULL,
                tp3_price REAL NOT NULL,
                tp4_price REAL NOT NULL,
                rr_ratio REAL NOT NULL,
                impulse_pct REAL NOT NULL,
                sl_pct REAL NOT NULL,
                sent_to_telegram INTEGER DEFAULT 0,
                telegram_msg_id INTEGER,
                w0_price REAL,
                w1_price REAL,
                w2_price REAL,
                w3_price REAL,
                w4_price REAL,
                w5_price REAL,
                wave_direction TEXT,
                origin_div INTEGER DEFAULT 0,
                w3_longest INTEGER DEFAULT 1,
                dur_bars INTEGER,
                dur_hours REAL,
                bar_timestamp INTEGER
            );

            -- Таблица 2: Отработка и статистика сделок
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL UNIQUE,
                checked_at TEXT,
                current_price REAL,
                tp1_hit INTEGER DEFAULT 0,
                tp1_hit_at TEXT,
                tp2_hit INTEGER DEFAULT 0,
                tp2_hit_at TEXT,
                tp3_hit INTEGER DEFAULT 0,
                tp3_hit_at TEXT,
                tp4_hit INTEGER DEFAULT 0,
                tp4_hit_at TEXT,
                sl_hit INTEGER DEFAULT 0,
                sl_hit_at TEXT,
                be_triggered INTEGER DEFAULT 0,
                be_hit_at TEXT,
                fail_fast_triggered INTEGER DEFAULT 0,
                fail_fast_hit_at TEXT,
                status TEXT DEFAULT 'active', -- 'active', 'closed_tp', 'closed_sl', 'closed_be', 'closed_fail_fast', 'expired'
                pos_size_usd REAL DEFAULT 0.0,
                pos_size_coins REAL DEFAULT 0.0,
                risk_usd REAL DEFAULT 10.0,
                final_pnl_pct REAL DEFAULT 0.0,
                net_pnl_usd REAL DEFAULT 0.0,
                notified_tp1 INTEGER DEFAULT 0,
                notified_tp2 INTEGER DEFAULT 0,
                notified_closed INTEGER DEFAULT 0,
                notified_fail_fast INTEGER DEFAULT 0,
                bars_to_first_tp INTEGER DEFAULT 0,
                bars_to_close INTEGER DEFAULT 0,
                max_favorable REAL DEFAULT 0.0,
                max_adverse REAL DEFAULT 0.0,
                FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE CASCADE
            );

            -- Таблица 3: Здоровье и метрики сканера
            CREATE TABLE IF NOT EXISTS scanner_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                pairs_scanned INTEGER NOT NULL,
                candidates_found INTEGER DEFAULT 0,
                signals_confirmed INTEGER DEFAULT 0,
                signals_rejected INTEGER DEFAULT 0,
                scan_duration_sec REAL NOT NULL,
                vision_api_status TEXT DEFAULT 'online',
                errors TEXT DEFAULT ''
            );

            -- Таблица 4: Реестр дедупликации сигналов (защита от сдвига экстремума и повторов)
            CREATE TABLE IF NOT EXISTS signal_dedup (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sig_key TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                direction TEXT NOT NULL,
                bar_timestamp INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                w5_price REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Индексы для быстрой фильтрации и аналитики
            CREATE INDEX IF NOT EXISTS idx_signals_sym_tf ON signals(symbol, interval);
            CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
            CREATE INDEX IF NOT EXISTS idx_signals_bar_ts ON signals(symbol, interval, bar_timestamp);
            CREATE INDEX IF NOT EXISTS idx_outcomes_status ON signal_outcomes(status);
            CREATE INDEX IF NOT EXISTS idx_outcomes_sig_id ON signal_outcomes(signal_id);
            CREATE INDEX IF NOT EXISTS idx_health_ts ON scanner_health(timestamp);
            CREATE INDEX IF NOT EXISTS idx_dedup_sym_tf ON signal_dedup(symbol, interval, direction);
            CREATE INDEX IF NOT EXISTS idx_dedup_bar_ts ON signal_dedup(bar_timestamp);
            """
        )
        _migrate_schema(conn)
    logger.info(f"База данных аналитики инициализирована: {db_path}")


def get_active_trade_for_symbol(
    symbol: str, db_path: Path | str = ANALYTICS_DB_PATH
) -> Optional[Dict[str, Any]]:
    """
    Возвращает активную сделку по указанному символу (status = 'active'), если таковая существует.
    Используется для предотвращения открытия конкурирующих или встречных сделок в трекере.
    """
    query = """
        SELECT
            s.id AS signal_id,
            s.symbol,
            s.interval,
            s.direction,
            s.entry_price,
            s.created_at,
            o.status
        FROM signals s
        JOIN signal_outcomes o ON s.id = o.signal_id
        WHERE s.symbol = ? AND o.status = 'active'
        ORDER BY s.id DESC
        LIMIT 1
    """
    with db_session(db_path) as conn:
        row = conn.execute(query, (symbol,)).fetchone()
        if row:
            return dict(row)
    return None


def is_signal_duplicate_in_db(
    symbol: str,
    interval: str,
    direction: str,
    bar_timestamp: int,
    w5_price: float,
    cooldown_bars: int = 6,
    db_path: Path | str = ANALYTICS_DB_PATH,
) -> Tuple[bool, str]:
    """
    Проверяет в базе данных, не является ли сигнал дубликатом уже сохранённого импульса.
    1. Точное совпадение ключа {symbol}_{interval}_{direction}_{bar_timestamp}.
    2. Сдвиг экстремума W5 на 1-3 свечи (тот же импульс при обновлении свечей).
    3. Недавний импульс в пределах cooldown_bars свечей с ценовым расхождением <= 1.5%.
    """
    tf_seconds_map = {
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    bar_sec = tf_seconds_map.get(interval, 900)
    sig_key = f"{symbol}_{interval}_{direction}_{bar_timestamp}"

    with db_session(db_path) as conn:
        # 1. Точное совпадение по sig_key
        exact = conn.execute(
            "SELECT id FROM signal_dedup WHERE sig_key = ? LIMIT 1", (sig_key,)
        ).fetchone()
        if exact:
            return True, f"exact_key_match ({sig_key})"

        # 2. Поиск по недавним сигналам той же монеты, ТФ и направления
        cutoff_ts = bar_timestamp - (cooldown_bars * bar_sec)
        future_cutoff_ts = bar_timestamp + (cooldown_bars * bar_sec)

        rows = conn.execute(
            """
            SELECT sig_key, bar_timestamp, w5_price, entry_price
            FROM signal_dedup
            WHERE symbol = ? AND interval = ? AND direction = ?
              AND bar_timestamp >= ? AND bar_timestamp <= ?
            ORDER BY bar_timestamp DESC
            """,
            (symbol, interval, direction, cutoff_ts, future_cutoff_ts),
        ).fetchall()

        for r in rows:
            prev_ts = r["bar_timestamp"]
            prev_w5 = r["w5_price"] if r["w5_price"] else r["entry_price"]
            delta_sec = abs(bar_timestamp - prev_ts)
            delta_bars = delta_sec / bar_sec if bar_sec > 0 else 0

            # Сдвиг экстремума W5 на 1-3 бара — 100% тот же импульс
            if delta_bars <= 3.0:
                return True, f"pivot_jitter_shift (delta={delta_bars:.1f} bars, key={r['sig_key']})"

            # В пределах кулдауна с ценовым расхождением <= 1.5%
            if prev_w5 > 0:
                price_diff_pct = abs(w5_price - prev_w5) / prev_w5
                if price_diff_pct <= 0.015:
                    return True, f"same_zone_impulse (delta={delta_bars:.1f} bars, diff={price_diff_pct*100:.2f}%)"

    return False, ""


def load_all_signal_keys(db_path: Path | str = ANALYTICS_DB_PATH) -> Set[str]:
    """
    Загружает полный набор ключей уже зарегистрированных сигналов из БД.
    Обеспечивает синхронизацию in-memory seen_signals при старте или перезапуске бота.
    """
    keys = set()
    try:
        with db_session(db_path) as conn:
            # Из таблицы signal_dedup
            dedup_rows = conn.execute("SELECT sig_key FROM signal_dedup").fetchall()
            for r in dedup_rows:
                keys.add(r["sig_key"])

            # Из таблицы signals
            sig_rows = conn.execute(
                "SELECT symbol, interval, direction, bar_timestamp FROM signals WHERE bar_timestamp IS NOT NULL"
            ).fetchall()
            for r in sig_rows:
                keys.add(f"{r['symbol']}_{r['interval']}_{r['direction']}_{r['bar_timestamp']}")
    except Exception as e:
        logger.warning(f"Ошибка загрузки ключей сигналов из БД: {e}")
    return keys


def save_signal(
    data: Dict[str, Any],
    is_active_trade: bool = True,
    db_path: Path | str = ANALYTICS_DB_PATH,
    telegram_msg_id: Optional[int] = None,
) -> int:
    """
    Сохраняет сигнал в таблицу `signals` и создаёт начальную запись в `signal_outcomes`.
    Если is_active_trade == True: статус в outcomes ставится 'active' (сопровождается трекером).
    Если is_active_trade == False: статус ставится 'info_only' (сигнал сохранён для истории,
    но не открывает дублирующуюся конкурирующую позицию).
    Также регистрирует сигнал в таблице `signal_dedup`.
    Возвращает `signal_id`.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    created_at = data.get("created_at", now_iso)
    bar_ts = data.get("bar_timestamp")
    w5_price = data.get("w5_price", data["entry_price"])
    sig_key = f"{data['symbol']}_{data['interval']}_{data['direction']}_{bar_ts}"
    tg_msg_id = telegram_msg_id or data.get("telegram_msg_id")

    with db_session(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO signals (
                created_at, symbol, interval, direction,
                algo_score, vision_score, vision_provider, vision_reason,
                entry_price, sl_price, tp1_price, tp2_price, tp3_price, tp4_price,
                rr_ratio, impulse_pct, sl_pct, sent_to_telegram, telegram_msg_id,
                w0_price, w1_price, w2_price, w3_price, w4_price, w5_price,
                wave_direction, origin_div, w3_longest, dur_bars, dur_hours,
                bar_timestamp
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?
            )
            """,
            (
                created_at,
                data["symbol"],
                data["interval"],
                data["direction"],
                data.get("algo_score", 0.0),
                data.get("vision_score", 0),
                data.get("vision_provider", "unknown"),
                data.get("vision_reason", ""),
                data["entry_price"],
                data["sl_price"],
                data["tp1_price"],
                data["tp2_price"],
                data["tp3_price"],
                data["tp4_price"],
                data.get("rr_ratio", 0.0),
                data.get("impulse_pct", 0.0),
                data.get("sl_pct", 0.0),
                1 if (data.get("sent_to_telegram") or tg_msg_id) else 0,
                tg_msg_id,
                data.get("w0_price"),
                data.get("w1_price"),
                data.get("w2_price"),
                data.get("w3_price"),
                data.get("w4_price"),
                data.get("w5_price"),
                data.get("wave_direction", ""),
                1 if data.get("origin_div") else 0,
                1 if data.get("w3_longest", True) else 0,
                data.get("dur_bars", 0),
                data.get("dur_hours", 0.0),
                bar_ts,
            ),
        )
        signal_id = cursor.lastrowid

        # Создаем запись в outcomes: 'active' или 'info_only'
        initial_status = "active" if is_active_trade else "info_only"
        pos_usd = float(data.get("pos_size_usd", 0.0))
        pos_coins = float(data.get("pos_size_coins", 0.0))
        risk_u = float(data.get("risk_budget_usd", data.get("risk_usd", 10.0)))

        cursor.execute(
            """
            INSERT INTO signal_outcomes (
                signal_id, checked_at, current_price, status,
                pos_size_usd, pos_size_coins, risk_usd
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (signal_id, now_iso, data["entry_price"], initial_status, pos_usd, pos_coins, risk_u),
        )

        # Регистрируем в signal_dedup
        cursor.execute(
            """
            INSERT OR REPLACE INTO signal_dedup (
                sig_key, symbol, interval, direction, bar_timestamp, entry_price, w5_price, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sig_key,
                data["symbol"],
                data["interval"],
                data["direction"],
                bar_ts or 0,
                data["entry_price"],
                w5_price,
                created_at,
            ),
        )

        return signal_id


def update_signal_outcome(
    signal_id: int, updates: Dict[str, Any], db_path: Path | str = ANALYTICS_DB_PATH
):
    """Обновляет состояние отработки сигнала в `signal_outcomes`."""
    if not updates:
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    updates["checked_at"] = now_iso

    fields = []
    values = []
    for k, v in updates.items():
        fields.append(f"{k} = ?")
        values.append(v)
    values.append(signal_id)

    query = f"UPDATE signal_outcomes SET {', '.join(fields)} WHERE signal_id = ?"
    with db_session(db_path) as conn:
        conn.execute(query, values)


def get_active_signals(db_path: Path | str = ANALYTICS_DB_PATH) -> List[Dict[str, Any]]:
    """
    Возвращает список всех активных сигналов, требующих мониторинга (status = 'active').
    Соединяет таблицу `signals` и `signal_outcomes`.
    """
    query = """
        SELECT
            s.id AS signal_id,
            s.created_at,
            s.symbol,
            s.interval,
            s.direction,
            s.entry_price,
            s.sl_price,
            s.tp1_price,
            s.tp2_price,
            s.tp3_price,
            s.tp4_price,
            s.rr_ratio,
            s.impulse_pct,
            s.sent_to_telegram,
            s.telegram_msg_id,
            s.bar_timestamp,
            o.current_price,
            o.tp1_hit,
            o.tp1_hit_at,
            o.tp2_hit,
            o.tp2_hit_at,
            o.tp3_hit,
            o.tp3_hit_at,
            o.tp4_hit,
            o.tp4_hit_at,
            o.sl_hit,
            o.sl_hit_at,
            o.be_triggered,
            o.be_hit_at,
            o.fail_fast_triggered,
            o.fail_fast_hit_at,
            o.status,
            o.pos_size_usd,
            o.pos_size_coins,
            o.risk_usd,
            o.final_pnl_pct,
            o.net_pnl_usd,
            o.notified_tp1,
            o.notified_tp2,
            o.notified_closed,
            o.notified_fail_fast,
            o.bars_to_first_tp,
            o.bars_to_close,
            o.max_favorable,
            o.max_adverse
        FROM signals s
        JOIN signal_outcomes o ON s.id = o.signal_id
        WHERE o.status = 'active'
        ORDER BY s.id ASC
    """
    with db_session(db_path) as conn:
        rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]



def log_scanner_health(
    iteration: int,
    pairs_scanned: int,
    candidates_found: int,
    signals_confirmed: int,
    signals_rejected: int,
    scan_duration_sec: float,
    vision_api_status: str = "online",
    errors: str = "",
    db_path: Path | str = ANALYTICS_DB_PATH,
):
    """Логирует состояние сканера после итерации."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO scanner_health (
                timestamp, iteration, pairs_scanned, candidates_found,
                signals_confirmed, signals_rejected, scan_duration_sec,
                vision_api_status, errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                iteration,
                pairs_scanned,
                candidates_found,
                signals_confirmed,
                signals_rejected,
                round(scan_duration_sec, 2),
                vision_api_status,
                errors,
            ),
        )


def get_analytics_summary(db_path: Path | str = ANALYTICS_DB_PATH) -> Dict[str, Any]:
    """
    Формирует комплексную аналитическую сводку по всем сигналам, монетам и результатам.
    Подходит для ежедневных отчетов, Telegram-дайджестов и обучения LLM.
    """
    with db_session(db_path) as conn:
        # Общие счётчики
        total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        if total_signals == 0:
            return {
                "total_signals": 0,
                "active_signals": 0,
                "closed_signals": 0,
                "win_rate_pct": 0.0,
                "symbols_count": 0,
            }

        sent_tg = conn.execute("SELECT COUNT(*) FROM signals WHERE sent_to_telegram = 1").fetchone()[0]

        # Статусы отработки
        status_rows = conn.execute(
            """
            SELECT status, COUNT(*) as cnt
            FROM signal_outcomes
            GROUP BY status
            """
        ).fetchall()
        statuses = {r["status"]: r["cnt"] for r in status_rows}

        active = statuses.get("active", 0)
        closed_tp = statuses.get("closed_tp", 0)
        closed_be = statuses.get("closed_be", 0)
        closed_sl = statuses.get("closed_sl", 0)
        expired = statuses.get("expired", 0)
        closed_total = closed_tp + closed_be + closed_sl + expired

        # Win Rate (TP4 + BE считаем безубыточно-прибыльными исходами)
        decided = closed_tp + closed_be + closed_sl
        win_rate = round(((closed_tp + closed_be) / decided * 100), 1) if decided > 0 else 0.0
        full_tp_win_rate = round((closed_tp / decided * 100), 1) if decided > 0 else 0.0

        # Достижения отдельных целей TP1-TP4
        tp_hits = conn.execute(
            """
            SELECT
                SUM(tp1_hit) as tp1,
                SUM(tp2_hit) as tp2,
                SUM(tp3_hit) as tp3,
                SUM(tp4_hit) as tp4,
                SUM(be_triggered) as be,
                SUM(sl_hit) as sl,
                AVG(CASE WHEN tp1_hit = 1 THEN bars_to_first_tp END) as avg_bars_tp1,
                AVG(CASE WHEN status != 'active' THEN bars_to_close END) as avg_bars_close,
                AVG(CASE WHEN status != 'active' THEN final_pnl_pct END) as avg_pnl,
                MAX(max_favorable) as max_mfe,
                AVG(max_favorable) as avg_mfe
            FROM signal_outcomes
            """
        ).fetchone()

        tp1_cnt = tp_hits["tp1"] or 0
        tp2_cnt = tp_hits["tp2"] or 0
        tp3_cnt = tp_hits["tp3"] or 0
        tp4_cnt = tp_hits["tp4"] or 0
        be_cnt = tp_hits["be"] or 0
        sl_cnt = tp_hits["sl"] or 0

        # Анализ по направлениям
        dir_rows = conn.execute(
            """
            SELECT direction, COUNT(*) as cnt
            FROM signals
            GROUP BY direction
            """
        ).fetchall()
        directions = {r["direction"]: r["cnt"] for r in dir_rows}

        # Анализ по таймфреймам
        tf_rows = conn.execute(
            """
            SELECT interval, COUNT(*) as cnt
            FROM signals
            GROUP BY interval
            ORDER BY cnt DESC
            """
        ).fetchall()
        timeframes = {r["interval"]: r["cnt"] for r in tf_rows}

        # Статистика по монетам (ТОП монеты по прибыльности и частоте)
        sym_rows = conn.execute(
            """
            SELECT
                s.symbol,
                COUNT(s.id) as signals_count,
                SUM(o.tp2_hit) as tp2_count,
                SUM(o.tp4_hit) as tp4_count,
                SUM(CASE WHEN o.status IN ('closed_tp', 'closed_be') THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN o.status = 'closed_sl' THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN o.status != 'active' THEN o.final_pnl_pct END) as avg_pnl
            FROM signals s
            JOIN signal_outcomes o ON s.id = o.signal_id
            GROUP BY s.symbol
            ORDER BY signals_count DESC
            LIMIT 10
            """
        ).fetchall()

        top_symbols = []
        for sr in sym_rows:
            dec = (sr["wins"] or 0) + (sr["losses"] or 0)
            sym_wr = round((sr["wins"] / dec * 100), 1) if dec > 0 else 0.0
            top_symbols.append(
                {
                    "symbol": sr["symbol"],
                    "signals": sr["signals_count"],
                    "win_rate": sym_wr,
                    "avg_pnl": round(sr["avg_pnl"] or 0.0, 2),
                    "tp2_count": sr["tp2_count"] or 0,
                    "tp4_count": sr["tp4_count"] or 0,
                }
            )

        # Последняя запись о здоровье сканера
        last_health = conn.execute(
            """
            SELECT * FROM scanner_health
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()

        return {
            "total_signals": total_signals,
            "sent_to_telegram": sent_tg,
            "active_signals": active,
            "closed_signals": closed_total,
            "statuses": {
                "active": active,
                "closed_tp": closed_tp,
                "closed_be": closed_be,
                "closed_sl": closed_sl,
                "expired": expired,
            },
            "win_rate_pct": win_rate,
            "full_tp_win_rate_pct": full_tp_win_rate,
            "tp_hit_rates": {
                "tp1_pct": round(tp1_cnt / total_signals * 100, 1) if total_signals else 0.0,
                "tp2_pct": round(tp2_cnt / total_signals * 100, 1) if total_signals else 0.0,
                "tp3_pct": round(tp3_cnt / total_signals * 100, 1) if total_signals else 0.0,
                "tp4_pct": round(tp4_cnt / total_signals * 100, 1) if total_signals else 0.0,
                "be_pct": round(be_cnt / total_signals * 100, 1) if total_signals else 0.0,
                "sl_pct": round(sl_cnt / total_signals * 100, 1) if total_signals else 0.0,
                "tp1_count": tp1_cnt,
                "tp2_count": tp2_cnt,
                "tp3_count": tp3_cnt,
                "tp4_count": tp4_cnt,
            },
            "execution_metrics": {
                "avg_bars_to_first_tp": round(tp_hits["avg_bars_tp1"] or 0.0, 1),
                "avg_bars_to_close": round(tp_hits["avg_bars_close"] or 0.0, 1),
                "avg_pnl_pct": round(tp_hits["avg_pnl"] or 0.0, 2),
                "max_mfe_pct": round(tp_hits["max_mfe"] or 0.0, 2),
                "avg_mfe_pct": round(tp_hits["avg_mfe"] or 0.0, 2),
            },
            "directions": directions,
            "timeframes": timeframes,
            "top_symbols": top_symbols,
            "last_health": dict(last_health) if last_health else None,
        }


def export_dataset_for_ml(db_path: Path | str = ANALYTICS_DB_PATH) -> pd.DataFrame:
    """
    Экспортирует весь датасет сигналов со всеми метриками и результатами в Pandas DataFrame
    для дальнейшего анализа, построения дашбордов и обучения моделей (CatBoost / LLM).
    """
    query = """
        SELECT
            s.*,
            o.tp1_hit, o.tp2_hit, o.tp3_hit, o.tp4_hit,
            o.sl_hit, o.be_triggered, o.status AS outcome_status,
            o.final_pnl_pct, o.bars_to_first_tp, o.bars_to_close,
            o.max_favorable, o.max_adverse
        FROM signals s
        LEFT JOIN signal_outcomes o ON s.id = o.signal_id
    """
    with db_session(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df
