"""
runtime_settings.py — Динамическое управление параметрами сканера и рисками.

Позволяет в реальном времени изменять через Telegram-бота:
1. Список отслеживаемых монет (добавление, удаление, переключение ON/OFF, сброс к ТОП-12).
2. Рабочие таймфреймы (15m, 1h, 5m, 4h).
3. Размер депозита, бюджет риска ($ или %) и кредитное плечо.
4. Минимальный порог качества ИИ (80%, 85%, 90%).

Все изменения сохраняются в таблице `bot_settings` в SQLite базе данных `ralph_analytics.db`
и мгновенно применяются в очередном цикле сканера без перезапуска контейнера.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ANALYTICS_DB_PATH,
    TOP_12_LEADERS,
    TOP_30_SYMBOLS,
    LIVE_TIMEFRAMES,
    RISK_BUDGET_USD,
    DEFAULT_LEVERAGE,
    TELEGRAM_MIN_SCORE,
)

logger = logging.getLogger("RalphRuntimeSettings")

_settings_lock = threading.Lock()
_cached_config: Optional[Dict[str, Any]] = None
_last_cache_time: float = 0.0
CACHE_TTL_SEC: float = 2.0  # Инвалидация локального кэша каждые 2 сек


def _ensure_settings_table(conn: sqlite3.Connection):
    """Создаёт таблицу настроек, если она ещё не существует."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bot_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )


def get_raw_setting(key: str, default: Any = None, db_path: Path | str = ANALYTICS_DB_PATH) -> Any:
    """Считывает сырое значение настройки из SQLite."""
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        _ensure_settings_table(conn)
        row = conn.execute("SELECT value FROM bot_settings WHERE key = ?", (key,)).fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]
    except Exception as e:
        logger.debug(f"Ошибка чтения настройки {key}: {e}")
    return default


def set_raw_setting(key: str, value: Any, db_path: Path | str = ANALYTICS_DB_PATH) -> bool:
    """Записывает настройку в SQLite."""
    global _cached_config
    val_str = json.dumps(value, ensure_ascii=False)
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        conn = sqlite3.connect(str(db_path), timeout=10.0)
        _ensure_settings_table(conn)
        conn.execute(
            """
            INSERT INTO bot_settings (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, val_str, now_iso),
        )
        conn.commit()
        conn.close()
        with _settings_lock:
            _cached_config = None  # Сброс кэша
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения настройки {key}: {e}")
        return False


def get_runtime_config(db_path: Path | str = ANALYTICS_DB_PATH) -> Dict[str, Any]:
    """
    Возвращает актуальную конфигурацию системы.
    Использует быстрый кэш в памяти с частой автосинхронизацией.
    """
    global _cached_config, _last_cache_time
    import time
    now = time.time()

    with _settings_lock:
        if _cached_config is not None and (now - _last_cache_time) < CACHE_TTL_SEC:
            return dict(_cached_config)

    # Загружаем из базы или выставляем дефолты
    active_symbols = get_raw_setting("active_symbols", None, db_path=db_path)
    if not active_symbols or not isinstance(active_symbols, list):
        active_symbols = list(TOP_12_LEADERS)
        set_raw_setting("active_symbols", active_symbols, db_path=db_path)

    live_timeframes = get_raw_setting("live_timeframes", None, db_path=db_path)
    if not live_timeframes or not isinstance(live_timeframes, list):
        live_timeframes = list(LIVE_TIMEFRAMES)
        set_raw_setting("live_timeframes", live_timeframes, db_path=db_path)

    deposit_usd = float(get_raw_setting("deposit_usd", 1000.0, db_path=db_path))
    risk_budget_usd = float(get_raw_setting("risk_budget_usd", float(RISK_BUDGET_USD), db_path=db_path))
    risk_pct = round((risk_budget_usd / deposit_usd * 100.0), 2) if deposit_usd > 0 else 1.0
    default_leverage = int(get_raw_setting("default_leverage", int(DEFAULT_LEVERAGE), db_path=db_path))
    min_score = int(get_raw_setting("min_score", int(TELEGRAM_MIN_SCORE), db_path=db_path))
    scanner_paused = bool(get_raw_setting("scanner_paused", False, db_path=db_path))

    cfg = {
        "active_symbols": active_symbols,
        "live_timeframes": live_timeframes,
        "deposit_usd": deposit_usd,
        "risk_budget_usd": risk_budget_usd,
        "risk_pct": risk_pct,
        "default_leverage": default_leverage,
        "min_score": min_score,
        "scanner_paused": scanner_paused,
    }

    with _settings_lock:
        _cached_config = cfg
        _last_cache_time = now

    return dict(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Управление монетами
# ═══════════════════════════════════════════════════════════════════════════

def add_symbol(symbol: str, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[bool, str]:
    """Добавляет монету в активный пул сканирования."""
    sym = symbol.strip().upper()
    if not sym.endswith("USDT"):
        sym += "USDT"

    cfg = get_runtime_config(db_path=db_path)
    current_symbols = list(cfg["active_symbols"])

    if sym in current_symbols:
        return False, f"Монета <b>{sym}</b> уже находится в списке мониторинга."

    current_symbols.append(sym)
    set_raw_setting("active_symbols", current_symbols, db_path=db_path)
    logger.info(f"Добавлена новая монета в пул: {sym} (всего: {len(current_symbols)})")
    return True, f"✅ Монета <b>{sym}</b> успешно добавлена в мониторинг!"


def remove_symbol(symbol: str, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[bool, str]:
    """Удаляет монету из пула сканирования."""
    sym = symbol.strip().upper()
    if not sym.endswith("USDT"):
        sym += "USDT"

    cfg = get_runtime_config(db_path=db_path)
    current_symbols = list(cfg["active_symbols"])

    if sym not in current_symbols:
        return False, f"Монета <b>{sym}</b> отсутствует в активном списке."

    if len(current_symbols) <= 1:
        return False, "⚠️ Нельзя удалить последнюю монету. В списке должен оставаться хотя бы 1 актив."

    current_symbols.remove(sym)
    set_raw_setting("active_symbols", current_symbols, db_path=db_path)
    logger.info(f"Удалена монета из пула: {sym} (осталось: {len(current_symbols)})")
    return True, f"🗑 Монета <b>{sym}</b> удалена из мониторинга."


def toggle_symbol(symbol: str, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[bool, str]:
    """Переключает статус монеты (включена / выключена)."""
    sym = symbol.strip().upper()
    if not sym.endswith("USDT"):
        sym += "USDT"

    cfg = get_runtime_config(db_path=db_path)
    current_symbols = list(cfg["active_symbols"])

    if sym in current_symbols:
        if len(current_symbols) <= 1:
            return False, "⚠️ Нельзя отключить последнюю активную пару."
        current_symbols.remove(sym)
        set_raw_setting("active_symbols", current_symbols, db_path=db_path)
        return False, f"⚪ <b>{sym}</b> отключен"
    else:
        current_symbols.append(sym)
        set_raw_setting("active_symbols", current_symbols, db_path=db_path)
        return True, f"🟢 <b>{sym}</b> включен"


def reset_symbols_to_default(db_path: Path | str = ANALYTICS_DB_PATH) -> List[str]:
    """Сбрасывает список монет к эталонному ТОП-12."""
    default_list = list(TOP_12_LEADERS)
    set_raw_setting("active_symbols", default_list, db_path=db_path)
    logger.info("Список монет сброшен к рекомендованным ТОП-12")
    return default_list


# ═══════════════════════════════════════════════════════════════════════════
# Управление таймфреймами
# ═══════════════════════════════════════════════════════════════════════════

ALLOWED_TIMEFRAMES = ["5m", "15m", "1h", "4h"]

def toggle_timeframe(tf: str, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[bool, str]:
    """Переключает статус таймфрейма."""
    tf = tf.strip().lower()
    if tf not in ALLOWED_TIMEFRAMES:
        return False, f"Неподдерживаемый таймфрейм: {tf}"

    cfg = get_runtime_config(db_path=db_path)
    current_tfs = list(cfg["live_timeframes"])

    if tf in current_tfs:
        if len(current_tfs) <= 1:
            return False, "⚠️ Нельзя отключить последний таймфрейм."
        current_tfs.remove(tf)
        set_raw_setting("live_timeframes", current_tfs, db_path=db_path)
        return False, f"❌ Таймфрейм <b>{tf}</b> выключен"
    else:
        # Добавляем в правильном порядке
        current_tfs.append(tf)
        ordered_tfs = [t for t in ALLOWED_TIMEFRAMES if t in current_tfs]
        set_raw_setting("live_timeframes", ordered_tfs, db_path=db_path)
        return True, f"✅ Таймфрейм <b>{tf}</b> включен"


# ═══════════════════════════════════════════════════════════════════════════
# Управление капиталом и рисками
# ═══════════════════════════════════════════════════════════════════════════

def set_deposit(amount: float, db_path: Path | str = ANALYTICS_DB_PATH) -> float:
    """Устанавливает размер торгового депозита в USD."""
    amount = max(10.0, float(amount))
    set_raw_setting("deposit_usd", amount, db_path=db_path)
    
    # Автоматически адаптируем риск в $, если он превышает 10% депо
    cfg = get_runtime_config(db_path=db_path)
    current_risk = cfg["risk_budget_usd"]
    if current_risk > amount * 0.10:
        new_risk = round(amount * 0.01, 2)  # 1%
        set_raw_setting("risk_budget_usd", new_risk, db_path=db_path)
    
    logger.info(f"Установлен депозит: ${amount:,.2f}")
    return amount


def set_risk_usd(amount: float, db_path: Path | str = ANALYTICS_DB_PATH) -> float:
    """Устанавливает фиксированный риск на сделку в USD."""
    amount = max(1.0, float(amount))
    set_raw_setting("risk_budget_usd", amount, db_path=db_path)
    logger.info(f"Установлен риск на сделку: ${amount:,.2f}")
    return amount


def set_risk_pct(pct: float, db_path: Path | str = ANALYTICS_DB_PATH) -> float:
    """Устанавливает риск в процентах от депозита."""
    pct = max(0.1, min(10.0, float(pct)))
    cfg = get_runtime_config(db_path=db_path)
    dep = cfg["deposit_usd"]
    risk_usd = round(dep * (pct / 100.0), 2)
    set_raw_setting("risk_budget_usd", risk_usd, db_path=db_path)
    logger.info(f"Установлен риск: {pct:.1f}% (${risk_usd:,.2f})")
    return risk_usd


def set_leverage(lev: int, db_path: Path | str = ANALYTICS_DB_PATH) -> int:
    """Устанавливает кредитное плечо."""
    lev = max(1, min(50, int(lev)))
    set_raw_setting("default_leverage", lev, db_path=db_path)
    logger.info(f"Установлено кредитное плечо: {lev}x")
    return lev


def set_min_score(score: int, db_path: Path | str = ANALYTICS_DB_PATH) -> int:
    """Устанавливает минимальный порог качества ИИ."""
    score = max(60, min(95, int(score)))
    set_raw_setting("min_score", score, db_path=db_path)
    logger.info(f"Установлен порог ИИ: {score}%")
    return score


def toggle_scanner_pause(db_path: Path | str = ANALYTICS_DB_PATH) -> bool:
    """Ставит сканер на паузу или снимает с паузы."""
    cfg = get_runtime_config(db_path=db_path)
    new_state = not cfg.get("scanner_paused", False)
    set_raw_setting("scanner_paused", new_state, db_path=db_path)
    return new_state
