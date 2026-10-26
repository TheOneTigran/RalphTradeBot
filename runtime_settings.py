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
        active_symbols = list(TOP_30_SYMBOLS)
        set_raw_setting("active_symbols", active_symbols, db_path=db_path)

    live_timeframes = get_raw_setting("live_timeframes", None, db_path=db_path)
    if not live_timeframes or not isinstance(live_timeframes, list):
        live_timeframes = list(LIVE_TIMEFRAMES)
        set_raw_setting("live_timeframes", live_timeframes, db_path=db_path)

    deposit_usd = float(get_raw_setting("deposit_usd", 500.0, db_path=db_path))
    risk_budget_usd = float(get_raw_setting("risk_budget_usd", float(RISK_BUDGET_USD), db_path=db_path))
    risk_pct = round((risk_budget_usd / deposit_usd * 100.0), 2) if deposit_usd > 0 else 0.5
    default_leverage = int(get_raw_setting("default_leverage", int(DEFAULT_LEVERAGE), db_path=db_path))
    min_score = int(get_raw_setting("min_score", int(TELEGRAM_MIN_SCORE), db_path=db_path))
    scanner_paused = bool(get_raw_setting("scanner_paused", False, db_path=db_path))
    night_mode_enabled = bool(get_raw_setting("night_mode_enabled", True, db_path=db_path))
    night_mode_start = int(get_raw_setting("night_mode_start", 0, db_path=db_path))
    night_mode_end = int(get_raw_setting("night_mode_end", 8, db_path=db_path))

    cfg = {
        "active_symbols": active_symbols,
        "live_timeframes": live_timeframes,
        "deposit_usd": deposit_usd,
        "risk_budget_usd": risk_budget_usd,
        "risk_pct": risk_pct,
        "default_leverage": default_leverage,
        "min_score": min_score,
        "scanner_paused": scanner_paused,
        "night_mode_enabled": night_mode_enabled,
        "night_mode_start": night_mode_start,
        "night_mode_end": night_mode_end,
    }

    with _settings_lock:
        _cached_config = cfg
        _last_cache_time = now

    return dict(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Управление монетами и поиск по тикеру (Binance Futures / Bybit)
# ═══════════════════════════════════════════════════════════════════════════

POPULAR_COINS_DEFAULT = [
    "NEARUSDT", "SUIUSDT", "KASUSDT", "DOGEUSDT",
    "UNIUSDT", "APTUSDT", "OPUSDT", "1000PEPEUSDT",
    "LTCUSDT", "BCHUSDT", "TAOUSDT", "ICPUSDT",
    "SOLUSDT", "ETHUSDT", "BTCUSDT", "AVAXUSDT",
]

FALLBACK_EXCHANGE_SYMBOLS = list(dict.fromkeys(
    list(TOP_30_SYMBOLS) + POPULAR_COINS_DEFAULT + [
        "TONUSDT", "WIFUSDT", "ONDOUSDT", "TIAUSDT", "SEIUSDT",
        "ORDIUSDT", "JUPUSDT", "ENAUSDT", "PENDLEUSDT", "FILUSDT",
        "ATOMUSDT", "HBARUSDT", "STXUSDT", "IMXUSDT", "MKRUSDT",
        "LDOUSDT", "RUNEUSDT", "GRTUSDT", "ALGOUSDT", "SANDUSDT",
        "MANAUSDT", "AXSUSDT", "APEUSDT", "GALAUSDT", "DYDXUSDT",
        "1000BONKUSDT", "1000FLOKIUSDT", "NOTUSDT", "EIGENUSDT", "ZROUSDT",
        "STRKUSDT", "ZKUSDT", "PYTHUSDT", "JTOUSDT", "WLDUSDT",
        "ARKMUSDT", "JASMYUSDT", "ENSUSDT", "CRVUSDT", "SNXUSDT",
        "COMPUSDT", "VETUSDT", "EGLDUSDT", "FLOWUSDT", "THETAUSDT",
        "EOSUSDT", "XTZUSDT", "NEOUSDT", "KAVAUSDT", "MINAUSDT",
        "CHZUSDT", "ZILUSDT", "ENJUSDT", "1INCHUSDT", "SUSHIUSDT",
    ]
))

_exchange_tickers_cache: List[Dict[str, Any]] = []
_exchange_tickers_ts: float = 0.0
EXCHANGE_CACHE_TTL_SEC: float = 120.0


def normalize_ticker_input(raw: str) -> str:
    """Нормализует пользовательский ввод тикера (например, 'sol/usdt' -> 'SOLUSDT')."""
    s = (raw or "").strip().upper()
    for ch in ("/", "-", "_", " ", ":"):
        s = s.replace(ch, "")
    return s


def _fetch_exchange_tickers_cached() -> List[Dict[str, Any]]:
    """
    Загружает и кэширует список всех USDT-фьючерсов с актуальными ценами и объёмами
    с Binance Futures (фоллбэк на Bybit Linear и локальный список).
    """
    global _exchange_tickers_cache, _exchange_tickers_ts
    import time
    import requests

    now = time.time()
    if _exchange_tickers_cache and (now - _exchange_tickers_ts) < EXCHANGE_CACHE_TTL_SEC:
        return _exchange_tickers_cache

    headers = {"User-Agent": "RalphTradeBot-TickerSearch/2.3"}
    results: List[Dict[str, Any]] = []

    # 1. Попытка через Binance Futures 24hr ticker
    try:
        resp = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", headers=headers, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                for item in data:
                    sym = str(item.get("symbol", ""))
                    if not sym.endswith("USDT") or "_" in sym:
                        continue
                    price = float(item.get("lastPrice") or 0.0)
                    if price <= 0:
                        continue
                    results.append({
                        "symbol": sym,
                        "base": sym[:-4],
                        "price": price,
                        "change_24h": float(item.get("priceChangePercent") or 0.0),
                        "vol_24h": float(item.get("quoteVolume") or 0.0),
                        "source": "Binance",
                    })
    except Exception as e:
        logger.debug(f"Binance ticker search fetch error: {e}")

    # 2. Фоллбэк через Bybit Linear
    if not results:
        try:
            resp_b = requests.get(
                "https://api.bybit.com/v5/market/tickers?category=linear",
                headers=headers,
                timeout=6,
            )
            if resp_b.status_code == 200:
                b_data = resp_b.json()
                rows = b_data.get("result", {}).get("list", [])
                for item in rows:
                    sym = str(item.get("symbol", ""))
                    if not sym.endswith("USDT"):
                        continue
                    price = float(item.get("lastPrice") or 0.0)
                    if price <= 0:
                        continue
                    results.append({
                        "symbol": sym,
                        "base": sym[:-4],
                        "price": price,
                        "change_24h": float(item.get("price24hPcnt") or 0.0) * 100.0,
                        "vol_24h": float(item.get("turnover24h") or 0.0),
                        "source": "Bybit",
                    })
        except Exception as e:
            logger.debug(f"Bybit ticker search fetch error: {e}")

    # 3. Локальный фоллбэк, если сеть недоступна
    if not results:
        for sym in FALLBACK_EXCHANGE_SYMBOLS:
            results.append({
                "symbol": sym,
                "base": sym[:-4] if sym.endswith("USDT") else sym,
                "price": 0.0,
                "change_24h": 0.0,
                "vol_24h": 0.0,
                "source": "Preset",
            })

    if results:
        _exchange_tickers_cache = results
        _exchange_tickers_ts = now
    return results


def search_exchange_symbols(
    query: str,
    limit: int = 8,
    db_path: Path | str = ANALYTICS_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Ищет инструменты по тикеру (или части тикера) на бирже и проверяет,
    находятся ли они уже в активном портфеле бота.
    """
    q = normalize_ticker_input(query)
    if not q:
        return []

    q_base = q[:-4] if q.endswith("USDT") and len(q) > 4 else q
    q_full = q if q.endswith("USDT") else f"{q}USDT"

    cfg = get_runtime_config(db_path=db_path)
    active_set = set(cfg["active_symbols"])

    tickers = _fetch_exchange_tickers_cached()
    scored: List[Tuple[int, float, Dict[str, Any]]] = []

    for item in tickers:
        sym = item["symbol"]
        base = item["base"]
        base_no_1000 = base[4:] if base.startswith("1000") else base

        if sym == q_full or base == q_base or base_no_1000 == q_base:
            rank = 0
        elif base.startswith(q_base) or base_no_1000.startswith(q_base):
            rank = 1
        elif q_base in base or q in sym:
            rank = 2
        else:
            continue

        entry = dict(item)
        entry["in_portfolio"] = sym in active_set
        scored.append((rank, -float(item.get("vol_24h", 0.0)), entry))

    scored.sort(key=lambda x: (x[0], x[1], x[2]["symbol"]))
    return [x[2] for x in scored[:limit]]


def get_portfolio_display_symbols(db_path: Path | str = ANALYTICS_DB_PATH) -> List[str]:
    """
    Возвращает полный список монет для отображения в меню портфеля:
    базовые популярные монеты + все пользовательские добавленные монеты + активные монеты.
    """
    cfg = get_runtime_config(db_path=db_path)
    active = list(cfg["active_symbols"])
    custom = get_raw_setting("custom_symbols", [], db_path=db_path)
    if not isinstance(custom, list):
        custom = []

    combined: List[str] = []
    for sym in POPULAR_COINS_DEFAULT + custom + active:
        if sym and sym not in combined:
            combined.append(sym)
    return combined


def add_symbol(symbol: str, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[bool, str]:
    """Добавляет монету в активный пул сканирования (с умным разрешением тикера)."""
    sym = normalize_ticker_input(symbol)
    if not sym:
        return False, "⚠️ Укажите корректный тикер монеты (например: <code>SOL</code> или <code>SOLUSDT</code>)."

    # Проверяем по биржевому списку для точного совпадения (например PEPE -> 1000PEPEUSDT)
    matches = search_exchange_symbols(sym, limit=5, db_path=db_path)
    q_base = sym[:-4] if sym.endswith("USDT") and len(sym) > 4 else sym
    resolved_sym = sym if sym.endswith("USDT") else f"{sym}USDT"

    if matches:
        exact = [
            m["symbol"]
            for m in matches
            if m["symbol"] == resolved_sym
            or m["base"] == q_base
            or (m["base"].startswith("1000") and m["base"][4:] == q_base)
        ]
        if exact:
            resolved_sym = exact[0]

    cfg = get_runtime_config(db_path=db_path)
    current_symbols = list(cfg["active_symbols"])

    if resolved_sym in current_symbols:
        return False, f"ℹ️ Инструмент <b>{resolved_sym}</b> уже находится в портфеле анализа."

    current_symbols.append(resolved_sym)
    set_raw_setting("active_symbols", current_symbols, db_path=db_path)

    # Сохраняем в список пользовательских монет, чтобы кнопка отображалась в меню портфеля
    custom = get_raw_setting("custom_symbols", [], db_path=db_path)
    if not isinstance(custom, list):
        custom = []
    if resolved_sym not in POPULAR_COINS_DEFAULT and resolved_sym not in custom:
        custom.append(resolved_sym)
        set_raw_setting("custom_symbols", custom, db_path=db_path)

    logger.info(f"Добавлена новая монета в пул: {resolved_sym} (всего: {len(current_symbols)})")
    return True, f"✅ Инструмент <b>{resolved_sym}</b> добавлен в портфель для анализа! (Всего в пуле: <b>{len(current_symbols)}</b>)"


def remove_symbol(symbol: str, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[bool, str]:
    """Удаляет монету из активного портфеля сканирования."""
    sym = normalize_ticker_input(symbol)
    if not sym.endswith("USDT"):
        sym += "USDT"

    cfg = get_runtime_config(db_path=db_path)
    current_symbols = list(cfg["active_symbols"])

    # Если ввели PEPE, а в пуле 1000PEPEUSDT
    if sym not in current_symbols:
        alt_sym = f"1000{sym}"
        if alt_sym in current_symbols:
            sym = alt_sym

    if sym not in current_symbols:
        return False, f"ℹ️ Инструмент <b>{sym}</b> сейчас не находится в активном портфеле."

    if len(current_symbols) <= 1:
        return False, "⚠️ Нельзя убрать последнюю монету. В портфеле должен оставаться хотя бы 1 инструмент."

    current_symbols.remove(sym)
    set_raw_setting("active_symbols", current_symbols, db_path=db_path)

    # Если это кастомная монета — убираем также из custom_symbols
    custom = get_raw_setting("custom_symbols", [], db_path=db_path)
    if isinstance(custom, list) and sym in custom:
        custom.remove(sym)
        set_raw_setting("custom_symbols", custom, db_path=db_path)

    logger.info(f"Удалена монета из пула: {sym} (осталось: {len(current_symbols)})")
    return True, f"🗑 Инструмент <b>{sym}</b> убран из портфеля анализа. (Осталось: <b>{len(current_symbols)}</b>)"


def toggle_symbol(symbol: str, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[bool, str]:
    """Переключает статус монеты (включена / выключена)."""
    sym = normalize_ticker_input(symbol)
    if not sym.endswith("USDT"):
        sym += "USDT"

    cfg = get_runtime_config(db_path=db_path)
    current_symbols = list(cfg["active_symbols"])

    if sym in current_symbols:
        if len(current_symbols) <= 1:
            return False, "⚠️ Нельзя отключить последнюю активную пару."
        current_symbols.remove(sym)
        set_raw_setting("active_symbols", current_symbols, db_path=db_path)
        return False, f"⚪ <b>{sym}</b> убран из портфеля"
    else:
        current_symbols.append(sym)
        set_raw_setting("active_symbols", current_symbols, db_path=db_path)
        custom = get_raw_setting("custom_symbols", [], db_path=db_path)
        if not isinstance(custom, list):
            custom = []
        if sym not in POPULAR_COINS_DEFAULT and sym not in custom:
            custom.append(sym)
            set_raw_setting("custom_symbols", custom, db_path=db_path)
        return True, f"🟢 <b>{sym}</b> добавлен в портфель"


def reset_symbols_to_default(db_path: Path | str = ANALYTICS_DB_PATH) -> List[str]:
    """Сбрасывает список монет к эталонному ТОП-12."""
    default_list = list(TOP_30_SYMBOLS)
    set_raw_setting("active_symbols", default_list, db_path=db_path)
    set_raw_setting("custom_symbols", [], db_path=db_path)
    logger.info("Список монет сброшен к рекомендованным ТОП-12")
    return default_list


# ═══════════════════════════════════════════════════════════════════════════
# Управление таймфреймами
# ═══════════════════════════════════════════════════════════════════════════

ALLOWED_TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]

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
        new_risk = round(amount * 0.005, 2)  # 0.5%
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


# ═══════════════════════════════════════════════════════════════════════════
# Управление ночным режимом (Night Mode)
# ═══════════════════════════════════════════════════════════════════════════

def get_msk_time() -> datetime:
    """Возвращает текущее время в часовом поясе Москвы (UTC+3)."""
    from datetime import timedelta
    return datetime.now(timezone.utc) + timedelta(hours=3)


def is_night_mode_active(cfg: Optional[Dict[str, Any]] = None, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[bool, str]:
    """
    Проверяет, действует ли сейчас ночной режим (по времени МСК).
    Возвращает (активен_ли_режим, строка_с_часами).
    """
    if cfg is None:
        cfg = get_runtime_config(db_path=db_path)

    if not cfg.get("night_mode_enabled", True):
        return False, "Выключен"

    start_h = int(cfg.get("night_mode_start", 0))
    end_h = int(cfg.get("night_mode_end", 8))
    end_h = int(cfg.get("night_mode_end", 8))

    now_msk = get_msk_time()
    cur_h = now_msk.hour
    if now_msk.weekday() == 6:
        return True, "Воскресенье — выходной (Пн–Сб 08–10, 15–24 МСК)"

    if start_h > end_h:
        # Переход через полночь, например 23:00 -> 09:00
        is_night = (cur_h >= start_h or cur_h < end_h)
    elif start_h < end_h:
        # Без перехода через полночь, например 01:00 -> 07:00
        is_night = (start_h <= cur_h < end_h)
    else:
        is_night = False

    hours_str = f"Тихие часы {start_h:02d}:00–{end_h:02d}:00 и 10:00–15:00 МСК (Прайм: 08–10, 15–24 МСК без ВС)"
    if start_h == 0 and end_h == 8 and (10 <= cur_h < 15):
        is_night = True
    return is_night, hours_str


def is_bar_allowed_by_schedule(
    bar_dt_utc: Any,
    interval: str,
    cfg: Optional[Dict[str, Any]] = None,
    db_path: Path | str = ANALYTICS_DB_PATH,
) -> Tuple[bool, str]:
    """
    Проверяет, попадает ли время закрывающейся сигнальной свечи в разрешённые прайм-диапазоны МСК:
    - Исключено Воскресенье (weekday == 6 по МСК).
    - Для интрадей ТФ (5m, 15m, 30m, 1h): разрешены часы 08:00–09:59 и 15:00–23:59 МСК.
    - Для 4h: разрешены все слоты Пн–Сб, кроме токсичных 07:00 и 11:00 МСК.
    - Для 1d: разрешены все дни Пн–Сб.
    """
    if cfg is None:
        cfg = get_runtime_config(db_path=db_path)

    if not cfg.get("night_mode_enabled", True):
        return True, "24/7"

    from datetime import timedelta
    if hasattr(bar_dt_utc, "to_pydatetime"):
        dt_u = bar_dt_utc.to_pydatetime()
    elif isinstance(bar_dt_utc, datetime):
        dt_u = bar_dt_utc
    else:
        dt_u = datetime.now(timezone.utc)

    if dt_u.tzinfo is None:
        dt_u = dt_u.replace(tzinfo=timezone.utc)
    dt_msk = dt_u.astimezone(timezone.utc) + timedelta(hours=3)

    if dt_msk.weekday() == 6:
        return False, "sunday_excluded_msk"

    hr = dt_msk.hour
    if interval in ("5m", "15m", "30m", "1h"):
        if hr in (8, 9, 15, 16, 17, 18, 19, 20, 21, 22, 23):
            return True, "prime_intraday_msk"
        return False, f"outside_prime_hours_msk ({hr:02d}:00)"
    elif interval == "4h":
        if hr in (7, 11):
            return False, f"forbidden_4h_slot_msk ({hr:02d}:00)"
        return True, "allowed_4h_msk"
    elif interval == "1d":
        return True, "allowed_1d_msk"

    return True, "allowed"
    return is_night, hours_str


def set_night_mode_hours(start_hour: int, end_hour: int, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[int, int]:
    """Устанавливает часы начала и окончания ночного режима (0-23 МСК)."""
    start_h = max(0, min(23, int(start_hour)))
    end_h = max(0, min(23, int(end_hour)))
    set_raw_setting("night_mode_start", start_h, db_path=db_path)
    set_raw_setting("night_mode_end", end_h, db_path=db_path)
    logger.info(f"Установлены часы ночного режима: {start_h:02d}:00 - {end_h:02d}:00 МСК")
    return start_h, end_h


def toggle_night_mode(db_path: Path | str = ANALYTICS_DB_PATH) -> bool:
    """Включает или выключает ночной режим."""
    cfg = get_runtime_config(db_path=db_path)
    new_state = not cfg.get("night_mode_enabled", True)
    set_raw_setting("night_mode_enabled", new_state, db_path=db_path)
    logger.info(f"Ночной режим: {'ВКЛ' if new_state else 'ВЫКЛ'}")
    return new_state


def adjust_night_mode_hour(which: str, delta: int, db_path: Path | str = ANALYTICS_DB_PATH) -> Tuple[int, int]:
    """Смещает час начала ('start') или конца ('end') на delta (-1 или +1)."""
    cfg = get_runtime_config(db_path=db_path)
    start_h = int(cfg.get("night_mode_start", 0))
    end_h = int(cfg.get("night_mode_end", 8))
    end_h = int(cfg.get("night_mode_end", 8))
    if which == "start":
        start_h = (start_h + delta) % 24
    elif which == "end":
        end_h = (end_h + delta) % 24
    set_night_mode_hours(start_h, end_h, db_path=db_path)
    return start_h, end_h
