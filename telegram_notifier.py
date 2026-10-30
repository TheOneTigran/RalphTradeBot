"""
telegram_notifier.py — Модуль отправки подтверждённых торговых сигналов в Telegram.

Требования:
1. Пропускает ТОЛЬКО сигналы с оценкой Vision AI >= 85% (Textbook).
2. Форматирует сообщение строго по шаблону:
   - Моноширинный тикер для быстрого копирования кликом: `BTCUSDT`
   - Эмодзи направления (🔴 Медвежий / 🟢 Бычий импульс)
   - RSI пивоты и текущий RSI
   - Обоснование на русском языке от ИИ
   - Дата и время обнаружения
3. Прикрепляет сгенерированный PNG график с разволновкой 0-5 и линиями дивергенций.
"""
from __future__ import annotations

import html
import io
import logging
from datetime import datetime
from typing import Optional

import requests

from config import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_MIN_SCORE
)

logger = logging.getLogger(__name__)


def _fmt_p(p: float) -> str:
    """Форматирует цену в зависимости от масштаба инструмента."""
    if p >= 1000.0:
        return f"${p:,.2f}"
    elif p >= 1.0:
        return f"${p:,.4f}"
    else:
        return f"${p:,.6f}"

def _fmt_pc(p: float) -> str:
    """Форматирует цену для копирования одним кликом (обёрнуто в <code> тег Telegram).
    Возвращает чистое число без $ — чтобы сразу вставить в ордер на бирже."""
    if p >= 1000.0:
        return f"<code>{p:.2f}</code>"
    elif p >= 1.0:
        return f"<code>{p:.4f}</code>"
    else:
        return f"<code>{p:.6f}</code>"


def _fmt_vol(v: float) -> str:
    """Форматирует объем в USDT."""
    return f"${v:,.1f}"


def _fmt_coins_code(amount: float) -> str:
    """Форматирует количество монет для 1-клик копирования (в теге <code>, без букв)."""
    if amount >= 1000.0:
        c_str = f"{amount:,.0f}".replace(",", "")
    elif amount >= 10.0:
        c_str = f"{amount:.2f}"
    elif amount >= 1.0:
        c_str = f"{amount:.3f}"
    elif amount >= 0.01:
        c_str = f"{amount:.4f}"
    else:
        c_str = f"{amount:.6f}"
    return f"<code>{c_str}</code>"


def format_signal_message(
    symbol: str,
    interval: str,
    direction: str,
    signal_price: float,
    swept_rsi: float = 0.0,
    signal_rsi: float = 0.0,
    current_rsi: float = 0.0,
    reason: str = "",
    score: int = 70,
    detection_time: Optional[datetime] = None,
    trade_plan: Optional[Any] = None,
    dur_bars: Optional[int] = None,
    dur_hours: Optional[float] = None,
    w0_rsi: Optional[float] = None,
    w0_status: Optional[str] = None,
    origin_div: bool = False,
    w3_longest: bool = True,
    conflict_note: Optional[str] = None,
    sec_to_close: int = 0,
) -> str:
    """Формирует текст сообщения для Telegram с 1-клик копируемыми уровнями и объемами."""
    dt = detection_time or datetime.now()
    dt_str = dt.strftime("%d.%m.%Y в %H:%M МСК")
    
    if direction == "SHORT":
        header = "🔴 <b>МЕДВЕЖИЙ ИМПУЛЬС ЗАВЕРШЁН (W5)</b>"
        dir_badge = "▼ <b>SHORT</b>"
        sl_note = "└ Стоп внутри свечи на уровне 83% (0.17 от экстремума W5)"
    else:
        header = "🟢 <b>БЫЧИЙ ИМПУЛЬС ЗАВЕРШЁН (W5)</b>"
        dir_badge = "▲ <b>LONG</b>"
        sl_note = "└ Стоп внутри свечи на уровне 83% (0.17 от экстремума W5)"

    conflict_banner = ""
    if conflict_note:
        conflict_banner = (
            f"⚠️ <b>ИНФОРМАЦИОННЫЙ СИГНАЛ (ПОЗИЦИЯ НЕ ОТКРЫВАЕТСЯ)</b>\n"
            f"└ <i>По активу уже сопровождается: {html.escape(str(conflict_note))}</i>\n"
            f"──────────────────────────\n\n"
        )

    if sec_to_close > 0:
        timing_line = f"⏳ До закрытия свечи: <b>~{sec_to_close} сек (~{max(1, round(sec_to_close/60))} мин)</b> <i>(проверьте расчёт и выставьте лимитку 0.618 после закрытия)</i>\n"
    else:
        timing_line = "⏳ Статус: <b>Свеча закрыта — ожидание коррекции 61.8% (0.618 Limit)</b>\n"

    if trade_plan is not None:
        entry_c = _fmt_pc(trade_plan.entry_price)
        sl_c = _fmt_pc(trade_plan.sl_price)
        tp1_c = _fmt_pc(trade_plan.tp1_price)
        tp2_c = _fmt_pc(trade_plan.tp2_price)

        pos_vol_str = _fmt_vol(getattr(trade_plan, 'pos_size_usd', 0.0))
        coins_c = _fmt_coins_code(getattr(trade_plan, 'pos_size_coins', 0.0))
        margin_str = f"${getattr(trade_plan, 'margin_usd', 0.0):.1f}"
        lev = getattr(trade_plan, 'leverage', 10)
        risk_usd = getattr(trade_plan, 'risk_budget_usd', 5.0)

        tp1_vol_s = _fmt_vol(getattr(trade_plan, 'tp1_size_usd', 0.0))
        tp1_coins_c = _fmt_coins_code(getattr(trade_plan, 'tp1_size_coins', 0.0))
        tp2_vol_s = _fmt_vol(getattr(trade_plan, 'tp2_size_usd', 0.0))
        tp2_coins_c = _fmt_coins_code(getattr(trade_plan, 'tp2_size_coins', 0.0))

        plan_block = (
            f"━━━ 🎯 <b>КУДА И ЧТО ВЫСТАВЛЯТЬ (ОРДЕРА):</b> ━━━\n\n"
            f"1️⃣ {dir_badge} │ Вход (Limit 61.8% / 0.618): {entry_c}\n"
            f"   └ Объем: {coins_c} монет (<b>{pos_vol_str}</b>)\n"
            f"   └ Маржа: <b>{margin_str}</b> (плечо {lev}x │ риск <b>${risk_usd:.1f}</b>)\n\n"
            f"2️⃣ 🛑 <b>STOP LOSS:</b> {sl_c} (−{trade_plan.sl_pct:.2f}%)\n"
            f"   {sl_note}\n\n"
            f"3️⃣ 🎯 <b>TP1 (1.5x │ +3.54R Net):</b> {tp1_c} (+{trade_plan.tp1_pct:.2f}%)\n"
            f"   └ Закрыть 40%: {tp1_coins_c} монет (<b>{tp1_vol_s}</b>) ➔ стоп в БУ\n\n"
            f"4️⃣ 🎯 <b>TP2 (3.5x │ +8.25R Net):</b> {tp2_c} (+{trade_plan.tp2_pct:.2f}%)\n"
            f"   └ Закрыть 60%: {tp2_coins_c} монет (<b>{tp2_vol_s}</b>)\n\n"
        )
    else:
        price_c = _fmt_pc(signal_price)
        plan_block = f"💰 Вход: {price_c}\n\n"

    msg = (
        f"{conflict_banner}"
        f"{header}\n\n"
        f"📌 <code>{symbol}</code> (Crypto Futures)\n"
        f"⏱️ Таймфрейм: <b>{interval}</b>\n"
        f"{timing_line}\n"
        f"{plan_block}"
        f"📅 {dt_str}"
    )
    return msg



import threading
import time

_tg_lock = threading.Lock()

def _send_tg_request(method_url: str, is_json: bool = False, retry_count: int = 4, **kwargs) -> Optional[requests.Response]:
    """Вспомогательная функция для отправки запросов в Telegram с защитой от флуда и повторами."""
    for attempt in range(retry_count):
        try:
            # Перематываем файловые потоки (BytesIO) в начало перед каждой попыткой отправки,
            # чтобы при повторной отправке после таймаута файл не оказывался пустым (HTTP 400: file must be non-empty)
            if "files" in kwargs and kwargs["files"]:
                for k, v in kwargs["files"].items():
                    if isinstance(v, (tuple, list)) and len(v) >= 2 and hasattr(v[1], "seek"):
                        v[1].seek(0)
                    elif hasattr(v, "seek"):
                        v.seek(0)

            if is_json:
                resp = requests.post(method_url, json=kwargs.get("json"), timeout=kwargs.get("timeout", 20))
            else:
                resp = requests.post(method_url, data=kwargs.get("data"), files=kwargs.get("files"), timeout=kwargs.get("timeout", 25))
            
            if resp.status_code == 200 and resp.json().get("ok"):
                return resp
            
            # Если ошибка парсинга HTML сущностей (400) — отправляем без parse_mode как чистый текст (гарантия доставки)
            if resp.status_code == 400 and "can't parse entities" in resp.text:
                logger.warning("⚠️ Ошибка HTML парсера Telegram. Пробуем повторную отправку без parse_mode...")
                if is_json and "json" in kwargs:
                    fb_json = dict(kwargs["json"])
                    fb_json.pop("parse_mode", None)
                    fb_resp = requests.post(method_url, json=fb_json, timeout=kwargs.get("timeout", 20))
                    if fb_resp.status_code == 200 and fb_resp.json().get("ok"):
                        return fb_resp
                elif "data" in kwargs:
                    fb_data = dict(kwargs["data"])
                    fb_data.pop("parse_mode", None)
                    # Если есть файлы, перематываем BytesIO в начало
                    if "files" in kwargs:
                        for k, v in kwargs["files"].items():
                            if hasattr(v[1], "seek"):
                                v[1].seek(0)
                    fb_resp = requests.post(method_url, data=fb_data, files=kwargs.get("files"), timeout=kwargs.get("timeout", 25))
                    if fb_resp.status_code == 200 and fb_resp.json().get("ok"):
                        return fb_resp

            # Если словили лимит Telegram 429
            if resp.status_code == 429:
                wait_sec = resp.json().get("parameters", {}).get("retry_after", 3)
                logger.warning(f"⏳ Telegram Flood Control: пауза {wait_sec} сек перед повтором...")
                time.sleep(wait_sec + 0.5)
                continue
            else:
                logger.error(f"❌ Telegram API HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(1.0)
        except Exception as e:
            logger.error(f"❌ Ошибка соединения с Telegram (попытка {attempt+1}/{retry_count}): {e}")
            time.sleep(2.0)
    return None


def send_signal_to_telegram(
    chart_png: bytes,
    symbol: str,
    interval: str,
    direction: str,
    signal_price: float,
    swept_rsi: float,
    signal_rsi: float,
    current_rsi: float,
    reason: str,
    score: int,
    detection_time: Optional[datetime] = None,
    trade_plan: Optional[Any] = None,
    dur_bars: Optional[int] = None,
    dur_hours: Optional[float] = None,
    w0_rsi: Optional[float] = None,
    w0_status: Optional[str] = None,
    origin_div: bool = False,
    w3_longest: bool = True,
    conflict_note: Optional[str] = None,
    bot_token: str = TELEGRAM_BOT_TOKEN,
    chat_id: str = TELEGRAM_CHAT_ID,
    min_score: int = TELEGRAM_MIN_SCORE,
    sec_to_close: int = 0,
) -> Optional[int]:
    """
    Отправляет подтверждённый сигнал в Telegram.
    Строгий фильтр: отправляются только сигналы с score >= min_score (85%).
    Использует потокобезопасную отправку с паузой для соблюдения Rate Limits Telegram.
    Возвращает message_id отправленного сообщения для последующих ответов-сопровождения.
    """
    if score < min_score:
        logger.info(f"⏭️ Пропуск отправки в TG: Score {score} < {min_score}% (порог Textbook)")
        return None

    if not bot_token or not chat_id:
        logger.warning("⚠️ Не заданы TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID")
        return None

    caption = format_signal_message(
        symbol=symbol,
        interval=interval,
        direction=direction,
        signal_price=signal_price,
        swept_rsi=swept_rsi,
        signal_rsi=signal_rsi,
        current_rsi=current_rsi,
        reason=reason,
        score=score,
        detection_time=detection_time,
        trade_plan=trade_plan,
        dur_bars=dur_bars,
        dur_hours=dur_hours,
        w0_rsi=w0_rsi,
        w0_status=w0_status,
        origin_div=origin_div,
        w3_longest=w3_longest,
        conflict_note=conflict_note,
        sec_to_close=sec_to_close,
    )

    url_photo = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    with _tg_lock:
        time.sleep(1.2)  # Пауза между сообщениями для соблюдения Telegram Rate Limit
        if len(caption) <= 1024:
            files = {"photo": ("signal_chart.png", io.BytesIO(chart_png), "image/png")}
            data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
            resp = _send_tg_request(url_photo, is_json=False, data=data, files=files)
            if resp:
                msg_id = resp.json().get("result", {}).get("message_id")
                logger.info(f"🚀 Сигнал {symbol} {direction} (Score {score}) успешно отправлен в Telegram! (MsgID: {msg_id})")
                return msg_id
        else:
            files = {"photo": ("signal_chart.png", io.BytesIO(chart_png), "image/png")}
            resp1 = _send_tg_request(url_photo, is_json=False, data={"chat_id": chat_id}, files=files)
            if resp1:
                time.sleep(0.5)
                resp2 = _send_tg_request(url_msg, is_json=True, json={"chat_id": chat_id, "text": caption, "parse_mode": "HTML"})
                if resp2:
                    msg_id = resp2.json().get("result", {}).get("message_id")
                    logger.info(f"🚀 Сигнал {symbol} {direction} (Score {score}) успешно отправлен в Telegram (двумя частями, MsgID: {msg_id})!")
                    return msg_id

    return None


def send_trade_update_reply(
    reply_to_message_id: int,
    text: str,
    bot_token: str = TELEGRAM_BOT_TOKEN,
    chat_id: str = TELEGRAM_CHAT_ID,
) -> Optional[int]:
    """
    Отправляет уведомление-сопровождение (TP, SL, БУ, Fail-Fast)
    строго в виде ОТВЕТА (reply) на оригинальный сигнал.
    """
    if not bot_token or not chat_id or not reply_to_message_id:
        return None

    url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "reply_to_message_id": reply_to_message_id,
        "allow_sending_without_reply": True,
    }

    with _tg_lock:
        time.sleep(0.5)
        resp = _send_tg_request(url_msg, is_json=True, json=payload)
        if resp and resp.status_code == 200:
            msg_id = resp.json().get("result", {}).get("message_id")
            logger.info(f"💬 Ответ на сигнал #{reply_to_message_id} успешно доставлен в Telegram (MsgID: {msg_id})")
            return msg_id

    return None

