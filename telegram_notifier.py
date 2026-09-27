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


def _fmt_vol(v: float) -> str:
    """Форматирует объем в USDT."""
    return f"${v:,.1f}"


def _fmt_coins(amount: float) -> str:
    """Форматирует количество монет."""
    if amount >= 1000.0:
        return f"{amount:,.0f}"
    elif amount >= 10.0:
        return f"{amount:,.2f}"
    elif amount >= 1.0:
        return f"{amount:,.3f}"
    else:
        return f"{amount:,.5f}"


def format_signal_message(
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
) -> str:
    """Формирует текст сообщения для Telegram с полноценным торговым планом и HTML-разметкой."""
    dt = detection_time or datetime.now()
    dt_str = dt.strftime("%d.%m.%Y в %H:%M")
    
    if direction == "SHORT":
        header = "🔴 <b>МЕДВЕЖИЙ ИМПУЛЬС ЗАВЕРШЁН (W5)</b>"
        dir_badge = "▼ <b>SHORT</b>"
        sl_note = "└ Стоп-Маркет за пиком W5"
        imp_sign = "+"
    else:
        header = "🟢 <b>БЫЧИЙ ИМПУЛЬС ЗАВЕРШЁН (W5)</b>"
        dir_badge = "▲ <b>LONG</b>"
        sl_note = "└ Стоп-Маркет за лоу W5"
        imp_sign = "-"

    # Если есть конфликт активной сделки — выводим информационную плашку
    conflict_banner = ""
    if conflict_note:
        conflict_banner = (
            f"⚠️ <b>ИНФОРМАЦИОННЫЙ СИГНАЛ (ПОЗИЦИЯ НЕ ОТКРЫВАЕТСЯ)</b>\n"
            f"└ <i>По активу уже сопровождается: {conflict_note}</i>\n"
            f"──────────────────────────\n\n"
        )

    fail_fast_str = "45 минут (3 свечи)" if interval == "15m" else ("3 часа (3 свечи)" if interval == "1h" else "3 свечи")

    # Если есть TradePlan
    if trade_plan is not None:
        entry_s = _fmt_p(trade_plan.entry_price)
        sl_s = _fmt_p(trade_plan.sl_price)
        tp1_s = _fmt_p(trade_plan.tp1_price)
        tp2_s = _fmt_p(trade_plan.tp2_price)
        tp3_s = _fmt_p(trade_plan.tp3_price)

        pos_vol_str = _fmt_vol(getattr(trade_plan, 'pos_size_usd', 0.0))
        coins_str = _fmt_coins(getattr(trade_plan, 'pos_size_coins', 0.0))
        margin_str = f"${getattr(trade_plan, 'margin_usd', 0.0):.1f}"
        lev = getattr(trade_plan, 'leverage', 10)
        risk_usd = getattr(trade_plan, 'risk_budget_usd', 10.0)

        tp1_vol_s = _fmt_vol(getattr(trade_plan, 'tp1_size_usd', 0.0))
        tp1_coins_s = _fmt_coins(getattr(trade_plan, 'tp1_size_coins', 0.0))
        tp2_vol_s = _fmt_vol(getattr(trade_plan, 'tp2_size_usd', 0.0))
        tp2_coins_s = _fmt_coins(getattr(trade_plan, 'tp2_size_coins', 0.0))

        plan_block = (
            f"━━━ 🎯 <b>КУДА И ЧТО ВЫСТАВЛЯТЬ (ОРДЕРА):</b> ━━━\n\n"
            f"1️⃣ {dir_badge} │ Вход: <b>{entry_s}</b>\n"
            f"   └ Объем: <b>{pos_vol_str}</b> (<b>{coins_str}</b> монет)\n"
            f"   └ Реком. маржа: <b>{margin_str}</b> (плечо {lev}x / риск ${risk_usd:.1f})\n\n"
            f"2️⃣ 🛑 <b>STOP LOSS:</b> <b>{sl_s}</b> (−{trade_plan.sl_pct:.2f}%)\n"
            f"   {sl_note}\n\n"
            f"3️⃣ 🎯 <b>TP1 (23.6% Фибо):</b> <b>{tp1_s}</b> (+{trade_plan.tp1_pct:.2f}%)\n"
            f"   └ Тип: Limit │ <b>Закрыть {int(trade_plan.tp1_share*100)}%</b> ({tp1_vol_s} / {tp1_coins_s} монет)\n\n"
            f"4️⃣ 🎯 <b>TP2 (38.2% Фибо):</b> <b>{tp2_s}</b> (+{trade_plan.tp2_pct:.2f}%)\n"
            f"   └ Тип: Limit │ <b>Закрыть {int(trade_plan.tp2_share*100)}%</b> ({tp2_vol_s} / {tp2_coins_s} монет)\n\n"
            f"5️⃣ 🏃 <b>Остаток {int(trade_plan.tp3_share*100)}% (Раннер):</b>\n"
            f"   └ Цель TP3: <b>{tp3_s}</b> или трейлинг по Chandelier ATR\n\n"
            f"━━━ 📋 <b>ЧТО ДЕЛАТЬ ПОСЛЕ ВХОДА (РЕГЛАМЕНТ):</b> ━━━\n\n"
            f"🛡️ <b>Безубыток:</b> Бот пришлёт ответ при взятии TP2 ➔ перенести SL в БУ (+0.1%).\n"
            f"⏱️ <b>Таймер {fail_fast_str}:</b> Бот сам отслеживает свечи. Если импульс угаснет (< +0.4R), бот пришлёт команду закрыть остаток по рынку (Fail-Fast).\n\n"
            f"📊 R:R (средневзвешенный): <b>{trade_plan.rr_ratio:.1f} : 1</b>\n"
            f"📈 RSI: <b>{swept_rsi:.1f} ➔ {signal_rsi:.1f}</b> (дивергенция)\n\n"
        )

        w0_extra = ""
        if w0_rsi is not None:
            w0_extra = f" (RSI {w0_rsi:.1f}"
            if origin_div:
                w0_extra += " • W0-DIV ✅)"
            else:
                w0_extra += ")"

        w_struct_block = (
            f"━━━ <b>ВОЛНОВАЯ СТРУКТУРА</b> ━━━\n\n"
            f"W0: {_fmt_p(trade_plan.w0_price)}{w0_extra}\n"
            f"W1: {_fmt_p(trade_plan.w1_price)} │ W2: {_fmt_p(trade_plan.w2_price)}\n"
            f"W3: {_fmt_p(trade_plan.w3_price)} │ W4: {_fmt_p(trade_plan.w4_price)}\n"
            f"W5: {_fmt_p(trade_plan.w5_price)} (экстремум импульса)\n\n"
            f"📏 Длина импульса: <b>{_fmt_p(trade_plan.impulse_range)} ({imp_sign}{trade_plan.impulse_pct:.2f}%)</b>\n"
        )
        if dur_hours and dur_bars:
            w_struct_block += f"⏳ Формирование: <b>{dur_hours:.1f}ч</b> ({dur_bars} бар.)\n"
        if w3_longest:
            w_struct_block += "🔄 W3 = самая длинная волна ✅\n"
        w_struct_block += "\n"
    else:
        price_str = _fmt_p(signal_price)
        plan_block = (
            f"💰 Цена: <b>{price_str}</b>\n"
            f"📊 RSI: <b>{swept_rsi:.1f} ➔ {signal_rsi:.1f}</b>\n\n"
        )
        w_struct_block = ""

    msg = (
        f"{conflict_banner}"
        f"{header}\n\n"
        f"📌 <code>{symbol}</code> (Crypto Futures)\n"
        f"⏱️ Таймфрейм: <b>{interval}</b>\n\n"
        f"{plan_block}"
        f"{w_struct_block}"
        f"━━━ <b>ОБОСНОВАНИЕ</b> ━━━\n\n"
        f"{reason}\n\n"
        f"🎯 Оценка ИИ: <b>{score}/100</b> (Textbook)\n"
        f"📅 {dt_str}"
    )
    return msg



import threading
import time

_tg_lock = threading.Lock()

def _send_tg_request(method_url: str, is_json: bool = False, retry_count: int = 3, **kwargs) -> Optional[requests.Response]:
    """Вспомогательная функция для отправки запросов в Telegram с защитой от флуда и повторами."""
    for attempt in range(retry_count):
        try:
            if is_json:
                resp = requests.post(method_url, json=kwargs.get("json"), timeout=kwargs.get("timeout", 20))
            else:
                resp = requests.post(method_url, data=kwargs.get("data"), files=kwargs.get("files"), timeout=kwargs.get("timeout", 25))
            
            if resp.status_code == 200 and resp.json().get("ok"):
                return resp
            
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

