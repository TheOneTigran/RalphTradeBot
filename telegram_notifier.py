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
) -> str:
    """Формирует текст сообщения для Telegram с HTML-разметкой."""
    dt = detection_time or datetime.now()
    dt_str = dt.strftime("%d.%m.%Y в %H:%M")
    
    if direction == "SHORT":
        header = "🔴 <b>Медвежий импульс</b>"
        rsi_zone = "🔴 Зона перекупленности"
        arrow = "↘️"
        action = "⚠️ <b>Потенциальный разворот вниз</b>"
    else:
        header = "🟢 <b>Бычий импульс</b>"
        rsi_zone = "🟢 Зона перепроданности"
        arrow = "↗️"
        action = "⚠️ <b>Потенциальный разворот вверх</b>"

    # Форматирование цены
    price_str = f"${signal_price:,.4f}" if signal_price < 10.0 else f"${signal_price:,.2f}"

    msg = (
        f"{header}\n\n"
        f"📌 <code>{symbol}</code> (Crypto)\n"
        f"⏱️ {interval}\n"
        f"💰 {price_str}\n"
        f"📊 RSI Пивоты: {swept_rsi:.1f} {arrow} {signal_rsi:.1f} ({rsi_zone})\n"
        f"📈 Текущий RSI(14): {current_rsi:.1f}\n\n"
        f"{action}\n\n"
        f"📝 <b>Обоснование:</b>\n"
        f"{reason}\n\n"
        f"📅 Обнаружено: {dt_str}\n"
        f"🎯 Оценка ИИ: <b>{score}/100</b> (Textbook)"
    )
    return msg


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
    bot_token: str = TELEGRAM_BOT_TOKEN,
    chat_id: str = TELEGRAM_CHAT_ID,
    min_score: int = TELEGRAM_MIN_SCORE,
) -> bool:
    """
    Отправляет подтверждённый сигнал в Telegram.
    Строгий фильтр: отправляются только сигналы с score >= min_score (85%).
    """
    if score < min_score:
        logger.info(f"⏭️ Пропуск отправки в TG: Score {score} < {min_score}% (порог Textbook)")
        return False

    if not bot_token or not chat_id:
        logger.warning("⚠️ Не заданы TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID")
        return False

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
    )

    url_photo = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    # Если caption укладывается в лимит Telegram (1024 символа), шлем фото с caption
    if len(caption) <= 1024:
        files = {"photo": ("signal_chart.png", io.BytesIO(chart_png), "image/png")}
        data = {
            "chat_id": chat_id,
            "caption": caption,
            "parse_mode": "HTML",
        }
        try:
            resp = requests.post(url_photo, data=data, files=files, timeout=20)
            if resp.status_code == 200 and resp.json().get("ok"):
                logger.info(f"🚀 Сигнал {symbol} {direction} (Score {score}) успешно отправлен в Telegram!")
                return True
            else:
                logger.error(f"❌ Ошибка Telegram sendPhoto: {resp.text}")
        except Exception as e:
            logger.error(f"❌ Исключение при отправке фото в Telegram: {e}")
    else:
        # Если длинный текст, шлем сначала фото, затем отдельное сообщение
        try:
            files = {"photo": ("signal_chart.png", io.BytesIO(chart_png), "image/png")}
            resp1 = requests.post(url_photo, data={"chat_id": chat_id}, files=files, timeout=20)
            url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            resp2 = requests.post(url_msg, json={"chat_id": chat_id, "text": caption, "parse_mode": "HTML"}, timeout=15)
            if resp2.status_code == 200 and resp2.json().get("ok"):
                logger.info(f"🚀 Сигнал {symbol} {direction} (Score {score}) успешно отправлен в Telegram (двумя частями)!")
                return True
        except Exception as e:
            logger.error(f"❌ Исключение при отправке в Telegram: {e}")

    return False
