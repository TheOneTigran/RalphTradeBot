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
) -> str:
    """Формирует текст сообщения для Telegram с полноценным торговым планом и HTML-разметкой."""
    dt = detection_time or datetime.now()
    dt_str = dt.strftime("%d.%m.%Y в %H:%M")
    
    if direction == "SHORT":
        header = "🔴 <b>МЕДВЕЖИЙ ИМПУЛЬС ЗАВЕРШЁН</b>"
        dir_badge = "▼ <b>SHORT</b>"
        sl_note = "└ За экстремумом W5"
        imp_sign = "+"
    else:
        header = "🟢 <b>БЫЧИЙ ИМПУЛЬС ЗАВЕРШЁН</b>"
        dir_badge = "▲ <b>LONG</b>"
        sl_note = "└ За экстремумом W5"
        imp_sign = "-"

    # Если есть TradePlan
    if trade_plan is not None:
        entry_s = _fmt_p(trade_plan.entry_price)
        sl_s = _fmt_p(trade_plan.sl_price)
        tp1_s = _fmt_p(trade_plan.tp1_price)
        tp2_s = _fmt_p(trade_plan.tp2_price)
        tp3_s = _fmt_p(trade_plan.tp3_price)
        tp4_s = _fmt_p(trade_plan.tp4_price)

        plan_block = (
            f"━━━ <b>ТОРГОВЫЙ ПЛАН</b> ━━━\n\n"
            f"{dir_badge} │ Вход: <b>{entry_s}</b>\n"
            f"🛑 SL: <b>{sl_s}</b> (−{trade_plan.sl_pct:.2f}%)\n"
            f"   {sl_note}\n\n"
            f"🎯 TP1: <b>{tp1_s}</b> (+{trade_plan.tp1_pct:.2f}%) → Закрыть 25% (23.6% Фибо)\n"
            f"🎯 TP2: <b>{tp2_s}</b> (+{trade_plan.tp2_pct:.2f}%) → Закрыть 35% (38.2% Фибо)\n"
            f"🎯 TP3: <b>{tp3_s}</b> (+{trade_plan.tp3_pct:.2f}%) → Закрыть 25% (50.0% Фибо)\n"
            f"🎯 TP4: <b>{tp4_s}</b> (+{trade_plan.tp4_pct:.2f}%) → Закрыть 15% (61.8% Фибо)\n\n"
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
        f"{header}\n\n"
        f"📌 <code>{symbol}</code> (Crypto)\n"
        f"⏱️ <b>{interval}</b>\n\n"
        f"{plan_block}"
        f"{w_struct_block}"
        f"━━━ <b>ОБОСНОВАНИЕ</b> ━━━\n\n"
        f"{reason}\n\n"
        f"🎯 Оценка ИИ: <b>{score}/100</b> (Textbook)\n"
        f"📅 {dt_str}"
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
    trade_plan: Optional[Any] = None,
    dur_bars: Optional[int] = None,
    dur_hours: Optional[float] = None,
    w0_rsi: Optional[float] = None,
    w0_status: Optional[str] = None,
    origin_div: bool = False,
    w3_longest: bool = True,
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
        trade_plan=trade_plan,
        dur_bars=dur_bars,
        dur_hours=dur_hours,
        w0_rsi=w0_rsi,
        w0_status=w0_status,
        origin_div=origin_div,
        w3_longest=w3_longest,
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
