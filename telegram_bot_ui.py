"""
telegram_bot_ui.py — Интерактивный Telegram-интерфейс управления и аналитики RalphTradeBot.

Функционал:
1. Подробная, красиво структурированная и полная статистика (PnL, Win Rate, PF, сделки, просадка, по ТФ и монетам).
2. Мониторинг активных сделок в реальном времени (вход, текущая цена, плавающий PnL, цели TP1-TP4, безубыток).
3. Интерактивное добавление и отключение монет (в 1 клик через inline-кнопки или команды /add, /del).
4. Управление таймфреймами (15m, 1h, 5m, 4h).
5. Настройка капитала: депозит, риск ($ или %), кредитное плечо и порог ИИ.
6. Работает параллельно с live_scanner.py через автономный long-polling поток на requests.
"""
from __future__ import annotations

import html
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TOP_12_LEADERS,
    TOP_30_SYMBOLS,
    COMMISSION_RATE,
)
from runtime_settings import (
    get_runtime_config,
    toggle_symbol,
    add_symbol,
    remove_symbol,
    reset_symbols_to_default,
    toggle_timeframe,
    set_deposit,
    set_risk_usd,
    set_risk_pct,
    set_leverage,
    set_min_score,
)
from analytics_db import (
    get_detailed_statistics,
    get_active_trades_detailed,
    get_analytics_summary,
    reset_analytics_data,
)

logger = logging.getLogger("RalphTelegramBotUI")

POPULAR_COINS = [
    "NEARUSDT", "SUIUSDT", "KASUSDT", "DOGEUSDT",
    "UNIUSDT", "APTUSDT", "OPUSDT", "1000PEPEUSDT",
    "LTCUSDT", "BCHUSDT", "TAOUSDT", "ICPUSDT",
    "SOLUSDT", "ETHUSDT", "BTCUSDT", "AVAXUSDT",
]


def _esc(val: Any) -> str:
    """Экранирует спецсимволы для безопасного HTML Telegram."""
    return html.escape(str(val))


def _fmt_price(p: float) -> str:
    """Форматирует цену актива."""
    if p >= 1000.0:
        return f"${p:,.2f}"
    elif p >= 1.0:
        return f"${p:,.4f}"
    else:
        return f"${p:,.6f}"


class TelegramUIController:
    """Контроллер интерактивного бота на long-polling."""

    def __init__(self, token: str = TELEGRAM_BOT_TOKEN):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RalphTradeBot-UI/2.3"})
        self.running = False
        self.last_update_id = 0
        self._thread: Optional[threading.Thread] = None
        self.setup_bot_commands()

    def setup_bot_commands(self):
        """Регистрирует список команд в нативном меню Telegram (синяя кнопка Menu)."""
        commands = [
            {"command": "menu", "description": "🏠 Главный терминал управления"},
            {"command": "stats", "description": "📊 Полная статистика PnL и Win Rate"},
            {"command": "trades", "description": "🎯 Активные сделки в трекере"},
            {"command": "coins", "description": "🪙 Управление монетами (вкл/выкл)"},
            {"command": "tf", "description": "⏱ Таймфреймы (15m, 1h, 5m, 4h)"},
            {"command": "settings", "description": "⚙️ Депозит, риск и плечо"},
            {"command": "reset", "description": "🗑 Сбросить историю и статистику"},
            {"command": "help", "description": "📖 Справка и команды"},
        ]
        self._call("setMyCommands", {"commands": commands})

    def _call(self, method: str, payload: Dict[str, Any], timeout: int = 15) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/{method}"
        try:
            resp = self.session.post(url, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    return data.get("result")
            elif resp.status_code == 429:
                wait_s = resp.json().get("parameters", {}).get("retry_after", 3)
                logger.warning(f"Telegram 429: пауза {wait_s}с...")
                time.sleep(wait_s + 0.5)
            else:
                logger.debug(f"Telegram API {method} error: {resp.text[:180]}")
        except Exception as e:
            logger.debug(f"Сетевая ошибка при вызове {method}: {e}")
        return None

    def send_message(self, chat_id: int | str, text: str, reply_markup: Optional[Dict[str, Any]] = None) -> Optional[int]:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        res = self._call("sendMessage", payload)
        return res.get("message_id") if res else None

    def edit_message(self, chat_id: int | str, message_id: int, text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        res = self._call("editMessageText", payload)
        return res is not None

    def answer_callback(self, callback_id: str, text: Optional[str] = None, show_alert: bool = False):
        payload = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text
            payload["show_alert"] = show_alert
        self._call("answerCallbackQuery", payload)

    # ═══════════════════════════════════════════════════════════════════════════
    # Генераторы экранов
    # ═══════════════════════════════════════════════════════════════════════════

    def render_main_menu(self) -> Tuple[str, Dict[str, Any]]:
        cfg = get_runtime_config()
        active_trades = get_active_trades_detailed()
        stats = get_detailed_statistics("all")

        status_icon = "⏸ <b>ПАУЗА</b>" if cfg.get("scanner_paused") else "🟢 <b>АКТИВЕН (24/7)</b>"
        pnl_sign = "+" if stats["net_pnl_usd"] >= 0 else ""
        pnl_emoji = "🟢" if stats["net_pnl_usd"] >= 0 else "🔴"

        text = (
            f"🤖 <b>RalphTradeBot — Терминал управления & Аналитика</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📡 <b>Статус сканера:</b> {status_icon}\n"
            f"⏱ <b>Таймфреймы:</b> <code>{', '.join(cfg['live_timeframes'])}</code>\n"
            f"🪙 <b>Монет в пуле:</b> <b>{len(cfg['active_symbols'])} пар</b>\n"
            f"💰 <b>Депозит:</b> ${cfg['deposit_usd']:,.0f} | <b>Риск:</b> ${cfg['risk_budget_usd']:,.1f} ({cfg['risk_pct']}%)\n"
            f"⚡ <b>Плечо:</b> {cfg['default_leverage']}x | <b>ИИ порог:</b> ≥{cfg['min_score']}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 <b>Открытых сделок:</b> <b>{len(active_trades)}</b>\n"
            f"{pnl_emoji} <b>Общий PnL (Net):</b> <b>{pnl_sign}${stats['net_pnl_usd']:,.2f} ({pnl_sign}{stats['net_pnl_r']:.1f}R)</b>\n"
            f"🏆 <b>Win Rate:</b> <b>{stats['win_rate']}%</b> (PF: <b>{stats['profit_factor']:.2f}</b>)\n\n"
            f"<i>Выберите нужный раздел в меню ниже:</i>"
        )

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "📊 Статистика & PnL", "callback_data": "screen:stats:all"},
                    {"text": f"🎯 Сделки ({len(active_trades)})", "callback_data": "screen:trades"},
                ],
                [
                    {"text": f"🪙 Монеты ({len(cfg['active_symbols'])})", "callback_data": "screen:coins"},
                    {"text": f"⏱ Таймфреймы ({len(cfg['live_timeframes'])})", "callback_data": "screen:tf"},
                ],
                [
                    {"text": "⚙️ Депозит & Риск", "callback_data": "screen:settings"},
                    {"text": "🔄 Обновить", "callback_data": "screen:main"},
                ],
            ]
        }
        return text, keyboard

    def render_stats_screen(self, period: str = "all") -> Tuple[str, Dict[str, Any]]:
        st = get_detailed_statistics(period)

        period_names = {
            "all": "🌐 За всё время",
            "today": "📅 За сегодня",
            "7d": "🗓 За 7 дней",
            "tf": "⏱ По таймфреймам",
            "coins": "🪙 Рейтинг монет",
        }
        period_title = period_names.get(period, "🌐 Статистика")

        if period in ("all", "today", "7d"):
            pnl_sign = "+" if st["net_pnl_usd"] >= 0 else ""
            pnl_emoji = "🟢" if st["net_pnl_usd"] >= 0 else "🔴"
            wr = st["win_rate"]
            wr_bar = "🟩" * int(wr // 10) + "⬜" * (10 - int(wr // 10))

            text = (
                f"📊 <b>ПОЛНАЯ СТАТИСТИКА ({period_title.upper()})</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{pnl_emoji} <b>Чистый PnL (Net):</b> <b>{pnl_sign}${st['net_pnl_usd']:,.2f}</b> (<b>{pnl_sign}{st['net_pnl_r']:.1f}R</b>)\n"
                f"└ <i>(за вычетом биржевых комиссий {COMMISSION_RATE*100*2:.1f}% roundtrip)</i>\n\n"
                f"🏆 <b>Win Rate:</b> <b>{wr}%</b>\n"
                f"<code>{wr_bar}</code>\n"
                f"├ <b>Profit Factor:</b> <b>{st['profit_factor']:.2f}</b>\n"
                f"├ <b>Full TP Win Rate:</b> <b>{st['full_tp_wr']}%</b>\n"
                f"└ <b>Ср. прибыль:</b> +${st['avg_win_usd']:.2f} │ <b>Ср. убыток:</b> -${st['avg_loss_usd']:.2f}\n\n"
                f"📈 <b>Количество сделок:</b>\n"
                f"├ Всего сигналов: <b>{st['total_signals']}</b>\n"
                f"├ В процессе (Active): <b>{st['active_trades']}</b>\n"
                f"├ Завершено (Closed): <b>{st['closed_trades']}</b>\n"
                f"└ <b>{st['wins']}</b> W (прибыль/БУ) │ <b>{st['losses']}</b> L (стоп) │ <b>{st['be_count']}</b> БУ\n\n"
                f"🎯 <b>Достижение целей (Fibonacci):</b>\n"
                f"├ <b>TP1 (23.6%):</b> {st['tp1_count']} раз (фиксация 40%)\n"
                f"├ <b>TP2 (38.2%):</b> {st['tp2_count']} раз (+30% и БУ)\n"
                f"├ <b>TP3 (50.0%):</b> {st['tp3_count']} раз (+30% runner)\n"
                f"├ <b>TP4 (61.8%):</b> {st['tp4_count']} раз\n"
                f"├ <b>Stop Loss:</b> {st['sl_count']} раз\n"
                f"└ <b>Fail-Fast (3 бара):</b> {st['fail_fast_count']} раз\n"
            )
            if st["best_trade_usd"] != 0:
                text += (
                    f"\n⭐ <b>Рекорды:</b>\n"
                    f"├ Лучший трейд: <b>+${st['best_trade_usd']:,.2f}</b>\n"
                    f"└ Худший трейд: <b>-${abs(st['worst_trade_usd']):,.2f}</b>\n"
                )

        elif period == "tf":
            text = (
                f"⏱ <b>СТАТИСТИКА ПО ТАЙМФРЕЙМАМ</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"<i>Сравнение эффективности импульсов на разных горизонтах:</i>\n\n"
            )
            tfs = st.get("tf_breakdown", {})
            if not tfs:
                text += "<i>Пока нет накопленных данных по таймфреймам.</i>\n"
            else:
                for tf_name, data in tfs.items():
                    sign = "+" if data["net_pnl_usd"] >= 0 else ""
                    em = "🟢" if data["net_pnl_usd"] >= 0 else "🔴"
                    text += (
                        f"⏰ <b>Таймфрейм {tf_name.upper()}:</b>\n"
                        f"├ Сигналов: <b>{data['signals']}</b> │ WR: <b>{data['win_rate']}%</b>\n"
                        f"├ Исходы: <b>{data['wins']}</b>W / <b>{data['losses']}</b>L\n"
                        f"└ {em} PnL: <b>{sign}${data['net_pnl_usd']:,.2f}</b>\n\n"
                    )

        elif period == "coins":
            text = (
                f"🪙 <b>РЕЙТИНГ МОНЕТ ПО ПРИБЫЛЬНОСТИ</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            )
            syms = st.get("symbol_breakdown", [])
            if not syms:
                text += "<i>Пока нет завершённых сделок по монетам.</i>\n"
            else:
                medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
                for idx, item in enumerate(syms[:10]):
                    medal = medals[idx] if idx < len(medals) else "▫️"
                    sign = "+" if item["net_pnl_usd"] >= 0 else ""
                    text += (
                        f"{medal} <code>{item['symbol']}</code>: <b>{sign}${item['net_pnl_usd']:,.2f}</b>\n"
                        f"   └ Сигналов: {item['signals']} │ WR: {item['win_rate']}% │ TP2: {item['tp2_count']}\n"
                    )

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "🌐 Все время", "callback_data": "screen:stats:all"},
                    {"text": "📅 Сегодня", "callback_data": "screen:stats:today"},
                    {"text": "🗓 7 дней", "callback_data": "screen:stats:7d"},
                ],
                [
                    {"text": "⏱ По ТФ", "callback_data": "screen:stats:tf"},
                    {"text": "🪙 Рейтинг монет", "callback_data": "screen:stats:coins"},
                ],
                [
                    {"text": "🗑 Сбросить статистику", "callback_data": "confirm:reset_stats"},
                ],
                [
                    {"text": "🔄 Обновить", "callback_data": f"screen:stats:{period}"},
                    {"text": "🔙 В главное меню", "callback_data": "screen:main"},
                ],
            ]
        }
        return text, keyboard

    def render_confirm_reset_screen(self) -> Tuple[str, Dict[str, Any]]:
        text = (
            f"⚠️ <b>ПОДТВЕРЖДЕНИЕ СБРОСА СТАТИСТИКИ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Вы действительно хотите обнулить всю накопленную историю сигналов, сделок и PnL?\n\n"
            f"• <b>Будет удалено:</b> вся история сигналов, результаты закрытых сделок, счетчики винрейта и кэш.\n"
            f"• <b>Будет сохранено:</b> ваши персональные настройки (выбранные монеты, таймфреймы, депозит, риск и плечо).\n\n"
            f"<i>Это действие необратимо. Отсчёт статистики начнётся с чистого листа.</i>"
        )
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "🔴 Да, сбросить статистику", "callback_data": "action:do_reset_stats"},
                ],
                [
                    {"text": "🟢 Отмена (Назад в статистику)", "callback_data": "screen:stats:all"},
                ],
            ]
        }
        return text, keyboard

    def render_active_trades_screen(self) -> Tuple[str, Dict[str, Any]]:
        trades = get_active_trades_detailed()

        if not trades:
            text = (
                f"🎯 <b>АКТИВНЫЕ СДЕЛКИ В РЕАЛЬНОМ ВРЕМЕНИ</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"🟢 <b>Сейчас открытых позиций нет.</b>\n"
                f"Сканер непрерывно проверяет графики в поиске завершённых волн 5.\n\n"
                f"При появлении эталонного сигнала (Score ≥ 85%) бот мгновенно отправит график, разволновку и карточку входа."
            )
        else:
            text = (
                f"🎯 <b>АКТИВНЫЕ СДЕЛКИ В ТРЕКЕРЕ ({len(trades)})</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            )
            for idx, t in enumerate(trades, 1):
                sym = t["symbol"]
                tf = t["interval"]
                dir_str = t["direction"]
                em_dir = "🟢 LONG" if dir_str == "LONG" else "🔴 SHORT"
                entry = float(t["entry_price"])
                cur = float(t["current_price"] or entry)
                pnl_usd = t["cur_pnl_usd"]
                pnl_pct = t["cur_pnl_pct"]
                pnl_sign = "+" if pnl_usd >= 0 else ""
                pnl_em = "🟩" if pnl_usd >= 0 else "🟥"

                tp1_st = "✅ ВЗЯТ" if t["tp1_hit"] else f"⏳ {_fmt_price(t['tp1_price'])}"
                tp2_st = "✅ ВЗЯТ" if t["tp2_hit"] else f"⏳ {_fmt_price(t['tp2_price'])}"
                be_st = "🛡 В БЕЗУБЫТКЕ" if t["be_triggered"] else f"{_fmt_price(t['sl_price'])}"

                text += (
                    f"<b>#{idx} <code>{sym}</code> ({tf} {em_dir})</b>\n"
                    f"├ Вход: <b>{_fmt_price(entry)}</b> │ Тек: <b>{_fmt_price(cur)}</b>\n"
                    f"├ {pnl_em} <b>PnL:</b> <b>{pnl_sign}${pnl_usd:,.2f} ({pnl_sign}{pnl_pct:.2f}%)</b>\n"
                    f"├ Объём: ${t['pos_size_usd']:,.1f} │ Риск: ${t['risk_usd']:.1f}\n"
                    f"├ TP1 (40%): {tp1_st}\n"
                    f"├ TP2 (30%): {tp2_st}\n"
                    f"└ Стоп: <b>{be_st}</b>\n\n"
                )

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "🔄 Обновить котировки", "callback_data": "screen:trades"},
                    {"text": "🔙 В главное меню", "callback_data": "screen:main"},
                ]
            ]
        }
        return text, keyboard

    def render_coins_screen(self) -> Tuple[str, Dict[str, Any]]:
        cfg = get_runtime_config()
        active = set(cfg["active_symbols"])

        text = (
            f"🪙 <b>УПРАВЛЕНИЕ МОНЕТАМИ СКАНЕРА</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Всего в активном пуле: <b>{len(active)} монет</b>\n"
            f"🟢 — монета сканируется в реальном времени\n"
            f"⚪ — монета отключена (нажмите, чтобы включить)\n\n"
            f"<i>Нажмите на монету для мгновенного переключения (ON/OFF) или отправьте команду:</i>\n"
            f"• <code>/add SOLUSDT</code> — добавить любую пару\n"
            f"• <code>/del NEARUSDT</code> — удалить пару\n"
        )

        buttons = []
        row = []
        for sym in POPULAR_COINS:
            short_name = sym.replace("USDT", "")
            icon = "🟢" if sym in active else "⚪"
            btn_text = f"{icon} {short_name}"
            row.append({"text": btn_text, "callback_data": f"toggle_coin:{sym}"})
            if len(row) == 4:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        buttons.append([
            {"text": "♻️ Сбросить к ТОП-12 Лидерам", "callback_data": "action:reset_coins"},
        ])
        buttons.append([
            {"text": "🔄 Обновить", "callback_data": "screen:coins"},
            {"text": "🔙 В главное меню", "callback_data": "screen:main"},
        ])

        return text, {"inline_keyboard": buttons}

    def render_timeframes_screen(self) -> Tuple[str, Dict[str, Any]]:
        cfg = get_runtime_config()
        active_tfs = set(cfg["live_timeframes"])

        text = (
            f"⏱ <b>УПРАВЛЕНИЕ ТАЙМФРЕЙМАМИ</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Активные таймфреймы: <b>{', '.join(cfg['live_timeframes'])}</b>\n\n"
            f"• <b>15m (Рекомендуется)</b> — Золотой стандарт для внутридневной торговли. Высокая точность W5 (PF 2.16-2.69).\n"
            f"• <b>1h (Рекомендуется)</b> — Идеально для спокойного свинг-трейдинга. Низкий шум и высокая прибыль на сделку.\n"
            f"• <b>5m (Шумный)</b> — Высокая частота сделок (55/день), требует быстрой реакции.\n"
            f"• <b>4h (Макро)</b> — Редкие эталонные циклы разворота крупных трендов.\n\n"
            f"<i>Нажмите на кнопку, чтобы включить или отключить ТФ:</i>"
        )

        buttons = [
            [
                {
                    "text": ("✅ 15m (Интрадей)" if "15m" in active_tfs else "❌ 15m"),
                    "callback_data": "toggle_tf:15m",
                },
                {
                    "text": ("✅ 1h (Свинг)" if "1h" in active_tfs else "❌ 1h"),
                    "callback_data": "toggle_tf:1h",
                },
            ],
            [
                {
                    "text": ("✅ 5m (Скальпинг)" if "5m" in active_tfs else "❌ 5m"),
                    "callback_data": "toggle_tf:5m",
                },
                {
                    "text": ("✅ 4h (Макро)" if "4h" in active_tfs else "❌ 4h"),
                    "callback_data": "toggle_tf:4h",
                },
            ],
            [
                {"text": "🔙 В главное меню", "callback_data": "screen:main"},
            ],
        ]
        return text, {"inline_keyboard": buttons}

    def render_settings_screen(self) -> Tuple[str, Dict[str, Any]]:
        cfg = get_runtime_config()

        dep = cfg["deposit_usd"]
        risk = cfg["risk_budget_usd"]
        pct = cfg["risk_pct"]
        lev = cfg["default_leverage"]
        score = cfg["min_score"]

        text = (
            f"⚙️ <b>НАСТРОЙКИ ДЕПОЗИТА И РИСК-МЕНЕДЖМЕНТА</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 <b>Торговый депозит:</b> <b>${dep:,.0f}</b>\n"
            f"🛡 <b>Риск на сделку:</b> <b>${risk:,.1f}</b> (<b>{pct}%</b> от депо)\n"
            f"⚡ <b>Кредитное плечо:</b> <b>{lev}x</b>\n"
            f"🧠 <b>Порог качества Vision AI:</b> <b>≥{score}%</b>\n\n"
            f"<i>Выберите быстрый пресет кнопками или введите команду:</i>\n"
            f"• <code>/set_dep 2500</code> — установить точный баланс депозита\n"
            f"• <code>/set_risk 25</code> — установить фиксированный риск ($)\n"
            f"• <code>/set_lev 15</code> — изменить кредитное плечо\n"
            f"• <code>/set_score 85</code> — изменить порог нейросети"
        )

        buttons = [
            # Депозиты
            [
                {"text": "💵 $500", "callback_data": "set_dep:500"},
                {"text": "💵 $1,000", "callback_data": "set_dep:1000"},
                {"text": "💵 $2,500", "callback_data": "set_dep:2500"},
                {"text": "💵 $5,000", "callback_data": "set_dep:5000"},
            ],
            # Риски в $
            [
                {"text": "🛡 $5", "callback_data": "set_risk:5"},
                {"text": "🛡 $10", "callback_data": "set_risk:10"},
                {"text": "🛡 $25", "callback_data": "set_risk:25"},
                {"text": "🛡 $50", "callback_data": "set_risk:50"},
            ],
            # Риски в %
            [
                {"text": "📊 0.5%", "callback_data": "set_pct:0.5"},
                {"text": "📊 1.0%", "callback_data": "set_pct:1.0"},
                {"text": "📊 2.0%", "callback_data": "set_pct:2.0"},
                {"text": "📊 3.0%", "callback_data": "set_pct:3.0"},
            ],
            # Плечо
            [
                {"text": "⚡ 5x", "callback_data": "set_lev:5"},
                {"text": "⚡ 10x", "callback_data": "set_lev:10"},
                {"text": "⚡ 15x", "callback_data": "set_lev:15"},
                {"text": "⚡ 20x", "callback_data": "set_lev:20"},
            ],
            # Порог ИИ
            [
                {"text": "🧠 80% (Мягкий)", "callback_data": "set_score:80"},
                {"text": "🧠 85% (Эталон)", "callback_data": "set_score:85"},
                {"text": "🧠 90% (Строгий)", "callback_data": "set_score:90"},
            ],
            [
                {"text": "🔄 Обновить", "callback_data": "screen:settings"},
                {"text": "🔙 В главное меню", "callback_data": "screen:main"},
            ],
        ]
        return text, {"inline_keyboard": buttons}

    # ═══════════════════════════════════════════════════════════════════════════
    # Обработчик Callback Query
    # ═══════════════════════════════════════════════════════════════════════════

    def handle_callback(self, cb: Dict[str, Any]):
        cb_id = cb["id"]
        data = cb.get("data", "")
        msg = cb.get("message", {})
        chat_id = msg.get("chat", {}).get("id")
        msg_id = msg.get("message_id")

        if not chat_id or not msg_id:
            self.answer_callback(cb_id)
            return

        toast_msg = None

        if data == "screen:main":
            text, kb = self.render_main_menu()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data.startswith("screen:stats:"):
            period = data.split(":")[-1]
            text, kb = self.render_stats_screen(period)
            self.edit_message(chat_id, msg_id, text, kb)

        elif data == "screen:trades":
            text, kb = self.render_active_trades_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data == "screen:coins":
            text, kb = self.render_coins_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data.startswith("toggle_coin:"):
            sym = data.split(":", 1)[1]
            is_active, toast_msg = toggle_symbol(sym)
            text, kb = self.render_coins_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data == "action:reset_coins":
            reset_symbols_to_default()
            toast_msg = "♻️ Пул монет сброшен к ТОП-12 лидерам!"
            text, kb = self.render_coins_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data == "screen:tf":
            text, kb = self.render_timeframes_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data.startswith("toggle_tf:"):
            tf = data.split(":", 1)[1]
            is_active, toast_msg = toggle_timeframe(tf)
            text, kb = self.render_timeframes_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data == "screen:settings":
            text, kb = self.render_settings_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data == "confirm:reset_stats":
            text, kb = self.render_confirm_reset_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data == "action:do_reset_stats":
            reset_analytics_data()
            toast_msg = "✅ Вся статистика успешно сброшена!"
            text, kb = self.render_stats_screen("all")
            self.edit_message(chat_id, msg_id, text, kb)

        elif data.startswith("set_dep:"):
            val = float(data.split(":")[1])
            set_deposit(val)
            toast_msg = f"💰 Депозит обновлён: ${val:,.0f}"
            text, kb = self.render_settings_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data.startswith("set_risk:"):
            val = float(data.split(":")[1])
            set_risk_usd(val)
            toast_msg = f"🛡 Риск на сделку: ${val:,.0f}"
            text, kb = self.render_settings_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data.startswith("set_pct:"):
            val = float(data.split(":")[1])
            risk_usd = set_risk_pct(val)
            toast_msg = f"📊 Риск: {val:.1f}% (${risk_usd:.1f})"
            text, kb = self.render_settings_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data.startswith("set_lev:"):
            val = int(data.split(":")[1])
            set_leverage(val)
            toast_msg = f"⚡ Кредитное плечо: {val}x"
            text, kb = self.render_settings_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        elif data.startswith("set_score:"):
            val = int(data.split(":")[1])
            set_min_score(val)
            toast_msg = f"🧠 Порог ИИ: {val}%"
            text, kb = self.render_settings_screen()
            self.edit_message(chat_id, msg_id, text, kb)

        self.answer_callback(cb_id, text=toast_msg)

    # ═══════════════════════════════════════════════════════════════════════════
    # Обработчик текстовых команд
    # ═══════════════════════════════════════════════════════════════════════════

    def handle_text_message(self, msg: Dict[str, Any]):
        chat_id = msg.get("chat", {}).get("id")
        text = (msg.get("text") or "").strip()

        if not chat_id or not text:
            return

        cmd = text.split()[0].lower() if text else ""

        if cmd in ("/start", "/menu", "меню", "start"):
            msg_text, kb = self.render_main_menu()
            self.send_message(chat_id, msg_text, kb)

        elif cmd in ("/stats", "/pnl", "статистика"):
            msg_text, kb = self.render_stats_screen("all")
            self.send_message(chat_id, msg_text, kb)

        elif cmd in ("/trades", "/positions", "сделки"):
            msg_text, kb = self.render_active_trades_screen()
            self.send_message(chat_id, msg_text, kb)

        elif cmd in ("/coins", "/symbols", "монеты"):
            msg_text, kb = self.render_coins_screen()
            self.send_message(chat_id, msg_text, kb)

        elif cmd in ("/tf", "/timeframes", "таймфреймы"):
            msg_text, kb = self.render_timeframes_screen()
            self.send_message(chat_id, msg_text, kb)

        elif cmd in ("/settings", "/risk", "настройки"):
            msg_text, kb = self.render_settings_screen()
            self.send_message(chat_id, msg_text, kb)

        elif cmd in ("/reset_stats", "/clear_stats", "/reset", "сброс"):
            msg_text, kb = self.render_confirm_reset_screen()
            self.send_message(chat_id, msg_text, kb)

        elif cmd == "/add":
            parts = text.split()
            if len(parts) >= 2:
                ok, note = add_symbol(parts[1])
                self.send_message(chat_id, note)
            else:
                self.send_message(chat_id, "⚠️ Использование: <code>/add SOLUSDT</code>")

        elif cmd in ("/del", "/remove"):
            parts = text.split()
            if len(parts) >= 2:
                ok, note = remove_symbol(parts[1])
                self.send_message(chat_id, note)
            else:
                self.send_message(chat_id, "⚠️ Использование: <code>/del SOLUSDT</code>")

        elif cmd in ("/set_dep", "/dep", "/deposit"):
            parts = text.split()
            if len(parts) >= 2 and parts[1].replace(".", "", 1).isdigit():
                val = float(parts[1])
                set_deposit(val)
                self.send_message(chat_id, f"✅ Торговый депозит установлен: <b>${val:,.2f}</b>")
            else:
                self.send_message(chat_id, "⚠️ Использование: <code>/set_dep 2500</code>")

        elif cmd in ("/set_risk", "/risk_usd"):
            parts = text.split()
            if len(parts) >= 2 and parts[1].replace(".", "", 1).isdigit():
                val = float(parts[1])
                set_risk_usd(val)
                self.send_message(chat_id, f"✅ Риск на сделку установлен: <b>${val:,.2f}</b>")
            else:
                self.send_message(chat_id, "⚠️ Использование: <code>/set_risk 25</code>")

        elif cmd in ("/set_lev", "/leverage"):
            parts = text.split()
            if len(parts) >= 2 and parts[1].isdigit():
                val = int(parts[1])
                set_leverage(val)
                self.send_message(chat_id, f"✅ Кредитное плечо установлено: <b>{val}x</b>")
            else:
                self.send_message(chat_id, "⚠️ Использование: <code>/set_lev 15</code>")

        elif cmd in ("/set_score", "/score"):
            parts = text.split()
            if len(parts) >= 2 and parts[1].isdigit():
                val = int(parts[1])
                set_min_score(val)
                self.send_message(chat_id, f"✅ Порог качества ИИ установлен: <b>{val}%</b>")
            else:
                self.send_message(chat_id, "⚠️ Использование: <code>/set_score 85</code>")

        elif cmd in ("/help", "помощь"):
            help_text = (
                f"📖 <b>Команды управления RalphTradeBot:</b>\n\n"
                f"• <code>/menu</code> — Главный дашборд управления\n"
                f"• <code>/stats</code> — Полная статистика PnL и Win Rate\n"
                f"• <code>/trades</code> — Текущие открытые позиции\n"
                f"• <code>/coins</code> — Меню переключения монет\n"
                f"• <code>/tf</code> — Выбор таймфреймов (15m, 1h, 5m, 4h)\n"
                f"• <code>/settings</code> — Настройки депо и плеча\n"
                f"• <code>/add BTCUSDT</code> — Добавить монету в пул\n"
                f"• <code>/del BTCUSDT</code> — Удалить монету из пула\n"
                f"• <code>/set_dep 2000</code> — Задать размер депозита ($)\n"
                f"• <code>/set_risk 25</code> — Задать риск на сделку ($)\n"
                f"• <code>/set_lev 10</code> — Задать кредитное плечо\n"
            )
            self.send_message(chat_id, help_text)

    # ═══════════════════════════════════════════════════════════════════════════
    # Long Polling цикл
    # ═══════════════════════════════════════════════════════════════════════════

    def run_polling(self):
        logger.info("🤖 Интерактивный Telegram-бот успешно запущен в режиме Long-Polling")
        self.running = True

        while self.running:
            try:
                payload = {
                    "offset": self.last_update_id + 1,
                    "timeout": 20,
                    "allowed_updates": ["message", "callback_query"],
                }
                res = self._call("getUpdates", payload, timeout=25)
                if res and isinstance(res, list):
                    for update in res:
                        self.last_update_id = update["update_id"]
                        if "callback_query" in update:
                            self.handle_callback(update["callback_query"])
                        elif "message" in update:
                            self.handle_text_message(update["message"])
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Ошибка в цикле Telegram long-polling: {e}")
                time.sleep(2.0)

    def start_in_thread(self) -> threading.Thread:
        """Запускает long-polling в фоновом потоке."""
        if self._thread and self._thread.is_alive():
            return self._thread
        self._thread = threading.Thread(target=self.run_polling, daemon=True, name="TelegramUIThread")
        self._thread.start()
        return self._thread

    def stop(self):
        self.running = False


_global_bot_controller: Optional[TelegramUIController] = None


def start_bot_ui_thread() -> Optional[threading.Thread]:
    """Глобальная точка входа для запуска UI-бота в потоке."""
    global _global_bot_controller
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN не задан, интерактивный бот не запущен.")
        return None
    _global_bot_controller = TelegramUIController(token=TELEGRAM_BOT_TOKEN)
    return _global_bot_controller.start_in_thread()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-5s │ %(message)s")
    bot = TelegramUIController()
    bot.run_polling()
