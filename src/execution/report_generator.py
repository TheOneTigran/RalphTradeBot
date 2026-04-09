"""
report_generator.py — Генератор аналитических отчетов.

Полностью исключена логика сайзинга и исполнения.
Фокус на аргументированных сигналах с Fibo, кластерами, ликвидностью
и дивергенциями для принятия решения человеком.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from src.events.models import SignalGeneratedEvent

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Генерирует подробные отчеты в Telegram."""

    @staticmethod
    def format_telegram_message(signal: SignalGeneratedEvent) -> str:
        """
        Форматирует SignalGeneratedEvent в профессиональный, 
        аргументированный текст для Telegram.
        """
        direction_emoji = "🟢 LONG" if signal.direction == "LONG" else "🔴 SHORT"
        prob = signal.probability_score * 100
        
        # Разбор фич
        c = signal.confluence_triggers
        
        # Fibo
        fibo_args = []
        if c.get("fibo_dist_618", 1.0) < 0.05:
            fibo_args.append("Golden Ratio (0.618)")
        elif c.get("fibo_dist_500", 1.0) < 0.05:
            fibo_args.append("Equilibrium (0.500)")
        elif c.get("fibo_dist_382", 1.0) < 0.05:
            fibo_args.append("Shallow Retrace (0.382)")
        else:
            fibo_args.append("Внутри зоны входа")
        fibo_str = " / ".join(fibo_args)

        # Liquidity
        sweep = c.get("liquidity_swept", c.get("liquidity_sweep", 0.0) == 1.0)
        liq_str = "✅ Снята целевая ликвидность (Sweep)" if sweep else "❌ Нет свипа"

        # Clusters & Volumes
        vol_z = c.get("cluster_volume_zscore", 0.0)
        absorption = c.get("absorption_detected", False)
        if absorption:
            vol_str = f"✅ Тормозящий объем (z-score: {vol_z:.1f})"
        elif vol_z > 1.5:
            vol_str = f"⚠️ Повышенный объем без явного торможения (z-score: {vol_z:.1f})"
        else:
            vol_str = f"❌ Обычный объем (z-score: {vol_z:.1f})"

        # OI & RSI
        oi_div = c.get("oi_divergence_flag", 0.0)
        if oi_div > 0:
            oi_str = "✅ Падение OI на проколе (Ликвидации!)"
        elif oi_div < 0:
            oi_str = "❌ Рост OI на проколе (Истинный пробой!)"
        else:
            oi_str = "➖ OI нейтрален / недоступен"

        # Формирование текста
        lines = [
            f"**{direction_emoji} {signal.symbol}**",
            f"Гипотеза: Завершается волна {signal.current_wave_hypothesis}",
            f"Степень: {signal.trend_degree}",
            f"Уверенность ML: **{prob:.0f}%**",
            "",
            "**Аргументы (Confluence):**",
            f"📍 Fibo: {fibo_str}",
            f"🔍 Ликвидность: {liq_str}",
            f"📊 Кластер: {vol_str}",
            f"〽️ Открытый Интерес: {oi_str}",
        ]

        if "sniper_trigger" in c and c["sniper_trigger"]:
            lines.insert(1, "⚡ **Early Warning (Sniper Trigger)** ⚡")
            lines.insert(2, "Вход до закрытия свечи!")
            
        lines.extend([
            "",
            "**Торговый План:**",
            f"Входная зона: {signal.entry_zone[0]} — {signal.entry_zone[1]}",
            f"Отмена гипотезы (Стоп): **{signal.invalidation_stop}**",
        ])
        
        if signal.take_profit_targets:
            lines.append("Цели (Фибо-расширения):")
            for i, tp in enumerate(signal.take_profit_targets):
                lines.append(f" 🎯 TP{i+1}: {tp}")

        return "\n".join(lines)

    @staticmethod
    def send_telegram_alert(message: str) -> None:
        """
        Отправляет сформированное сообщение в Telegram, если настроены ключи.
        """
        import os
        import requests
        
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        if not token or not chat_id:
            logger.debug("Telegram credentials not found, alert not sent.")
            return
            
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code != 200:
                logger.error("Telegram Error: %s", resp.text)
        except Exception as e:
            logger.error("Failed to send Telegram alert: %s", e)
