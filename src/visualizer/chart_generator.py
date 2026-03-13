"""
chart_generator.py — Модуль визуализации графиков.

Использует mplfinance для генерации графиков со свечами, 
векторами (трендовыми линиями) и горизонтальными зонами Fib-кластеров.
"""
from __future__ import annotations

import os
import logging
from typing import List, Dict

import re
import pandas as pd
import numpy as np
import mplfinance as mpf

from src.core.models import Vector, FibCluster, TradePlan

logger = logging.getLogger(__name__)


def generate_chart(
    symbol: str,
    timeframe: str,
    candles: List[Dict],
    vectors: List[Vector],
    fib_clusters: List[FibCluster],
    save_path: str,
    plan: TradePlan | None = None,
) -> str | None:
    """
    Генерирует и сохраняет график (PNG) с разметкой.

    Рисует:
      - OHLCV свечи.
      - Векторы (свинги) поверх цены (через alines).
      - Сильные кластеры Фибоначчи (hline).

    Args:
        symbol:       Торговая пара.
        timeframe:    Таймфрейм.
        candles:      Сырые свечи [{ts, open, high, low, close, volume}].
        vectors:      Список объектов Vector (рассчитанные свинги).
        fib_clusters: Список объектов FibCluster.
        save_path:    Путь для сохранения .png файла.

    Returns:
        Абсолютный путь к файлу или None при ошибке.
    """
    if not candles:
        logger.warning("Нет свечей для генерации графика %s", symbol)
        return None

    try:
        # 1. Подготовка DataFrame для mplfinance
        df = pd.DataFrame(candles)
        # ccxt возвращает ts в мс
        df['Date'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('Date', inplace=True)
        # Переименовываем колонки для mplfinance
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)

        # Оставляем только нужные колонки
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Используем полный датафрейм (обычно 500 свечей).
        # Это гарантирует, что все рассчитанные векторы лежат внутри графика.
        plot_df = df

        # 2. Подготовка линий векторов (alines)
        # Векторы заданы ts и price.
        # mplfinance alines формат: список списков кортежей (timestamp/datetime, price).
        # [(date1, price1), (date2, price2), ...]
        alines = []
        if vectors:
            # Создаем одну непрерывную линию зигзага (векторы соединены)
            # Отфильтруем только те векторы, которые попадают в окно plot_df
            first_plot_ts = plot_df.index[0]
            
            zigzag_line = []
            for vec in vectors:
                vec_start_dt = pd.to_datetime(vec.start_time, unit='ms')
                vec_end_dt = pd.to_datetime(vec.end_time, unit='ms')
                
                # Добавляем если вектор актуален для графика
                if vec_end_dt >= first_plot_ts:
                    if not zigzag_line:
                        zigzag_line.append((vec_start_dt, vec.start_price))
                    zigzag_line.append((vec_end_dt, vec.end_price))
            
            if zigzag_line:
                alines.append(zigzag_line)

        # 3. Подготовка горизонтальных линий для кластеров Фибоначчи
        hlines = []
        hline_colors = []
        
        # Берем только самые сильные кластеры (strength >= 2), максимум 4 штуки
        strong_clusters = [c for c in fib_clusters if c.strength >= 2][:4]
        for cl in strong_clusters:
            hlines.append(cl.price)
            # Синие пунктиры для кластеров Фибоначчи
            hline_colors.append('blue' if cl.is_support else 'purple')

        # 3.5. Подготовка линий торгового плана (вход, тейк-профит, стоп-лосс)
        def _extract_price(text: str) -> float | None:
            if not text:
                return None
            m = re.search(r'\b\d+(?:\.\d+)?\b', str(text).replace(',', ''))
            return float(m.group(0)) if m else None

        if plan:
            params = plan.trade_params or {}
            trig = plan.trigger_prices or {}
            
            # Вход (серый)
            entry_price = _extract_price(trig.get("confirmation_level")) or _extract_price(trig.get("entry_zone"))
            if entry_price:
                hlines.append(entry_price)
                hline_colors.append('gray')
                
            # Стоп-лосс (красный)
            sl_price = _extract_price(params.get("stop_loss"))
            if sl_price:
                hlines.append(sl_price)
                hline_colors.append('red')
                
            # Тейк-профит (зеленый) - берем первый TP
            tps = params.get("take_profit_levels", [])
            tp_price = _extract_price(tps[0]) if tps else None
            if tp_price:
                hlines.append(tp_price)
                hline_colors.append('green')

        # 4. Настройка стиля графика (Dark Premium)
        market_colors = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',  # Цвета во вкусе TradingView
            edge='inherit',
            wick='inherit',
            volume='#26a69a44', # Полупрозрачный объем
            ohlc='i'
        )
        
        # Глубокий темный фон
        s = mpf.make_mpf_style(
            base_mpf_style='nightclouds',
            marketcolors=market_colors,
            gridcolor='#2d3436', # Сетка
            gridstyle='--',
            facecolor='#131722', # Основной фон (как в TradingView)
            edgecolor='#2d3436',
            figcolor='#0d1117'    # Внешний фон
        )

        # Настройка заголовка с разметкой
        title = f"\n{symbol} [{timeframe}] RalphTradeBot"
        if plan and plan.wave_count_label:
            label = plan.wave_count_label
            if len(label) > 80:
                label = label[:77] + "..."
            title += f"\n{label}"

        # Дополнительные кварги для mplfinance
        kwargs = dict(
            type='candle',
            volume=True,
            style=s,
            title=title,
            ylabel='Price',
            ylabel_lower='Volume',
            figsize=(16, 9),
            returnfig=True,
            tight_layout=True,
            datetime_format='%H:%M %d.%m',
            xrotation=35
        )

        # Добавляем векторы, если есть
        if alines:
            kwargs['alines'] = dict(alines=alines, colors=['#5d81f4'], linewidths=1.8, alpha=0.9)

        # Добавляем кластеры, если есть
        if hlines:
            kwargs['hlines'] = dict(hlines=hlines, colors=hline_colors, linestyle='dotted', alpha=0.5, linewidths=1.2)

        # 5. Отрисовка
        fig, axes = mpf.plot(plot_df, **kwargs)
        ax = axes[0]  # Главная ось с ценами
        
        # 6. Отрисовка зон (fill_between) для Сделки
        # Проверяем, что это не режим WAIT
        if plan and plan.trade_params and 'WAIT' not in str(plan.trade_params.get("direction", "")).upper():
            if 'entry_price' in locals() and entry_price:
                x_indices = range(len(plot_df))
                if 'tp_price' in locals() and tp_price is not None:
                    ax.fill_between(x_indices, entry_price, tp_price, color='#26a69a', alpha=0.1, zorder=0)
                if 'sl_price' in locals() and sl_price is not None:
                    ax.fill_between(x_indices, entry_price, sl_price, color='#ef5350', alpha=0.1, zorder=0)

        # 7. Отрисовка меток из waves_breakdown (градация волн)
        if plan and hasattr(plan, 'waves_breakdown') and plan.waves_breakdown:
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            
            # Словарь для избежания наложения надписей (будем смещать по оси Y)
            plotted_texts = {}
            
            for w in plan.waves_breakdown:
                tf_str = str(w.timeframe).upper()
                fontsize = 12
                # ТФ-специфичные цвета для меток
                if '1W' in tf_str: 
                    color = '#f48fb1' # Pinkish (Primary)
                    fontsize = 14
                elif '1D' in tf_str: 
                    color = '#ffffff' # White (Intermediate)
                    fontsize = 13
                elif '4H' in tf_str: color = '#ffcc80' # Light Orange (Minor)
                elif '1H' in tf_str: color = '#fff59d' # Light Yellow (Minute)
                elif '15M' in tf_str: color = '#80deea' # Light Cyan (Minuette)
                elif '5M' in tf_str: color = '#a5d6a7' # Light Green (Subminuette)
                else: color = '#cfd8dc'
                
                # Ищем свечу по точному времени
                has_time = getattr(w, 'end_time', 0) > 0
                idx_date = None

                if has_time:
                    expected_date = pd.to_datetime(w.end_time, unit='ms')
                    diff_time = np.abs(plot_df.index - expected_date)
                    min_idx = diff_time.argmin()
                    idx_date = plot_df.index[min_idx]
                    
                    # Определяем, верхняя это точка или нижняя для направления стрелки
                    price_val = w.end_price
                    p_high = plot_df.loc[idx_date, 'High']
                    p_low = plot_df.loc[idx_date, 'Low']
                    is_top = abs(p_high - price_val) < abs(p_low - price_val)
                    anchor_price = p_high if is_top else p_low
                else:
                    continue # Для чистоты рисуем только те, где есть точное время
                    
                row_idx = plot_df.index.get_indexer([idx_date], method='nearest')[0]
                
                # Защита от наложения
                offset_multiplier = plotted_texts.get(row_idx, 0) + 1
                plotted_texts[row_idx] = offset_multiplier
                
                # Смещение метки
                offset_val = y_range * (0.02 * offset_multiplier + 0.03)
                y_pos = anchor_price + offset_val if is_top else anchor_price - offset_val
                va = 'bottom' if is_top else 'top'
                
                ax.annotate(
                    str(w.wave_name),
                    xy=(row_idx, anchor_price),
                    xytext=(row_idx, y_pos),
                    color=color, fontsize=fontsize, fontweight='bold',
                    ha='center', va=va,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.6),
                    bbox=dict(facecolor='#1e222d', alpha=0.9, edgecolor=color, lw=0.5, boxstyle='round,pad=0.2')
                )

        # 8. Сохранение и финализация
        fig.savefig(save_path, dpi=110, bbox_inches='tight', facecolor='#0d1117')
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        logger.info("График %s [%s] успешно сгенерирован: %s", symbol, timeframe, save_path)
        return os.path.abspath(save_path)

    except Exception as e:
        logger.error("Ошибка при генерации графика %s: %s", symbol, e)
        return None
