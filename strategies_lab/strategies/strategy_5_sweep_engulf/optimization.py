"""
optimization.py — Исследовательский модуль оптимизации выходов для Стратегии 5 (Sweep & Engulf + 50% Limit).

Цели:
1. Масштабное тестирование на массиве исторических эпох (Bull Run 2020-2021, Crypto Winter 2022,
   ETF Rally 2023-2024, Mega Bull Run 2025, Macro Bear Market 2026).
2. Сбор единого пула исполненных сделок (вход строго 50% лимит на 1-2 свече, SL строго за тенью).
3. Анализ MFE (Maximum Favorable Excursion) в R — определение потенциала движения цены до выбивания стопа.
4. Сравнительный A/B тест архитектур выходов на ОДИНАКОВОМ пуле сделок:
   - Модель A: Фиксированные тейки (1:1.5, 1:2.0, 1:2.5, 1:3.0, 1:4.0, 1:5.0 Net R)
   - Модель B: Частичный фикс (50% на TP1) + перенос стопа в БУ (Breakeven) + удержание остатка (Runner)
     до:
     * Обратного поглощения (Reverse Candle Engulfing)
     * Экстремума / дивергенции RSI (Wilder RSI <= 30 для Short, >= 70 для Long)
     * Любого из первых наступивших обратных сигналов
5. Расчёт институциональных KPI: Win Rate, Profit Factor, Net PnL ($), Net EV (R), Max Drawdown, комиссии.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from strategies_lab.engine import load_all_market_data, detect_all_signals_in_history
from strategies_lab.strategies.strategy_5_sweep_engulf.patterns import (
    detect_sweep_engulf_candle,
    check_limit_fill,
    calculate_net_1_to_3_tp,
    SweepEngulfPattern,
    FillCheckResult,
)
from strategies_lab.strategies.strategy_5_sweep_engulf.config import (
    RESULTS_DIR,
    FEE_RATE,
    ACCOUNT_DEPOSIT_USD,
    RISK_PER_TRADE_USD,
    MIN_ELLIOTT_SCORE,
)

logger = logging.getLogger("OptimizationLab")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)

DATASETS_BASE = Path("c:/Users/user/Desktop/данные для бектестов")

HISTORICAL_REGIMES = [
    ("1_Bull_Run_2020_2021", "🚀 Бычий рынок (2020-2021)"),
    ("2_Crypto_Winter_2022", "🥶 Крипто-зима / Крах FTX (2022)"),
    ("4_ETF_Rally_2023_2024", "🏢 Институциональный рост ETF (2023-2024)"),
    ("8_Mega_Bull_Run_2025", "🏔️ Мега-буллран (2025)"),
    ("9_Macro_Bear_Market_2026", "🩸 Медвежий макро-тренд (2025-2026)"),
]

TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT",
    "SUIUSDT", "AAVEUSDT", "1000PEPEUSDT",
]


@dataclass
class FilledTradeContext:
    """Контекст исполненной сделки со всеми будущими свечами для симуляции разных выходов."""
    trade_id: str
    regime_id: str
    symbol: str
    interval: str
    direction: str
    entry_bar_abs: int
    entry_time: Any
    entry_price: float
    initial_sl: float
    risk_dist: float
    risk_pct: float
    pos_size_usd: float
    future_candles: pd.DataFrame
    rsi_future: np.ndarray
    pattern: SweepEngulfPattern


@dataclass
class SimulatedOutcome:
    """Результат выхода из сделки по определенному сценарию."""
    exit_model: str
    exit_time: Any
    exit_price: float
    exit_reason: str
    bars_held: int
    net_pnl_usd: float
    net_pnl_r: float
    gross_pnl_usd: float
    fee_usd: float
    mfe_r: float
    mae_r: float
    extra_info: str = ""


def collect_filled_trades_across_regimes(
    regimes: List[Tuple[str, str]],
    symbols: List[str],
    interval: str = "15m",
    fill_window_bars: int = 2,
    future_window_bars: int = 150,
) -> List[FilledTradeContext]:
    """
    Загружает котировки всех эпох, ищет свечи S, исполняет 50% лимит и собирает пул сделок.
    """
    all_filled_trades: List[FilledTradeContext] = []
    trade_seq = 0

    for reg_dir_name, reg_label in regimes:
        reg_path = DATASETS_BASE / reg_dir_name
        if not reg_path.exists():
            logger.warning(f"Папка {reg_dir_name} не найдена, пропускаем...")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"⏳ Загрузка эпохи: {reg_label} [{reg_dir_name}]...")
        mdata = load_all_market_data(data_dir=reg_path, symbols=symbols, interval=interval)
        if not mdata:
            logger.warning(f"Данные не загружены для {reg_dir_name}")
            continue

        signals = detect_all_signals_in_history(mdata, interval=interval, min_score=MIN_ELLIOTT_SCORE)
        logger.info(f"  Найдено 5-волновых импульсов W0-W5: {len(signals)}")

        s_count = 0
        filled_count = 0

        for sig in signals:
            sym_dict = mdata.get(sig.symbol)
            if not sym_dict:
                continue
            full_df = sym_dict["main"]
            w5_bar = sig.w5_bar

            future_slice = full_df.iloc[w5_bar + 1 : w5_bar + 1 + 15]
            if len(future_slice) == 0:
                continue

            prev_c = sig.raw_df.iloc[-2] if len(sig.raw_df) >= 2 else None
            pat = detect_sweep_engulf_candle(
                w5_candle=sig.raw_df.iloc[-1],
                future_candles=future_slice,
                direction=sig.direction,
                prev_candle=prev_c,
            )

            if not pat.found:
                continue
            s_count += 1

            # Проверяем исполнение лимита 50%
            fill_res = check_limit_fill(
                pattern=pat,
                future_candles=future_slice,
                direction=sig.direction,
                fill_window_bars=fill_window_bars,
            )

            if not fill_res.is_filled:
                continue

            # Сделка исполнена!
            trade_seq += 1
            filled_count += 1

            entry_p = pat.limit_entry_price
            sl_p = pat.sl_price
            risk_dist = abs(entry_p - sl_p)
            risk_pct = (risk_dist / entry_p) * 100.0 if entry_p > 0 else 1.0
            pos_size_usd = RISK_PER_TRADE_USD / (risk_pct / 100.0)

            # Извлекаем окно выполнения сделки
            abs_fill_bar = w5_bar + 1 + fill_res.fill_rel_idx
            exec_candles = full_df.iloc[abs_fill_bar : abs_fill_bar + future_window_bars].copy()

            # Считаем RSI для окна выполнения
            rsi_period = 7 if interval in ("5m", "15m") else 14
            rsi_full = calculate_rsi_wilder(full_df['close'].values.astype(np.float64), period=rsi_period)
            rsi_exec = rsi_full[abs_fill_bar : abs_fill_bar + future_window_bars]

            t_ctx = FilledTradeContext(
                trade_id=f"T{trade_seq:04d}_{sig.symbol}_{interval}_{sig.direction}_{abs_fill_bar}",
                regime_id=reg_dir_name,
                symbol=sig.symbol,
                interval=interval,
                direction=sig.direction,
                entry_bar_abs=abs_fill_bar,
                entry_time=fill_res.fill_time,
                entry_price=entry_p,
                initial_sl=sl_p,
                risk_dist=risk_dist,
                risk_pct=risk_pct,
                pos_size_usd=pos_size_usd,
                future_candles=exec_candles,
                rsi_future=rsi_exec,
                pattern=pat,
            )
            all_filled_trades.append(t_ctx)

        logger.info(f"  Сформировали свечу S: {s_count} | Исполнен лимит 50%: {filled_count} ({filled_count/s_count*100:.1f}%)" if s_count > 0 else "  Свечей S: 0")

    logger.info(f"\n{'='*70}")
    logger.info(f"🏆 ВСЕГО СОБРАНО ИСПОЛНЕННЫХ СДЕЛОК: {len(all_filled_trades)} по всем эпохам!")
    return all_filled_trades


def simulate_fixed_tp(ctx: FilledTradeContext, target_r: float, fee_rate: float = FEE_RATE) -> SimulatedOutcome:
    """
    Симулирует выход по строго фиксированному Net R тейку с учётом комиссий.
    """
    direction = ctx.direction
    entry = ctx.entry_price
    sl = ctx.initial_sl
    pos_usd = ctx.pos_size_usd

    tp_price, _, _ = calculate_net_1_to_3_tp(entry, sl, direction, fee_rate=fee_rate, target_net_r=target_r)

    candles = ctx.future_candles
    n = len(candles)
    if n == 0:
        return SimulatedOutcome(
            exit_model=f"Fixed_TP_{target_r}R",
            exit_time=ctx.entry_time,
            exit_price=entry,
            exit_reason="no_data",
            bars_held=0,
            net_pnl_usd=0.0,
            net_pnl_r=0.0,
            gross_pnl_usd=0.0,
            fee_usd=0.0,
            mfe_r=0.0,
            mae_r=0.0,
        )

    highs = candles['high'].values
    lows = candles['low'].values
    closes = candles['close'].values
    times = candles.index

    mfe_val = 0.0
    mae_val = 0.0

    exit_bar = n - 1
    exit_p = closes[-1]
    exit_reason = "timeout"

    for b in range(n):
        h, l = highs[b], lows[b]
        fav = (entry - l) if direction == "SHORT" else (h - entry)
        adv = (h - entry) if direction == "SHORT" else (entry - l)
        mfe_val = max(mfe_val, fav)
        mae_val = max(mae_val, adv)

        if direction == "SHORT":
            if h >= sl:
                exit_bar = b
                exit_p = sl
                exit_reason = "sl"
                break
            elif l <= tp_price:
                exit_bar = b
                exit_p = tp_price
                exit_reason = "tp"
                break
        else:
            if l <= sl:
                exit_bar = b
                exit_p = sl
                exit_reason = "sl"
                break
            elif h >= tp_price:
                exit_bar = b
                exit_p = tp_price
                exit_reason = "tp"
                break

    # PnL
    pct = (entry - exit_p) / entry * 100.0 if direction == "SHORT" else (exit_p - entry) / entry * 100.0
    gross_usd = pos_usd * (pct / 100.0)
    in_fee = pos_usd * fee_rate
    out_fee = max(0.0, pos_usd + gross_usd) * fee_rate
    tot_fee = in_fee + out_fee
    net_usd = gross_usd - tot_fee
    net_r = net_usd / RISK_PER_TRADE_USD

    mfe_r = mfe_val / ctx.risk_dist if ctx.risk_dist > 0 else 0.0
    mae_r = mae_val / ctx.risk_dist if ctx.risk_dist > 0 else 0.0

    return SimulatedOutcome(
        exit_model=f"Fixed_TP_{target_r}R",
        exit_time=times[exit_bar],
        exit_price=exit_p,
        exit_reason=exit_reason,
        bars_held=exit_bar + 1,
        net_pnl_usd=round(net_usd, 2),
        net_pnl_r=round(net_r, 2),
        gross_pnl_usd=round(gross_usd, 2),
        fee_usd=round(tot_fee, 2),
        mfe_r=round(mfe_r, 2),
        mae_r=round(mae_r, 2),
    )


def simulate_partial_tp_breakeven_runner(
    ctx: FilledTradeContext,
    tp1_r: float = 2.0,
    tp1_share: float = 0.50,
    runner_exit_type: str = "reverse_engulf",  # "reverse_engulf", "rsi_extreme", "combined"
    fee_rate: float = FEE_RATE,
) -> SimulatedOutcome:
    """
    Симулирует:
    1. При достижении TP1 (например, +2.0R):
       - Фиксируется tp1_share (например 50%) позиции.
       - Стоп-лосс на остаток (1 - tp1_share) мгновенно переносится в БУ (Breakeven + покрытие комиссий).
    2. Остаток позиции тянется до:
       - Если runner_exit_type == "reverse_engulf":
         Свеча поглощения в ОБРАТНУЮ сторону (для SHORT — бычье тело поглощает предыдущее; для LONG — медвежье).
       - Если runner_exit_type == "rsi_extreme":
         RSI уходит в противоположную зону экстремума (<= 30 для SHORT, >= 70 для LONG).
       - Если runner_exit_type == "combined":
         Любое из двух условий выше.
       - Если срабатывает перенесённый стоп БУ (цена развернулась к точке входа).
    """
    direction = ctx.direction
    entry = ctx.entry_price
    sl = ctx.initial_sl
    pos_usd = ctx.pos_size_usd

    tp1_price, _, _ = calculate_net_1_to_3_tp(entry, sl, direction, fee_rate=fee_rate, target_net_r=tp1_r)

    # Цена безубытка с запасом на комиссию
    if direction == "SHORT":
        be_price = entry * (1.0 - 2.0 * fee_rate)
    else:
        be_price = entry * (1.0 + 2.0 * fee_rate)

    candles = ctx.future_candles
    rsi_arr = ctx.rsi_future
    n = len(candles)
    if n == 0:
        return SimulatedOutcome(
            exit_model=f"Partial_{tp1_r}R_{runner_exit_type}",
            exit_time=ctx.entry_time, exit_price=entry, exit_reason="no_data",
            bars_held=0, net_pnl_usd=0.0, net_pnl_r=0.0, gross_pnl_usd=0.0,
            fee_usd=0.0, mfe_r=0.0, mae_r=0.0,
        )

    highs = candles['high'].values
    lows = candles['low'].values
    opens = candles['open'].values
    closes = candles['close'].values
    times = candles.index

    mfe_val = 0.0
    mae_val = 0.0

    tp1_hit = False
    tp1_bar = -1

    current_sl = sl
    runner_exit_bar = -1
    runner_exit_price = 0.0
    runner_exit_reason = ""

    # Проход по свечам
    for b in range(n):
        h, l, o, c = highs[b], lows[b], opens[b], closes[b]
        fav = (entry - l) if direction == "SHORT" else (h - entry)
        adv = (h - entry) if direction == "SHORT" else (entry - l)
        mfe_val = max(mfe_val, fav)
        mae_val = max(mae_val, adv)

        # Фаза 1: До взятия TP1
        if not tp1_hit:
            # Проверяем начальный SL
            if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                # Полный стоп-лосс на всей позиции
                runner_exit_bar = b
                runner_exit_price = current_sl
                runner_exit_reason = "sl"
                break

            # Проверяем достижение TP1
            if (direction == "SHORT" and l <= tp1_price) or (direction == "LONG" and h >= tp1_price):
                tp1_hit = True
                tp1_bar = b
                current_sl = be_price  # Переносим стоп в БУ!
                continue

        # Фаза 2: После взятия TP1 (тянем Runner со стопом в БУ)
        else:
            # 1. Проверяем выбивание БУ
            if (direction == "SHORT" and h >= current_sl) or (direction == "LONG" and l <= current_sl):
                runner_exit_bar = b
                runner_exit_price = current_sl
                runner_exit_reason = "be"
                break

            # 2. Проверяем сигнал на выход Runner
            prev_b = b - 1
            if prev_b >= 0:
                p_open, p_close = opens[prev_b], closes[prev_b]
                p_body_top = max(p_open, p_close)
                p_body_bottom = min(p_open, p_close)

                is_reverse_engulf = False
                if direction == "SHORT":
                    # Бычье поглощение: свеча закрылась выше тела предыдущей
                    if c > o and c > p_body_top:
                        is_reverse_engulf = True
                else:
                    # Медвежье поглощение: свеча закрылась ниже тела предыдущей
                    if c < o and c < p_body_bottom:
                        is_reverse_engulf = True

                # RSI условие
                is_rsi_exit = False
                if b < len(rsi_arr):
                    curr_rsi = rsi_arr[b]
                    if direction == "SHORT" and curr_rsi <= 30.0:
                        is_rsi_exit = True
                    elif direction == "LONG" and curr_rsi >= 70.0:
                        is_rsi_exit = True

                # Принятие решения по выходу Runner
                should_exit_runner = False
                if runner_exit_type == "reverse_engulf" and is_reverse_engulf:
                    should_exit_runner = True
                    runner_exit_reason = "runner_reverse_engulf"
                elif runner_exit_type == "rsi_extreme" and is_rsi_exit:
                    should_exit_runner = True
                    runner_exit_reason = "runner_rsi_extreme"
                elif runner_exit_type == "combined" and (is_reverse_engulf or is_rsi_exit):
                    should_exit_runner = True
                    runner_exit_reason = "runner_engulf" if is_reverse_engulf else "runner_rsi"

                if should_exit_runner:
                    runner_exit_bar = b
                    runner_exit_price = c
                    break

    # Если до конца окна позиция не закрылась
    if runner_exit_bar == -1:
        runner_exit_bar = n - 1
        runner_exit_price = closes[-1]
        runner_exit_reason = "timeout"

    # Расчёт PnL частей позиции
    in_fee = pos_usd * fee_rate

    if not tp1_hit:
        # Позиция полностью закрылась по SL (или таймауту)
        pct = (entry - runner_exit_price) / entry * 100.0 if direction == "SHORT" else (runner_exit_price - entry) / entry * 100.0
        gross_usd = pos_usd * (pct / 100.0)
        out_fee = max(0.0, pos_usd + gross_usd) * fee_rate
        tot_fee = in_fee + out_fee
        net_usd = gross_usd - tot_fee
        net_r = net_usd / RISK_PER_TRADE_USD
        final_exit_reason = runner_exit_reason
    else:
        # Часть 1 (TP1)
        part1_usd = pos_usd * tp1_share
        pct1 = (entry - tp1_price) / entry * 100.0 if direction == "SHORT" else (tp1_price - entry) / entry * 100.0
        gross1 = part1_usd * (pct1 / 100.0)
        fee1 = (part1_usd + max(0.0, part1_usd + gross1)) * fee_rate

        # Часть 2 (Runner)
        runner_share = 1.0 - tp1_share
        part2_usd = pos_usd * runner_share
        pct2 = (entry - runner_exit_price) / entry * 100.0 if direction == "SHORT" else (runner_exit_price - entry) / entry * 100.0
        gross2 = part2_usd * (pct2 / 100.0)
        fee2 = (part2_usd + max(0.0, part2_usd + gross2)) * fee_rate

        gross_usd = gross1 + gross2
        tot_fee = fee1 + fee2
        net_usd = gross_usd - tot_fee
        net_r = net_usd / RISK_PER_TRADE_USD
        final_exit_reason = f"tp1+{runner_exit_reason}"

    mfe_r = mfe_val / ctx.risk_dist if ctx.risk_dist > 0 else 0.0
    mae_r = mae_val / ctx.risk_dist if ctx.risk_dist > 0 else 0.0

    return SimulatedOutcome(
        exit_model=f"Partial_{tp1_r}R_{runner_exit_type}",
        exit_time=times[runner_exit_bar],
        exit_price=runner_exit_price,
        exit_reason=final_exit_reason,
        bars_held=runner_exit_bar + 1,
        net_pnl_usd=round(net_usd, 2),
        net_pnl_r=round(net_r, 2),
        gross_pnl_usd=round(gross_usd, 2),
        fee_usd=round(tot_fee, 2),
        mfe_r=round(mfe_r, 2),
        mae_r=round(mae_r, 2),
    )


def run_full_optimization_matrix(
    filled_trades: List[FilledTradeContext],
) -> Dict[str, Any]:
    """
    Прогоняет все модели выходов на едином пуле сделок и строит сравнительную матрицу.
    """
    models_to_test = [
        # Фиксированные тейки
        ("Fixed_1.5R", lambda t: simulate_fixed_tp(t, target_r=1.5)),
        ("Fixed_2.0R", lambda t: simulate_fixed_tp(t, target_r=2.0)),
        ("Fixed_2.5R", lambda t: simulate_fixed_tp(t, target_r=2.5)),
        ("Fixed_3.0R (Базовый)", lambda t: simulate_fixed_tp(t, target_r=3.0)),
        ("Fixed_4.0R", lambda t: simulate_fixed_tp(t, target_r=4.0)),
        ("Fixed_5.0R", lambda t: simulate_fixed_tp(t, target_r=5.0)),

        # Модели с частичным фиксом и БУ
        ("50%@2R + БУ + Обратное Поглощение", lambda t: simulate_partial_tp_breakeven_runner(t, tp1_r=2.0, tp1_share=0.5, runner_exit_type="reverse_engulf")),
        ("50%@2R + БУ + RSI Экстремум (30/70)", lambda t: simulate_partial_tp_breakeven_runner(t, tp1_r=2.0, tp1_share=0.5, runner_exit_type="rsi_extreme")),
        ("50%@2R + БУ + Любой обратный сигнал", lambda t: simulate_partial_tp_breakeven_runner(t, tp1_r=2.0, tp1_share=0.5, runner_exit_type="combined")),
        ("50%@3R + БУ + Обратное Поглощение", lambda t: simulate_partial_tp_breakeven_runner(t, tp1_r=3.0, tp1_share=0.5, runner_exit_type="reverse_engulf")),
        ("50%@3R + БУ + Любой обратный сигнал", lambda t: simulate_partial_tp_breakeven_runner(t, tp1_r=3.0, tp1_share=0.5, runner_exit_type="combined")),
    ]

    model_results = {}
    detailed_rows = []

    for model_name, sim_func in models_to_test:
        outcomes: List[SimulatedOutcome] = []
        for t in filled_trades:
            out = sim_func(t)
            outcomes.append(out)
            detailed_rows.append({
                "trade_id": t.trade_id,
                "regime": t.regime_id,
                "symbol": t.symbol,
                "direction": t.direction,
                "model": model_name,
                "net_pnl_usd": out.net_pnl_usd,
                "net_pnl_r": out.net_pnl_r,
                "exit_reason": out.exit_reason,
                "bars_held": out.bars_held,
                "mfe_r": out.mfe_r,
                "mae_r": out.mae_r,
            })

        # Метрики
        pnls = [o.net_pnl_usd for o in outcomes]
        r_mults = [o.net_pnl_r for o in outcomes]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        fees = sum(o.fee_usd for o in outcomes)

        total_trades = len(outcomes)
        win_count = len(wins)
        loss_count = len(losses)
        be_count = total_trades - win_count - loss_count

        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
        net_pnl = sum(pnls)
        ev_r = np.mean(r_mults) if r_mults else 0.0

        # Макс просадка
        equity_curve = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity_curve)
        dd = peak - equity_curve
        max_dd_usd = np.max(dd) if len(dd) > 0 else 0.0
        max_dd_pct = (max_dd_usd / ACCOUNT_DEPOSIT_USD) * 100.0

        model_results[model_name] = {
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "breakevens": be_count,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "net_pnl_usd": round(net_pnl, 2),
            "net_ev_r": round(ev_r, 2),
            "max_dd_pct": round(max_dd_pct, 1),
            "total_fees_usd": round(fees, 2),
        }

    return {
        "model_results": model_results,
        "detailed_rows": detailed_rows,
    }


def analyze_mfe_distribution(filled_trades: List[FilledTradeContext]) -> Dict[str, Any]:
    """
    Анализирует распределение Maximum Favorable Excursion (MFE) в R для всех сделок.
    Показывает, какой % сделок реально доходит до 1R, 2R, 3R, 4R, 5R, 7R, 10R.
    """
    mfes = []
    for t in filled_trades:
        # Симулируем удержание до бесконечности (пока не выбьет SL), чтобы замерить пиковый MFE
        direction = t.direction
        entry = t.entry_price
        sl = t.initial_sl
        dist = t.risk_dist
        highs = t.future_candles['high'].values
        lows = t.future_candles['low'].values

        mfe_val = 0.0
        for b in range(len(highs)):
            h, l = highs[b], lows[b]
            # Если выбило стоп
            if (direction == "SHORT" and h >= sl) or (direction == "LONG" and l <= sl):
                break
            fav = (entry - l) if direction == "SHORT" else (h - entry)
            mfe_val = max(mfe_val, fav)

        mfe_r = (mfe_val / dist) if dist > 0 else 0.0
        mfes.append(mfe_r)

    mfes = np.array(mfes)
    total = len(mfes)
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0]
    reach_stats = {}
    for th in thresholds:
        count = int(np.sum(mfes >= th))
        pct = (count / total * 100.0) if total > 0 else 0.0
        reach_stats[f">={th:.1f}R"] = {
            "count": count,
            "pct": round(pct, 1),
        }

    return {
        "total_trades": total,
        "median_mfe_r": round(float(np.median(mfes)), 2) if total > 0 else 0.0,
        "mean_mfe_r": round(float(np.mean(mfes)), 2) if total > 0 else 0.0,
        "p75_mfe_r": round(float(np.percentile(mfes, 75)), 2) if total > 0 else 0.0,
        "p90_mfe_r": round(float(np.percentile(mfes, 90)), 2) if total > 0 else 0.0,
        "reach_stats": reach_stats,
    }
