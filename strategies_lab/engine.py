"""
engine.py — Универсальный математический бэктест-движок для strategies_lab v2.0.

Функционал:
1. Multi-TF загрузка: хранит параллельно 1m котировки и ресемплированные 15m свечи.
2. Обнаружение математических импульсов Эллиотта v2.2 (Score >= 80).
3. Финансовое моделирование:
   - Начальный депозит: $1,000
   - Риск на сделку: 1.0% ($10.00)
   - Реалистичная комиссия: 0.1% на вход + 0.1% на выход (суммарно ~0.2% от объема)
4. Количественный аудит MFE и MAE (Maximum Favorable / Adverse Excursion) для победителей и проигравших.
5. Расчёт институциональных метрик: Net PnL ($ и %), Net R, Profit Factor, Max Drawdown %, Шарп, Сортино, Slippage Tax.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Any
import numpy as np
import pandas as pd

import sys
LAB_DIR = Path(__file__).parent
BASE_DIR = LAB_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from signal_scanner import calculate_rsi_wilder
from scipy.signal import argrelextrema
from elliott_detector import detect_elliott_impulse
from strategies_lab.strategies.base_strategy import BaseStrategy, TradeSignal, EntryDecision, TradeResult

logger = logging.getLogger("StrategiesLabEngine")


def _interval_to_minutes(interval: str) -> int:
    unit = interval[-1].lower()
    val = int(interval[:-1])
    if unit == 'm':
        return val
    elif unit == 'h':
        return val * 60
    elif unit == 'd':
        return val * 1440
    return 15


def resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """Ресемплирует минутные свечи в целевой таймфрейм."""
    minutes = _interval_to_minutes(target_interval)
    if minutes <= 1:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime_utc' in df.columns:
            df.index = pd.to_datetime(df['datetime_utc'], utc=True)
        elif 'timestamp_ms' in df.columns:
            df.index = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
        else:
            df.index = pd.to_datetime(df.iloc[:, 0], unit='ms', utc=True)

    rule = f"{minutes}min"
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if 'volume' in df.columns else 'first',
    }
    res = df.resample(rule).agg(agg).dropna()
    return res


def load_all_market_data(
    data_dir: Path = BASE_DIR / "data",
    symbols: Optional[List[str]] = None,
    interval: str = "15m",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Загружает рыночные данные в формате Multi-TF:
    market_data[symbol] = {
        "main": resampled_df (15m),
        "1m": raw_1m_df (с DateTimeIndex для микроструктуры)
    }
    """
    market_data = {}
    if not data_dir.exists():
        logger.error(f"Папка с данными не найдена: {data_dir}")
        return market_data

    files = list(data_dir.glob("*_1m.parquet"))
    for f in files:
        sym = f.name.replace("_1m.parquet", "").upper()
        if symbols and sym not in symbols:
            continue
        try:
            raw = pd.read_parquet(f)
            if not isinstance(raw.index, pd.DatetimeIndex):
                if 'datetime_utc' in raw.columns:
                    raw.index = pd.to_datetime(raw['datetime_utc'], utc=True)
                elif 'timestamp_ms' in raw.columns:
                    raw.index = pd.to_datetime(raw['timestamp_ms'], unit='ms', utc=True)
                else:
                    raw.index = pd.to_datetime(raw.iloc[:, 0], unit='ms', utc=True)

            resampled = resample_ohlcv(raw, interval)
            if len(resampled) >= 150:
                market_data[sym] = {
                    "main": resampled,
                    "1m": raw,
                }
        except Exception as e:
            logger.warning(f"Ошибка загрузки {f.name}: {e}")

    logger.info(f"Загружено инструментов: {len(market_data)} (Multi-TF: main={interval} + 1m)")
    return market_data


def detect_all_signals_in_history(
    market_data: Dict[str, Dict[str, pd.DataFrame]],
    interval: str = "15m",
    min_score: float = 80.0,
) -> List[TradeSignal]:
    """
    Сканирует историю на появление валидных импульсов Эллиотта (Score >= min_score).
    Каждый сигнал снабжается ссылкой на 1m-данные для Multi-TF стратегий.
    """
    signals: List[TradeSignal] = []
    seen_keys = set()

    for symbol, tf_dict in market_data.items():
        df = tf_dict["main"]
        df_1m = tf_dict["1m"]
        n = len(df)
        if n < 100:
            continue

        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)
        close = df['close'].values.astype(np.float64)
        open_p = df['open'].values.astype(np.float64)

        rsi_len = 7 if interval in ("5m", "15m") else 14
        rsi = calculate_rsi_wilder(close, period=rsi_len)

        high_pivots = argrelextrema(high, np.greater_equal, order=3)[0]
        low_pivots = argrelextrema(low, np.less_equal, order=3)[0]

        # 1. Bearish Signals (SHORT)
        if len(high_pivots) >= 2:
            for i in range(1, len(high_pivots)):
                p1 = int(high_pivots[i - 1])
                p2 = int(high_pivots[i])
                dist = p2 - p1
                if dist < 6 or dist > 45 or p2 >= n - 15:
                    continue
                if high[p2] <= high[p1]:
                    continue
                r1, r2 = float(rsi[p1]), float(rsi[p2])
                if r2 >= r1 or (r1 - r2) < 2.5 or r1 < 64.0 or r2 < 64.0:
                    continue
                if len(high[p1 + 1 : p2]) > 0 and np.max(high[p1 + 1 : p2]) > high[p2]:
                    continue

                algo_res = detect_elliott_impulse(
                    high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                    w3_bar=p1, w5_bar=p2, direction="SHORT", rsi=rsi[:p2 + 1],
                )
                if algo_res.is_valid and algo_res.score >= min_score:
                    sig_key = f"{symbol}_{interval}_SHORT_{p2}"
                    if sig_key not in seen_keys:
                        seen_keys.add(sig_key)
                        w0_bar = algo_res.wave_indices.get("W0", max(0, p1 - 20))
                        signals.append(
                            TradeSignal(
                                symbol=symbol,
                                interval=interval,
                                direction="SHORT",
                                w5_bar=p2,
                                w5_time=df.index[p2],
                                w5_price=float(high[p2]),
                                w3_bar=p1,
                                w3_price=float(high[p1]),
                                w0_bar=w0_bar,
                                w0_price=algo_res.wave_points.get("W0", float(low[w0_bar])),
                                algo_score=algo_res.score,
                                wave_points=algo_res.wave_points,
                                raw_df=df.iloc[:p2 + 1],
                                df_1m=df_1m,
                            )
                        )

        # 2. Bullish Signals (LONG)
        if len(low_pivots) >= 2:
            for i in range(1, len(low_pivots)):
                p1 = int(low_pivots[i - 1])
                p2 = int(low_pivots[i])
                dist = p2 - p1
                if dist < 6 or dist > 45 or p2 >= n - 15:
                    continue
                if low[p2] >= low[p1]:
                    continue
                r1, r2 = float(rsi[p1]), float(rsi[p2])
                if r2 <= r1 or (r2 - r1) < 2.5 or r1 > 36.0 or r2 > 36.0:
                    continue
                if len(low[p1 + 1 : p2]) > 0 and np.min(low[p1 + 1 : p2]) < low[p2]:
                    continue

                algo_res = detect_elliott_impulse(
                    high=high[:p2 + 1], low=low[:p2 + 1], close=close[:p2 + 1], open_p=open_p[:p2 + 1],
                    w3_bar=p1, w5_bar=p2, direction="LONG", rsi=rsi[:p2 + 1],
                )
                if algo_res.is_valid and algo_res.score >= min_score:
                    sig_key = f"{symbol}_{interval}_LONG_{p2}"
                    if sig_key not in seen_keys:
                        seen_keys.add(sig_key)
                        w0_bar = algo_res.wave_indices.get("W0", max(0, p1 - 20))
                        signals.append(
                            TradeSignal(
                                symbol=symbol,
                                interval=interval,
                                direction="LONG",
                                w5_bar=p2,
                                w5_time=df.index[p2],
                                w5_price=float(low[p2]),
                                w3_bar=p1,
                                w3_price=float(low[p1]),
                                w0_bar=w0_bar,
                                w0_price=algo_res.wave_points.get("W0", float(high[w0_bar])),
                                algo_score=algo_res.score,
                                wave_points=algo_res.wave_points,
                                raw_df=df.iloc[:p2 + 1],
                                df_1m=df_1m,
                            )
                        )

    signals.sort(key=lambda s: s.w5_time)
    logger.info(f"Обнаружено исторических импульсов Эллиотта (Score >= {min_score}): {len(signals)}")
    return signals


@dataclass
class BacktestSummary:
    """Сводные финансовые и квантовые метрики стратегии."""
    strategy_name: str
    total_signals: int
    executed_trades: int
    filtered_signals: int
    wins: int
    losses: int
    breakevens: int
    win_rate_pct: float
    full_win_rate_pct: float
    profit_factor: float
    total_net_pnl_pct: float
    avg_trade_pnl_pct: float
    expectancy_r: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    # Финансовые показатели (Депозит $1,000, Риск 1% = $10, Комиссия 0.1% + 0.1%)
    initial_deposit_usd: float = 1000.0
    final_deposit_usd: float = 1000.0
    net_profit_usd: float = 0.0
    total_fees_usd: float = 0.0
    avg_position_size_usd: float = 0.0
    # MFE / MAE метрики
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    avg_mfe_r: float = 0.0
    avg_mae_r: float = 0.0
    avg_bars_held: float = 0.0
    avg_slippage_tax_pct: float = 0.0
    losses_saved_vs_baseline: int = 0
    mfe_mae_analysis: Dict[str, Any] = field(default_factory=dict)
    trade_results: List[TradeResult] = field(default_factory=list)


def compute_mfe_mae_analysis(trades: List[TradeResult]) -> Dict[str, Any]:
    """
    Проводит глубокий институциональный квантовый анализ распределения MFE и MAE.
    Отвечает на ключевые вопросы:
    1. Насколько можно сузить базовый Stop-Loss?
    2. Где математически оптимально фиксировать тейк-профиты?
    """
    if not trades:
        return {}

    winners = [t for t in trades if t.net_pnl_usd > 0]
    losers = [t for t in trades if t.net_pnl_usd < 0]

    # Анализ MAE для победителей (сколько просадки испытывает успешный трейд)
    win_mae_pct = [t.mae_pct for t in winners]
    win_mae_r = [t.mae_r for t in winners]

    # Анализ MFE для всех сделок (куда реально доходит цена)
    all_mfe_pct = [t.mfe_pct for t in trades]
    all_mfe_r = [t.mfe_r for t in trades]

    # Сколько сделок дошло до уровней 1R, 2R, 3R, 4R
    reached_1r = len([t for t in trades if t.mfe_r >= 1.0])
    reached_2r = len([t for t in trades if t.mfe_r >= 2.0])
    reached_3r = len([t for t in trades if t.mfe_r >= 3.0])
    reached_4r = len([t for t in trades if t.mfe_r >= 4.0])

    stats = {
        "total_analyzed": len(trades),
        "winners_count": len(winners),
        "losers_count": len(losers),
        "win_mae_percentiles_r": {
            "p50_median": round(float(np.percentile(win_mae_r, 50)), 2) if win_mae_r else 0.0,
            "p75": round(float(np.percentile(win_mae_r, 75)), 2) if win_mae_r else 0.0,
            "p90": round(float(np.percentile(win_mae_r, 90)), 2) if win_mae_r else 0.0,
            "p95": round(float(np.percentile(win_mae_r, 95)), 2) if win_mae_r else 0.0,
        },
        "all_mfe_percentiles_r": {
            "p25": round(float(np.percentile(all_mfe_r, 25)), 2) if all_mfe_r else 0.0,
            "p50_median": round(float(np.percentile(all_mfe_r, 50)), 2) if all_mfe_r else 0.0,
            "p75": round(float(np.percentile(all_mfe_r, 75)), 2) if all_mfe_r else 0.0,
            "p90": round(float(np.percentile(all_mfe_r, 90)), 2) if all_mfe_r else 0.0,
        },
        "mfe_reach_rates_pct": {
            "reached_1r_pct": round(reached_1r / len(trades) * 100.0, 1),
            "reached_2r_pct": round(reached_2r / len(trades) * 100.0, 1),
            "reached_3r_pct": round(reached_3r / len(trades) * 100.0, 1),
            "reached_4r_pct": round(reached_4r / len(trades) * 100.0, 1),
        }
    }
    return stats


def run_strategy_backtest(
    strategy: BaseStrategy,
    signals: List[TradeSignal],
    market_data: Dict[str, Dict[str, pd.DataFrame]],
    baseline_losses: Optional[Dict[str, bool]] = None,
    future_window_bars: int = 80,
    initial_deposit: float = 1000.0,
) -> BacktestSummary:
    """
    Запускает бэктест конкретной стратегии с точным финансовым учётом.
    """
    executed: List[TradeResult] = []
    filtered_count = 0
    losses_saved = 0

    for idx, sig in enumerate(signals):
        sig_id = f"{sig.symbol}_{sig.interval}_{sig.direction}_{sig.w5_bar}"
        tf_dict = market_data.get(sig.symbol)
        if tf_dict is None:
            continue
        full_df = tf_dict["main"]
        if len(full_df) <= sig.w5_bar + 1:
            continue

        future_slice = full_df.iloc[sig.w5_bar + 1 : sig.w5_bar + 1 + future_window_bars]
        if len(future_slice) < 5:
            continue

        decision = strategy.evaluate_entry(sig, future_slice)

        if decision.should_enter:
            res = strategy.simulate_execution(decision, sig.direction, future_slice)
            res.signal_id = sig_id
            res.symbol = sig.symbol
            res.interval = sig.interval
            executed.append(res)
        else:
            filtered_count += 1
            if baseline_losses and baseline_losses.get(sig_id, False):
                losses_saved += 1

    total = len(signals)
    exec_cnt = len(executed)

    if exec_cnt == 0:
        return BacktestSummary(
            strategy_name=strategy.name,
            total_signals=total,
            executed_trades=0,
            filtered_signals=filtered_count,
            wins=0, losses=0, breakevens=0,
            win_rate_pct=0.0, full_win_rate_pct=0.0,
            profit_factor=0.0, total_net_pnl_pct=0.0,
            avg_trade_pnl_pct=0.0, expectancy_r=0.0,
            max_drawdown_pct=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, losses_saved_vs_baseline=losses_saved,
            trade_results=[],
        )

    # Финансовые агрегаты
    net_pnls_usd = [t.net_pnl_usd for t in executed]
    fees_usd = [t.fee_usd for t in executed]
    pos_sizes = [t.position_size_usd for t in executed]
    raw_pnls = [t.final_pnl_pct for t in executed]
    r_mults = [t.pnl_r for t in executed]
    slippage_taxes = [t.slippage_tax_pct for t in executed if t.slippage_tax_pct > 0]

    wins_list = [p for p in net_pnls_usd if p > 0.05]
    losses_list = [p for p in net_pnls_usd if p < -0.05]
    be_list = [p for p in net_pnls_usd if -0.05 <= p <= 0.05]

    wins_cnt = len(wins_list)
    loss_cnt = len(losses_list)
    be_cnt = len(be_list)

    win_rate = round((wins_cnt + be_cnt) / exec_cnt * 100.0, 1)
    full_win_rate = round(wins_cnt / exec_cnt * 100.0, 1)

    gross_profit = sum(wins_list)
    gross_loss = abs(sum(losses_list))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    total_net_profit_usd = round(sum(net_pnls_usd), 2)
    final_deposit = round(initial_deposit + total_net_profit_usd, 2)
    total_net_pnl_pct = round((total_net_profit_usd / initial_deposit) * 100.0, 2)

    avg_pnl_pct = round(float(np.mean(raw_pnls)), 2)
    expectancy_r = round(float(np.mean(r_mults)), 2)
    total_fees = round(sum(fees_usd), 2)
    avg_pos_size = round(float(np.mean(pos_sizes)), 2) if pos_sizes else 0.0
    avg_slip_tax = round(float(np.mean(slippage_taxes)), 2) if slippage_taxes else 0.0

    # Max Drawdown по долларовой кривой капитала
    cum_equity = initial_deposit + np.cumsum(net_pnls_usd)
    running_max = np.maximum.accumulate(cum_equity)
    drawdowns_pct = (running_max - cum_equity) / running_max * 100.0
    max_dd = round(float(np.max(drawdowns_pct)) if len(drawdowns_pct) > 0 else 0.0, 2)

    # Sharpe / Sortino
    std_r = float(np.std(r_mults)) if len(r_mults) > 1 else 1.0
    sharpe = round((expectancy_r / std_r) * np.sqrt(252), 2) if std_r > 0 else 0.0

    neg_r = [r for r in r_mults if r < 0]
    downside_r = float(np.std(neg_r)) if len(neg_r) > 1 else 1.0
    sortino = round((expectancy_r / downside_r) * np.sqrt(252), 2) if downside_r > 0 else 0.0

    avg_mfe = round(float(np.mean([t.mfe_pct for t in executed])), 2)
    avg_mae = round(float(np.mean([t.mae_pct for t in executed])), 2)
    avg_mfe_r = round(float(np.mean([t.mfe_r for t in executed])), 2)
    avg_mae_r = round(float(np.mean([t.mae_r for t in executed])), 2)
    avg_bars = round(float(np.mean([t.bars_held for t in executed])), 1)

    mfe_mae_stats = compute_mfe_mae_analysis(executed)

    return BacktestSummary(
        strategy_name=strategy.name,
        total_signals=total,
        executed_trades=exec_cnt,
        filtered_signals=filtered_count,
        wins=wins_cnt,
        losses=loss_cnt,
        breakevens=be_cnt,
        win_rate_pct=win_rate,
        full_win_rate_pct=full_win_rate,
        profit_factor=profit_factor,
        total_net_pnl_pct=total_net_pnl_pct,
        avg_trade_pnl_pct=avg_pnl_pct,
        expectancy_r=expectancy_r,
        max_drawdown_pct=max_dd,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        initial_deposit_usd=initial_deposit,
        final_deposit_usd=final_deposit,
        net_profit_usd=total_net_profit_usd,
        total_fees_usd=total_fees,
        avg_position_size_usd=avg_pos_size,
        avg_mfe_pct=avg_mfe,
        avg_mae_pct=avg_mae,
        avg_mfe_r=avg_mfe_r,
        avg_mae_r=avg_mae_r,
        avg_bars_held=avg_bars,
        avg_slippage_tax_pct=avg_slip_tax,
        losses_saved_vs_baseline=losses_saved,
        mfe_mae_analysis=mfe_mae_stats,
        trade_results=executed,
    )
