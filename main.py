"""
main.py — Единая точка входа RalphTradeBot V3.

Режимы запуска:
  python main.py --mode live     : Real-time анализ (WS + DAG + Signals)
  python main.py --mode backtest : Прогон по историческим данным
  python main.py --mode train    : Обучение ML-модели на HITL-разметке
  python main.py --mode label    : Запуск Streamlit HITL Dashboard
  python main.py --mode fetch    : Загрузка исторических данных в DuckDB
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import numpy as np

from src.core.config import SYMBOLS, TIMEFRAMES, CANDLE_LIMIT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ralph")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: FETCH — Загрузка исторических данных в DuckDB
# ═════════════════════════════════════════════════════════════════════════════

def run_fetch(symbols: list[str], timeframes: list[str]):
    """Загружает OHLCV с биржи и сохраняет в DuckDB."""
    from src.fetcher.data_fetcher import fetch_ohlcv
    from src.storage.duckdb_store import get_store

    store = get_store()
    for symbol in symbols:
        for tf in timeframes:
            try:
                candles = fetch_ohlcv(symbol, tf, limit=CANDLE_LIMIT)
                n = store.upsert_ohlcv(symbol, tf, candles)
                logger.info("Saved %d candles: %s [%s]", n, symbol, tf)
            except Exception as e:
                logger.error("Failed to fetch %s [%s]: %s", symbol, tf, e)
    store.close()


# ═════════════════════════════════════════════════════════════════════════════
# MODE: BACKTEST — Offline анализ по историческим данным
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(symbol: str, timeframe: str):
    """Прогоняет Wave Engine + ML + Execution по историческим данным."""
    from src.storage.duckdb_store import get_store
    from src.wave_engine.extremum_finder import ExtremumFinder
    from src.wave_engine.hypothesis_dag import HypothesisDAG
    from src.confluence.feature_extractor import FeatureExtractor
    from src.confluence.ml_scorer import MLScorer
    from src.confluence.signal_filter import SignalFilter
    from src.execution.report_generator import ReportGenerator
    from src.ingestion.liquidity_mapper import LiquidityMapper

    store = get_store()
    candles = store.get_ohlcv(symbol, timeframe)
    if not candles:
        logger.error("No data for %s [%s]. Run --mode fetch first.", symbol, timeframe)
        return

    high = np.array([c["high"] for c in candles])
    low = np.array([c["low"] for c in candles])
    close = np.array([c["close"] for c in candles])
    timestamps = np.array([c["ts"] for c in candles])

    logger.info("Backtest: %s [%s], %d candles", symbol, timeframe, len(candles))

    # 1. Extrema
    finder = ExtremumFinder(mode="single")
    from src.core.config import ATR_FRACTAL_SETTINGS
    settings = ATR_FRACTAL_SETTINGS.get(timeframe, {"fractal_n": 2, "atr_mult": 1.5, "atr_period": 14})
    extrema = finder.find(high, low, close, timestamps, **settings)
    logger.info("Found %d extrema", len(extrema))

    # 2. Liquidity Map
    liq = LiquidityMapper(symbol, timeframe)
    liq.build_from_history(high, low, close, timestamps)

    # 3. DAG
    dag = HypothesisDAG()
    for ext in extrema:
        dag.ingest_extremum(ext)

    top = dag.get_top_hypotheses(5)
    completed = dag.completed_hypotheses
    logger.info("DAG: %d active, %d completed", len(dag.active_hypotheses), len(completed))

    # 4. Score top hypotheses
    scorer = MLScorer()
    sf = SignalFilter(threshold=0.60)
    rg = ReportGenerator()

    signals = []
    for hyp in (completed[-10:] + top):
        features = FeatureExtractor.extract_features(hyp)
        prob = scorer.predict_proba(features)

        signal = sf.evaluate(symbol, hyp, features, prob)
        if signal:
            msg = rg.format_telegram_message(signal)
            signals.append(msg)
            print("\n" + "=" * 60)
            print(msg)
            print("=" * 60)

    logger.info("Backtest complete. Generated %d signals.", len(signals))
    store.close()


# ═════════════════════════════════════════════════════════════════════════════
# MODE: LIVE — Real-time WS streaming + DAG + Signals
# ═════════════════════════════════════════════════════════════════════════════

async def run_live(symbol: str, timeframes_list: list[str]):
    """Real-time режим: WS → ClusterBuilder → LiquidityMapper → DAG → Signals."""
    from src.events.bus import get_event_bus
    from src.events.models import EventType, NewCandleEvent
    from src.ingestion.ws_streamer import WSStreamer
    from src.ingestion.cluster_builder import ClusterBuilder
    from src.ingestion.liquidity_mapper import LiquidityMapper
    from src.ingestion.sniper_trigger import SniperTrigger
    from src.wave_engine.extremum_finder import ExtremumFinder
    from src.wave_engine.hypothesis_dag import HypothesisDAG
    from src.confluence.feature_extractor import FeatureExtractor
    from src.confluence.ml_scorer import MLScorer
    from src.confluence.signal_filter import SignalFilter
    from src.execution.report_generator import ReportGenerator
    from src.storage.duckdb_store import get_store

    bus = get_event_bus()
    store = get_store()

    # Components
    primary_tf = timeframes_list[0] if timeframes_list else "1h"
    streamer = WSStreamer(symbol, timeframes_list)
    cluster = ClusterBuilder(symbol, primary_tf, tick_size=0.01)
    liq_mapper = LiquidityMapper(symbol, primary_tf)
    sniper = SniperTrigger()
    dag = HypothesisDAG()
    finder = ExtremumFinder(mode="single")
    scorer = MLScorer()
    sf = SignalFilter()
    rg = ReportGenerator()

    # Preload history for liquidity map
    candles = store.get_ohlcv(symbol, primary_tf)
    if candles:
        high = np.array([c["high"] for c in candles])
        low = np.array([c["low"] for c in candles])
        close = np.array([c["close"] for c in candles])
        ts = np.array([c["ts"] for c in candles])
        liq_mapper.build_from_history(high, low, close, ts)
        logger.info("Preloaded %d historical candles for liquidity map", len(candles))

    # Wire up events
    async def on_candle(event: NewCandleEvent):
        if event.timeframe != primary_tf:
            return

        # Store candle
        store.upsert_ohlcv(event.symbol, event.timeframe, [{
            "ts": event.ts, "open": event.open, "high": event.high,
            "low": event.low, "close": event.close, "volume": event.volume,
        }])

        # Liquidity check
        liq_result = liq_mapper.on_new_candle({
            "ts": event.ts, "open": event.open, "high": event.high,
            "low": event.low, "close": event.close, "volume": event.volume,
        })

        # Sniper reset (new candle = reset sticky states)
        sniper.reset_candle_state(event.symbol)

        # Re-run extremum detection on latest data
        all_candles = store.get_ohlcv(event.symbol, event.timeframe)
        if len(all_candles) < 30:
            return

        h = np.array([c["high"] for c in all_candles[-200:]])
        l = np.array([c["low"] for c in all_candles[-200:]])
        c = np.array([c["close"] for c in all_candles[-200:]])
        t = np.array([c_["ts"] for c_ in all_candles[-200:]])

        from src.core.config import ATR_FRACTAL_SETTINGS
        settings = ATR_FRACTAL_SETTINGS.get(event.timeframe, {})
        extrema = finder.find(h, l, c, t, **settings)

        # Feed last extremum to DAG
        if extrema:
            dag.ingest_extremum(extrema[-1])

        # Score top hypothesis
        top = dag.get_top_hypotheses(1)
        if top:
            hyp = top[0]
            context = {
                "cluster_volume_zscore": cluster.get_context_for_ml().get("cluster_volume_zscore", 0),
                "liquidity_sweep": liq_result.get("liquidity_sweep", 0),
            }
            features = FeatureExtractor.extract_features(hyp, context)
            prob = scorer.predict_proba(features)
            signal = sf.evaluate(event.symbol, hyp, features, prob)

            if signal:
                msg = rg.format_telegram_message(signal)
                logger.info("SIGNAL GENERATED:\n%s", msg)
                rg.send_telegram_alert(msg)

    bus.subscribe(EventType.NEW_CANDLE, on_candle)

    # Start
    await bus.start()
    await cluster.start(streamer.trade_queue)
    await streamer.start()

    logger.info("=== RalphTradeBot V3 LIVE mode started ===")
    logger.info("Symbol: %s | TFs: %s | Primary: %s", symbol, timeframes_list, primary_tf)

    try:
        while True:
            await asyncio.sleep(60)
            logger.info("Heartbeat | Streamer: %s | DAG active: %d",
                       streamer.stats, len(dag.active_hypotheses))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await streamer.stop()
        await cluster.stop()
        await bus.stop()
        store.close()


# ═════════════════════════════════════════════════════════════════════════════
# MODE: TRAIN — Обучение ML модели
# ═════════════════════════════════════════════════════════════════════════════

def run_train():
    """Запускает пайплайн обучения ML-модели на HITL-разметке."""
    from src.confluence.training_pipeline import train_model
    success = train_model()
    if success:
        logger.info("Model training complete.")
    else:
        logger.warning("Training skipped (insufficient data).")


# ═════════════════════════════════════════════════════════════════════════════
# MODE: LABEL — Streamlit HITL Dashboard
# ═════════════════════════════════════════════════════════════════════════════

def run_label():
    """Запускает Streamlit dashboard для HITL разметки."""
    import subprocess
    app_path = "src/labeling/app.py"
    logger.info("Launching HITL Dashboard: streamlit run %s", app_path)
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RalphTradeBot V3 — Elliott Wave Engine")
    parser.add_argument("--mode", choices=["live", "backtest", "train", "label", "fetch"],
                        default="backtest", help="Режим работы")
    parser.add_argument("--symbol", default=SYMBOLS[0], help="Торговая пара (default: BTCUSDT)")
    parser.add_argument("--timeframe", default="1h", help="Основной таймфрейм (default: 1h)")

    args = parser.parse_args()

    logger.info("RalphTradeBot V3 | Mode: %s | Symbol: %s | TF: %s",
                args.mode, args.symbol, args.timeframe)

    if args.mode == "fetch":
        run_fetch([args.symbol], TIMEFRAMES)

    elif args.mode == "backtest":
        run_backtest(args.symbol, args.timeframe)

    elif args.mode == "live":
        asyncio.run(run_live(args.symbol, TIMEFRAMES))

    elif args.mode == "train":
        run_train()

    elif args.mode == "label":
        run_label()


if __name__ == "__main__":
    main()
