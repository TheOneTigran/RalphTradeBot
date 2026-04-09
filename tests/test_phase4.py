"""E2E test: Full pipeline from Hypothesis -> Features -> ML Score -> Signal -> Report."""

from src.wave_engine.extremum_finder import Extremum
from src.wave_engine.hypothesis_dag import WaveHypothesis, PatternType, WaveDegree
from src.confluence.feature_extractor import FeatureExtractor
from src.confluence.ml_scorer import MLScorer
from src.confluence.signal_filter import SignalFilter
from src.execution.report_generator import ReportGenerator
from src.ingestion.sniper_trigger import SniperTrigger, SniperSetup


def test_full_pipeline():
    """End-to-end: DAG hypothesis -> Telegram Telegram report."""
    # 1. Build a hypothesis (simulating DAG output)
    hyp = WaveHypothesis(
        pattern_type=PatternType.IMPULSE,
        degree=WaveDegree.INTERMEDIATE,
        is_bullish=True,
        confidence=0.72,
    )
    hyp.points = [
        Extremum(price=60000, timestamp=1000, index=0, is_high=False),
        Extremum(price=65000, timestamp=2000, index=10, is_high=True),
        Extremum(price=62000, timestamp=3000, index=20, is_high=False),
        Extremum(price=71000, timestamp=4000, index=30, is_high=True),
        Extremum(price=67000, timestamp=5000, index=40, is_high=False),
    ]
    hyp.features = {"alternation_depth": True, "w3_extended": True}

    # 2. Extract features
    context = {
        "cluster_volume_zscore": 2.8,
        "liquidity_sweep": 1.0,
        "rsi_divergence": 1.0,
        "volume_delta_ratio": 0.3,
        "funding_extreme": 0.4,
        "move_in_atr": 1.5,
    }
    features = FeatureExtractor.extract_features(hyp, context)

    # 3. ML scoring
    scorer = MLScorer()
    probability = scorer.predict_proba(features)

    # 4. Signal filter
    sf = SignalFilter(threshold=0.65)
    signal = sf.evaluate("BTCUSDT", hyp, features, probability)
    assert signal is not None

    # 5. Report Report
    rg = ReportGenerator()
    telegram_msg = rg.format_telegram_message(signal)
    
    msg_bytes = ("\n--- TELEGRAM MESSAGE ---\n" + telegram_msg + "\n--- END ---\n").encode("utf-8")
    import sys
    sys.stdout.buffer.write(msg_bytes)
    print("Full pipeline test PASSED")


def test_sniper_trigger():
    """Test SniperTrigger with synthetic data."""
    sniper = SniperTrigger()
    setup = SniperSetup(
        hypothesis_id="test-123", symbol="BTCUSDT", direction="LONG",
        fibo_zone=[60000, 61000], invalidation_level=59500,
        take_profit_targets=[63000, 65000], degree="INTERMEDIATE",
        pattern_type="IMPULSE", confidence=0.85,
    )
    sniper.arm(setup)
    sniper.on_tick("BTCUSDT", price=60200, absorption=True, sweep=True)
    assert sniper.fired_count == 1
    print("SniperTrigger test PASSED")


if __name__ == "__main__":
    test_full_pipeline()
    print()
    test_sniper_trigger()
    print()
    print("ALL Phase 4 tests PASSED")
