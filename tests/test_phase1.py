"""Smoke tests for Phase 1: Ingestion Pipeline."""
import asyncio
import numpy as np

from src.ingestion.cluster_builder import ClusterBuilder
from src.ingestion.liquidity_mapper import LiquidityMapper


def test_cluster_builder():
    """Test ClusterBuilder with synthetic trade data."""
    cb = ClusterBuilder(symbol="BTCUSDT", timeframe="1h", tick_size=1.0)

    # Simulate 500 trades within one candle
    np.random.seed(42)
    for i in range(500):
        price = 100.0 + np.random.randn() * 5
        qty = abs(np.random.randn()) * 10
        is_sell = np.random.rand() > 0.5
        cb.ingest_trade({
            "trade_id": i,
            "price": price,
            "quantity": qty,
            "is_buyer_maker": is_sell,
            "trade_time": 1000 + i,
        })

    result = cb.flush(candle_ts=1000)
    assert result["poc_price"] > 0, "POC should be positive"
    assert result["total_volume"] > 0, "Total volume should be positive"
    assert result["vah_price"] >= result["val_price"], "VAH should be >= VAL"
    assert isinstance(result["absorption_detected"], bool)
    print(f"ClusterBuilder: POC={result['poc_price']:.2f}, "
          f"VAH={result['vah_price']:.2f}, VAL={result['val_price']:.2f}, "
          f"Delta={result['delta']:.2f}, ZScore={result['cluster_volume_zscore']:.2f}, "
          f"Absorption={result['absorption_detected']}")

    # Test that multiple flushes build z-score history
    for candle in range(5):
        for i in range(200):
            cb.ingest_trade({
                "trade_id": 1000 + candle * 200 + i,
                "price": 100 + np.random.randn() * 3,
                "quantity": abs(np.random.randn()) * 10,
                "is_buyer_maker": np.random.rand() > 0.5,
                "trade_time": 2000 + candle * 200 + i,
            })
        r = cb.flush(candle_ts=2000 + candle * 3600000)
    
    # After 6 total flushes, z-score should be calculable
    print(f"  Z-Score after 6 candles: {r['cluster_volume_zscore']:.3f}")
    print("ClusterBuilder test PASSED")


def test_liquidity_mapper():
    """Test LiquidityMapper with synthetic price data."""
    np.random.seed(42)

    # Generate price data with clear peaks and valleys
    n = 200
    t = np.linspace(0, 4 * np.pi, n)
    base = 100 + 20 * np.sin(t) + np.random.randn(n) * 2
    high = base + abs(np.random.randn(n)) * 3
    low = base - abs(np.random.randn(n)) * 3
    close = base + np.random.randn(n) * 1
    timestamps = (np.arange(n) * 3600000).astype(int)

    mapper = LiquidityMapper(symbol="BTCUSDT", timeframe="1h", min_age_candles=5)
    pools = mapper.build_from_history(high, low, close, timestamps)

    active = mapper.get_active_pools()
    total = len(pools)
    swept_count = sum(1 for p in pools if p.swept)
    
    print(f"LiquidityMapper: {total} pools found, {swept_count} swept, {len(active)} active")
    assert total > 0, "Should find at least some pools"

    # Simulate a sweep: price drops below a support pool then closes above
    support_pools = [p for p in active if not p.is_high]
    if support_pools:
        target = support_pools[0]
        candle = {
            "ts": int(timestamps[-1] + 3600000),
            "open": target.price + 1,
            "high": target.price + 2,
            "low": target.price - 3,   # Pierce below
            "close": target.price + 0.5,  # Close above = SWEEP
            "volume": 1000,
        }
        result = mapper.on_new_candle(candle)
        print(f"  Sweep test: liquidity_sweep={result['liquidity_sweep']}, "
              f"direction={result['sweep_direction']}, "
              f"nearest_pool={result['nearest_pool_distance']:.4f}")
        assert result["liquidity_sweep"] == 1.0, "Should detect sweep"
        assert result["sweep_direction"] == "bullish", "Should be bullish sweep"
    
    # ML context
    ctx = mapper.get_context_for_ml(close[-1])
    print(f"  ML context: {ctx}")
    print("LiquidityMapper test PASSED")


def test_event_integration():
    """Test that ClusterBuilder integrates with EventBus on candle close."""
    async def _run():
        from src.events.bus import get_event_bus, reset_event_bus
        from src.events.models import NewCandleEvent, EventType

        # Use the global singleton (same instance ClusterBuilder will use)
        reset_event_bus()
        bus = get_event_bus()
        cb = ClusterBuilder(symbol="BTCUSDT", timeframe="1h", tick_size=1.0)
        
        cluster_events = []

        async def capture_cluster(e):
            cluster_events.append(e)

        bus.subscribe(EventType.CLUSTER_UPDATED, capture_cluster)
        bus.subscribe(EventType.NEW_CANDLE, cb._on_candle_close)
        
        # Ingest some trades
        for i in range(100):
            cb.ingest_trade({
                "trade_id": i, "price": 50000 + i, "quantity": 0.1,
                "is_buyer_maker": i % 3 == 0, "trade_time": 1000 + i,
            })
        
        # Start bus loop for async dispatch
        await bus.start()

        candle_event = NewCandleEvent(
            symbol="BTCUSDT", timeframe="1h", ts=1000,
            open=50000, high=50100, low=49900, close=50050, volume=100,
        )
        await bus.publish(candle_event)

        # Wait for both events to process (NewCandle -> handler -> ClusterUpdated)
        import asyncio as _aio
        for _ in range(20):
            await _aio.sleep(0.1)
            if cluster_events:
                break
        await bus.stop()
        reset_event_bus()
        
        assert len(cluster_events) == 1, f"Expected 1 ClusterUpdatedEvent, got {len(cluster_events)}"
        ce = cluster_events[0]
        assert ce.poc_price > 0
        print(f"EventBus integration: ClusterUpdatedEvent emitted, POC={ce.poc_price:.2f}")
        print("Event integration test PASSED")
    
    asyncio.run(_run())


if __name__ == "__main__":
    test_cluster_builder()
    print()
    test_liquidity_mapper()
    print()
    test_event_integration()
    print()
    print("ALL Phase 1 tests PASSED")
