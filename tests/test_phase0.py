"""Quick smoke test for Phase 0 components."""
import asyncio
from src.events.bus import EventBus
from src.events.models import EventType, NewCandleEvent

async def test_event_bus():
    bus = EventBus()
    received = []
    
    async def on_candle(event):
        received.append(event)
    
    bus.subscribe(EventType.NEW_CANDLE, on_candle)
    
    # Direct dispatch (no loop)
    evt = NewCandleEvent(
        symbol="BTCUSDT", timeframe="1h", ts=1000,
        open=100, high=105, low=95, close=102, volume=5000
    )
    await bus.publish(evt)
    
    assert len(received) == 1, f"Expected 1 event, got {len(received)}"
    assert received[0].symbol == "BTCUSDT"
    assert received[0].close == 102
    
    # Test with background loop
    await bus.start()
    evt2 = NewCandleEvent(
        symbol="ETHUSDT", timeframe="4h", ts=2000,
        open=3000, high=3100, low=2900, close=3050, volume=10000
    )
    await bus.publish(evt2)
    await asyncio.sleep(0.3)
    assert len(received) == 2, f"Expected 2 events, got {len(received)}"
    
    stats = bus.stats
    await bus.stop()
    
    print(f"Events processed: {stats['total_events']}")
    print(f"Subscribers: {stats['subscriber_counts']}")
    print("EventBus test PASSED")

if __name__ == "__main__":
    asyncio.run(test_event_bus())
