import json, sys

data = json.load(open(r'c:\Users\user\Desktop\RalphTradeBot\reports\BTCUSDT_raw_2026-03-12_15-26-14.json', encoding='utf-8'))
for tf in data.get('context', {}).get('timeframes', []):
    ws = tf.get('mathematical_wave_state', '')
    sys.stdout.write(f"\n=== {tf['timeframe']} ===\n")
    sys.stdout.write(ws[:1500] + "\n")
    sys.stdout.flush()
