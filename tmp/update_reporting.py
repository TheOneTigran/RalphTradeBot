
import os

# 1. Update math_actor.py for leniency
path_actor = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path_actor, 'r', encoding='utf-8') as f:
    actor_text = f.read()
actor_text = actor_text.replace('extrema[-10:]', 'extrema[-20:]')
with open(path_actor, 'w', encoding='utf-8') as f:
    f.write(actor_text)

# 2. Update runner.py to include RR_FILTERED trades
path_runner = r'c:\Users\user\Desktop\RalphTradeBot\src\backtester\runner.py'
with open(path_runner, 'r', encoding='utf-8') as f:
    runner_text = f.read()

# Change the logic to NOT skip WAIT plans but check if they are RR Filtered
old_filter = r'''            direction = str(plan.trade_params.get("direction", "")).upper()
            if "WAIT" in direction or not direction:
                continue'''

new_filter = r'''            direction = str(plan.trade_params.get("direction", "")).upper()
            if not direction: continue
            
            if "WAIT" in direction:
                if plan.wave_count_label == "RR Filtered":
                    # Логируем отсеянный по RR паттерн
                    results.append({
                        "ts": ts,
                        "date": dt.strftime('%Y-%m-%d %H:%M'),
                        "direction": "WAIT (RR)",
                        "status": "RR_FILTERED",
                        "rr_ratio": 0,
                        "pnl_pct": 0,
                        "pnl_usd": 0,
                        "fee_usd": 0,
                        "balance": round(current_balance, 2),
                        "pos_size_usd": 0,
                        "leverage": 0,
                        "reasoning": plan.detailed_logic[:200]
                    })
                continue'''

runner_text = runner_text.replace(old_filter, new_filter)
with open(path_runner, 'w', encoding='utf-8') as f:
    f.write(runner_text)

print("Updated actor freshness and runner reporting.")
