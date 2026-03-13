
import os

path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Clean up debug prints and logs to keep output clean
text = text.replace('print(f"!!! [RR_FILTER] Plan rejected for {best_tf} {best_s.pattern_type}"); ', '')
text = text.replace('print(f"    [RR_FAIL] {direction} {tf} {s.pattern_type} RR={reward/risk:.2f}"); ', '')
text = text.replace('logger.info(f"    [DEBUG]', '# logger.info(f"    [DEBUG]')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Cleaned up math_actor.py")
