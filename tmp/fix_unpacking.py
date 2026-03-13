
import os

path = r'c:\Users\user\Desktop\RalphTradeBot\src\core\math_actor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix unpacking error at line 119 (approx)
text = text.replace('alt_tf, alt_s = all_valid_structures[1]', 'alt_tf, alt_s, _alt_atr = all_valid_structures[1]')

# Double check other unpackings
text = text.replace('for tf, s in all_valid_structures:', 'for tf, s, atr in all_valid_structures:')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("Fixed tuple unpacking in math_actor.py")
