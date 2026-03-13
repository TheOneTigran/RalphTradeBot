
import sys
import os

filepath = r'c:\Users\user\Desktop\RalphTradeBot\src\math_engine\wave_analyzer.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# I'll use a string replacement since I know the exact structure
bullish_search = """    if bullish:
        if not (p1.price > p0.price and p2.price < p1.price and
                p3.price > p2.price and p4.price < p3.price and
                p5.price > p4.price):
            return None"""

bullish_replace = """    if bullish:
        if not (p1.price > p0.price and p2.price < p1.price and
                p3.price > p2.price and p4.price < p3.price and
                p5.price > p4.price):
            return None
        if p3.price <= p1.price: return None"""

bearish_search = """    else:
        if not (p1.price < p0.price and p2.price > p1.price and
                p3.price < p2.price and p4.price > p3.price and
                p5.price < p4.price):
            return None"""

bearish_replace = """    else:
        if not (p1.price < p0.price and p2.price > p1.price and
                p3.price < p2.price and p4.price > p3.price and
                p5.price < p4.price):
            return None
        if p3.price >= p1.price: return None"""

if bullish_search in content:
    content = content.replace(bullish_search, bullish_replace)
    print("Replaced bullish")
else:
    print("Bullish search not found")

if bearish_search in content:
    content = content.replace(bearish_search, bearish_replace)
    print("Replaced bearish")
else:
    print("Bearish search not found")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
