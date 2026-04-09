
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

def plot_wave_structure(symbol, timeframe, candles, wave_structure, output_dir="reports/charts"):
    """
    Отрисовывает свечной график и накладывает волновую структуру.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.DataFrame(candles)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    
    plt.figure(figsize=(15, 8))
    plt.style.use('dark_background')
    
    # 1. Рисуем цену (упрощенно линией или свечами если данных мало)
    plt.plot(df['datetime'], df['close'], color='gray', alpha=0.3, label='Price')
    
    # 2. Отрисовываем волны
    pts = wave_structure.points
    wave_x = [pd.to_datetime(p.timestamp, unit='ms') for p in pts]
    wave_y = [p.price for p in pts]
    
    # Цвет в зависимости от направления
    color = 'cyan' if wave_structure.direction == "БЫЧИЙ" else 'orange'
    
    # Сами линии волн
    plt.plot(wave_x, wave_y, color=color, linewidth=3, marker='o', markersize=8, label=f"{wave_structure.pattern_type} {wave_structure.direction}")
    
    # Подписи точек (0, 1, 2, 3, 4, 5...)
    for p in pts:
        plt.annotate(p.label, 
                     (pd.to_datetime(p.timestamp, unit='ms'), p.price),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', 
                     color='white', 
                     fontsize=12, 
                     fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.5))

    plt.title(f"{symbol} [{timeframe}] - {wave_structure.pattern_type} ({wave_structure.confidence:.1f}%)", fontsize=16)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.2)
    plt.legend()
    
    # Сохраняем
    filename = f"{symbol}_{timeframe}_{wave_structure.pattern_type}_{datetime.now().strftime('%H%M%S')}.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    return path
