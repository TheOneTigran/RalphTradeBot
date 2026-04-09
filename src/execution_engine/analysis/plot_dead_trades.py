"""
plot_dead_trades.py — Визуальная отладка убыточных сделок.
Берет 5 случайных сделок со статусом FILLED_SL из backtest_results.csv и рисует график.
"""
import os
import random
import pandas as pd
import plotly.graph_objects as go

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(root_dir, "src", "execution_engine", "tests", "data")
    fixture_path = os.path.join(data_dir, "btc_1m.csv")
    
    if not os.path.exists(csv_path):
        print("Сначала запустите run_backtest_demo.py для генерации CSV.")
        return
        
    df_trades = pd.read_csv(csv_path)
    df_dead = df_trades[df_trades["Status"] == "FILLED_SL"]
    
    if df_dead.empty:
        print("Убыточных сделок не найдено (или файл пуст).")
        return
        
    # Берем до 5 случайных сделок
    sample_size = min(5, len(df_dead))
    dead_samples = df_dead.sample(sample_size).to_dict('records')
    
    print(f"Выбрано {sample_size} убыточных сделок для визуализации.")
    
    # Читаем свечи для контекста
    df_candles = pd.read_csv(fixture_path)
    if 'ts' in df_candles.columns:
        df_candles['datetime'] = pd.to_datetime(df_candles['ts'], unit='ms')
    elif 'timestamp' in df_candles.columns:
        df_candles['datetime'] = pd.to_datetime(df_candles['timestamp'], unit='ms')
    
    # Сортируем
    df_candles = df_candles.sort_values('datetime').reset_index()
    
    for i, trade in enumerate(dead_samples):
        # Парсим время
        entry_time = pd.to_datetime(trade['Entry_Time'])
        exit_time = pd.to_datetime(trade['Exit_Time'])
        
        # Находим индексы свечей
        # Поскольку это 1H свечи названные 1m в тестах, мы просто найдем их по времени
        mask_entry = df_candles['datetime'] <= entry_time
        if mask_entry.any():
            start_idx = df_candles[mask_entry].index[-1] - 10 # 10 свечей до входа
            start_idx = max(0, start_idx)
        else:
            start_idx = 0
            
        mask_exit = df_candles['datetime'] >= exit_time
        if mask_exit.any():
            end_idx = df_candles[mask_exit].index[0] + 10 # 10 свечей после выхода
            end_idx = min(len(df_candles)-1, end_idx)
        else:
            end_idx = len(df_candles)-1
            
        context_df = df_candles.iloc[start_idx:end_idx+1]
        
        # Строим график
        fig = go.Figure()
        
        # 1. Свечи
        fig.add_trace(go.Candlestick(
            x=context_df['datetime'],
            open=context_df['open'],
            high=context_df['high'],
            low=context_df['low'],
            close=context_df['close'],
            name='Market'
        ))
        
        # 2. Уровни сделки
        entry_p = float(trade['Entry_Price'])
        # Мы не сохраняли SL/TP в CSV, но мы можем аппроксимировать или просто показать вход и выход
        exit_p = float(trade['Exit_Price'])
        
        # Линия входа
        fig.add_hline(y=entry_p, line_dash="dash", line_color="blue", annotation_text="Entry")
        # Линия выхода (в данном случае это SL)
        fig.add_hline(y=exit_p, line_dash="solid", line_color="red", annotation_text=f"SL Hit ({trade['Direction']})")
        
        # Отметки времени
        fig.add_vline(x=entry_time, line_dash="dot", line_color="gray")
        fig.add_vline(x=exit_time, line_dash="dot", line_color="gray")
        
        fig.update_layout(
            title=f"Dead Trade #{i+1} | {trade['Direction']} | Duration: {trade['Trade_Duration_Bars']} bars",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )
        
        out_html = os.path.join(root_dir, f"dead_trade_{i+1}.html")
        fig.write_html(out_html)
        print(f"[{i+1}] Сохранен график: {out_html}")
        
if __name__ == "__main__":
    main()
