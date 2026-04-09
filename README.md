# 🌊 RalphTradeBot — Elliott Wave HITL Platform

Аналитический ассистент для разволновки крипто-рынков по Эллиотту с ML-слоем.

## Деплой публичного дашборда (Streamlit Community Cloud)

### 1. Залить код на GitHub

```bash
# В папке проекта
git init
git add .
git commit -m "Initial commit"
gh repo create ralph-trade-bot --public --push
```

### 2. Задеплоить на Streamlit Cloud

1. Зайди на **https://share.streamlit.io**
2. Нажми **New app**
3. Выбери репозиторий `ralph-trade-bot`
4. **Main file path:** `app_community.py`
5. Нажми **Deploy!**

Через ~2 минуты получишь публичную ссылку вида `https://yourname-ralph-trade-bot.streamlit.app`

### 3. Запуск локально (для разработки)

```bash
# Локальная версия (с DuckDB, полный движок)
python main.py --mode label

# Облачная версия (с SQLite, для тестирования деплоя)  
streamlit run app_community.py
```

### Архитектура данных

| Сценарий | БД | Описание |
|---|---|---|
| Локально | DuckDB (`data/ralph.duckdb`) | Полный OLAP, кластеры, тики |
| Облако (Streamlit Cloud) | SQLite (`data/community.sqlite3`) | OHLCV подгружается с Bybit live |

### Команды

```bash
python main.py --mode fetch    # Скачать 5000 свечей в DuckDB
python main.py --mode label    # Запустить локальный HITL
python main.py --mode train    # Обучить XGBoost на размеченных данных
python main.py --mode live     # Forward Testing + Telegram алерты
```
