FROM python:3.12-slim

WORKDIR /app

# Системные зависимости для matplotlib, PIL и сетевых запросов
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создание директорий для результатов
RUN mkdir -p results/confirmed_signals results/rejected_signals data

ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

CMD ["python", "live_scanner.py", "--interval", "60", "--workers", "8"]
