#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Проверка виртуального окружения..."
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
fi

echo "Установка зависимостей..."
./venv/bin/pip install -q -r requirements.txt

echo "Запуск скрипта..."
./venv/bin/python main.py

echo "Готово!"
