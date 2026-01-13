# -*- coding: utf-8 -*-
"""
Скрипт для конвертации JSON файлов в JSONL формат для together.ai fine-tuning
"""
import json
import os
import sys

def convert_json_to_jsonl(json_file_path, jsonl_file_path=None):
    """
    Конвертирует JSON файл (массив объектов) в JSONL формат (каждая строка - JSON объект)
    
    Args:
        json_file_path: путь к входному JSON файлу
        jsonl_file_path: путь к выходному JSONL файлу (если None, создается автоматически)
    """
    if jsonl_file_path is None:
        jsonl_file_path = json_file_path.replace('.json', '.jsonl')
    
    print(f"Чтение JSON файла: {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON файл должен содержать массив объектов")
    
    print(f"Конвертация {len(data)} записей в JSONL формат...")
    with open(jsonl_file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"✓ JSONL файл создан: {jsonl_file_path}")
    print(f"  Размер файла: {os.path.getsize(jsonl_file_path) / 1024 / 1024:.2f} MB")
    return jsonl_file_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python convert_to_jsonl.py <путь_к_json_файлу> [путь_к_jsonl_файлу]")
        print("\nПримеры:")
        print("  python convert_to_jsonl.py output/gaado-somali-dataset_light_train.json")
        print("  python convert_to_jsonl.py output/gaado-somali-dataset_light_train.json output/training.jsonl")
        sys.exit(1)
    
    json_file = sys.argv[1]
    jsonl_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(json_file):
        print(f"Ошибка: файл {json_file} не найден!")
        sys.exit(1)
    
    convert_json_to_jsonl(json_file, jsonl_file)
