# -*- coding: utf-8 -*-
"""
Скрипт для разделения JSONL файла на training и validation наборы для together.ai fine-tuning
"""
import json
import os
import sys
import random

def split_jsonl(input_file, train_file=None, validation_file=None, train_ratio=0.9, shuffle=True, seed=42):
    """
    Разделяет JSONL файл на training и validation наборы
    
    Args:
        input_file: путь к входному JSONL файлу
        train_file: путь к выходному training файлу (если None, создается автоматически)
        validation_file: путь к выходному validation файлу (если None, создается автоматически)
        train_ratio: доля данных для training (по умолчанию 0.9 = 90%)
        shuffle: перемешивать ли данные перед разделением
        seed: seed для случайного перемешивания
    """
    if train_file is None:
        train_file = input_file.replace('.jsonl', '_train.jsonl')
    if validation_file is None:
        validation_file = input_file.replace('.jsonl', '_validation.jsonl')
    
    print(f"Чтение файла: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Всего строк: {total_lines}")
    
    # Парсим JSON объекты
    data = []
    for line in lines:
        if line.strip():
            data.append(json.loads(line))
    
    # Перемешиваем, если нужно
    if shuffle:
        random.seed(seed)
        random.shuffle(data)
        print(f"Данные перемешаны (seed={seed})")
    
    # Разделяем на train и validation
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    validation_data = data[train_size:]
    
    print(f"\nРазделение данных:")
    print(f"  Training: {len(train_data)} строк ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validation: {len(validation_data)} строк ({len(validation_data)/len(data)*100:.1f}%)")
    
    # Сохраняем training файл
    print(f"\nСохранение training файла: {train_file}...")
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"✓ Training файл создан: {train_file}")
    print(f"  Размер: {os.path.getsize(train_file) / 1024 / 1024:.2f} MB")
    
    # Сохраняем validation файл
    print(f"\nСохранение validation файла: {validation_file}...")
    with open(validation_file, 'w', encoding='utf-8') as f:
        for item in validation_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"✓ Validation файл создан: {validation_file}")
    print(f"  Размер: {os.path.getsize(validation_file) / 1024 / 1024:.2f} MB")
    
    return train_file, validation_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python split_train_val.py <input.jsonl> [train_ratio] [--no-shuffle]")
        print("\nПараметры:")
        print("  input.jsonl    - входной JSONL файл")
        print("  train_ratio    - доля данных для training (по умолчанию 0.9 = 90%)")
        print("  --no-shuffle   - не перемешивать данные перед разделением")
        print("\nПримеры:")
        print("  python split_train_val.py output/gaado-somali-dataset_light_train.jsonl")
        print("  python split_train_val.py output/gaado-somali-dataset_light_train.jsonl 0.8")
        print("  python split_train_val.py output/gaado-somali-dataset_light_train.jsonl 0.9 --no-shuffle")
        sys.exit(1)
    
    input_file = sys.argv[1]
    train_ratio = 0.9
    shuffle = True
    
    if len(sys.argv) > 2:
        if sys.argv[2] == '--no-shuffle':
            shuffle = False
        else:
            try:
                train_ratio = float(sys.argv[2])
                if not 0 < train_ratio < 1:
                    print("Ошибка: train_ratio должен быть между 0 и 1")
                    sys.exit(1)
            except ValueError:
                print(f"Ошибка: неверное значение train_ratio: {sys.argv[2]}")
                sys.exit(1)
    
    if len(sys.argv) > 3 and sys.argv[-1] == '--no-shuffle':
        shuffle = False
    
    if not os.path.exists(input_file):
        print(f"Ошибка: файл {input_file} не найден!")
        sys.exit(1)
    
    split_jsonl(input_file, train_ratio=train_ratio, shuffle=shuffle)
