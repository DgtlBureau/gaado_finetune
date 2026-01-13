# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение токена Hugging Face из переменных окружения
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN не найден! Создайте файл .env и добавьте туда: HF_TOKEN=your_token_here\n"
        "Или установите переменную окружения: export HF_TOKEN=your_token_here"
    )

login(HF_TOKEN)

SOURCE_DATASET = "almanach/topxgen-gemma-3-27b-and-nllb-3.3b"
SPLIT_NAME = "Somali"
DATASET_NAME = "gaado-somali-dataset"
FULL_SPLIT_NAME = "train"
LIGHT_SPLIT_NAME = "light_train"
OUTPUT_DIR = "output"
MAX_ROWS = 1000

print(f"Загрузка датасета {SOURCE_DATASET}...")
ds = load_dataset(SOURCE_DATASET)
full_dataset = ds[SPLIT_NAME]

print(f"Создание light версии ({MAX_ROWS} строк)...")
light_dataset = full_dataset.select(range(min(MAX_ROWS, len(full_dataset))))
print(f"Полный датасет: {len(full_dataset)} строк")
print(f"Light датасет: {len(light_dataset)} строк")

def format_for_qwen(example):
    chat_text = (
        f"<|im_start|>user\nTranslate from Somali to English: {example['target']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['source']}<|im_end|>"
    )
    return {"text": chat_text}

print("Конвертация полного датасета в формат ChatML...")
clean_full_dataset = full_dataset.map(
    format_for_qwen,
    remove_columns=full_dataset.column_names,
    desc="Formatting full dataset to ChatML"
)

print("Конвертация light датасета в формат ChatML...")
clean_light_dataset = light_dataset.map(
    format_for_qwen,
    remove_columns=light_dataset.column_names,
    desc="Formatting light dataset to ChatML"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

full_parquet_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_{FULL_SPLIT_NAME}.parquet")
full_json_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_{FULL_SPLIT_NAME}.json")
light_parquet_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_{LIGHT_SPLIT_NAME}.parquet")
light_json_path = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_{LIGHT_SPLIT_NAME}.json")

print(f"Сохранение полного датасета в Parquet: {full_parquet_path}...")
df_full = clean_full_dataset.to_pandas()
df_full.to_parquet(full_parquet_path, engine="pyarrow", compression="snappy", index=False)

print(f"Сохранение полного датасета в JSON: {full_json_path}...")
json_data_full = [{"text": item["text"]} for item in clean_full_dataset]
with open(full_json_path, "w", encoding="utf-8") as f:
    json.dump(json_data_full, f, ensure_ascii=False, indent=2)

# Создание JSONL файла для together.ai
full_jsonl_path = full_json_path.replace('.json', '.jsonl')
print(f"Сохранение полного датасета в JSONL (для together.ai): {full_jsonl_path}...")
with open(full_jsonl_path, "w", encoding="utf-8") as f:
    for item in json_data_full:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"Сохранение light датасета в Parquet: {light_parquet_path}...")
df_light = clean_light_dataset.to_pandas()
df_light.to_parquet(light_parquet_path, engine="pyarrow", compression="snappy", index=False)

print(f"Сохранение light датасета в JSON: {light_json_path}...")
json_data_light = [{"text": item["text"]} for item in clean_light_dataset]
with open(light_json_path, "w", encoding="utf-8") as f:
    json.dump(json_data_light, f, ensure_ascii=False, indent=2)

# Создание JSONL файла для together.ai
light_jsonl_path = light_json_path.replace('.json', '.jsonl')
print(f"Сохранение light датасета в JSONL (для together.ai): {light_jsonl_path}...")
with open(light_jsonl_path, "w", encoding="utf-8") as f:
    for item in json_data_light:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"Загрузка split '{FULL_SPLIT_NAME}' в датасет {DATASET_NAME}...")
clean_full_dataset.push_to_hub(DATASET_NAME, split=FULL_SPLIT_NAME)

print(f"Загрузка split '{LIGHT_SPLIT_NAME}' в датасет {DATASET_NAME}...")
clean_light_dataset.push_to_hub(DATASET_NAME, split=LIGHT_SPLIT_NAME)

print(f"Готово! Файлы сохранены в папку {OUTPUT_DIR}/")
print(f"Split '{FULL_SPLIT_NAME}':")
print(f"  - {full_parquet_path}")
print(f"  - {full_json_path}")
print(f"  - {full_jsonl_path} (для together.ai)")
print(f"Split '{LIGHT_SPLIT_NAME}':")
print(f"  - {light_parquet_path}")
print(f"  - {light_json_path}")
print(f"  - {light_jsonl_path} (для together.ai)")
print(f"Датасет {DATASET_NAME} с двумя сплитами загружен в Hugging Face:")
print(f"  - {DATASET_NAME} (split: {FULL_SPLIT_NAME}, {len(clean_full_dataset)} строк)")
print(f"  - {DATASET_NAME} (split: {LIGHT_SPLIT_NAME}, {len(clean_light_dataset)} строк)")
print(f"\nДля together.ai используйте JSONL файлы:")
print(f"  - {full_jsonl_path}")
print(f"  - {light_jsonl_path}")
print(f"\n⚠️  ВАЖНО: Для together.ai нужны training и validation файлы!")
print(f"Разделите JSONL файлы на train/validation с помощью:")
print(f"  python3 split_train_val.py {full_jsonl_path}")
print(f"  python3 split_train_val.py {light_jsonl_path}")
