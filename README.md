# Dataset Converter for Qwen2.5 Fine-tuning

Простой скрипт для конвертации датасетов переводов в формат ChatML для fine-tuning модели Qwen2.5.

## Установка

1. Создайте виртуальное окружение (если еще не создано):
```bash
python3 -m venv venv
```

2. Активируйте виртуальное окружение:
   - На macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   - На Windows:
   ```bash
   venv\Scripts\activate
   ```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` на основе `.env.example`:
```bash
cp .env.example .env
```

5. Откройте `.env` и добавьте ваш токен Hugging Face с правами WRITE (можно получить в [settings/tokens](https://huggingface.co/settings/tokens)):
```bash
HF_TOKEN=your_huggingface_token_here
```

## Использование

1. Убедитесь, что файл `.env` создан и содержит ваш токен Hugging Face

2. При необходимости измените настройки в `main.py`:
   - `SOURCE_DATASET` - исходный датасет
   - `SPLIT_NAME` - название split (по умолчанию "Somali")
   - `NEW_DATASET_NAME` - имя нового датасета в вашем профиле

3. Запустите bash-скрипт (рекомендуется):
```bash
./run.sh
```

Или запустите вручную:
```bash
source venv/bin/activate
python main.py
```

Bash-скрипт `run.sh` автоматически создаст виртуальное окружение (если его нет), установит зависимости и запустит основной скрипт.

Скрипт загрузит датасет, конвертирует его в формат ChatML и загрузит обратно в ваш профиль на Hugging Face.

## Формат вывода

Данные конвертируются в формат ChatML:
```
<|im_start|>user
Translate to Somali: [английский текст]<|im_end|>
<|im_start|>assistant
[сомалийский перевод]<|im_end|>
```
