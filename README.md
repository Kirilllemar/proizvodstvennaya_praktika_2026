# FoodVision — Вариант 4 (Распознавание блюда по фотографии)

Веб-приложение для производственной практики: классификация блюд с сравнением 5 архитектур CNN.

**Кроссплатформенно:** Windows, macOS, Linux.

## Требования

- Python **3.10+** ([python.org](https://www.python.org/downloads/))
- Интернет при первом запуске (скачивание весов PyTorch ~100 МБ)
- Браузер (Chrome, Firefox, Edge, Safari)

## Быстрый запуск

### Windows

Дважды щёлкните **`start.bat`** или в командной строке:

```cmd
cd food_recognition
start.bat
```

### macOS / Linux

```bash
cd food_recognition
chmod +x start.sh
./start.sh
```

### Любая ОС (универсально)

```bash
cd food_recognition
python start.py
```

`start.py` сам создаст `venv`, установит зависимости и запустит сервер.

Откройте в браузере: **http://127.0.0.1:5000**

## Ручная установка

```bash
cd food_recognition
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python run.py
```

## Переменные окружения (необязательно)

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `FOODVISION_HOST` | `127.0.0.1` | Адрес сервера |
| `FOODVISION_PORT` | `5000` | Порт |
| `FOODVISION_DEBUG` | `1` | Режим отладки Flask (`0` — выкл.) |

Пример:

```bash
FOODVISION_PORT=8080 python run.py
```

## Что реализовано

| Требование практики | Где в проекте |
|---------------------|---------------|
| Постановка задачи (вариант 4) | Главная, «О проекте» |
| Описание данных | `/about`, `data/experiments.json` |
| 5 архитектур CNN | `/compare` |
| Метрики (Acc, P, R, F1) | Таблица сравнения |
| Confusion Matrix | `/confusion` |
| Демо: фото → топ-3 на русском + уверенность | `/recognize` |
| Поиск нескольких блюд (сетка 3×3) | `/recognize` (чекбокс) |
| PDF с кириллицей | `static/fonts/DejaVuSans.ttf` |
| История запусков | SQLite, `/history` |
| Удаление истории | `/history` — по записи или полная очистка |
| Экспорт JSON / Excel / PDF | Кнопки на главной и в истории |
| Анализ ошибок | `/compare`, `/confusion` |

## Архитектуры

1. ResNet-18
2. ResNet-50
3. MobileNetV3-Small
4. **EfficientNet-B0** (лучшая)
5. DenseNet-121

## Структура

```
food_recognition/
  start.py            ← универсальный запуск (Windows / macOS / Linux)
  start.bat           ← запуск на Windows
  start.sh            ← запуск на macOS / Linux
  run.py              ← точка входа Flask
  requirements.txt
  requirements-dev.txt  ← playwright (для скриншотов отчёта)
  app/                ← логика Flask
  templates/          ← страницы
  static/             ← CSS, загрузки
  data/               ← метрики экспериментов, БД
```

## Скриншоты для отчёта (опционально)

```bash
pip install -r requirements-dev.txt
playwright install chromium
python ../capture_screenshots.py
```

## Демонстрация для защиты

1. Главная — статистика и этапы практики
2. «Распознать» — загрузить фото еды, получить топ-3 на русском
3. Включить «Искать несколько блюд» — показать режим 3×3 на фото со столом
4. Включить «Сравнить все 5 моделей» — показать разницу во времени
5. «Архитектуры» — таблица метрик для отчёта
6. «Confusion Matrix» — матрица ошибок
7. Скачать PDF/Excel для отчёта (кириллица в PDF)

## Устранение неполадок

**Python не найден (Windows)** — переустановите Python с галочкой «Add Python to PATH».

**Ошибка PyTorch** — обновите pip: `python -m pip install -U pip`, затем снова `pip install -r requirements.txt`.

**Порт 5000 занят** — `FOODVISION_PORT=8080 python run.py`

**Медленный первый запуск** — загрузка моделей в память занимает 30–60 с.
