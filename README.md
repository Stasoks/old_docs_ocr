# Hybrid HTR and table OCR pipeline

Пайплайн для распознавания исторических русскоязычных PDF и
изображений. Он объединяет детекцию строк, рукописный OCR, поиск и распознавание
таблиц через VLM, а затем извлекает структурированные параметры скважин из
полученного документа.

Основной исполняемый файл — `hybrid_htr_table_pipeline.py`.

## Архитектура

Пайплайн использует четыре модели с разными задачами:

1. **RF-DETR** — детекция текстовых строк и текстовых регионов.
   По умолчанию загружается checkpoint
   `Kansallisarkisto/rfdetr_textline_textregion_detection_model`. Исходный
   `.pth` один раз конвертируется в тензорный `.safetensors`, и на
   инференсе загружается уже без `pickle`.
2. **TrOCR** — распознавание строк, не относящихся к таблицам. По умолчанию
   используется `Kansallisarkisto/cyrillic-large-handwritten`.
3. **Qwen3-VL 4B** — поиск таблиц на полной странице и распознавание содержимого
   каждого найденного табличного crop. Модель вызывается через OpenAI-compatible
   API LM Studio; модель по умолчанию — `qwen/qwen3-vl-4b`.
4. **Qwen3.6 27B** — анализ очищенного OCR JSON и извлечение параметров скважин.
   Модель по умолчанию — `qwen/qwen3.6-27b`.

RF-DETR и TrOCR загружаются в процесс Python один раз. При использовании
`--keep-alive` они остаются в памяти между документами. Для моделей LM Studio
в каждый запрос передаётся TTL; значение по умолчанию — 86400 секунд.

## Последовательность обработки

### 1. Подготовка страниц

- PDF рендерится с заданным DPI, по умолчанию 250.
- Выполняется deskew и безопасная обрезка внешних полей.
- Страница делится на горизонтальные перекрывающиеся тайлы.
- Значение `--tile-count 1` отключает деление: детектор получает целую страницу.

### 2. Детекция

RF-DETR определяет текстовые строки и регионы. Детекции из соседних тайлов
переносятся в общую систему координат страницы и дедуплицируются по IoU,
перекрытию и принадлежности строк регионам.

### 3. Поиск таблиц

По умолчанию полная подготовленная страница передаётся Qwen3-VL. Модель
возвращает список bbox таблиц в системе `normalized_1000`. Координаты:

- ограничиваются границами страницы;
- переводятся в пиксели подготовленной страницы;
- проверяются по минимальному размеру и площади;
- дедуплицируются перед распознаванием.

Если полностраничный запрос завершился ошибкой, включается fallback: кандидаты
таблиц выбираются из регионов RF-DETR по площади, форме и количеству строк.
Успешный ответ `tables=[]` считается корректным отсутствием таблиц и не включает
fallback.

### 4. OCR таблиц и строк

Каждый уникальный табличный bbox вырезается один раз и передаётся Qwen3-VL со
строгим запросом на двумерный массив `rows`. Строки, перекрытые успешно
распознанной таблицей, не отправляются в TrOCR повторно.

Остальные строки распознаются TrOCR батчами. Пустые ответы, циклические повторы,
подозрительная латиница и чрезмерно длинные строки отмечаются флагами качества.
Повторная проверка таких строк через Qwen3-VL по умолчанию отключена, поскольку
создаёт отдельный VLM-запрос на каждую строку. При необходимости её можно
включить флагом `--qwen-review-suspicious-lines`.

### 5. Сборка документа

Результаты TrOCR и Qwen сортируются по координатам и объединяются в блоки одной
страницы. Полный технический результат сохраняется в `document.json`. Он
содержит текст, таблицы, геометрию, сведения о моделях, маршрутизацию и
диагностические данные.

### 6. Очистка JSON

Для document-level LLM создаётся `cleaned_document.json`. В нём остаются только:

- `pages[].file` — имя исходного файла;
- `pages[].page` — исходный номер страницы;
- `pages[].blocks[].type` — тип блока;
- `pages[].blocks[].content` — текст блока.

Табличные строки сохраняются как текст в `content`; отдельная геометрия
и служебная структура `rows` удаляются.

BBox, размеры страниц, ID строк, confidence, пути к crop и прочая служебная
геометрия удаляются.

### 7. Извлечение параметров скважин

Очищенный документ целиком передаётся Qwen3.6 27B с системным промптом из
`well_extraction_prompt.txt`. Ответ ограничивается JSON Schema с девятью
каноническими параметрами.

После ответа выполняется локальная детерминированная проверка:

- номер страницы должен присутствовать в исходном OCR JSON;
- `evidence` должна буквально находиться на указанной странице;
- `raw_value` должна присутствовать в `evidence`;
- имена параметров и confidence должны входить в разрешённые множества;
- имя файла и страница сверяются с `pages[].file/page`;
- `row_number` перенумеровывается последовательно;
- неподтверждённые записи отбрасываются и фиксируются в `warnings`.

Итог сохраняется в `well_extraction.json`.

## Требования

- Linux;
- Python 3.10 или новее;
- NVIDIA GPU и совместимая CUDA-сборка PyTorch для GPU-инференса;
- LM Studio с включённым OpenAI-compatible API server;
- достаточно RAM/VRAM для выбранных квантов Qwen3-VL и Qwen3.6 27B.

Зависимости Python перечислены в `requirements_htr.txt`.

## Установка

Создайте виртуальное окружение:

```bash
python3 -m venv .venv_htr
source .venv_htr/bin/activate
python -m pip install --upgrade pip
```

Для NVIDIA сначала установите подходящие CUDA wheels `torch` и `torchvision`
для вашей версии драйвера, затем установите остальные зависимости:

```bash
pip install -r requirements_htr.txt
```

В LM Studio необходимо скачать и разрешить JIT-загрузку либо вручную загрузить:

- `qwen/qwen3-vl-4b`;
- `qwen/qwen3.6-27b`.

Запустите Local Server на `http://localhost:1234`. Пайплайн обращается к
OpenAI-compatible endpoint `http://localhost:1234/v1`.

## Быстрый запуск

```bash
python3 hybrid_htr_table_pipeline.py document.pdf \
  --output-dir hybrid_htr_output \
  --overwrite
```

Результаты появятся в `hybrid_htr_output/document/`, где `document` — имя
исходного файла без расширения.

Обработка выбранных страниц:

```bash
python3 hybrid_htr_table_pipeline.py document.pdf \
  --pages "1,3-5" \
  --output-dir hybrid_htr_output \
  --overwrite
```

Последовательная обработка нескольких документов без выгрузки локальных
моделей:

```bash
python3 hybrid_htr_table_pipeline.py first.pdf second.pdf \
  --output-dir hybrid_htr_output \
  --keep-alive \
  --overwrite
```

После обработки переданных файлов режим `--keep-alive` продолжит принимать пути
из stdin. Для завершения введите `exit`.

## Полезные параметры

| Параметр | Назначение |
| --- | --- |
| `--pages "1,3-5"` | Обработать только выбранные страницы |
| `--dpi 250` | Разрешение рендера PDF |
| `--tile-count 1` | Передать детектору целую страницу без тайлинга |
| `--detector-cpu` | Запустить RF-DETR на CPU |
| `--ocr-device auto|cuda|cpu` | Выбрать устройство TrOCR |
| `--ocr-batch-size N` | Настроить размер OCR-батча |
| `--qwen-table-localization off` | Отключить VLM-поиск таблиц и использовать RF-DETR эвристику |
| `--no-qwen-review-suspicious-lines` | Не отправлять сомнительные строки в VLM; это default |
| `--no-extract-well-data` | Остановиться после создания очищенного OCR JSON |
| `--lmstudio-url URL` | Изменить OpenAI-compatible endpoint LM Studio |
| `--qwen-ttl SECONDS` | Настроить время удержания моделей LM Studio |
| `--save-debug` | Сохранить дополнительные изображения и диагностические файлы |
| `--skip-detection` | Использовать ранее сохранённые detection manifests |
| `--continue-on-error` | Продолжить обработку следующих входных файлов после ошибки |

Полный список параметров:

```bash
python3 hybrid_htr_table_pipeline.py --help
```

## Веб-интерфейс

Streamlit-интерфейс принимает один PDF, TIFF, JPEG или ZIP, запускает
полный пайплайн и формирует PDF-отчёт с таблицей параметров скважины.
RF-DETR и TrOCR кэшируются в процессе Streamlit и остаются в памяти между
запросами. Одновременные задания выполняются последовательно, чтобы не переполнять
GPU.

### Установка

Активируйте окружение проекта и установите все зависимости:

```bash
cd ~/python_prjcts/qwen_ocr
source .venv_htr/bin/activate
python3 -m pip install -r requirements_htr.txt
```

Если остальные зависимости уже установлены, можно добавить только веб-интерфейс и
генератор PDF:

```bash
python3 -m pip install "streamlit>=1.40.0" "reportlab>=4.2.0"
```

Перед запуском поднимите OpenAI-compatible API в LM Studio и убедитесь, что доступны
модели `qwen/qwen3-vl-4b` и `qwen/qwen3.6-27b`. Адрес по умолчанию —
`http://localhost:1234/v1`.

### Локальный запуск

```bash
cd ~/python_prjcts/qwen_ocr
source .venv_htr/bin/activate
streamlit run streamlit_app.py
```

После запуска откройте `http://localhost:8501`. Остановка сервера — `Ctrl+C` в
терминале. Первая обработка займёт больше времени, поскольку веса RF-DETR и TrOCR
загружаются в память. Последующие задания переиспользуют эти же экземпляры моделей.

### Использование

1. Нажмите «Выберите файл» и загрузите PDF, TIFF, JPEG или ZIP.
2. Нажмите «Распознать и создать отчёт».
3. Дождитесь завершения OCR, распознавания таблиц и извлечения параметров.
4. Скачайте итоговый `PDF-отчёт`.
5. При необходимости раскройте «Дополнительные файлы» и скачайте:

   - очищенный OCR JSON;
   - JSON с извлечёнными параметрами;
   - журнал обработки.

Для ZIP создаётся один общий отчёт. Вложенные ZIP не обрабатываются; файлы
неподдерживаемых типов внутри архива пропускаются и отмечаются в журнале.

### Настройка моделей

Параметры Streamlit-сервиса задаются переменными окружения до запуска:

| Переменная | Назначение | По умолчанию |
| --- | --- | --- |
| `HTR_DETECTOR_WEIGHTS` | Путь к RF-DETR `.safetensors` | Скачивание/конвертация из Hugging Face |
| `HTR_DETECTOR_CPU` | `1`, чтобы запустить RF-DETR на CPU | `0` |
| `HTR_OCR_MODEL` | ID Hugging Face или локальная папка TrOCR | `Kansallisarkisto/cyrillic-large-handwritten` |
| `HTR_OCR_DEVICE` | `auto`, `cuda` или `cpu` | `auto` |
| `HTR_OCR_DTYPE` | `auto`, `float32`, `float16` или `bfloat16` | `auto` |
| `LMSTUDIO_URL` | OpenAI-compatible endpoint | `http://localhost:1234/v1` |
| `QWEN_VL_MODEL` | ID Qwen VL на API-сервере | `qwen/qwen3-vl-4b` |
| `QWEN_EXTRACTION_MODEL` | ID финальной текстовой модели | `qwen/qwen3.6-27b` |
| `WELL_EXTRACTION_PROMPT` | Путь к промпту извлечения | `well_extraction_prompt.txt` |

После изменения переменных перезапустите Streamlit, чтобы пересоздать кэш моделей.

### Запуск в локальной сети или офлайн

Для офлайн-развёртывания передайте RF-DETR `.safetensors` и полную папку TrOCR,
задайте их локальные пути и отключите обращения Transformers к Hugging Face:

```bash
export HTR_DETECTOR_WEIGHTS=/opt/qwen_ocr/models/rfdetr.safetensors
export HTR_OCR_MODEL=/opt/qwen_ocr/models/trocr
export LMSTUDIO_URL=http://qwen-server:1234/v1
export QWEN_VL_MODEL=qwen/qwen3-vl-4b
export QWEN_EXTRACTION_MODEL=qwen/qwen3.6-27b
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.maxUploadSize 2048
```

Параметр `--server.maxUploadSize` задаёт максимальный размер загрузки в мегабайтах. Порт
`8501` должен быть доступен клиентам. В самом приложении нет аутентификации, поэтому при
публикации в общей сети используйте внутренний reverse proxy или другой корпоративный
слой доступа.

## Веса RF-DETR в safetensors

При обычном запуске ничего делать вручную не нужно: пайплайн один
раз преобразует скачанный RF-DETR и сохранит результат в
`~/.cache/qwen_ocr/weights/`. Каждый тензор проверяется на точное совпадение.

Для ручной конвертации доверенного checkpoint:

```bash
python3 convert_rfdetr_to_safetensors.py model.pth model.safetensors
```

Готовый файл можно передать явно:

```bash
python3 hybrid_htr_table_pipeline.py document.pdf \
  --detector-weights /path/to/model.safetensors \
  --output-dir hybrid_htr_output \
  --overwrite
```

## Структура результатов

```text
hybrid_htr_output/
└── document/
    ├── document.json
    ├── cleaned_document.json
    ├── well_extraction.json
    ├── all_text.txt
    └── page_001/
        ├── detections.json
        ├── ocr_json_001.json
        ├── page_001.txt
        ├── lines/
        └── vlm_regions/
```

Если финальная текстовая модель недоступна, технический `document.json` и
`cleaned_document.json` уже остаются на диске, а `document.json` получает статус
ошибки postprocessing.

## Файлы репозитория

- `hybrid_htr_table_pipeline.py` — основной гибридный пайплайн.
- `historical_russian_htr_pipeline.py` — рендер, предобработка, RF-DETR,
  геометрические операции и базовые HTR-компоненты.
- `convert_rfdetr_to_safetensors.py` — ручная конвертация доверенного
  RF-DETR `.pth`/`.pt` в `.safetensors` с последующей точной проверкой.
- `streamlit_app.py` — веб-интерфейс загрузки и скачивания отчёта.
- `strip_ocr_for_llm.py` — очистка OCR JSON от геометрии и метаданных.
- `json_to_well_pdf.py` — преобразование `well_extraction.json` в PDF-отчёт.
- `well_extraction_prompt.txt` — системный и пользовательский промпты финального
  извлечения параметров.
- `requirements_htr.txt` — зависимости Python.
