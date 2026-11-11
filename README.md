# Sprint 4 Project - Мультимодальная нейросеть для определения калорийности блюда по его описанию и фотографии
Проект реализуется в рамках 4-ого спринта обучения на курсе "Инженер по глубокому обучению нейросетей" (Яндекс Практикум)

Создаваемая нейросеть, должна определять калорийность блюд.
Модель обучается на датасете с фотографиями тарелок с разными блюдами и текстовым описанием их ингредиентного состава.

## Датасет
[Датасет (1.3 ГБ)](https://disk.yandex.ru/d/kz9g5msVqtahiw), содержит фотографии блюд, описание ингредиентов, а также общую массу порции и калорийность
- `data/ingredients.csv` 
    - `id`, `ingr` - названия ингредиентов
- `data/dish.csv` 
    - `dish_id`
    - `total_calories` - общее число калорий - целевая переменная
    - `ingredients` - список всех `Id` ингредиентов в формате `ingr_0000000122;ingr_0000000026;..`
    - `split` - данные уже разбиты на `train/test`
    - `total_mass`, `total_calories`
- `data/images` - фотографии блюд, директория соответствует `dish_id

## Настройка виртуального окружения

В качестве менеджера окружения используется **uv**, среда **Python 3.10**.

### Создание virtual environment (Python 3.10)

```bash
uv venv --python 3.10
```

Будет создана директория `.venv`

### Активация venv

```bash
source .venv/bin/activate
```
Большинство `uv` команд (`uv run ...`) работают без ручной активации.

### Установка зависимостей

Зависимости перечислены в `sprint4_requirements.txt` (https://code.s3.yandex.net/deep-learning/sprint4_requirements(1).txt)

При работе на ВМ с GPU:
```bash
uv add -r sprint4_requirements.txt
```

При работе без ВМ с GPU (на локальном компьютере) отключаем лишние зависимости:
```bash
uv add -r sprint4_requirements_no_gpu.txt
```

```
# nvidia-cublas-cu12==12.6.4.1
# nvidia-cuda-cupti-cu12==12.6.80
# nvidia-cuda-nvrtc-cu12==12.6.77
# nvidia-cuda-runtime-cu12==12.6.77
# nvidia-cudnn-cu12==9.5.1.17
# nvidia-cufft-cu12==11.3.0.4
# nvidia-cufile-cu12==1.11.1.6
# nvidia-curand-cu12==10.3.7.77
# nvidia-cusolver-cu12==11.7.1.2
# nvidia-cusparse-cu12==12.5.4.2
# nvidia-cusparselt-cu12==0.6.3
# nvidia-nccl-cu12==2.26.2
# nvidia-nvjitlink-cu12==12.6.85
# nvidia-nvtx-cu12==12.6.77
...
#triton==3.3.1
```
### Общие команды

```bash
# Run a specific Python script
uv run python your_script.py

# Re-sync environment to match pyproject.toml / uv.lock
uv sync

# Remove the environment
rm -rf .venv      # Linux/macOS
rmdir /s .venv    # Windows cmd
```

## Настройка переменных окружения

Подговьте `.env` файл:

```env
TODO: ...
```
## Структура директорий проекта

- `data` - данные датасета
- `scripts` - код обучения и работы с данными
- `notebook.ipynb` - ноутбук с EDA и запусками обучения
- `README.md` - описание проекта и результатов