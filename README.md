# Music Emotion Recognition

Исследование и разработка методов глубокого обучения для анализа эмоционального содержания музыкальных фрагментов, их классификации и генерации новых музыкальных отрывков с заданными эмоциями.

## Датасет

**PANDA / TAFFC** (Panda et al., IEEE TAFFC 2018) — 900 клипов по примерно 30 секунд, размечены по модели Рассела.

| Папка | Индекс | Эмоция |
|-------|--------|--------|
| Q1 | 0 | Happy — радость (+Valence, +Arousal) |
| Q2 | 1 | Angry — гнев/напряжение (−Valence, +Arousal) |
| Q3 | 2 | Sad — грусть (−Valence, −Arousal) |
| Q4 | 3 | Relaxed — спокойствие (+Valence, −Arousal) |

> Данные не хранятся в репозитории. Датасет доступен по адресу: https://mir.dei.uc.pt/downloads.html

## Структура проекта

```
music-emotions/
├── data/                               # не в репозитории
│   ├── Q1/                             # Happy   (225 .mp3)
│   ├── Q2/                             # Angry   (225 .mp3)
│   ├── Q3/                             # Sad     (225 .mp3)
│   └── Q4/                             # Relaxed (225 .mp3)
│
├── src/
│   ├── baseline/
│   │   ├── dataset.py                  # Dataset, STFT-спектрограммы, сплит 70/20/10
│   │   ├── model.py                    # CNN архитектура
│   │   ├── train.py                    # Обучение baseline
│   │   └── evaluate.py                 # Метрики + confusion matrix
│   └── mert/
│       ├── dataset_mert_librosa.py     # Dataset для MERT с librosa
│       ├── dataset_mert_torchaudio.py  # Dataset для MERT с torchaudio
│       ├── model_mert.py               # MERT-v1-95M + классификатор
│       ├── train_mert.py               # Обучение MERT
│       └── evaluate_mert.py            # Метрики + confusion matrix
│
├── scripts/
│   ├── preprocess.py                   # Визуализация STFT-спектрограмм по классам
│   └── predict.py                      # Инференс на одном аудиофайле
│
├── outputs/
│   ├── baseline_cnn/                   # Baseline CNN для классификации
│   │   ├── checkpoints/
│   │   │   ├── best_model.pt
│   │   │   └── last_model.pt
│   │   └── plots/
│   │       ├── confusion_matrix.png
│   │       └── training_curves.png
│   │
│   ├── mert/                           # Результаты MERT
│   │   ├── checkpoints/
│   │   │   └── best_mert_model.pt
│   │   └── plots/
│   │       ├── mert_confusion_matrix.png
│   │       └── mert_training_curves.png
│   │
│   ├── mert_best/                      # Результаты MERT с лучшенной конфигурацией
│   │   ├── checkpoints/
│   │   │   └── best_mert_model.pt
│   │   └── plots/
│   │       ├── mert_confusion_matrix.png
│   │       └── mert_training_curves.png
│   │
│   └── spectrograms/                   # Визуализации спектрограмм по классам
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Классификация эмоций.

### Baseline CNN (`outputs/baseline_cnn/`)

Реализован согласно заданию — сверточная нейронная сеть на STFT-спектрограммах, обученная с нуля. Обучалась 30 эпох на GTX 1660 Ti. Итоговая val accuracy примерно 62% при сильной нестабильности: val loss скачет, модель переобучается.

### MERT mert (`outputs/mert/`)

Первая попытка Transfer Learning на базе [m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) — музыкальной модели, предобученной на 160 000 часах аудио. MERT заморожен, обучается только легкий классификатор (~198K параметров). Запускалась на GTX 1660 Ti (6 ГБ VRAM), параметры: `layer_idx=5, epochs=20, dropout=0.5`. Val accuracy 71.67% — заметно лучше baseline, обучение стабильнее.

### MERT mert_best (`outputs/mert_best/`) <- рекомендуется

После анализа результатов `mert` была проведена серия экспериментов на RTX 3080 с разными конфигурациями (`layer_idx` 5/7/8, `epochs` 20/40, `lr` 1e-3/3e-4). Лучший результат показала конфигурация `layer_idx=7, epochs=40, lr=3e-4` — val accuracy **73.89%**. Именно эта модель рекомендуется для дальнейшего использования.

## Сравнение результатов

| Модель | Параметры | Val Acc | Happy | Angry | Sad | Relaxed |
|--------|-----------|---------|-------|-------|-----|---------|
| Baseline CNN | epochs=30 | ~62% | 73.91% | 65.22% | 43.48% | 73.91% |
| MERT mert_home | layer=5, epochs=20 | 71.67% | 91.30% | 65.22% | 56.52% | 60.87% |
| **MERT mert_best** | **layer=7, epochs=40, lr=3e-4** | **73.89%** | 86.96% | 73.91% | 52.17% | 73.91% |

Основная сложность во всех моделях — различение Sad и Relaxed: оба класса акустически похожи (медленный темп, низкое возбуждение).


## Обнаружение эмоциональных переходов

- [ ] Нарезка треков на окна по 5 секунд
- [ ] Разметка окон через классификатор из задания 2
- [ ] LSTM для анализа временных последовательностей эмоций
- [ ] Метрики: ROC-AUC, FPR, TPR, ROC-кривая

## Генерация музыкальных фрагментов

- [ ] VAE или GAN для генерации спектрограмм с заданной эмоцией
- [ ] Обратное STFT для синтеза аудио из спектрограммы
- [ ] Метрики: Cosine Similarity, MOS

## Анализ взаимодействия музыки и эмоций

- [ ] Извлечение признаков: BPM, громкость, частотный спектр
- [ ] Регрессионная модель (Random Forest) для оценки влияния признаков
- [ ] SHAP для интерпретации важности признаков
- [ ] Визуализация зависимостей

## Оценка качества и эмоциональной точности

- [ ] Сравнение классификации (CNN), обнаружения переходов (LSTM), генерации (VAE)
- [ ] Анализ ошибок: случаи неверной классификации (Sad vs Relaxed)
- [ ] Оценка ограничений генерации сложных структур

## Установка

```bash
# 1. Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# 2. PyTorch с CUDA (обязательно перед остальным)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Остальные зависимости
pip install -r requirements.txt
```

## Использование модели классификации

### Оценка на тестовой выборке

```bash
python src/mert/evaluate_mert.py \
    --data_dir data \
    --checkpoint outputs/mert_best/checkpoints/best_mert_model.pt \
    --batch_size 8
```

При `CUDA out of memory` уменьши `--batch_size` до 4.

### Предсказание эмоции одного файла

```bash
python scripts/predict.py \
    --audio data/Q1/some_file.mp3 \
    --checkpoint outputs/mert_best/checkpoints/best_mert_model.pt
```

### Загрузка модели в коде

```python
import torch
from src.mert.model_mert import get_mert_model
from src.mert.dataset_mert_torchaudio import build_mert_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка модели
model = get_mert_model(layer_idx=7, device=device)
ckpt  = torch.load(
    "outputs/mert_best/checkpoints/best_mert_model.pt",
    map_location=device
)
model.head.load_state_dict(ckpt["head_state"])
model.eval()

# DataLoader
train_loader, val_loader, test_loader = build_mert_dataloaders(
    "data", batch_size=8
)

# Инференс
with torch.no_grad():
    for input_values, attention_mask, labels in test_loader:
        input_values = input_values.to(device)
        logits = model(input_values, attention_mask.to(device))
        probs  = logits.softmax(dim=-1)   # вероятности по 4 классам
        preds  = logits.argmax(dim=-1)    # предсказанный класс
```

### Маппинг классов

```python
IDX_TO_LABEL = {0: "Happy", 1: "Angry", 2: "Sad", 3: "Relaxed"}
```

## Повторное обучение

### Baseline CNN

```bash
python src/baseline/train.py --data_dir data --epochs 30 --batch_size 32
python src/baseline/evaluate.py --data_dir data
```

### MERT (лучшая конфигурация)

```bash
python src/mert/train_mert.py \
    --data_dir data \
    --epochs 40 \
    --dropout 0.2 \
    --batch_size 16 \
    --workers 0 \
    --layer_idx 7 \
    --lr 3e-4

python src/mert/evaluate_mert.py --data_dir data --batch_size 8
```