import random
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# Маппинг

QUADRANT_TO_IDX = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}

IDX_TO_LABEL = {
    0: "Happy",
    1: "Angry",
    2: "Sad",
    3: "Relaxed",
}

NUM_CLASSES = 4

# Параметры для спектрограммы

SAMPLE_RATE = 22050  # Гц
DURATION = 30  # секунд
N_FFT = 2048  # размер окна STFT
HOP_LENGTH = 512  # шаг окна
SPEC_HEIGHT = 128  # высота спектрограммы после ресайза
SPEC_WIDTH = 128  # ширина спектрограммы после ресайза


def load_audio(path: str, sr: int = SAMPLE_RATE,
               duration: float = DURATION) -> np.ndarray:
    """Загружает аудиофайл, обрезает/дополняет до нужной длины."""
    y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        # дополнение нулями (pad)
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]
    return y


def audio_to_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE,
                         n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
                         height: int = SPEC_HEIGHT,
                         width: int = SPEC_WIDTH) -> np.ndarray:
    """
    Преобразует аудио в лог-спектрограмму (dB) через STFT,
    затем масштабирует до (height x width) и нормализует.
    Возвращает массив формы (1, height, width) — один канал.
    """
    # STFT -> амплитудный спектр
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Перевод в дБ
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Ресайз до фиксированного размера (height x width)
    from PIL import Image
    img = Image.fromarray(S_db).resize((width, height), Image.BILINEAR)
    S_resized = np.array(img)

    # Нормализация в диапазон [0, 1]
    S_min, S_max = S_resized.min(), S_resized.max()
    if S_max - S_min > 1e-6:
        S_norm = (S_resized - S_min) / (S_max - S_min)
    else:
        S_norm = np.zeros_like(S_resized, dtype=np.float32)

    # Добавляем канальную размерность: (1, H, W)
    return S_norm.astype(np.float32)[np.newaxis, ...]


# Dataset

class MusicEmotionDataset(Dataset):
    """
    Загружает аудиофайлы из папок Q1–Q4 и конвертирует их в спектрограммы.

    Параметры:
        data_dir    : путь к папке data/ (содержит Q1, Q2, Q3, Q4)
        file_list   : список (path, label_idx); если None — сканирует data_dir
        transform   : опциональный PyTorch transform для тензора спектрограммы
        cache       : если True, кешируем спектрограммы в памяти (ускоряет
                      повторные эпохи, но требует RAM)
    """

    AUDIO_EXTS = {".mp3"}

    def __init__(self, data_dir: str | Path,
                 file_list: list[tuple[str, int]] | None = None,
                 transform=None,
                 cache: bool = False):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.cache = cache
        self._cache_store: dict[int, torch.Tensor] = {}

        if file_list is not None:
            self.samples = file_list
        else:
            self.samples = self._scan()

    def _scan(self) -> list[tuple[str, int]]:
        samples = []
        for quad, idx in QUADRANT_TO_IDX.items():
            folder = self.data_dir / quad
            if not folder.exists():
                print(f"Папка {folder} не найдена — пропускаем")
                continue
            for fpath in sorted(folder.iterdir()):
                if fpath.suffix.lower() in self.AUDIO_EXTS:
                    samples.append((str(fpath), idx))
        if not samples:
            raise RuntimeError(
                f"Аудиофайлы не найдены в {self.data_dir}. "
                "Убедитесь, что папки Q1–Q4 содержат .mp3 файлы."
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.cache and idx in self._cache_store:
            spec_tensor = self._cache_store[idx]
            _, label = self.samples[idx]
            return spec_tensor, label

        path, label = self.samples[idx]
        try:
            y = load_audio(path)
            spec = audio_to_spectrogram(y)
        except Exception as e:
            print(f"Не удалось загрузить {path}: {e}")
            spec = np.zeros((1, SPEC_HEIGHT, SPEC_WIDTH), dtype=np.float32)

        spec_tensor = torch.from_numpy(spec)

        if self.transform is not None:
            spec_tensor = self.transform(spec_tensor)

        if self.cache:
            self._cache_store[idx] = spec_tensor

        return spec_tensor, label


# Разбивка на train, val, test

def split_dataset(dataset: MusicEmotionDataset,
                  train_ratio: float = 0.70,
                  val_ratio: float = 0.20,
                  seed: int = 42
                  ) -> tuple[Subset, Subset, Subset]:
    rng = random.Random(seed)

    # группируем индексы по классу
    class_indices: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
    for i, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(i)

    train_idx, val_idx, test_idx = [], [], []

    for cls, indices in class_indices.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx += indices[:n_train]
        val_idx += indices[n_train:n_train + n_val]
        test_idx += indices[n_train + n_val:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset,
                                                                        test_idx)


def build_dataloaders(data_dir: str | Path,
                      batch_size: int = 32,
                      num_workers: int = 2,
                      cache: bool = False,
                      seed: int = 42
                      ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Главная точка входа: сканирует data_dir, делает сплит,
    возвращает три DataLoader'а.

    Пример:
        train_loader, val_loader, test_loader = build_dataloaders("data")
    """
    dataset = MusicEmotionDataset(data_dir, cache=cache)

    n = len(dataset)
    print(f"Всего файлов: {n}")
    counts = {}
    for _, lbl in dataset.samples:
        counts[lbl] = counts.get(lbl, 0) + 1
    for idx, cnt in sorted(counts.items()):
        print(f"  {IDX_TO_LABEL[idx]:8s} (Q{idx + 1}): {cnt} файлов")

    train_set, val_set, test_set = split_dataset(dataset, seed=seed)
    print(f"\nСплит: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, val_loader, test_loader
