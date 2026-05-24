import random
from pathlib import Path

import functools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import Wav2Vec2FeatureExtractor

from src.baseline.dataset import QUADRANT_TO_IDX, IDX_TO_LABEL, NUM_CLASSES
from src.mert.model_mert import MERT_SR, get_processor

DURATION = 15
AUDIO_EXTS = {".mp3"}


# Dataset

class MERTMusicDataset(Dataset):
    """
    Загружает аудиофайлы, ресемплирует до 24 КГц и нормализует
    с помощью Wav2Vec2FeatureExtractor из MERT.

    Параметры:
        data_dir   : путь к папке data/ (содержит Q1–Q4)
        processor  : Wav2Vec2FeatureExtractor от MERT
        file_list  : список (path, label_idx); если None — сканирует data_dir
        cache      : кешировать тензоры в RAM
    """

    def __init__(self, data_dir: str | Path,
                 processor: Wav2Vec2FeatureExtractor,
                 file_list: list[tuple[str, int]] | None = None,
                 cache: bool = False):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.cache = cache
        self._store: dict[int, torch.Tensor] = {}

        self.samples = file_list if file_list is not None else self._scan()

    def _scan(self) -> list[tuple[str, int]]:
        samples = []
        for quad, idx in QUADRANT_TO_IDX.items():
            folder = self.data_dir / quad
            if not folder.exists():
                print(f"{folder} не найдена — пропускаем")
                continue
            for fpath in sorted(folder.iterdir()):
                if fpath.suffix.lower() in AUDIO_EXTS:
                    samples.append((str(fpath), idx))
        if not samples:
            raise RuntimeError(
                f"Аудиофайлы не найдены в {self.data_dir}. "
                "Убедитесь, что папки Q1–Q4 содержат .mp3 файлы."
            )
        return samples

    def _load_waveform(self, path: str) -> np.ndarray:
        """Загружает аудио и ресемплирует до MERT_SR (24 КГц)."""
        import torchaudio
        import torchaudio.transforms as T

        waveform, sr = torchaudio.load(path)

        # Моно
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Ресемплинг до 24 КГц
        if sr != MERT_SR:
            waveform = T.Resample(orig_freq=sr, new_freq=MERT_SR)(waveform)

        # Обрезка/паддинг до нужной длины
        target_len = int(MERT_SR * DURATION)
        y = waveform.squeeze(0).numpy()
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]
        return y

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.cache and idx in self._store:
            return self._store[idx], self.samples[idx][1]

        path, label = self.samples[idx]
        try:
            y = self._load_waveform(path)
        except Exception as e:
            print(f"ERROR {path}: {e}")
            y = np.zeros(int(MERT_SR * DURATION), dtype=np.float32)

        # Нормализация через процессор MERT
        # return_tensors="pt" дает (1, time) - убираем батчевую размерность
        inputs = self.processor(
            y,
            sampling_rate=MERT_SR,
            return_tensors="pt",
            padding=False,
        )
        waveform = inputs["input_values"].squeeze(0)  # (time,)

        if self.cache:
            self._store[idx] = waveform

        return waveform, label


def mert_collate_fn(batch: list[tuple[torch.Tensor, int]],
                    processor=None):
    """
    Паддит waveform'ы до одной длины внутри батча и строит attention_mask.
    Нужен потому что MERT принимает батч через процессор.
    """
    waveforms = [item[0].numpy() for item in batch]
    labels = [item[1] for item in batch]

    encoded = processor(
        waveforms,
        sampling_rate=MERT_SR,
        return_tensors="pt",
        padding=True,
    )

    return (
        encoded["input_values"],  # (B, max_len)
        encoded.get("attention_mask"),  # (B, max_len) или None
        torch.tensor(labels, dtype=torch.long),
    )


def split_dataset(dataset: MERTMusicDataset,
                  train_ratio: float = 0.70,
                  val_ratio: float = 0.20,
                  seed: int = 42) -> tuple[Subset, Subset, Subset]:
    """Стратифицированный сплит — такой же, как в dataset.py."""
    rng = random.Random(seed)
    class_indices: dict[int, list[int]] = {i: [] for i in range(NUM_CLASSES)}
    for i, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for indices in class_indices.values():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx += indices[:n_train]
        val_idx += indices[n_train:n_train + n_val]
        test_idx += indices[n_train + n_val:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset,
                                                                        test_idx)


def build_mert_dataloaders(data_dir: str | Path,
                           batch_size: int = 16,
                           num_workers: int = 2,
                           cache: bool = False,
                           seed: int = 42
                           ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Пример:
        train_loader, val_loader, test_loader = build_mert_dataloaders("data")
    """
    processor = get_processor()
    dataset = MERTMusicDataset(data_dir, processor, cache=cache)

    n = len(dataset)
    print(f"Всего файлов: {n}")
    counts = {}
    for _, lbl in dataset.samples:
        counts[lbl] = counts.get(lbl, 0) + 1
    for idx, cnt in sorted(counts.items()):
        print(f"  {IDX_TO_LABEL[idx]:8s} (Q{idx + 1}): {cnt} файлов")

    train_set, val_set, test_set = split_dataset(dataset, seed=seed)
    print(f"\nСплит: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    collate = functools.partial(mert_collate_fn, processor=processor)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             collate_fn=collate, pin_memory=True)

    return train_loader, val_loader, test_loader
