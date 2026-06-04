from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from src.baseline.dataset import IDX_TO_LABEL, NUM_CLASSES, QUADRANT_TO_IDX
except ImportError:  # Allows direct local experiments if src is not on sys.path.
    QUADRANT_TO_IDX = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    IDX_TO_LABEL = {0: "Happy", 1: "Angry", 2: "Sad", 3: "Relaxed"}
    NUM_CLASSES = 4

from src.generation.audio_utils import AudioConfig, load_audio, wav_to_logmel

AUDIO_EXTS = {".mp3"}
DEFAULT_DATA_DIR = r"E:/ITMO/2 семестр/ГлубокоеОбучение/Dataset"
REQUIRED_QUADRANTS = ("Q1", "Q2", "Q3", "Q4")


def validate_generation_data_dir(data_dir: str | Path) -> Path:
    """Validate external PANDA/TAFFC dataset layout before training/evaluation."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise RuntimeError(
            f"Dataset directory does not exist: {data_path}\n"
            f"Pass a valid path with --data_dir or place the dataset at the default path:\n"
            f"  {DEFAULT_DATA_DIR}"
        )
    if not data_path.is_dir():
        raise RuntimeError(f"Dataset path is not a directory: {data_path}")

    missing = [quad for quad in REQUIRED_QUADRANTS if not (data_path / quad).is_dir()]
    if missing:
        expected = "\n".join(f"  {data_path / quad}" for quad in REQUIRED_QUADRANTS)
        raise RuntimeError(
            f"Dataset directory has invalid structure: {data_path}\n"
            f"Missing required folders: {', '.join(missing)}\n"
            f"Expected folders:\n{expected}"
        )

    empty = []
    for quad in REQUIRED_QUADRANTS:
        if not any((data_path / quad).glob("*.mp3")):
            empty.append(quad)
    if empty:
        raise RuntimeError(
            f"Dataset folders contain no .mp3 files: {', '.join(empty)}\n"
            f"Expected .mp3 files under {data_path}/Q1..Q4"
        )

    return data_path


class GenerationMelDataset(Dataset):
    """Log-mel dataset for conditional audio generation.

    Train samples use random 5-second crops. Validation/test samples use a
    deterministic center crop or zero padding. For train, files can be repeated
    via segments_per_file to expose more random crops per epoch.
    """

    def __init__(
        self,
        data_dir: str | Path,
        file_list: list[tuple[str, int]] | None = None,
        cfg: AudioConfig | None = None,
        split: str = "train",
        segments_per_file: int = 1,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.cfg = cfg or AudioConfig()
        self.split = split
        self.segments_per_file = max(1, int(segments_per_file))
        self.seed = seed
        self.samples = file_list if file_list is not None else self._scan()

        if split == "train" and self.segments_per_file > 1:
            self.index = [i for i in range(len(self.samples)) for _ in range(self.segments_per_file)]
        else:
            self.index = list(range(len(self.samples)))

    def _scan(self) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for quad, idx in QUADRANT_TO_IDX.items():
            folder = self.data_dir / quad
            if not folder.exists():
                print(f"{folder} not found, skipping")
                continue
            for fpath in sorted(folder.iterdir()):
                if fpath.suffix.lower() in AUDIO_EXTS:
                    samples.append((str(fpath), idx))
        if not samples:
            raise RuntimeError(
                f"No .mp3 files found in {self.data_dir}. Expected data/Q1..Q4 folders."
            )
        return samples

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample_idx = self.index[idx]
        path, label = self.samples[sample_idx]
        crop_mode = "random" if self.split == "train" else "center"
        rng = np.random.default_rng(self.seed + idx) if self.split != "train" else None

        try:
            wav = load_audio(path, self.cfg, crop_mode=crop_mode, rng=rng)
            mel = wav_to_logmel(wav, self.cfg)
        except Exception as exc:
            print(f"Failed to load {path}: {exc}")
            mel = np.zeros((self.cfg.n_mels, self.cfg.target_frames), dtype=np.float32)

        mel_tensor = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, frames)
        return mel_tensor, int(label)


def split_file_list(
    samples: list[tuple[str, int]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[str, int]]] = {i: [] for i in range(NUM_CLASSES)}
    for item in samples:
        by_class[item[1]].append(item)

    train, val, test = [], [], []
    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)
        n = len(cls_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(cls_samples[:n_train])
        val.extend(cls_samples[n_train:n_train + n_val])
        test.extend(cls_samples[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _print_counts(name: str, samples: list[tuple[str, int]]) -> None:
    counts = {i: 0 for i in range(NUM_CLASSES)}
    for _, label in samples:
        counts[label] += 1
    readable = ", ".join(f"{IDX_TO_LABEL[i]}={counts[i]}" for i in range(NUM_CLASSES))
    print(f"{name}: {len(samples)} files ({readable})")


def build_generation_dataloaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    batch_size: int = 16,
    workers: int = 2,
    seed: int = 42,
    cfg: AudioConfig | None = None,
    train_segments_per_file: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    cfg = cfg or AudioConfig()
    data_dir = validate_generation_data_dir(data_dir)
    base = GenerationMelDataset(data_dir, cfg=cfg, split="scan", seed=seed)
    train_files, val_files, test_files = split_file_list(base.samples, seed=seed)

    _print_counts("train", train_files)
    _print_counts("val", val_files)
    _print_counts("test", test_files)

    train_ds = GenerationMelDataset(
        data_dir,
        file_list=train_files,
        cfg=cfg,
        split="train",
        segments_per_file=train_segments_per_file,
        seed=seed,
    )
    val_ds = GenerationMelDataset(data_dir, file_list=val_files, cfg=cfg, split="val", seed=seed)
    test_ds = GenerationMelDataset(data_dir, file_list=test_files, cfg=cfg, split="test", seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
