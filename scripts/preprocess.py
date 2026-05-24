import argparse
import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    QUADRANT_TO_IDX, IDX_TO_LABEL,
    SAMPLE_RATE, N_FFT, HOP_LENGTH,
    load_audio,
)

AUDIO_EXTS = {".mp3"}


# Сбор файлов

def collect_files(data_dir: Path, n_samples: int, seed: int) -> dict[str, list[Path]]:
    """Возвращает словарь {quadrant: [path, ...]}."""
    rng = random.Random(seed)
    result = {}
    for quad in QUADRANT_TO_IDX:
        folder = data_dir / quad
        if not folder.exists():
            print(f"{folder} не найдена — пропускаем")
            continue
        files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in AUDIO_EXTS]
        if not files:
            print(f"{folder} пуста — пропускаем")
            continue
        result[quad] = rng.sample(files, min(n_samples, len(files)))
    return result


# Визуализация одного файла

def plot_stft_spectrogram(ax, y: np.ndarray, sr: int, title: str) -> None:
    """Рисует лог-спектрограмму на переданном axes."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="log",
        ax=ax, cmap="magma",
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Частота (Гц)")


# Сводный грид: по одному примеру на класс

def save_overview_grid(files_by_quad: dict[str, list[Path]],
                       out_path: Path) -> None:
    """
    4 квадранта × 1 пример -> сетка 2×2 на одном рисунке.
    """
    quads = list(files_by_quad.keys())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, quad in zip(axes, quads):
        path = files_by_quad[quad][0]
        label = IDX_TO_LABEL[QUADRANT_TO_IDX[quad]]
        try:
            y = load_audio(str(path))
            plot_stft_spectrogram(ax, y, SAMPLE_RATE,
                                  title=f"{quad} — {label}\n{path.name}")
        except Exception as e:
            ax.set_title(f"{quad}: ошибка загрузки\n{e}")
            ax.axis("off")

    plt.suptitle("Спектрограммы STFT по эмоциональным классам", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Обзорный грид сохранён: {out_path}")


# N примеров на класс -> отдельные PNG

def save_per_class_plots(files_by_quad: dict[str, list[Path]],
                         out_dir: Path) -> None:
    """
    Для каждого квадранта сохраняет один PNG с N спектрограммами.
    """
    for quad, paths in files_by_quad.items():
        label = IDX_TO_LABEL[QUADRANT_TO_IDX[quad]]
        n = len(paths)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, path in zip(axes, paths):
            try:
                y = load_audio(str(path))
                plot_stft_spectrogram(ax, y, SAMPLE_RATE, title=path.name)
            except Exception as e:
                ax.set_title(f"Ошибка: {e}")
                ax.axis("off")

        fig.suptitle(f"{quad} — {label}", fontsize=12)
        plt.tight_layout()
        out = out_dir / f"{quad}_{label.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {quad} ({label}): {n} спектрограмм → {out}")


# Статистика датасета

def print_dataset_stats(data_dir: Path) -> None:
    print("\nСтатистика датасета:")
    print(f"  {'Квадрант':8s}  {'Эмоция':10s}  {'Файлов':>7}")
    print("  " + "-" * 32)
    total = 0
    for quad, idx in QUADRANT_TO_IDX.items():
        folder = data_dir / quad
        if not folder.exists():
            continue
        n = sum(1 for f in folder.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        total += n
        print(f"  {quad:8s}  {IDX_TO_LABEL[idx]:10s}  {n:>7}")
    print("  " + "-" * 32)
    print(f"  {'Итого':8s}  {'':10s}  {total:>7}")


def main(args) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_dataset_stats(data_dir)

    files_by_quad = collect_files(data_dir, n_samples=args.n_samples, seed=args.seed)
    if not files_by_quad:
        print("Не найдено ни одного аудиофайла. Проверьте --data_dir.")
        return

    print(f"\nВизуализируем по {args.n_samples} файл(а) на класс...")

    save_overview_grid(files_by_quad, out_dir / "overview_grid.png")
    save_per_class_plots(files_by_quad, out_dir)

    print(f"\nВсе изображения сохранены в {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Визуализация спектрограмм")
    p.add_argument("--data_dir", default="data", type=str)
    p.add_argument("--output_dir", default="outputs/spectrograms", type=str)
    p.add_argument("--n_samples", default=3, type=int)
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
