import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baseline.dataset import build_dataloaders, IDX_TO_LABEL, NUM_CLASSES
from src.baseline.model   import get_model


# Сбор предсказаний

@torch.no_grad()
def get_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Прогоняет весь loader, возвращает (all_preds, all_labels)."""
    model.eval()
    all_preds, all_labels = [], []

    for specs, labels in tqdm(loader, desc="Inference"):
        specs = specs.to(device, non_blocking=True)
        logits = model(specs)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


# Метрики

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 55}")
    print(f"  Accuracy : {acc:.4f}  ({acc:.2%})")
    print(f"{'=' * 55}")

    print("\nКлассификационный отчёт:")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))


# Confusion matrix

def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

    cm = confusion_matrix(y_true, y_pred)

    # Нормализованная (в %) и абсолютная версии рядом
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title in zip(
            axes,
            [cm, cm.astype(float) / cm.sum(axis=1, keepdims=True)],
            ["d", ".2%"],
            ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(
            data,
            annot=True, fmt=fmt,
            xticklabels=label_names, yticklabels=label_names,
            cmap="Blues", ax=ax,
            linewidths=0.5, linecolor="white",
        )
        ax.set_xlabel("Предсказанный класс", fontsize=11)
        ax.set_ylabel("Истинный класс", fontsize=11)
        ax.set_title(title, fontsize=12, pad=10)

    plt.tight_layout()
    out = plots_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nМатрица ошибок сохранена: {out}")


# Per-class метрики (таблица)

def print_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    print("\nМетрики по классам:")
    print(f"  {'Класс':10s}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 42)
    for i, name in enumerate(label_names):
        print(f"  {name:10s}  {precision[i]:>10.4f}  {recall[i]:>8.4f}  {f1[i]:>8.4f}")
    print("  " + "-" * 42)

    for avg in ("macro", "weighted"):
        p = precision_score(y_true, y_pred, average=avg, zero_division=0)
        r = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f = f1_score(y_true, y_pred, average=avg, zero_division=0)
        print(f"  {avg:10s}  {p:>10.4f}  {r:>8.4f}  {f:>8.4f}")


def evaluate(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем модель
    model = get_model(device=device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    epoch = ckpt.get("epoch", "?")
    print(f"\nЗагружен чекпоинт: {args.checkpoint}  (epoch {epoch})")

    # Данные — только test_loader нужен
    _, _, test_loader = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        seed=args.seed,
    )
    print(f"Тестовых примеров: {len(test_loader.dataset)}")

    y_pred, y_true = get_predictions(model, test_loader, device)

    print_metrics(y_true, y_pred)
    print_per_class_metrics(y_true, y_pred)

    plots_dir = Path(args.output_dir) / "plots"
    save_confusion_matrix(y_true, y_pred, plots_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Оценка EmotionCNN")
    p.add_argument("--data_dir", default="data", type=str)
    p.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt",
                   type=str)
    p.add_argument("--output_dir", default="outputs", type=str)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
