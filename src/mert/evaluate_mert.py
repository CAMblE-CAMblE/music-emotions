import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mert.dataset_mert_torchaudio import build_mert_dataloaders
from src.mert.model_mert import get_mert_model
from src.baseline.dataset import IDX_TO_LABEL, NUM_CLASSES


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for input_values, attention_mask, labels in tqdm(loader, desc="Inference"):
        input_values = input_values.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        preds = model(input_values, attention_mask).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def save_confusion_matrix(y_true, y_pred, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
            axes,
            [cm, cm.astype(float) / cm.sum(axis=1, keepdims=True)],
            ["d", ".2%"],
            ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt,
                    xticklabels=label_names, yticklabels=label_names,
                    cmap="Blues", ax=ax, linewidths=0.5, linecolor="white")
        ax.set_xlabel("Предсказанный класс");
        ax.set_ylabel("Истинный класс")
        ax.set_title(title)

    plt.tight_layout()
    out = plots_dir / "mert_confusion_matrix.png"
    plt.savefig(out, dpi=150);
    plt.close()
    print(f"\nМатрица ошибок сохранена: {out}")


def evaluate(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Восстанавливаем модель
    ckpt = torch.load(args.checkpoint, map_location=device)
    layer_idx = ckpt.get("layer_idx", 5)
    model = get_mert_model(layer_idx=layer_idx, device=device)
    model.head.load_state_dict(ckpt["head_state"])
    print(f"\nЗагружен чекпоинт: {args.checkpoint}  "
          f"(epoch {ckpt.get('epoch', '?')}, layer={layer_idx})")

    _, _, test_loader = build_mert_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.workers, seed=args.seed,
    )
    print(f"Тестовых примеров: {len(test_loader.dataset)}")

    y_pred, y_true = get_predictions(model, test_loader, device)

    label_names = [IDX_TO_LABEL[i] for i in range(NUM_CLASSES)]
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'=' * 55}")
    print(f"  Accuracy : {acc:.4f}  ({acc:.2%})")
    print(f"{'=' * 55}")
    print("\nКлассификационный отчет:")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    save_confusion_matrix(y_true, y_pred, Path(args.output_dir) / "plots")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", type=str)
    p.add_argument("--checkpoint", default="outputs/checkpoints/best_mert_model.pt",
                   type=str)
    p.add_argument("--output_dir", default="outputs", type=str)
    p.add_argument("--batch_size", default=4, type=int)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
