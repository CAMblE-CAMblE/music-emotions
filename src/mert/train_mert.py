import argparse
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mert.dataset_mert_torchaudio import build_mert_dataloaders
from src.mert.model_mert import get_mert_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Одна эпоха обучения

def train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for input_values, attention_mask, labels in tqdm(loader, desc="  train",
                                                     leave=False):
        input_values = input_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(input_values, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_values.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += input_values.size(0)

    return total_loss / total, correct / total


# Валидация

@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for input_values, attention_mask, labels in tqdm(loader, desc="  val  ",
                                                     leave=False):
        input_values = input_values.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        logits = model(input_values, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * input_values.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += input_values.size(0)

    return total_loss / total, correct / total


# Графики

def save_plots(history: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(
            axes,
            [("train_loss", "val_loss"), ("train_acc", "val_acc")],
            ["Loss", "Accuracy"],
    ):
        ax.plot(epochs, history[key[0]], label="Train")
        ax.plot(epochs, history[key[1]], label="Val")
        ax.set_xlabel("Epoch");
        ax.set_ylabel(title)
        ax.set_title(f"MERT — {title}");
        ax.legend();
        ax.grid(True)

    plt.tight_layout()
    out = plots_dir / "mert_training_curves.png"
    plt.savefig(out, dpi=150);
    plt.close()
    print(f"  График сохранён: {out}")


def train(args) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 55}")
    print(f"  Модель     : MERT-v1-95M (Feature Extraction)")
    print(f"  Слой MERT  : {args.layer_idx}")
    print(f"  Устройство : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"{'=' * 55}\n")

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_mert_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        cache=args.cache,
        seed=args.seed,
    )

    model = get_mert_model(layer_idx=args.layer_idx,
                           dropout=args.dropout, device=device)
    criterion = nn.CrossEntropyLoss()

    # Оптимизатор только для параметров головы (MERT заморожен)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_ckpt = ckpt_dir / "best_mert_model.pt"

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'LR':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer,
                                            device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>7.2%}  {lr_now:>8.2e}  "
              f"({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Сохраняем только веса головы — MERT не нужно сохранять повторно
            torch.save({
                "epoch": epoch,
                "head_state": model.head.state_dict(),
                "layer_idx": args.layer_idx,
                "val_acc": val_acc,
                "args": vars(args),
            }, best_ckpt)
            print(f"  ✓ Новый лучший val_acc={val_acc:.2%} → {best_ckpt}")

    last_ckpt = ckpt_dir / "last_mert_model.pt"
    torch.save({"epoch": args.epochs, "head_state": model.head.state_dict()}, last_ckpt)

    print(f"\nЛучшая val accuracy: {best_val_acc:.2%}")
    save_plots(history, plots_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Обучение MERT Feature Extraction")
    p.add_argument("--data_dir", default="data", type=str)
    p.add_argument("--output_dir", default="outputs", type=str)
    p.add_argument("--epochs", default=20, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--dropout", default=0.3, type=float)
    p.add_argument("--layer_idx", default=5, type=int,
                   help="Слой MERT для эмбеддингов (0-12; 5 оптимален для EMO)")
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--cache", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
