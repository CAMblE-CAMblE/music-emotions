import argparse
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Импорт из src
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.baseline.dataset import build_dataloaders
from src.baseline.model import get_model


# Воспроизводимость

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

    for specs, labels in tqdm(loader, desc="  train", leave=False):
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += specs.size(0)

    return total_loss / total, correct / total


# Валидация

@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for specs, labels in tqdm(loader, desc="  val  ", leave=False):
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(specs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * specs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += specs.size(0)

    return total_loss / total, correct / total


# Сохранение графиков

def save_training_plots(history: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    out = plots_dir / "training_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  График сохранён: {out}")


def train(args) -> None:
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 55}")
    print(f"  Устройство : {device}")
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"{'=' * 55}\n")

    # Директории для сохранения
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Данные
    train_loader, val_loader, _ = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        cache=args.cache,
        seed=args.seed,
    )

    # Модель
    model = get_model(dropout=args.dropout, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_ckpt = ckpt_dir / "best_model.pt"

    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
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

        # Сохраняем лучший чекпоинт
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, best_ckpt)
            print(f"  ✓ Новый лучший val_acc={val_acc:.2%} — сохранён в {best_ckpt}")

    # Последний чекпоинт
    last_ckpt = ckpt_dir / "last_model.pt"
    torch.save({"epoch": args.epochs, "model_state": model.state_dict()}, last_ckpt)

    print(f"\nЛучшая val accuracy: {best_val_acc:.2%}")
    print(f"Чекпоинты: {ckpt_dir}")

    save_training_plots(history, plots_dir)


# Парсинг аргументов

def parse_args():
    p = argparse.ArgumentParser(description="Обучение EmotionCNN")
    p.add_argument("--data_dir", default="data", type=str)
    p.add_argument("--output_dir", default="outputs", type=str)
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--dropout", default=0.5, type=float)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--cache", action="store_true",
                   help="Кешировать спектрограммы в RAM")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
