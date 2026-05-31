import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import TransitionDataset
from model import CNNLSTMTransitionDetector

# параметры
DATA_DIR      = "data"
OUTPUT_DIR    = "outputs/transitions"
NUM_TRACKS    = 1000       # сколько синтетических треков генерировать
SEQ_LEN       = 9         # сколько окон подаём в LSTM за раз (9 окон = 45 сек)
BATCH_SIZE    = 16
EPOCHS        = 30
LR            = 3e-4
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.3
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
# ─────────────────────────────────────────────────────────────────────

os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"устройство: {device}")


class SequenceDataset(Dataset):
    """
    оборачиваем плоский список окон в последовательности длиной SEQ_LEN.
    каждый семпл — SEQ_LEN последовательных окон + их метки.
    скользящее окно с шагом 1.
    """
    def __init__(self, windows, seq_len):
        self.windows = windows
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.windows) - self.seq_len + 1) // self.seq_len

    def __getitem__(self, idx):
        idx = idx * self.seq_len
        stfts  = [self.windows[idx + i][0] for i in range(self.seq_len)]
        labels = [self.windows[idx + i][1] for i in range(self.seq_len)]
        stfts  = torch.stack(stfts)   # (seq_len, 1, freq, time)
        labels = torch.stack(labels)  # (seq_len,)
        return stfts, labels


def compute_pos_weight(windows):
    # считаем вес для положительного класса — компенсируем дисбаланс
    n_pos = sum(1 for _, l in windows if int(l) == 1)
    n_neg = len(windows) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    weight = n_neg / n_pos
    print(f"pos_weight = {weight:.2f}  (переходов {n_pos}, нет переходов {n_neg})")
    return torch.tensor(weight)


def train():
    # генерируем данные
    flat_dataset = TransitionDataset(DATA_DIR, num_tracks=NUM_TRACKS)
    windows = flat_dataset.windows

    # делим окна на train/val/test до оборачивания в SequenceDataset,
    # чтобы последовательности не перемешивались между сплитами
    n = len(windows)
    n_test = int(n * TEST_RATIO)
    n_val  = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    train_windows = windows[:n_train]
    val_windows   = windows[n_train:n_train + n_val]
    test_windows  = windows[n_train + n_val:]

    train_ds = SequenceDataset(train_windows, SEQ_LEN)
    val_ds   = SequenceDataset(val_windows,   SEQ_LEN)
    test_ds  = SequenceDataset(test_windows,  SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    print(f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)} последовательностей")

    # модель
    model = CNNLSTMTransitionDetector(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    pos_weight = compute_pos_weight(windows).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        running_loss = 0.0
        for stfts, labels in train_loader:
            stfts  = stfts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(stfts)           # (batch, seq_len)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # val
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for stfts, labels in val_loader:
                stfts  = stfts.to(device)
                labels = labels.to(device)
                logits = model(stfts)
                loss   = criterion(logits, labels)
                running_val += loss.item()

        val_loss = running_val / len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"эпоха {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pt"))

    # последняя модель на всякий случай
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "checkpoints", "last_model.pt"))

    # кривые обучения
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses,   label="val loss")
    plt.xlabel("эпоха")
    plt.ylabel("loss")
    plt.legend()
    plt.title("кривые обучения")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "training_curves.png"))
    plt.show()
    print("готово. лучший val_loss:", round(best_val_loss, 4))

    return test_loader, model


if __name__ == "__main__":
    train()
