import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

from dataset import TransitionDataset
from model import CNNLSTMTransitionDetector
from train import SequenceDataset, SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

DATA_DIR    = "data"
OUTPUT_DIR  = "outputs/transitions"
CHECKPOINT  = os.path.join(OUTPUT_DIR, "checkpoints", "best_model.pt")
NUM_TRACKS  = 1000
BATCH_SIZE  = 16
TEST_RATIO  = 0.15
VAL_RATIO   = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate():
    # пересоздаём тот же датасет с тем же seed
    flat_dataset = TransitionDataset(DATA_DIR, num_tracks=NUM_TRACKS, seed=42)
    windows = flat_dataset.windows

    n = len(windows)
    n_test  = int(n * TEST_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    test_windows = windows[n_train + n_val:]
    test_ds      = SequenceDataset(test_windows, SEQ_LEN)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # загружаем модель
    model = CNNLSTMTransitionDetector(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for stfts, labels in test_loader:
            stfts = stfts.to(device)
            logits = model(stfts)                        # (batch, seq_len)
            probs  = torch.sigmoid(logits).cpu().numpy() # вероятности
            all_probs.append(probs.flatten())
            all_labels.append(labels.numpy().flatten())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # метрики
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nROC-AUC: {auc:.4f}")

    # порог 0.5 для classification report
    preds = (all_probs >= 0.5).astype(int)
    print("\n", classification_report(all_labels, preds, target_names=["нет перехода", "переход"]))

    # ROC-кривая
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-кривая (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="случайный классификатор")
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.title("ROC-кривая: обнаружение переходов")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "roc_curve.png"))
    plt.show()

    print(f"ROC-кривая сохранена в {OUTPUT_DIR}/plots/roc_curve.png")


if __name__ == "__main__":
    evaluate()
