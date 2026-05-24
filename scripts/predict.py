import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_audio, audio_to_spectrogram, IDX_TO_LABEL
from src.model import get_model


def predict(audio_path: str, checkpoint: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем модель
    model = get_model(device=device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Обрабатываем аудио
    y = load_audio(audio_path)
    spec = audio_to_spectrogram(y)  # (1, H, W) float32
    x = torch.from_numpy(spec).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = IDX_TO_LABEL[pred_idx]

    print(f"\nФайл: {audio_path}")
    print(f"Эмоция: {pred_label}  (уверенность {probs[pred_idx]:.1%})")
    print("\nВероятности по классам:")
    for i, name in IDX_TO_LABEL.items():
        bar = "█" * int(probs[i] * 30)
        print(f"  {name:8s}  {probs[i]:5.1%}  {bar}")


def parse_args():
    p = argparse.ArgumentParser(description="Предсказание эмоции по аудиофайлу")
    p.add_argument("--audio", required=True, type=str)
    p.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt",
                   type=str)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(args.audio, args.checkpoint)
