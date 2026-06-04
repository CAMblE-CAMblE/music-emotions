from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation import IDX_TO_LABEL, LABEL_TO_IDX
from src.generation.audio_utils import AudioConfig, wav_to_logmel
from src.generation.dataset import DEFAULT_DATA_DIR, build_generation_dataloaders
from src.generation.model_cvae import ConditionalVAE
from src.generation.model_unet_ae import ConditionalUNetAutoencoder
from src.generation.model_vqvae import ConditionalVQVAE


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg not in {"auto", "cuda"}:
        return device_arg
    if not torch.cuda.is_available():
        if device_arg == "cuda":
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
        return "cpu"

    major, minor = torch.cuda.get_device_capability(0)
    current_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and current_arch not in supported_arches:
        message = (
            f"GPU {torch.cuda.get_device_name(0)} has compute capability {current_arch}, "
            f"but this PyTorch build supports: {', '.join(sorted(supported_arches))}. "
            "Use --device cpu or install an older PyTorch CUDA build that supports this GPU."
        )
        if device_arg == "cuda":
            raise RuntimeError(message)
        print(f"WARNING: {message}\nFalling back to CPU because --device auto was used.")
        return "cpu"

    return "cuda"


def load_generation_model(checkpoint: str | Path, device: str, requested_model_type: str) -> tuple[torch.nn.Module, AudioConfig, str]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = AudioConfig(**ckpt.get("audio_config", {}))
    model_type = ckpt.get("model_type", "cvae") if requested_model_type == "auto" else requested_model_type

    if model_type == "cvae":
        model = ConditionalVAE(
            latent_dim=ckpt.get("latent_dim", 128),
            num_classes=ckpt.get("num_classes", 4),
            n_mels=cfg.n_mels,
            frames=cfg.target_frames,
        ).to(device)
    elif model_type == "unet_ae":
        args = ckpt.get("args", {})
        model = ConditionalUNetAutoencoder(
            num_classes=ckpt.get("num_classes", 4),
            condition_dim=ckpt.get("condition_dim", args.get("condition_dim", 32)),
            n_mels=cfg.n_mels,
            frames=cfg.target_frames,
        ).to(device)
    elif model_type == "vqvae":
        args = ckpt.get("args", {})
        model = ConditionalVQVAE(
            num_embeddings=ckpt.get("num_embeddings", args.get("num_embeddings", 512)),
            embedding_dim=ckpt.get("embedding_dim", args.get("embedding_dim", 128)),
            commitment_cost=ckpt.get("commitment_cost", args.get("commitment_cost", 0.25)),
            num_classes=ckpt.get("num_classes", 4),
            condition_dim=ckpt.get("condition_dim", args.get("condition_dim", 32)),
            n_mels=cfg.n_mels,
            frames=cfg.target_frames,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type={model_type!r}; expected cvae, unet_ae, or vqvae")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, model_type


def reconstruct(model: torch.nn.Module, mel: torch.Tensor, labels: torch.Tensor, model_type: str) -> torch.Tensor:
    if model_type == "cvae":
        recon, _, _ = model(mel, labels)
        return recon
    if model_type == "unet_ae":
        return model(mel, labels)
    if model_type == "vqvae":
        recon, _, _, _ = model(mel, labels)
        return recon
    raise ValueError(f"Unknown model_type={model_type!r}")


@torch.no_grad()
def reconstruction_metrics(model, loader, device: str, model_type: str) -> dict[str, float]:
    totals = {"mse": 0.0, "mae": 0.0, "spectral_convergence": 0.0, "cosine_similarity": 0.0, "n": 0}
    for mel, labels in tqdm(loader, desc="reconstruction", leave=False):
        mel = mel.to(device)
        labels = labels.to(device)
        recon = reconstruct(model, mel, labels, model_type)
        batch = mel.size(0)
        diff = recon - mel
        mse = F.mse_loss(recon, mel, reduction="mean")
        mae = F.l1_loss(recon, mel, reduction="mean")
        sc = torch.linalg.vector_norm(diff.flatten(1), dim=1) / torch.linalg.vector_norm(mel.flatten(1), dim=1).clamp(min=1e-8)
        cos = F.cosine_similarity(mel.flatten(1), recon.flatten(1), dim=1, eps=1e-8)
        totals["mse"] += mse.item() * batch
        totals["mae"] += mae.item() * batch
        totals["spectral_convergence"] += sc.mean().item() * batch
        totals["cosine_similarity"] += cos.mean().item() * batch
        totals["n"] += batch
    n = max(1, totals["n"])
    return {k: totals[k] / n for k in ("mse", "mae", "spectral_convergence", "cosine_similarity")}


@torch.no_grad()
def cvae_diversity_metrics(model, device: str, num_per_class: int = 8) -> dict[str, float]:
    all_distances = []
    per_class = {}
    for label, name in IDX_TO_LABEL.items():
        labels = torch.full((num_per_class,), label, dtype=torch.long, device=device)
        samples = model.sample(labels, device=device).flatten(1).cpu()
        distances = []
        for i, j in combinations(range(samples.size(0)), 2):
            distances.append(torch.mean(torch.abs(samples[i] - samples[j])).item())
        value = float(np.mean(distances)) if distances else 0.0
        per_class[name] = value
        all_distances.extend(distances)
    return {
        "mean_pairwise_l1": float(np.mean(all_distances)) if all_distances else 0.0,
        "per_class_pairwise_l1": per_class,
    }


def samples_diversity_metrics(samples_dir: str | Path, cfg: AudioConfig) -> dict:
    wavs = sorted(Path(samples_dir).glob("*.wav"))
    if len(wavs) < 2:
        return {"available": False, "reason": f"Need at least 2 .wav files in {samples_dir}"}

    grouped: dict[int, list[torch.Tensor]] = {i: [] for i in range(4)}
    all_mels = []
    for path in wavs:
        label = _label_from_wav_name(path)
        wav, _ = librosa.load(path, sr=cfg.sr, mono=True, duration=cfg.duration)
        if len(wav) < cfg.target_samples:
            wav = np.pad(wav, (0, cfg.target_samples - len(wav)), mode="constant")
        else:
            wav = wav[: cfg.target_samples]
        mel = torch.from_numpy(wav_to_logmel(wav.astype(np.float32), cfg)).flatten()
        all_mels.append(mel)
        if label is not None:
            grouped[label].append(mel)

    def mean_pairwise(items: list[torch.Tensor]) -> float:
        distances = [torch.mean(torch.abs(items[i] - items[j])).item() for i, j in combinations(range(len(items)), 2)]
        return float(np.mean(distances)) if distances else 0.0

    return {
        "available": True,
        "samples": len(wavs),
        "mean_pairwise_l1": mean_pairwise(all_mels),
        "per_class_pairwise_l1": {IDX_TO_LABEL[label]: mean_pairwise(items) for label, items in grouped.items()},
    }


def _label_from_wav_name(path: Path) -> int | None:
    prefix = path.stem.split("_")[0].capitalize()
    return LABEL_TO_IDX.get(prefix)


def evaluate_mert(samples_dir: str | Path, mert_checkpoint: str | Path, device: str, out_dir: Path) -> dict:
    """Evaluate generated WAV files with the existing MERT classifier if available."""
    samples_dir = Path(samples_dir)
    wavs = sorted(samples_dir.glob("*.wav"))
    labeled_wavs = [(p, _label_from_wav_name(p)) for p in wavs]
    labeled_wavs = [(p, y) for p, y in labeled_wavs if y is not None]
    if not labeled_wavs:
        return {"available": False, "reason": f"No labeled .wav files found in {samples_dir}"}
    if not Path(mert_checkpoint).exists():
        return {"available": False, "reason": f"MERT checkpoint not found: {mert_checkpoint}"}

    try:
        from src.mert.model_mert import MERT_SR, get_mert_model, get_processor
    except Exception as exc:
        return {"available": False, "reason": f"Could not import MERT modules: {exc}"}

    try:
        ckpt = torch.load(mert_checkpoint, map_location=device)
        layer_idx = ckpt.get("layer_idx", 7)
        model = get_mert_model(layer_idx=layer_idx, device=device)
        model.head.load_state_dict(ckpt["head_state"])
        model.eval()
        processor = get_processor()
    except Exception as exc:
        return {"available": False, "reason": f"Could not load MERT evaluator: {exc}"}

    y_true, y_pred = [], []
    target_len = int(MERT_SR * 15)

    with torch.no_grad():
        for path, target in tqdm(labeled_wavs, desc="mert", leave=False):
            wav, _ = librosa.load(path, sr=MERT_SR, mono=True, duration=15.0)
            if len(wav) < target_len:
                wav = np.pad(wav, (0, target_len - len(wav)), mode="constant")
            else:
                wav = wav[:target_len]
            encoded = processor(wav, sampling_rate=MERT_SR, return_tensors="pt", padding=False)
            input_values = encoded["input_values"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            pred = int(model(input_values, attention_mask).argmax(dim=1).item())
            y_true.append(target)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    per_class = {}
    for label, name in IDX_TO_LABEL.items():
        idxs = [i for i, y in enumerate(y_true) if y == label]
        per_class[name] = float(np.mean([y_pred[i] == label for i in idxs])) if idxs else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    names = [IDX_TO_LABEL[i] for i in range(4)]
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=names, yticklabels=names, cmap="Blues")
    plt.xlabel("MERT prediction")
    plt.ylabel("Generation condition")
    plt.title("Generated emotion confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "generation_confusion_matrix.png", dpi=150)
    plt.close()

    return {
        "available": True,
        "samples": len(y_true),
        "emotion_accuracy": float(acc),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate(args) -> None:
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg, model_type = load_generation_model(args.checkpoint, device, args.model_type)

    _, val_loader, test_loader = build_generation_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        cfg=cfg,
    )
    loader = val_loader if args.split == "val" else test_loader

    metrics = {
        "checkpoint": args.checkpoint,
        "model_type": model_type,
        "split": args.split,
        "reconstruction": reconstruction_metrics(model, loader, device, model_type),
        "diversity": (
            cvae_diversity_metrics(model, device, args.num_generated_per_class)
            if model_type == "cvae"
            else samples_diversity_metrics(args.samples_dir, cfg)
        ),
        "audio_config": cfg.to_dict(),
    }

    if not args.skip_mert:
        metrics["mert_evaluation"] = evaluate_mert(args.samples_dir, args.mert_checkpoint, device, out_dir / "plots")
    else:
        metrics["mert_evaluation"] = {"available": False, "reason": "Skipped by --skip_mert"}

    metrics_path = out_dir / "generation_evaluation.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved evaluation metrics to {metrics_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate generated audio quality and emotion accuracy")
    p.add_argument("--checkpoint", default="outputs/generation/unet_ae_final/checkpoints/best_model.pt", type=str)
    p.add_argument("--model_type", choices=["auto", "cvae", "unet_ae", "vqvae"], default="auto", type=str)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--out_dir", default="outputs/generation/unet_ae_final", type=str)
    p.add_argument("--samples_dir", default="outputs/generation/unet_ae_final/samples", type=str)
    p.add_argument("--mert_checkpoint", default="outputs/mert_best/checkpoints/best_mert_model.pt", type=str)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--num_generated_per_class", default=8, type=int)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--skip_mert", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
