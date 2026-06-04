from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation import IDX_TO_LABEL
from src.generation.audio_utils import (
    AudioConfig,
    load_audio,
    logmel_to_wav_griffin_lim,
    save_spectrogram_png,
    save_wav,
    wav_to_logmel,
)
from src.generation.dataset import (
    DEFAULT_DATA_DIR,
    GenerationMelDataset,
    split_file_list,
    validate_generation_data_dir,
)
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


def load_reconstruction_model(
    checkpoint: str | Path,
    device: str,
    requested_model_type: str,
) -> tuple[torch.nn.Module, AudioConfig, dict, str]:
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
    return model, cfg, ckpt, model_type


def reconstruction_from_latent(
    model: ConditionalVAE,
    mel: torch.Tensor,
    label: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, logvar = model.encode(mel, label)
    if mode == "reconstruct_only":
        z = mu
    elif mode == "sample_latent":
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
    else:
        raise ValueError(f"Unknown mode: {mode}")
    recon = model.decode(z, label)
    return recon, mu, logvar


def reconstruct_batch(
    model: torch.nn.Module,
    mel: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if model_type == "cvae":
        return reconstruction_from_latent(model, mel, labels, mode)  # type: ignore[arg-type]
    if model_type == "unet_ae":
        return model(mel, labels), None, None
    if model_type == "vqvae":
        recon, _, _, _ = model(mel, labels)
        return recon, None, None
    raise ValueError(f"Unknown model_type={model_type!r}")


def compute_batch_metrics(
    real: torch.Tensor,
    recon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    diff = recon - real
    mse = diff.pow(2).flatten(1).mean(dim=1)
    mae = diff.abs().flatten(1).mean(dim=1)
    spectral_convergence = (
        torch.linalg.vector_norm(diff.flatten(1), dim=1)
        / torch.linalg.vector_norm(real.flatten(1), dim=1).clamp(min=1e-8)
    )
    cosine_similarity = torch.nn.functional.cosine_similarity(
        real.flatten(1),
        recon.flatten(1),
        dim=1,
        eps=1e-8,
    )
    return mse, mae, spectral_convergence, cosine_similarity


def summarize_metric_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "mse": 0.0,
            "mae": 0.0,
            "spectral_convergence": 0.0,
            "cosine_similarity_mean": 0.0,
            "cosine_similarity_std": 0.0,
            "count": 0,
        }
    cosine_values = [r["cosine_similarity"] for r in rows]
    return {
        "mse": float(np.mean([r["mse"] for r in rows])),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "spectral_convergence": float(np.mean([r["spectral_convergence"] for r in rows])),
        "cosine_similarity_mean": float(np.mean(cosine_values)),
        "cosine_similarity_std": float(np.std(cosine_values)),
        "count": len(rows),
    }


def plot_comparison(
    out_path: Path,
    real_mel: np.ndarray,
    recon_mel: np.ndarray,
    title: str,
) -> None:
    diff = np.abs(real_mel - recon_mel)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), squeeze=False)
    axes = axes[:, 0]

    images = [
        (real_mel, "Real mel", "magma", 0.0, 1.0),
        (recon_mel, "Reconstructed mel", "magma", 0.0, 1.0),
        (diff, "Absolute error", "viridis", 0.0, float(max(diff.max(), 1e-6))),
    ]
    for ax, (data, name, cmap, vmin, vmax) in zip(axes, images):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("frames")
        ax.set_ylabel("mel bins")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)

    fig.suptitle(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_latent_histogram(out_path: Path, mu_values: np.ndarray) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(mu_values.reshape(-1), bins=80, color="#2d5f7f", alpha=0.85)
    plt.title("Latent mu histogram")
    plt.xlabel("mu value")
    plt.ylabel("count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_test_file_list(data_dir: str | Path, seed: int) -> list[tuple[str, int]]:
    data_dir = validate_generation_data_dir(data_dir)
    base = GenerationMelDataset(data_dir, split="scan", seed=seed)
    _, _, test_files = split_file_list(base.samples, seed=seed)
    return test_files


def collect_selected_examples(
    test_files: list[tuple[str, int]],
    per_class: int,
    seed: int,
) -> list[tuple[str, int]]:
    grouped: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for item in test_files:
        grouped[item[1]].append(item)

    rng = random.Random(seed)
    selected: list[tuple[str, int]] = []
    for label in range(4):
        items = grouped[label]
        rng.shuffle(items)
        selected.extend(items[:per_class])
    return selected


@torch.no_grad()
def evaluate_full_test_metrics(
    model: torch.nn.Module,
    test_files: list[tuple[str, int]],
    cfg: AudioConfig,
    args,
    device: str,
    model_type: str,
) -> tuple[dict, np.ndarray]:
    dataset = GenerationMelDataset(args.data_dir, file_list=test_files, cfg=cfg, split="test", seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    rows: list[dict] = []
    latent_chunks = []

    for mel, labels in tqdm(loader, desc="test reconstruction metrics", leave=False):
        mel = mel.to(device)
        labels = labels.to(device)
        recon, mu, _ = reconstruct_batch(model, mel, labels, args.mode, model_type)
        mse, mae, sc, cos = compute_batch_metrics(mel, recon)
        if mu is not None:
            latent_chunks.append(mu.detach().cpu().numpy())

        for i in range(mel.size(0)):
            label = int(labels[i].item())
            rows.append(
                {
                    "label": label,
                    "class": IDX_TO_LABEL[label],
                    "mse": float(mse[i].item()),
                    "mae": float(mae[i].item()),
                    "spectral_convergence": float(sc[i].item()),
                    "cosine_similarity": float(cos[i].item()),
                }
            )

    by_class = {}
    for label, name in IDX_TO_LABEL.items():
        by_class[name] = summarize_metric_rows([r for r in rows if r["label"] == label])

    metrics = {
        "overall": summarize_metric_rows(rows),
        "by_class": by_class,
        "mode": args.mode,
        "split": "test",
    }
    mu_values = np.concatenate(latent_chunks, axis=0) if latent_chunks else np.empty((0, 0), dtype=np.float32)
    return metrics, mu_values


@torch.no_grad()
def save_debug_examples(
    model: torch.nn.Module,
    examples: list[tuple[str, int]],
    cfg: AudioConfig,
    args,
    device: str,
    model_type: str,
) -> list[dict]:
    out_dir = Path(args.out_dir)
    real_dir = out_dir / "real"
    recon_dir = out_dir / "reconstructed"
    comparison_dir = out_dir / "comparisons"
    real_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, (path, label) in enumerate(tqdm(examples, desc="saving debug examples")):
        class_name = IDX_TO_LABEL[label]
        stem = f"sample_{idx:03d}"

        wav = load_audio(path, cfg=cfg, crop_mode="center")
        real_mel = wav_to_logmel(wav, cfg)
        mel_tensor = torch.from_numpy(real_mel).unsqueeze(0).unsqueeze(0).to(device)
        label_tensor = torch.tensor([label], dtype=torch.long, device=device)

        recon_tensor, mu, logvar = reconstruct_batch(model, mel_tensor, label_tensor, args.mode, model_type)
        recon_mel = recon_tensor[0, 0].detach().cpu().numpy()
        recon_wav = logmel_to_wav_griffin_lim(recon_mel, cfg=cfg, n_iter=args.griffin_lim_iters)

        save_wav(real_dir / f"{stem}.wav", wav, cfg)
        save_spectrogram_png(real_dir / f"{stem}.png", real_mel, cfg, title=f"Real {class_name}")
        save_wav(recon_dir / f"{stem}.wav", recon_wav, cfg)
        save_spectrogram_png(recon_dir / f"{stem}.png", recon_mel, cfg, title=f"Reconstructed {class_name}")

        if idx < args.num_comparisons:
            plot_comparison(
                comparison_dir / f"comparison_{idx:03d}.png",
                real_mel,
                recon_mel,
                title=f"{stem} | {class_name}",
            )

        mse, mae, sc, cos = compute_batch_metrics(mel_tensor, recon_tensor)
        item = {
            "index": idx,
            "stem": stem,
            "source_path": str(path),
            "label": label,
            "class": class_name,
            "real_wav": str(real_dir / f"{stem}.wav"),
            "real_png": str(real_dir / f"{stem}.png"),
            "reconstructed_wav": str(recon_dir / f"{stem}.wav"),
            "reconstructed_png": str(recon_dir / f"{stem}.png"),
            "mse": float(mse[0].item()),
            "mae": float(mae[0].item()),
            "spectral_convergence": float(sc[0].item()),
            "cosine_similarity": float(cos[0].item()),
        }
        if mu is not None and logvar is not None:
            item.update(
                {
                    "mu_mean": float(mu.mean().item()),
                    "mu_std": float(mu.std().item()),
                    "logvar_mean": float(logvar.mean().item()),
                }
            )
        manifest.append(item)

    return manifest


def latent_stats(mu_values: np.ndarray) -> dict:
    if mu_values.size == 0:
        return {"available": False, "reason": "model has no latent mu values", "count": 0}
    return {
        "available": True,
        "count": int(mu_values.shape[0]),
        "latent_dim": int(mu_values.shape[1]),
        "mean_mu": float(mu_values.mean()),
        "std_mu": float(mu_values.std()),
        "min_mu": float(mu_values.min()),
        "max_mu": float(mu_values.max()),
        "mean_abs_mu": float(np.abs(mu_values).mean()),
    }


def evaluate(args) -> None:
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, ckpt, model_type = load_reconstruction_model(args.checkpoint, device, args.model_type)
    test_files = build_test_file_list(args.data_dir, args.seed)
    selected = collect_selected_examples(test_files, args.examples_per_class, args.seed)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {model_type}")
    print(f"Dataset: {validate_generation_data_dir(args.data_dir)}")
    print(f"Output: {out_dir}")
    print(f"Mode: {args.mode}")
    print(f"Selected debug examples: {len(selected)}")

    metrics, mu_values = evaluate_full_test_metrics(model, test_files, cfg, args, device, model_type)
    stats = latent_stats(mu_values)
    metrics["latent"] = stats
    metrics["checkpoint"] = {
        "path": str(args.checkpoint),
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "model_type": model_type,
        "latent_dim": ckpt.get("latent_dim"),
    }
    metrics["audio_config"] = cfg.to_dict()

    manifest = save_debug_examples(model, selected, cfg, args, device, model_type)
    metrics["debug_examples"] = manifest

    if stats.get("available"):
        plot_latent_histogram(out_dir / "latent_histogram.png", mu_values)

    metrics_path = out_dir / "reconstruction_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: metrics[k] for k in ("overall", "by_class", "latent")}, ensure_ascii=False, indent=2))
    print(f"Saved real examples to: {out_dir / 'real'}")
    print(f"Saved reconstructed examples to: {out_dir / 'reconstructed'}")
    print(f"Saved comparisons to: {out_dir / 'comparisons'}")
    if stats.get("available"):
        print(f"Saved latent histogram to: {out_dir / 'latent_histogram.png'}")
    else:
        print("Skipped latent histogram: model has no latent mu values")
    print(f"Saved metrics to: {metrics_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose reconstruction quality for CVAE, U-Net AE, or VQ-VAE")
    p.add_argument("--checkpoint", default="outputs/generation/cvae_mel/checkpoints/best_model.pt", type=str)
    p.add_argument("--model_type", choices=["auto", "cvae", "unet_ae", "vqvae"], default="auto", type=str)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--out_dir", default="outputs/generation/reconstruction_debug", type=str)
    p.add_argument("--examples_per_class", default=5, type=int)
    p.add_argument("--num_comparisons", default=8, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--workers", default=0, type=int)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--griffin_lim_iters", default=32, type=int)
    p.add_argument("--mode", choices=["reconstruct_only", "sample_latent"], default="reconstruct_only")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
