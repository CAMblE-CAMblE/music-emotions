from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation.audio_utils import AudioConfig
from src.generation.dataset import DEFAULT_DATA_DIR, build_generation_dataloaders
from src.generation.model_vqvae import ConditionalVQVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    l1 = F.l1_loss(recon, target)
    if loss_type == "l1":
        return l1
    if loss_type == "mse":
        return F.mse_loss(recon, target)
    if loss_type == "mixed":
        return 0.8 * l1 + 0.2 * F.mse_loss(recon, target)
    raise ValueError(f"Unknown loss_type={loss_type!r}")


def run_epoch(model, loader, optimizer, device: str, loss_type: str, vq_weight: float, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "vq": 0.0, "perplexity": 0.0, "n": 0}
    desc = "train" if train else "val"

    for mel, labels in tqdm(loader, desc=f"  {desc}", leave=False):
        mel = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            recon, vq_loss, _, perplexity = model(mel, labels)
            recon_loss = reconstruction_loss(recon, mel, loss_type)
            loss = recon_loss + vq_weight * vq_loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite {desc} loss: loss={loss.item()}, recon={recon_loss.item()}, "
                    f"vq={vq_loss.item()}, perplexity={perplexity.item()}"
                )
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch = mel.size(0)
        totals["loss"] += loss.item() * batch
        totals["recon"] += recon_loss.item() * batch
        totals["vq"] += vq_loss.item() * batch
        totals["perplexity"] += perplexity.item() * batch
        totals["n"] += batch

    n = max(1, totals["n"])
    return {k: totals[k] / n for k in ("loss", "recon", "vq", "perplexity")}


def save_training_curves(history: dict[str, list[float]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, history["train_recon"], label="train")
    axes[1].plot(epochs, history["val_recon"], label="val")
    axes[1].set_title("Reconstruction")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(epochs, history["train_vq"], label="train vq")
    axes[2].plot(epochs, history["val_vq"], label="val vq")
    axes[2].plot(epochs, history["train_perplexity"], label="train perplexity")
    axes[2].plot(epochs, history["val_perplexity"], label="val perplexity")
    axes[2].set_title("VQ loss / perplexity")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close()


@torch.no_grad()
def save_recon_examples(model, loader, device: str, plots_dir: Path, max_items: int = 6) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    mel, labels = next(iter(loader))
    mel = mel[:max_items].to(device)
    labels = labels[:max_items].to(device)
    recon, _, indices, _ = model(mel, labels)

    n = mel.size(0)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 8), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(mel[i, 0].cpu().numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        axes[0, i].set_title(f"input {int(labels[i])}")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].cpu().numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        axes[1, i].set_title("recon")
        axes[1, i].axis("off")
        axes[2, i].imshow(indices[i].cpu().numpy(), origin="lower", aspect="auto", cmap="tab20")
        axes[2, i].set_title("code indices")
        axes[2, i].axis("off")
    plt.tight_layout()
    plt.savefig(plots_dir / "recon_examples.png", dpi=150)
    plt.close()


def save_checkpoint(path: Path, model, optimizer, epoch: int, val_loss: float, args, cfg: AudioConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_type": "vqvae",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "num_classes": 4,
            "num_embeddings": args.num_embeddings,
            "embedding_dim": args.embedding_dim,
            "commitment_cost": args.commitment_cost,
            "condition_dim": args.condition_dim,
            "audio_config": cfg.to_dict(),
            "args": vars(args),
        },
        path,
    )


def train(args) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = AudioConfig()
    train_loader, val_loader, _ = build_generation_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        cfg=cfg,
        train_segments_per_file=args.train_segments_per_file,
    )

    model = ConditionalVQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        num_classes=4,
        condition_dim=args.condition_dim,
        n_mels=cfg.n_mels,
        frames=cfg.target_frames,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = {
        "train_loss": [], "train_recon": [], "train_vq": [], "train_perplexity": [],
        "val_loss": [], "val_recon": [], "val_vq": [], "val_perplexity": [],
    }
    best_val = float("inf")

    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print("Model: ConditionalVQVAE")
    print(f"Input shape: (B, 1, {cfg.n_mels}, {cfg.target_frames})")
    print(f"Code grid: ({cfg.n_mels // 16}, {cfg.target_frames // 16})")
    print(f"Codebook: {args.num_embeddings} x {args.embedding_dim}")
    print(f"Loss type: {args.loss_type}; vq_weight={args.vq_weight}; commitment_cost={args.commitment_cost}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.loss_type, args.vq_weight, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, args.loss_type, args.vq_weight, train=False)

        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            history[f"{prefix}_loss"].append(metrics["loss"])
            history[f"{prefix}_recon"].append(metrics["recon"])
            history[f"{prefix}_vq"].append(metrics["vq"])
            history[f"{prefix}_perplexity"].append(metrics["perplexity"])

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} "
            f"val_recon={val_metrics['recon']:.5f} val_vq={val_metrics['vq']:.5f} "
            f"val_perplexity={val_metrics['perplexity']:.2f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(ckpt_dir / "best_model.pt", model, optimizer, epoch, best_val, args, cfg)

    save_checkpoint(ckpt_dir / "last_model.pt", model, optimizer, args.epochs, history["val_loss"][-1], args, cfg)
    save_training_curves(history, plots_dir)
    save_recon_examples(model, val_loader, device, plots_dir)

    metrics = {
        "best_val_loss": best_val,
        "history": history,
        "model_type": "vqvae",
        "audio_config": cfg.to_dict(),
        "args": vars(args),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved checkpoints to {ckpt_dir}")
    print(f"Saved plots to {plots_dir}")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Conditional VQ-VAE for log-mel music generation")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--out_dir", default="outputs/generation/vqvae_final", type=str)
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--loss_type", choices=["l1", "mixed", "mse"], default="l1", type=str)
    p.add_argument("--vq_weight", default=1.0, type=float)
    p.add_argument("--num_embeddings", default=512, type=int)
    p.add_argument("--embedding_dim", default=128, type=int)
    p.add_argument("--commitment_cost", default=0.25, type=float)
    p.add_argument("--condition_dim", default=32, type=int)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--train_segments_per_file", default=4, type=int)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
