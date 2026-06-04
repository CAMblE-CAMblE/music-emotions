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
from src.generation.model_cvae import ConditionalVAE


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


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def cvae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    recon_loss_type: str = "mixed",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse = F.mse_loss(recon, target)
    l1 = F.l1_loss(recon, target)
    if recon_loss_type == "mixed":
        recon_loss = mse + 0.5 * l1
    elif recon_loss_type == "l1":
        recon_loss = l1
    elif recon_loss_type == "mse":
        recon_loss = mse
    else:
        raise ValueError(f"Unknown recon_loss_type={recon_loss_type!r}")
    if beta == 0.0:
        # Autoencoder baseline: do not evaluate KL at all. If logvar overflows,
        # 0 * NaN would still poison the total loss.
        kl = torch.zeros((), device=target.device, dtype=target.dtype)
    else:
        kl = kl_divergence(mu, logvar)
    total = recon_loss + beta * kl
    return total, recon_loss, kl


def beta_for_epoch(epoch: int, epochs: int, beta_max: float) -> float:
    if epochs <= 1:
        return beta_max
    warmup_epochs = max(1, int(0.5 * epochs))
    return min(beta_max, beta_max * epoch / warmup_epochs)


def run_epoch(model, loader, optimizer, device, beta: float, train: bool, recon_loss_type: str) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    desc = "train" if train else "val"

    for mel, labels in tqdm(loader, desc=f"  {desc}", leave=False):
        mel = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(mel, labels)
            loss, recon_loss, kl = cvae_loss(recon, mel, mu, logvar, beta, recon_loss_type)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss detected in {'train' if train else 'val'} epoch: "
                    f"loss={loss.item()}, recon={recon_loss.item()}, kl={kl.item()}, "
                    f"beta={beta}, recon_loss_type={recon_loss_type}. "
                    "Try lower lr/beta_max or inspect the checkpoint."
                )
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch = mel.size(0)
        totals["loss"] += loss.item() * batch
        totals["recon"] += recon_loss.item() * batch
        totals["kl"] += kl.item() * batch
        totals["n"] += batch

    n = max(1, totals["n"])
    return {"loss": totals["loss"] / n, "recon": totals["recon"] / n, "kl": totals["kl"] / n}


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

    axes[2].plot(epochs, history["train_kl"], label="train")
    axes[2].plot(epochs, history["val_kl"], label="val")
    axes[2].plot(epochs, history["beta"], label="beta")
    axes[2].set_title("KL and beta")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "training_curves.png", dpi=150)
    plt.close()


@torch.no_grad()
def save_recon_examples(model, loader, device, plots_dir: Path, max_items: int = 6) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    mel, labels = next(iter(loader))
    mel = mel[:max_items].to(device)
    labels = labels[:max_items].to(device)
    recon, _, _ = model(mel, labels)

    n = mel.size(0)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(mel[i, 0].cpu().numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        axes[0, i].set_title(f"input {int(labels[i])}")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0].cpu().numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
        axes[1, i].set_title("recon")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(plots_dir / "recon_examples.png", dpi=150)
    plt.close()


def save_checkpoint(path: Path, model, optimizer, epoch: int, val_loss: float, args, cfg: AudioConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "latent_dim": args.latent_dim,
            "num_classes": 4,
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

    model = ConditionalVAE(
        latent_dim=args.latent_dim,
        num_classes=4,
        n_mels=cfg.n_mels,
        frames=cfg.target_frames,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss": [], "val_recon": [], "val_kl": [], "beta": [],
    }
    best_val = float("inf")

    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(f"Input shape: (B, 1, {cfg.n_mels}, {cfg.target_frames})")
    print(f"Reconstruction loss: {args.recon_loss_type}")

    for epoch in range(1, args.epochs + 1):
        beta = beta_for_epoch(epoch, args.epochs, args.beta_max)
        train_metrics = run_epoch(model, train_loader, optimizer, device, beta, train=True, recon_loss_type=args.recon_loss_type)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, beta, train=False, recon_loss_type=args.recon_loss_type)

        history["train_loss"].append(train_metrics["loss"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_recon"].append(val_metrics["recon"])
        history["val_kl"].append(val_metrics["kl"])
        history["beta"].append(beta)

        print(
            f"epoch {epoch:03d}/{args.epochs} beta={beta:.5f} "
            f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} "
            f"val_recon={val_metrics['recon']:.5f} val_kl={val_metrics['kl']:.5f}"
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
        "audio_config": cfg.to_dict(),
        "args": vars(args),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved checkpoints to {ckpt_dir}")
    print(f"Saved plots to {plots_dir}")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Conditional VAE for emotion-conditioned music generation")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--out_dir", default="outputs/generation/cvae_mel", type=str)
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--latent_dim", default=128, type=int)
    p.add_argument("--beta_max", default=0.01, type=float)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--train_segments_per_file", default=4, type=int)
    p.add_argument("--recon_loss_type", choices=["mixed", "l1", "mse"], default="mixed", type=str)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
