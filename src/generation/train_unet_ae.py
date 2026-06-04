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
from src.generation.model_unet_ae import ConditionalUNetAutoencoder


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
    if loss_type == "mixed":
        mse = F.mse_loss(recon, target)
        return 0.8 * l1 + 0.2 * mse
    raise ValueError(f"Unknown loss_type={loss_type!r}")


def run_epoch(model, loader, optimizer, device: str, loss_type: str, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "n": 0}
    desc = "train" if train else "val"

    for mel, labels in tqdm(loader, desc=f"  {desc}", leave=False):
        mel = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            recon = model(mel, labels)
            loss = reconstruction_loss(recon, mel, loss_type)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {desc} loss detected: {loss.item()} with loss_type={loss_type}")
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch = mel.size(0)
        totals["loss"] += loss.item() * batch
        totals["n"] += batch

    return {"loss": totals["loss"] / max(1, totals["n"])}


def save_training_curves(history: dict[str, list[float]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Conditional U-Net AE reconstruction loss")
    plt.grid(True)
    plt.legend()
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
    recon = model(mel, labels)

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
            "model_type": "unet_ae",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "num_classes": 4,
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

    model = ConditionalUNetAutoencoder(
        num_classes=4,
        condition_dim=args.condition_dim,
        n_mels=cfg.n_mels,
        frames=cfg.target_frames,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    print(f"Device: {device}")
    print(f"Output: {out_dir}")
    print(f"Model: ConditionalUNetAutoencoder")
    print(f"Input shape: (B, 1, {cfg.n_mels}, {cfg.target_frames})")
    print(f"Loss type: {args.loss_type}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.loss_type, train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, device, args.loss_type, train=False)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f}"
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
        "model_type": "unet_ae",
        "audio_config": cfg.to_dict(),
        "args": vars(args),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved checkpoints to {ckpt_dir}")
    print(f"Saved plots to {plots_dir}")
    print(f"Saved metrics to {out_dir / 'metrics.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Conditional U-Net Autoencoder for log-mel reconstruction")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--out_dir", default="outputs/generation/unet_ae_final", type=str)
    p.add_argument("--epochs", default=5, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--loss_type", choices=["l1", "mixed"], default="l1", type=str)
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--workers", default=2, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--train_segments_per_file", default=4, type=int)
    p.add_argument("--condition_dim", default=32, type=int)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
