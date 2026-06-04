from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

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
from src.generation.dataset import DEFAULT_DATA_DIR
from src.generation.model_vqvae import ConditionalVQVAE
from src.generation.synthesize_cvae_unet import parse_emotion, resolve_device, select_source_examples


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vqvae(checkpoint: str | Path, device: str) -> tuple[ConditionalVQVAE, AudioConfig]:
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = AudioConfig(**ckpt.get("audio_config", {}))
    args = ckpt.get("args", {})
    model_type = ckpt.get("model_type", "vqvae")
    if model_type != "vqvae":
        raise RuntimeError(f"Expected a VQ-VAE checkpoint, got model_type={model_type!r}")

    model = ConditionalVQVAE(
        num_embeddings=ckpt.get("num_embeddings", args.get("num_embeddings", 512)),
        embedding_dim=ckpt.get("embedding_dim", args.get("embedding_dim", 128)),
        commitment_cost=ckpt.get("commitment_cost", args.get("commitment_cost", 0.25)),
        num_classes=ckpt.get("num_classes", 4),
        condition_dim=ckpt.get("condition_dim", args.get("condition_dim", 32)),
        n_mels=cfg.n_mels,
        frames=cfg.target_frames,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def replace_code_indices(indices: torch.Tensor, num_embeddings: int, replace_prob: float) -> torch.Tensor:
    if replace_prob <= 0.0:
        return indices
    if replace_prob > 1.0:
        raise ValueError("--replace_prob must be in [0, 1]")
    mask = torch.rand_like(indices.float()) < replace_prob
    random_codes = torch.randint(0, num_embeddings, size=indices.shape, device=indices.device)
    return torch.where(mask, random_codes, indices)


def mode_tag(mode: str, replace_prob: float) -> str:
    if mode == "vary_codes":
        return f"replace{int(round(replace_prob * 100)):03d}"
    return mode


@torch.no_grad()
def synthesize(args) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, cfg = load_vqvae(args.checkpoint, device)
    labels = parse_emotion(args.emotion)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = mode_tag(args.mode, args.replace_prob)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {out_dir}")
    print(f"Mode: {args.mode}")
    print(f"Emotion: {args.emotion}")
    if args.mode == "vary_codes":
        print(f"Replace prob: {args.replace_prob}")

    if args.mode == "sample_codes":
        for label in labels:
            name = IDX_TO_LABEL[label].lower()
            label_tensor = torch.full((args.num_samples,), label, dtype=torch.long, device=device)
            mels = model.sample_random_codes(label_tensor, device=device).detach().cpu()
            for idx, mel in enumerate(mels):
                stem = f"{name}_{tag}_{idx:03d}"
                wav = logmel_to_wav_griffin_lim(mel[0], cfg=cfg, n_iter=args.griffin_lim_iters)
                save_wav(out_dir / f"{stem}.wav", wav, cfg=cfg)
                save_spectrogram_png(out_dir / f"{stem}.png", mel[0], cfg=cfg, title=f"{IDX_TO_LABEL[label]} VQ-VAE random codes")
                print(f"  saved {stem}.wav / {stem}.png")
        return

    selected = select_source_examples(args.data_dir, labels, args.split, args.num_samples, args.seed)
    for label in labels:
        name = IDX_TO_LABEL[label].lower()
        for idx, (path, _) in enumerate(selected[label]):
            wav = load_audio(path, cfg=cfg, crop_mode="center")
            mel = wav_to_logmel(wav, cfg)
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            label_tensor = torch.tensor([label], dtype=torch.long, device=device)

            z_e = model.encode(mel_tensor, label_tensor)
            z_q, _, indices, _ = model.quantize(z_e)
            if args.mode == "vary_codes":
                varied_indices = replace_code_indices(indices, model.num_embeddings, args.replace_prob)
                generated = model.decode_indices(varied_indices, label_tensor)
            elif args.mode == "reconstruct":
                generated = model.decode(z_q, label_tensor)
            else:
                raise ValueError(f"Unknown mode={args.mode!r}")

            generated_mel = generated[0, 0].detach().cpu()
            generated_wav = logmel_to_wav_griffin_lim(generated_mel, cfg=cfg, n_iter=args.griffin_lim_iters)
            stem = f"{name}_{tag}_{idx:03d}"
            save_wav(out_dir / f"{stem}.wav", generated_wav, cfg=cfg)
            save_spectrogram_png(out_dir / f"{stem}.png", generated_mel, cfg=cfg, title=f"{IDX_TO_LABEL[label]} VQ-VAE {tag}")
            print(f"  saved {stem}.wav / {stem}.png from {Path(path).name}")


def parse_args():
    p = argparse.ArgumentParser(description="Synthesize audio with Conditional VQ-VAE")
    p.add_argument("--checkpoint", default="outputs/generation/vqvae_final/checkpoints/best_model.pt", type=str)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--out_dir", default="outputs/generation/vqvae_final/samples", type=str)
    p.add_argument("--emotion", default="all", type=str, help="all, Happy/Angry/Sad/Relaxed, or 0..3")
    p.add_argument("--num_samples", default=5, type=int)
    p.add_argument("--mode", choices=["reconstruct", "vary_codes", "sample_codes"], default="vary_codes")
    p.add_argument("--replace_prob", default=0.10, type=float)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--griffin_lim_iters", default=32, type=int)
    return p.parse_args()


if __name__ == "__main__":
    synthesize(parse_args())
