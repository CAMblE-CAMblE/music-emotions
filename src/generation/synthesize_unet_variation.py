from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation import IDX_TO_LABEL
from src.generation.audio_utils import (
    load_audio,
    logmel_to_wav_griffin_lim,
    save_spectrogram_png,
    save_wav,
    wav_to_logmel,
)
from src.generation.dataset import DEFAULT_DATA_DIR
from src.generation.synthesize_cvae_unet import load_model, parse_emotion, resolve_device, select_source_examples

SUPPORTED_SIGMAS = (0.01, 0.05, 0.1, 0.2)


def sigma_tag(sigma: float) -> str:
    return f"sigma{int(round(sigma * 1000)):03d}"


def skip_tag(skip_scale: float) -> str:
    return f"skip{int(round(skip_scale * 100)):03d}"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def synthesize_variations(args) -> None:
    if args.sigma not in SUPPORTED_SIGMAS:
        supported = ", ".join(str(v) for v in SUPPORTED_SIGMAS)
        raise ValueError(f"Unsupported --sigma {args.sigma}. Supported values: {supported}")

    set_seed(args.seed)
    device = resolve_device(args.device)
    model, cfg, model_type = load_model(args.checkpoint, device, requested_model_type="unet_ae")
    if model_type != "unet_ae":
        raise RuntimeError(f"Variation synthesis requires a U-Net AE checkpoint, got model_type={model_type!r}")

    labels = parse_emotion(args.emotion)
    selected = select_source_examples(args.data_dir, labels, args.split, args.num_samples, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = sigma_tag(args.sigma)
    skip = skip_tag(args.skip_scale)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {out_dir}")
    print(f"Emotion: {args.emotion}")
    print(f"Split: {args.split}")
    print(f"Sigma: {args.sigma}")
    print(f"Skip scale: {args.skip_scale}")
    print("Mode: latent perturbation generation with scaled U-Net skips")

    for label in labels:
        emotion_name = IDX_TO_LABEL[label]
        prefix = emotion_name.lower()
        for idx, (path, _) in enumerate(selected[label]):
            wav = load_audio(path, cfg=cfg, crop_mode="center")
            mel = wav_to_logmel(wav, cfg)
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            label_tensor = torch.tensor([label], dtype=torch.long, device=device)

            bottleneck, skips = model.encode_with_skips(mel_tensor)
            noise = args.sigma * torch.randn_like(bottleneck)
            perturbed_bottleneck = bottleneck + noise
            scaled_skips = tuple(skip_feature * args.skip_scale for skip_feature in skips)
            generated_mel = model.decode(perturbed_bottleneck, label_tensor, skips=scaled_skips)[0, 0].detach().cpu()
            generated_wav = logmel_to_wav_griffin_lim(generated_mel, cfg=cfg, n_iter=args.griffin_lim_iters)

            stem = f"{prefix}_{tag}_{skip}_{idx:03d}"
            save_wav(out_dir / f"{stem}.wav", generated_wav, cfg=cfg)
            save_spectrogram_png(
                out_dir / f"{stem}.png",
                generated_mel,
                cfg=cfg,
                title=f"{emotion_name} variation | sigma={args.sigma} | skip_scale={args.skip_scale}",
            )
            print(f"  saved {stem}.wav / {stem}.png from {Path(path).name}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate U-Net AE music variations by perturbing bottleneck features")
    p.add_argument("--checkpoint", default="outputs/generation/unet_ae_final/checkpoints/best_model.pt", type=str)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--emotion", default="Happy", type=str, help="all, Happy/Angry/Sad/Relaxed, or 0..3")
    p.add_argument("--num_samples", default=5, type=int)
    p.add_argument("--sigma", choices=SUPPORTED_SIGMAS, default=0.05, type=float)
    p.add_argument("--skip_scale", default=1.0, type=float, help="Scale U-Net skip connections during decoding. Try 0.3 or 0.0 for stronger variation.")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--out_dir", default="outputs/generation/variation", type=str)
    p.add_argument("--griffin_lim_iters", default=32, type=int)
    return p.parse_args()


if __name__ == "__main__":
    synthesize_variations(parse_args())
