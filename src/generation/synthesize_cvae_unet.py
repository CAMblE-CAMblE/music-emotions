from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.generation import IDX_TO_LABEL, LABEL_TO_IDX
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


def parse_emotion(value: str) -> list[int]:
    if value.lower() == "all":
        return [0, 1, 2, 3]
    if value.isdigit():
        idx = int(value)
        if idx not in IDX_TO_LABEL:
            raise ValueError("Emotion index must be 0..3")
        return [idx]
    normalized = value.strip().capitalize()
    if normalized not in LABEL_TO_IDX:
        valid = ", ".join(["all", "0", "1", "2", "3"] + list(LABEL_TO_IDX))
        raise ValueError(f"Unknown emotion {value!r}. Valid values: {valid}")
    return [LABEL_TO_IDX[normalized]]


def load_model(checkpoint: str | Path, device: str, requested_model_type: str) -> tuple[torch.nn.Module, AudioConfig, str]:
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
        ckpt_args = ckpt.get("args", {})
        model = ConditionalUNetAutoencoder(
            num_classes=ckpt.get("num_classes", 4),
            condition_dim=ckpt.get("condition_dim", ckpt_args.get("condition_dim", 32)),
            n_mels=cfg.n_mels,
            frames=cfg.target_frames,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type={model_type!r}; expected cvae or unet_ae")

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, model_type


def select_source_examples(data_dir: str | Path, labels: list[int], split: str, num_samples: int, seed: int) -> dict[int, list[tuple[str, int]]]:
    data_dir = validate_generation_data_dir(data_dir)
    base = GenerationMelDataset(data_dir, split="scan", seed=seed)
    train_files, val_files, test_files = split_file_list(base.samples, seed=seed)
    if split == "train":
        pool = train_files
    elif split == "val":
        pool = val_files
    elif split == "test":
        pool = test_files
    else:
        raise ValueError(f"Unknown split={split!r}")

    rng = random.Random(seed)
    selected: dict[int, list[tuple[str, int]]] = {}
    for label in labels:
        items = [item for item in pool if item[1] == label]
        if len(items) < num_samples:
            raise RuntimeError(
                f"Not enough {IDX_TO_LABEL[label]} examples in {split} split: "
                f"need {num_samples}, found {len(items)}"
            )
        rng.shuffle(items)
        selected[label] = items[:num_samples]
    return selected


@torch.no_grad()
def synthesize(args) -> None:
    device = resolve_device(args.device)
    model, cfg, model_type = load_model(args.checkpoint, device, args.model_type)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = parse_emotion(args.emotion)
    print(f"Generating to {out_dir}")
    print(f"Model type: {model_type}")

    if model_type == "cvae":
        for label in labels:
            name = IDX_TO_LABEL[label].lower()
            batch_labels = torch.full((args.num_samples,), label, dtype=torch.long, device=device)
            samples = model.sample(batch_labels, device=device).cpu()

            for i, mel in enumerate(samples):
                stem = f"{name}_{i:03d}"
                wav = logmel_to_wav_griffin_lim(mel, cfg=cfg, n_iter=args.griffin_lim_iters)
                save_wav(out_dir / f"{stem}.wav", wav, cfg=cfg)
                save_spectrogram_png(out_dir / f"{stem}.png", mel, cfg=cfg, title=f"{IDX_TO_LABEL[label]} generated")
                print(f"  saved {stem}.wav / {stem}.png")
        return

    selected = select_source_examples(args.data_dir, labels, args.split, args.num_samples, args.seed)
    for label in labels:
        name = IDX_TO_LABEL[label].lower()
        for i, (path, _) in enumerate(selected[label]):
            stem = f"{name}_{i:03d}"
            wav = load_audio(path, cfg=cfg, crop_mode="center")
            mel = wav_to_logmel(wav, cfg)
            mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
            label_tensor = torch.tensor([label], dtype=torch.long, device=device)
            recon = model(mel_tensor, label_tensor)[0, 0].detach().cpu()
            generated_wav = logmel_to_wav_griffin_lim(recon, cfg=cfg, n_iter=args.griffin_lim_iters)
            save_wav(out_dir / f"{stem}.wav", generated_wav, cfg=cfg)
            save_spectrogram_png(out_dir / f"{stem}.png", recon, cfg=cfg, title=f"{IDX_TO_LABEL[label]} U-Net reconstruction")
            print(f"  saved {stem}.wav / {stem}.png from {Path(path).name}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate emotion-conditioned audio samples from CVAE or U-Net AE")
    p.add_argument("--checkpoint", default="outputs/generation/unet_ae_final/checkpoints/best_model.pt", type=str)
    p.add_argument("--model_type", choices=["auto", "cvae", "unet_ae"], default="auto", type=str)
    p.add_argument("--out_dir", default="outputs/generation/unet_ae_final/samples", type=str)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR, type=str)
    p.add_argument("--emotion", default="all", type=str, help="all, Happy/Angry/Sad/Relaxed, or 0..3")
    p.add_argument("--num_samples", default=5, type=int)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--griffin_lim_iters", default=32, type=int)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


if __name__ == "__main__":
    synthesize(parse_args())
