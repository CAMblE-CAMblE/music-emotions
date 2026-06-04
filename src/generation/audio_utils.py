from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AudioConfig:
    sr: int = 22050
    duration: float = 5.0
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    target_frames: int = 432
    fmin: float = 0.0
    fmax: float | None = None
    top_db: float = 80.0

    @property
    def target_samples(self) -> int:
        return int(round(self.sr * self.duration))

    def to_dict(self) -> dict:
        return asdict(self)


def load_audio(
    path: str | Path,
    cfg: AudioConfig | None = None,
    crop_mode: str = "center",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load mono audio, then crop or pad to cfg.duration seconds."""
    cfg = cfg or AudioConfig()
    y, _ = librosa.load(path, sr=cfg.sr, mono=True)
    target = cfg.target_samples

    if len(y) >= target:
        if crop_mode == "random":
            if rng is None:
                rng = np.random.default_rng()
            start = int(rng.integers(0, len(y) - target + 1))
        elif crop_mode == "center":
            start = max(0, (len(y) - target) // 2)
        else:
            raise ValueError(f"Unknown crop_mode={crop_mode!r}; expected random or center")
        y = y[start:start + target]
    else:
        y = np.pad(y, (0, target - len(y)), mode="constant")

    return y.astype(np.float32, copy=False)


def _fix_frames(mel: np.ndarray, target_frames: int) -> np.ndarray:
    if mel.shape[1] > target_frames:
        return mel[:, :target_frames]
    if mel.shape[1] < target_frames:
        return np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])), mode="constant")
    return mel


def wav_to_logmel(wav: np.ndarray, cfg: AudioConfig | None = None) -> np.ndarray:
    """Convert waveform to normalized log-mel spectrogram in [0, 1]."""
    cfg = cfg or AudioConfig()
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
        center=True,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=cfg.top_db)
    mel_norm = (mel_db + cfg.top_db) / cfg.top_db
    mel_norm = np.clip(mel_norm, 0.0, 1.0)
    mel_norm = _fix_frames(mel_norm, cfg.target_frames)
    return mel_norm.astype(np.float32, copy=False)


def logmel_to_wav_griffin_lim(
    logmel: np.ndarray | torch.Tensor,
    cfg: AudioConfig | None = None,
    n_iter: int = 32,
) -> np.ndarray:
    """Invert normalized log-mel to waveform using Griffin-Lim.

    This is a baseline synthesis path. It is listenable enough for inspection, but
    it will sound phasey/metallic compared with a neural vocoder.
    """
    cfg = cfg or AudioConfig()
    if isinstance(logmel, torch.Tensor):
        logmel = logmel.detach().cpu().float().numpy()
    logmel = np.asarray(logmel, dtype=np.float32)
    if logmel.ndim == 3:
        logmel = logmel.squeeze(0)
    logmel = _fix_frames(logmel, cfg.target_frames)
    logmel = np.clip(logmel, 0.0, 1.0)

    mel_db = logmel * cfg.top_db - cfg.top_db
    mel_power = librosa.db_to_power(mel_db)
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        n_iter=n_iter,
    )
    if len(wav) < cfg.target_samples:
        wav = np.pad(wav, (0, cfg.target_samples - len(wav)), mode="constant")
    else:
        wav = wav[:cfg.target_samples]

    if np.max(np.abs(wav)) > 1e-8:
        wav = wav / np.max(np.abs(wav)) * 0.95
    return wav.astype(np.float32, copy=False)


def save_wav(path: str | Path, wav: np.ndarray, cfg: AudioConfig | None = None) -> None:
    cfg = cfg or AudioConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wav, cfg.sr)


def save_spectrogram_png(
    path: str | Path,
    mel: np.ndarray | torch.Tensor,
    cfg: AudioConfig | None = None,
    title: str | None = None,
) -> None:
    cfg = cfg or AudioConfig()
    if isinstance(mel, torch.Tensor):
        mel = mel.detach().cpu().float().numpy()
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim == 3:
        mel = mel.squeeze(0)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    librosa.display.specshow(
        mel,
        sr=cfg.sr,
        hop_length=cfg.hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
    )
    plt.colorbar(format="%.2f", label="normalized log-mel")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
