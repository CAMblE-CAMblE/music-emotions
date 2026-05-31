import os
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

SAMPLE_RATE = 22050
WINDOW_SEC = 5
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC
TARGET_TRACK_SEC = 45

# минимальная и максимальная длина одного отрезка внутри трека (в секундах)
MIN_SEGMENT_SEC = 8
MAX_SEGMENT_SEC = 20


def load_audio_files(data_dir):
    # собираем все файлы по классам
    class_dirs = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
    files_by_class = {0: [], 1: [], 2: [], 3: []}

    for folder, label in class_dirs.items():
        folder_path = os.path.join(data_dir, folder)
        for fname in os.listdir(folder_path):
            if fname.endswith(".mp3"):
                files_by_class[label].append(os.path.join(folder_path, fname))

    return files_by_class


def load_and_resample(path):
    waveform, sr = torchaudio.load(path)
    # моно
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # ресемплинг если нужно
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0)  # (samples,)


def build_synthetic_track(files_by_class):
    """
    склеиваем несколько отрезков из разных классов в один трек ~45 сек.
    длина каждого отрезка случайная, чтобы стыки не совпадали с границами окон.
    возвращает: waveform (samples,), список (start_sample, end_sample, label)
    """
    total_samples = TARGET_TRACK_SEC * SAMPLE_RATE
    segments = []  # (waveform, label)
    accumulated = 0

    prev_label = None
    while accumulated < total_samples:
        # выбираем класс, отличный от предыдущего (чтобы переходы были)
        label = random.randint(0, 3)
        if label == prev_label:
            label = (label + 1) % 4
        prev_label = label

        seg_sec = random.uniform(MIN_SEGMENT_SEC, MAX_SEGMENT_SEC)
        seg_samples = int(seg_sec * SAMPLE_RATE)

        # берём случайный файл этого класса
        path = random.choice(files_by_class[label])
        audio = load_and_resample(path)

        # вырезаем случайный кусок нужной длины
        if len(audio) <= seg_samples:
            # файл короче нужного — берём всё что есть
            chunk = audio
        else:
            start = random.randint(0, len(audio) - seg_samples)
            chunk = audio[start:start + seg_samples]

        segments.append((chunk, label))
        accumulated += len(chunk)

    # склеиваем
    waveform = torch.cat([s[0] for s in segments])

    # считаем границы отрезков в семплах
    boundaries = []
    pos = 0
    for chunk, label in segments:
        boundaries.append((pos, pos + len(chunk), label))
        pos += len(chunk)

    return waveform, boundaries


def make_window_labels(waveform_len, boundaries):
    """
    нарезаем трек на окна по 5 сек и для каждого окна ставим метку:
    1 - если стык попадает внутрь окна или ровно на его левую границу
    0 - иначе
    """
    labels = []
    window_start = 0

    while window_start + WINDOW_SAMPLES <= waveform_len:
        window_end = window_start + WINDOW_SAMPLES
        is_transition = 0

        for i in range(1, len(boundaries)):
            styk = boundaries[i][0]  # начало нового отрезка = момент стыка

            # стык внутри окна
            if window_start < styk < window_end:
                is_transition = 1
                break

            # стык ровно на левой границе окна - метим это окно
            if styk == window_start:
                is_transition = 1
                break

        labels.append(is_transition)
        window_start += WINDOW_SAMPLES

    return labels


def compute_stft(waveform_chunk):
    """
    считаем STFT для одного окна, возвращаем (freq_bins, time_frames) в дБ
    """
    n_fft = 1024
    hop_length = 512
    stft = torch.stft(
        waveform_chunk,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True
    )
    magnitude = stft.abs()
    # переводим в дБ, клипаем чтобы не было -inf
    db = 20 * torch.log10(magnitude.clamp(min=1e-5))
    return db  # (freq_bins, time_frames)


class TransitionDataset(Dataset):
    def __init__(self, data_dir, num_tracks=500, seed=42):
        random.seed(seed)
        torch.manual_seed(seed)

        self.files_by_class = load_audio_files(data_dir)
        self.windows = []   # список (stft_tensor, label)

        print(f"генерируем {num_tracks} синтетических треков...")
        for i in range(num_tracks):
            if i % 20 == 0:
                print(f"  трек {i}/{num_tracks}")

            waveform, boundaries = build_synthetic_track(self.files_by_class)
            labels = make_window_labels(len(waveform), boundaries)

            window_start = 0
            for label in labels:
                chunk = waveform[window_start:window_start + WINDOW_SAMPLES]
                stft = compute_stft(chunk)
                # нормализуем каждое окно отдельно
                stft = (stft - stft.mean()) / (stft.std() + 1e-8)
                # добавляем канал для CNN: (1, freq, time)
                self.windows.append((stft.unsqueeze(0), torch.tensor(label, dtype=torch.float32)))
                window_start += WINDOW_SAMPLES

        n_pos = sum(1 for _, l in self.windows if l == 1)
        n_neg = sum(1 for _, l in self.windows if l == 0)
        print(f"всего окон: {len(self.windows)}, переходов: {n_pos}, нет переходов: {n_neg}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        stft, label = self.windows[idx]
        return stft, torch.tensor(label, dtype=torch.float32)
