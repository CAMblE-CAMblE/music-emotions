# Conditional U-Net AE commands

Production-like генеративный пайплайн использует Conditional U-Net Autoencoder как основную модель.

Путь результата:

```text
outputs/generation/unet_ae_final/
```

## 1. Обучение, 50 эпох

PowerShell, одна строка:

```powershell
python src/generation/train_unet_ae.py --out_dir outputs/generation/unet_ae_final --epochs 50 --batch_size 16 --lr 0.001 --loss_type l1 --device auto --workers 0 --train_segments_per_file 4
```

Что сохранится:

- `outputs/generation/unet_ae_final/checkpoints/best_model.pt`
- `outputs/generation/unet_ae_final/checkpoints/last_model.pt`
- `outputs/generation/unet_ae_final/plots/training_curves.png`
- `outputs/generation/unet_ae_final/plots/recon_examples.png`
- `outputs/generation/unet_ae_final/metrics.json`

## 2. Reconstruction diagnostics

Запускать после обучения:

```powershell
python src/generation/evaluate_reconstruction.py --checkpoint outputs/generation/unet_ae_final/checkpoints/best_model.pt --model_type unet_ae --out_dir outputs/generation/unet_ae_final/reconstruction_debug --examples_per_class 5 --num_comparisons 8 --batch_size 8 --workers 0 --device auto --mode reconstruct_only
```

Что сохранится:

- `outputs/generation/unet_ae_final/reconstruction_debug/real/`
- `outputs/generation/unet_ae_final/reconstruction_debug/reconstructed/`
- `outputs/generation/unet_ae_final/reconstruction_debug/comparisons/`
- `outputs/generation/unet_ae_final/reconstruction_debug/reconstruction_metrics.json`

В `reconstruction_metrics.json` считаются:

- MSE
- MAE
- Spectral Convergence
- Cosine Similarity overall mean/std
- Cosine Similarity per-class mean/std

## 3. Генерация WAV для прослушивания

Для U-Net AE генерация реализована как reconstruction-based synthesis:

```text
real audio -> log-mel -> Conditional U-Net AE -> reconstructed mel -> Griffin-Lim -> wav
```

PowerShell, одна строка:

```powershell
python src/generation/synthesize_cvae_unet.py --checkpoint outputs/generation/unet_ae_final/checkpoints/best_model.pt --model_type unet_ae --out_dir outputs/generation/unet_ae_final/samples --emotion all --num_samples 5 --split test --device auto
```

Что сохранится:

- `outputs/generation/unet_ae_final/samples/happy_000.wav`
- `outputs/generation/unet_ae_final/samples/happy_000.png`
- `outputs/generation/unet_ae_final/samples/angry_000.wav`
- `outputs/generation/unet_ae_final/samples/sad_000.wav`
- `outputs/generation/unet_ae_final/samples/relaxed_000.wav`

## 4. Evaluation generated audio

Запускать после генерации WAV:

```powershell
python src/generation/evaluate_generation.py --checkpoint outputs/generation/unet_ae_final/checkpoints/best_model.pt --model_type unet_ae --out_dir outputs/generation/unet_ae_final/evaluation --samples_dir outputs/generation/unet_ae_final/samples --split test --batch_size 8 --workers 0 --device auto
```

Что оценивается:

- reconstruction MSE / MAE / Spectral Convergence / Cosine Similarity;
- diversity по pairwise L1 между generated mel из WAV;
- MERT-based emotion accuracy, если доступен `outputs/mert_best/checkpoints/best_mert_model.pt`.

## 5. MOS subjective evaluation

Шаблон для ручной оценки:

```text
outputs/generation/mos_template.csv
```

Колонки:

```text
sample_path,target_emotion,quality_score_1_5,emotion_match_1_5,comment
```
