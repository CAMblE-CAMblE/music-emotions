# Conditional VQ-VAE commands

Экспериментальная ветка: `generation-vqvae`.

VQ-VAE pipeline:

```text
log-mel -> conditional encoder -> discrete codebook -> conditional decoder -> mel -> Griffin-Lim -> wav
```

## 1. Обучение, 50 эпох

PowerShell, одна строка:

```powershell
python src/generation/train_vqvae.py --out_dir outputs/generation/vqvae_final --epochs 50 --batch_size 16 --lr 0.001 --loss_type l1 --vq_weight 1.0 --num_embeddings 512 --embedding_dim 128 --commitment_cost 0.25 --device auto --workers 0 --train_segments_per_file 4
```

Что сохранится:

- `outputs/generation/vqvae_final/checkpoints/best_model.pt`
- `outputs/generation/vqvae_final/checkpoints/last_model.pt`
- `outputs/generation/vqvae_final/plots/training_curves.png`
- `outputs/generation/vqvae_final/plots/recon_examples.png`
- `outputs/generation/vqvae_final/metrics.json`

## 2. Reconstruction diagnostics

```powershell
python src/generation/evaluate_reconstruction.py --checkpoint outputs/generation/vqvae_final/checkpoints/best_model.pt --model_type vqvae --out_dir outputs/generation/vqvae_final/reconstruction_debug --examples_per_class 5 --num_comparisons 8 --batch_size 8 --workers 0 --device auto --mode reconstruct_only
```

Что смотреть:

- `outputs/generation/vqvae_final/reconstruction_debug/comparisons/`
- `outputs/generation/vqvae_final/reconstruction_debug/reconstructed/`
- `outputs/generation/vqvae_final/reconstruction_debug/reconstruction_metrics.json`

## 3. VQ-VAE variation generation

Заменяет часть discrete code indices случайными codebook entries.

Первый эксперимент:

```powershell
python src/generation/synthesize_vqvae.py --checkpoint outputs/generation/vqvae_final/checkpoints/best_model.pt --out_dir outputs/generation/vqvae_final/samples --emotion Happy --mode vary_codes --replace_prob 0.10 --num_samples 5 --split test --device auto
```

Если вариации слабые:

```powershell
python src/generation/synthesize_vqvae.py --checkpoint outputs/generation/vqvae_final/checkpoints/best_model.pt --out_dir outputs/generation/vqvae_final/samples --emotion Happy --mode vary_codes --replace_prob 0.25 --num_samples 5 --split test --device auto
```

По всем эмоциям:

```powershell
python src/generation/synthesize_vqvae.py --checkpoint outputs/generation/vqvae_final/checkpoints/best_model.pt --out_dir outputs/generation/vqvae_final/samples --emotion all --mode vary_codes --replace_prob 0.10 --num_samples 5 --split test --device auto
```

## 4. Random code sampling

Это наиболее близко к настоящей генерации без real mel, но без обученного prior может звучать шумно.

```powershell
python src/generation/synthesize_vqvae.py --checkpoint outputs/generation/vqvae_final/checkpoints/best_model.pt --out_dir outputs/generation/vqvae_final/random_samples --emotion all --mode sample_codes --num_samples 5 --device auto
```

## 5. Evaluation generated WAV

После генерации samples:

```powershell
python src/generation/evaluate_generation.py --checkpoint outputs/generation/vqvae_final/checkpoints/best_model.pt --model_type vqvae --out_dir outputs/generation/vqvae_final/evaluation --samples_dir outputs/generation/vqvae_final/samples --split test --batch_size 8 --workers 0 --device auto
```

Если MERT не запускается в окружении:

```powershell
python src/generation/evaluate_generation.py --checkpoint outputs/generation/vqvae_final/checkpoints/best_model.pt --model_type vqvae --out_dir outputs/generation/vqvae_final/evaluation --samples_dir outputs/generation/vqvae_final/samples --split test --batch_size 8 --workers 0 --device auto --skip_mert
```

## Рекомендации

- Начать с `replace_prob=0.10`.
- Если почти reconstruction, попробовать `replace_prob=0.25`.
- Если структура ломается, вернуться к `replace_prob=0.05` или `0.10`.
- `sample_codes` использовать как sanity-check: без prior это не обязано звучать музыкально.
