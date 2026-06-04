# Conditional U-Net AE variation commands

Новый режим использует уже обученный Conditional U-Net Autoencoder и не требует переобучения.

Базовая схема с ослаблением skip connections:

```text
real mel -> encoder -> bottleneck -> bottleneck + sigma * noise -> decoder with skip_scale * skips -> new mel -> Griffin-Lim -> wav
```

## Первый рекомендуемый эксперимент со слабыми skips

PowerShell, одна строка:

```powershell
python src/generation/synthesize_unet_variation.py --checkpoint outputs/generation/unet_ae_final/checkpoints/best_model.pt --emotion Happy --sigma 0.1 --skip_scale 0.3 --num_samples 5 --split test --device auto
```

## Если изменения все еще не слышны

Отключить skips полностью:

```powershell
python src/generation/synthesize_unet_variation.py --checkpoint outputs/generation/unet_ae_final/checkpoints/best_model.pt --emotion Happy --sigma 0.1 --skip_scale 0.0 --num_samples 5 --split test --device auto
```

## Более мягкий вариант

```powershell
python src/generation/synthesize_unet_variation.py --checkpoint outputs/generation/unet_ae_final/checkpoints/best_model.pt --emotion Happy --sigma 0.05 --skip_scale 0.3 --num_samples 5 --split test --device auto
```

## Стресс-тест

```powershell
python src/generation/synthesize_unet_variation.py --checkpoint outputs/generation/unet_ae_final/checkpoints/best_model.pt --emotion Happy --sigma 0.2 --skip_scale 0.0 --num_samples 5 --split test --device auto
```

## Генерация по всем эмоциям

```powershell
python src/generation/synthesize_unet_variation.py --checkpoint outputs/generation/unet_ae_final/checkpoints/best_model.pt --emotion all --sigma 0.1 --skip_scale 0.3 --num_samples 5 --split test --device auto
```

## Output

Файлы сохраняются в:

```text
outputs/generation/variation/
```

Примеры имен:

```text
happy_sigma100_skip030_000.wav
happy_sigma100_skip030_000.png
happy_sigma100_skip000_000.wav
happy_sigma100_skip000_000.png
```

## Интерпретация

Если `skip_scale=1.0`, результат почти совпадает с оригиналом, потому что U-Net skips передают детальную структуру исходного mel.

Если `skip_scale=0.3`, часть локальной структуры ослабляется, bottleneck noise должен стать заметнее.

Если `skip_scale=0.0`, decoder работает без skip features. Это даст максимальное отличие, но может резко ухудшить качество, потому что модель обучалась со skips.
