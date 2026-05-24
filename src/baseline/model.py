import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, pool: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionCNN(nn.Module):
    """
    Сверточная сеть для классификации эмоций.

    Параметры:
        num_classes : количество классов (по умолчанию 4)
        dropout     : вероятность dropout перед выходным слоем
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.5):
        super().__init__()

        # Сверточные блоки
        # Вход: (B, 1, 128, 128)
        self.features = nn.Sequential(
            ConvBlock(1, 32),  # → (B, 32,  64, 64)
            ConvBlock(32, 64),  # → (B, 64,  32, 32)
            ConvBlock(64, 128),  # → (B, 128, 16, 16)
            ConvBlock(128, 256),  # → (B, 256,  8,  8)
        )

        # Global Average Pooling: (B, 256, 8, 8) → (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # (B, 256, 8, 8)
        x = self.gap(x)  # (B, 256, 1, 1)
        x = self.classifier(x)  # (B, num_classes)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Возвращает вероятности через softmax (для инференса)."""
        return F.softmax(self.forward(x), dim=-1)


def count_parameters(model: nn.Module) -> int:
    """Количество обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(num_classes: int = 4, dropout: float = 0.5,
              device: str | None = None) -> EmotionCNN:
    """
    Создаёт и возвращает модель, перемещая на нужное устройство.
    Пример:
        model = get_model()
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmotionCNN(num_classes=num_classes, dropout=dropout).to(device)
    print(
        f"Модель: EmotionCNN | Параметры: {count_parameters(model):,} | Устройство: {device}")
    return model


if __name__ == "__main__":
    # Быстрая проверка формы
    model = get_model()
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(f"Входной тензор: {dummy.shape} → Выход: {out.shape}")
