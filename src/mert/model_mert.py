import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MERT_MODEL_ID = "m-a-p/MERT-v1-95M"
MERT_SR = 24000  # MERT-v1 обучен на 24 КГц
EMO_BEST_LAYER = 5  # лучший слой для emotion recognition по документации MERT


# Классификатор поверх MERT-эмбеддингов

class EmotionHead(nn.Module):
    """
    Легкий классификатор: Linear → ReLU → Dropout → Linear.
    Вход: (batch, 768) — эмбеддинги из одного слоя MERT
    Выход: (batch, num_classes)
    """

    def __init__(self, input_dim: int = 768,
                 hidden_dim: int = 256,
                 num_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MERTEmotionClassifier(nn.Module):
    """
    MERT-v1-95M (заморожен) + обучаемый EmotionHead.

    Параметры:
        layer_idx   : какой скрытый слой MERT использовать как эмбеддинг
                      (0-12 для MERT-v1-95M; 5 — оптимально для EMO)
        num_classes : число классов эмоций
        dropout     : dropout в классификаторе
    """

    def __init__(self, layer_idx: int = EMO_BEST_LAYER,
                 num_classes: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.layer_idx = layer_idx

        print(f"Загружаем {MERT_MODEL_ID} ...")
        self.mert = AutoModel.from_pretrained(
            MERT_MODEL_ID,
            trust_remote_code=True,
            output_hidden_states=True,  # нужны все скрытые состояния
        )

        # Замораживаем все параметры MERT
        for param in self.mert.parameters():
            param.requires_grad = False
        print(
            f"  MERT заморожен. Параметров: {sum(p.numel() for p in self.mert.parameters()):,}")

        # Обучаемый классификатор
        hidden_dim = self.mert.config.hidden_size  # 768 для 95M
        self.head = EmotionHead(hidden_dim, num_classes=num_classes, dropout=dropout)
        print(
            f"  Классификатор: {sum(p.numel() for p in self.head.parameters()):,} параметров")
        print(f"  Слой MERT для эмбеддингов: {layer_idx}")

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Аргументы:
            input_values  : (batch, time)  — нормализованный waveform 24 КГц
            attention_mask: (batch, time)  — маска паддинга (опционально)

        Возвращает:
            logits: (batch, num_classes)
        """
        with torch.no_grad():
            outputs = self.mert(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # hidden_states: кортеж из 13 тензоров (batch, time_steps, 768)
        # берем нужный слой
        hidden = outputs.hidden_states[self.layer_idx]  # (B, T, 768)

        # mean pooling по временной оси -> (B, 768)
        if attention_mask is not None:
            # учитываем паддинг: маска для hidden states короче waveform
            # приводим маску к длине hidden states
            ratio = hidden.size(1) / input_values.size(1)
            h_mask = F.interpolate(
                attention_mask.float().unsqueeze(1),
                size=hidden.size(1),
                mode="nearest",
            ).squeeze(1)  # (B, T_hidden)
            h_mask = h_mask.unsqueeze(-1)  # (B, T_hidden, 1)
            emb = (hidden * h_mask).sum(1) / h_mask.sum(1).clamp(min=1e-9)
        else:
            emb = hidden.mean(dim=1)  # (B, 768)

        return self.head(emb)

    def predict_proba(self, input_values: torch.Tensor,
                      attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return F.softmax(self.forward(input_values, attention_mask), dim=-1)


def get_processor() -> Wav2Vec2FeatureExtractor:
    """
    Загружает процессор MERT для нормализации waveform.
    Используется в датасете и при инференсе.
    """
    return Wav2Vec2FeatureExtractor.from_pretrained(
        MERT_MODEL_ID, trust_remote_code=True
    )


def get_mert_model(layer_idx: int = EMO_BEST_LAYER,
                   num_classes: int = 4,
                   dropout: float = 0.3,
                   device: str | None = None) -> MERTEmotionClassifier:
    """
    Создает модель и перемещает на устройство.

    Пример:
        model = get_mert_model()
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MERTEmotionClassifier(layer_idx, num_classes, dropout).to(device)
    print(f"  Устройство: {device}")
    return model


if __name__ == "__main__":
    # Быстрая проверка: создаем модель и прогоняем фейковый батч
    model = get_mert_model()
    # 2 клипа по 3 секунды @ 24 КГц
    dummy = torch.randn(2, 3 * MERT_SR)
    out = model(dummy)
    print(f"\nВход: {dummy.shape} → Выход: {out.shape}")
    assert out.shape == (2, 4), "Ошибка формы выхода"
    print("OK")
