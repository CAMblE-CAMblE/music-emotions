import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    сжимает спектрограмму одного окна в вектор признаков
    вход: (batch, 1, freq_bins, time_frames)
    выход: (batch, 128)
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))  # фиксируем размер до (4,4)

        self.relu = nn.ReLU()
        self.fc   = nn.Linear(128 * 4 * 4, 128)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc(x))
        return x  # (batch, 128)


class CNNLSTMTransitionDetector(nn.Module):
    """
    принимает последовательность окон трека, возвращает вероятность перехода для каждого окна.
    вход: (batch, seq_len, 1, freq_bins, time_frames)
    выход: (batch, seq_len)
    """
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = CNNEncoder()
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1, freq, time)
        batch, seq_len, c, freq, time = x.shape

        # прогоняем каждое окно через CNN
        x = x.view(batch * seq_len, c, freq, time)
        features = self.cnn(x)                        # (batch*seq_len, 128)
        features = features.view(batch, seq_len, 128) # (batch, seq_len, 128)

        # прогоняем последовательность через LSTM
        lstm_out, _ = self.lstm(features)             # (batch, seq_len, hidden)
        lstm_out = self.dropout(lstm_out)

        # для каждого окна предсказываем вероятность перехода
        logits = self.fc_out(lstm_out).squeeze(-1)    # (batch, seq_len)
        return logits
