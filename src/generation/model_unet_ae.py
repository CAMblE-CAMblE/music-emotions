from __future__ import annotations

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalUNetAutoencoder(nn.Module):
    """Conditional U-Net autoencoder for normalized log-mel spectrograms.

    Input:  (B, 1, 128, 432)
    Labels: (B,)
    Output: (B, 1, 128, 432), normalized to [0, 1]
    """

    def __init__(
        self,
        num_classes: int = 4,
        condition_dim: int = 32,
        n_mels: int = 128,
        frames: int = 432,
    ):
        super().__init__()
        if n_mels % 16 != 0 or frames % 16 != 0:
            raise ValueError("n_mels and frames must be divisible by 16")

        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.n_mels = n_mels
        self.frames = frames
        self.h4 = n_mels // 16
        self.w4 = frames // 16

        self.down1 = DownBlock(1, 32)     # (B, 32, 64, 216)
        self.down2 = DownBlock(32, 64)    # (B, 64, 32, 108)
        self.down3 = DownBlock(64, 128)   # (B, 128, 16, 54)
        self.down4 = DownBlock(128, 256)  # (B, 256, 8, 27)

        self.label_embedding = nn.Embedding(num_classes, condition_dim)

        self.up1 = UpBlock(256 + condition_dim, 128)  # -> (B, 128, 16, 54)
        self.up2 = UpBlock(128 + 128, 64)             # -> (B, 64, 32, 108)
        self.up3 = UpBlock(64 + 64, 32)               # -> (B, 32, 64, 216)
        self.up4 = UpBlock(32 + 32, 16)               # -> (B, 16, 128, 432)
        self.out = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _condition_maps(self, labels: torch.Tensor, height: int, width: int) -> torch.Tensor:
        emb = self.label_embedding(labels)
        return emb[:, :, None, None].expand(-1, -1, height, width)

    def encode_with_skips(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        s1 = self.down1(x)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        bottleneck = self.down4(s3)
        return bottleneck, (s1, s2, s3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, _ = self.encode_with_skips(x)
        return bottleneck

    def decode(
        self,
        bottleneck: torch.Tensor,
        labels: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if skips is None:
            batch = bottleneck.size(0)
            device = bottleneck.device
            dtype = bottleneck.dtype
            s1 = torch.zeros(batch, 32, self.n_mels // 2, self.frames // 2, device=device, dtype=dtype)
            s2 = torch.zeros(batch, 64, self.n_mels // 4, self.frames // 4, device=device, dtype=dtype)
            s3 = torch.zeros(batch, 128, self.n_mels // 8, self.frames // 8, device=device, dtype=dtype)
        else:
            s1, s2, s3 = skips

        cond = self._condition_maps(labels.to(bottleneck.device), bottleneck.size(2), bottleneck.size(3))
        x = torch.cat([bottleneck, cond.to(bottleneck.device)], dim=1)

        x = self.up1(x)
        x = self.up2(torch.cat([x, s3], dim=1))
        x = self.up3(torch.cat([x, s2], dim=1))
        x = self.up4(torch.cat([x, s1], dim=1))
        return self.out(x)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encode_with_skips(x)
        return self.decode(bottleneck, labels, skips=skips)


if __name__ == "__main__":
    model = ConditionalUNetAutoencoder()
    mel = torch.rand(2, 1, 128, 432)
    labels = torch.tensor([0, 3])
    out = model(mel, labels)
    print(mel.shape, out.shape)
