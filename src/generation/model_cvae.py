from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalVAE(nn.Module):
    """Conditional VAE for normalized log-mel inputs.

    Expected input shape: (B, 1, 128, 432). Labels are emotion indices 0..3.
    The encoder receives label condition maps, and the decoder receives a learned
    label embedding concatenated with z.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 4,
        condition_dim: int = 16,
        n_mels: int = 128,
        frames: int = 432,
    ):
        super().__init__()
        if n_mels % 16 != 0 or frames % 16 != 0:
            raise ValueError("n_mels and frames must be divisible by 16 for this minimal CVAE")

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.n_mels = n_mels
        self.frames = frames
        self.h4 = n_mels // 16
        self.w4 = frames // 16
        self.flat_dim = 256 * self.h4 * self.w4

        self.encoder = nn.Sequential(
            ConvBlock(1 + num_classes, 32),   # (B, 32, 64, 216)
            ConvBlock(32, 64),                # (B, 64, 32, 108)
            ConvBlock(64, 128),               # (B, 128, 16, 54)
            ConvBlock(128, 256),              # (B, 256, 8, 27)
        )
        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 512),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.label_embedding = nn.Embedding(num_classes, condition_dim)
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.flat_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            DeconvBlock(256, 128),             # (B, 128, 16, 54)
            DeconvBlock(128, 64),              # (B, 64, 32, 108)
            DeconvBlock(64, 32),               # (B, 32, 64, 216)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),                      # normalized log-mel in [0, 1]
        )

    def _condition_maps(self, labels: torch.Tensor, height: int, width: int) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        return one_hot[:, :, None, None].expand(-1, -1, height, width)

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self._condition_maps(labels, x.size(2), x.size(3)).to(x.device)
        h = self.encoder(torch.cat([x, cond], dim=1))
        h = self.fc_enc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels.to(z.device))
        z_cond = torch.cat([z, label_emb], dim=1)
        h = self.fc_dec(z_cond)
        h = h.view(z.size(0), 256, self.h4, self.w4)
        return self.decoder(h)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, labels: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        labels = labels.to(device)
        z = torch.randn(labels.size(0), self.latent_dim, device=device)
        return self.decode(z, labels)
