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
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
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
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 128, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: (B, C, H, W). Quantization is performed over channel vectors.
        z = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z = z.view(-1, self.embedding_dim)

        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat_z @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).type(flat_z.dtype)
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view_as(z)

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        quantized = z + (quantized - z).detach()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        indices = indices.view(z_e.size(0), z_e.size(2), z_e.size(3))
        return quantized, vq_loss, indices, perplexity

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        z = self.embedding(indices)
        return z.permute(0, 3, 1, 2).contiguous()


class ConditionalVQVAE(nn.Module):
    """Conditional VQ-VAE for normalized log-mel spectrograms.

    Input:  (B, 1, 128, 432)
    Labels: (B,)
    Encoder output / code grid: (B, embedding_dim, 8, 27)
    Output: (B, 1, 128, 432), normalized to [0, 1]
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        num_classes: int = 4,
        condition_dim: int = 32,
        n_mels: int = 128,
        frames: int = 432,
    ):
        super().__init__()
        if n_mels % 16 != 0 or frames % 16 != 0:
            raise ValueError("n_mels and frames must be divisible by 16")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.n_mels = n_mels
        self.frames = frames
        self.h4 = n_mels // 16
        self.w4 = frames // 16

        self.encoder = nn.Sequential(
            ConvBlock(1 + num_classes, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, embedding_dim),
        )
        self.pre_vq = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.label_embedding = nn.Embedding(num_classes, condition_dim)

        self.decoder = nn.Sequential(
            DeconvBlock(embedding_dim + condition_dim, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def _condition_maps(self, labels: torch.Tensor, height: int, width: int) -> torch.Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        return one_hot[:, :, None, None].expand(-1, -1, height, width)

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self._condition_maps(labels.to(x.device), x.size(2), x.size(3)).to(x.device)
        z_e = self.encoder(torch.cat([x, cond], dim=1))
        return self.pre_vq(z_e)

    def quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.quantizer(z_e)

    def decode(self, z_q: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_emb = self.label_embedding(labels.to(z_q.device))
        cond = label_emb[:, :, None, None].expand(-1, -1, z_q.size(2), z_q.size(3))
        return self.decoder(torch.cat([z_q, cond], dim=1))

    def decode_indices(self, indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.decode_indices(indices.to(labels.device))
        return self.decode(z_q, labels)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encode(x, labels)
        z_q, vq_loss, indices, perplexity = self.quantize(z_e)
        recon = self.decode(z_q, labels)
        return recon, vq_loss, indices, perplexity

    @torch.no_grad()
    def sample_random_codes(self, labels: torch.Tensor, device: torch.device | str | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        labels = labels.to(device)
        indices = torch.randint(
            low=0,
            high=self.num_embeddings,
            size=(labels.size(0), self.h4, self.w4),
            device=device,
        )
        return self.decode_indices(indices, labels)
