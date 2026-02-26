"""
LandmarkTransformerV4 — exact replica of the training architecture.
Matches the checkpoint produced by the Colab training notebook (parameters.md).
"""

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Weighted-sum pooling over the time dimension using a learned score."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            h:    (B, T, d_model)
            mask: (B, T) — 1 for valid frames, 0 for padding
        Returns:
            pooled: (B, d_model)
            w:      (B, T) attention weights
        """
        logits = self.score(h.float()).squeeze(-1)          # (B, T)
        logits = logits.masked_fill(mask <= 0, -1e4)
        w = torch.softmax(logits, dim=1).to(h.dtype)       # (B, T)
        pooled = (h * w.unsqueeze(-1)).sum(dim=1)           # (B, d_model)
        return pooled, w


class LandmarkTransformerV4(nn.Module):
    """
    Transformer-based sign language classifier.

    Architecture (matches training checkpoint):
        inp   = Linear(input_dim → d_model) + LayerNorm + Dropout
        encoder = TransformerEncoder (norm_first=True, GELU, batch_first=True)
        pool  = AttentionPool
        head  = LayerNorm + Dropout + Linear(d_model → num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.inp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.pool = AttentionPool(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, input_dim) landmark features
            mask: (B, T) 1=valid frame, 0=padding
        Returns:
            logits: (B, num_classes)
        """
        h = self.inp(x)
        key_pad = (mask <= 0)                               # True = ignore
        h = self.encoder(h, src_key_padding_mask=key_pad)
        pooled, _ = self.pool(h, mask)
        return self.head(pooled)
