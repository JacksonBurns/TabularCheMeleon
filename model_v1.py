"""Reconstruct the ORIGINAL v1 from-scratch ICL model (continuous-label, scalar
head, present_embed) so the icl_logs/fixeddim_icl.pt checkpoint can be loaded and
evaluated on the polaris benchmarks for a clean head-to-head comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class LegacyConfig:
    embed_dim: int = 2048
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_context: int = 512
    task: str = "regression"
    n_classes: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    def to_dict(self):
        return asdict(self)


def _sinusoidal_pe(max_len, d_model):
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class LegacyICLTransformer(nn.Module):
    def __init__(self, cfg: LegacyConfig):
        super().__init__()
        self.cfg = cfg
        D, dm = cfg.embed_dim, cfg.d_model
        self.feature_proj = nn.Sequential(nn.Linear(D, dm), nn.GELU(), nn.Linear(dm, dm))
        self.feature_norm = nn.LayerNorm(dm)
        self.label_proj = nn.Linear(1, dm)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        self.present_embed = nn.Embedding(2, dm)
        self.register_buffer("pe", _sinusoidal_pe(cfg.max_context, dm))
        enc = nn.TransformerEncoderLayer(
            d_model=dm, nhead=cfg.n_heads, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, dm), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(dm, 1))

    def forward(self, x, y, mask, key_padding_mask=None):
        B, N, D = x.shape
        dm = self.cfg.d_model
        h = self.feature_norm(self.feature_proj(x))
        label_part = self.label_proj(y)
        present = (~mask).long()
        present_e = self.present_embed(present)
        mask_e = self.mask_token.expand(B, N, -1)
        tok_label = torch.where(mask.unsqueeze(-1), mask_e, label_part)
        tokens = h + tok_label + present_e + self.pe[:N].unsqueeze(0)
        z = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.head(z)


def load_legacy(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = LegacyConfig(**ck["cfg"])
    m = LegacyICLTransformer(cfg).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m
