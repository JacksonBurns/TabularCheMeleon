"""Fixed-dim in-context-learning transformer for CheMeleon embeddings.

This is the model that exploits the *relaxed* assumptions. Compared to a
generic tabular FM (TabPFN/TabICL), which must handle an arbitrary number of
columns in an arbitrary order, this model knows the input is a fixed 2048-dim
vector in a fixed, meaningful order. Concretely:

  * No column/feature-index embeddings and no column-permutation invariance.
    A tabular FM builds each row's embedding by summing per-column terms
    (so it cannot tell column 3 from column 2045). Here the whole vector is
    projected once through `feature_proj` (Linear D -> d_model), so the model
    can learn per-dimension weights and exploit the known structure.
  * No categorical/numerical type dispatch - all inputs are continuous.
  * The context window is over *examples* (rows), exactly as in TabPFN/TabICL:
    each example is one token = feature projection + label embedding (or a
    learnable mask token for queries) + positional encoding.

Label encoding (`label_mode`):
  * 'bin'   : the (standardised) label is discretised to one of n_bins bins and
              read out through a LEARNED bin embedding. The head predicts a
              distribution over bins. This is TabPFN's design and the signal is
              rich enough for attention to condition on (a continuous scalar
              nudge collapses the model to a constant predictor - see scale
              study).
  * 'cont'  : continuous scalar label embedding + pinball/quantile head
              (kept for comparison).

Inference is pure in-context learning: feed labeled context molecules +
unlabeled query molecules in one forward pass, read the query predictions.
No per-task parameter updates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


@dataclass
class ModelConfig:
    embed_dim: int = 2048
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    dim_feedforward: int = 1536
    dropout: float = 0.1
    max_context: int = 512          # max examples in one context window
    task: str = "regression"        # 'regression' | 'classification'
    n_classes: int = 2
    label_mode: str = "bin"         # 'bin' | 'cont'
    feature_mode: str = "mlp"       # 'mlp' (2-layer GELU+LN) | 'linear' (single Linear)
    n_bins: int = 32                # bins for the label distribution head
    bin_range: float = 5.0          # labels standardised to ~N(0,1); bins span [-r, r]
    n_quantiles: int = 21           # for pinball/quantile ('cont' mode)
    loss: str = "pinball"           # 'mse' | 'pinball' (cont mode only)
    label_scale: float = 4.0        # scale of the continuous label nudge
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    def to_dict(self) -> dict:
        return asdict(self)


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (max_len, d_model)


class FixedDimICLTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        D, dm = cfg.embed_dim, cfg.d_model

        # THE key relaxation: one structured projection of the full ordered
        # vector, instead of an order-agnostic per-column sum.
        if cfg.feature_mode == "linear":
            # single linear projection (preserves linear structure for
            # in-context linear regression, as TabPFN's ~linear per-column
            # encoding does)
            self.feature_proj = nn.Linear(D, dm)
        else:  # 'mlp'
            self.feature_proj = nn.Sequential(
                nn.Linear(D, dm),
                nn.GELU(),
                nn.Linear(dm, dm),
            )
        self.feature_norm = nn.LayerNorm(dm)

        self.mask_token = nn.Parameter(torch.randn(1, 1, dm) * 0.02)

        if cfg.task == "regression" and cfg.label_mode == "bin":
            # bin centers in standardised label space
            edges = torch.linspace(-cfg.bin_range, cfg.bin_range, cfg.n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            self.register_buffer("bin_edges", edges)
            self.register_buffer("bin_centers", centers)
            self.label_embed = nn.Embedding(cfg.n_bins, dm)
            head_out = cfg.n_bins
        elif cfg.task == "regression":
            self.label_proj = nn.Linear(1, dm)
            # The feature token after LayerNorm has ~unit RMS per dim. The label
            # signal must be of COMPARABLE magnitude or attention is dominated
            # by features and the per-token label is effectively noise -> the
            # model cannot pair context molecules with their labels and collapses
            # to a constant predictor. Scale the continuous label nudge up to
            # unit RMS to match the feature token.
            self.label_scale = float(cfg.label_scale)
            if cfg.loss == "pinball":
                self.register_buffer("quantile_levels",
                                     torch.linspace(0.5 / cfg.n_quantiles,
                                                    1.0 - 0.5 / cfg.n_quantiles,
                                                    cfg.n_quantiles))
                head_out = cfg.n_quantiles
            else:
                head_out = 1
        else:  # classification
            self.label_embed = nn.Embedding(cfg.n_classes, dm)
            head_out = cfg.n_classes

        self.register_buffer("pe", _sinusoidal_pe(cfg.max_context, dm))

        enc = nn.TransformerEncoderLayer(
            d_model=dm, nhead=cfg.n_heads, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=cfg.n_layers,
                                             enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(dm),
            nn.Linear(dm, dm),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dm, head_out),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding) and m is getattr(self, "label_embed", None):
                # label embeddings must be UNIT RMS (like the LayerNorm'd feature
                # token) so attention can read the per-token label; the default
                # 0.02 init is ~50x too weak and causes mean-predictor collapse
                nn.init.normal_(m.weight, std=1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------ #
    def _bin_of(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B,N,1) standardised labels -> (B,N) bin indices."""
        idx = torch.searchsorted(self.bin_edges, y.squeeze(-1), right=True)
        return idx.clamp(0, self.cfg.n_bins - 1)

    def point_pred(self, out: torch.Tensor) -> torch.Tensor:
        """Map head output to a point prediction in standardised units."""
        if self.cfg.task == "regression" and self.cfg.label_mode == "bin":
            probs = F.softmax(out, dim=-1)                 # (.., n_bins)
            return (probs * self.bin_centers).sum(-1, keepdim=True)
        if self.cfg.task == "regression" and self.cfg.loss == "pinball":
            k = self.cfg.n_quantiles // 2
            return out[..., k:k + 1]
        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        dm = self.cfg.d_model

        h = self.feature_proj(x)
        h = self.feature_norm(h)

        if self.cfg.task == "regression" and self.cfg.label_mode == "bin":
            bin_idx = self._bin_of(y)                      # (B,N)
            lab = self.label_embed(bin_idx)                # (B,N,dm)
        elif self.cfg.task == "regression":
            lab = self.label_proj(y) * self.label_scale    # (B,N,dm)
        else:
            lab = self.label_embed(y.squeeze(-1).long())   # (B,N,dm)

        mask_e = self.mask_token.expand(B, N, -1)
        tok_label = torch.where(mask.unsqueeze(-1), mask_e, lab)

        tokens = h + tok_label + self.pe[:N].unsqueeze(0)
        z = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        return self.head(z)


class FixedDimICLLightning(pl.LightningModule):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
        self.cfg = cfg
        self.model = FixedDimICLTransformer(cfg)

    def forward(self, x, y, mask, key_padding_mask=None):
        return self.model(x, y, mask, key_padding_mask)

    def _step(self, batch, prefix: str) -> torch.Tensor:
        x, y, mask = batch["x"], batch["y"], batch["mask"]
        kpm = batch.get("key_padding_mask")
        out = self(x, y, mask, kpm)
        valid = mask & (~(kpm.bool() if kpm is not None else
                          mask.new_zeros(mask.shape, dtype=torch.bool)))
        yq = y[valid].squeeze(-1)
        point = self.model.point_pred(out)[valid].squeeze(-1)
        if self.cfg.task == "regression" and self.cfg.label_mode == "bin":
            target_bin = self.model._bin_of(y)[valid]
            loss = F.cross_entropy(out[valid], target_bin)
        elif self.cfg.task == "regression" and self.cfg.loss == "pinball":
            levels = self.model.quantile_levels
            err = yq.unsqueeze(-1) - out[valid]
            loss = torch.mean(torch.max(levels * err, (levels - 1) * err))
        elif self.cfg.task == "regression":
            loss = F.mse_loss(point, yq)
        else:
            loss = F.cross_entropy(out[valid], y[valid].long().squeeze(-1))
            with torch.no_grad():
                acc = (out[valid].argmax(-1) == y[valid].long().squeeze(-1)).float().mean()
            self.log(f"{prefix}/acc", acc, on_step=(prefix == "train"), on_epoch=True,
                     prog_bar=(prefix == "val"), batch_size=int(valid.sum()))

        with torch.no_grad():
            mae = F.l1_loss(point, yq)
        self.log(f"{prefix}/mae", mae, on_step=(prefix == "train"), on_epoch=True,
                 prog_bar=(prefix == "val"), batch_size=int(valid.sum()))
        self.log(f"{prefix}/loss", loss, on_step=(prefix == "train"), on_epoch=True,
                 prog_bar=(prefix == "val"), batch_size=int(valid.sum()))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay, betas=(0.9, 0.98),
        )
        epochs = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 1
        total_steps = max(epochs, 1)
        warmup_steps = max(int(total_steps * 0.1), 1)
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
