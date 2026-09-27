import math
import random
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from config import MIN_CONTEXT, MAX_CONTEXT, NUM_QUERIES


# ==============================================================================
# 1. Chemically-Informed Synthetic Task Prior
# ==============================================================================

class ChemicallyInformedSyntheticPrior:
    """
    Simulates realistic Structure-Activity Relationship (SAR) landscapes:
    - Global linear/shallow physicochemical trends (MW/lipophilicity baselines)
    - Localized RBF pharmacophore binding modes around active chemical clusters
    - Activity cliffs: steep, localized non-linear drops in potency between close analogs
    - Heteroscedastic observational assay noise
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_molecules, embed_dim)
        Returns:
            y: (N_molecules, 1) standardized continuous bioactivity targets
        """
        device = x.device
        n_samples = x.size(0)

        # 1. Global Physicochemical Trend (Linear projection)
        w_global = torch.randn(self.embed_dim, 1, device=device) / math.sqrt(self.embed_dim)
        y_global = x @ w_global

        # 2. Localized Binding Pockets (RBF Pharmacophores centered on actual molecules)
        num_pharmacophores = random.randint(1, 4)
        active_indices = torch.randperm(n_samples)[:num_pharmacophores]
        centroids = x[active_indices]  # (num_pharmacophores, embed_dim)

        dists = torch.cdist(x, centroids, p=2)  # (N, num_pharmacophores)
        gamma = random.uniform(0.5, 3.0)
        amplitudes = torch.empty(num_pharmacophores, 1, device=device).uniform_(1.5, 4.0)
        y_binding = torch.sum(amplitudes.T * torch.exp(-gamma * (dists ** 2)), dim=1, keepdim=True)

        # 3. Activity Cliffs (Sudden discontinuous drop for a fraction of close analogs)
        cliff_mask = torch.zeros(n_samples, 1, device=device)
        if random.random() < 0.6:  # 60% of assays have steep activity cliffs
            cliff_center_idx = active_indices[0]
            dist_to_cliff = torch.norm(x - x[cliff_center_idx], dim=1, keepdim=True)
            # Find close structural analogs
            cliff_analog_indices = (dist_to_cliff < 1.2) & (dist_to_cliff > 0.05)
            # Apply severe negative potency penalty to create the cliff
            cliff_mask[cliff_analog_indices] = -random.uniform(2.5, 5.0)

        # 4. Composite Target
        y_raw = y_global + y_binding + cliff_mask

        # 5. Heteroscedastic / Heavy-Tailed Assay Noise
        noise_std = random.uniform(0.05, 0.25)
        # 20% probability of Student-t heavy-tailed experimental outlier noise
        if random.random() < 0.2:
            noise = torch.distributions.StudentT(df=3.0).sample(y_raw.shape).to(device) * noise_std
        else:
            noise = torch.randn_like(y_raw) * noise_std

        y = y_raw + noise

        # Standardize target to N(0, 1) across the episode
        y = (y - y.mean()) / (y.std() + 1e-7)
        return y


# ==============================================================================
# 2. Log-Uniform Episodic Dataset
# ==============================================================================

class InMemoryMoleculeICLDataset(Dataset):
    def __init__(
        self,
        embeddings: torch.Tensor,
        min_context: int = MIN_CONTEXT,
        max_context: int = MAX_CONTEXT,
        num_queries: int = NUM_QUERIES,
        num_episodes: int = 10_000,
    ):
        self.embeddings = embeddings
        self.embed_dim = embeddings.shape[1]
        self.min_context = min_context
        self.max_context = max_context
        self.num_queries = num_queries
        self.num_episodes = num_episodes
        self.prior = ChemicallyInformedSyntheticPrior(embed_dim=self.embed_dim)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int):
        n_samples = len(self.embeddings)

        # LOG-UNIFORM SAMPLING: Equal training mass across orders of magnitude
        log_min = math.log2(self.min_context)
        log_max = math.log2(self.max_context)
        n_context = int(round(2 ** random.uniform(log_min, log_max)))
        n_context = max(self.min_context, min(self.max_context, n_context))

        total_needed = n_context + self.num_queries
        sample_indices = torch.randperm(n_samples)[:total_needed]
        x_selected = self.embeddings[sample_indices]

        # Generate SAR target landscape
        y_selected = self.prior(x_selected)

        return {
            "x_ctx": x_selected[:n_context],
            "y_ctx": y_selected[:n_context],
            "x_query": x_selected[n_context:],
            "y_query": y_selected[n_context:],
            "n_ctx": n_context,
            "n_query": self.num_queries,
        }


def pad_icl_batch_collate(batch):
    batch_size = len(batch)
    max_ctx = max(item["n_ctx"] for item in batch)
    max_query = max(item["n_query"] for item in batch)
    embed_dim = batch[0]["x_ctx"].size(-1)

    x_all = torch.zeros(batch_size, max_ctx + max_query, embed_dim)
    y_all = torch.zeros(batch_size, max_ctx + max_query, 1)
    y_query_gt = torch.zeros(batch_size, max_query, 1)
    sample_mask = torch.zeros(batch_size, max_ctx + max_query, dtype=torch.bool)
    context_lens = []

    for i, item in enumerate(batch):
        nc = item["n_ctx"]
        nq = item["n_query"]
        context_lens.append(nc)

        x_all[i, :nc] = item["x_ctx"]
        y_all[i, :nc] = item["y_ctx"]
        sample_mask[i, :nc] = True

        x_all[i, nc:nc + nq] = item["x_query"]
        sample_mask[i, nc:nc + nq] = True
        y_query_gt[i, :nq] = item["y_query"]

    return {
        "x": x_all,
        "y": y_all,
        "y_query_gt": y_query_gt,
        "sample_mask": sample_mask,
        "context_lens": context_lens,
        "query_len": max_query,
    }


# ==============================================================================
# 3. Inductive Transformer Backbone
# ==============================================================================

class InductiveCheMeleonTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_bins: int = 64,  # Distributional output discretization
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_bins = num_bins

        self.x_proj = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.y_proj = nn.Linear(1, d_model)
        self.query_target_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Predicts logits over discretized continuous outcome bins (TabPFN Bar distribution)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_bins),
        )

    def _build_inductive_mask(
        self,
        batch_size: int,
        total_len: int,
        context_lens: list[int],
        query_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Strictly Inductive Attention Mask:
        - Context rows attend strictly to Context rows.
        - Query rows attend to Context rows and ONLY to themselves on the diagonal.
        - Query rows CANNOT attend to other Query rows.
        """
        mask = torch.zeros(batch_size, total_len, total_len, device=device)
        for b, nc in enumerate(context_lens):
            # 1. Context cannot attend to Query
            mask[b, :nc, nc:] = float("-inf")
            # 2. Query cannot attend to other Queries
            mask[b, nc:, nc:] = float("-inf")
            # 3. Allow Query self-attention along diagonal
            for q in range(query_len):
                mask[b, nc + q, nc + q] = 0.0

        return mask

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context_lens: list[int],
        query_len: int,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        h_x = self.x_proj(x)

        h_tokens = torch.zeros_like(h_x)
        for b, nc in enumerate(context_lens):
            # Context tokens encode real targets
            h_tokens[b, :nc] = h_x[b, :nc] + self.y_proj(y[b:b+1, :nc]).squeeze(0)
            # Query tokens receive learned query mask token
            h_tokens[b, nc:nc + query_len] = (
                h_x[b, nc:nc + query_len] + self.query_target_token.expand(1, query_len, -1).squeeze(0)
            )

        attn_mask = self._build_inductive_mask(B, N, context_lens, query_len, x.device)
        num_heads = self.transformer.layers[0].self_attn.num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        out = self.transformer(
            src=h_tokens,
            mask=attn_mask,
            src_key_padding_mask=~sample_mask if sample_mask is not None else None,
        )

        query_logits = []
        for b, nc in enumerate(context_lens):
            q_out = out[b, nc:nc + query_len]
            query_logits.append(self.head(q_out))

        return torch.stack(query_logits, dim=0)  # (B, query_len, num_bins)


# ==============================================================================
# 4. Lightning Module with Riemann Distribution Loss & Bagged Inference
# ==============================================================================

class TabularCheMeleonLightningModule(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int = 32,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        num_bins: int = 64,
        support_min: float = -4.0,
        support_max: float = 4.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Setup Riemann distribution support grid
        bin_edges = torch.linspace(support_min, support_max, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_centers", bin_centers)

        self.model = InductiveCheMeleonTransformer(
            embed_dim=embed_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_bins=num_bins,
        )

    def forward(self, batch):
        return self.model(
            x=batch["x"],
            y=batch["y"],
            context_lens=batch["context_lens"],
            query_len=batch["query_len"],
            sample_mask=batch["sample_mask"],
        )

    def _distribution_to_moments(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts output logits into expected value E[Y] and variance Var[Y]."""
        probs = F.softmax(logits, dim=-1)
        mean = torch.sum(probs * self.bin_centers, dim=-1, keepdim=True)
        var = torch.sum(probs * ((self.bin_centers - mean) ** 2), dim=-1, keepdim=True)
        return mean, var

    def training_step(self, batch, batch_idx):
        logits = self(batch)  # (B, Q, num_bins)
        targets = batch["y_query_gt"]  # (B, Q, 1)

        # Digitize continuous target into discrete bin indices (Bar distribution)
        target_bins = torch.bucketize(targets, self.bin_edges[1:-1]).squeeze(-1)

        loss = F.cross_entropy(logits.view(-1, self.num_bins), target_bins.view(-1))
        with torch.no_grad():
            pred_mean, _ = self._distribution_to_moments(logits)
            mae = F.l1_loss(pred_mean, targets)

        batch_size = batch["x"].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("train/mae", mae, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch["y_query_gt"]
        target_bins = torch.bucketize(targets, self.bin_edges[1:-1]).squeeze(-1)

        loss = F.cross_entropy(logits.view(-1, self.num_bins), target_bins.view(-1))
        pred_mean, _ = self._distribution_to_moments(logits)

        batch_size = batch["x"].size(0)
        total_mae = F.l1_loss(pred_mean, targets)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/mae_overall", total_mae, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        # 1. Compute MAE per episode: shape (B,)
        per_episode_mae = (pred_mean - targets).abs().mean(dim=(1, 2))

        # 2. Get individual context length per episode: shape (B,)
        ctx_lens = torch.tensor(batch["context_lens"], device=targets.device)

        few_shot_mask = ctx_lens < 32
        rich_context_mask = ctx_lens >= 64

        # 3. Log using the actual count of matching episodes as batch_size
        if few_shot_mask.any():
            self.log(
                "val/mae_few_shot",
                per_episode_mae[few_shot_mask].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=int(few_shot_mask.sum()),
                sync_dist=True,
            )

        if rich_context_mask.any():
            self.log(
                "val/mae_rich_context",
                per_episode_mae[rich_context_mask].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=int(rich_context_mask.sum()),
                sync_dist=True,
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer.max_epochs else 50,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

# ==============================================================================
# 6. DataModule & Training Script
# ==============================================================================

class TabularCheMeleonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        embeddings_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
        val_split: float = 0.2,
        episodes_per_epoch: int = 10_000,
        val_episodes: int = 1_000,
        min_context: int = MIN_CONTEXT,
        max_context: int = MAX_CONTEXT,
        num_queries: int = NUM_QUERIES,
    ):
        super().__init__()
        self.embeddings_path = Path(embeddings_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.episodes_per_epoch = episodes_per_epoch
        self.val_episodes = val_episodes
        self.min_context = min_context
        self.max_context = max_context
        self.num_queries = num_queries
        self.embed_dim: Optional[int] = None

    def setup(self, stage: Optional[str] = None):
        if self.embeddings_path.is_file():
            embeddings = torch.load(self.embeddings_path, weights_only=True)
        else:
            files = sorted(self.embeddings_path.glob("*.pt"))
            if not files:
                raise FileNotFoundError(f"No .pt files found in {self.embeddings_path}")
            embeddings = torch.concat([torch.load(f, weights_only=True) for f in files])

        self.embed_dim = embeddings.shape[1]
        n_total = len(embeddings)
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
        train_size = int((1.0 - self.val_split) * n_total)

        self.train_ds = InMemoryMoleculeICLDataset(
            embeddings=embeddings[indices[:train_size]],
            min_context=self.min_context,
            max_context=self.max_context,
            num_queries=self.num_queries,
            num_episodes=self.episodes_per_epoch,
        )
        self.val_ds = InMemoryMoleculeICLDataset(
            embeddings=embeddings[indices[train_size:]],
            min_context=self.min_context,
            max_context=self.max_context,
            num_queries=self.num_queries,
            num_episodes=self.val_episodes,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pad_icl_batch_collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pad_icl_batch_collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--episodes-per-epoch", type=int, default=10_000)
    parser.add_argument("--val-episodes", type=int, default=1_000)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-bins", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-context", type=int, default=MIN_CONTEXT)
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--log-dir", type=str, default="tabular_chemeleon_logs")
    args = parser.parse_args()

    datamodule = TabularCheMeleonDataModule(
        embeddings_path=args.embeddings_path,
        batch_size=args.batch_size,
        episodes_per_epoch=args.episodes_per_epoch,
        val_episodes=args.val_episodes,
        min_context=args.min_context,
        max_context=args.max_context,
        num_queries=args.num_queries,
    )
    datamodule.setup()

    model = TabularCheMeleonLightningModule(
        embed_dim=datamodule.embed_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        num_bins=args.num_bins,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logger = TensorBoardLogger(save_dir=args.log_dir, name="inductive_tabular_chemeleon", default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename="best-{epoch:02d}-{val_loss:.4f}")
    early_stopping = EarlyStopping(monitor="val/loss", patience=args.patience, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)

    # save the model for inference, ignoring optimizer state and other training artifacts
    model_path = Path(args.log_dir) / "tabularchemeleon.pt"
    trainer.save_checkpoint(model_path, weights_only=True)
