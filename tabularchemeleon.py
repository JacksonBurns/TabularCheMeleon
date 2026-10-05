import math
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from config import MIN_CONTEXT, MAX_CONTEXT, NUM_QUERIES


# ==============================================================================
# 1. Chemically-Grounded Bioactivity Manifold Prior
# ==============================================================================

class BioactivityManifoldPrior:
    """
    Simulates authentic biological Structure-Activity Relationships (SAR):
    1. Free-Wilson Additive SAR (directional latent substituent vectors & synergy)
    2. Multi-Pocket Pharmacophore GP (cosine-metric kernel with adaptive local bandwidth)
    3. Cooperative Hill Kinetics (sigmoidal receptor binding curves for kinase assays)
    4. Non-monotonic ADMET Surfaces (parabolic lipophilicity and solubility windows)
    5. Structural Activity Cliffs (bidirectional shifts strictly on high-similarity pairs)
    6. Heavy-tailed assay noise & experimental plate censoring limits
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    # --------------------------------------------------------------------------
    # Sub-Prior 1: Free-Wilson Additive SAR (Substituent Vectors & Synergy)
    # --------------------------------------------------------------------------
    def _free_wilson_prior(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        k_axes = min(self.embed_dim, random.randint(3, 8))
        
        # Orthonormal projection axes simulating chemical property axes
        proj_matrix, _ = torch.linalg.qr(torch.randn(self.embed_dim, k_axes, device=device))
        projections = x @ proj_matrix  # (N, k_axes)

        y = torch.zeros(x.size(0), 1, device=device)
        for i in range(k_axes):
            weight = random.uniform(0.8, 2.5) * (1 if random.random() > 0.5 else -1)
            # Mix of linear and saturating substituent responses
            if random.random() > 0.5:
                y += weight * torch.tanh(projections[:, i : i + 1])
            else:
                y += weight * projections[:, i : i + 1]

        # Second-order cooperative synergy or steric clash
        if k_axes >= 2:
            synergy_weight = random.uniform(-1.5, 1.5)
            y += synergy_weight * (projections[:, 0:1] * projections[:, 1:2])

        return y

    # --------------------------------------------------------------------------
    # Sub-Prior 2: Multi-Pocket Pharmacophore Cosine GP
    # --------------------------------------------------------------------------
    def _cosine_gp_prior(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        n_samples = x.size(0)
        x_norm = F.normalize(x, p=2, dim=-1)

        num_centroids = random.randint(2, 5)
        centroid_idx = torch.randperm(n_samples, device=device)[:num_centroids]
        centroids = x_norm[centroid_idx]

        # Angular distance in [0, 2]
        angular_dist = 1.0 - (x_norm @ centroids.T)

        # Adaptive lengthscale based on the episode's local similarity distribution
        median_dist = torch.median(angular_dist) + 1e-5
        lengthscale = median_dist * random.uniform(0.6, 1.8)
        gamma = 1.0 / (2.0 * (lengthscale ** 2) + 1e-6)

        amplitudes = torch.empty(num_centroids, 1, device=device).uniform_(-3.5, 3.5)
        y = torch.exp(-gamma * (angular_dist ** 2)) @ amplitudes
        return y

    # --------------------------------------------------------------------------
    # Sub-Prior 3: Cooperative Hill Kinetics (Kinase Saturation Curves)
    # --------------------------------------------------------------------------
    def _hill_kinetics_prior(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        n_samples = x.size(0)
        x_norm = F.normalize(x, p=2, dim=-1)

        num_pockets = random.randint(1, 3)
        pocket_idx = torch.randperm(n_samples, device=device)[:num_pockets]
        centroids = x_norm[pocket_idx]

        sims = x_norm @ centroids.T  # (N, num_pockets) in [-1, 1]
        
        # Effective latent ligand affinity concentration
        alpha = random.uniform(2.0, 4.0)
        ligand_potency = torch.exp(alpha * sims)

        hill_coeff = random.uniform(0.9, 2.2)
        kd = torch.empty(1, num_pockets, device=device).uniform_(1.5, 4.5) ** hill_coeff
        e_max = torch.empty(num_pockets, 1, device=device).uniform_(2.0, 5.0)

        y = ((ligand_potency ** hill_coeff) / (kd + (ligand_potency ** hill_coeff))) @ e_max
        return y

    # --------------------------------------------------------------------------
    # Sub-Prior 4: Non-monotonic ADMET Surfaces (Solubility & LogP Sweet Spots)
    # --------------------------------------------------------------------------
    def _admet_surface_prior(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        k_active = min(self.embed_dim, random.randint(4, 10))
        active_indices = torch.randperm(self.embed_dim, device=device)[:k_active]
        x_sub = x[:, active_indices]

        # Linear background
        w_lin = torch.randn(k_active, 1, device=device) / math.sqrt(k_active)
        y = x_sub @ w_lin

        # Parabolic optimum (e.g., Hansch lipophilicity parabola)
        optimum = random.uniform(-1.0, 1.0)
        y -= 0.6 * ((x_sub[:, 0:1] - optimum) ** 2)
        return y

    # --------------------------------------------------------------------------
    # Main Generator Execution
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        n_samples = x.size(0)

        # 1. Sample primary landscape
        p = random.random()
        if p < 0.30:
            y = self._free_wilson_prior(x)
        elif p < 0.60:
            y = self._cosine_gp_prior(x)
        elif p < 0.85:
            y = self._hill_kinetics_prior(x)
        else:
            y = self._admet_surface_prior(x)

        # 2. Add weak global linear baseline
        w_bg = torch.randn(self.embed_dim, 1, device=device) / math.sqrt(self.embed_dim)
        y = y + 0.25 * (x @ w_bg)

        # 3. Inject Activity Cliffs strictly into genuine structural analogs
        if random.random() < 0.60 and n_samples > 4:
            x_norm = F.normalize(x, p=2, dim=-1)
            sim_matrix = x_norm @ x_norm.T

            # Exclude self-similarity along diagonal
            sim_matrix.fill_diagonal_(0.0)
            high_sim_pairs = torch.nonzero(sim_matrix > 0.80)

            if len(high_sim_pairs) > 0:
                cliff_count = min(3, max(1, len(high_sim_pairs) // 4))
                perm = torch.randperm(len(high_sim_pairs))[:cliff_count]
                for p_idx in perm:
                    target_mol = high_sim_pairs[p_idx][1].item()
                    cliff_mag = random.uniform(2.5, 5.0) * (1 if random.random() > 0.5 else -1)
                    y[target_mol] += cliff_mag

        # 4. Realistic Assay Censoring (Plate Detection Limits)
        if random.random() < 0.25:
            if random.random() > 0.5:
                floor_val = torch.quantile(y, random.uniform(0.05, 0.15))
                y = torch.clamp(y, min=floor_val)
            else:
                ceil_val = torch.quantile(y, random.uniform(0.85, 0.95))
                y = torch.clamp(y, max=ceil_val)

        # 5. Composite Heteroscedastic Noise
        noise_std = random.uniform(0.03, 0.18)
        if random.random() < 0.20:
            noise = torch.distributions.StudentT(df=3.0).sample(y.shape).to(device) * noise_std
        else:
            noise = torch.randn_like(y) * noise_std
        y = y + noise

        # 6. Standardize target distribution across episode
        y = (y - y.mean()) / (y.std() + 1e-7)
        return y


# ==============================================================================
# 2. Cluster-Aware Chemical Episodic Dataset
# ==============================================================================

class ChemicalEpisodicDataset(Dataset):
    """
    Episodic Dataset that mirrors real-world drug discovery workflows:
    - 40% Congeneric Analog Series: Dense clusters around a seed compound (SAR learning).
    - 35% Multi-Scaffold Hit Expansion: 2 to 4 separate hit clusters (scaffold ranking).
    - 25% Global Screening Decks: Broad uniform sampling (physicochemical baselines).
    """
    def __init__(
        self,
        embeddings: torch.Tensor,
        min_context: int = MIN_CONTEXT,
        max_context: int = MAX_CONTEXT,
        num_queries: int = NUM_QUERIES,
        num_episodes: int = 20_000,
    ):
        self.embeddings = embeddings.float()
        self.embed_dim = embeddings.shape[1]
        self.n_samples = len(embeddings)
        self.min_context = min_context
        self.max_context = max_context
        self.num_queries = num_queries
        self.num_episodes = num_episodes
        self.prior = BioactivityManifoldPrior(embed_dim=self.embed_dim)

    def __len__(self) -> int:
        return self.num_episodes

    def _sample_indices(self, total_needed: int) -> torch.Tensor:
        p = random.random()
        candidate_pool_size = min(self.n_samples, 2048)

        # Mode A: 40% Congeneric Analog Series (Tight local chemical cluster)
        if p < 0.40:
            seed_idx = random.randint(0, self.n_samples - 1)
            seed = self.embeddings[seed_idx : seed_idx + 1]

            pool_idx = torch.randperm(self.n_samples)[:candidate_pool_size]
            pool = self.embeddings[pool_idx]

            sims = F.cosine_similarity(seed, pool, dim=-1)
            topk = torch.topk(sims, k=min(total_needed, len(pool)), largest=True).indices
            selected = pool_idx[topk]

            if len(selected) < total_needed:
                deficit = total_needed - len(selected)
                padding = torch.randperm(self.n_samples)[:deficit]
                selected = torch.cat([selected, padding])

            return selected[torch.randperm(total_needed)]

        # Mode B: 35% Multi-Scaffold Expansion (2-3 distinct chemical families)
        elif p < 0.75:
            num_families = random.randint(2, 3)
            per_family = total_needed // num_families
            family_indices = []

            for _ in range(num_families):
                seed_idx = random.randint(0, self.n_samples - 1)
                seed = self.embeddings[seed_idx : seed_idx + 1]
                pool_idx = torch.randperm(self.n_samples)[:1024]
                pool = self.embeddings[pool_idx]

                sims = F.cosine_similarity(seed, pool, dim=-1)
                topk = torch.topk(sims, k=min(per_family, len(pool)), largest=True).indices
                family_indices.append(pool_idx[topk])

            selected = torch.cat(family_indices)
            if len(selected) < total_needed:
                deficit = total_needed - len(selected)
                padding = torch.randperm(self.n_samples)[:deficit]
                selected = torch.cat([selected, padding])

            return selected[:total_needed][torch.randperm(total_needed)]

        # Mode C: 25% Global Diverse Library (Uniform sampling)
        else:
            return torch.randperm(self.n_samples)[:total_needed]

    def __getitem__(self, idx: int):
        # Sample context length on a log-uniform scale
        log_min = math.log2(self.min_context)
        log_max = math.log2(self.max_context)
        n_context = int(round(2 ** random.uniform(log_min, log_max)))
        n_context = max(self.min_context, min(self.max_context, n_context))

        total_needed = n_context + self.num_queries
        sample_indices = self._sample_indices(total_needed)
        x_selected = self.embeddings[sample_indices]

        # Generate realistic SAR bioactivity landscape
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
    query_len = batch[0]["n_query"]
    embed_dim = batch[0]["x_ctx"].size(-1)

    x_all = torch.zeros(batch_size, max_ctx + query_len, embed_dim)
    y_all = torch.zeros(batch_size, max_ctx + query_len, 1)
    y_query_gt = torch.zeros(batch_size, query_len, 1)
    sample_mask = torch.zeros(batch_size, max_ctx + query_len, dtype=torch.bool)
    context_lens = []

    for i, item in enumerate(batch):
        nc = item["n_ctx"]
        context_lens.append(nc)

        x_all[i, :nc] = item["x_ctx"]
        y_all[i, :nc] = item["y_ctx"]
        sample_mask[i, :nc] = True

        # Queries sit immediately after context
        x_all[i, nc:nc + query_len] = item["x_query"]
        sample_mask[i, nc:nc + query_len] = True
        y_query_gt[i] = item["y_query"]

    return {
        "x": x_all,
        "y": y_all,
        "y_query_gt": y_query_gt,
        "sample_mask": sample_mask,
        "context_lens": context_lens,
        "query_len": query_len,
    }


# ==============================================================================
# 3. Inductive Transformer Backbone
# ==============================================================================

class InductiveCheMeleonTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_bins: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_bins = num_bins

        # Input feature projection
        self.x_proj = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Target projection (MLP allows non-linear token representations in d_model)
        self.y_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.query_target_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

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
        mask = torch.zeros(batch_size, total_len, total_len, device=device)
        for b, nc in enumerate(context_lens):
            # Context cannot attend to Queries or Padding
            mask[b, :nc, nc:] = float("-inf")
            # Queries cannot attend to other Queries
            mask[b, nc:nc + query_len, nc:nc + query_len] = float("-inf")
            # Queries attend to themselves on the diagonal
            for q in range(query_len):
                mask[b, nc + q, nc + q] = 0.0
            # Queries cannot attend to padding beyond active tokens
            if nc + query_len < total_len:
                mask[b, nc:nc + query_len, nc + query_len:] = float("-inf")

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
            # Context tokens encode inputs + known targets
            h_tokens[b, :nc] = h_x[b, :nc] + self.y_proj(y[b:b+1, :nc]).squeeze(0)
            # Query tokens receive the learned query target token
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
# 4. Lightning Module with Two-Hot Smooth Target Distribution Loss
# ==============================================================================

class TabularCheMeleonLightningModule(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 8,
        dim_feedforward: int = 512,
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

        # Setup Riemann support grid
        bin_edges = torch.linspace(support_min, support_max, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_centers", bin_centers)
        self.bin_width = (support_max - support_min) / num_bins

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
        probs = F.softmax(logits, dim=-1)
        mean = torch.sum(probs * self.bin_centers, dim=-1, keepdim=True)
        var = torch.sum(probs * ((self.bin_centers - mean) ** 2), dim=-1, keepdim=True)
        return mean, var

    def _two_hot_encoding(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Smooth target interpolation between adjacent support bins.
        Eliminates step-discretization quantization noise.
        """
        c0 = self.bin_centers[0]
        c_last = self.bin_centers[-1]
        clamped = torch.clamp(targets, min=c0.item(), max=c_last.item())

        unit_pos = (clamped - c0) / self.bin_width
        idx_lower = torch.floor(unit_pos).long()
        idx_upper = torch.clamp(idx_lower + 1, max=self.num_bins - 1)

        w_upper = unit_pos - idx_lower.float()
        w_lower = 1.0 - w_upper

        B, Q, _ = targets.shape
        soft_targets = torch.zeros(B, Q, self.num_bins, device=targets.device)
        soft_targets.scatter_add_(-1, idx_lower, w_lower)
        soft_targets.scatter_add_(-1, idx_upper, w_upper)
        return soft_targets

    def training_step(self, batch, batch_idx):
        logits = self(batch)  # (B, Q, num_bins)
        targets = batch["y_query_gt"]  # (B, Q, 1)

        soft_targets = self._two_hot_encoding(targets)
        loss = F.cross_entropy(logits.view(-1, self.num_bins), soft_targets.view(-1, self.num_bins))

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

        soft_targets = self._two_hot_encoding(targets)
        loss = F.cross_entropy(logits.view(-1, self.num_bins), soft_targets.view(-1, self.num_bins))

        pred_mean, _ = self._distribution_to_moments(logits)
        mae = F.l1_loss(pred_mean, targets)

        batch_size = batch["x"].size(0)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/mae_overall", mae, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        per_episode_mae = (pred_mean - targets).abs().mean(dim=(1, 2))
        ctx_lens = torch.tensor(batch["context_lens"], device=targets.device)

        few_shot_mask = ctx_lens < 32
        rich_context_mask = ctx_lens >= 64

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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
        )
        total_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ==============================================================================
# 5. DataModule & Training Entrypoint
# ==============================================================================

class TabularCheMeleonDataModule(pl.LightningDataModule):
    def __init__(
        self,
        embeddings_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
        val_split: float = 0.15,
        episodes_per_epoch: int = 16_000,
        val_episodes: int = 1_280,
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

        self.train_ds = ChemicalEpisodicDataset(
            embeddings=embeddings[indices[:train_size]],
            min_context=self.min_context,
            max_context=self.max_context,
            num_queries=self.num_queries,
            num_episodes=self.episodes_per_epoch,
        )
        self.val_ds = ChemicalEpisodicDataset(
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
    parser.add_argument("--episodes-per-epoch", type=int, default=16_000)
    parser.add_argument("--val-episodes", type=int, default=1_280)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-bins", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-context", type=int, default=MIN_CONTEXT)
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
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

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="tabular_chemeleon",
        default_hp_metric=False,
    )

    # Checkpoint template correctly formatted to match the logged metric key
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val/loss:.4f}",
    )
    early_stopping = EarlyStopping(
        monitor="val/loss",
        patience=args.patience,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=25,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=datamodule)

    # Save finalized model checkpoint for inference
    model_path = Path(args.log_dir) / "tabularchemeleon_production.pt"
    trainer.save_checkpoint(model_path, weights_only=True)
    print(f"Production model successfully trained and saved to: {model_path}")
