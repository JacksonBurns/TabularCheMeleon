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

try:
    from config import MIN_CONTEXT, MAX_CONTEXT, NUM_QUERIES
except ImportError:
    MIN_CONTEXT = 32
    MAX_CONTEXT = 1024
    NUM_QUERIES = 64


# ==============================================================================
# 1. Chemically-Grounded Empirical Bioactivity Prior
# ==============================================================================

class EmpiricalBioactivityPrior:
    """
    Bioactivity prior generating continuous SAR response manifolds:
    1. Local Multi-Pocket Cosine GP (angular distance with adaptive bandwidth)
    2. Cooperative Receptor Hill Kinetics (sigmoidal dose-response curves)
    3. Free-Wilson Additive SAR (directional PCA empirical projections + synergy)
    4. Non-monotonic ADMET Surfaces (Hansch lipophilicity & solubility parabolas)
    5. Structural activity cliffs strictly on close analogs (0.80 < sim < 0.98)
    6. Heteroscedastic noise & plate censoring limits
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def _sample_empirical_directions(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Derives k orthogonal axes from empirical chemical variation via truncated SVD."""
        n_samples = x.size(0)
        k_clamped = min(k, n_samples - 1, self.embed_dim)
        if k_clamped < 1:
            return torch.randn(self.embed_dim, k, device=x.device) / math.sqrt(self.embed_dim)

        x_centered = x - x.mean(dim=0, keepdim=True)
        try:
            _, _, V = torch.pca_lowrank(x_centered, q=k_clamped)
            proj = V[:, :k_clamped]
            if k_clamped < k:
                pad = torch.randn(self.embed_dim, k - k_clamped, device=x.device) / math.sqrt(self.embed_dim)
                proj = torch.cat([proj, pad], dim=1)
            return proj
        except Exception:
            return torch.randn(self.embed_dim, k, device=x.device) / math.sqrt(self.embed_dim)

    def _cosine_gp_prior(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        n_samples = x.size(0)
        x_norm = F.normalize(x, p=2, dim=-1)

        num_centroids = random.randint(2, 5)
        centroid_idx = torch.randperm(n_samples, device=device)[:num_centroids]
        centroids = x_norm[centroid_idx]

        angular_dist = 1.0 - (x_norm @ centroids.T)  # (N, num_centroids) in [0, 2]
        median_dist = torch.median(angular_dist) + 1e-5
        lengthscale = median_dist * random.uniform(0.6, 1.8)
        gamma = 1.0 / (2.0 * (lengthscale ** 2) + 1e-6)

        amplitudes = torch.empty(num_centroids, 1, device=device).uniform_(-3.5, 3.5)
        return torch.exp(-gamma * (angular_dist ** 2)) @ amplitudes

    def _hill_kinetics_prior(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        n_samples = x.size(0)
        x_norm = F.normalize(x, p=2, dim=-1)

        num_pockets = random.randint(1, 3)
        pocket_idx = torch.randperm(n_samples, device=device)[:num_pockets]
        centroids = x_norm[pocket_idx]

        sims = x_norm @ centroids.T
        alpha = random.uniform(2.0, 4.0)
        ligand_potency = torch.exp(alpha * sims)

        hill_coeff = random.uniform(0.9, 2.2)
        kd = torch.empty(1, num_pockets, device=device).uniform_(1.5, 4.5) ** hill_coeff
        e_max = torch.empty(num_pockets, 1, device=device).uniform_(2.0, 5.0)

        return ((ligand_potency ** hill_coeff) / (kd + (ligand_potency ** hill_coeff))) @ e_max

    def _free_wilson_prior(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        k_axes = min(self.embed_dim, random.randint(3, 8))
        proj_matrix = self._sample_empirical_directions(x, k_axes)
        projections = x @ proj_matrix

        y = torch.zeros(x.size(0), 1, device=device)
        for i in range(k_axes):
            weight = random.uniform(0.8, 2.5) * (1 if random.random() > 0.5 else -1)
            y += weight * (torch.tanh(projections[:, i : i + 1]) if random.random() > 0.5 else projections[:, i : i + 1])

        if k_axes >= 2:
            synergy = random.uniform(-1.5, 1.5)
            y += synergy * (projections[:, 0:1] * projections[:, 1:2])
        return y

    def _scaffold_gam_prior(self, x: torch.Tensor) -> torch.Tensor:
        k_active = min(self.embed_dim, random.randint(4, 10))
        proj_matrix = self._sample_empirical_directions(x, k_active)
        projections = x @ proj_matrix

        y = torch.sum(projections, dim=1, keepdim=True) / math.sqrt(k_active)
        optimum = random.uniform(-1.0, 1.0)
        y -= 0.6 * ((projections[:, 0:1] - optimum) ** 2)
        return y

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        n_samples = x.size(0)

        p = random.random()
        if p < 0.35:
            y = self._cosine_gp_prior(x)
        elif p < 0.65:
            y = self._hill_kinetics_prior(x)
        elif p < 0.85:
            y = self._free_wilson_prior(x)
        else:
            y = self._scaffold_gam_prior(x)

        # Baseline linear drift
        dir_bg = self._sample_empirical_directions(x, 1)
        y = y + 0.25 * (x @ dir_bg)

        # Structural activity cliffs on high-similarity analogs
        if random.random() < 0.65 and n_samples > 4:
            x_norm = F.normalize(x, p=2, dim=-1)
            sim_matrix = x_norm @ x_norm.T
            sim_matrix.fill_diagonal_(0.0)
            high_sim_pairs = torch.nonzero((sim_matrix > 0.80) & (sim_matrix < 0.98))

            if len(high_sim_pairs) > 0:
                cliff_count = min(3, max(1, len(high_sim_pairs) // 4))
                perm = torch.randperm(len(high_sim_pairs))[:cliff_count]
                for p_idx in perm:
                    target_mol = high_sim_pairs[p_idx][1].item()
                    cliff_mag = random.uniform(2.5, 5.0) * (1 if random.random() > 0.5 else -1)
                    y[target_mol] += cliff_mag

        # Plate censoring limits
        if random.random() < 0.25:
            if random.random() > 0.5:
                y = torch.clamp(y, min=torch.quantile(y, random.uniform(0.05, 0.15)))
            else:
                y = torch.clamp(y, max=torch.quantile(y, random.uniform(0.85, 0.95)))

        # Heteroscedastic noise
        noise_std = random.uniform(0.03, 0.18)
        if random.random() < 0.20:
            noise = torch.distributions.StudentT(df=3.0).sample(y.shape).to(device) * noise_std
        else:
            noise = torch.randn_like(y) * noise_std
        y = y + noise

        return (y - y.mean()) / (y.std() + 1e-7)


# ==============================================================================
# 2. Distance-Stratified Chemical Episodic Dataset
# ==============================================================================

class ChemicalEpisodicDataset(Dataset):
    """
    Episodic Dataset generating realistic chemical context topologies:
    - 30% Tight Congeneric Series (sim > 0.80): Local SAR and cliff learning.
    - 35% Scaffold-Hopping Series (0.35 < sim < 0.78): Bemis-Murcko benchmark coverage.
    - 20% Multi-Family Expansions: 2-3 distinct scaffolds (scaffold ranking).
    - 15% Global Diverse Decks: Broad library screening coverage.
    """
    def __init__(
        self,
        embeddings: torch.Tensor,
        min_context: int = MIN_CONTEXT,
        max_context: int = MAX_CONTEXT,
        num_queries: int = NUM_QUERIES,
        num_episodes: int = 16_000,
    ):
        self.embeddings = embeddings.float()
        self.embed_dim = embeddings.shape[1]
        self.n_samples = len(embeddings)
        self.min_context = min_context
        self.max_context = max_context
        self.num_queries = num_queries
        self.num_episodes = num_episodes
        self.prior = EmpiricalBioactivityPrior(embed_dim=self.embed_dim)

    def __len__(self) -> int:
        return self.num_episodes

    def _sample_indices(self, total_needed: int) -> torch.Tensor:
        p = random.random()
        pool_size = min(self.n_samples, max(2048, total_needed * 2))
        pool_idx = torch.randperm(self.n_samples)[:pool_size]
        pool = self.embeddings[pool_idx]
        seed = pool[0:1]
        sims = F.cosine_similarity(seed, pool, dim=-1)

        # Mode A: 30% Tight Congeneric Series
        if p < 0.30:
            topk = torch.topk(sims, k=min(total_needed, len(pool)), largest=True).indices
            selected = pool_idx[topk]

        # Mode B: 35% Scaffold-Hopping Series
        elif p < 0.65:
            mid_mask = (sims >= 0.35) & (sims <= 0.78)
            mid_indices = pool_idx[mid_mask]
            if len(mid_indices) >= total_needed:
                selected = mid_indices[torch.randperm(len(mid_indices))[:total_needed]]
            else:
                topk = torch.topk(sims, k=min(total_needed, len(pool)), largest=True).indices
                selected = pool_idx[topk]

        # Mode C: 20% Multi-Family Expansions
        elif p < 0.85:
            half = total_needed // 2
            seed_b = pool[1:2]
            sims_b = F.cosine_similarity(seed_b, pool, dim=-1)
            topk_a = torch.topk(sims, k=min(half, len(pool)), largest=True).indices
            topk_b = torch.topk(sims_b, k=min(half, len(pool)), largest=True).indices
            selected = torch.cat([pool_idx[topk_a], pool_idx[topk_b]])

        # Mode D: 15% Global Diverse Library
        else:
            selected = torch.randperm(self.n_samples)[:total_needed]

        if len(selected) < total_needed:
            deficit = total_needed - len(selected)
            selected = torch.cat([selected, torch.randperm(self.n_samples)[:deficit]])

        return selected[:total_needed][torch.randperm(total_needed)]

    def __getitem__(self, idx: int):
        log_min = math.log2(self.min_context)
        log_max = math.log2(self.max_context)
        n_context = int(round(2 ** random.uniform(log_min, log_max)))
        n_context = max(self.min_context, min(self.max_context, n_context))

        total_needed = n_context + self.num_queries
        sample_indices = self._sample_indices(total_needed)
        x_selected = self.embeddings[sample_indices]
        y_selected = self.prior(x_selected)

        return {
            "x_ctx": x_selected[:n_context],
            "y_ctx": y_selected[:n_context],
            "x_query": x_selected[n_context:],
            "y_query": y_selected[n_context:],
            "n_ctx": n_context,
            "n_query": self.num_queries,
        }


def pad_icl_perceiver_collate(batch):
    """
    Pads context sets to max_ctx in batch while building an explicit key_padding_mask.
    Queries maintain fixed dimension num_queries without padding.
    """
    batch_size = len(batch)
    max_ctx = max(item["n_ctx"] for item in batch)
    query_len = batch[0]["n_query"]
    embed_dim = batch[0]["x_ctx"].size(-1)

    x_ctx_padded = torch.zeros(batch_size, max_ctx, embed_dim)
    y_ctx_padded = torch.zeros(batch_size, max_ctx, 1)
    ctx_key_padding_mask = torch.ones(batch_size, max_ctx, dtype=torch.bool)  # True = Ignore

    x_query = torch.zeros(batch_size, query_len, embed_dim)
    y_query = torch.zeros(batch_size, query_len, 1)
    context_lens = []

    for i, item in enumerate(batch):
        nc = item["n_ctx"]
        context_lens.append(nc)

        x_ctx_padded[i, :nc] = item["x_ctx"]
        y_ctx_padded[i, :nc] = item["y_ctx"]
        ctx_key_padding_mask[i, :nc] = False  # False = Active token

        x_query[i] = item["x_query"]
        y_query[i] = item["y_query"]

    return {
        "x_ctx": x_ctx_padded,
        "y_ctx": y_ctx_padded,
        "ctx_key_padding_mask": ctx_key_padding_mask,
        "x_query": x_query,
        "y_query": y_query,
        "context_lens": context_lens,
    }


# ==============================================================================
# 3. Feature-Attention & Latent Perceiver Architecture
# ==============================================================================

class InContextFeatureAttention(nn.Module):
    """
    Computes assay-conditioned feature modulation:
    Evaluates covariance between feature projections and context targets,
    dynamically scaling latent channels that correlate with bioactivity in this assay.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_ctx: (B, N_ctx, d_model)
            y_ctx: (B, N_ctx, 1)
            ctx_mask: (B, N_ctx) bool mask (True = padding)
        Returns:
            channel_scales: (B, 1, d_model) modulation multipliers in [0, 2]
        """
        valid_mask = (~ctx_mask).float().unsqueeze(-1)  # (B, N_ctx, 1)
        denom = torch.clamp(valid_mask.sum(dim=1, keepdim=True), min=1.0)

        # Centered covariances between feature projections and continuous targets
        mean_h = (h_ctx * valid_mask).sum(dim=1, keepdim=True) / denom
        mean_y = (y_ctx * valid_mask).sum(dim=1, keepdim=True) / denom

        cov = ((h_ctx - mean_h) * (y_ctx - mean_y) * valid_mask).sum(dim=1) / denom.squeeze(-1)
        gate = self.gate_mlp(cov).unsqueeze(1)  # (B, 1, d_model)
        return 2.0 * gate


class LatentContextPerceiverTransformer(nn.Module):
    """
    Linear-Complexity In-Context Perceiver Transformer:
    1. In-line projection from native embedding dimension (e.g. 2048 -> d_model)
    2. Dynamic in-context feature gating
    3. Context Compressor: M learned latents cross-attend into N context molecules: O(M * N)
    4. Latent Engine: TransformerEncoder processing compressed assay memory: O(M^2)
    5. Query Decoder: Candidate molecules cross-attend into assay memory: O(Q * M)
    """
    def __init__(
        self,
        embed_dim: int = 2048,
        d_model: int = 256,
        m_latents: int = 128,
        nhead: int = 8,
        num_latent_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        num_bins: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.m_latents = m_latents
        self.num_bins = num_bins

        # In-line feature projection replacing offline autoencoders
        self.in_proj = nn.Sequential(
            nn.Linear(embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.feature_attention = InContextFeatureAttention(d_model=d_model)

        # Clean additive target projection
        self.y_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Learned latent assay summary tokens
        self.latent_tokens = nn.Parameter(torch.randn(1, m_latents, d_model) * 0.02)

        # Stage 1: Context Compressor (Cross-Attention)
        self.norm_latents = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.ctx_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.compressor_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Stage 2: Latent Transformer Engine
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.latent_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_latent_layers)

        # Stage 3: Query Decoder (Cross-Attention)
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_assay_memory = nn.LayerNorm(d_model)
        self.query_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.query_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Readout prediction head over Riemann continuous outcome bins
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_bins),
        )

    def forward(
        self,
        x_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        ctx_key_padding_mask: torch.Tensor,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        B = x_ctx.size(0)

        # 1. In-line molecular projection
        h_ctx_raw = self.in_proj(x_ctx)      # (B, N_ctx, d_model)
        h_query_raw = self.in_proj(x_query)  # (B, N_query, d_model)

        # 2. Dynamic in-context feature gating
        channel_scales = self.feature_attention(h_ctx_raw, y_ctx, ctx_key_padding_mask)
        h_ctx_feat = h_ctx_raw * channel_scales
        h_query_feat = h_query_raw * channel_scales

        # 3. Additive token formation for context
        h_ctx_tokens = h_ctx_feat + self.y_proj(y_ctx)

        # 4. Context Compressor (Cross-Attention: Latents -> Context)
        latents_q = self.norm_latents(self.latent_tokens.expand(B, -1, -1))
        ctx_kv = self.norm_ctx(h_ctx_tokens)

        z_compressed, _ = self.ctx_cross_attn(
            query=latents_q,
            key=ctx_kv,
            value=ctx_kv,
            key_padding_mask=ctx_key_padding_mask,
        )
        z = latents_q + z_compressed
        z = z + self.compressor_ffn(z)

        # 5. Latent Transformer Engine
        z_assay = self.latent_transformer(z)  # (B, M, d_model)

        # 6. Query Readout (Cross-Attention: Query -> Compressed Assay Memory)
        q_norm = self.norm_query(h_query_feat)
        mem_kv = self.norm_assay_memory(z_assay)

        q_readout, _ = self.query_cross_attn(
            query=q_norm,
            key=mem_kv,
            value=mem_kv,
        )
        h_out = q_norm + q_readout
        h_out = h_out + self.query_ffn(h_out)

        # 7. Distribution logits
        return self.head(h_out)  # (B, N_query, num_bins)


# ==============================================================================
# 4. Lightning Module with Two-Hot Continuous Riemann Loss
# ==============================================================================

class TabularCheMeleonLightningModule(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int = 2048,
        d_model: int = 256,
        m_latents: int = 128,
        nhead: int = 8,
        num_latent_layers: int = 8,
        dim_feedforward: int = 1024,
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

        self.model = LatentContextPerceiverTransformer(
            embed_dim=embed_dim,
            d_model=d_model,
            m_latents=m_latents,
            nhead=nhead,
            num_latent_layers=num_latent_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_bins=num_bins,
        )

    def forward(self, batch):
        return self.model(
            x_ctx=batch["x_ctx"],
            y_ctx=batch["y_ctx"],
            ctx_key_padding_mask=batch["ctx_key_padding_mask"],
            x_query=batch["x_query"],
        )

    def _distribution_to_moments(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        mean = torch.sum(probs * self.bin_centers, dim=-1, keepdim=True)
        var = torch.sum(probs * ((self.bin_centers - mean) ** 2), dim=-1, keepdim=True)
        return mean, var

    def _two_hot_encoding(self, targets: torch.Tensor) -> torch.Tensor:
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
        logits = self(batch)
        targets = batch["y_query"]

        soft_targets = self._two_hot_encoding(targets)
        loss = F.cross_entropy(logits.view(-1, self.num_bins), soft_targets.view(-1, self.num_bins))

        with torch.no_grad():
            pred_mean, _ = self._distribution_to_moments(logits)
            mae = F.l1_loss(pred_mean, targets)

        batch_size = batch["x_ctx"].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("train/mae", mae, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        targets = batch["y_query"]

        soft_targets = self._two_hot_encoding(targets)
        loss = F.cross_entropy(logits.view(-1, self.num_bins), soft_targets.view(-1, self.num_bins))

        pred_mean, _ = self._distribution_to_moments(logits)
        mae = F.l1_loss(pred_mean, targets)

        batch_size = batch["x_ctx"].size(0)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/mae_overall", mae, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)

        per_episode_mae = (pred_mean - targets).abs().mean(dim=(1, 2))
        ctx_lens = torch.tensor(batch["context_lens"], device=targets.device)

        few_shot_mask = ctx_lens < 64
        rich_context_mask = ctx_lens >= 256

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
# 5. DataModule & Production Training Entrypoint
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
            collate_fn=pad_icl_perceiver_collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pad_icl_perceiver_collate,
            num_workers=self.num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--episodes-per-epoch", type=int, default=32_000)  # 1,000 steps per epoch
    parser.add_argument("--val-episodes", type=int, default=1_600)          # 50 validation batches
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--m-latents", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)                    # head_dim = 64
    parser.add_argument("--num-latent-layers", type=int, default=10)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--num-bins", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)      # Slightly lower for d_model=512
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-context", type=int, default=32)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--num-queries", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed")
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
        m_latents=args.m_latents,
        nhead=args.nhead,
        num_latent_layers=args.num_latent_layers,
        dim_feedforward=args.dim_feedforward,
        num_bins=args.num_bins,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="tabular_chemeleon_perceiver",
        default_hp_metric=False,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=25,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=datamodule)

    model_path = Path(args.log_dir) / "tabularchemeleon_perceiver.pt"
    trainer.save_checkpoint(model_path, weights_only=True)
    print(f"Perceiver model successfully trained and saved to: {model_path}")
