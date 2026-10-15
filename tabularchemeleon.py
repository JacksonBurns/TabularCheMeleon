from __future__ import annotations

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
    from config import MIN_CONTEXT, MAX_CONTEXT, NUM_QUERIES, TRAIN_MIN_CONTEXT, TRAIN_MAX_CONTEXT
except ImportError:
    MIN_CONTEXT = 16
    MAX_CONTEXT = 5000
    NUM_QUERIES = 128
    TRAIN_MIN_CONTEXT = 16
    TRAIN_MAX_CONTEXT = 1024


# ==============================================================================
# 1. Manifold-Locked QSAR & Activity-Cliff Synthetic Prior
# ==============================================================================
class EmpiricalBioactivityPrior:
    """
    Infinite-task synthetic QSAR generator over the fixed 2048-D CheMeleon manifold.
    Combines:
      1. Fixed-Coordinate Sparse & Group-Sparse SwiGLU SCMs (exploits fixed coordinate identity)
      2. Multi-Scale Spectral Mixture GP surfaces (RBF + Matern-5/2 in raw & SVD-whitened spaces)
      3. Soft Oblique Decision Tree ensembles (piecewise / rule-based SAR)
      4. Cooperative Hill Kinetics & Hansch non-monotonic parabolas
      5. Matched-Molecular-Pair (MMP) directional gradients & sharp Activity Cliffs
    """

    def __init__(self, embed_dim: int = 2048):
        self.embed_dim = embed_dim

    def _sample_empirical_directions(self, x: torch.Tensor, k: int) -> torch.Tensor:
        n_samples = x.size(0)
        k_clamped = min(k, n_samples - 1, self.embed_dim)
        if k_clamped < 1:
            v = torch.randn(self.embed_dim, k, device=x.device)
            return F.normalize(v, p=2, dim=0)
        x_centered = x - x.mean(dim=0, keepdim=True)
        try:
            _, _, V = torch.pca_lowrank(x_centered, q=k_clamped)
            proj = V[:, :k_clamped]
            if k_clamped < k:
                pad = F.normalize(
                    torch.randn(self.embed_dim, k - k_clamped, device=x.device), p=2, dim=0
                )
                proj = torch.cat([proj, pad], dim=1)
            return proj
        except Exception:
            v = torch.randn(self.embed_dim, k, device=x.device)
            return F.normalize(v, p=2, dim=0)

    def _sparse_coordinate_scm(self, x: torch.Tensor) -> torch.Tensor:
        """Selects sparse fixed CheMeleon coordinates and applies a 2-layer SwiGLU SCM."""
        device = x.device
        n_active = random.randint(16, 192)
        idx = torch.randperm(self.embed_dim, device=device)[:n_active]
        x_sub = x[:, idx]  # (N, n_active)
        hidden = random.randint(32, 96)
        w1 = torch.randn(n_active, hidden * 2, device=device) / math.sqrt(n_active)
        b1 = torch.randn(1, hidden * 2, device=device) * 0.2
        h_raw = x_sub @ w1 + b1
        h1, h2 = h_raw.chunk(2, dim=-1)
        h = F.silu(h1) * h2
        w2 = torch.randn(hidden, 1, device=device) / math.sqrt(hidden)
        return h @ w2

    def _multiscale_kernel_gp(self, x: torch.Tensor) -> torch.Tensor:
        """Spectral mixture of RBF and Matern-5/2 kernels in raw or task-whitened space."""
        device = x.device
        n = x.size(0)
        if random.random() < 0.50 and n > 12:
            # Operate in local SVD-whitened (Mahalanobis) subspace
            xc = x - x.mean(dim=0, keepdim=True)
            g = xc @ xc.T
            evals, evecs = torch.linalg.eigh(g + 1e-4 * torch.eye(n, device=device))
            pos = evals > 1e-3
            if pos.sum() >= 4:
                z = evecs[:, pos] * math.sqrt(n)
            else:
                z = F.normalize(xc, p=2, dim=-1)
        else:
            # Sparse coordinate-weighted Euclidean space
            weights = torch.rand(1, self.embed_dim, device=device) ** random.uniform(1.5, 4.0)
            z = (x - x.mean(dim=0, keepdim=True)) * weights

        n_centers = min(n, random.randint(6, 24))
        c_idx = torch.randperm(n, device=device)[:n_centers]
        centers = z[c_idx]
        dist = torch.cdist(z, centers, p=2)
        med = torch.median(dist) + 1e-4

        y = torch.zeros(n, 1, device=device)
        # Sum 2-3 lengthscale regimes (mimicking Spectral Mixture Kernel)
        for _ in range(random.randint(2, 3)):
            ell = med * random.uniform(0.25, 2.2)
            amps = torch.randn(n_centers, 1, device=device)
            if random.random() < 0.5:
                k_mat = torch.exp(-0.5 * (dist / ell) ** 2)
            else:
                r = math.sqrt(5.0) * dist / ell
                k_mat = (1.0 + r + (r ** 2) / 3.0) * torch.exp(-r)
            y = y + (k_mat @ amps) / math.sqrt(n_centers)
        return y

    def _oblique_tree_ensemble(self, x: torch.Tensor) -> torch.Tensor:
        """Soft oblique decision tree ensemble of depth 3 over empirical/sparse directions."""
        device = x.device
        n = x.size(0)
        n_trees = random.randint(4, 10)
        k_dirs = n_trees * 3
        dirs = self._sample_empirical_directions(x, k_dirs)
        proj = x @ dirs  # (N, k_dirs)
        proj = (proj - proj.mean(0, keepdim=True)) / (proj.std(0, keepdim=True) + 1e-5)
        thresh = torch.empty(1, k_dirs, device=device).uniform_(-1.2, 1.2)
        temp = random.uniform(3.0, 10.0)
        gates = torch.sigmoid(temp * (proj - thresh)).view(n, n_trees, 3)

        g0, g1, g2 = gates[:, :, 0:1], gates[:, :, 1:2], gates[:, :, 2:3]
        leaves = torch.cat(
            [
                g0 * g1 * g2,
                g0 * g1 * (1 - g2),
                g0 * (1 - g1) * g2,
                g0 * (1 - g1) * (1 - g2),
                (1 - g0) * g1 * g2,
                (1 - g0) * g1 * (1 - g2),
                (1 - g0) * (1 - g1) * g2,
                (1 - g0) * (1 - g1) * (1 - g2),
            ],
            dim=-1,
        )  # (N, n_trees, 8)
        leaf_vals = torch.randn(1, n_trees, 8, device=device)
        return (leaves * leaf_vals).sum(dim=(1, 2)).unsqueeze(-1) / math.sqrt(n_trees)

    def _hill_and_parabola_sar(self, x: torch.Tensor) -> torch.Tensor:
        """Combines Free-Wilson directional effects with Hansch optima and Hill saturation."""
        device = x.device
        k_axes = random.randint(4, 10)
        proj_mat = self._sample_empirical_directions(x, k_axes)
        p = x @ proj_mat
        p = (p - p.mean(0, keepdim=True)) / (p.std(0, keepdim=True) + 1e-5)

        weights = torch.randn(k_axes, 1, device=device)
        y = p @ weights
        if k_axes >= 2:
            # Hansch lipophilicity parabola + bilinear synergy
            opt = random.uniform(-0.8, 0.8)
            y = y - random.uniform(0.4, 1.2) * ((p[:, 0:1] - opt) ** 2)
            y = y + random.uniform(-1.2, 1.2) * (p[:, 0:1] * p[:, 1:2])
        if random.random() < 0.5:
            y = torch.tanh(y * random.uniform(0.6, 1.5)) * 3.0
        return y

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        n_samples = x.size(0)
        x_std = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-5)

        r = random.random()
        if r < 0.30:
            y = self._sparse_coordinate_scm(x_std)
        elif r < 0.65:
            y = self._multiscale_kernel_gp(x_std)
        elif r < 0.85:
            y = self._oblique_tree_ensemble(x_std)
        else:
            y = self._hill_and_parabola_sar(x_std)

        # Blend with secondary mechanism 40% of the time for rich multi-scale structure
        if random.random() < 0.40:
            y2 = self._sparse_coordinate_scm(x_std)
            y = (y / (y.std() + 1e-5)) + 0.35 * (y2 / (y2.std() + 1e-5))

        y = (y - y.mean()) / (y.std() + 1e-6)

        # Inject Matched-Molecular-Pair (MMP) local activity cliffs on close analogs
        if random.random() < 0.55 and n_samples > 8:
            x_norm = F.normalize(x_std, p=2, dim=-1)
            sim_matrix = x_norm @ x_norm.T
            sim_matrix.fill_diagonal_(-1.0)
            high_sim_pairs = torch.nonzero((sim_matrix > 0.75) & (sim_matrix < 0.985))
            if len(high_sim_pairs) > 0:
                cliff_count = min(max(2, n_samples // 25), len(high_sim_pairs))
                perm = torch.randperm(len(high_sim_pairs), device=device)[:cliff_count]
                cliff_dir = torch.randn(self.embed_dim, 1, device=device) / math.sqrt(self.embed_dim)
                for p_idx in perm:
                    i_mol = high_sim_pairs[p_idx, 0].item()
                    j_mol = high_sim_pairs[p_idx, 1].item()
                    delta = ((x_std[j_mol : j_mol + 1] - x_std[i_mol : i_i + 1 if (i_i := i_mol) >= 0 else 1]) @ cliff_dir).item()
                    sign = 1.0 if delta >= 0 else -1.0
                    mag = random.uniform(1.5, 3.2) * sign
                    y[j_mol] = y[i_mol] + mag

        # Observation noise (Gaussian or heavy-tailed Student-t)
        noise_std = random.uniform(0.02, 0.15)
        if random.random() < 0.20:
            noise = torch.distributions.StudentT(df=3.5).sample(y.shape).to(device) * (noise_std * 0.6)
        else:
            noise = torch.randn_like(y) * noise_std
        y = y + noise

        # Assay plate censoring 15% of the time
        if random.random() < 0.15:
            if random.random() < 0.5:
                y = torch.clamp(y, min=torch.quantile(y, random.uniform(0.03, 0.10)))
            else:
                y = torch.clamp(y, max=torch.quantile(y, random.uniform(0.90, 0.97)))

        return (y - y.mean()) / (y.std() + 1e-6)


# ==============================================================================
# 2. Distance- & Scaffold-Stratified Episodic Dataset (Zero-Copy Shared RAM)
# ==============================================================================
class ChemicalEpisodicDataset(Dataset):
    """
    Zero-copy Episodic Dataset referencing a shared embeddings tensor via valid_indices.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        valid_indices: Optional[torch.Tensor] = None,
        min_context: int = TRAIN_MIN_CONTEXT,
        max_context: int = TRAIN_MAX_CONTEXT,
        num_queries: int = NUM_QUERIES,
        num_episodes: int = 16_000,
    ):
        # Keep underlying storage as-is (float16 or float32) in shared memory; cast only per episode
        self.embeddings = embeddings
        self.embed_dim = embeddings.shape[1]
        self.valid_indices = (
            valid_indices if valid_indices is not None else torch.arange(len(embeddings), dtype=torch.int64)
        )
        self.n_samples = len(self.valid_indices)
        self.min_context = min_context
        self.max_context = max_context
        self.num_queries = num_queries
        self.num_episodes = num_episodes
        self.prior = EmpiricalBioactivityPrior(embed_dim=self.embed_dim)

    def __len__(self) -> int:
        return self.num_episodes

    def _sample_episode_split(self, n_context: int, n_query: int) -> Tuple[torch.Tensor, torch.Tensor]:
        total_needed = n_context + n_query
        p = random.random()
        pool_size = min(self.n_samples, max(2048, total_needed * 2))

        # Use O(pool_size) randint instead of O(N_total) randperm to avoid RAM churn
        local_idx = torch.randint(0, self.n_samples, (pool_size,))
        pool_idx = self.valid_indices[local_idx]
        pool = self.embeddings[pool_idx].float()
        seed = pool[0:1]
        sims = F.cosine_similarity(seed, pool, dim=-1)

        if p < 0.25:
            # Mode A: Tight Congeneric Series
            topk = torch.topk(sims, k=min(total_needed, len(pool)), largest=True).indices
            selected = pool_idx[topk]
            selected = selected[torch.randperm(len(selected))]
            return selected[:n_context], selected[n_context:total_needed]

        elif p < 0.55:
            # Mode B: Explicit Scaffold-Shift Split
            topk = torch.topk(sims, k=min(total_needed, len(pool)), largest=True).indices
            ordered = pool_idx[topk]
            if random.random() < 0.7:
                n_shift = int(n_query * 0.8)
                q_shift = ordered[n_context : n_context + n_shift]
                rem = torch.cat([ordered[:n_context], ordered[n_context + n_shift :]])
                rem = rem[torch.randperm(len(rem))]
                q_rand = rem[: (n_query - n_shift)]
                ctx_idx = rem[(n_query - n_shift) : (n_query - n_shift + n_context)]
                q_idx = torch.cat([q_shift, q_rand])[torch.randperm(n_query)]
                return ctx_idx, q_idx
            else:
                ordered = ordered[torch.randperm(len(ordered))]
                return ordered[:n_context], ordered[n_context:total_needed]

        elif p < 0.80:
            # Mode C: Multi-Family Series (2 distinct scaffolds)
            half = total_needed // 2
            rem_need = total_needed - half
            sims_b = F.cosine_similarity(pool[1:2], pool, dim=-1)
            topk_a = torch.topk(sims, k=min(half, len(pool)), largest=True).indices
            topk_b = torch.topk(sims_b, k=min(rem_need, len(pool)), largest=True).indices
            selected = torch.cat([pool_idx[topk_a], pool_idx[topk_b]])
            selected = selected[torch.randperm(len(selected))]
            return selected[:n_context], selected[n_context:total_needed]

        else:
            # Mode D: Global Diverse Library
            selected = pool_idx[:total_needed]
            return selected[:n_context], selected[n_context:total_needed]

    def __getitem__(self, idx: int):
        log_min = math.log2(max(16, self.min_context))
        log_max = math.log2(max(32, self.max_context))
        n_context = int(round(2 ** random.uniform(log_min, log_max)))
        n_context = max(self.min_context, min(self.max_context, n_context))
        total_needed = n_context + self.num_queries

        ctx_indices, q_indices = self._sample_episode_split(n_context, self.num_queries)
        all_indices = torch.cat([ctx_indices, q_indices], dim=0)
        if len(all_indices) < total_needed:
            deficit = total_needed - len(all_indices)
            extra = self.valid_indices[torch.randint(0, self.n_samples, (deficit,))]
            all_indices = torch.cat([all_indices, extra])

        # Cast only the sampled episode to float32
        x_selected = self.embeddings[all_indices[:total_needed]].float()
        y_selected = self.prior(x_selected)

        y_ctx_raw = y_selected[:n_context]
        y_q_raw = y_selected[n_context:]
        ym = y_ctx_raw.mean()
        ys = y_ctx_raw.std().clamp_min(1e-5)
        y_ctx = ((y_ctx_raw - ym) / ys).clamp(-4.5, 4.5)
        y_query = ((y_q_raw - ym) / ys).clamp(-4.5, 4.5)

        return {
            "x_ctx": x_selected[:n_context],
            "y_ctx": y_ctx,
            "x_query": x_selected[n_context:],
            "y_query": y_query,
            "n_ctx": n_context,
            "n_query": self.num_queries,
        }


def pad_icl_perceiver_collate(batch):
    """
    Pads variable-length context sets to max_ctx in batch while building ctx_key_padding_mask.
    """
    batch_size = len(batch)
    max_ctx = max(item["n_ctx"] for item in batch)
    query_len = batch[0]["n_query"]
    embed_dim = batch[0]["x_ctx"].size(-1)

    x_ctx_padded = torch.zeros(batch_size, max_ctx, embed_dim)
    y_ctx_padded = torch.zeros(batch_size, max_ctx, 1)
    ctx_key_padding_mask = torch.ones(batch_size, max_ctx, dtype=torch.bool)  # True = Padding

    x_query = torch.zeros(batch_size, query_len, embed_dim)
    y_query = torch.zeros(batch_size, query_len, 1)
    context_lens = []

    for i, item in enumerate(batch):
        nc = item["n_ctx"]
        context_lens.append(nc)
        x_ctx_padded[i, :nc] = item["x_ctx"]
        y_ctx_padded[i, :nc] = item["y_ctx"]
        ctx_key_padding_mask[i, :nc] = False  # False = Valid context token
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
# 3. Analytic Leave-One-Out (LOO) Multi-Metric Basis Bank
# ==============================================================================
class AnalyticLOOBasisBank(nn.Module):
    """
    Zero-leakage, in-graph closed-form solver executed in float32.
    Constructs three task-specific chemical geometries strictly from (x_ctx, y_ctx):
      1. Raw Standardized CheMeleon Space (2048-D ambient geometry)
      2. Dual-Gram SVD-Whitened Mahalanobis Space (denoised low-rank manifold for N < 500)
      3. Shrinkage-Correlation (|r_xy|-Weighted) Supervised Subspace
    Evaluates 18 closed-form basis experts (multi-scale RBF, Matern-5/2, Dual Ridge, and multi-k kNN)
    and computes exact Leave-One-Out (LOO) predictions, residuals, and simplex stacking weights.
    """

    NUM_EXPERTS = 18

    def __init__(self, embed_dim: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim

    @staticmethod
    def _krr_with_exact_loo(
        K_cc: torch.Tensor,
        K_qc: torch.Tensor,
        y_c: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves (K_cc + alpha * I) beta = y_c and uses the Craven-Wahba identity:
            y_loo_i = y_i - beta_i / (A^{-1})_{ii}
        to obtain exact Leave-One-Out predictions on all N_ctx training molecules simultaneously.
        """
        n = K_cc.size(0)
        device = K_cc.device
        reg = max(alpha, 1e-4)
        A = K_cc + reg * torch.eye(n, device=device, dtype=torch.float32)

        L, info = torch.linalg.cholesky_ex(A)
        if info.item() != 0:
            A = K_cc + max(reg * 10.0, 1e-2) * torch.eye(n, device=device, dtype=torch.float32)
            L, _ = torch.linalg.cholesky_ex(A)
            L = torch.nan_to_num(L, nan=1.0)

        beta = torch.cholesky_solve(y_c, L)  # (N, 1)
        A_inv = torch.cholesky_inverse(L)    # (N, N)
        diag_inv = A_inv.diagonal().unsqueeze(-1).clamp_min(1e-6)
        loo_err = (beta / diag_inv).clamp(-6.0, 6.0)
        y_loo = (y_c - loo_err).clamp(-5.0, 5.0)
        y_q = (K_qc @ beta).clamp(-5.0, 5.0)
        return y_loo.squeeze(-1), y_q.squeeze(-1)

    @staticmethod
    def _knn_with_exact_loo(
        dist_cc: torch.Tensor,
        dist_qc: torch.Tensor,
        y_c: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse-distance weighted kNN with diagonal-masked exact Leave-One-Out on context."""
        n = dist_cc.size(0)
        kk = max(1, min(k, n - 1))
        y_flat = y_c.squeeze(-1)  # (N,)

        # Context LOO: mask self-distance with +inf
        d_cc = dist_cc.clone()
        d_cc.fill_diagonal_(float("inf"))
        d_top_c, idx_c = torch.topk(d_cc, k=kk, dim=-1, largest=False)
        w_c = 1.0 / (d_top_c + 1e-4)
        w_c = w_c / w_c.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        y_loo = (w_c * y_flat[idx_c]).sum(dim=-1)

        # Query prediction
        d_top_q, idx_q = torch.topk(dist_qc, k=kk, dim=-1, largest=False)
        w_q = 1.0 / (d_top_q + 1e-4)
        w_q = w_q / w_q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        y_q = (w_q * y_flat[idx_q]).sum(dim=-1)
        return y_loo.clamp(-5.0, 5.0), y_q.clamp(-5.0, 5.0)

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        x_ctx_loc: torch.Tensor,
        y_ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
        x_query_loc: torch.Tensor,
        r_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x_ctx_loc: (B, N_ctx, D) float32 locally-standardized context embeddings
            y_ctx: (B, N_ctx, 1) float32 standardized context labels
            ctx_mask: (B, N_ctx) bool padding mask (True = padded)
            x_query_loc: (B, N_q, D) float32 locally-standardized query embeddings
            r_xy: (B, D) float32 empirical feature-target correlations
        """
        B, N_max, D = x_ctx_loc.shape
        N_q = x_query_loc.size(1)
        device = x_ctx_loc.device
        M = self.NUM_EXPERTS

        y_loo_all = torch.zeros(B, N_max, M, device=device, dtype=torch.float32)
        y_q_all = torch.zeros(B, N_q, M, device=device, dtype=torch.float32)
        loo_mse_all = torch.ones(B, M, device=device, dtype=torch.float32)
        loo_corr_all = torch.zeros(B, M, device=device, dtype=torch.float32)
        w_loo_all = torch.full((B, M), 1.0 / M, device=device, dtype=torch.float32)

        # Pairwise similarity matrices for Transformer attention bias: (B, N_q, N_max) & (B, N_max, N_max)
        sim_qc_bank = torch.zeros(B, 4, N_q, N_max, device=device, dtype=torch.float32)
        sim_cc_bank = torch.zeros(B, 4, N_max, N_max, device=device, dtype=torch.float32)
        eff_rank_ratio = torch.zeros(B, 1, device=device, dtype=torch.float32)

        for b in range(B):
            valid = ~ctx_mask[b]
            nc = int(valid.sum().item())
            if nc < 4:
                continue

            xc = x_ctx_loc[b, :nc].float()   # (nc, D)
            yc = y_ctx[b, :nc].float()       # (nc, 1)
            xq = x_query_loc[b].float()      # (N_q, D)
            corr = r_xy[b].float()           # (D,)

            # 1. Space A: Raw Standardized Space
            d_cc_raw = torch.cdist(xc, xc, p=2)
            d_qc_raw = torch.cdist(xq, xc, p=2)
            triu_idx = torch.triu_indices(nc, nc, offset=1, device=device)
            med_raw = (
                torch.median(d_cc_raw[triu_idx[0], triu_idx[1]]).clamp_min(1e-2)
                if triu_idx.numel() > 0
                else torch.tensor(math.sqrt(D), device=device)
            )

            # Cosine similarity
            xc_n = F.normalize(xc, p=2, dim=-1)
            xq_n = F.normalize(xq, p=2, dim=-1)
            cos_cc = (xc_n @ xc_n.T).clamp(-1.0, 1.0)
            cos_qc = (xq_n @ xc_n.T).clamp(-1.0, 1.0)

            # 2. Space B: Fast Dual-Gram SVD-Whitened Mahalanobis Space
            xc_mean = xc.mean(dim=0, keepdim=True)
            xc_c = xc - xc_mean
            xq_c = xq - xc_mean
            G_cc = xc_c @ xc_c.T  # (nc, nc)
            G_qc = xq_c @ xc_c.T  # (N_q, nc)
            evals, evecs = torch.linalg.eigh(G_cc + 1e-5 * torch.eye(nc, device=device))
            evals = evals.clamp_min(0.0)
            # Flip to descending order
            evals = evals.flip(0)
            evecs = evecs.flip(1)
            total_var = evals.sum().clamp_min(1e-6)
            cum_var = torch.cumsum(evals, dim=0) / total_var
            r_keep = int(torch.searchsorted(cum_var, torch.tensor(0.999, device=device)).item()) + 1
            r_keep = max(2, min(r_keep, nc - 1))
            eff_rank_ratio[b, 0] = float(r_keep) / float(nc)

            s_r = evals[:r_keep].clamp_min(1e-4)
            u_r = evecs[:, :r_keep]  # (nc, r_keep)
            # Exact whitened coordinates: Z_ctx = U_r, Z_query = G_qc @ U_r @ diag(1 / s_r)
            z_c_white = u_r * math.sqrt(r_keep)
            z_q_white = (G_qc @ (u_r / s_r.unsqueeze(0))) * math.sqrt(r_keep)
            d_cc_white = torch.cdist(z_c_white, z_c_white, p=2)
            d_qc_white = torch.cdist(z_q_white, z_c_white, p=2)
            med_white = (
                torch.median(d_cc_white[triu_idx[0], triu_idx[1]]).clamp_min(1e-2)
                if triu_idx.numel() > 0
                else torch.tensor(math.sqrt(r_keep), device=device)
            )

            # 3. Space C: Correlation-Weighted (|r_xy|) Supervised Subspace
            w_c = corr.abs().clamp_min(0.02)
            w_c = w_c / (w_c.pow(2).mean().sqrt() + 1e-6)
            xc_corr = xc * w_c.unsqueeze(0)
            xq_corr = xq * w_c.unsqueeze(0)
            d_cc_corr = torch.cdist(xc_corr, xc_corr, p=2)
            d_qc_corr = torch.cdist(xq_corr, xc_corr, p=2)
            med_corr = (
                torch.median(d_cc_corr[triu_idx[0], triu_idx[1]]).clamp_min(1e-2)
                if triu_idx.numel() > 0
                else med_raw
            )

            # Store normalized similarity kernels for Transformer attention heads
            sim_cc_bank[b, 0, :nc, :nc] = cos_cc
            sim_cc_bank[b, 1, :nc, :nc] = torch.exp(-0.5 * (d_cc_raw / med_raw) ** 2)
            sim_cc_bank[b, 2, :nc, :nc] = torch.exp(-0.5 * (d_cc_white / med_white) ** 2)
            sim_cc_bank[b, 3, :nc, :nc] = torch.exp(-0.5 * (d_cc_corr / med_corr) ** 2)

            sim_qc_bank[b, 0, :, :nc] = cos_qc
            sim_qc_bank[b, 1, :, :nc] = torch.exp(-0.5 * (d_qc_raw / med_raw) ** 2)
            sim_qc_bank[b, 2, :, :nc] = torch.exp(-0.5 * (d_qc_white / med_white) ** 2)
            sim_qc_bank[b, 3, :, :nc] = torch.exp(-0.5 * (d_qc_corr / med_corr) ** 2)

            def matern52(d: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
                r = math.sqrt(5.0) * d / ell
                return (1.0 + r + (r ** 2) / 3.0) * torch.exp(-r)

            def rbf(d: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
                return torch.exp(-0.5 * (d / ell) ** 2)

            experts_loo = []
            experts_q = []

            # --- Group 1: Raw Space Kernels (6 experts: multi-scale RBF + Matern-5/2) ---
            for l_frac, alpha in [(0.4, 0.05), (0.85, 0.08), (1.6, 0.15)]:
                ell = med_raw * l_frac
                l_p, q_p = self._krr_with_exact_loo(rbf(d_cc_raw, ell), rbf(d_qc_raw, ell), yc, alpha)
                experts_loo.append(l_p)
                experts_q.append(q_p)
            for l_frac, alpha in [(0.5, 0.05), (1.0, 0.10), (1.8, 0.20)]:
                ell = med_raw * l_frac
                l_p, q_p = self._krr_with_exact_loo(
                    matern52(d_cc_raw, ell), matern52(d_qc_raw, ell), yc, alpha
                )
                experts_loo.append(l_p)
                experts_q.append(q_p)

            # --- Group 2: SVD-Whitened Mahalanobis Kernels (5 experts: Matern-5/2 + RBF + Whitened Ridge) ---
            for l_frac, alpha in [(0.6, 0.08), (1.2, 0.12), (2.2, 0.20)]:
                ell = med_white * l_frac
                l_p, q_p = self._krr_with_exact_loo(
                    matern52(d_cc_white, ell), matern52(d_qc_white, ell), yc, alpha
                )
                experts_loo.append(l_p)
                experts_q.append(q_p)
            for l_frac, alpha in [(0.8, 0.10), (1.5, 0.25)]:
                ell = med_white * l_frac
                l_p, q_p = self._krr_with_exact_loo(
                    rbf(d_cc_white, ell), rbf(d_qc_white, ell), yc, alpha
                )
                experts_loo.append(l_p)
                experts_q.append(q_p)

            # --- Group 3: Correlation-Filtered Supervised Kernels (4 experts) ---
            for l_frac, alpha in [(0.6, 0.08), (1.2, 0.15)]:
                ell = med_corr * l_frac
                l_p, q_p = self._krr_with_exact_loo(
                    rbf(d_cc_corr, ell), rbf(d_qc_corr, ell), yc, alpha
                )
                experts_loo.append(l_p)
                experts_q.append(q_p)
            for l_frac, alpha in [(0.8, 0.10), (1.6, 0.20)]:
                ell = med_corr * l_frac
                l_p, q_p = self._krr_with_exact_loo(
                    matern52(d_cc_corr, ell), matern52(d_qc_corr, ell), yc, alpha
                )
                experts_loo.append(l_p)
                experts_q.append(q_p)

            # --- Group 4: Dual Linear Ridge & Multi-k Local kNN (3 experts) ---
            # Dual Linear Ridge (scaled by D)
            l_p, q_p = self._krr_with_exact_loo(G_cc / D, G_qc / D, yc, alpha=0.5)
            experts_loo.append(l_p)
            experts_q.append(q_p)
            # Local kNN (k=5 raw Euclidean, k=12 cosine)
            l_knn5, q_knn5 = self._knn_with_exact_loo(d_cc_raw, d_qc_raw, yc, k=5)
            experts_loo.append(l_knn5)
            experts_q.append(q_knn5)
            l_knn12, q_knn12 = self._knn_with_exact_loo(1.0 - cos_cc, 1.0 - cos_qc, yc, k=12)
            experts_loo.append(l_knn12)
            experts_q.append(q_knn12)

            Y_loo_b = torch.stack(experts_loo, dim=-1)  # (nc, M)
            Y_q_b = torch.stack(experts_q, dim=-1)      # (N_q, M)
            y_loo_all[b, :nc] = Y_loo_b
            y_q_all[b] = Y_q_b

            # Compute exact LOO MSE and LOO Pearson r per expert on the active support set
            yc_flat = yc.squeeze(-1)  # (nc,)
            mse_b = ((Y_loo_b - yc_flat.unsqueeze(-1)) ** 2).mean(dim=0)  # (M,)
            yc_c = yc_flat - yc_flat.mean()
            yl_c = Y_loo_b - Y_loo_b.mean(dim=0, keepdim=True)
            corr_b = (yl_c * yc_c.unsqueeze(-1)).sum(dim=0) / (
                yl_c.norm(dim=0) * yc_c.norm() + 1e-6
            )
            loo_mse_all[b] = mse_b
            loo_corr_all[b] = corr_b.clamp(-1.0, 1.0)

            # Analytic LOO stacking weights: combine temperature-scaled LOO skill with non-negative ridge
            rel_mse = mse_b - mse_b.min()
            temp = max(4.0, min(25.0, math.sqrt(nc) * 0.8))
            log_score = -temp * rel_mse + 2.0 * corr_b.clamp_min(0.0)
            w_soft = F.softmax(log_score, dim=-1)
            w_loo_all[b] = w_soft

        return {
            "y_loo": y_loo_all,              # (B, N_max, M)
            "y_q_basis": y_q_all,            # (B, N_q, M)
            "loo_mse": loo_mse_all,          # (B, M)
            "loo_corr": loo_corr_all,        # (B, M)
            "w_loo": w_loo_all,              # (B, M)
            "sim_cc": sim_cc_bank,           # (B, 4, N_max, N_max)
            "sim_qc": sim_qc_bank,           # (B, 4, N_q, N_max)
            "eff_rank_ratio": eff_rank_ratio # (B, 1)
        }


# ==============================================================================
# 4. Fixed-Coordinate Subspace Tokenizer & Relational Delta-ICL Architecture
# ==============================================================================
class CoordinateHyperGate(nn.Module):
    """
    Exploits the fixed 2048-D CheMeleon coordinate identity before any projection:
      1. Modulates all 2,048 dimensions using shrinkage-stabilized support-set correlation r_xy
         and support-set log-standard deviation.
      2. Projects into K = 8 semantic chemical subspace tokens per molecule.
      3. Runs Intra-Molecule Subspace Self-Attention across the K = 8 tokens.
    """

    def __init__(self, embed_dim: int = 2048, n_subspaces: int = 8, d_sub: int = 96, d_model: int = 384):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_subspaces = n_subspaces
        self.d_sub = d_sub
        self.d_model = d_model

        # Learned per-coordinate sensitivity weights locked to CheMeleon's 2048 dimensions
        self.coord_gain = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.coord_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Dual projection: fixed coordinate chunks + full dense cross-coordinate projection
        self.dense_subspace_proj = nn.Linear(embed_dim * 2, n_subspaces * d_sub)
        self.subspace_pos = nn.Parameter(torch.randn(1, 1, n_subspaces, d_sub) * 0.02)
        self.subspace_norm1 = nn.LayerNorm(d_sub)
        self.subspace_attn = nn.MultiheadAttention(
            embed_dim=d_sub, num_heads=4, dropout=0.05, batch_first=True
        )
        self.subspace_ffn = nn.Sequential(
            nn.LayerNorm(d_sub),
            nn.Linear(d_sub, d_sub * 2),
            nn.GELU(),
            nn.Linear(d_sub * 2, d_sub),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(n_subspaces * d_sub),
            nn.Linear(n_subspaces * d_sub, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        x_glob: torch.Tensor,
        x_loc: torch.Tensor,
        r_xy: torch.Tensor,
        log_sigma_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_glob: (B, N, 2048) pool-standardized embeddings
            x_loc:  (B, N, 2048) task-standardized embeddings
            r_xy:   (B, 2048) shrinkage-stabilized feature-target correlation
            log_sigma_ctx: (B, 2048) task log standard deviation
        Returns:
            h_mol: (B, N, d_model)
        """
        B, N, D = x_glob.shape
        K, d_s = self.n_subspaces, self.d_sub

        gate_in = torch.cat([r_xy, log_sigma_ctx], dim=-1)  # (B, 2*D)
        g = 2.0 * torch.sigmoid(
            self.gate_net(gate_in).unsqueeze(1)
            + self.coord_gain * r_xy.abs().unsqueeze(1) * 2.0
            + self.coord_bias
        )  # (B, 1, D)

        x_gated = x_loc * g
        x_cat = torch.cat([x_gated, x_glob], dim=-1)  # (B, N, 2*D)
        sub_tokens = self.dense_subspace_proj(x_cat).view(B, N, K, d_s) + self.subspace_pos

        # Intra-molecule subspace self-attention: (B*N, K, d_s)
        flat = sub_tokens.view(B * N, K, d_s)
        flat_norm = self.subspace_norm1(flat)
        attn_out, _ = self.subspace_attn(flat_norm, flat_norm, flat_norm, need_weights=False)
        flat = flat + attn_out
        flat = flat + self.subspace_ffn(flat)

        pooled = flat.reshape(B, N, K * d_s)
        return self.out_proj(pooled)


class ContinuousLabelEncoder(nn.Module):
    """Encodes standardized scalar targets y via multi-frequency Fourier + linear features."""

    def __init__(self, d_model: int):
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(0.25), math.log(8.0), 16))
        self.register_buffer("freqs", freqs)
        self.net = nn.Sequential(
            nn.Linear(1 + 32, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, N, 1)
        angles = y * self.freqs.view(1, 1, -1)
        fourier = torch.cat([torch.sin(angles), torch.cos(angles), y], dim=-1)
        return self.net(fourier)


class RelationalDeltaICLBlock(nn.Module):
    """
    Inductive Set-to-Set Transformer Block with Multi-Metric Kernel Bias:
      - Context tokens self-attend to refine local neighborhood & cliff representations.
      - Query tokens cross-attend strictly to Context tokens (queries never attend to queries).
      - Attention logits incorporate learned head-specific weights over the 4 chemical
        similarity matrices (Cosine, Raw RBF, SVD-Whitened RBF, Correlation-Weighted RBF).
    """

    def __init__(self, d_model: int = 384, nhead: int = 8, dim_feedforward: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Head-specific weights over the 4 chemical similarity matrices
        self.cc_sim_weights = nn.Parameter(torch.zeros(nhead, 4))
        self.qc_sim_weights = nn.Parameter(torch.zeros(nhead, 4))
        nn.init.constant_(self.cc_sim_weights[:, 0], 0.5)
        nn.init.constant_(self.qc_sim_weights[:, 0], 0.8)

        # Context Self-Attention
        self.norm_c1 = nn.LayerNorm(d_model)
        self.c_qkv = nn.Linear(d_model, d_model * 3)
        self.c_proj = nn.Linear(d_model, d_model)
        self.c_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Query-to-Context Cross-Attention (Decoupled Key & Value projections)
        self.norm_q1 = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_out = nn.Linear(d_model, d_model)
        self.q_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _biased_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sim_bank: torch.Tensor,
        sim_weights: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L_q, _ = q.shape
        L_k = k.size(1)
        H, d_h = self.nhead, self.head_dim

        q_h = q.view(B, L_q, H, d_h).transpose(1, 2)  # (B, H, L_q, d_h)
        k_h = k.view(B, L_k, H, d_h).transpose(1, 2)  # (B, H, L_k, d_h)
        v_h = v.view(B, L_k, H, d_h).transpose(1, 2)  # (B, H, L_k, d_h)

        # Compute multi-metric similarity bias: (B, H, L_q, L_k)
        # sim_bank: (B, 4, L_q, L_k), sim_weights: (H, 4)
        attn_bias = torch.einsum("bmlk,hm->bhlk", sim_bank.to(q.dtype), sim_weights.to(q.dtype))
        if key_padding_mask is not None:
            attn_bias = attn_bias.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        out = F.scaled_dot_product_attention(
            q_h, k_h, v_h, attn_mask=attn_bias, dropout_p=self.dropout.p if self.training else 0.0
        )
        return out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

    def forward(
        self,
        h_ctx: torch.Tensor,
        h_query: torch.Tensor,
        sim_cc: torch.Tensor,
        sim_qc: torch.Tensor,
        ctx_key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Context Self-Attention
        c_norm = self.norm_c1(h_ctx)
        cq, ck, cv = self.c_qkv(c_norm).chunk(3, dim=-1)
        c_attn = self._biased_mha(
            cq, ck, cv, sim_cc, self.cc_sim_weights, ctx_key_padding_mask
        )
        h_ctx = h_ctx + self.dropout(self.c_proj(c_attn))
        h_ctx = h_ctx + self.dropout(self.c_ffn(h_ctx))

        # 2. Query-to-Context Cross-Attention
        q_norm = self.norm_q1(h_query)
        kv_norm = self.norm_kv(h_ctx)
        qq = self.q_proj(q_norm)
        kk = self.k_proj(kv_norm)
        vv = self.v_proj(kv_norm)
        q_attn = self._biased_mha(
            qq, kk, vv, sim_qc, self.qc_sim_weights, ctx_key_padding_mask
        )
        h_query = h_query + self.dropout(self.q_out(q_attn))
        h_query = h_query + self.dropout(self.q_ffn(h_query))
        return h_ctx, h_query


class LatentContextPerceiverTransformer(nn.Module):
    """
    CheMeleon-PFN (Dual-Resolution Relational Transformer Neural Process).
    Retains class name `LatentContextPerceiverTransformer` so existing scripts and checkpoints
    import without breaking, while replacing the lossy Perceiver bottleneck with:
      1. Global + Local Standardization & Shrinkage-Correlation Feature Gating (CoordinateHyperGate)
      2. Analytic Leave-One-Out (LOO) Multi-Metric Basis Bank (18 experts across Raw, SVD-Whitened, and Corr spaces)
      3. Full-Context Relational Delta-ICL Blocks (exact Query-to-Context cross-attention)
      4. Matched-Molecular-Pair (Delta-x) Local Cliff Correction & LOO-Calibrated Residual Distribution Head
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        d_model: int = 384,
        m_latents: int = 256,  # Kept in signature for CLI backward compatibility
        nhead: int = 8,
        num_latent_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        num_bins: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_bins = num_bins
        self.num_experts = AnalyticLOOBasisBank.NUM_EXPERTS

        self.basis_bank = AnalyticLOOBasisBank(embed_dim=embed_dim)
        self.coord_tokenizer = CoordinateHyperGate(
            embed_dim=embed_dim, n_subspaces=8, d_sub=96, d_model=d_model
        )
        self.label_encoder = ContinuousLabelEncoder(d_model=d_model)

        # Inject context LOO predictions & LOO residuals into context tokens
        self.ctx_basis_proj = nn.Sequential(
            nn.Linear(self.num_experts * 2 + 1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        # Inject query basis predictions + neighbor distance leverage into query tokens
        self.query_basis_proj = nn.Sequential(
            nn.Linear(self.num_experts + 6, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Stack of Relational Delta-ICL Transformer layers
        n_layers = min(max(4, num_latent_layers), 8)
        self.layers = nn.ModuleList(
            [
                RelationalDeltaICLBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Explicit Matched-Molecular-Pair (Delta-x) Local Cliff Aggregator over top-16 neighbors
        self.mmp_delta_net = nn.Sequential(
            nn.Linear(d_model * 2 + 4, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2),  # [delta_y_shift, pair_logit_weight]
        )

        # Query-adaptive expert gate + non-parametric residual readout
        # Task summary features: loo_mse (M) + loo_corr (M) + log(N_ctx) + eff_rank_ratio
        task_feat_dim = self.num_experts * 2 + 2
        self.expert_gate_head = nn.Sequential(
            nn.Linear(d_model + task_feat_dim, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, self.num_experts),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model + task_feat_dim),
            nn.Linear(d_model + task_feat_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # [residual_mean, log_std_offset]
        )
        self.bin_residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_bins),
        )

        # Zero-init residual gates so epoch-0 starts directly at the LOO-optimal basis stack!
        nn.init.zeros_(self.expert_gate_head[-1].weight)
        nn.init.zeros_(self.expert_gate_head[-1].bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.bin_residual_head[-1].weight)
        nn.init.zeros_(self.bin_residual_head[-1].bias)
        self.gate_scale = nn.Parameter(torch.tensor(0.25))
        self.res_scale = nn.Parameter(torch.tensor(0.15))
        self.mmp_scale = nn.Parameter(torch.tensor(0.10))

    @staticmethod
    def _compute_support_stats(
        x_ctx_glob: torch.Tensor,
        y_ctx: torch.Tensor,
        ctx_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes support-set mean, std, and shrinkage-stabilized correlation r_xy."""
        valid = (~ctx_mask).float().unsqueeze(-1)  # (B, N_ctx, 1)
        n_valid = valid.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1, 1)

        mu_ctx = (x_ctx_glob * valid).sum(dim=1, keepdim=True) / n_valid
        var_ctx = (((x_ctx_glob - mu_ctx) ** 2) * valid).sum(dim=1, keepdim=True) / n_valid
        sd_ctx = torch.sqrt(var_ctx + 1e-4)

        ym = (y_ctx * valid).sum(dim=1, keepdim=True) / n_valid
        xc = (x_ctx_glob - mu_ctx) * valid
        yc = (y_ctx - ym) * valid
        cov = (xc * yc).sum(dim=1)  # (B, D)
        denom = (
            torch.sqrt((xc ** 2).sum(dim=1) * (yc ** 2).sum(dim=1) + 1e-6)
        )
        r_raw = (cov / denom).clamp(-1.0, 1.0)
        # Empirical Bayes shrinkage for small sample sizes
        n_s = n_valid.squeeze(-1)  # (B, 1)
        shrink = torch.sqrt(n_s / (n_s + 24.0))
        r_xy = r_raw * shrink
        return mu_ctx.squeeze(1), sd_ctx.squeeze(1), r_xy

    def forward_detailed(
        self,
        x_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        ctx_key_padding_mask: torch.Tensor,
        x_query: torch.Tensor,
        pool_mean: Optional[torch.Tensor] = None,
        pool_std: Optional[torch.Tensor] = None,
        bin_centers: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        B, N_ctx, D = x_ctx.shape
        N_q = x_query.size(1)

        # 1. Global pool standardization
        if pool_mean is not None and pool_std is not None:
            p_mu = pool_mean.view(1, 1, D)
            p_sd = pool_std.view(1, 1, D).clamp_min(1e-6)
            x_ctx_glob = (x_ctx - p_mu) / p_sd
            x_query_glob = (x_query - p_mu) / p_sd
        else:
            x_ctx_glob = x_ctx
            x_query_glob = x_query

        # 2. Support-set local statistics & correlation vector r_xy
        mu_ctx, sd_ctx, r_xy = self._compute_support_stats(
            x_ctx_glob.float(), y_ctx.float(), ctx_key_padding_mask
        )
        x_ctx_loc = ((x_ctx_glob.float() - mu_ctx.unsqueeze(1)) / (sd_ctx.unsqueeze(1) + 1e-3)).clamp(-10.0, 10.0)
        x_query_loc = ((x_query_glob.float() - mu_ctx.unsqueeze(1)) / (sd_ctx.unsqueeze(1) + 1e-3)).clamp(-10.0, 10.0)
        log_sigma_ctx = torch.log(sd_ctx + 1e-3)

        # 3. Analytic Leave-One-Out (LOO) Multi-Metric Basis Bank
        basis_out = self.basis_bank(
            x_ctx_loc=x_ctx_loc,
            y_ctx=y_ctx.float(),
            ctx_mask=ctx_key_padding_mask,
            x_query_loc=x_query_loc,
            r_xy=r_xy,
        )
        y_loo = basis_out["y_loo"].to(x_ctx.dtype)            # (B, N_ctx, M)
        y_q_basis = basis_out["y_q_basis"].to(x_ctx.dtype)    # (B, N_q, M)
        loo_mse = basis_out["loo_mse"].to(x_ctx.dtype)        # (B, M)
        loo_corr = basis_out["loo_corr"].to(x_ctx.dtype)      # (B, M)
        w_loo = basis_out["w_loo"].to(x_ctx.dtype)            # (B, M)
        sim_cc = basis_out["sim_cc"]                          # (B, 4, N_ctx, N_ctx)
        sim_qc = basis_out["sim_qc"]                          # (B, 4, N_q, N_ctx)
        eff_rank = basis_out["eff_rank_ratio"].to(x_ctx.dtype)

        # 4. Fixed-Coordinate Subspace Tokenization
        h_ctx_mol = self.coord_tokenizer(
            x_ctx_glob.to(x_ctx.dtype),
            x_ctx_loc.to(x_ctx.dtype),
            r_xy.to(x_ctx.dtype),
            log_sigma_ctx.to(x_ctx.dtype),
        )
        h_query_mol = self.coord_tokenizer(
            x_query_glob.to(x_ctx.dtype),
            x_query_loc.to(x_ctx.dtype),
            r_xy.to(x_ctx.dtype),
            log_sigma_ctx.to(x_ctx.dtype),
        )

        # 5. Enrich Context Tokens with Target + LOO Residual Features
        loo_resid = (y_ctx - y_loo).clamp(-5.0, 5.0)
        ctx_basis_feat = torch.cat([y_loo, loo_resid, y_ctx], dim=-1)
        h_ctx = h_ctx_mol + self.label_encoder(y_ctx) + self.ctx_basis_proj(ctx_basis_feat)

        # Enrich Query Tokens with Basis Predictions + Chemical Leverage Stats
        max_sim_cos = sim_qc[:, 0].max(dim=-1, keepdim=True).values
        max_sim_raw = sim_qc[:, 1].max(dim=-1, keepdim=True).values
        max_sim_white = sim_qc[:, 2].max(dim=-1, keepdim=True).values
        max_sim_corr = sim_qc[:, 3].max(dim=-1, keepdim=True).values
        basis_mean = (y_q_basis * w_loo.unsqueeze(1)).sum(dim=-1, keepdim=True)
        basis_std = torch.sqrt(
            ((y_q_basis - basis_mean) ** 2 * w_loo.unsqueeze(1)).sum(dim=-1, keepdim=True) + 1e-5
        )
        q_leverage = torch.cat(
            [y_q_basis, max_sim_cos, max_sim_raw, max_sim_white, max_sim_corr, basis_mean, basis_std],
            dim=-1,
        ).to(x_ctx.dtype)
        h_query = h_query_mol + self.query_basis_proj(q_leverage)

        # 6. Relational Delta-ICL Transformer Layers
        for layer in self.layers:
            h_ctx, h_query = layer(
                h_ctx=h_ctx,
                h_query=h_query,
                sim_cc=sim_cc,
                sim_qc=sim_qc,
                ctx_key_padding_mask=ctx_key_padding_mask,
            )

        # 7. Matched-Molecular-Pair (Delta-x) Local Cliff Correction over Top-k Neighbors
        k_mmp = min(16, N_ctx)
        # Rank neighbors by blended similarity
        blended_sim = (
            0.35 * sim_qc[:, 0] + 0.25 * sim_qc[:, 1] + 0.20 * sim_qc[:, 2] + 0.20 * sim_qc[:, 3]
        )
        if ctx_key_padding_mask is not None:
            blended_sim = blended_sim.masked_fill(ctx_key_padding_mask.unsqueeze(1), -1e4)
        top_sim, top_idx = torch.topk(blended_sim, k=k_mmp, dim=-1, largest=True)  # (B, N_q, k_mmp)

        # Gather neighbor molecular embeddings, labels, and LOO stack predictions
        idx_exp_d = top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        h_ctx_exp = h_ctx_mol.unsqueeze(1).expand(-1, N_q, -1, -1)
        h_nn = torch.gather(h_ctx_exp, 2, idx_exp_d)  # (B, N_q, k_mmp, d_model)
        delta_h = h_query_mol.unsqueeze(2) - h_nn      # (B, N_q, k_mmp, d_model)

        y_ctx_flat = y_ctx.squeeze(-1)  # (B, N_ctx)
        y_loo_stack = (y_loo * w_loo.unsqueeze(1)).sum(dim=-1)  # (B, N_ctx)
        y_nn = torch.gather(y_ctx_flat.unsqueeze(1).expand(-1, N_q, -1), 2, top_idx).unsqueeze(-1)
        yloo_nn = torch.gather(y_loo_stack.unsqueeze(1).expand(-1, N_q, -1), 2, top_idx).unsqueeze(-1)
        resid_nn = y_nn - yloo_nn  # Local cliff indicator on neighbor i

        mmp_in = torch.cat(
            [
                delta_h,
                h_query.unsqueeze(2).expand(-1, -1, k_mmp, -1),
                top_sim.unsqueeze(-1).to(x_ctx.dtype),
                y_nn.to(x_ctx.dtype),
                resid_nn.to(x_ctx.dtype),
                (basis_mean.unsqueeze(2) - y_nn.to(x_ctx.dtype)),
            ],
            dim=-1,
        )
        mmp_out = self.mmp_delta_net(mmp_in)  # (B, N_q, k_mmp, 2)
        mmp_shift = mmp_out[..., 0]
        mmp_w = F.softmax(mmp_out[..., 1] + top_sim.to(x_ctx.dtype) * 4.0, dim=-1)
        delta_mmp = (mmp_w * (resid_nn.squeeze(-1).to(x_ctx.dtype) + mmp_shift)).sum(dim=-1, keepdim=True)

        # 8. Query-Conditioned Expert Weighting + Residual Readout
        valid_counts = (~ctx_key_padding_mask).float().sum(dim=1, keepdim=True).clamp_min(1.0)
        log_n = torch.log2(valid_counts) / 12.0
        task_summary = torch.cat([loo_mse, loo_corr, log_n.to(x_ctx.dtype), eff_rank], dim=-1)  # (B, 2M+2)
        task_summary_q = task_summary.unsqueeze(1).expand(-1, N_q, -1)
        readout_in = torch.cat([h_query, task_summary_q], dim=-1)

        delta_gate = self.expert_gate_head(readout_in)  # (B, N_q, M)
        log_w_loo = torch.log(w_loo.clamp_min(1e-6)).unsqueeze(1)
        w_final = F.softmax(log_w_loo + self.gate_scale * delta_gate, dim=-1)

        y_basis_combined = (w_final * y_q_basis).sum(dim=-1, keepdim=True)  # (B, N_q, 1)
        res_out = self.residual_head(readout_in)  # (B, N_q, 2)
        res_mean = res_out[..., 0:1]
        log_std_offset = res_out[..., 1:2]

        y_point = y_basis_combined + self.res_scale * res_mean + self.mmp_scale * delta_mmp

        # Calibrated predictive uncertainty combining LOO MSE, ensemble spread, and neural offset
        expected_loo_var = (w_final * loo_mse.unsqueeze(1)).sum(dim=-1, keepdim=True).clamp_min(1e-3)
        ensemble_var = ((y_q_basis - y_basis_combined) ** 2 * w_final).sum(dim=-1, keepdim=True)
        base_std = torch.sqrt(expected_loo_var + ensemble_var + 1e-4)
        pred_std = (base_std * torch.exp(0.3 * torch.tanh(log_std_offset))).clamp(0.05, 2.5)

        # 9. Construct 64-bin Riemann Logits anchored at (y_point, pred_std) + fine neural shape logits
        if bin_centers is None:
            edges = torch.linspace(-4.0, 4.0, self.num_bins + 1, device=x_ctx.device, dtype=x_ctx.dtype)
            centers = 0.5 * (edges[:-1] + edges[1:])
        else:
            centers = bin_centers.to(device=x_ctx.device, dtype=x_ctx.dtype)

        gauss_logits = -0.5 * ((centers.view(1, 1, -1) - y_point) / pred_std) ** 2
        fine_logits = self.bin_residual_head(h_query)
        logits = gauss_logits + 0.2 * fine_logits

        return {
            "logits": logits,
            "y_point": y_point,
            "pred_std": pred_std,
            "y_basis": y_basis_combined,
            "w_final": w_final,
        }

    def forward(
        self,
        x_ctx: torch.Tensor,
        y_ctx: torch.Tensor,
        ctx_key_padding_mask: torch.Tensor,
        x_query: torch.Tensor,
        pool_mean: Optional[torch.Tensor] = None,
        pool_std: Optional[torch.Tensor] = None,
        bin_centers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.forward_detailed(
            x_ctx=x_ctx,
            y_ctx=y_ctx,
            ctx_key_padding_mask=ctx_key_padding_mask,
            x_query=x_query,
            pool_mean=pool_mean,
            pool_std=pool_std,
            bin_centers=bin_centers,
        )
        return out["logits"]


# ==============================================================================
# 5. Lightning Module with Composite Huber + Pearson + Riemann Loss
# ==============================================================================
class TabularCheMeleonLightningModule(pl.LightningModule):
    def __init__(
        self,
        embed_dim: int = 2048,
        d_model: int = 384,
        m_latents: int = 256,
        nhead: int = 8,
        num_latent_layers: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        learning_rate: float = 2.5e-4,
        weight_decay: float = 1e-4,
        num_bins: int = 64,
        support_min: float = -4.0,
        support_max: float = 4.0,
        pool_mean: Optional[torch.Tensor] = None,
        pool_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pool_mean", "pool_std"])
        self.num_bins = num_bins
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        bin_edges = torch.linspace(support_min, support_max, num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_centers", bin_centers)
        self.bin_width = (support_max - support_min) / num_bins

        # Persistent global manifold standardization buffers saved inside checkpoint
        if pool_mean is not None and pool_std is not None:
            self.register_buffer("pool_mean", pool_mean.float().view(embed_dim))
            self.register_buffer("pool_std", pool_std.float().view(embed_dim).clamp_min(1e-6))
        else:
            self.register_buffer("pool_mean", torch.zeros(embed_dim))
            self.register_buffer("pool_std", torch.ones(embed_dim))

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

    def set_pool_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.pool_mean.copy_(mean.float().view(-1))
        self.pool_std.copy_(std.float().view(-1).clamp_min(1e-6))

    def forward_detailed(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model.forward_detailed(
            x_ctx=batch["x_ctx"],
            y_ctx=batch["y_ctx"],
            ctx_key_padding_mask=batch["ctx_key_padding_mask"],
            x_query=batch["x_query"],
            pool_mean=self.pool_mean,
            pool_std=self.pool_std,
            bin_centers=self.bin_centers,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_detailed(batch)["logits"]

    def _distribution_to_moments(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits.float(), dim=-1)
        centers = self.bin_centers.float()
        mean = torch.sum(probs * centers, dim=-1, keepdim=True)
        var = torch.sum(probs * ((centers - mean) ** 2), dim=-1, keepdim=True)
        return mean, var

    def _two_hot_encoding(self, targets: torch.Tensor) -> torch.Tensor:
        c0 = self.bin_centers[0]
        c_last = self.bin_centers[-1]
        clamped = torch.clamp(targets.float(), min=c0.item(), max=c_last.item())
        unit_pos = (clamped - c0) / self.bin_width
        idx_lower = torch.floor(unit_pos).long().clamp(0, self.num_bins - 1)
        idx_upper = torch.clamp(idx_lower + 1, max=self.num_bins - 1)
        w_upper = unit_pos - idx_lower.float()
        w_lower = 1.0 - w_upper

        B, Q, _ = targets.shape
        soft_targets = torch.zeros(B, Q, self.num_bins, device=targets.device, dtype=torch.float32)
        soft_targets.scatter_add_(-1, idx_lower, w_lower)
        soft_targets.scatter_add_(-1, idx_upper, w_upper)
        return soft_targets

    def _compute_loss_and_metrics(self, batch: dict[str, torch.Tensor]):
        out = self.forward_detailed(batch)
        logits = out["logits"].float()
        y_point = out["y_point"].float()
        pred_std = out["pred_std"].float()
        targets = batch["y_query"].float()

        # 1. Two-Hot Riemann Cross-Entropy Loss
        soft_targets = self._two_hot_encoding(targets)
        loss_riemann = F.cross_entropy(
            logits.view(-1, self.num_bins), soft_targets.view(-1, self.num_bins)
        )

        # 2. Direct Point + Distribution Mean Huber Loss (optimizes MSE/MAE directly)
        dist_mean, _ = self._distribution_to_moments(logits)
        pred_blend = 0.5 * y_point + 0.5 * dist_mean
        loss_huber = F.smooth_l1_loss(pred_blend, targets, beta=0.25)

        # 3. Cosine / Pearson Correlation Loss per episode (optimizes r / spearman directly)
        p_c = pred_blend - pred_blend.mean(dim=1, keepdim=True)
        t_c = targets - targets.mean(dim=1, keepdim=True)
        corr = (p_c * t_c).sum(dim=1) / (
            torch.sqrt((p_c ** 2).sum(dim=1) * (t_c ** 2).sum(dim=1) + 1e-6)
        )
        loss_corr = (1.0 - corr).mean()

        # 4. Calibrated Gaussian NLL Loss
        loss_nll = 0.5 * (
            ((y_point.detach() - targets) / pred_std) ** 2 + 2.0 * torch.log(pred_std)
        ).mean()

        total_loss = loss_huber + 0.35 * loss_corr + 0.25 * loss_riemann + 0.05 * loss_nll
        mae = F.l1_loss(pred_blend.detach(), targets)
        return total_loss, mae, corr.mean().detach(), pred_blend.detach()

    def training_step(self, batch, batch_idx):
        loss, mae, corr, _ = self._compute_loss_and_metrics(batch)
        bs = batch["x_ctx"].size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("train/mae", mae, on_step=True, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        self.log("train/pearson", corr, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae, corr, pred_mean = self._compute_loss_and_metrics(batch)
        targets = batch["y_query"].float()
        bs = batch["x_ctx"].size(0)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("val/mae_overall", mae, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log("val/pearson", corr, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

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
        total_epochs = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs),
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}


# ==============================================================================
# 6. Memory-Efficient DataModule & Production Training Entrypoint
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
        min_context: int = TRAIN_MIN_CONTEXT,
        max_context: int = TRAIN_MAX_CONTEXT,
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
        self.pool_mean: Optional[torch.Tensor] = None
        self.pool_std: Optional[torch.Tensor] = None
        self.train_ds: Optional[ChemicalEpisodicDataset] = None
        self.val_ds: Optional[ChemicalEpisodicDataset] = None

    @staticmethod
    def _streaming_mean_std(t: torch.Tensor, chunk_size: int = 65_536) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes exact mean and std in chunks without allocating a full float32 copy in RAM."""
        n, d = t.shape
        sum_x = torch.zeros(d, dtype=torch.float64)
        sum_x2 = torch.zeros(d, dtype=torch.float64)
        for i in range(0, n, chunk_size):
            chunk = t[i : i + chunk_size].double()
            sum_x += chunk.sum(dim=0)
            sum_x2 += (chunk ** 2).sum(dim=0)
        mean = sum_x / n
        var = (sum_x2 / n - mean ** 2).clamp_min(1e-12)
        return mean.float(), torch.sqrt(var).float()

    def setup(self, stage: Optional[str] = None):
        # Prevent double initialization when called manually before trainer.fit()
        if self.train_ds is not None and self.val_ds is not None:
            return

        if self.embeddings_path.is_file():
            raw = torch.load(self.embeddings_path, map_location="cpu", weights_only=True)
            self.embed_dim = raw.shape[1]
            self.pool_mean, self.pool_std = self._streaming_mean_std(raw)
            embeddings = raw.half()
            del raw
        else:
            files = sorted(self.embeddings_path.glob("*.pt"))
            if not files:
                raise FileNotFoundError(f"No .pt files found in {self.embeddings_path}")
            # Pre-scan shapes so we allocate ONE float16 buffer instead of a list + concat copy
            shapes = []
            for f in files:
                t = torch.load(f, map_location="cpu", weights_only=True)
                shapes.append(t.shape)
                del t
            n_total = sum(s[0] for s in shapes)
            self.embed_dim = shapes[0][1]
            embeddings = torch.empty((n_total, self.embed_dim), dtype=torch.float16)
            offset = 0
            for f, s in zip(files, shapes):
                t = torch.load(f, map_location="cpu", weights_only=True)
                embeddings[offset : offset + s[0]] = t.half()
                offset += s[0]
                del t
            self.pool_mean, self.pool_std = self._streaming_mean_std(embeddings)

        # Put the single float16 tensor in shared OS memory so workers never copy it
        embeddings.share_memory_()

        n_total = len(embeddings)
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
        train_size = int((1.0 - self.val_split) * n_total)
        train_indices = indices[:train_size].share_memory_()
        val_indices = indices[train_size:].share_memory_()

        self.train_ds = ChemicalEpisodicDataset(
            embeddings=embeddings,
            valid_indices=train_indices,
            min_context=self.min_context,
            max_context=self.max_context,
            num_queries=self.num_queries,
            num_episodes=self.episodes_per_epoch,
        )
        self.val_ds = ChemicalEpisodicDataset(
            embeddings=embeddings,
            valid_indices=val_indices,
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
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pad_icl_perceiver_collate,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )


if __name__ == "__main__":
    import argparse
    import shutil

    # Cap CPU OpenMP threads per worker so 4 workers don't spawn 64 BLAS/SVD buffers
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", type=str, default="data/pool_full.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--episodes-per-epoch", type=int, default=16_000)
    parser.add_argument("--val-episodes", type=int, default=1_280)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--m-latents", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-latent-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1536)
    parser.add_argument("--num-bins", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-context", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=1024)
    parser.add_argument("--num-queries", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--log-dir", type=str, default="tabular_chemeleon_logs")
    args = parser.parse_args()

    datamodule = TabularCheMeleonDataModule(
        embeddings_path=args.embeddings_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
        pool_mean=datamodule.pool_mean,
        pool_std=datamodule.pool_std,
    )

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="chemeleon_pfn",
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
    if checkpoint_callback.best_model_path:
        shutil.copyfile(checkpoint_callback.best_model_path, model_path)
    else:
        trainer.save_checkpoint(model_path, weights_only=True)
    print(f"CheMeleon-PFN model successfully trained and saved to: {model_path}")
