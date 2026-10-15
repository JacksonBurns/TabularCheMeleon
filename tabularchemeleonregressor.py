from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin

from tabularchemeleon import TabularCheMeleonLightningModule

try:
    from config import MIN_CONTEXT, MAX_CONTEXT, NUM_QUERIES
except ImportError:
    MIN_CONTEXT = 16
    MAX_CONTEXT = 5000
    NUM_QUERIES = 128


class TabularCheMeleonRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn & eval.py compatible In-Context Bioactivity Regressor for CheMeleon-PFN.

    Key Capabilities:
      - 100% Full-Context Conditioning up to max_context_size (5000+ molecules) in a single pass.
      - Automatic Skewness-Aware Target Normalization (handles unlogged right-skewed clearance/half-life/VDss
        and bounded left-skewed PPBR assays automatically from support-set y statistics).
      - Hybrid kNN + Diversity Context Retrieval for ultra-large screening decks (N > max_context_size).
      - Calibrated predictive uncertainty and threshold-based classification via the 64-bin Riemann head.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        max_context_size: int = MAX_CONTEXT,
        chunk_size: int = 256,
        num_bags: int = 8,
        bag_fraction: float = 1.0,
        bag_strategy: Literal["hybrid", "knn", "random"] = "hybrid",
        knn_ratio: float = 0.70,
        clip_target_norm: float = 4.2,
        skew_threshold: float = 1.6,
        autocast: bool = True,
        pool_stats_path: str = "data/pool_full.pt",
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_context_size = max_context_size
        self.chunk_size = chunk_size
        self.num_bags = num_bags
        self.bag_fraction = bag_fraction
        self.bag_strategy = bag_strategy
        self.knn_ratio = knn_ratio
        self.clip_target_norm = clip_target_norm
        self.skew_threshold = skew_threshold
        self.autocast = autocast
        self.pool_stats_path = pool_stats_path

        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self.model = TabularCheMeleonLightningModule.load_from_checkpoint(
                checkpoint_path, map_location=self.device
            )
        else:
            # Zero-shot fallback (uses the analytic LOO basis bank + zero-initialized residual gates)
            self.model = TabularCheMeleonLightningModule()

        # Ensure pool_mean and pool_std are populated if checkpoint had default zeros/ones
        if (
            torch.allclose(self.model.pool_mean.cpu(), torch.zeros_like(self.model.pool_mean.cpu()))
            and Path(self.pool_stats_path).exists()
        ):
            pool = torch.load(self.pool_stats_path, map_location="cpu", weights_only=True).float()
            self.model.set_pool_stats(pool.mean(dim=0), pool.std(dim=0))

        self.model.to(self.device)
        self.model.eval()

        self.bin_centers = self.model.bin_centers.to(self.device)
        self.embed_dim = getattr(self.model.model, "embed_dim", 2048)

        self.is_fitted_ = False
        self.X_context_: Optional[torch.Tensor] = None
        self.X_context_norm_: Optional[torch.Tensor] = None
        self.y_context_norm_: Optional[torch.Tensor] = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0
        self.y_transform_mode_: str = "identity"
        self.y_shift_: float = 0.0
        self.y_min_: float = 0.0
        self.y_max_: float = 0.0

    def _fit_target_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        Detects severe target skewness strictly from y_ctx and applies an invertible transform
        before standardization so extreme outliers do not distort L2/kernel/attention weights.
        """
        y_flat = y.view(-1).double()
        n = len(y_flat)
        self.y_min_ = float(y_flat.min().item())
        self.y_max_ = float(y_flat.max().item())

        if n < 16:
            self.y_transform_mode_ = "identity"
            y_trans = y_flat
        else:
            mu = y_flat.mean()
            sd = y_flat.std().clamp_min(1e-8)
            skew = float((((y_flat - mu) / sd) ** 3).mean().item())
            q50 = float(torch.quantile(y_flat, 0.50).item())
            q95 = float(torch.quantile(y_flat, 0.95).item())
            q05 = float(torch.quantile(y_flat, 0.05).item())
            upper_lower_ratio = (q95 - q50) / max(q50 - q05, 1e-6)

            if self.y_min_ >= 0.0 and skew > self.skew_threshold and upper_lower_ratio > 2.2:
                # Strictly positive right-skewed assay (e.g., Clearance, Half-life, VDss)
                self.y_transform_mode_ = "pos_log"
                self.y_shift_ = max(abs(q50) * 0.05, 1e-3)
                y_trans = torch.log(y_flat.clamp_min(0.0) + self.y_shift_)
            elif (
                self.y_min_ >= 0.0
                and self.y_max_ <= 100.5
                and skew < -self.skew_threshold
            ):
                # Left-skewed percentage assay piled near 100% (e.g., PPBR)
                self.y_transform_mode_ = "neg_log_100"
                self.y_shift_ = max((100.0 - q50) * 0.10, 0.25)
                y_trans = -torch.log((100.0 - y_flat).clamp_min(0.0) + self.y_shift_)
            else:
                self.y_transform_mode_ = "identity"
                self.y_shift_ = 0.0
                y_trans = y_flat

        if n <= 1:
            self.y_mean_ = float(y_trans.mean().item())
            self.y_scale_ = 1.0
        else:
            self.y_mean_ = float(y_trans.mean().item())
            s_val = float(y_trans.std().item())
            self.y_scale_ = s_val if (not np.isnan(s_val) and s_val > 1e-6) else 1.0

        y_norm = ((y_trans - self.y_mean_) / self.y_scale_).float()
        return y_norm.unsqueeze(-1)

    def _invert_target_transform(
        self, y_norm_mean: torch.Tensor, y_norm_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverts standardization and any monotonic skew transform back to raw target units."""
        y_t = y_norm_mean * self.y_scale_ + self.y_mean_
        s_t = (y_norm_std * self.y_scale_).clamp_min(1e-6)

        if self.y_transform_mode_ == "pos_log":
            # Median/damped log-normal inversion avoids exponential blow-up on outlier queries
            y_raw = torch.exp(y_t.clamp(max=self.y_mean_ + 3.5 * self.y_scale_)) - self.y_shift_
            y_raw = y_raw.clamp_min(max(0.0, self.y_min_ * 0.5))
            std_raw = y_raw.abs() * s_t
            return y_raw, std_raw
        elif self.y_transform_mode_ == "neg_log_100":
            # Invert y_t = -log(100 - y + shift) -> y = 100 + shift - exp(-y_t + 0.25 * s_t^2)
            unlogged = torch.exp((-y_t + 0.20 * (s_t ** 2)).clamp(max=6.0))
            y_raw = (100.0 + self.y_shift_ - unlogged).clamp(self.y_min_, 100.0)
            std_raw = unlogged * s_t
            return y_raw, std_raw
        else:
            return y_t, s_t

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        """Stores and indexes the assay conditioning context."""
        X_tensor = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.clone().detach().float()
        y_tensor = torch.from_numpy(y).float() if isinstance(y, np.ndarray) else y.clone().detach().float()

        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        if X_tensor.shape[1] != self.embed_dim:
            raise ValueError(
                f"Feature dimension mismatch: model expects {self.embed_dim}, got {X_tensor.shape[1]}"
            )

        y_norm = self._fit_target_transform(y_tensor)
        self.y_context_norm_ = torch.clamp(
            y_norm, -self.clip_target_norm, self.clip_target_norm
        ).to(self.device)

        self.X_context_ = X_tensor.to(self.device)
        p_mu = self.model.pool_mean.unsqueeze(0)
        p_sd = self.model.pool_std.unsqueeze(0)
        self.X_context_norm_ = F.normalize((self.X_context_ - p_mu) / p_sd, p=2, dim=-1)
        self.is_fitted_ = True
        return self

    def _sample_context_bag(
        self,
        x_q_chunk: torch.Tensor,
        bag_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_total = self.X_context_.size(0)
        if n_total <= bag_size:
            return self.X_context_, self.y_context_norm_

        if self.bag_strategy == "random":
            indices = torch.randperm(n_total, device=self.device)[:bag_size]
            return self.X_context_[indices], self.y_context_norm_[indices]

        p_mu = self.model.pool_mean.unsqueeze(0)
        p_sd = self.model.pool_std.unsqueeze(0)
        x_q_norm = F.normalize((x_q_chunk - p_mu) / p_sd, p=2, dim=-1)
        cosine_sims = x_q_norm @ self.X_context_norm_.T
        max_sims = cosine_sims.max(dim=0).values

        if self.bag_strategy == "knn":
            topk_indices = torch.topk(max_sims, k=bag_size, largest=True).indices
            return self.X_context_[topk_indices], self.y_context_norm_[topk_indices]

        k_near = int(bag_size * self.knn_ratio)
        k_rand = bag_size - k_near
        nearest_indices = torch.topk(max_sims, k=k_near, largest=True).indices
        mask = torch.ones(n_total, dtype=torch.bool, device=self.device)
        mask[nearest_indices] = False
        remaining = torch.where(mask)[0]
        rand_picks = remaining[torch.randperm(len(remaining), device=self.device)[:k_rand]]
        combined = torch.cat([nearest_indices, rand_picks])
        return self.X_context_[combined], self.y_context_norm_[combined]

    @torch.inference_mode()
    def _predict_internal(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total_probs, y_norm_pred, y_norm_std) on CPU."""
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit(X, y) before predict().")

        X_query = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.clone().detach().float()
        if X_query.dim() == 1:
            X_query = X_query.unsqueeze(0)

        num_queries = X_query.size(0)
        n_total_ctx = self.X_context_.size(0)

        # Chemically sort queries only when subsampling a massive context (> max_context_size)
        if n_total_ctx > self.max_context_size and num_queries > self.chunk_size:
            try:
                q_centered = X_query - X_query.mean(dim=0, keepdim=True)
                _, _, V = torch.pca_lowrank(q_centered, q=1)
                sort_order = torch.argsort((q_centered @ V[:, :1]).squeeze(-1))
                unsort_order = torch.argsort(sort_order)
                X_query_sorted = X_query[sort_order]
            except Exception:
                sort_order = None
                X_query_sorted = X_query
        else:
            sort_order = None
            X_query_sorted = X_query

        if n_total_ctx > self.max_context_size:
            bag_size = self.max_context_size
            effective_num_bags = self.num_bags
        elif self.num_bags > 1 and self.bag_fraction < 1.0 and n_total_ctx > MIN_CONTEXT:
            bag_size = max(MIN_CONTEXT, int(n_total_ctx * self.bag_fraction))
            effective_num_bags = self.num_bags
        else:
            bag_size = n_total_ctx
            effective_num_bags = 1

        all_probs, all_means, all_stds = [], [], []
        use_autocast = self.autocast and self.device.type == "cuda"
        centers_cpu = self.bin_centers.cpu().float()

        for q_idx in range(0, num_queries, self.chunk_size):
            x_q_chunk = X_query_sorted[q_idx : q_idx + self.chunk_size].to(self.device)
            ctx_x_list, ctx_y_list = [], []
            for _ in range(effective_num_bags):
                x_c, y_c = self._sample_context_bag(x_q_chunk, bag_size)
                ctx_x_list.append(x_c)
                ctx_y_list.append(y_c)

            x_ctx_batch = torch.stack(ctx_x_list, dim=0)
            y_ctx_batch = torch.stack(ctx_y_list, dim=0)
            x_q_batch = x_q_chunk.unsqueeze(0).expand(effective_num_bags, -1, -1)
            ctx_key_padding_mask = torch.zeros(
                effective_num_bags, bag_size, dtype=torch.bool, device=self.device
            )

            batch_dict = {
                "x_ctx": x_ctx_batch,
                "y_ctx": y_ctx_batch,
                "ctx_key_padding_mask": ctx_key_padding_mask,
                "x_query": x_q_batch,
            }

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                out = self.model.forward_detailed(batch_dict)

            logits = out["logits"].float()
            y_point = out["y_point"].float().squeeze(-1)
            pred_std = out["pred_std"].float().squeeze(-1)

            probs = F.softmax(logits, dim=-1).mean(dim=0).cpu()  # (q_len, num_bins)
            dist_mean = torch.sum(probs * centers_cpu, dim=-1)
            dist_var = torch.sum(probs * ((centers_cpu - dist_mean.unsqueeze(-1)) ** 2), dim=-1)

            # Blend exact continuous residual head with Riemann distribution expectation
            point_mean = 0.65 * y_point.mean(dim=0).cpu() + 0.35 * dist_mean
            point_std = 0.50 * pred_std.mean(dim=0).cpu() + 0.50 * torch.sqrt(dist_var.clamp_min(1e-6))

            all_probs.append(probs)
            all_means.append(point_mean)
            all_stds.append(point_std)

        total_probs = torch.cat(all_probs, dim=0)
        total_means = torch.cat(all_means, dim=0)
        total_stds = torch.cat(all_stds, dim=0)

        if sort_order is not None:
            inv = unsort_order.cpu()
            total_probs = total_probs[inv]
            total_means = total_means[inv]
            total_stds = total_stds[inv]

        return total_probs, total_means, total_stds

    @torch.inference_mode()
    def predict_distribution(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        probs, _, _ = self._predict_internal(X)
        return probs

    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, y_norm_mean, y_norm_std = self._predict_internal(X)
        y_pred, std_pred = self._invert_target_transform(y_norm_mean, y_norm_std)
        return y_pred.numpy(), std_pred.numpy()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        preds, _ = self.predict_with_uncertainty(X)
        return preds

    def predict_classification(
        self,
        X: Union[np.ndarray, torch.Tensor],
        active_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probs, y_norm_mean, y_norm_std = self._predict_internal(X)
        if self.y_transform_mode_ == "pos_log":
            t_trans = math.log(max(active_threshold, 0.0) + self.y_shift_)
        elif self.y_transform_mode_ == "neg_log_100":
            t_trans = -math.log(max(100.0 - active_threshold, 0.0) + self.y_shift_)
        else:
            t_trans = float(active_threshold)

        threshold_norm = (t_trans - self.y_mean_) / self.y_scale_
        active_mask = self.bin_centers.cpu() >= threshold_norm
        prob_riemann = probs[:, active_mask].sum(dim=-1)
        prob_gauss = 1.0 - 0.5 * (
            1.0 + torch.erf((threshold_norm - y_norm_mean) / (y_norm_std * math.sqrt(2.0) + 1e-6))
        )
        prob_active = (0.5 * prob_riemann + 0.5 * prob_gauss).clamp(0.0, 1.0).numpy()
        pred_class = (prob_active >= 0.5).astype(int)
        return prob_active, pred_class

    def fit_predict(
        self,
        X_context: Union[np.ndarray, torch.Tensor],
        y_context: Union[np.ndarray, torch.Tensor],
        X_query: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        self.fit(X_context, y_context)
        return self.predict(X_query)

    def score(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]) -> float:
        y_pred = self.predict(X)
        y_true = y.detach().cpu().numpy().squeeze() if isinstance(y, torch.Tensor) else np.asarray(y).squeeze()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-8:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))
