from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.base import BaseEstimator, RegressorMixin

from tabularchemeleon import TabularCheMeleonLightningModule
from config import MIN_CONTEXT, MAX_CONTEXT, NUM_QUERIES


class TabularCheMeleonRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible In-Context Bioactivity Regressor.
    
    Features:
    - Chemically-sorted query batching to prevent cross-scaffold context dilution.
    - Angular Cosine kNN + Diversity Context Bagging aligned with empirical SAR priors.
    - True Monte Carlo context subsampling across both few-shot and rich-context regimes.
    - Vectorized multi-bag parallel forward passes on GPU (single batch execution).
    - Z-score normalization preventing target clamping on skewed screening assays.
    - Supports continuous bioactivity regression and thresholded classification.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        max_context_size: int = MAX_CONTEXT,
        chunk_size: int = NUM_QUERIES,
        num_bags: int = 16,
        bag_fraction: float = 0.85,
        bag_strategy: Literal["hybrid", "knn", "random"] = "hybrid",
        knn_ratio: float = 0.65,
        clip_target_norm: float = 3.8,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_context_size = max_context_size
        self.chunk_size = chunk_size
        self.num_bags = num_bags
        self.bag_fraction = bag_fraction
        self.bag_strategy = bag_strategy
        self.knn_ratio = knn_ratio
        self.clip_target_norm = clip_target_norm

        self.model = TabularCheMeleonLightningModule.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        self.bin_centers = self.model.bin_centers.to(self.device)
        self.embed_dim = self.model.model.embed_dim

        self.is_fitted_ = False
        self.X_context_: Optional[torch.Tensor] = None
        self.X_context_norm_: Optional[torch.Tensor] = None
        self.y_context_norm_: Optional[torch.Tensor] = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        """
        Stores and indexes the assay conditioning context.
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X.clone().detach().float()

        if isinstance(y, np.ndarray):
            y_tensor = torch.from_numpy(y).float()
        else:
            y_tensor = y.clone().detach().float()

        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        if X_tensor.shape[1] != self.embed_dim:
            raise ValueError(
                f"Feature dimension mismatch: model expects {self.embed_dim}, got {X_tensor.shape[1]}"
            )

        # Standard Z-score normalization matching the synthetic training prior
        self.y_mean_ = torch.mean(y_tensor).item()
        self.y_scale_ = torch.std(y_tensor).item()
        if self.y_scale_ < 1e-6:
            self.y_scale_ = 1.0  # Guard against flat target arrays

        # Normalize and clamp within Riemann support grid [-4.0, 4.0]
        y_norm = (y_tensor - self.y_mean_) / self.y_scale_
        self.y_context_norm_ = torch.clamp(
            y_norm, -self.clip_target_norm, self.clip_target_norm
        ).to(self.device)

        self.X_context_ = X_tensor.to(self.device)
        # Precompute L2 normalized vectors for fast cosine similarity retrieval
        self.X_context_norm_ = F.normalize(self.X_context_, p=2, dim=-1)
        self.is_fitted_ = True
        return self

    def _sample_context_bag(
        self,
        x_q_chunk: torch.Tensor,
        bag_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_total = self.X_context_.size(0)

        # Uniform random sampling
        if self.bag_strategy == "random" or n_total <= bag_size:
            indices = torch.randperm(n_total, device=self.device)[:bag_size]
            return self.X_context_[indices], self.y_context_norm_[indices]

        # Angular Cosine Similarity Retrieval
        x_q_norm = F.normalize(x_q_chunk, p=2, dim=-1)
        # Similarity matrix: (current_q_len, n_total)
        cosine_sims = x_q_norm @ self.X_context_norm_.T
        # Maximum similarity to ANY candidate molecule in this query chunk
        max_sims = cosine_sims.max(dim=0).values  # (n_total,)

        if self.bag_strategy == "knn":
            topk_indices = torch.topk(max_sims, k=bag_size, largest=True).indices
            perm = torch.randperm(bag_size, device=self.device)
            return self.X_context_[topk_indices[perm]], self.y_context_norm_[topk_indices[perm]]

        # Hybrid: nearest chemical analogs + diverse background anchors
        k_near = int(bag_size * self.knn_ratio)
        k_rand = bag_size - k_near

        nearest_indices = torch.topk(max_sims, k=k_near, largest=True).indices

        mask = torch.ones(n_total, dtype=torch.bool, device=self.device)
        mask[nearest_indices] = False
        remaining_indices = torch.where(mask)[0]

        rand_picks = remaining_indices[
            torch.randperm(len(remaining_indices), device=self.device)[:k_rand]
        ]
        combined_indices = torch.cat([nearest_indices, rand_picks])

        perm = torch.randperm(bag_size, device=self.device)
        final_indices = combined_indices[perm]
        return self.X_context_[final_indices], self.y_context_norm_[final_indices]

    @torch.inference_mode()
    def predict_distribution(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluates bagged predictive distributions over the discretized Riemann bins.
        
        Returns:
            probs: (N_query, num_bins) tensor of probability distributions on CPU.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit(X, y) before predict().")

        if isinstance(X, np.ndarray):
            X_query = torch.from_numpy(X).float()
        else:
            X_query = X.clone().detach().float()

        num_queries = X_query.size(0)
        n_total_ctx = self.X_context_.size(0)

        # ----------------------------------------------------------------------
        # Chemically sort queries to prevent context dilution across scaffolds
        # ----------------------------------------------------------------------
        if num_queries > self.chunk_size:
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

        # Determine effective context bag size with proper subsampling
        if n_total_ctx > self.max_context_size:
            bag_size = self.max_context_size
            effective_num_bags = self.num_bags
        elif n_total_ctx > MIN_CONTEXT:
            bag_size = max(MIN_CONTEXT, int(n_total_ctx * self.bag_fraction))
            effective_num_bags = self.num_bags
        else:
            bag_size = n_total_ctx
            effective_num_bags = 1

        all_chunk_probs = []

        # Stream candidate queries in chunks
        for q_idx in range(0, num_queries, self.chunk_size):
            x_q_chunk = X_query_sorted[q_idx : q_idx + self.chunk_size].to(self.device)
            current_q_len = x_q_chunk.size(0)

            ctx_x_list = []
            ctx_y_list = []
            for _ in range(effective_num_bags):
                x_c, y_c = self._sample_context_bag(x_q_chunk, bag_size)
                ctx_x_list.append(x_c)
                ctx_y_list.append(y_c)

            # Vectorized multi-bag batch construction: (effective_num_bags, bag_size + q_len, D)
            x_ctx_batch = torch.stack(ctx_x_list, dim=0)
            y_ctx_batch = torch.stack(ctx_y_list, dim=0)

            x_q_batch = x_q_chunk.unsqueeze(0).expand(effective_num_bags, -1, -1)
            dummy_y_batch = torch.zeros(effective_num_bags, current_q_len, 1, device=self.device)

            x_combined = torch.cat([x_ctx_batch, x_q_batch], dim=1)
            y_combined = torch.cat([y_ctx_batch, dummy_y_batch], dim=1)

            batch_dict = {
                "x": x_combined,
                "y": y_combined,
                "context_lens": [bag_size] * effective_num_bags,
                "query_len": current_q_len,
                "sample_mask": None,
            }

            # Single parallel forward pass for all bags
            logits = self.model(batch_dict)  # (effective_num_bags, current_q_len, num_bins)
            probs = F.softmax(logits, dim=-1)

            # Average probability distributions across context bags
            avg_probs = probs.mean(dim=0)  # (current_q_len, num_bins)
            all_chunk_probs.append(avg_probs.cpu())

        total_probs = torch.cat(all_chunk_probs, dim=0)  # (num_queries, num_bins)

        # Restore original query indexing order if sorted
        if sort_order is not None:
            total_probs = total_probs[unsort_order.cpu()]

        return total_probs

    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts continuous values along with predictive uncertainty standard deviations.
        
        Returns:
            y_preds: (N,) Expected value in original physical units (e.g. pIC50, LogS).
            y_stds:  (N,) Combined predictive uncertainty standard deviation in physical units.
        """
        probs = self.predict_distribution(X)  # (N, num_bins)
        bin_centers = self.bin_centers.cpu()  # (num_bins,)

        # Compute moments in normalized space
        expected_y_norm = torch.sum(probs * bin_centers, dim=-1)  # (N,)
        var_norm = torch.sum(probs * ((bin_centers - expected_y_norm.unsqueeze(-1)) ** 2), dim=-1)  # (N,)
        std_norm = torch.sqrt(torch.clamp(var_norm, min=0.0))  # (N,)

        # De-standardize back to original physical assay units
        y_pred = (expected_y_norm * self.y_scale_) + self.y_mean_
        std_pred = std_norm * self.y_scale_

        return y_pred.numpy(), std_pred.numpy()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Standard scikit-learn continuous regression interface."""
        preds, _ = self.predict_with_uncertainty(X)
        return preds

    def predict_classification(
        self,
        X: Union[np.ndarray, torch.Tensor],
        active_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs binary active/inactive classification via Riemann posterior tail integration.
        
        Args:
            X: Candidate molecule embeddings
            active_threshold: Cutoff in raw assay units (e.g. 6.0 for pIC50 >= 1 uM)
            
        Returns:
            prob_active: (N,) Posterior probability P(Y >= threshold) in [0.0, 1.0]
            pred_class: (N,) Binary classification call (1 for Active, 0 for Inactive)
        """
        probs = self.predict_distribution(X)  # (N, num_bins)
        threshold_norm = (active_threshold - self.y_mean_) / self.y_scale_
        active_mask = (self.bin_centers.cpu() >= threshold_norm)

        prob_active = probs[:, active_mask].sum(dim=-1).numpy()
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
        """Computes the R^2 coefficient of determination."""
        y_pred = self.predict(X)
        if isinstance(y, torch.Tensor):
            y_true = y.detach().cpu().numpy().squeeze()
        else:
            y_true = np.asarray(y).squeeze()

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-8:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))
