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
    MIN_CONTEXT = 32
    MAX_CONTEXT = 2048
    NUM_QUERIES = 64


class TabularCheMeleonRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible In-Context Bioactivity Regressor for Perceiver Architectures.
    
    Features:
    - Native support for LatentContextPerceiverTransformer (separate x_ctx, y_ctx, x_query).
    - 100% Context Conditioning for N <= max_context_size in a single forward pass.
    - Adaptive Hybrid kNN + Diversity Bagging for massive screening decks (N > 2048).
    - Chemically-sorted query batching to eliminate cross-scaffold context dilution.
    - Autocast FP16 inference matching 16-mixed pre-training.
    - Continuous potency regression and threshold-based binary classification.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        max_context_size: int = MAX_CONTEXT,
        chunk_size: int = NUM_QUERIES,
        num_bags: int = 16,
        bag_fraction: float = 1.0,  # 1.0 uses all context compounds when N <= max_context_size
        bag_strategy: Literal["hybrid", "knn", "random"] = "hybrid",
        knn_ratio: float = 0.65,
        clip_target_norm: float = 3.8,
        autocast: bool = True,
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
        self.autocast = autocast

        # Load checkpoint and configure evaluation state
        self.model = TabularCheMeleonLightningModule.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        self.bin_centers = self.model.bin_centers.to(self.device)
        if hasattr(self.model.model, "embed_dim"):
            self.embed_dim = self.model.model.embed_dim
        elif hasattr(self.model, "hparams") and "embed_dim" in self.model.hparams:
            self.embed_dim = self.model.hparams.embed_dim
        else:
            self.embed_dim = 2048

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

        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        if X_tensor.shape[1] != self.embed_dim:
            raise ValueError(
                f"Feature dimension mismatch: model expects {self.embed_dim}, got {X_tensor.shape[1]}"
            )

        # Standard Z-score normalization matching the synthetic training prior
        if len(y_tensor) <= 1:
            self.y_mean_ = torch.mean(y_tensor).item()
            self.y_scale_ = 1.0
        else:
            self.y_mean_ = torch.mean(y_tensor).item()
            std_val = torch.std(y_tensor).item()
            self.y_scale_ = std_val if (not np.isnan(std_val) and std_val > 1e-6) else 1.0

        # Normalize and clamp within Riemann support grid [-4.0, 4.0]
        y_norm = (y_tensor - self.y_mean_) / self.y_scale_
        self.y_context_norm_ = torch.clamp(
            y_norm, -self.clip_target_norm, self.clip_target_norm
        ).to(self.device)

        self.X_context_ = X_tensor.to(self.device)
        self.X_context_norm_ = F.normalize(self.X_context_, p=2, dim=-1)
        self.is_fitted_ = True
        return self

    def _sample_context_bag(
        self,
        x_q_chunk: torch.Tensor,
        bag_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_total = self.X_context_.size(0)

        # If assay fits in bag budget, return all context compounds
        if n_total <= bag_size or self.bag_strategy == "random":
            indices = torch.randperm(n_total, device=self.device)[:bag_size]
            return self.X_context_[indices], self.y_context_norm_[indices]

        # Angular Cosine Similarity Retrieval
        x_q_norm = F.normalize(x_q_chunk, p=2, dim=-1)
        cosine_sims = x_q_norm @ self.X_context_norm_.T
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
        Evaluates predictive distributions over the discretized Riemann bins.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit(X, y) before predict().")

        if isinstance(X, np.ndarray):
            X_query = torch.from_numpy(X).float()
        else:
            X_query = X.clone().detach().float()

        if X_query.dim() == 1:
            X_query = X_query.unsqueeze(0)

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

        # Determine context sizing and parallel execution plan
        if n_total_ctx > self.max_context_size:
            bag_size = self.max_context_size
            effective_num_bags = self.num_bags
        elif self.num_bags > 1 and self.bag_fraction < 1.0 and n_total_ctx > MIN_CONTEXT:
            bag_size = max(MIN_CONTEXT, int(n_total_ctx * self.bag_fraction))
            effective_num_bags = self.num_bags
        else:
            # Single forward pass conditioning on 100% of context data
            bag_size = n_total_ctx
            effective_num_bags = 1

        all_chunk_probs = []
        use_autocast = self.autocast and self.device.type == "cuda"

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

            # Vectorized multi-bag batch construction: (B_bags, bag_size, D)
            x_ctx_batch = torch.stack(ctx_x_list, dim=0)
            y_ctx_batch = torch.stack(ctx_y_list, dim=0)
            x_q_batch = x_q_chunk.unsqueeze(0).expand(effective_num_bags, -1, -1)

            # Active tokens: padding mask is False
            ctx_key_padding_mask = torch.zeros(
                effective_num_bags, bag_size, dtype=torch.bool, device=self.device
            )

            batch_dict = {
                "x_ctx": x_ctx_batch,
                "y_ctx": y_ctx_batch,
                "ctx_key_padding_mask": ctx_key_padding_mask,
                "x_query": x_q_batch,
            }

            # Parallel Perceiver forward pass
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
                logits = self.model(batch_dict)  # (effective_num_bags, current_q_len, num_bins)

            probs = F.softmax(logits.float(), dim=-1)
            avg_probs = probs.mean(dim=0)  # (current_q_len, num_bins)
            all_chunk_probs.append(avg_probs.cpu())

        total_probs = torch.cat(all_chunk_probs, dim=0)

        # Restore original query indexing order
        if sort_order is not None:
            total_probs = total_probs[unsort_order.cpu()]

        return total_probs

    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts continuous values along with predictive uncertainty standard deviations.
        """
        probs = self.predict_distribution(X)
        bin_centers = self.bin_centers.cpu()

        expected_y_norm = torch.sum(probs * bin_centers, dim=-1)
        var_norm = torch.sum(probs * ((bin_centers - expected_y_norm.unsqueeze(-1)) ** 2), dim=-1)
        std_norm = torch.sqrt(torch.clamp(var_norm, min=0.0))

        y_pred = (expected_y_norm * self.y_scale_) + self.y_mean_
        std_pred = std_norm * self.y_scale_

        return y_pred.numpy(), std_pred.numpy()

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        preds, _ = self.predict_with_uncertainty(X)
        return preds

    def predict_classification(
        self,
        X: Union[np.ndarray, torch.Tensor],
        active_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.predict_distribution(X)
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
