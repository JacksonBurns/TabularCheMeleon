from pathlib import Path
from typing import Literal, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError:
    class BaseEstimator: pass
    class RegressorMixin: pass

from tabularchemeleon import TabularCheMeleonLightningModule
from config import MIN_CONTEXT, MAX_CONTEXT, NUM_QUERIES

class TabularCheMeleonRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        max_context_size: int = MAX_CONTEXT,  # Configured for MAX_CONTEXT models
        chunk_size: int = NUM_QUERIES,         # Matches NUM_QUERIES-query evaluation blocks
        num_bags: int = 16,
        bag_strategy: Literal["hybrid", "knn", "random"] = "hybrid",
        knn_ratio: float = 0.65,
        robust_scaling: bool = True,
        clip_target_norm: float = 3.5,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_context_size = max_context_size
        self.chunk_size = chunk_size
        self.num_bags = num_bags
        self.bag_strategy = bag_strategy
        self.knn_ratio = knn_ratio
        self.robust_scaling = robust_scaling
        self.clip_target_norm = clip_target_norm

        self.model = TabularCheMeleonLightningModule.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        self.bin_centers = self.model.bin_centers.to(self.device)
        self.embed_dim = self.model.model.embed_dim

        self.is_fitted_ = False
        self.X_context_: Optional[torch.Tensor] = None
        self.y_context_norm_: Optional[torch.Tensor] = None
        self.y_mean_: float = 0.0
        self.y_scale_: float = 1.0

    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
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
            raise ValueError(f"Feature dimension mismatch: expected {self.embed_dim}, got {X_tensor.shape[1]}")

        # Robust scaling using Median and IQR
        if self.robust_scaling and len(y_tensor) >= 8:
            q25 = torch.quantile(y_tensor, 0.25).item()
            q75 = torch.quantile(y_tensor, 0.75).item()
            iqr = q75 - q25
            self.y_mean_ = torch.median(y_tensor).item()
            self.y_scale_ = (iqr / 1.349) if iqr > 1e-6 else (torch.std(y_tensor).item() + 1e-7)
        else:
            self.y_mean_ = torch.mean(y_tensor).item()
            self.y_scale_ = torch.std(y_tensor).item() + 1e-7

        y_norm = (y_tensor - self.y_mean_) / self.y_scale_
        self.y_context_norm_ = torch.clamp(y_norm, -self.clip_target_norm, self.clip_target_norm).to(self.device)
        self.X_context_ = X_tensor.to(self.device)
        self.is_fitted_ = True
        return self

    def _sample_context_bag(self, x_q_chunk: torch.Tensor, bag_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_total = self.X_context_.size(0)

        if n_total <= bag_size:
            perm = torch.randperm(n_total, device=self.device)
            return self.X_context_[perm], self.y_context_norm_[perm]

        if self.bag_strategy == "random":
            indices = torch.randperm(n_total, device=self.device)[:bag_size]
            return self.X_context_[indices], self.y_context_norm_[indices]

        # Calculate distances from query chunk to full context library
        dists = torch.cdist(x_q_chunk, self.X_context_, p=2)
        min_dists = dists.min(dim=0).values

        if self.bag_strategy == "knn":
            topk_indices = torch.topk(min_dists, k=bag_size, largest=False).indices
            return self.X_context_[topk_indices], self.y_context_norm_[topk_indices]

        # Hybrid: nearest neighbors + random diverse anchors
        k_near = int(bag_size * self.knn_ratio)
        k_rand = bag_size - k_near

        nearest_indices = torch.topk(min_dists, k=k_near, largest=False).indices
        mask = torch.ones(n_total, dtype=torch.bool, device=self.device)
        mask[nearest_indices] = False
        remaining_indices = torch.where(mask)[0]

        rand_picks = remaining_indices[torch.randperm(len(remaining_indices), device=self.device)[:k_rand]]
        combined_indices = torch.cat([nearest_indices, rand_picks])

        perm = torch.randperm(bag_size, device=self.device)
        return self.X_context_[combined_indices[perm]], self.y_context_norm_[combined_indices[perm]]

    @torch.inference_mode()
    def predict_with_uncertainty(self, X: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit(X, y) before predict().")

        if isinstance(X, np.ndarray):
            X_query = torch.from_numpy(X).float()
        else:
            X_query = X.clone().detach().float()

        num_queries = X_query.size(0)
        effective_bag_size = min(self.max_context_size, self.X_context_.size(0))

        all_means = []
        all_stds = []

        for q_idx in range(0, num_queries, self.chunk_size):
            x_q_chunk = X_query[q_idx : q_idx + self.chunk_size].to(self.device)
            # FIX: Index 0 is the number of molecules in the chunk
            current_q_len = x_q_chunk.size(0)
            dummy_y = torch.zeros(1, current_q_len, 1, device=self.device)

            bag_probs = []
            for _ in range(self.num_bags):
                x_ctx_sub, y_ctx_sub = self._sample_context_bag(x_q_chunk, effective_bag_size)

                x_combined = torch.cat([x_ctx_sub.unsqueeze(0), x_q_chunk.unsqueeze(0)], dim=1)
                y_combined = torch.cat([y_ctx_sub.unsqueeze(0), dummy_y], dim=1)

                batch_dict = {
                    "x": x_combined,
                    "y": y_combined,
                    "context_lens": [effective_bag_size],
                    "query_len": current_q_len,
                    "sample_mask": None,
                }

                logits = self.model(batch_dict).squeeze(0)  # (current_q_len, num_bins)
                bag_probs.append(F.softmax(logits, dim=-1))

            avg_probs = torch.stack(bag_probs, dim=0).mean(dim=0)

            # Continuous moments
            expected_y_norm = torch.sum(avg_probs * self.bin_centers, dim=-1, keepdim=True)
            var_norm = torch.sum(avg_probs * ((self.bin_centers - expected_y_norm) ** 2), dim=-1, keepdim=True)
            std_norm = torch.sqrt(var_norm)

            y_pred = (expected_y_norm * self.y_scale_) + self.y_mean_
            std_pred = std_norm * self.y_scale_

            all_means.append(y_pred.cpu())
            all_stds.append(std_pred.cpu())

        y_preds = torch.cat(all_means, dim=0).numpy().squeeze(-1)
        y_stds = torch.cat(all_stds, dim=0).numpy().squeeze(-1)

        return y_preds, y_stds

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        preds, _ = self.predict_with_uncertainty(X)
        return preds

    def fit_predict(self, X_context: Union[np.ndarray, torch.Tensor], y_context: Union[np.ndarray, torch.Tensor], X_query: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        self.fit(X_context, y_context)
        return self.predict(X_query)