from pathlib import Path
from typing import Tuple
import torch
import torch.nn.functional as F

from tabularchemeleon import TabularCheMeleonLightningModule


class CheMeleonPredictor:
    """
    Inference engine for TabularCheMeleon models.
    Supports both continuous bioactivity regression with uncertainty intervals
    and binary classification via Riemann posterior tail integration.
    """
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.model = TabularCheMeleonLightningModule.load_from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Fixed bin support from the model (e.g., 64 bins in [-4.0, 4.0])
        self.bin_centers = self.model.bin_centers.to(self.device)
        self.embed_dim = self.model.model.embed_dim

    @torch.inference_mode()
    def _evaluate_distribution(
        self,
        x_context: torch.Tensor,
        y_context_norm: torch.Tensor,
        x_query: torch.Tensor,
        num_bags: int = 16,
        bag_fraction: float = 0.85,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Evaluates bagged predictive distributions over discretized support bins."""
        n_total_ctx = x_context.size(0)
        num_queries = x_query.size(0)
        bag_size = max(8, int(n_total_ctx * bag_fraction))

        all_query_probs = []

        for q_idx in range(0, num_queries, chunk_size):
            x_q_chunk = x_query[q_idx : q_idx + chunk_size].unsqueeze(0).to(self.device)
            current_q_len = x_q_chunk.size(1)
            dummy_y = torch.zeros(1, current_q_len, 1, device=self.device)

            bag_probs = []
            for _ in range(num_bags):
                perm = torch.randperm(n_total_ctx)[:bag_size]
                x_ctx_sub = x_context[perm].unsqueeze(0).to(self.device)
                y_ctx_sub = y_context_norm[perm].unsqueeze(0).to(self.device)

                x_combined = torch.cat([x_ctx_sub, x_q_chunk], dim=1)
                y_combined = torch.cat([y_ctx_sub, dummy_y], dim=1)

                batch_dict = {
                    "x": x_combined,
                    "y": y_combined,
                    "context_lens": [bag_size],
                    "query_len": current_q_len,
                    "sample_mask": None,
                }

                logits = self.model(batch_dict).squeeze(0)  # (current_q_len, num_bins)
                probs = F.softmax(logits, dim=-1)
                bag_probs.append(probs)

            # Average probability distributions across bags
            avg_probs = torch.stack(bag_probs, dim=0).mean(dim=0)
            all_query_probs.append(avg_probs)

        return torch.cat(all_query_probs, dim=0)  # (N_query, num_bins)

    def predict_regression(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
        num_bags: int = 16,
        bag_fraction: float = 0.85,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs continuous bioactivity prediction.

        Returns:
            mean: (N_query, 1) Expected value in original physical units (e.g. pIC50).
            std:  (N_query, 1) Predictive uncertainty in original physical units.
        """
        if y_context.dim() == 1:
            y_context = y_context.unsqueeze(-1)

        # Standardize context targets to align with training prior
        mu, sigma = y_context.mean(), y_context.std() + 1e-7
        y_context_norm = (y_context - mu) / sigma

        probs = self._evaluate_distribution(
            x_context=x_context,
            y_context_norm=y_context_norm,
            x_query=x_query,
            num_bags=num_bags,
            bag_fraction=bag_fraction,
            chunk_size=chunk_size,
        )

        # Calculate moments in normalized space
        expected_y_norm = torch.sum(probs * self.bin_centers, dim=-1, keepdim=True)
        var_norm = torch.sum(probs * ((self.bin_centers - expected_y_norm) ** 2), dim=-1, keepdim=True)
        std_norm = torch.sqrt(var_norm)

        # Rescale predictions back to raw physical assay units
        expected_y = (expected_y_norm * sigma) + mu
        expected_std = std_norm * sigma

        return expected_y.cpu(), expected_std.cpu()

    def predict_classification(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
        active_threshold: float,
        num_bags: int = 16,
        bag_fraction: float = 0.85,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs binary active/inactive classification via tail probability integration.

        Args:
            active_threshold: Cutoff in raw assay units (e.g. 6.0 for pIC50 >= 1 uM).

        Returns:
            prob_active: (N_query, 1) Posterior probability P(Y >= threshold).
            predicted_class: (N_query, 1) Binary prediction (1 for Active, 0 for Inactive).
        """
        if y_context.dim() == 1:
            y_context = y_context.unsqueeze(-1)

        mu, sigma = y_context.mean(), y_context.std() + 1e-7
        y_context_norm = (y_context - mu) / sigma

        # Map active cutoff to normalized support grid
        threshold_norm = (active_threshold - mu) / sigma
        active_bins_mask = (self.bin_centers >= threshold_norm)

        probs = self._evaluate_distribution(
            x_context=x_context,
            y_context_norm=y_context_norm,
            x_query=x_query,
            num_bags=num_bags,
            bag_fraction=bag_fraction,
            chunk_size=chunk_size,
        )

        # Integrate tail: sum probabilities of all bins above the cutoff
        prob_active = probs[:, active_bins_mask].sum(dim=-1, keepdim=True)
        predicted_class = (prob_active >= 0.5).long()

        return prob_active.cpu(), predicted_class.cpu()


# ==============================================================================
# Demonstration Script
# ==============================================================================

if __name__ == "__main__":
    checkpoint_file = "tabular_chemeleon_logs/inductive_tabular_chemeleon/version_0/checkpoints/best-epoch=12-val_loss=0.0821.ckpt"

    # Initialize predictor
    predictor = CheMeleonPredictor(checkpoint_path=checkpoint_file)
    embed_dim = predictor.embed_dim

    # --------------------------------------------------------------------------
    # Shared Experimental Context (Validated Wet-Lab Assay Hits)
    # --------------------------------------------------------------------------
    num_context = 48
    x_context = torch.randn(num_context, embed_dim)
    # Measured binding affinities (e.g., pIC50 ranging from 4.2 to 8.5)
    y_context = torch.empty(num_context, 1).uniform_(4.2, 8.5)

    # --------------------------------------------------------------------------
    # Candidate Screening Deck
    # --------------------------------------------------------------------------
    num_candidates = 500
    x_candidates = torch.randn(num_candidates, embed_dim)

    # ==========================================================================
    # Demo 1: Regression Task (Predicting exact continuous potency + uncertainty)
    # ==========================================================================
    print("\n--- Running Continuous Regression ---")
    pred_pIC50, pred_uncertainty = predictor.predict_regression(
        x_context=x_context,
        y_context=y_context,
        x_query=x_candidates,
        num_bags=8,
        chunk_size=64,
    )

    print(f"Regression outputs shape: {pred_pIC50.shape}")
    print("Sample Candidate Predictions (Mean +/- 1 Sigma):")
    for i in range(5):
        val = pred_pIC50[i].item()
        sigma = pred_uncertainty[i].item()
        print(f"  Molecule {i+1:02d}: pIC50 = {val:.2f} +/- {sigma:.2f}")

    # ==========================================================================
    # Demo 2: Binary Classification Task (Hit Picking: pIC50 >= 6.0)
    # ==========================================================================
    print("\n--- Running Binary Classification (Hit Threshold: pIC50 >= 6.0) ---")
    active_cutoff = 6.0  # Corresponds to <= 1 uM potency

    prob_active, class_pred = predictor.predict_classification(
        x_context=x_context,
        y_context=y_context,
        x_query=x_candidates,
        active_threshold=active_cutoff,
        num_bags=8,
        chunk_size=64,
    )

    print(f"Classification outputs shape: {prob_active.shape}")
    print("Sample Candidate Hit Probabilities:")
    for i in range(5):
        p = prob_active[i].item()
        call = "ACTIVE" if class_pred[i].item() == 1 else "INACTIVE"
        print(f"  Molecule {i+1:02d}: P(Active) = {p:.3f} -> Predicted: {call}")

    num_hits = (class_pred == 1).sum().item()
    print(f"\nTotal Identified Hits: {num_hits} / {num_candidates} ({100 * num_hits / num_candidates:.1f}%)")
