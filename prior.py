"""Strong-signal, low-intrinsic-dimension smooth synthetic target prior.

REVISED after diagnosis: the first version's functions (3-8 additive terms,
full-width MLPs) were too complex to recover from ~64 sparse points in 2048-dim,
so the in-context signal was weak and the model collapsed to the mean predictor.

This version makes each function depend on only a FEW (2-4) dominant directions
of the embedding - mirroring how TabICL restricts its SCM targets to a small
number of "cause" features. On such a low-dimensional structure a small context
carries a strong, recoverable signal, so a well-trained in-context model can
clearly beat both the mean and (ideally) kNN.

All families are SMOOTH and low-complexity:
  * linear   : y = x . w                                  (1 direction)
  * additive : y = sum_{j<k} g_j(x . v_j), k=2..4         (2-4 directions)
  * rbf_gp   : RBF sum in a k=2..3-dim projection         (2-3 directions)
  * mlp      : random 2-layer MLP on a k=2..4-dim proj.   (2-4 directions)
Deterministic given a seed, vectorised, standardised per task.
"""
from __future__ import annotations

import math

import torch


def _randn(gen: torch.Generator, *shape, device) -> torch.Tensor:
    return torch.randn(*shape, generator=gen, device=device)


class SyntheticPrior:
    def __init__(
        self,
        embed_dim: int = 2048,
        p_linear: float = 0.25,
        p_additive: float = 0.40,
        p_rbf: float = 0.20,
        p_mlp: float = 0.15,
        noise_range: tuple[float, float] = (0.02, 0.05),
        seed: int | None = None,
    ) -> None:
        self.D = embed_dim
        weights = [p_linear, p_additive, p_rbf, p_mlp]
        s = sum(weights)
        self.probs = [w / s for w in weights]
        self.noise_range = noise_range
        self.rng = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()

    def _directions(self, x: torch.Tensor, k: int) -> torch.Tensor:
        V = _randn(self.rng, self.D, k, device=x.device)
        return V / V.norm(dim=0, keepdim=True).clamp_min(1e-6)

    # ------------------------------------------------------------------ #
    def _linear(self, x: torch.Tensor) -> torch.Tensor:
        w = _randn(self.rng, self.D, 1, device=x.device) / math.sqrt(self.D)
        return (x @ w).squeeze(-1)

    def _additive(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        k = int(torch.randint(1, 3, (1,), generator=self.rng).item())  # 1..2 dominant
        V = self._directions(x, k)
        P = x @ V  # (n, k)
        y = torch.zeros(n, device=x.device)
        for j in range(k):
            p = P[:, j]
            kind = int(torch.randint(0, 3, (1,), generator=self.rng).item())
            w = _randn(self.rng, 1, device=x.device).item() * 3.0
            b = _randn(self.rng, 1, device=x.device).item() * 0.5
            if kind == 0:
                y = y + w * p
            elif kind == 1:
                y = y + w * torch.tanh(p + b)
            else:
                y = y + w * torch.sin(p + b)
        return y

    def _rbf_gp(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        k = int(torch.randint(1, 3, (1,), generator=self.rng).item())  # 1..2-dim
        V = self._directions(x, k)
        Z = x @ V  # (n, k)
        n_c = int(torch.randint(6, 11, (1,), generator=self.rng).item())
        idx = torch.randperm(n, generator=self.rng, device=x.device)[:n_c]
        C = Z[idx]
        l = float(torch.rand(1, generator=self.rng).item()) * 1.2 + 0.5
        gamma = 1.0 / (2.0 * l * l)
        d2 = torch.cdist(Z, C) ** 2
        a = _randn(self.rng, n_c, 1, device=x.device) * 2.0
        return (torch.exp(-gamma * d2) @ a).squeeze(-1)

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(1, 3, (1,), generator=self.rng).item())  # 1..2-dim
        V = self._directions(x, k)
        Z = x @ V  # (n, k)
        H = int(torch.randint(16, 33, (1,), generator=self.rng).item())
        W1 = _randn(self.rng, k, H, device=x.device) / math.sqrt(k)
        b1 = _randn(self.rng, H, device=x.device)
        W2 = _randn(self.rng, H, 1, device=x.device) / math.sqrt(H)
        b2 = _randn(self.rng, 1, device=x.device)
        act = torch.tanh if int(torch.randint(0, 2, (1,), generator=self.rng).item()) == 0 \
            else torch.nn.functional.gelu
        h = act(Z @ W1 + b1)
        return (h @ W2 + b2).squeeze(-1)

    # ------------------------------------------------------------------ #
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r = float(torch.rand(1, generator=self.rng).item())
        cum = 0.0
        y = None
        for name, p in zip(["linear", "additive", "rbf_gp", "mlp"], self.probs):
            cum += p
            if r < cum:
                y = getattr(self, f"_{name}")(x)
                break
        if y is None:
            y = self._linear(x)

        sigma = self.noise_range[0] + (self.noise_range[1] - self.noise_range[0]) * \
            float(torch.rand(1, generator=self.rng).item())
        y = y + _randn(self.rng, x.size(0), device=x.device) * sigma

        mean = y.mean()
        std = y.std()
        if std < 1e-6:
            std = torch.ones_like(std)
        return ((y - mean) / std).unsqueeze(-1)
