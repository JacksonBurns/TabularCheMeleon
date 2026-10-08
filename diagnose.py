"""Diagnostic: what did the ICL model actually learn?

Compares, on identical synthetic episodes, three predictors for the queries:
  1. MEAN   : predict the context mean (the trivial baseline)
  2. kNN    : predict each query as the mean of its k nearest context neighbors
              (in embedding space) - a strong baseline for SMOOTH functions
  3. MODEL  : the trained in-context transformer

If MODEL ~= kNN >> mean, the model learned smooth local interpolation (good for
smooth priors, but that's what kNN already gives). If MODEL > kNN, the model is
learning the underlying function structure (extrapolating beyond local
neighbors) - the real in-context-regression capability. If MODEL ~= mean, it
learned little.

Also reports the per-function-family breakdown to see which task types transfer.
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
from prior import SyntheticPrior
from train import load_model

torch.manual_seed(123)

D = 2048
N_EP = 400
CTX = 64
Q = 16


def gen_episode(pool, prior, n_ctx, n_q, g):
    idx = torch.randperm(len(pool), generator=g)[:n_ctx + n_q]
    x = pool[idx]
    y = prior(x).squeeze(-1)
    return x[:n_ctx], y[:n_ctx], x[n_ctx:], y[n_ctx:]


def knn_pred(x_ctx, y_ctx, x_q, k=8):
    # cosine kNN
    xc = F.normalize(x_ctx, dim=-1)
    xq = F.normalize(x_q, dim=-1)
    sim = xq @ xc.T  # (nq, nctx)
    kk = min(k, len(x_ctx))
    nn_idx = sim.topk(kk, dim=-1).indices
    nn_y = y_ctx[nn_idx]
    return nn_y.mean(dim=-1)


def main():
    device = "cuda"
    pool = torch.load("data/pool_full.pt", map_location="cpu", weights_only=True).float()
    mu, sd = pool.mean(0), pool.std(0).clamp_min(1e-6)
    pool = (pool - mu) / sd

    prior = SyntheticPrior(embed_dim=D, seed=7)
    model = load_model(Path("icl_logs_v2/fixeddim_icl_v2.pt"), device=device).eval()

    g = torch.Generator().manual_seed(999)
    tot = {"mean": 0.0, "knn": 0.0, "model": 0.0}
    cnt = 0
    with torch.no_grad():
        for _ in range(N_EP):
            xc, yc, xq, yq = gen_episode(pool, prior, CTX, Q, g)
            xc, xq, yc, yq = xc.to(device), xq.to(device), yc.to(device), yq.to(device)
            ym, ys = yc.mean(), yc.std().clamp_min(1e-6)
            yc_std = ((yc - ym) / ys).unsqueeze(-1)

            # model in-context
            X_all = torch.cat([xc, xq], 0)
            y_all = torch.cat([yc_std, torch.zeros(Q, 1, device=device)], 0)
            mask = torch.zeros(len(X_all), dtype=torch.bool, device=device)
            mask[CTX:] = True
            pred = model(X_all.unsqueeze(0), y_all.unsqueeze(0), mask.unsqueeze(0))[0]
            pred_model = (pred[CTX:].squeeze(-1) * ys + ym).cpu()

            pred_mean = torch.full((Q,), ym.item(), device=device)
            pred_knn = knn_pred(xc, yc, xq, k=8)

            yq_c = yq.cpu()
            tot["mean"] += (pred_mean.cpu() - yq_c).abs().mean().item()
            tot["knn"] += (pred_knn.cpu() - yq_c).abs().mean().item()
            tot["model"] += (pred_model - yq_c).abs().mean().item()
            cnt += 1
    for k in tot:
        print(f"{k:8s} MAE (raw-std units): {tot[k]/cnt:.4f}")
    base = tot["mean"] / cnt
    print(f"\nmodel beats mean by {(base - tot['model']/cnt)/base*100:.1f}%")
    print(f"model beats kNN by {(tot['knn']/cnt - tot['model']/cnt)/(tot['knn']/cnt)*100:.1f}%")


if __name__ == "__main__":
    main()
