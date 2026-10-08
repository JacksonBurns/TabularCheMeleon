"""Decisive test: does bin-label encoding fix the mean-predictor collapse?

Pure linear episodes (strong known signal: ridge recovers +12..26% over mean).
Trains the fixed-dim ICL transformer and measures in-context prediction quality.
If corr(truth) rises and pred-std approaches truth-std, bin-label ICL works.
"""
import math
import torch
import lightning.pytorch as pl
from model import ModelConfig, FixedDimICLLightning
from train import EpisodicDataModule

torch.manual_seed(0)
D = 2048
pool = torch.load("data/pool_full.pt", map_location="cpu", weights_only=True).float()
mu, sd = pool.mean(0), pool.std(0).clamp_min(1e-6)
pool = (pool - mu) / sd


class LinearOnlyPrior:
    def __init__(self, embed_dim=D, seed=0):
        self.rng = torch.Generator().manual_seed(seed)

    def __call__(self, x):
        w = torch.randn(x.size(1), 1, generator=self.rng, device=x.device) / math.sqrt(x.size(1))
        y = (x @ w).squeeze(-1) + torch.randn(x.size(0), generator=self.rng, device=x.device) * 0.05
        return ((y - y.mean()) / y.std()).unsqueeze(-1)


prior = LinearOnlyPrior()
cfg = ModelConfig(embed_dim=D, d_model=256, n_layers=6, n_heads=8, dim_feedforward=1024,
                  max_context=256, task="regression", label_mode="bin", n_bins=32,
                  learning_rate=1e-3)
lm = FixedDimICLLightning(cfg)
dm = EpisodicDataModule(pool=pool, prior=prior, batch_episodes=32, num_workers=8,
                        train_episodes=16000, val_episodes=512,
                        min_ctx=32, max_ctx=128, max_query=16, standardise_pool=False)
trainer = pl.Trainer(max_epochs=3, accelerator="cuda", devices=1, precision="bf16-mixed",
                     log_every_n_steps=400, num_sanity_val_steps=0, enable_progress_bar=False)
trainer.fit(lm, dm)


@torch.inference_mode()
def measure(n=300):
    g = torch.Generator().manual_seed(5)
    lm.model = lm.model.to("cuda").eval()
    P, Y = [], []
    for _ in range(n):
        idx = torch.randperm(len(pool), generator=g)[:80]
        x = pool[idx]
        y = prior(x).squeeze(-1)
        xc, yc, xq, yq = x[:64], y[:64], x[64:], y[64:]
        ym, ys = yc.mean(), yc.std()
        yc_s = ((yc - ym) / ys).unsqueeze(-1)
        X_all = torch.cat([xc, xq], 0).cuda()
        y_all = torch.cat([yc_s.cuda(), torch.zeros(16, 1, device="cuda")], 0)
        mask = torch.zeros(80, dtype=torch.bool, device="cuda"); mask[64:] = True
        out = lm.model(X_all.unsqueeze(0), y_all.unsqueeze(0), mask.unsqueeze(0))[0]
        pm = (lm.model.point_pred(out)[64:].squeeze(-1) * ys + ym).cpu()
        P.append(pm); Y.append(yq)
    P = torch.cat(P); Y = torch.cat(Y)
    corr = torch.corrcoef(torch.stack([P, Y]))[0, 1].item()
    return P, Y, corr


P, Y, corr = measure()
mean_mae = (torch.full_like(Y, Y.mean()) - Y).abs().mean().item()
model_mae = (P - Y).abs().mean().item()
print(f"\npred mean={P.mean():.3f} std={P.std():.3f}   truth std={Y.std():.3f}")
print(f"pred-truth correlation = {corr:.3f}")
print(f"mean MAE={mean_mae:.4f}   model MAE={model_mae:.4f}")
print(f"model beats mean by {(mean_mae - model_mae) / mean_mae * 100:+.1f}%"
      f"  ({'LEARNED' if model_mae < mean_mae * 0.9 else 'NOT LEARNED'})")
