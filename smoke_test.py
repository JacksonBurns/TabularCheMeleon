"""Smoke test: prior, model forward, full Lightning train loop, and ICL predict."""
import torch
import torch.nn.functional as F
from prior import SyntheticPrior
from model import FixedDimICLTransformer, ModelConfig, FixedDimICLLightning
from train import EpisodicICLDataset, EpisodicDataModule, collate

torch.manual_seed(0)
D = 2048
N = 64

# 1) prior
prior = SyntheticPrior(embed_dim=D, seed=0)
x = torch.randn(N, D)
y = prior(x)
assert y.shape == (N, 1)
print("prior ok, y std", round(y.std().item(), 3))

# 2) model forward, batched (B=1) with padding
cfg = ModelConfig(embed_dim=D, d_model=64, n_layers=2, n_heads=4,
                  dim_feedforward=128, max_context=256, task="regression")
m = FixedDimICLTransformer(cfg)
xb = x.unsqueeze(0); yb = y.unsqueeze(0)
mask = torch.zeros(1, N, dtype=torch.bool); mask[0, 40:] = True
kpm = torch.zeros(1, N, dtype=torch.bool)
out = m(xb, yb, mask, kpm)
print("forward ok, out", tuple(out.shape), "| params",
      round(sum(p.numel() for p in m.parameters()) / 1e6, 2), "M")
loss = F.mse_loss(out[0, 40:, 0], y[40:, 0])
loss.backward()
print("backward ok, loss", round(loss.item(), 4))

# 3) full Lightning train loop on the real pool (tiny model, few steps)
pool = torch.load("data/pool_full.pt", map_location="cpu", weights_only=True)
print("pool", tuple(pool.shape))

lm_cfg = ModelConfig(embed_dim=D, d_model=128, n_layers=3, n_heads=4,
                     dim_feedforward=256, max_context=300, task="regression",
                     learning_rate=3e-4)
lm = FixedDimICLLightning(lm_cfg)
dm = EpisodicDataModule(pool=pool, prior=prior, batch_episodes=4,
                        train_episodes=8, val_episodes=4,
                        min_ctx=16, max_ctx=64, max_query=16, standardise_pool=True)

import lightning.pytorch as pl
trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1,
                     log_every_n_steps=1, num_sanity_val_steps=2)
trainer.fit(lm, dm)
print("LIGHTNING TRAIN LOOP OK")

# 4) in-context predict with the (untrained) model, sanity of path
mu = pool.mean(0); sd = pool.std(0).clamp_min(1e-6)
xs = (pool[:32] - mu) / sd
Xq = (pool[32:48] - mu) / sd
ymu = torch.tensor([0.0]); ysd = torch.tensor([1.0])
X_all = torch.cat([xs, Xq]); y_all = torch.cat([xs.new_zeros(32, 1), Xq.new_zeros(16, 1)])
mk = torch.zeros(48, dtype=torch.bool); mk[32:] = True
pred = m(X_all.unsqueeze(0), y_all.unsqueeze(0), mk.unsqueeze(0))[0]
print("ICL predict path ok, preds", tuple(pred[32:].shape))
print("ALL SMOKE TESTS PASSED")
