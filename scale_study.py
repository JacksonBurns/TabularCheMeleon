"""Scale study: does in-context LINEAR regression emerge with more compute?

Trains the fixed-dim ICL transformer on pure linear episodes (strong, known
signal: ridge recovers +12..26% over mean) and, every few epochs, measures the
in-context prediction quality. If correlation with truth rises and pred-std
approaches truth-std as compute grows, the capability is a compute gap (the
honest answer to the user's 'how do we close the disconnect'). If it stays flat
at ~0 correlation / tiny std, it's a fundamental failure at this architecture
scale.
"""
import math
import torch
import torch.nn.functional as F
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
                  max_context=256, task="regression", loss="pinball", n_quantiles=21,
                  label_scale=4.0, learning_rate=1e-3)
lm = FixedDimICLLightning(cfg)
dm = EpisodicDataModule(pool=pool, prior=prior, batch_episodes=32, num_workers=8,
                        train_episodes=16000, val_episodes=512,
                        min_ctx=32, max_ctx=128, max_query=16, standardise_pool=False)
dm.setup("fit")


@torch.inference_mode()
def evaluate():
    g = torch.Generator().manual_seed(5)
    lm.model.eval()
    P, Y = [], []
    for _ in range(200):
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
        pm = (lm.model.median(out)[64:].squeeze(-1) * ys + ym).cpu()
        P.append(pm); Y.append(yq)
    P = torch.cat(P); Y = torch.cat(Y)
    corr = torch.corrcoef(torch.stack([P, Y]))[0, 1].item()
    mean_mae = (torch.full_like(Y, Y.mean()) - Y).abs().mean().item()
    model_mae = (P - Y).abs().mean().item()
    return corr, P.std().item(), Y.std().item(), mean_mae, model_mae


# manual training loop so we can evaluate on a schedule
lm.model = lm.model.to("cuda")
opt = torch.optim.AdamW(lm.parameters(), lr=cfg.learning_rate, weight_decay=1e-4, betas=(0.9, 0.98))
n_epochs = 30
steps_per_epoch = 16000 // 32
total = n_epochs * steps_per_epoch
warmup = total // 10
def lr_lambda(s):
    if s < warmup: return (s + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

print("epoch  corr    pred_std  truth_std  mean_mae  model_mae  beat_mean%")
eval_every = 3
for ep in range(n_epochs):
    lm.model.train()
    dl = dm.train_dataloader()
    for batch in dl:
        x, y, mask, kpm = batch["x"].cuda(), batch["y"].cuda(), batch["mask"].cuda(), batch["key_padding_mask"].cuda()
        out = lm.model(x, y, mask, kpm)
        valid = mask & ~kpm
        yq = y[valid].squeeze(-1)
        pq = out[valid]
        levels = lm.model.quantile_levels.cuda()
        err = yq.unsqueeze(-1) - pq
        loss = torch.mean(torch.max(levels * err, (levels - 1) * err))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step(); sched.step()
    if (ep + 1) % eval_every == 0:
        corr, ps, ts, mm, om = evaluate()
        print(f"{ep+1:5d}  {corr:+.3f}  {ps:.3f}     {ts:.3f}     {mm:.4f}   {om:.4f}   "
              f"{(mm-om)/mm*100:+.1f}%", flush=True)
print("done")
