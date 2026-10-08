"""Two decisive diagnostics.

(1) CONTEXT-SENSITIVITY probe on the already-trained linear model: for a fixed
    query, does the prediction change when we (a) shuffle context labels,
    (b) replace context with its mean, (c) use a different context? If the
    prediction is invariant to context, the model is NOT using context at all.

(2) SINGLE-DIRECTION linear training test: y = x.v (v a single random unit
    direction, 1-D intrinsic). This is the CLEANEST strong-signal in-context
    task: 64 context points genuinely determine the 1-D projection. If the
    model learns THIS (corr rises, pred_std ~ truth_std), the collapse was a
    task-difficulty problem (high intrinsic dim), not an architecture failure.
    If it still collapses, the obstacle is deeper (training dynamics at scale).
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


class SingleDirPrior:
    """y = x.v for a single random unit direction v (1-D intrinsic)."""
    def __init__(self, embed_dim=D, seed=0):
        self.D = embed_dim
        self.rng = torch.Generator().manual_seed(seed)

    def __call__(self, x):
        v = torch.randn(self.D, 1, generator=self.rng, device=x.device)
        v = v / v.norm()
        y = (x @ v).squeeze(-1) + torch.randn(x.size(0), generator=self.rng, device=x.device) * 0.05
        return ((y - y.mean()) / y.std()).unsqueeze(-1)


# ---- (1) context-sensitivity probe on the multi-dir linear model ----
def probe():
    torch.manual_seed(0)
    prior = LinearOnlyPrior()
    cfg = ModelConfig(embed_dim=D, d_model=256, n_layers=6, n_heads=8, dim_feedforward=1024,
                      max_context=256, task="regression", label_mode="bin", n_bins=32,
                      learning_rate=1e-3)
    lm = FixedDimICLLightning(cfg)
    dm = EpisodicDataModule(pool=pool, prior=prior, batch_episodes=32, num_workers=8,
                            train_episodes=8000, val_episodes=256,
                            min_ctx=32, max_ctx=128, max_query=16, standardise_pool=False)
    trainer = pl.Trainer(max_epochs=1, accelerator="cuda", devices=1, precision="bf16-mixed",
                         log_every_n_steps=1000, num_sanity_val_steps=0, enable_progress_bar=False)
    trainer.fit(lm, dm)
    lm.model = lm.model.to("cuda").eval()

    @torch.inference_mode()
    def pred_for(xc, yc, xq):
        ym, ys = yc.mean(), yc.std()
        yc_s = ((yc - ym) / ys).unsqueeze(-1)
        X_all = torch.cat([xc, xq], 0).cuda()
        y_all = torch.cat([yc_s.cuda(), torch.zeros(len(xq), 1, device="cuda")], 0)
        mask = torch.zeros(len(X_all), dtype=torch.bool, device="cuda"); mask[len(xc):] = True
        out = lm.model(X_all.unsqueeze(0), y_all.unsqueeze(0), mask.unsqueeze(0))[0]
        return (lm.model.point_pred(out)[len(xc):].squeeze(-1) * ys + ym).cpu()

    g = torch.Generator().manual_seed(7)
    idx = torch.randperm(len(pool), generator=g)[:96]
    x = pool[idx]; y = prior(x).squeeze(-1)
    xc, yc, xq = x[:64], y[:64], x[64:80]
    p_base = pred_for(xc, yc, xq)
    p_shuf = pred_for(xc, torch.roll(yc, 3), xq)          # shuffle labels
    p_meanctx = pred_for(pool[:64], torch.randn(64, generator=g), xq)  # diff context
    p_flipped = pred_for(xc, -yc, xq)                      # flip labels
    print("\n[1] CONTEXT-SENSITIVITY (higher = uses context):")
    print(f"    |p_base - p_shuffled_labels|  = {(p_base - p_shuf).abs().mean().item():.4f}")
    print(f"    |p_base - p_flipped_labels|   = {(p_base - p_flipped).abs().mean().item():.4f}")
    print(f"    |p_base - p_diff_context|     = {(p_base - p_meanctx).abs().mean().item():.4f}")
    print(f"    (if all ~0, the model ignores context entirely)")
    torch.cuda.empty_cache()


# ---- (2) single-direction linear training ----
def single_dir(n_epochs=6):
    torch.manual_seed(0)
    prior = SingleDirPrior()
    cfg = ModelConfig(embed_dim=D, d_model=256, n_layers=6, n_heads=8, dim_feedforward=1024,
                      max_context=256, task="regression", label_mode="bin", n_bins=32,
                      learning_rate=1e-3)
    lm = FixedDimICLLightning(cfg)
    dm = EpisodicDataModule(pool=pool, prior=prior, batch_episodes=32, num_workers=8,
                            train_episodes=8000, val_episodes=512,
                            min_ctx=32, max_ctx=128, max_query=16, standardise_pool=False)
    dm.setup("fit")

    @torch.inference_mode()
    def measure(n=300):
        g = torch.Generator().manual_seed(5)
        lm.model.eval()
        P, Y = [], []
        for _ in range(n):
            idx = torch.randperm(len(pool), generator=g)[:80]
            x = pool[idx]; y = prior(x).squeeze(-1)
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
        mean_mae = (torch.full_like(Y, Y.mean()) - Y).abs().mean().item()
        model_mae = (P - Y).abs().mean().item()
        return P.std().item(), corr, mean_mae, model_mae

    # manual loop with periodic eval
    lm.model = lm.model.to("cuda")
    opt = torch.optim.AdamW(lm.parameters(), lr=cfg.learning_rate, weight_decay=1e-4, betas=(0.9, 0.98))
    steps_per_epoch = 8000 // 32
    total = n_epochs * steps_per_epoch
    warmup = total // 10
    def lr_lambda(s):
        if s < warmup: return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(total - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"\n[2] SINGLE-DIRECTION linear, {n_epochs} epochs:")
    print("    epoch  pred_std  corr    mean_mae  model_mae  beat_mean%")
    for ep in range(n_epochs):
        lm.model.train()
        for batch in dm.train_dataloader():
            xb, yb, mb, kb = batch["x"].cuda(), batch["y"].cuda(), batch["mask"].cuda(), batch["key_padding_mask"].cuda()
            out = lm.model(xb, yb, mb, kb)
            valid = mb & ~kb
            yq = yb[valid].squeeze(-1)
            target_bin = lm.model._bin_of(yb)[valid]
            loss = torch.nn.functional.cross_entropy(out[valid], target_bin)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step(); sched.step()
        ps, corr, mm, om = measure()
        print(f"    {ep+1:5d}  {ps:.3f}   {corr:+.3f}  {mm:.4f}   {om:.4f}   {(mm-om)/mm*100:+.1f}% "
              f"{'LEARNED' if om < mm*0.9 else ''}", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe()
    else:
        probe()
        single_dir()
