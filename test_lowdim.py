"""Control: does in-context linear regression work in LOW dim?

Same model, same linear task, but the feature space is projected to `F` dims
(32 here) BEFORE the model. If the ICL transformer learns in-context regression
in 32-dim (corr rises, pred-std ~ truth-std) but not in 2048-dim, the feature
bottleneck is the obstacle - and the fix is a better/more-capacity feature
projection, not a different ICL architecture.
"""
import math
import torch
import lightning.pytorch as pl
from model import ModelConfig, FixedDimICLLightning
from train import EpisodicDataModule

torch.manual_seed(0)


class LinearPrior:
    def __init__(self, embed_dim, seed=0):
        self.D = embed_dim
        self.rng = torch.Generator().manual_seed(seed)

    def __call__(self, x):
        w = torch.randn(x.size(1), 1, generator=self.rng, device=x.device) / math.sqrt(x.size(1))
        y = (x @ w).squeeze(-1) + torch.randn(x.size(0), generator=self.rng, device=x.device) * 0.05
        return ((y - y.mean()) / y.std()).unsqueeze(-1)


def run(F, tag):
    torch.manual_seed(0)
    pool = torch.load("data/pool_full.pt", map_location="cpu", weights_only=True).float()
    mu, sd = pool.mean(0), pool.std(0).clamp_min(1e-6)
    pool = (pool - mu) / sd
    # project to F dims (random orthonormal)
    g = torch.Generator().manual_seed(1)
    Wp = torch.randn(2048, F, generator=g)
    Wp, _ = torch.linalg.qr(Wp)  # orthonormal columns
    poolF = pool @ Wp  # (N, F)
    poolF = (poolF - poolF.mean(0)) / poolF.std(0).clamp_min(1e-6)

    prior = LinearPrior(F)
    cfg = ModelConfig(embed_dim=F, d_model=128, n_layers=6, n_heads=8, dim_feedforward=512,
                      max_context=256, task="regression", label_mode="bin", n_bins=32,
                      learning_rate=1e-3)
    lm = FixedDimICLLightning(cfg)
    dm = EpisodicDataModule(pool=poolF, prior=prior, batch_episodes=32, num_workers=8,
                            train_episodes=16000, val_episodes=512,
                            min_ctx=32, max_ctx=128, max_query=16, standardise_pool=False)
    trainer = pl.Trainer(max_epochs=3, accelerator="cuda", devices=1, precision="bf16-mixed",
                         log_every_n_steps=1000, num_sanity_val_steps=0, enable_progress_bar=False)
    trainer.fit(lm, dm)

    @torch.inference_mode()
    def measure(n=300):
        gg = torch.Generator().manual_seed(5)
        lm.model = lm.model.to("cuda").eval()
        P, Y = [], []
        for _ in range(n):
            idx = torch.randperm(len(poolF), generator=gg)[:80]
            x = poolF[idx]
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
        mean_mae = (torch.full_like(Y, Y.mean()) - Y).abs().mean().item()
        model_mae = (P - Y).abs().mean().item()
        return P.std().item(), Y.std().item(), corr, mean_mae, model_mae

    ps, ts, corr, mm, om = measure()
    print(f"\n[{tag}] F={F}: pred_std={ps:.3f} truth_std={ts:.3f} corr={corr:.3f} "
          f"mean_mae={mm:.4f} model_mae={om:.4f} beat_mean={(mm-om)/mm*100:+.1f}% "
          f"{'LEARNED' if om < mm*0.9 else 'NOT LEARNED'}", flush=True)
    torch.cuda.empty_cache()


run(32, "low-dim")
run(128, "mid-dim")
