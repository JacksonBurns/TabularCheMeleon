"""Learned low-rank Mahalanobis metric (STABLE) for KRR.

Z = X @ V, kernel RBF in Z.  V (D x r) INIT = SVD-whitening transform (the
fixed correlation-exploiting metric).  Refine V with:
    loss = -val_metric + LAMBDA * ||V - V0||^2        # anchor to the good init
Length scale l FIXED at the SVD-whitening median distance (not co-optimized,
which couples badly and was the divergence driver).  float32, low lr, so V can
only make a *bounded* refinement of the SVD metric.

Fairly tests: does a data-driven learned metric beat the fixed SVD-whitening
transform (and the raw ambient metric)?

Two-stage: optimize V on <=1200-pt subsample val; refit KRR (fixed alpha) on
FULL train; score untouched test.
"""
from __future__ import annotations
import datetime, json, warnings
import numpy as np, pandas as pd, polaris as po
import torch
warnings.filterwarnings("ignore")
import kernel_baseline as kb

SEED = 0
R_METRIC = 16
STEPS = 200
ALPHA = 0.1
SUB = 1200
LAMBDA = 0.5
LR = 0.005


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


def main():
    cache = torch_load(kb.CACHE)
    HEAD = ["polaris/pkis2-ret-wt-reg-v2", "polaris/pkis2-kit-wt-reg-v2",
            "polaris/pkis2-egfr-wt-reg-v2", "polaris/adme-fang-solu-1",
            "polaris/adme-fang-rppb-1", "polaris/adme-fang-hppb-1",
            "polaris/adme-fang-perm-1", "polaris/adme-fang-rclint-1",
            "polaris/adme-fang-hclint-1"]
    out = open("kernel_learned_metric_results.md", "w")
    out.write(f"# Learned Low-Rank Mahalanobis Metric (STABLE, SVD-anchored)\n\n"
              f"{datetime.datetime.now()}\n\nZ=X@V, V init=SVD-whitening, "
              f"loss=-val+{LAMBDA}||V-V0||^2, l fixed. Refit full train, score test.\n\n")
    summary = {}
    for name in HEAD:
        d = cache[name]
        Xtr = d["X_train"].float().numpy().astype("float64")
        Xte = d["X_test"].float().numpy().astype("float64")
        ytr = np.asarray(d["y_train"], float)
        ytr_s = (ytr - ytr.mean()) / (ytr.std() + 1e-12)
        mname = d["metric"]
        hgb = kb.is_higher_better(mname)
        n = len(ytr)
        b = po.load_benchmark(name)

        mu, sd = kb.standardize_fit(Xtr)
        Xtr_s = kb.standardize_apply(Xtr, mu, sd)
        Xte_s = kb.standardize_apply(Xte, mu, sd)

        # SVD whitening transform (init)
        V_svd, s_svd, wmu, r = kb.whiten_fit(Xtr_s, 0.999)
        V0 = V_svd / s_svd
        Zs0 = Xtr_s @ V0
        l = kb.median_pairwise_dist(Zs0)

        # subsample
        rng = np.random.default_rng(SEED)
        sub = rng.choice(n, min(n, SUB), replace=False) if n > SUB else np.arange(n)
        perm = rng.permutation(len(sub))
        nv = max(8, int(0.2 * len(sub)))
        vi, si = perm[:nv], perm[nv:]
        Xs = Xtr_s[sub[si]]; Xv = Xtr_s[sub[vi]]
        ys = ytr_s[sub[si]]; yv = ytr_s[sub[vi]]

        Xs_t = torch.tensor(Xs, dtype=torch.float32)
        Xv_t = torch.tensor(Xv, dtype=torch.float32)
        ys_t = torch.tensor(ys, dtype=torch.float32)
        yv_t = torch.tensor(yv, dtype=torch.float32)
        V0_t = torch.tensor(V0, dtype=torch.float32)
        V = V0_t.clone().requires_grad_(True)
        opt = torch.optim.Adam([V], lr=LR)

        best_V, best_score = V0_t.clone(), -np.inf
        for step in range(STEPS):
            opt.zero_grad()
            Zs = Xs_t @ V
            Zv = Xv_t @ V
            d2_ss = (Zs ** 2).sum(1)[:, None] + (Zs ** 2).sum(1)[None, :] - 2.0 * (Zs @ Zs.T)
            d2_vs = (Zv ** 2).sum(1)[:, None] + (Zs ** 2).sum(1)[None, :] - 2.0 * (Zv @ Zs.T)
            Kss = torch.exp(-0.5 * d2_ss / (l * l)) + (ALPHA + 1e-6) * torch.eye(len(Zs))
            Kvs = torch.exp(-0.5 * d2_vs / (l * l))
            beta = torch.linalg.solve(Kss, ys_t)
            pv = Kvs @ beta
            if hgb:
                pv_c = pv - pv.mean(); yv_c = yv_t - yv_t.mean()
                score = (pv_c * yv_c).sum() / (pv_c.norm() * yv_c.norm() + 1e-6)
            else:
                score = -((pv - yv_t) ** 2).mean()
            loss = -score + LAMBDA * ((V - V0_t) ** 2).mean()
            loss.backward()
            opt.step()
            if score.item() > best_score:
                best_score = score.item()
                best_V = V.detach().clone()

        def full_eval(Vn):
            Ztr = Xtr_s @ Vn; Zte = Xte_s @ Vn
            d2_tr = (Ztr ** 2).sum(1)[:, None] + (Ztr ** 2).sum(1)[None, :] - 2.0 * (Ztr @ Ztr.T)
            d2_te = (Zte ** 2).sum(1)[:, None] + (Ztr ** 2).sum(1)[None, :] - 2.0 * (Zte @ Ztr.T)
            Ktr = np.exp(-0.5 * d2_tr / (l * l)) + (ALPHA + 1e-8) * np.eye(n)
            beta = np.linalg.solve(Ktr, ytr_s)
            pred = (np.exp(-0.5 * d2_te / (l * l)) @ beta) * ytr.std() + ytr.mean()
            return float(b.evaluate(pred.astype("float64")).results
                         .query(f"Metric == '{mname}'")["Score"].values[0])

        learned = full_eval(best_V.numpy())
        svd_ref = full_eval(V0)
        # true ambient raw reference: use full standardized (no projection)
        Ztr_r = Xtr_s; Zte_r = Xte_s
        d2_tr = (Ztr_r ** 2).sum(1)[:, None] + (Ztr_r ** 2).sum(1)[None, :] - 2.0 * (Ztr_r @ Ztr_r.T)
        d2_te = (Zte_r ** 2).sum(1)[:, None] + (Ztr_r ** 2).sum(1)[None, :] - 2.0 * (Zte_r @ Ztr_r.T)
        Ktr = np.exp(-0.5 * d2_tr / (l * l)) + (ALPHA + 1e-8) * np.eye(n)
        beta = np.linalg.solve(Ktr, ytr_s)
        raw_ref = float((b.evaluate((np.exp(-0.5 * d2_te / (l * l)) @ beta) * ytr.std() + ytr.mean()
                      ).results.query(f"Metric == '{mname}'")["Score"].values[0]))

        moved = float(np.linalg.norm(best_V.numpy() - V0) / (np.linalg.norm(V0) + 1e-9))
        summary[name] = dict(learned=learned, svd_white=svd_ref, raw_ambient=raw_ref,
                             metric=mname, n_train=n, val_opt=round(best_score, 4),
                             V_moved_pct=round(moved * 100, 1))
        print(f"{name.replace('polaris/',''):26s} learned={learned:.4f}  "
              f"svdWhite={svd_ref:.4f}  raw={raw_ref:.4f}  (V moved {moved*100:.1f}%)", flush=True)
        out.write(f"\n## `{name}`\n\nlearned={learned:.4f}  svdWhite={svd_ref:.4f}  "
                  f"raw={raw_ref:.4f}  V_moved={moved*100:.1f}%\n")
    out.close()
    with open("kernel_learned_metric_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
