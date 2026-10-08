"""Kernel / GP 'inference-time global attention' over CheMeleon embeddings.

Thesis (answering the user's reframe):
  The from-scratch transformer tried to LEARN "attention over all the data"
  (in-context regression) and collapsed to the mean predictor. But kernel
  regression / Gaussian Processes give you *global attention over all your
  data, by construction, with zero training*: the prediction for a query is a
  soft-weighted average of EVERY training point, weights = kernel similarity.
  A trained wide network's predictions ARE kernel regression (NTK), so this is
  the right thing the transformer was supposed to approximate - but analytic,
  so it cannot collapse.

  Where to exploit CheMeleon's correlated 2048-dim columns: the KERNEL METRIC.
  Naive Euclidean on standardized features drowns in ambient dimension (all
  pairwise distances ~ sqrt(2*2048)). Rank-aware WHITENING (SVD, drop noise
  components, decorrelate) reduces to the effective rank and makes
  Euclidean-in-whitened == Mahalanobis-in-original. That is the ablation.

Models (all closed-form, no learned params): kNN, Ridge, KRR(RBF), GP(RBF).
Spaces: raw (standardized) vs whitened (rank-aware).
HP selection: two-stage - grid on a <=1200-point train subsample, refit the
winner on the FULL train set, score the untouched test set with polaris.

Run:  ~/miniforge3/envs/chemeleon/bin/python kernel_baseline.py
"""
from __future__ import annotations

import datetime
import json
import warnings

import numpy as np
import pandas as pd
import polaris as po
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

CACHE = "data/bench_embeddings.pt"
SEED = 0
SUBCAP = 1200                    # max train pts for HP selection
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]      # KRR/GP reg (std-y space)
L_FRACS = [0.1, 0.25, 0.5, 1.0, 2.0]               # RBF length / median dist
KNN_KS = [5, 10, 20, 50, 100]
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
RANK_FRACS = [0.999]                            # whitening variance cutoff
MODELS = ["KRR", "GP", "kNN", "Ridge"]


# ---------- kernels / spaces ----------
def rbf(A, B, l):
    A2 = (A ** 2).sum(1)[:, None]
    B2 = (B ** 2).sum(1)[None, :]
    d2 = np.clip(A2 + B2 - 2.0 * (A @ B.T), 0.0, None)
    return np.exp(-d2 / (2.0 * l * l))


def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0).clip(min=1e-8)
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def whiten_fit(X, rank_frac):
    mu = X.mean(0)
    Xc = X - mu
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    cum = np.cumsum(var) / max(var.sum(), 1e-12)
    r = int(np.searchsorted(cum, rank_frac) + 1)
    r = max(1, min(r, len(s)))
    V = Vt[:r].T
    return V, s[:r], mu, r


def whiten_apply(X, V, s, mu):
    return (X - mu) @ V / s


def median_pairwise_dist(X, cap=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(n, cap), replace=False)
    Xs = X[idx]
    A2 = (Xs ** 2).sum(1)[:, None]
    B2 = (Xs ** 2).sum(1)[None, :]
    d2 = np.clip(A2 + B2 - 2.0 * (Xs @ Xs.T), 0.0, None)
    iu = np.triu_indices(len(Xs), k=1)
    return float(np.median(np.sqrt(d2[iu])))


# ---------- models (fit on std-y, predict std-y) ----------
def krr(Xtr, ytr, Xte, alpha, l):
    K = rbf(Xtr, Xtr, l) + (alpha + 1e-10) * np.eye(len(Xtr))
    beta = np.linalg.solve(K, ytr)
    return rbf(Xte, Xtr, l) @ beta


def gp(Xtr, ytr, Xte, alpha):
    K = rbf(Xtr, Xtr, alpha) + (1.0 + 1e-10) * np.eye(len(Xtr))
    L = np.linalg.cholesky(K)
    v = np.linalg.solve(L.T, np.linalg.solve(L, ytr))
    return rbf(Xte, Xtr, alpha) @ v


def knn(Xtr, ytr, Xte, k):
    A2 = (Xtr ** 2).sum(1)[None, :]
    B2 = (Xte ** 2).sum(1)[:, None]
    d = np.sqrt(np.clip(A2 + B2 - 2.0 * (Xte @ Xtr.T), 0.0, None))
    kk = min(k, len(Xtr))
    nn = np.argpartition(d, kk, axis=1)[:, :kk]
    w = 1.0 / (d[np.arange(len(d))[:, None], nn] + 1e-6)
    w /= w.sum(1, keepdims=True)
    return (w * ytr[nn]).sum(1)


def ridge(Xtr, ytr, Xte, alpha):
    A = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    beta = np.linalg.solve(A, Xtr.T @ ytr)
    return Xte @ beta


def predict(model, Xtr, ytr, Xte, **kw):
    if model == "KRR":
        return krr(Xtr, ytr, Xte, alpha=kw["alpha"], l=kw["l"])
    if model == "GP":
        return gp(Xtr, ytr, Xte, alpha=kw["l"])
    if model == "kNN":
        return knn(Xtr, ytr, Xte, k=kw["k"])
    if model == "Ridge":
        return ridge(Xtr, ytr, Xte, alpha=kw["alpha"])
    raise ValueError(model)


def hp_candidates(model, dmed):
    if model in ("KRR", "GP"):
        out = []
        for a in ALPHAS:
            for lf in L_FRACS:
                out.append(dict(alpha=a, l=lf * dmed))
        return out
    if model == "kNN":
        return [dict(k=k) for k in KNN_KS]
    return [dict(alpha=a) for a in RIDGE_ALPHAS]


# ---------- metrics ----------
def bench_metric(pred, true, name):
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    if name == "pearsonr":
        return float(pearsonr(pred, true)[0])
    if name == "spearmanr":
        return float(spearmanr(pred, true)[0])
    if name == "mean_squared_error":
        return float(np.mean((pred - true) ** 2))
    if name == "mean_absolute_error":
        return float(np.mean(np.abs(pred - true)))
    return float(np.mean((pred - true) ** 2))


def is_higher_better(name):
    return name in {"pearsonr", "spearmanr"}


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


def main():
    cache = torch_load(CACHE)
    lb = None
    out = open("kernel_baseline_results.md", "w")
    out.write(f"# Kernel / GP 'Inference-Time Global Attention' over CheMeleon\n\n"
              f"timestamp: {datetime.datetime.now()}\n\n"
              f"Closed-form kernel methods (no learned params, cannot collapse).\n"
              f"Ablation: raw standardized-Euclidean vs rank-aware WHITENED (Mahalanobis) metric.\n"
              f"HP selected on a <=1200-pt train subsample; refit on full train; scored on test.\n\n")

    summary = {}
    for name, d in cache.items():
        if "REGRESSION" not in d["task_type"]:
            continue
        Xtr_full = d["X_train"].float().numpy().astype("float64")
        Xte = d["X_test"].float().numpy().astype("float64")
        ytr = np.asarray(d["y_train"], float)
        ytr_s = (ytr - ytr.mean()) / (ytr.std() + 1e-12)
        mname = d["metric"]
        hgb = is_higher_better(mname)
        n = len(ytr)

        # ---- HP selection on a subsample ----
        rng = np.random.default_rng(SEED)
        if n > SUBCAP:
            sub_idx = rng.choice(n, SUBCAP, replace=False)
        else:
            sub_idx = np.arange(n)
        Xtr_sub, ytr_s_sub = Xtr_full[sub_idx], ytr_s[sub_idx]
        perm = rng.permutation(len(sub_idx))
        n_val = max(8, int(0.2 * len(sub_idx)))
        val_idx, sel_idx = perm[:n_val], perm[n_val:]

        spaces_sub = {}
        # raw
        mu, sd = standardize_fit(Xtr_sub)
        spaces_sub["raw"] = (standardize_apply(Xtr_sub, mu, sd), (mu, sd))
        # whitened
        V, s, wmu, r = whiten_fit(Xtr_sub, 0.999)
        spaces_sub[f"white"] = (whiten_apply(Xtr_sub, V, s, wmu), (V, s, wmu, r))

        best = {}
        for sname, (S_sub, meta) in spaces_sub.items():
            dmed = median_pairwise_dist(S_sub)
            for model in MODELS:
                bv = -np.inf if hgb else np.inf
                bkw = None
                for kw in hp_candidates(model, dmed):
                    try:
                        pv = predict(model, S_sub[sel_idx], ytr_s_sub[sel_idx],
                                     S_sub[val_idx], **kw)
                    except Exception:
                        continue
                    sc = bench_metric(pv, ytr_s_sub[val_idx], mname)
                    if (hgb and sc > bv) or ((not hgb) and sc < bv):
                        bv, bkw = sc, kw
                best[(model, sname)] = (bv, bkw)

        # ---- refit winners on FULL train, predict test, score ----
        b = po.load_benchmark(name)
        mu, sd = standardize_fit(Xtr_full)
        V, s, wmu, r = whiten_fit(Xtr_full, 0.999)
        full_spaces = {
            "raw": (standardize_apply(Xtr_full, mu, sd), standardize_apply(Xte, mu, sd)),
            "white": (whiten_apply(Xtr_full, V, s, wmu), whiten_apply(Xte, V, s, wmu)),
        }
        row = {"bench": name, "metric": mname, "n_train": n, "n_test": len(Xte),
               "rank_kept": int(r)}
        for (model, sname), (val_score, kw) in best.items():
            if not kw:
                continue
            S_tr, S_te = full_spaces[sname]
            pred = predict(model, S_tr, ytr_s, S_te, **kw)
            pred_raw = pred * ytr.std() + ytr.mean()
            val = b.evaluate(pred_raw.astype("float64")).results
            score = val.query(f"Metric == '{mname}'")["Score"].values[0]
            row[f"{model}/{sname}"] = float(score)
            row[f"{model}/{sname}_hp"] = kw

        # mean-predictor baseline (the collapse the transformer did)
        mean_pred = np.full(len(Xte), ytr.mean())
        mval = b.evaluate(mean_pred.astype("float64")).results
        row["mean/mean"] = float(mval.query(f"Metric == '{mname}'")["Score"].values[0])

        # leaderboard rank
        META = {"bench", "metric", "n_train", "n_test", "rank_kept"}
        score_keys = [k for k, v in list(row.items())
                      if isinstance(v, float) and k not in META and not k.endswith("_hp")]
        try:
            lb = lb if lb is not None else pd.read_csv(
                "polaris_benchmarks_leaderboard_20260709_150240.csv")
            subset = lb[lb["Benchmark_ID"] == name].copy()
            for k in score_keys:
                v = row[k]
                lbc = pd.concat([subset, pd.DataFrame([{"Name": k, mname: v}])],
                                ignore_index=True)
                lbc[mname] = pd.to_numeric(lbc[mname], errors="coerce")
                lbc = lbc.dropna(subset=[mname]).sort_values(
                    by=mname, ascending=not hgb).reset_index(drop=True)
                if lbc.shape[0] >= 5:
                    row[k + "_rank"] = int(lbc.query(f"Name == '{k}'").index[0] + 1)
        except Exception as e:
            print("  (leaderboard rank skipped:", e, ")")

        summary[name] = row
        print(f"\n### {name}  (n_tr={n}, rank_kept={r}, {mname})", flush=True)
        for k in score_keys:
            print(f"  {k:16s} {row[k]:.4f}", flush=True)

        out.write(f"\n## `{name}`\n\n")
        out.write(pd.DataFrame([row]).T.to_markdown())
        out.write("\n")

    # ---- aggregate ----
    META = {"bench", "metric", "n_train", "n_test", "rank_kept"}
    model_names = sorted({k for r in summary.values() for k, v in r.items()
                          if isinstance(v, float) and k not in META
                          and not k.endswith(("_hp", "_rank"))})
    out.write("\n# Summary (all configs, lower is better for error metrics)\n\n")
    out.write("| benchmark | metric | " + " | ".join(model_names) + " |\n")
    out.write("|---|---|" + "---:|" * len(model_names) + "\n")
    for name, r in summary.items():
        cells = [f"{r.get(m, float('nan')):.3f}" for m in model_names]
        out.write(f"| {name} | {r['metric']} | " + " | ".join(cells) + " |\n")

    out.write("\n## Average rank (lower better)\n\n")
    rank_rows = {}
    for m in model_names:
        ranks = [r[m + "_rank"] for r in summary.values() if m + "_rank" in r]
        if ranks:
            rank_rows[m] = sum(ranks) / len(ranks)
    out.write(pd.DataFrame(rank_rows, index=["avg_rank"]).T
              .sort_values("avg_rank").to_markdown(floatfmt=".2f"))
    out.close()

    with open("kernel_baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n\nAVERAGE RANK:")
    for m, r in sorted(rank_rows.items(), key=lambda x: x[1]):
        print(f"  {m:16s} {r:.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
