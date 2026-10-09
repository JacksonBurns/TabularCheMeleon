"""GP / kernel exploration over BOTH feature spaces.

Extends kernel_baseline.py (RBF + KRR on CheMeleon raw/white) with:

  (c) HETEROSCEDASTIC noise - input-dependent sigma(x), fit by iteratively
      reweighted least squares (IRWLS) with a linear-in-features log-noise model.
  (d) ALTERNATIVE KERNELS - Matern nu=5/2, Spectral Mixture Kernel (SMK, fit by
      GP marginal MLE), and structured fingerprint kernels (Tanimoto on the
      count-FP block, and Tanimoto(FP) x RBF(desc) product).

Feature spaces (head-to-head):
  - "chemeleon" : frozen CheMeleon 2048-dim GNN embedding (raw + SVD-white)
  - "descriptor": 2048-dim radius-3 Morgan COUNT fingerprint + 325 RDKit
                  descriptors = 2373-dim classical descriptor embedding

All GP models are inductive (test points never see each other), closed-form,
and cannot collapse to the mean predictor.

Two-stage HP selection (matches kernel_baseline.py): grid on a <=1200-pt train
subsample (20% val), pick the winner, refit on FULL train, score untouched test
with polaris.

Run:  ~/miniforge3/envs/chemeleon/bin/python gp_explore.py
"""
from __future__ import annotations

import datetime
import json
import os
import warnings

import numpy as np
import pandas as pd
import polaris as po
from scipy import optimize
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

CACHE_CHEM = "data/bench_embeddings.pt"
CACHE_DESC = "data/bench_descriptor_embeddings.pt"
SEED = 0
SUBCAP = 1200                       # max train pts for HP selection
L_FRACS = [0.1, 0.25, 0.5, 1.0, 2.0]        # signal lengthscale / median dist
NOISES = [1e-3, 1e-2, 1e-1, 1.0]            # homoscedastic noise (std-y space)
SMK_M = 3                                 # spectral mixture components
HETO_ITERS = 3                            # IRWLS iterations for heteroscedastic

# Which kernel to run in which space
CHEM_KERNELS = ["rbf", "matern52", "smk", "rbf_heto"]
DESC_KERNELS = ["rbf", "matern52", "smk", "tanimoto_fp",
                "tanimoto_x_rbf", "rbf_heto"]


# ---------------- spaces ----------------
def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0).clip(min=1e-8)
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def whiten_fit(X, rank_frac=0.999):
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


def sq_dist(A, B):
    A2 = (A ** 2).sum(1)[:, None]
    B2 = (B ** 2).sum(1)[None, :]
    return np.clip(A2 + B2 - 2.0 * (A @ B.T), 0.0, None)


# ---------------- kernels (all return K(A,B), symmetric PSD) ----------------
def k_rbf(A, B, l):
    return np.exp(-sq_dist(A, B) / (2.0 * l * l))


def k_matern52(A, B, l):
    # Matern nu=5/2: (1 + r*sqrt(3) + 3r^2/2) * exp(-r*sqrt(3))
    r = np.sqrt(sq_dist(A, B))
    sr3 = r * np.sqrt(3.0)
    return (1.0 + sr3 + 0.5 * sr3 * sr3) * np.exp(-sr3) / (l ** 0)  # l absorbed
    # NOTE: we pass l already scaled so the kernel is in units of r; see caller.


def k_matern52_scaled(A, B, l):
    r = np.sqrt(sq_dist(A, B))
    sr3 = r * np.sqrt(3.0) / l
    return (1.0 + sr3 + 0.5 * sr3 * sr3) * np.exp(-sr3)


def k_tanimoto_fp(A, B):
    # Tanimoto (Jaccard) on count-FP block. A,B are FP-only matrices (n,2048).
    ab = A @ B.T
    aa = (A ** 2).sum(1)[:, None]
    bb = (B ** 2).sum(1)[None, :]
    denom = aa + bb - ab
    denom[denom == 0] = 1e-12
    return ab / denom


def k_smk(A, B, amps, widths):
    d2 = sq_dist(A, B)
    K = np.zeros(d2.shape)
    for a, w in zip(amps, widths):
        K += a * np.exp(-w * d2)
    return K


def _smk_nll(params, X, y, M):
    la = params[:M]
    lw = params[M:2 * M]
    log_noise = params[2 * M]
    log_var = params[2 * M + 1]
    amps = np.exp(la)
    widths = np.exp(lw)
    noise = np.exp(log_noise)
    K = k_smk(X, X, amps, widths) * np.exp(log_var) + noise * np.eye(len(X))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return (0.5 * y @ alpha + np.sum(np.log(np.diag(L)))
            + 0.5 * len(y) * np.log(2 * np.pi))


def fit_smk(X, y, M=3, seed=0):
    """Fit SMK hyperparams by GP marginal MLE. Returns (amps, widths, noise, var)."""
    rng = np.random.default_rng(seed)
    dmed = median_pairwise_dist(X)
    # init: M components spread in width, amps ~ 1/M
    base_w = 1.0 / (2.0 * dmed * dmed)
    lw0 = np.log(np.array([base_w * f for f in [0.1, 1.0, 10.0][:M]]))
    la0 = np.full(M, np.log(1.0 / M))
    x0 = np.concatenate([la0, lw0, [0.0, 0.0]])
    res = optimize.minimize(_smk_nll, x0, args=(X, y, M), method="L-BFGS-B",
                            options={"maxiter": 200, "gtol": 1e-5})
    p = res.x
    amps = np.exp(p[:M])
    widths = np.exp(p[M:2 * M])
    noise = float(np.exp(p[2 * M]))
    var = float(np.exp(p[2 * M + 1]))
    return {"amps": amps, "widths": widths, "noise": noise, "var": var}


# ---------------- GP predictors (fit on std-y, predict std-y) ----------------
def gp_solve(K, y, noise_diag):
    """alpha = (K + noise_diag)^{-1} y ; returns alpha. noise_diag: scalar or (n,)."""
    A = K.copy()
    if np.isscalar(noise_diag):
        A[np.diag_indices_from(A)] += noise_diag
    else:
        A[np.diag_indices_from(A)] += noise_diag
    try:
        L = np.linalg.cholesky(A)
        return np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        A += 1e-6 * np.eye(len(A))
        return np.linalg.solve(A, y)


def predict_homo(kernel, Xtr, ytr, Xte, l=None, noise=None, smk=None):
    K = _kernel_matrix(kernel, Xtr, Xtr, l, smk)
    Ks = _kernel_matrix(kernel, Xte, Xtr, l, smk)
    alpha = gp_solve(K, ytr, noise if noise is not None else 1e-2)
    return Ks @ alpha


def _kernel_matrix(kernel, A, B, l=None, smk=None):
    if kernel == "rbf":
        return k_rbf(A, B, l)
    if kernel == "matern52":
        return k_matern52_scaled(A, B, l)
    if kernel == "smk":
        return k_smk(A, B, smk["amps"], smk["widths"]) * smk["var"]
    if kernel == "tanimoto_fp":
        return k_tanimoto_fp(A, B)
    if kernel == "tanimoto_x_rbf":
        # A,B are full descriptor vectors; split into FP block and desc block
        fp = A[:, :2048]; dp = A[:, 2048:]
        fpb = B[:, :2048]; dpb = B[:, 2048:]
        return k_tanimoto_fp(fp, fpb) * k_rbf(dp, dpb, l)
    raise ValueError(kernel)


def predict_heto(kernel, Xtr, ytr, Xte, l, noise, M=3, smk=None, iters=3):
    """Heteroscedastic GP with linear-in-features log-noise, fit by IRWLS."""
    # start homoscedastic
    sigma2 = np.full(len(ytr), float(noise))
    for _ in range(iters):
        K = _kernel_matrix(kernel, Xtr, Xtr, l, smk)
        alpha = gp_solve(K, ytr, sigma2)
        train_pred = K @ alpha
        resid2 = np.maximum((ytr - train_pred) ** 2, 1e-10)
        # fit log sigma2 = w^T x + b by ridge on features
        Z = np.column_stack([Xtr, np.ones(len(Xtr))])
        lam = 1e-2 * Z.shape[1]
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ np.log(resid2))
        sigma2 = np.exp(Z @ w)
        sigma2 = np.clip(sigma2, 1e-6, None)
    K = _kernel_matrix(kernel, Xtr, Xtr, l, smk)
    alpha = gp_solve(K, ytr, sigma2)
    Ks = _kernel_matrix(kernel, Xte, Xtr, l, smk)
    return Ks @ alpha, sigma2


# ---------------- metrics ----------------
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


# ---------------- HP selection ----------------
def select_hp(space_name, kernels, Xtr_sub, ytr_s_sub, sel, val, mname, hgb):
    """Return {(kernel): (best_val_score, kw)} over the subsample."""
    best = {}
    for kernel in kernels:
        bv = -np.inf if hgb else np.inf
        bkw = None
        if kernel == "smk":
            try:
                smk = fit_smk(Xtr_sub[sel], ytr_s_sub[sel], M=SMK_M, seed=SEED)
                pv = predict_homo("smk", Xtr_sub[sel], ytr_s_sub[sel],
                                  Xtr_sub[val], smk=dict(smk))
                sc = bench_metric(pv, ytr_s_sub[val], mname)
                if (hgb and sc > bv) or ((not hgb) and sc < bv):
                    bv, bkw = sc, dict(smk=dict(smk))
            except Exception:
                pass
        elif kernel == "tanimoto_fp":
            # no lengthscale; just noise grid
            for noise in NOISES:
                try:
                    pv = predict_homo("tanimoto_fp", Xtr_sub[sel], ytr_s_sub[sel],
                                      Xtr_sub[val], noise=noise)
                    sc = bench_metric(pv, ytr_s_sub[val], mname)
                    if (hgb and sc > bv) or ((not hgb) and sc < bv):
                        bv, bkw = sc, dict(noise=noise)
                except Exception:
                    pass
        elif kernel == "tanimoto_x_rbf":
            for l in L_FRACS:
                for noise in NOISES:
                    try:
                        # need a length for the desc RBF block
                        desc = Xtr_sub[sel][:, 2048:]
                        ld = l * median_pairwise_dist(desc)
                        pv = predict_homo("tanimoto_x_rbf", Xtr_sub[sel],
                                          ytr_s_sub[sel], Xtr_sub[val],
                                          l=ld, noise=noise)
                        sc = bench_metric(pv, ytr_s_sub[val], mname)
                        if (hgb and sc > bv) or ((not hgb) and sc < bv):
                            bv, bkw = sc, dict(l=ld, noise=noise)
                    except Exception:
                        pass
        else:
            # rbf, matern52, rbf_heto : grid lengthscale x noise
            dmed = median_pairwise_dist(Xtr_sub)
            for l in L_FRACS:
                for noise in NOISES:
                    ll = l * dmed
                    try:
                        if kernel == "rbf_heto":
                            pv, _ = predict_heto("rbf", Xtr_sub[sel], ytr_s_sub[sel],
                                                 Xtr_sub[val], l=ll, noise=noise,
                                                 iters=HETO_ITERS)
                        else:
                            pv = predict_homo(kernel, Xtr_sub[sel], ytr_s_sub[sel],
                                              Xtr_sub[val], l=ll, noise=noise)
                        sc = bench_metric(pv, ytr_s_sub[val], mname)
                        if (hgb and sc > bv) or ((not hgb) and sc < bv):
                            bv, bkw = sc, dict(l=ll, noise=noise, kernel=kernel)
                    except Exception:
                        pass
        best[kernel] = (bv, bkw)
    return best


def run_final(space_name, kernels, Xtr, ytr_s, Xte, mname, best):
    """Refit winners on FULL train, predict test, score with polaris benchmark."""
    row = {}
    for kernel, (val_score, kw) in best.items():
        if not kw:
            continue
        try:
            if kernel == "smk":
                pred = predict_homo("smk", Xtr, ytr_s, Xte, smk=kw["smk"])
            elif kernel == "tanimoto_fp":
                pred = predict_homo("tanimoto_fp", Xtr, ytr_s, Xte,
                                    noise=kw["noise"])
            elif kernel == "tanimoto_x_rbf":
                pred = predict_homo("tanimoto_x_rbf", Xtr, ytr_s, Xte,
                                    l=kw["l"], noise=kw["noise"])
            elif kernel == "rbf_heto":
                pred, sigma2 = predict_heto("rbf", Xtr, ytr_s, Xte,
                                            l=kw["l"], noise=kw["noise"],
                                            iters=HETO_ITERS)
                row[kernel + "_hetoscale"] = float(np.exp(np.mean(np.log(sigma2))))
            else:
                pred = predict_homo(kernel, Xtr, ytr_s, Xte,
                                    l=kw["l"], noise=kw["noise"])
            row[kernel] = pred  # store std-y pred; destandardize later
            row[kernel + "_val"] = float(val_score)
        except Exception as e:
            row[kernel + "_err"] = str(e)
    return row


def main():
    chem = torch_load(CACHE_CHEM)
    desc = torch_load(CACHE_DESC)
    out = open("gp_explore_results.md", "w")
    out.write(f"# GP / Kernel Exploration: CheMeleon vs Descriptor Embeddings\n\n"
              f"timestamp: {datetime.datetime.now()}\n\n"
              f"Inductive GP (test points independent). Kernels: RBF, Matern-5/2,\n"
              f"Spectral Mixture (MLE), Tanimoto(FP), Tanimoto(FP)xRBF(desc),\n"
              f"Heteroscedastic RBF (IRWLS linear log-noise).\n"
              f"Spaces: CheMeleon 2048 (raw+white) vs descriptor 2373 (Morgan count + RDKit).\n"
              f"HP: grid on <=1200-pt subsample (20% val), refit full train, score test.\n\n")

    summary = {}
    # resume: skip benchmarks already completed in a prior (killed) run
    if os.path.exists("gp_explore_summary.json"):
        try:
            with open("gp_explore_summary.json") as f:
                summary = json.load(f)
            if summary:
                print(f"Resuming: {len(summary)} benchmarks already done.", flush=True)
        except Exception:
            summary = {}
    all_names = list(chem.keys())
    for name in all_names:
        if name in summary:
            print(f"\n### {name}  (cached, skipping)", flush=True)
            continue
        d = chem[name]
        Xtr_c = d["X_train"].float().numpy().astype("float64")
        Xte_c = d["X_test"].float().numpy().astype("float64")
        ytr = np.asarray(d["y_train"], float)
        mname = d["metric"]
        hgb = is_higher_better(mname)
        n = len(ytr)
        ytr_s = (ytr - ytr.mean()) / (ytr.std() + 1e-12)

        Xtr_d = desc[name]["X_train"].float().numpy().astype("float64")
        Xte_d = desc[name]["X_test"].float().numpy().astype("float64")

        # subsample for HP
        rng = np.random.default_rng(SEED)
        sub_idx = rng.choice(n, SUBCAP, replace=False) if n > SUBCAP else np.arange(n)
        perm = rng.permutation(len(sub_idx))
        n_val = max(8, int(0.2 * len(sub_idx)))
        val_idx, sel_idx = perm[:n_val], perm[n_val:]

        row = {"bench": name, "metric": mname, "n_train": n, "n_test": d["n_test"]}

        b = po.load_benchmark(name)
        destd = lambda p: p * ytr.std() + ytr.mean()

        # ---- CHEMELEON space ----
        mu, sd = standardize_fit(Xtr_c[sub_idx])
        V, s, wmu, r = whiten_fit(Xtr_c[sub_idx])
        chem_sub = {
            "raw": standardize_apply(Xtr_c[sub_idx], mu, sd),
            "white": whiten_apply(Xtr_c[sub_idx], V, s, wmu),
        }
        for sname, Ssub in chem_sub.items():
            best = select_hp(f"chem/{sname}", CHEM_KERNELS, Ssub, ytr_s[sub_idx],
                             sel_idx, val_idx, mname, hgb)
            # full spaces
            mu_f, sd_f = standardize_fit(Xtr_c)
            V_f, s_f, wmu_f, r_f = whiten_fit(Xtr_c)
            Sfull = (standardize_apply(Xtr_c, mu_f, sd_f) if sname == "raw"
                     else whiten_apply(Xtr_c, V_f, s_f, wmu_f))
            Ste = (standardize_apply(Xte_c, mu_f, sd_f) if sname == "raw"
                   else whiten_apply(Xte_c, V_f, s_f, wmu_f))
            finals = run_final(f"chem/{sname}", CHEM_KERNELS, Sfull, ytr_s, Ste,
                               mname, best)
            for k, v in finals.items():
                if isinstance(v, np.ndarray):
                    pv = destd(v)
                    val = b.evaluate(pv.astype("float64")).results
                    row[f"chem/{sname}/{k}"] = float(
                        val.query(f"Metric == '{mname}'")["Score"].values[0])
                else:
                    row[f"chem/{sname}/{k}"] = v
            row[f"chem/{sname}/rank_kept"] = int(r_f)

        # ---- DESCRIPTOR space ----
        # standardize the whole 2373 vector for RBF/Matern/SMK/heto
        mu_d, sd_d = standardize_fit(Xtr_d[sub_idx])
        desc_sub = standardize_apply(Xtr_d[sub_idx], mu_d, sd_d)
        best_d = select_hp("desc", DESC_KERNELS, desc_sub, ytr_s[sub_idx],
                           sel_idx, val_idx, mname, hgb)
        mu_d_f, sd_d_f = standardize_fit(Xtr_d)
        Sfull_d = standardize_apply(Xtr_d, mu_d_f, sd_d_f)
        Ste_d = standardize_apply(Xte_d, mu_d_f, sd_d_f)
        finals_d = run_final("desc", DESC_KERNELS, Sfull_d, ytr_s, Ste_d, mname, best_d)
        for k, v in finals_d.items():
            if isinstance(v, np.ndarray):
                pv = destd(v)
                val = b.evaluate(pv.astype("float64")).results
                row[f"desc/{k}"] = float(
                    val.query(f"Metric == '{mname}'")["Score"].values[0])
            else:
                row[f"desc/{k}"] = v

        summary[name] = row
        # incremental checkpoint so a kill doesn't lose progress
        with open("gp_explore_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n### {name}  (n_tr={n}, {mname})", flush=True)
        score_keys = [k for k, v in row.items()
                      if isinstance(v, float) and not k.endswith(
                          ("_val", "_rank_kept", "_hetoscale", "_err"))]
        for k in score_keys:
            print(f"  {k:34s} {row[k]:.4f}", flush=True)

        out.write(f"\n## `{name}`  (n_tr={n}, {mname})\n\n")
        out.write(pd.DataFrame([row]).T.to_markdown())
        out.write("\n")

    # ---- aggregate table ----
    META = {"bench", "metric", "n_train", "n_test"}
    model_names = sorted({k for r in summary.values() for k, v in r.items()
                          if isinstance(v, float) and k not in META
                          and not k.endswith(("_val", "_rank_kept", "_hetoscale",
                                              "_err"))})
    out.write("\n# Summary (per-benchmark score; direction depends on metric)\n\n")
    out.write("| benchmark | " + " | ".join(model_names) + " |\n")
    out.write("|---|" + "---:|" * len(model_names) + "\n")
    for name, r in summary.items():
        cells = [f"{r.get(m, float('nan')):.3f}" for m in model_names]
        out.write(f"| {name} | " + " | ".join(cells) + " |\n")

    # direction-aware average: for higher-better metrics average the score,
    # for lower-better average the negative (so one number = "better is larger")
    out.write("\n# Normalized average (higher = better across all metrics)\n\n")
    norm = {}
    for m in model_names:
        vals = []
        for r in summary.values():
            if m in r and isinstance(r[m], float) and not np.isnan(r[m]):
                vals.append(r[m] if is_higher_better(r["metric"]) else -r[m])
        if vals:
            norm[m] = float(np.mean(vals))
    out.write(pd.DataFrame(norm, index=["normalized_avg"]).T
              .sort_values("normalized_avg", ascending=False)
              .to_markdown(floatfmt=".3f"))
    out.close()

    with open("gp_explore_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n\nNORMALIZED AVG (higher=better):")
    for m, v in sorted(norm.items(), key=lambda x: -x[1]):
        print(f"  {m:34s} {v:.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
