"""Adaptive kernel ensemble: blend raw (ambient) + whitened (Mahalanobis) KRR.

Motivation: whitening wins on data-poor benchmarks (rppb: 0.54->0.79) but
loses on data-rich ones (perm, hclint). If the correlation structure is the
lever, an ensemble that picks/weights the two metrics per benchmark (by val)
should capture both and beat any single metric. Two-stage: grid each space on
a <=1200-pt subsample, then refit on full train, blend by val weight, score test.
"""
from __future__ import annotations
import datetime, json, warnings
import numpy as np, pandas as pd, polaris as po
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings("ignore")

import kernel_baseline as kb
SEED = 0
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
L_FRACS = [0.1, 0.25, 0.5, 1.0, 2.0]


def torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


def main():
    cache = torch_load(kb.CACHE)
    out = open("kernel_ensemble_results.md", "w")
    out.write(f"# Adaptive Raw+White KRR Ensemble\n\n{datetime.datetime.now()}\n\n")
    summary = {}
    for name, d in cache.items():
        if "REGRESSION" not in d["task_type"]:
            continue
        Xtr = d["X_train"].float().numpy().astype("float64")
        Xte = d["X_test"].float().numpy().astype("float64")
        ytr = np.asarray(d["y_train"], float)
        ytr_s = (ytr - ytr.mean()) / (ytr.std() + 1e-12)
        mname = d["metric"]
        hgb = kb.is_higher_better(mname)
        n = len(ytr)

        # subsample for HP selection
        rng = np.random.default_rng(SEED)
        sub = rng.choice(n, min(n, kb.SUBCAP), replace=False) if n > kb.SUBCAP else np.arange(n)
        perm = rng.permutation(len(sub))
        nv = max(8, int(0.2 * len(sub)))
        vi, si = perm[:nv], perm[nv:]

        def build_spaces(Xf):
            mu, sd = kb.standardize_fit(Xf)
            raw = kb.standardize_apply(Xf, mu, sd)
            V, s, wmu, r = kb.whiten_fit(Xf, 0.999)
            white = kb.whiten_apply(Xf, V, s, wmu)
            return raw, white, (mu, sd), (V, s, wmu, r)

        raw_sub, white_sub, raw_meta, white_meta = build_spaces(Xtr[sub])
        # select best (alpha,l) per space on val
        best = {}
        for sname, S in [("raw", raw_sub), ("white", white_sub)]:
            dmed = kb.median_pairwise_dist(S)
            bv, bkw = (-np.inf if hgb else np.inf), None
            for a in ALPHAS:
                for lf in L_FRACS:
                    try:
                        pv = kb.krr(S[si], ytr_s[sub][si], S[vi], alpha=a, l=lf * dmed)
                    except Exception:
                        continue
                    sc = kb.bench_metric(pv, ytr_s[sub][vi], mname)
                    if (hgb and sc > bv) or ((not hgb) and sc < bv):
                        bv, bkw = sc, (a, lf * dmed)
            best[sname] = (bv, bkw)

        # refit on full train, compute val weights, blend
        raw, white, raw_meta, white_meta = build_spaces(Xtr)
        # need a held-out val from full train for blending (use same indices mapped)
        # recompute a val split on full train (disjoint from test)
        perm2 = rng.permutation(n)
        nval = max(8, int(0.2 * n))
        vidx, tridx = perm2[:nval], perm2[nval:]
        preds_val = {}
        for sname, (S, meta) in {"raw": (raw, raw_meta), "white": (white, white_meta)}.items():
            a, l = best[sname][1]
            preds_val[sname] = kb.krr(S[tridx], ytr_s[tridx], S[vidx], alpha=a, l=l)
        # blend weight by val performance
        def sc(p):
            return kb.bench_metric(p, ytr_s[vidx], mname)
        sr, sw = sc(preds_val["raw"]), sc(preds_val["white"])
        # softmax-ish weight; if one is clearly better, favor it
        if hgb:
            w = np.array([max(sr, 0), max(sw, 0)]); w = w / (w.sum() + 1e-9)
        else:
            inv = np.array([1.0 / max(sr, 1e-9), 1.0 / max(sw, 1e-9)]); w = inv / inv.sum()
        # predict test with blend
        Xte_raw = kb.standardize_apply(Xte, raw_meta[0], raw_meta[1])
        Xte_white = kb.whiten_apply(Xte, white_meta[0], white_meta[1], white_meta[2])
        pred_raw = kb.krr(raw, ytr_s, Xte_raw,
                          alpha=best["raw"][1][0], l=best["raw"][1][1])
        pred_white = kb.krr(white, ytr_s, Xte_white,
                            alpha=best["white"][1][0], l=best["white"][1][1])
        blend = w[0] * pred_raw + w[1] * pred_white
        blend_raw = blend * ytr.std() + ytr.mean()

        b = po.load_benchmark(name)
        val = b.evaluate(blend_raw.astype("float64")).results
        score = val.query(f"Metric == '{mname}'")["Score"].values[0]
        summary[name] = dict(score=float(score), w_raw=round(float(w[0]), 3),
                             w_white=round(float(w[1]), 3),
                             metric=mname, n_train=n)
        print(f"{name:38s} blend={score:.4f}  w_raw={w[0]:.2f} w_white={w[1]:.2f}", flush=True)
        out.write(f"\n## `{name}`\n\nblend={score:.4f}  w_raw={w[0]:.3f}  w_white={w[1]:.3f}\n")
    out.close()
    with open("kernel_ensemble_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
