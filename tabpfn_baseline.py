"""Baseline: frozen CheMeleon embeddings -> TabPFN v2 regressor (in-context).

The direct reference point for the research idea: 'passing CheMeleon embeddings
through a tabular foundation model (TabPFN)'. Runs the SAME CheMeleon 2048-dim
embeddings (shared cache) through TabPFN and scores the same polaris metrics,
so it is directly comparable to eval_fixeddim.py on identical inputs.

Self-contained (no polaris needed): computes pearsonr / spearmanr / MAE / MSE
directly with scipy/sklearn - the same definitions polaris uses.

Usage (in the tabpfn conda env):
    python tabpfn_baseline.py [cache.pt] [leaderboard.csv]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tabpfn import TabPFNRegressor


def metric(y_true, y_pred, name):
    name = name.lower()
    if name in ("mean_absolute_error", "mae"):
        return float(mean_absolute_error(y_true, y_pred))
    if name in ("mean_squared_error", "mse"):
        return float(mean_squared_error(y_true, y_pred))
    if name in ("pearsonr", "pearson"):
        r, _ = stats.pearsonr(y_true, y_pred)
        return float(r)
    if name in ("spearmanr", "spearman"):
        r, _ = stats.spearmanr(y_true, y_pred)
        return float(r)
    if name in ("r2",):
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"unknown metric {name}")


def main() -> None:
    cache_path = sys.argv[1] if len(sys.argv) > 1 else "data/bench_embeddings.pt"
    leaderboard_path = sys.argv[2] if len(sys.argv) > 2 else \
        "polaris_benchmarks_leaderboard_20260709_150240.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    lb = pd.read_csv(leaderboard_path) if Path(leaderboard_path).exists() else None
    out = open("tabpfn_baseline_results.md", "w")
    perf, ranks = {}, []

    for name, d in cache.items():
        if "REGRESSION" not in d["task_type"]:
            print(f"Skipping {name} (not regression)")
            continue
        X_tr = d["X_train"].numpy()
        y_tr = np.asarray(d["y_train"], dtype=np.float64)
        y_te = np.asarray(d["y_test"], dtype=np.float64)
        X_te = d["X_test"].numpy()
        mname = d["metric"]

        reg = TabPFNRegressor(device=device)
        preds = np.asarray(reg.fit(X_tr, y_tr).predict(X_te), dtype=np.float64)
        val = metric(y_te, preds, mname)

        out.write(f"\n## `{name}`  ({mname})\n\nTabPFN-CheMeleon = {val:.4f}\n")
        if lb is not None:
            subset = lb[lb["Benchmark_ID"] == name].copy()
            lbc = pd.concat([subset, pd.DataFrame([{"Name": "TabPFN-CheMeleon",
                                                    mname: val}])],
                            ignore_index=True).sort_values(
                by=mname,
                ascending=mname in {"mean_absolute_error", "mean_squared_error"}
            ).reset_index(drop=True)
            if lbc.shape[0] >= 5:
                ranks.append(lbc.query("Name == 'TabPFN-CheMeleon'").index[0] + 1)
            out.write(lbc[["Name", mname]].to_markdown(index=False) + "\n")

        perf[name] = {mname: float(val)}
        print(f"{name}: {mname} = {val:.4f}  (train={d['n_train']} test={d['n_test']})")

    out.write(f"\n# Summary\n\nAverage rank ({len(ranks)}): "
              f"{(sum(ranks) / len(ranks) if ranks else float('nan')):.2f}\n\n")
    out.write("results_dict = " + json.dumps(perf, indent=2) + "\n")
    out.close()
    print(f"\nTabPFN baseline done. Avg rank = "
          f"{(sum(ranks) / len(ranks) if ranks else float('nan')):.2f}")


if __name__ == "__main__":
    main()
