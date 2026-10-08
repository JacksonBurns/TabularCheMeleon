"""Reference baseline: frozen CheMeleon embeddings -> TabICL v2 (in-context).

The open tabular FM closest to TabPFN, and the direct reference for the
user's premise ('passing CheMeleon embeddings through a tabular foundation
model'). Embeds benchmark SMILES with frozen CheMeleon (full 2048-dim) and
runs TabICL v2 in-context regression, scoring with polaris - the identical
pipeline to eval_fixeddim.py, so results are directly comparable.

Run in the chemeleon env with tabicl on PYTHONPATH:
    PYTHONPATH=/home/jackson/tabicl-chemeleon/src \
      python tabicl_baseline.py [leaderboard.csv]
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import torch
import pandas as pd
import polaris as po
from polaris.utils.types import TargetType

from tabicl import TabICLRegressor
from featurize import CheMeleonEmbedder

BENCHMARKS = (
    "polaris/pkis2-ret-wt-reg-v2", "polaris/pkis2-kit-wt-reg-v2",
    "polaris/pkis2-egfr-wt-reg-v2", "polaris/adme-fang-solu-1",
    "polaris/adme-fang-rppb-1", "polaris/adme-fang-hppb-1",
    "polaris/adme-fang-perm-1", "polaris/adme-fang-rclint-1",
    "polaris/adme-fang-hclint-1", "tdcommons/lipophilicity-astrazeneca",
    "tdcommons/ppbr-az", "tdcommons/clearance-hepatocyte-az",
    "tdcommons/half-life-obach", "tdcommons/clearance-microsome-az",
    "tdcommons/vdss-lombardo", "tdcommons/caco2-wang", "tdcommons/ld50-zhu",
)


def main() -> None:
    leaderboard_path = sys.argv[1] if len(sys.argv) > 1 else \
        "polaris_benchmarks_leaderboard_20260709_150240.csv"
    ckpt = "/home/jackson/tabicl-chemeleon/tabicl-regressor-v2-20260212.ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    get_emb = CheMeleonEmbedder(device=device, type="full")
    lb = pd.read_csv(leaderboard_path)
    out = open("tabicl_baseline_results.md", "w")
    out.write(f"# TabICL-v2 (CheMeleon embeddings) Baseline\n\ntimestamp: {datetime.datetime.now()}\n\n")
    perf, ranks = {}, []

    for name in BENCHMARKS:
        b = po.load_benchmark(name)
        s_col, t_col = list(b.input_cols)[0], list(b.target_cols)[0]
        train, test = b.get_train_test_split()
        tdf, tdf2 = train.as_dataframe(), test.as_dataframe()
        if b.target_types[t_col] != TargetType.REGRESSION:
            print(f"Skipping {name} (not regression)")
            continue

        X_tr = get_emb(tdf[s_col].to_list()).cpu().numpy().astype("float32")
        y_tr = tdf[t_col].to_numpy().astype("float32")
        X_te = get_emb(tdf2[s_col].to_list()).cpu().numpy().astype("float32")
        torch.cuda.empty_cache()

        reg = TabICLRegressor(model_path=ckpt, allow_auto_download=False, device=device,
                              n_estimators=8, use_amp=True)
        preds = reg.fit(X_tr, y_tr).predict(X_te)

        results = b.evaluate(preds.astype("float64")).results
        val = results.query(f"Metric == '{b.main_metric.label}'")["Score"].values[0]

        subset = lb[lb["Benchmark_ID"] == name].copy()
        lbc = pd.concat([subset, pd.DataFrame([{"Name": "TabICL-CheMeleon",
                                                b.main_metric.label: val}])],
                        ignore_index=True).sort_values(
            by=b.main_metric.label,
            ascending=b.main_metric.label in {"mean_absolute_error", "mean_squared_error"}
        ).reset_index(drop=True)
        if lbc.shape[0] >= 5:
            ranks.append(lbc.query("Name == 'TabICL-CheMeleon'").index[0] + 1)

        out.write(f"\n## `{name}`\n\n{results.to_markdown()}\n\n"
                  f"{lbc[['Name', b.main_metric.label]].to_markdown(index=False)}\n")
        perf[name] = {b.main_metric.label: float(val)}
        print(f"{name}: {b.main_metric.label} = {val:.4f}", flush=True)

    out.write(f"\n# Summary\n\nAverage rank ({len(ranks)}): "
              f"{(sum(ranks) / len(ranks) if ranks else float('nan')):.2f}\n\n"
              f"results_dict = {json.dumps(perf, indent=4)}\n")
    out.close()
    print(f"\nTabICL baseline done. Avg rank = "
          f"{(sum(ranks) / len(ranks) if ranks else float('nan')):.2f}")


if __name__ == "__main__":
    main()
