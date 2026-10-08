"""Evaluate the v1 from-scratch ICL model on the polaris benchmarks (the same 9
that the TabICL v2 reference baseline ran), for a clean head-to-head table."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import pandas as pd
import polaris as po
from polaris.utils.types import TargetType

from model_v1 import load_legacy
from featurize import CheMeleonEmbedder
from eval_fixeddim import load_pool_stats

BENCHMARKS = (
    "polaris/pkis2-ret-wt-reg-v2", "polaris/pkis2-kit-wt-reg-v2",
    "polaris/pkis2-egfr-wt-reg-v2", "polaris/adme-fang-solu-1",
    "polaris/adme-fang-rppb-1", "polaris/adme-fang-hppb-1",
    "polaris/adme-fang-perm-1", "polaris/adme-fang-rclint-1",
    "polaris/adme-fang-hclint-1",
)


@torch.inference_mode()
def predict_in_context(model, X_ctx, y_ctx, X_query, max_context, chunk=64):
    n_ctx, n_q = len(X_ctx), len(X_query)
    device = X_ctx.device
    ym = y_ctx.mean(); ys = y_ctx.std()
    if ys < 1e-6:
        ys = torch.ones_like(ys)
    y_ctx_std = (y_ctx - ym) / ys
    if n_ctx + chunk > max_context:
        keep = max(1, max_context - chunk)
        perm = torch.randperm(n_ctx)[:keep]
        X_ctx, y_ctx_std, n_ctx = X_ctx[perm], y_ctx_std[perm], keep
    preds = []
    for s in range(0, n_q, chunk):
        X_q = X_query[s:s + chunk]; nq = X_q.size(0)
        X_all = torch.cat([X_ctx, X_q], 0)
        y_all = torch.cat([y_ctx_std, torch.zeros(nq, 1, device=device)], 0)
        mask = torch.zeros(len(X_all), dtype=torch.bool, device=device); mask[n_ctx:] = True
        pred = model(X_all.unsqueeze(0), y_all.unsqueeze(0), mask.unsqueeze(0))[0]
        preds.append(pred[n_ctx:].squeeze(-1))
    return torch.cat(preds, 0) * ys + ym


def main():
    device = "cuda"
    model = load_legacy("icl_logs/fixeddim_icl.pt", device=device)
    mu, sd = load_pool_stats("data/pool_full.pt")
    mu, sd = mu.to(device), sd.to(device)
    get_emb = CheMeleonEmbedder(device=device, type="full")
    lb = pd.read_csv("polaris_benchmarks_leaderboard_20260709_150240.csv")
    out = open("eval_v1_results.md", "w")
    perf, ranks = {}, []
    for name in BENCHMARKS:
        b = po.load_benchmark(name)
        s_col, t_col = list(b.input_cols)[0], list(b.target_cols)[0]
        train, test = b.get_train_test_split()
        tdf, tdf2 = train.as_dataframe(), test.as_dataframe()
        if b.target_types[t_col] != TargetType.REGRESSION:
            continue
        X_ctx = ((get_emb(tdf[s_col].to_list()) - mu) / sd)
        y_ctx = torch.tensor(tdf[t_col].to_list(), dtype=torch.float32).unsqueeze(-1).to(device)
        X_q = ((get_emb(tdf2[s_col].to_list()) - mu) / sd)
        preds = predict_in_context(model, X_ctx, y_ctx, X_q, model.cfg.max_context).cpu()
        torch.cuda.empty_cache()
        results = b.evaluate(preds.flatten().numpy()).results
        val = results.query(f"Metric == '{b.main_metric.label}'")["Score"].values[0]
        subset = lb[lb["Benchmark_ID"] == name].copy()
        lbc = pd.concat([subset, pd.DataFrame([{"Name": "ScratchICL-v1", b.main_metric.label: val}])],
                        ignore_index=True).sort_values(
            by=b.main_metric.label,
            ascending=b.main_metric.label in {"mean_absolute_error", "mean_squared_error"}
        ).reset_index(drop=True)
        if lbc.shape[0] >= 5:
            ranks.append(lbc.query("Name == 'ScratchICL-v1'").index[0] + 1)
        out.write(f"\n## `{name}`\n\n{results.to_markdown()}\n\n"
                  f"{lbc[['Name', b.main_metric.label]].to_markdown(index=False)}\n")
        perf[name] = {b.main_metric.label: float(val)}
        print(f"{name}: {b.main_metric.label} = {val:.4f}", flush=True)
    out.write(f"\n# Summary\n\nAverage rank ({len(ranks)}): "
              f"{(sum(ranks) / len(ranks) if ranks else float('nan')):.2f}\n")
    out.close()
    print(f"\nv1 done. Avg rank = {(sum(ranks) / len(ranks) if ranks else float('nan')):.2f}")


if __name__ == "__main__":
    main()
