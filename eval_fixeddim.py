"""Evaluate the fixed-dim ICL transformer on polaris benchmarks (in-context).

Uses the shared CheMeleon embedding cache (data/bench_embeddings.pt) so this
and the TabPFN baseline score on *identical* inputs. Mirrors the repo's
eval.py output (leaderboard snippet + eval_results.md).

In-context prediction:
  * standardise embeddings with pool statistics (as in pre-training)
  * standardise the TRAIN labels to mean-0/std-1 (as per-task in training)
  * one forward pass over [context + queries]; read query predictions
  * un-standardise; score with polaris
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

from model import FixedDimICLTransformer
from train import load_model


def load_pool_stats(pool_path: str):
    pool = torch.load(pool_path, map_location="cpu", weights_only=True).float()
    return pool.mean(0), pool.std(0).clamp_min(1e-6)


@torch.inference_mode()
def predict_in_context(model, X_ctx, y_ctx, X_query, max_context: int,
                       chunk: int = 64):
    """X/y in *raw* (unstandardised) units; returns raw predictions for queries.

    Queries are processed in chunks so that (context + chunk) <= max_context.
    """
    n_ctx, n_q = len(X_ctx), len(X_query)
    device = X_ctx.device

    ym = y_ctx.mean()
    ys = y_ctx.std()
    if ys < 1e-6:
        ys = torch.ones_like(ys)
    y_ctx_std = (y_ctx - ym) / ys

    # cap context so that context + one query chunk fits
    if n_ctx + chunk > max_context:
        keep = max(1, max_context - chunk)
        perm = torch.randperm(n_ctx)[:keep]
        X_ctx, y_ctx_std = X_ctx[perm], y_ctx_std[perm]
        n_ctx = keep

    preds = []
    for s in range(0, n_q, chunk):
        X_q = X_query[s:s + chunk]
        nq = X_q.size(0)
        X_all = torch.cat([X_ctx, X_q], dim=0)
        y_all = torch.cat([y_ctx_std, torch.zeros(nq, 1, device=device)], dim=0)
        mask = torch.zeros(len(X_all), dtype=torch.bool, device=device)
        mask[n_ctx:] = True
        pred = model(X_all.unsqueeze(0), y_all.unsqueeze(0), mask.unsqueeze(0))[0]
        preds.append(pred[n_ctx:].squeeze(-1))
    pred_q = torch.cat(preds, dim=0)
    return pred_q * ys + ym


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python eval_fixeddim.py /path/to/model.ckpt [leaderboard.csv]")
        exit(1)
    model_ckpt = sys.argv[1]
    leaderboard_path = sys.argv[2] if len(sys.argv) > 2 else \
        "polaris_benchmarks_leaderboard_20260709_150240.csv"
    pool_path = sys.argv[3] if len(sys.argv) > 3 else "data/pool_full.pt"
    cache_path = sys.argv[4] if len(sys.argv) > 4 else "data/bench_embeddings.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(model_ckpt), device=device)
    mu, sd = load_pool_stats(pool_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    def std_embed(X):
        return (X - mu) / sd

    lb = pd.read_csv(leaderboard_path)
    out = open("eval_results_fixeddim.md", "w")
    out.write(f"# FixedDim-ICL Eval Results\n\ntimestamp: {datetime.datetime.now()}\n"
              f"checkpoint: {model_ckpt}\n\n")
    perf, ranks = {}, []
    for name, d in cache.items():
        if "REGRESSION" not in d["task_type"]:
            print(f"Skipping {name} (not regression)")
            continue
        X_ctx = std_embed(d["X_train"]).to(device)
        y_ctx = torch.tensor(d["y_train"], dtype=torch.float32).unsqueeze(-1).to(device)
        X_q = std_embed(d["X_test"]).to(device)

        preds = predict_in_context(model, X_ctx, y_ctx, X_q, model.cfg.max_context).cpu()
        torch.cuda.empty_cache()

        b = po.load_benchmark(name)
        results = b.evaluate(preds.flatten().numpy()).results
        val = results.query(f"Metric == '{b.main_metric.label}'")["Score"].values[0]

        subset = lb[lb["Benchmark_ID"] == name].copy()
        lbc = pd.concat([subset, pd.DataFrame([{"Name": "FixedDimICL",
                                                b.main_metric.label: val}])],
                        ignore_index=True).sort_values(
            by=b.main_metric.label,
            ascending=b.main_metric.label in {"mean_absolute_error", "mean_squared_error"}
        ).reset_index(drop=True)
        if lbc.shape[0] >= 5:
            ranks.append(lbc.query("Name == 'FixedDimICL'").index[0] + 1)

        out.write(f"\n## `{name}`\n\n{results.to_markdown()}\n\n"
                  f"{lbc[['Name', b.main_metric.label]].to_markdown(index=False)}\n")
        perf[name] = {b.main_metric.label: float(val)}
        print(f"{name}: {b.main_metric.label} = {val:.4f}")
    out.write(f"\n# Summary\n\nAverage rank ({len(ranks)}): "
              f"{(sum(ranks) / len(ranks) if ranks else float('nan')):.2f}\n\n"
              f"results_dict = {json.dumps(perf, indent=4)}\n")
    out.close()
    print(f"\nFixedDimICL done. Avg rank = "
          f"{(sum(ranks) / len(ranks) if ranks else float('nan')):.2f}")


if __name__ == "__main__":
    main()
