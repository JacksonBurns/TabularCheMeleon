"""Precompute frozen CheMeleon 2048-dim embeddings for all polaris benchmark
SMILES into a single cache, so eval and baselines share identical inputs.

Output: data/bench_embeddings.pt  -> dict[benchmark_name] = {
    'smiles_train': [...], 'y_train': [...], 'smiles_test': [...],
    'X_train': (n,D) tensor, 'X_test': (n,D) tensor, 'target': str,
    'metric': str, 'task_type': str
}
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import polaris as po
from polaris.utils.types import TargetType

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = CheMeleonEmbedder(device=device, type="full")
    cache = {}
    for name in BENCHMARKS:
        b = po.load_benchmark(name)
        s_col, t_col = list(b.input_cols)[0], list(b.target_cols)[0]
        train, test = b.get_train_test_split()
        tdf, tdf2 = train.as_dataframe(), test.as_dataframe()
        tt = b.target_types[t_col]
        X_tr = emb(tdf[s_col].to_list()).to("cpu").float()
        X_te = emb(tdf2[s_col].to_list()).to("cpu").float()
        torch.cuda.empty_cache()
        cache[name] = {
            "smiles_train": tdf[s_col].to_list(),
            "y_train": tdf[t_col].to_list(),
            "smiles_test": tdf2[s_col].to_list(),
            "y_test": tdf2[t_col].to_list(),
            "X_train": X_tr, "X_test": X_te,
            "target": t_col, "metric": b.main_metric.label,
            "task_type": str(tt), "n_train": len(tdf), "n_test": len(tdf2),
        }
        print(f"{name}: train={len(tdf)} test={len(tdf2)} {tt} -> {b.main_metric.label}")
    out = Path("data/bench_embeddings.pt")
    torch.save(cache, out)
    print(f"saved {len(cache)} benchmarks -> {out}")


if __name__ == "__main__":
    main()
