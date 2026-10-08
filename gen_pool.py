"""Generate a fixed pool of CheMeleon 2048-dim embeddings for pre-training.

The pool is a random sample of molecules from the PubChem SMILES database,
embedded with the frozen CheMeleon message-passing model (mean-aggregated,
full 2048-dim output). Saved as a single .pt tensor of shape (N_pool, 2048).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from featurize import CheMeleonEmbedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-database", type=str,
                        default=str(Path(__file__).parent / "data" / "cleaned_pubchem_1MM.smiles"))
    parser.add_argument("--pool-size", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-path", type=str,
                        default=str(Path(__file__).parent / "data" / "pool_full.pt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.smiles_database) as fh:
        all_smiles = [ln.strip() for ln in fh if ln.strip()]
    print(f"loaded {len(all_smiles)} smiles")

    rng = np.random.default_rng(args.seed)
    n = min(args.pool_size, len(all_smiles))
    idx = rng.choice(len(all_smiles), size=n, replace=False)
    pool_smiles = [all_smiles[i] for i in idx]

    emb = CheMeleonEmbedder(device="cuda", type="full")
    chunks = []
    for i in range(0, n, args.batch_size):
        batch = pool_smiles[i:i + args.batch_size]
        x = emb(batch).to("cpu").float()
        chunks.append(x)
        torch.cuda.empty_cache()
        if (i // args.batch_size) % 5 == 0:
            print(f"  {i}/{n} ({i / n * 100:.1f}%)")

    pool = torch.cat(chunks, dim=0)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pool, out)
    print(f"saved pool {tuple(pool.shape)} -> {out}")


if __name__ == "__main__":
    main()
