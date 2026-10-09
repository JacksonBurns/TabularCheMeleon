"""Generate the molecular-descriptor feature cache for all polaris/tdcommons benchmarks.

Feature vector (per SMILES) = concatenation of:
  - 2048-dim radius-3 Morgan COUNT fingerprint (counts, NOT binary)
  - 325 RDKit 2D descriptors (rdkit.Chem.Descriptors module)
Total: 2373-dim.

This is the "classical descriptor embedding" alternative to the 2048-dim frozen
CheMeleon GNN embedding, for head-to-head GP/kernel comparison.

Output: data/bench_descriptor_embeddings.pt
  dict[benchmark_name] = {
      'smiles_train': [...], 'y_train': [...], 'smiles_test': [...],
      'X_train': (n,D) float tensor, 'X_test': (n,D) float tensor,
      'target': str, 'metric': str, 'task_type': str,
      'n_train': int, 'n_test': int,
  }
Same schema as data/bench_embeddings.pt (CheMeleon) so the eval harness is shared.

Run:  ~/miniforge3/envs/chemeleon/bin/python gen_descriptor_embeddings.py
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("RDLogger", "error")
warnings.filterwarnings("ignore")

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

SRC_CACHE = "data/bench_embeddings.pt"
OUT_CACHE = "data/bench_descriptor_embeddings.pt"
FP_DIM = 2048
RADIUS = 3
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Canonical RDKit descriptor functions: all public callables in Descriptors that
# take a Mol and return a finite scalar. Built once at import (module-global so
# it survives the process-pool fork).
_DESC_FNS = []
_DESC_LABELS = []
for _name in sorted(dir(Descriptors)):
    _obj = getattr(Descriptors, _name)
    if callable(_obj) and not _name.startswith("_") and _name[0].isupper():
        _DESC_FNS.append(_obj)
        _DESC_LABELS.append(_name)
N_DESC = len(_DESC_FNS)
TOTAL_DIM = FP_DIM + N_DESC


def _morgan_count(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    arr = np.zeros(FP_DIM, dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetHashedMorganFingerprint(mol, RADIUS, nBits=FP_DIM)
    for i, v in fp.GetNonzeroElements().items():
        arr[i] = v
    return arr


def _rdkit_descs(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    out = np.zeros(N_DESC, dtype=np.float32)
    if mol is None:
        return out
    for j, fn in enumerate(_DESC_FNS):
        try:
            v = fn(mol)
            if isinstance(v, (int, float)) and np.isfinite(v):
                out[j] = v
        except Exception:
            out[j] = 0.0
    return out


def _feat_one(smi: str) -> np.ndarray:
    return np.concatenate([_morgan_count(smi), _rdkit_descs(smi)])


def _feat_batch(smi_list):
    return [_feat_one(s) for s in smi_list]


def main():
    src = torch.load(SRC_CACHE, map_location="cpu", weights_only=False)
    print(f"Loaded {len(src)} benchmarks from {SRC_CACHE}", flush=True)
    print(f"Feature dim: {FP_DIM} (Morgan count r={RADIUS}) + {N_DESC} (RDKit desc) "
          f"= {TOTAL_DIM}", flush=True)

    out = {}
    for name, d in src.items():
        smi_tr = list(d["smiles_train"])
        smi_te = list(d["smiles_test"])
        all_smi = smi_tr + smi_te

        # parallel featurize
        feats = [None] * len(all_smi)
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {ex.submit(_feat_batch, chunk): i
                    for i, chunk in enumerate(
                        [all_smi[k:k + 500] for k in range(0, len(all_smi), 500)])}
            for fut in as_completed(futs):
                ci = futs[fut]
                chunk = all_smi[ci * 500:(ci + 1) * 500]
                for k, s in enumerate(chunk):
                    feats[ci * 500 + k] = fut.result()[k]

        X = torch.tensor(np.stack(feats), dtype=torch.float32)
        out[name] = {
            "smiles_train": smi_tr,
            "y_train": d["y_train"],
            "smiles_test": smi_te,
            "X_train": X[:len(smi_tr)],
            "X_test": X[len(smi_tr):],
            "target": d["target"],
            "metric": d["metric"],
            "task_type": d["task_type"],
            "n_train": len(smi_tr),
            "n_test": len(smi_te),
        }
        print(f"  {name:48s} n_tr={len(smi_tr):5d} n_te={len(smi_te):5d} "
              f"dim={X.shape[1]}  fp_nz_avg={X[:len(smi_tr)][:FP_DIM].ne(0).float().mean(0).mean():.1f}",
              flush=True)

    torch.save(out, OUT_CACHE)
    print(f"\nSaved {len(out)} benchmarks to {OUT_CACHE}", flush=True)
    print(f"Descriptor labels ({N_DESC}): {N_DESC} total", flush=True)


if __name__ == "__main__":
    main()
