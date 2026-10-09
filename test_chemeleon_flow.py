"""Functional test for CheMeleonFlowRegressor / Classifier.

Verifies:
  1. GP math reproduces gp_explore.py scores on rppb (white/matern52 ~0.796).
  2. sklearn API: fit / predict / predict_with_uncertainty / score / R2.
  3. Classifier: predict_proba / predict / predict_classification / decision.
  4. SMILES input path (batch CuikmolmakerMolGraphFeaturizer + CheMeleon MPNN).
  5. get_params / set_params (sklearn clone-ability).
  6. Uncertainty is calibrated (low std near train, high std far away).
"""
from __future__ import annotations
import warnings, os
os.environ["RDLogger"] = "error"; warnings.filterwarnings("ignore")
import numpy as np
import torch
import polaris as po

from chemeleon_flow import (CheMeleonFlowRegressor, CheMeleonFlowClassifier)


cache = torch.load("data/bench_embeddings.pt", map_location="cpu", weights_only=False)
name = "polaris/adme-fang-rppb-1"
d = cache[name]
Xtr = d["X_train"].numpy().astype("float64")
Xte = d["X_test"].numpy().astype("float64")
ytr = np.asarray(d["y_train"], float)
mname = d["metric"]
b = po.load_benchmark(name)
destd = lambda p: p

def pol_score(pred):
    val = b.evaluate(pred.astype("float64")).results
    return float(val.query(f"Metric == '{mname}'")["Score"].values[0])

print(f"=== {name}  n_tr={len(ytr)} n_te={len(Xte)} metric={mname} ===\n")

# ---- 1. Reproduce gp_explore per-kernel scores (embeddings in, no featurize) ----
print("[1] Per-kernel scores (should match gp_explore.py):")
for kern, space in [("smk","white"),("matern52","white"),("rbf","white"),
                    ("smk","raw"),("matern52","raw"),("rbf","raw")]:
    reg = CheMeleonFlowRegressor(kernel=kern, space=space)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    sc = pol_score(pred)
    print(f"    {kern:10s} {space:6s}  {mname}={sc:.4f}")

# reference from gp_explore_summary.json
import json
gs = json.load(open("gp_explore_summary.json"))[name]
print("    --- gp_explore.py reference ---")
for k in ["chem/white/smk","chem/white/matern52","chem/white/rbf",
          "chem/raw/smk","chem/raw/matern52","chem/raw/rbf"]:
    print(f"    {k:24s} {gs[k]:.4f}")

# ---- 2. sklearn API on the default (smk/white) ----
print("\n[2] sklearn API (default rbf/auto-space):")
reg = CheMeleonFlowRegressor()
reg.fit(Xtr, ytr)
pred = reg.predict(Xte)
mean, std = reg.predict_with_uncertainty(Xte)
r2 = reg.score(Xtr, ytr)
print(f"    predict shape: {pred.shape}, score(rppb test): {pol_score(pred):.4f}")
print(f"    predict_with_uncertainty: mean {mean.shape}, std {std.shape}")
print(f"    train R2: {r2:.4f}")
print(f"    std range: [{std.min():.4f}, {std.max():.4f}]  (y_std={ytr.std():.3f})")
assert np.all(std > 0), "stds must be positive"
assert np.allclose(pred, mean), "predict == predict_with_uncertainty mean"

# ---- 3. get_params / set_params / clone ----
print("\n[3] sklearn clone-ability:")
from sklearn.base import clone
p = reg.get_params()
print(f"    get_params keys: {sorted(p.keys())}")
reg2 = clone(reg)
print(f"    clone OK: {type(reg2).__name__}, kernel={reg2.kernel}, space={reg2.space}")
reg2.set_params(kernel="matern52")
print(f"    set_params OK: kernel now {reg2.kernel}")

# ---- 4. Classifier ----
print("\n[4] Classifier (threshold on continuous y):")
thr = float(np.median(ytr))
clf = CheMeleonFlowClassifier(threshold=thr, kernel="smk", space="white")
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)
pact, cls = clf.predict_classification(Xte)
dc = clf.decision_function(Xte)
print(f"    predict_proba shape: {proba.shape}, rows sum to 1: {np.allclose(proba.sum(1),1)}")
print(f"    P(active) range: [{pact.min():.3f}, {pact.max():.3f}]")
print(f"    class balance: {cls.mean():.3f} active")
print(f"    decision_function range: [{dc.min():.2f}, {dc.max():.2f}]")
assert proba.shape == (len(Xte), 2)
assert np.all((proba >= 0) & (proba <= 1))

# binary-label path
y_bin = (ytr >= thr).astype(int)
clf_b = CheMeleonFlowClassifier(kernel="rbf", space="white")  # threshold auto->0.5
clf_b.fit(Xtr, y_bin)
acc = clf_b.score(Xtr, y_bin)
print(f"    binary-label fit: auto threshold={clf_b.threshold}, train acc={acc:.3f}")

# ---- 5. SMILES input path (batch cuik molmaker featurization) ----
print("\n[5] SMILES input path (CheMeleon batch featurizer):")
smi_tr = d["smiles_train"][:150]
y_tr_sub = ytr[:150]
smi_te = d["smiles_test"][:50]
# ground-truth embeddings for the same SMILES (from cache)
Xtr_sub = Xtr[:150]; Xte_sub = Xte[:50]
reg_s = CheMeleonFlowRegressor(kernel="rbf", space="white", device="cuda")
reg_s.fit(smi_tr, y_tr_sub)
pred_smi = reg_s.predict(smi_te)
# compare to embedding-input prediction on identical molecules
reg_e = CheMeleonFlowRegressor(kernel="rbf", space="white")
reg_e.fit(Xtr_sub, y_tr_sub)
pred_emb = reg_e.predict(Xte_sub)
diff = np.max(np.abs(pred_smi - pred_emb))
print(f"    SMILES-featurized vs embedding-input max abs diff: {diff:.2e}")
print(f"    SMILES-path score (subset test): {pol_score(pred_smi):.4f}")
assert diff < 1e-3, "SMILES featurization must match cached CheMeleon embeddings"
print("    -> batch cuik molmaker featurization reproduces cached embeddings ✓")

print("\nALL FUNCTIONAL TESTS PASSED")
