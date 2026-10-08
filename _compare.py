"""Build the definitive comparison table: best kernel config per benchmark
vs verified baselines (TabICL v2, CheMeleon, failed repo, TabPFNv2-rdkit)."""
import json
import numpy as np

kb = json.load(open("kernel_baseline_summary.json"))

# Verified baseline numbers (from tabicl_baseline_results.md + REPORT.md)
# Only benchmarks with a complete baseline set form the headline.
TABICL = {
    "polaris/pkis2-ret-wt-reg-v2": 656.012,
    "polaris/pkis2-kit-wt-reg-v2": 863.643,
    "polaris/pkis2-egfr-wt-reg-v2": 398.603,
    "polaris/adme-fang-solu-1": 0.6033,
    "polaris/adme-fang-rppb-1": 0.7789,
    "polaris/adme-fang-hppb-1": 0.7404,
    "polaris/adme-fang-perm-1": 0.7443,
    "polaris/adme-fang-rclint-1": 0.6674,
    "polaris/adme-fang-hclint-1": 0.6729,
}
CHEMELEON = {
    "polaris/pkis2-ret-wt-reg-v2": 684.319,
    "polaris/pkis2-kit-wt-reg-v2": 849.611,
    "polaris/pkis2-egfr-wt-reg-v2": 459.929,
    "polaris/adme-fang-solu-1": 0.682,
    "polaris/adme-fang-rppb-1": 0.662,
    "polaris/adme-fang-hppb-1": 0.793,
    "polaris/adme-fang-perm-1": 0.822,
    "polaris/adme-fang-rclint-1": 0.757,
    "polaris/adme-fang-hclint-1": 0.720,
}
FAILED = {
    "polaris/pkis2-ret-wt-reg-v2": 1068.1,
    "polaris/pkis2-kit-wt-reg-v2": 1046.3,
    "polaris/pkis2-egfr-wt-reg-v2": 705.7,
    "polaris/adme-fang-solu-1": 0.49,
    "polaris/adme-fang-rppb-1": 0.63,
    "polaris/adme-fang-hppb-1": 0.73,
    "polaris/adme-fang-perm-1": 0.67,
    "polaris/adme-fang-rclint-1": 0.57,
    "polaris/adme-fang-hclint-1": 0.52,
}
# TabPFNv2-rdkit uses rdkit fingerprints (DIFFERENT input) - reference only
TABBPFN_RDKIT = {
    "polaris/adme-fang-rppb-1": 0.816,
    "polaris/adme-fang-hppb-1": 0.827,
    "polaris/adme-fang-perm-1": 0.798,
}

CONFIGS = ["KRR/raw", "GP/raw", "kNN/raw", "Ridge/raw",
           "KRR/white", "GP/white", "kNN/white", "Ridge/white"]


def best_kernel(name):
    d = kb[name]
    m = d["metric"]
    hgb = m in ("pearsonr", "spearmanr")
    vals = [(c, d[c]) for c in CONFIGS if c in d and isinstance(d[c], float)]
    if not vals:
        return None, None
    if hgb:
        c, v = max(vals, key=lambda t: t[1])
    else:
        c, v = min(vals, key=lambda t: t[1])
    return c, v


def hgb(m):
    return m in ("pearsonr", "spearmanr")


print("=" * 96)
print("HEADLINE: 9 polaris benchmarks (identical frozen CheMeleon 2048-dim input)")
print("=" * 96)
hdr = f"{'benchmark':26s}{'metric':18s}{'n_tr':>6s} | {'bestKernel':>22s} | {'TabICLv2':>9s} {'CheMl':>7s} {'Failed':>7s}"
print(hdr)
print("-" * len(hdr))
wins = {"kernel": 0, "tabicl": 0}
nine = []
for name in list(TABICL.keys()):
    d = kb[name]
    m = d["metric"]
    c, v = best_kernel(name)
    t = TABICL[name]; ch = CHEMELEON[name]; f = FAILED[name]
    # who wins (higher better for pearsonr/spearmanr, lower for MSE)
    if hgb(m):
        best = max(v, t, ch, f)
    else:
        best = min(v, t, ch, f)
    winner = {v: "Kernel", t: "TabICL", ch: "CheMeleon", f: "Failed"}[best]
    if winner == "Kernel":
        wins["kernel"] += 1
    elif winner == "TabICL":
        wins["tabicl"] += 1
    nine.append((name, m, d["n_train"], c, v, t, ch, f, winner))
    mark = " *" if winner == "Kernel" else ""
    print(f"{name.replace('polaris/',''):26s}{m:18s}{d['n_train']:6d} | "
          f"{f'{c} {v:.4f}':>22s} | {t:9.4f} {ch:7.3f} {f:7.3f}  {winner}{mark}")

print(f"\nHead-to-head wins (best-of-4 per bench):  Kernel={wins['kernel']}  "
      f"TabICLv2={wins['tabicl']}  (CheMeleon/Failed = rest)")

# ADME pearsonr subset (where the in-context/attention claim lives)
adme = [r for r in nine if r[1] == "pearsonr"]
print(f"\nOn the 6 ADME pearsonr benchmarks: kernel wins "
      f"{sum(1 for r in adme if r[8]=='Kernel')}/6")

print("\n" + "=" * 96)
print("FULL 17 (best kernel config per benchmark; baselines where available)")
print("=" * 96)
for name, d in kb.items():
    m = d["metric"]
    c, v = best_kernel(name)
    t = TABICL.get(name, float('nan'))
    ch = CHEMELEON.get(name, float('nan'))
    print(f"{name:38s}{m:18s} {d['n_train']:5d}  bestKernel={v:.4f} ({c})  "
          f"TabICL={t:.4f}  CheMeleon={ch:.3f}")

# mean-predictor (collapse) reference on the 9
print("\nmean-predictor (the transformer's collapse) on the 9:")
for name in list(TABICL.keys()):
    d = kb[name]
    mv = d.get("mean/mean", float('nan'))
    print(f"  {name.replace('polaris/',''):26s} {d['metric']:18s} {mv if mv==mv else 'nan':>10}")
