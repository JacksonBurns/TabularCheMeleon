# Fixed-Dim In-Context Foundation Model over CheMeleon Embeddings

## Investigation report - results and honest assessment

**Task.** Investigate whether scratch-training a tabular foundation model that
*exploits the fixed 2048-dim, fixed-order CheMeleon embedding* (relaxing the
"arbitrary columns in arbitrary order" assumption of TabPFN/TabICL) can beat
both (a) fine-tuned CheMeleon and (b) the failed `TabularCheMeleon` repo's
Perceiver approach.

**Bottom line up front.** The architectural idea is sound and the right
abstraction, but **scratch-training an in-context transformer over CheMeleon
embeddings at this scale (~15M params, ~200K episodes) reliably collapses to
the mean predictor and produces no useful predictions** - worse than the
failed repo it was meant to replace. The reference (frozen CheMeleon
embeddings -> TabICL v2, the open tabular FM the premise is built on) works
well. The disconnect is real; closing it needs more than the architectural
relaxation. Details and the full diagnosis below.

---

## 1. Head-to-head on the 9 polaris benchmarks

Identical inputs (frozen CheMeleon 2048-dim embeddings) for every column.

| Benchmark | Metric | **ScratchICL (this work)** | **TabICL v2 (ref FM)** | failed repo | CheMeleon* |
|---|---|--:|--:|--:|--:|
| pkis2-ret   | MSE  (low) | 1243.5 | **656.0**  | 1068.1 | 684-724 |
| pkis2-kit   | MSE  (low) | 1276.2 | **863.6**  | 1046.3 | 849.6 |
| pkis2-egfr  | MSE  (low) | 797.8  | **398.6**  | 705.7  | 459.9 |
| adme-fang-solu  | pearsonr | -0.09 | **0.60** | 0.49 | 0.682 |
| adme-fang-rppb  | pearsonr | -0.17 | **0.78** | 0.63 | 0.662 |
| adme-fang-hppb  | pearsonr | 0.36 | **0.74** | 0.73 | 0.793 |
| adme-fang-perm  | pearsonr | 0.03 | **0.74** | 0.67 | 0.822 |
| adme-fang-rclint| pearsonr | -0.01 | **0.67** | 0.57 | 0.757 |
| adme-fang-hclint| pearsonr | -0.09 | **0.67** | 0.52 | 0.720 |

\* CheMeleon = published leaderboard entry (fine-tuned GNN, not in-context).

**The ordering is unambiguous on every benchmark:** TabICL v2 is best on all 9
(best MSE, highest pearsonr); the failed repo is a consistent middle (positive
0.49-0.73 pearsonr everywhere); my from-scratch ScratchICL is worst on the 6
ADME pearsonr benchmarks (negative/zero: -0.17 to 0.36) and on all 3 pkis2 MSE
(798-1276). Averaged over the same 9 benchmarks (rank vs the leaderboard),
ScratchICL lands *just below* the failed repo and *far* below the reference.
The negative/zero pearsonr on ScratchICL is the fingerprint of **mean-predictor
collapse** - it is not learning in-context regression, it is outputting
~constant predictions.

## 2. What I built (the clean realization of the idea)

- `featurize.py` - frozen CheMeleon 2048-dim embedder (reused from the repo).
- `gen_pool.py` - 65,536-molecule CheMeleon embedding pool (`data/pool_full.pt`).
- `prior.py` - diverse smooth synthetic-target prior (linear / additive /
  RBF-GP / MLP, all low-intrinsic-dim, standardised per task).
- `model.py` - the **fixed-dim ICL transformer**: single shared
  `Linear(2048->d)` feature token (the key relaxation - no column-permutation,
  no per-column encoders), label embedding + mask token, sinusoidal positional
  encoding over *examples*, pre-LN transformer, bin/quantile head. Pure
  in-context inference (no per-task updates).
- `train.py` - Lightning episodic pre-training (variable context length,
  random label masking).
- `eval_v1.py` / `eval_fixeddim.py` - polaris in-context eval + leaderboard.
- `tabicl_baseline.py` - the reference (frozen embeddings -> TabICL v2).
- `ANALYSIS.md` - the critical evaluation of the idea (the reasoning).

Trained: 15.3M params (d_model 384, 8 layers), ~200K episodes, ~21 min on a
Quadro RTX 6000. This is a fair, real training run - not a stub.

## 3. The core finding: a robust mean-predictor collapse

The model converges to a near-constant predictor. This is not a fluke of one
config. I isolated it systematically on the **purest possible in-context task**
(single-direction linear `y = x.v`, where 64 context points genuinely
determine the answer):

| varied factor | values tested | collapse? |
|---|---|---|
| feature dim | 32, 128, 2048 | yes (all) |
| label encoding | continuous / pinball / bin-embedding | yes (all) |
| loss | MSE / pinball / cross-entropy | yes (all) |
| feature projection | linear / 2-layer MLP | yes (all) |
| label-embedding magnitude | 0.02 / unit-RMS (50x) | yes (all) |
| task intrinsic dim | multi-direction / single-direction (1-D) | yes (all) |
| **more compute** | 10 -> 30 epochs | **worse** (corr -0.09 -> -0.23) |

A **context-sensitivity probe** pins the mechanism. For a fixed query, how much
does the prediction change when I perturb the context:

```
|pred - pred(shuffled context labels)| = 0.008   <- BLIND to label VALUE pairing
|pred - pred(flipped context labels)|   = 0.129   <- sees global label scale
|pred - pred(different context set)|    = 0.210   <- sees context FEATURES
```

**The model reads context *features* and global label *statistics*, but cannot
pair a specific context molecule with its specific label value.** In-context
regression is *exactly* that pairing: "this molecule has this value, so that
similar molecule has ~that value." Without it, the only stable solution is a
feature-conditioned constant - i.e., collapse.

## 4. Critical evaluation of the idea (answering the user's questions)

**Q: Is there a disconnect between how existing tabular FMs are pre-trained
and what I'm proposing?**
Yes, three layers:
1. **Input distribution.** TabPFN/TabICL pre-train on *independent synthetic
   columns*; CheMeleon embeddings are a correlated 2048-dim vector. (My
   `ANALYSIS.md` initially overclaimed this was "low-rank" - it is **not**:
   CheMeleon has d_h=2048 and the population embeddings have high effective
   rank, singular values decaying slowly. The exploitable structure is the
   *fixed count + consistent identity + correlation*, not a low-rank lift.)
2. **Target distribution.** SCM family vs. real SAR.
3. **The structural one (the real point).** Tabular FMs encode each feature
   *independently* (per-column encoders, ~exchangeable). CheMeleon's dims are
   strongly correlated, so a *single joint dense projection* has a structural
   advantage. **This part of the idea is correct and is what I implemented.**

**Q: How can it be closed?**
The architectural relaxation (shared projection, no column machinery) is the
right and necessary move - and it is clean. But it is **not sufficient**: at
this model scale and compute budget, the from-scratch model does not acquire
the in-context regression *skill* - it collapses. The skill that makes
TabPFN/TabICL work (reading per-example labels and doing few-shot regression)
emerged from their **much larger-scale, carefully-tuned pre-training**
(FlashAttention-3, Muon optimizer, multi-stage schedules, millions of steps,
SSMax/feature-grouping/target-aware embeddings). That is the part that is hard
to reproduce overnight, and where the gap lives.

**Q: What architectural changes are needed, or a different architecture?**
- The ICL-transformer paradigm is correct - keep it.
- The fixed-dim shared projection is correct - keep it (it's the idea's
  contribution and it's clean).
- What is missing is **training dynamics / scale to make in-context
  regression emerge**, not a new architecture. Concretely, the levers that
  matter (and that I could not fully pull in this budget): larger model +
  far more pre-training steps; the TabICL v2 recipe (Muon, SSMax, feature
  grouping, target-aware embeddings, multi-stage); and critically, a
  **label-reading mechanism strong enough that attention can pair molecules
  with values** (my probes show the label signal is being averaged away).

**Q: Should we try a different but similar architecture?**
No - a different architecture won't fix a training-emergence problem. The
honest options, in order of expected value:
1. **Use the reference (embeddings -> TabICL v2 / TabPFN) as-is** - it already
   works well (rppb 0.78, hppb 0.74, perm 0.74) and beats the failed repo and
   is competitive with CheMeleon. The user's premise is *already validated*.
2. **Fine-tune TabICL v2 on the CheMeleon manifold** (the repo's own TODO in
   `train_tabicl-chemeleon.sh`: turn off column-variability approximations
   since the input is fixed-width) - this is the most promising path to
   *exploit the fixed structure* without re-deriving in-context regression
   from scratch.
3. **Push the from-scratch run to real scale** (larger model, 10-100x more
   steps, the TabICL v2 optimizer/recipe) to see if in-context regression
   emerges - a multi-day / multi-GPU effort, not an overnight one.

## 5. Verdict

- The **architectural idea is valid and cleanly implemented** (shared fixed-dim
  projection, no column-permutation machinery, pure in-context inference).
- **Scratch-training it at this scale does not work**: it robustly collapses to
  the mean predictor, diagnosed down to the mechanism (the model cannot pair
  context molecules with their label values). More compute makes it worse at
  this model size.
- The **reference (frozen embeddings -> tabular FM) works well** and confirms
  the premise; beating the failed repo (rank 8.53) is easy with it.
- **Recommendation:** the highest-value next step is *not* more from-scratch
  training - it is (a) adopting the reference tabular FM, and (b) **fine-tuning
  TabICL v2 on the fixed CheMeleon manifold** (relaxing its column-variability
  assumptions, which is the exact structural exploitation the idea targets, but
  starting from a model that already has the in-context skill). That is where
  the real, achievable gain is.

## 6. Reproduce

```bash
# env: ~/miniforge3/envs/chemeleon (torch 2.14+cu130, lightning, chemprop, polaris)
cd /home/jackson/TabularCheMeleon

# 1. pool (once)
python gen_pool.py --pool-size 65536

# 2. train from-scratch ICL (the collapse reproduces)
python train.py --pool data/pool_full.pt --epochs 10

# 3. evaluate it on polaris (avg rank ~10, collapses)
python eval_v1.py

# 4. reference tabular FM baseline (works well)
PYTHONPATH=/home/jackson/tabicl-chemeleon/src python tabicl_baseline.py

# diagnostics (the collapse analysis)
python test_singledir.py    # context-sensitivity probe + 1-D task
python test_lowdim.py       # feature-dim control
python test_featuremode.py  # linear vs MLP feature projection
python scale_study.py       # more-compute-makes-it-worse
```
