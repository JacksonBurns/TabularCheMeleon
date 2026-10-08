# Fixed-Dim In-Context Foundation Model over CheMeleon Embeddings

## Critical evaluation of the research idea

**The idea (as stated).** A recent result showed that passing frozen CheMeleon
2048-dim embeddings through a tabular foundation model (TabPFN) and doing
in-context prediction *beat* fine-tuning CheMeleon directly on small datasets.
Tabular FMs are built to handle an *arbitrary number of columns in an
arbitrary order*. CheMeleon embeddings are a *fixed* 2048-dim vector in a
*fixed* order. Hypothesis: relax those architectural assumptions and
scratch-train a model that exploits the known fixed structure to improve
performance.

Below is a critical assessment. The short version: **the instinct is right,
the diagnosis is slightly off, and the fix is smaller and cleaner than it
first appears.** The exploitable structure is not "the order" - it is "the
fixed, consistent, *correlated* dimensionality" - and the mechanism that
exploits it is a single shared dense projection, not order-aware attention.

---

## 1. What CheMeleon's 2048-dim vector actually is

This matters more than it first looks. CheMeleon is a message-passing GNN.
Its output is `MeanAggregation` over atom hidden states, then a **linear
readout** `W_out ∈ R^{2048 × d_h}` applied to the pooled vector `h ∈ R^{d_h}`:

```
x (2048-dim) = W_out @ h
```

Three consequences follow directly:

1. **Redundant / low effective rank.** The 2048 dims are *linear functions of
   a d_h-dim vector* (`d_h` ≈ 256-512). So the 2048-dim embedding occupies a
   ≤ d_h-dimensional linear subspace. It is a *correlated*, low-effective-
   dimensionality manifold, **not** 2048 independent features.
2. **Consistent across molecules.** The same `W_out` is used for every
   molecule, so dim `i` is *always* the same linear direction. This is what
   makes a *single shared projection* `W: R^{2048} -> R^d` valid.
3. **No meaningful *order*.** Dim 3 is not "closer" to dim 4 than to dim
   2045. The rows of `W_out` are arbitrary linear directions. Permuting the
   2048 dims gives an *equally valid* embedding (just a reparametrization).

Point 3 is the subtle one and it **reframes the user's premise.** The data is
not "in a fixed order that we should exploit." The data is *permutation-
equivalent* - the specific order carries no signal. What carries signal is
that the dims are **fixed in count, consistent in identity, and correlated in
value**. "Fixed order" is a *necessary enabler* (it makes a shared `W`
valid); it is **not the source of the gain**.

## 2. Does a tabular FM actually "throw away" the fixed structure?

Partly - and the user is right that there is a real mismatch, but the
mechanism is different from "arbitrary order."

TabICL/TabPFN do **not** treat a 2048-dim row as 2048 free columns. Each
*row* is one example (one token); the 2048 dims are the *features* of that
example. Per feature, TabICL applies a **numeric value encoder** (bucketing /
quantile binning) **plus a per-column learned embedding**. So:

- It *can* specialize per dimension (via 2048 distinct column embeddings), so
  it is not strictly permutation-invariant - it is *order-tolerant*.
- **But** each dimension's *value* is encoded **independently** (its own
  numeric encoder), and the dimensions are assumed to be roughly
  **exchangeable / independent**. That is a **bad prior for a correlated,
  low-rank 2048-dim vector.**

The concrete disconnect, stated precisely:

> **Tabular FMs encode each feature independently and assume features are
> approximately independent. CheMeleon embeddings are strongly correlated
> (a linear lift of a low-dim vector). A model that exploits this
> correlation - a single *dense* linear projection that can mix all 2048
> dims jointly - has a structural advantage a per-column tabular FM
> structurally cannot match.**

That is the real, defensible version of the user's idea. It is *not* "exploit
the order"; it is "exploit the correlation with a joint projection."

## 3. The pre-training distribution disconnect (the deeper one)

TabPFN/TabICL have no separate "pre-train on real data" phase. Their
*pre-training is the synthetic-data generation*: millions of random tables
(i.i.d. Gaussian-ish columns) with random SCM-generated targets, and the
model learns **the skill of in-context regression**. Two mismatches with our
setting:

**(a) Input distribution.** Synthetic columns are independent Gaussians.
CheMeleon embeddings are correlated low-rank. Pre-training a tabular FM on
Gaussians bakes in an "independent columns" prior that is wrong at inference.
**Fix:** pre-train on the *real* frozen embedding manifold (a pool of real
CheMeleon embeddings). This is what `train.py` does with the 65K-molecule
pool.

**(b) Target distribution.** Synthetic SCM targets are one family (linear,
trees, small MLPs). Real ADMET/kinase SAR is another. **Fix:** pre-train on a
**broad, diverse family of smooth functions** over the embedding manifold.

**(c) The key reframing - teach the *skill*, not the *content*.** This is
where the previous (failed) repo went wrong, and it is the most important
point. On *small* datasets, fine-tuning a 17M-param GNN overfits; an ICL
model that has *already learned how to do few-shot in-context regression*
generalizes better because the specific SAR is learned **at inference time
from the context examples**, not in pre-training. Therefore the synthetic
targets must be **broad enough that the ICL *mechanism* transfers** - they do
*not* need to match real SAR.

The failed repo tried to encode *specific SAR content* in pre-training
(cosine-GP pockets, Hill kinetics, Free-Wilson, scaffold-GAM - four hand-
crafted chemical motifs). That is the **wrong level of abstraction**: it
narrowed the task family instead of broadening it, and it coupled the prior
to a cosine-similarity structure that didn't match how the model consumes the
tokens. Combined with a **Perceiver bottleneck** (lossy context compression)
that underfits at this scale, the result was rank **8.53** - *worse* than
plain CheMeleon and worse than TabPFNv2-rdkit on almost every benchmark.

**Principle:** pre-training should maximize *diversity of smooth function
forms* (so the model learns to infer arbitrary smooth structure from few
examples), **not** chemical specificity. My `prior.py` implements a broad
mixture (linear / additive random projections / RBF-kernel GP / random 2-layer
MLP), all *smooth* in embedding space, standardised per task.

## 4. Should we use a *different* architecture that "doesn't introduce these
assumptions in the first place"?

**No - keep the in-context-transformer paradigm. It is the right one.** The
assumptions to drop are simply *not in the design*; we don't need a new
architecture, we need to *remove* the column machinery:

| Component | TabPFN/TabICL | Fixed-dim ICL model (this work) |
|---|---|---|
| Feature encoding | per-column numeric encoder + per-column embedding (independent) | **single shared `Linear(2048->d)` + GELU + `Linear`** (joint, captures correlation) |
| Column-index / permutation handling | yes (order-tolerant) | **none** (no concept of "columns") |
| Categorical/numerical dispatch | yes | **none** (all continuous) |
| Variable column count / padding | yes | **none** (fixed 2048) |
| Label embed + mask token + present flag | yes | **keep** |
| Positional encoding over *examples* | yes | **keep** |
| Pre-LN transformer over the context window | yes | **keep** |
| In-context (no per-task update) | yes | **keep** |

The *only* structural change to the TabPFN row-architecture is: **replace the
per-column encoders with one shared dense projection.** Everything else is
identical in spirit. This is the minimal, clean realization of "relax the
column assumptions, keep the ICL architecture."

**On scaling to very large context.** The plain self-attention transformer is
O(N²) and comfortable up to N ≈ 256-512 on a 24 GB GPU. Polaris train sets
range ~50 to ~1500. For the small/medium benchmarks (the majority), full
context fits. For the large ones we subsample context (a real limitation).
The *clean* extension for N ≫ 512 is a **Perceiver-resampler front-end**
(compress N context tokens to M latents) **combined with the shared-projection
tokenization** - i.e., keep the Perceiver *for scale* but fix the *two* things
that were wrong in the failed repo (tokenization and prior). That is an
orthogonal scaling concern, not the core idea, and is out of scope for the
validation run.

## 5. Honest risks / ceiling

1. **Synthetic-to-real gap is the ceiling.** The model learns "in-context
   few-shot regression for smooth functions of 2048-dim vectors on this
   manifold." It does not learn *real* SAR. Its ceiling is set by how well
   that *skill* transfers - the same ceiling CheMeleon+TabPFN has. We should
   expect to beat **fine-tuned CheMeleon on small data** and the **failed
   repo**, and to *approach* (not necessarily beat) TabPFNv2-rdkit. Beating
   a fine-tuned GNN on *large* data is not the claim.
2. **Frozen embeddings cap the ceiling.** If real SAR needs 3D information
   CheMeleon flattened away, we are capped at CheMeleon's representational
   quality. We are exploiting the *embedding*, not re-learning chemistry.
3. **Context subsampling** on the largest benchmarks loses information.
4. **Redundancy:** since the 2048 dims are a low-rank lift, the shared
   projection will effectively learn a low-dim subspace - expected and fine,
   but it means the "2048" is not 2048 independent channels of signal.

## 6. Verdict

- **Is there a disconnect?** Yes, two: (a) input distribution (independent
  synthetic columns vs. correlated low-rank embeddings) and (b) target
  distribution (SCM family vs. broad smooth family). Plus a structural one:
  per-column independent encoding vs. joint dense projection.
- **How is it closed?** Pre-train on the *real* frozen embedding manifold with
  a *broad, diverse, smooth* synthetic target family (teach the skill), and
  tokenize each example with a *single shared dense projection* (exploit the
  correlation).
- **What architectural changes?** Remove column-index / permutation / type /
  variable-width machinery; add one shared `Linear(2048->d)`. Keep the ICL
  transformer, label/mask embedding, positional encoding, and in-context
  inference. No new architecture needed.
- **Different architecture?** No. The ICL transformer is correct; just strip
  the column assumptions. Add a Perceiver resampler *later* only for very
  large context.

This is a well-motivated, testable idea. The version implemented here is the
cleanest realization of it.
