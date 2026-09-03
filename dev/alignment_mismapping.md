# Is the alignment-built `M` equivalent to the simulate-and-map one?

**Yes, on this reference set — and it is closer to the measurement than the measurement is
to itself.** `‖M_align − M_sim‖_F = 0.430` against a reseed noise floor of `0.593`. The
alignment kernel costs **0.011 s** where simulate+map costs 5.2 s (**460×**) and needs no
mapper, no container and no error model.

Reproduce with `python dev/alignment_mismapping.py` (needs docker + torch/pyro, ~5 min).
Raw numbers in `alignment_mismapping.csv`.

Step 3 of [docs/alignment_mismapping_plan.md](../docs/alignment_mismapping_plan.md); step 1 was
[amplicon_distance_census.md](amplicon_distance_census.md).

## Setup

Same 21-genome / 81-amplicon B. uniformis V4 set, same truth (the confusable pair at
3% / 17%), same 50k observed reads and 300 simulated reads per reference as
[error_rate_sensitivity.md](error_rate_sensitivity.md) — whose `mapseq`-in-docker wrapper,
read simulators and Pyro fit this script imports rather than re-implements.

`M_sim` is the pipeline default (flat 0.005, seed 1). Its precision is the same model
re-simulated at seed 99.

## Matrix space

| candidate | ‖·−M_sim‖_F | beyond one run's noise | mean diag | TV med / max | support J | ρ off-diag |
|---|---|---|---|---|---|---|
| reseed (flat 0.005, seed 99) | **0.593** *(the floor)* | 0.420 *(its own noise)* | 0.3037 | 0.063 / 0.140 | 0.726 | 0.844 |
| trained error model | 0.593 | 0.419 *(its own noise)* | 0.3107 | 0.063 / 0.120 | 0.594 | 0.781 |
| **alignment tau=0** | **0.430** | **0.093** | 0.3086 | **0.047 / 0.093** | 0.584 | 0.776 |
| alignment tau=1 | 2.100 | 2.058 | 0.2526 | 0.057 / 0.817 | 0.646 | 0.809 |
| alignment tau=2 | 2.205 | 2.165 | 0.2469 | 0.057 / 0.824 | 0.648 | 0.809 |
| shuffled control | 5.795 | 5.780 | **0.2915** | 0.813 / 0.917 | 0.024 | −0.010 |

*"Beyond one run's noise"* decomposes the distance: the floor compares two noisy
measurements, so `floor² ≈ 2·noise²` and a **noiseless** estimator that agreed perfectly
would still score `floor/√2 = 0.420`, not zero. The column is
`√(F² − floor²/2)` — for the noiseless alignment matrix that is its systematic difference
from the measurement (0.093); for the two simulated candidates it just returns their own
sampling noise (0.42), since they have no systematic difference from `M_sim` by
construction.

## Downstream

| generator | candidate | L1 | err *b. uniformis* | err *BU_JCM13286* |
|---|---|---|---|---|
| trained | *(none: naive observed)* | 0.0215 | +0.0055 | −0.0038 |
| trained | simulate+map (flat 0.005) | 0.0191 | +0.0023 | −0.0020 |
| trained | **alignment tau=0** | **0.0158** | **−0.0003** | **+0.0002** |
| trained | shuffled control | 0.0529 | +0.0086 | +0.0033 |
| flat 1% | *(none: naive observed)* | 0.0279 | +0.0073 | −0.0069 |
| flat 1% | simulate+map (flat 0.005) | 0.0241 | +0.0041 | −0.0049 |
| flat 1% | **alignment tau=0** | **0.0194** | **+0.0008** | **−0.0019** |
| flat 1% | shuffled control | 0.0508 | +0.0106 | −0.0001 |

## Reading it

1. **It passes, with room.** 0.430 is below the 0.593 floor, and the part that isn't
   sampling noise is 0.093 — a systematic difference **4.5× smaller than the noise a
   single simulate+map run contributes**. Per-row total variation is also *lower* than the
   reseed's (median 0.047 vs 0.063, max 0.093 vs 0.140): there is no row where the
   alignment matrix is unusually wrong.
2. **The mean diagonal cannot tell you this, and step 1 was not sufficient.** The shuffled
   control scores `mean diag = 0.2915` — indistinguishable from the measured 0.30 — while
   being wrong in every way that matters (F = 5.8, support Jaccard 0.024, ρ ≈ 0, L1 twice
   the naive baseline). Shuffling preserves the distance multiset and therefore the cluster
   size distribution. The census confirmed the *mechanism*; only this comparison confirms
   the matrix.
3. **The control has teeth**, which is what makes 1 believable: every metric fails it, and
   its downstream fit (L1 0.051–0.053) is worse than applying no correction at all.
4. **Where alignment is weakest: the low-identity tail.** Support Jaccard 0.584 and
   ρ = 0.776 are the two metrics where it sits below the reseed (0.726 / 0.844) — it lands
   exactly where the *trained* model sits (0.594 / 0.781), i.e. at the level of a
   genuinely different error model rather than a re-run of the same one. A hard tie cluster
   emits no mass to a reference 10 bases away; mapseq occasionally does. That mass is small
   enough not to move `‖·‖_F` or the fit, and it is the first thing to look at if a future
   reference set fails.
5. **tau > 0 fails, decisively.** F jumps to 2.1 and TV max to 0.82: a handful of rows are
   completely wrong while the median row is fine. Merging references one base apart is not
   a softer version of the truth, it is a different and wrong claim. tau=0 is not a fitted
   value — there was nothing to fit (plan §5.4 is moot).
6. **Downstream it is at least as good, and best on the pair the pipeline exists for.**
   L1 0.0158 / 0.0194 vs 0.0191 / 0.0241 for simulate+map, and the confusable pair comes
   back at −0.0003 / +0.0002 and +0.0008 / −0.0019 — the best of any candidate under both
   generators. Do not over-read the ordering: L1 wobbles by ~0.003 between runs (mapseq is
   not bit-reproducible across threads, which also moves the floor by ~0.006), so the
   honest claim is "indistinguishable from, and never worse than, simulate+map".
7. **460× cheaper, and that undersells it.** 0.011 s vs 5.2 s excludes what the alignment
   path also removes: the `mapseq -mscluster` build, the container pull, and — under
   `--sim_error_model trained` — the entire skiver training subworkflow.

## Align mode's error, by backend and condition

`align` builds `M` from **whole-reference** alignments only. Two backends produce those
distances: `edlib` (in-process, lossless k-mer filter, IUPAC-aware) and `minimap2` (the
binary in its biocontainer, index built once). Reproduce with
`python dev/alignment_mismapping.py` and `--inject-n 0.25`.

| condition | candidate | ‖·−M_sim‖_F | systematic part | mean diag | TV med / max | support J | ρ |
|---|---|---|---|---|---|---|---|
| clean DB *(floor 0.604)* | reseed | 0.604 | 0.427 | 0.3087 | 0.060 / 0.127 | 0.709 | 0.833 |
| clean DB | **align edlib** | **0.462** | **0.176** | 0.3086 | 0.050 / 0.113 | 0.590 | 0.780 |
| clean DB | **align minimap2** | **0.462** | **0.176** | 0.3086 | 0.050 / 0.113 | 0.590 | 0.780 |
| 27% ambiguous *(floor 0.536)* | reseed | 0.536 | 0.379 | – | – | – | – |
| 27% ambiguous | **align edlib** | **1.421** | **1.406** | – | – | – | – |
| 27% ambiguous | **align minimap2** | **5.104** | **5.097** | – | – | – | – |
| 27% ambiguous | edlib, ambiguity = mismatch | 5.104 | 5.097 | – | – | – | – |

Downstream (clean DB), composition L1 against truth: naive observed 0.0229 / 0.0255,
simulate+map 0.0177 / 0.0206, **align 0.0156 / 0.0191 on either backend**, shuffled control
0.0524 / 0.0499. The confusable pair comes back at −0.0002 / +0.0002 and +0.0007 / −0.0018.

1. **On a clean reference set the two backends are indistinguishable** — identical to four
   decimal places on every metric, because both resolve the same tie clusters. The choice
   between them is cost and scaling, not accuracy.
2. **On an ambiguous reference set they are not.** `align minimap2` scores **exactly** what
   the pre-fix "ambiguity = mismatch" kernel scored (5.097): minimap2's `NM` counts an `N`
   as a mismatch, so an `N`-bearing reference is split out of the cluster it belongs to.
   The IUPAC handling that fixes this lives in edlib's `additionalEqualities` and has no
   equivalent in a PAF. 68 pairs disagree on `d ≤ 0`. The estimator now warns when it is
   given a PAF and the references carry ambiguity codes.
3. **Cost at this size favours edlib**: 0.10 s vs 0.60 s at 81 references, the difference
   being container startup. minimap2 wins only where indexing pays — see
   [the backend benchmark](alignment_backend_benchmark.md).
4. **Read windows are not modelled at all, on purpose.** A query shorter than the reference
   sees a window of it and is confusable in ways whole-sequence distance cannot represent.
   Rather than approximate it, `align` refuses `--sim_read_len` and the pipeline directs
   short/unmerged reads to `simulate`, which measures it. Paired samples are merged into
   whole-fragment queries first, so this is the exception rather than the norm.

**Choosing a backend:** `edlib` unless the reference set is large enough that indexing
pays; `edlib` regardless if it carries IUPAC ambiguity.

## IUPAC ambiguity in the references (`N`)

Draft-genome 16S carries ambiguity codes. edlib scores them as plain mismatches by
default, so a single `N` in one copy of an otherwise-identical pair splits the tie cluster
and hands *both* references identity rows. `build_mismapping_align.py` now passes
`additionalEqualities` so two symbols match when the base sets they stand for overlap.

Probed by injecting one `N` at a random position into 22 of the 81 reference amplicons —
**in the mapseq DB only**, with reads still simulated from the un-`N`'d sequences, since a
reference's ambiguity is an assembly artefact and the organism has a real base there
(`python dev/alignment_mismapping.py --inject-n 0.25`, numbers in
`alignment_mismapping_ambiguity.csv`).

| candidate | ‖·−M_sim‖_F | mean diag | TV max | support J |
|---|---|---|---|---|
| reseed *(floor)* | **0.539** | 0.3047 | 0.160 | 0.769 |
| **alignment tau=0 (ambiguity matches)** | **2.450** | 0.3086 | 0.663 | 0.705 |
| alignment tau=0 (*ambiguity = mismatch*, pre-fix) | 5.111 | 0.5556 | 1.000 | 0.376 |

1. **The bug is real and large.** Scoring ambiguity as a mismatch drives the mean diagonal
   to 0.556 against a measured 0.305 — it invents distinctions mapseq does not make — with
   TV max 1.0 (rows completely wrong) and support Jaccard 0.376, barely above the shuffled
   control's neighbourhood. It breaks the cluster of **57 of 81** references.
2. **The fix recovers the clustering exactly.** With the equalities, all 81 cluster sizes
   equal those of the un-`N`'d reference set: no spurious splits, and no spurious merges
   either (0 pairs at clean-distance 1 collapse to 0). At the sequence level this is the
   right answer, and `‖·‖_F` drops from 5.111 to 2.450.
3. **It still does not pass, and the reason is mapseq, not the kernel.** The mapper is
   *not* ambiguity-agnostic: an `N`-bearing reference receives **0.19×** the incoming mass
   of its clean cluster partners (57 references in mixed clusters) — one mismatch is enough
   for a clean duplicate to win the read. Directly: `‖M_measured(N-DB) −
   M_measured(clean-DB)‖_F = 2.481`, which is essentially the whole residual 2.450. The
   equalities kernel reproduces the *clean* matrix perfectly; the gap is entirely the
   mapper's `N`-penalty, which no ambiguity-agnostic distance can express.
4. **The mean diagonal is blind to all of it again.** Measured 0.3063 with `N`s vs 0.3056
   without, while the matrices differ by 2.481 — mass moves *within* clusters, which the
   diagonal cannot see. Third time this metric has been uninformative.

5. **Modelling the penalty helps, and is now the default.** Down-weighting a cluster
   member as `w ** (its ambiguous positions)` recovers ~40% of the residual
   (2.455 → 1.351 at this injection). The fitted `w` matches the independently measured
   mass ratio in 5 of 5 configurations — but that ratio itself ranges 0.18–0.97 across
   draws, so `--align_ambiguity_weight` defaults to 0.3 as a compromise rather than an
   optimum, and is a no-op on ambiguity-free sets. Details and the sweep:
   [ambiguity_weight_sweep.md](ambiguity_weight_sweep.md).

**Practical reading:** the fix is necessary and removes a catastrophic failure mode, and
the penalty model recovers much of what is left — but at best `‖·‖_F` is 1.35–1.88 against
a ~0.58 floor. On a DB where a meaningful fraction of references carry ambiguity, `align`
and `simulate` genuinely disagree: use `simulate`, or drop/repair the ambiguous
references.

## Does a low-identity tail help?

The tie cluster puts *exactly zero* mass on a reference a few bases away, and step 3 found
that to be its one measurable weakness (support Jaccard 0.588 against a floor of 0.737).
The obvious fix is to give it a tail. With full distances available — the k-mer filter
supplies them by raising its bound — that is directly testable.

First, what the measurement's tail actually looks like: **0.0135 mass per row, 198 entries
above 1e-3 out of 6198 possible off-cluster pairs**, at median distance 10 (90th percentile
17, max 52). Sparse and mid-range, not a smooth decay.

| kernel | ‖·−M_sim‖_F | mean diag | support J | ρ off-diag |
|---|---|---|---|---|
| reseed *(floor)* | 0.593 | 0.3082 | 0.737 | 0.852 |
| tie cluster tau=0 | **0.462** | 0.3086 | 0.588 | 0.778 |
| softmax `exp(-2.0 d)` | 0.872 | 0.2885 | 0.658 | 0.443 |
| softmax `exp(-0.5 d)` | 1.847 | 0.2514 | 0.694 | 0.442 |
| tie + 0.02 mass over d≤5 | **0.455** | 0.3069 | **0.702** | **0.834** |
| tie + 0.02 mass over d≤20 | 0.454 | 0.3050 | 0.569 | 0.622 |
| tie + 0.20 mass over d≤5 | 0.638 | 0.2914 | 0.702 | 0.834 |

1. **A smooth exponential tail is the wrong shape.** Every β makes `‖·‖_F` worse, and ρ
   collapses to 0.44 — softmax orders all 6000 off-diagonal pairs by distance when the
   truth is that ~97% of them are exactly zero.
2. **A small, tightly truncated tail does help** on the metrics that flagged the problem:
   support Jaccard 0.588 → 0.702 and ρ 0.778 → 0.834, most of the way to the floor's
   0.737 / 0.852. The `‖·‖_F` gain (0.462 → 0.455) is within run-to-run noise.
3. **But it does not survive being checked against a different measurement.** Against
   `M_sim` built from the *trained* error model, the plain tie cluster scores support
   Jaccard **0.966** and ρ **0.983** — nearly perfect — and adding the tail drops it to
   0.730 / 0.859. Against the flat-0.005 reseed it helps (0.583 → 0.658); at read-len 150
   it hurts slightly (0.695 → 0.684).

**So the tail was not shipped**, and step 3's "missing tail" limitation is now better
understood: it is not a fixed property of the kernel but a property of *which error model
built the reference measurement*. A flat 0.5% substitution rate scatters more reads onto
distant references than the trained model does, so it has a fatter tail for the kernel to
miss. Fitting a tail means fitting the simulator's error rate — the exact thing the
alignment method exists to avoid.

## What this does not show

One reference set, and the easiest kind: 73 of its 81 references are exact duplicates.
Point 4 says where the model is thin, and point 5 says the kernel has no knob to absorb a
set that is *near*-identical rather than identical — the binomial-race kernel (plan §4
rung 4) is the designed answer there, and remains unbuilt and untested. Before making this
the pipeline default rather than an option, run this same study on a reference set whose
members differ by one or two bases.
