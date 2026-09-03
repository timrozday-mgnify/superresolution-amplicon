# Which all-vs-all alignment backend should build `M`?

**A lossless k-mer filter, keeping edlib as the distance oracle.** It removes 10–12× of
the work, drops nothing by construction, and adds no dependency. minimap2 (via `mappy`) is
a further ~3× at n = 8000 and had **recall 1.000** on the real reference set — but it is a
heuristic, and the exact filter already removes the ceiling for the sets this pipeline
builds.

Reproduce with `python dev/alignment_backend_benchmark.py` (needs `mappy` for the
comparison rows). Raw numbers in `alignment_backend_benchmark.csv`.

## The problem

The tie-cluster kernel needs, per reference, the set of references within `tau` edit
operations — naively n² alignments. Two ways to skip most of them.

## What was tried

| backend | what it is | verdict |
|---|---|---|
| **k-mer filter + edlib** | q-gram lemma: a length-`L` sequence has `L-k+1` k-mers and one edit destroys at most `k`, so two sequences within `e` edits share ≥ `L-k+1-k*e`. Anything sharing fewer is skipped without alignment; survivors are aligned with edlib's `k=` bound so it also gives up early *inside* an alignment. | **Shipped.** Provably lossless, no new dependency. |
| `mappy` (minimap2) | Real minimizer index, `asm5`/`asm10`/`sr`/`map-ont` presets. | **Measured, not shipped.** See below. |
| plain edlib all-pairs | What was there before. | Kept as the exact path (`max_distance=None`) for the census histogram. |

## minimap2 as a prefilter, on the real 81-amplicon set

| preset | recall | precision | clusters exactly right | time |
|---|---|---|---|---|
| asm5 | **1.000** | 0.901 | 65/81 | 0.018 s |
| asm10 | **1.000** | 0.833 | 54/81 | 0.015 s |
| sr | **1.000** | 0.427 | 18/81 | 0.024 s |
| map-ont | **1.000** | 0.821 | 52/81 | 0.021 s |

Recall 1.000 everywhere, precision 0.43–0.90: minimap2 never missed a true distance-0
pair but returns extra candidates. That is a **prefilter** profile, not a distance oracle —
its output would still have to be confirmed by alignment, so it competes with the k-mer
filter, not with edlib.

## Cost against reference-set size

Synthesised from the real amplicons in the same duplicate-heavy shape, since that is what
makes the problem hard.

| n | edlib all-pairs | k-mer filter + edlib | speed-up | mappy asm5 (candidates only) |
|---|---|---|---|---|
| 500 | 3.71 s | 0.37 s | **10.0×** | 0.37 s |
| 2000 | 61.20 s | 5.17 s | **11.8×** | 4.70 s |
| 8000 | *~16 min (projected)* | 82.21 s | ~12× | 25.19 s |

## Reading it

1. **The filter is free correctness.** Bounded distances equal exact ones (clipped to the
   bound) on the real set at every `tau` tried, and `M` is byte-identical to the validated
   matrix. It is a pure cost change, asserted in the estimator's `--demo`.
2. **10–12×, and still quadratic-ish.** The synthetic sets are deliberately
   duplicate-heavy, so most references genuinely *have* many neighbours and the candidate
   lists stay long. The filter removes the constant, not the exponent.
3. **minimap2 is ~3× better again at n = 8000** and would scale further. It is the
   documented upgrade path, not the default: it is heuristic (recall 1.000 is a
   measurement on one reference set, not a guarantee), it adds a dependency, and the
   reference sets this pipeline builds are per-sample genome collections — 81 amplicons
   here, where the whole operation is 0.01 s.
4. **This changes cost, not accuracy.** None of the known limitations in
   [the plan](../docs/alignment_mismapping_plan.md) are addressed by it. One tension worth
   recording: the bound deliberately discards distances beyond `tau`, while a future
   soft-tail kernel (plan rung 3, the fix for the under-dispersion bias) would need
   exactly those distances. They pull in opposite directions.

## What this does not show

Ambiguity is charged like an edit in the bound (its k-mers are skipped), which is
conservative but untested past the injected-`N` sets. The k-mer size (15) was not tuned —
it is short enough that a 250bp amplicon has k-mers to spare against the bound and long
enough to be specific, and no sweep was run. Memory is O(total k-mers); a minimizer sketch
would trade a weaker bound for less of it, and was not needed here.
