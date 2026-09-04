# Which all-vs-all alignment backend should build `M`?

**Both are shipped.** `--align_backend edlib` (default) aligns in-process behind a
lossless k-mer filter — 10–12× less work, drops nothing by construction, no container.
`--align_backend minimap2` runs the binary in its biocontainer with an index built once,
which scales better and is a real mapper's own view of which references look alike. They
give **identical `M`** on the reference set tested. `mappy` was evaluated and rejected: it
exposes `-N` but not `-p`, and cannot reach the settings that matter.

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

## What minimap2 actually returns, by edit distance

A single recall number here is misleading: 282 of the 6198 true off-diagonal pairs are at
distance 0, and those are the easy ones. Broken out — true pairs found, per band, on the
real 81-amplicon set:

| preset | d=0 | d=1 | d=2-5 | d=6-20 | d=21-60 | d>60 |
|---|---|---|---|---|---|---|
| asm5 N=500 | 282/282 | 38/38 | **6/68** | **0/846** | 0/2306 | 0/2940 |
| asm10 N=500 | 282/282 | 38/38 | 66/68 | **0/846** | 0/2306 | 0/2940 |
| **sr** N=500 | 282/282 | 38/38 | 68/68 | **822/846** | 54/2306 | 0/2940 |
| map-ont N=500 | 282/282 | 38/38 | 68/68 | **0/846** | 0/2306 | 0/2940 |
| asm5 N=5 | 282/282 | 33/38 | 4/68 | 0/846 | 0/2306 | 0/2940 |

**minimap2 is not keeping all alignments, and `-N` is not the binding constraint.** It
keeps the top `-N` hits *and* drops anything scoring below `-p` (secondary-to-primary
ratio, default 0.8) of the primary — and `mappy` exposes `-N` but not `-p`. Comparing
`N=500` with `N=5` shows `-N` binding only slightly (38 vs 33 pairs at d=1); the presets'
score and chaining thresholds do the rest. `sr` is by far the most permissive and still
stops around d≈20.

## minimap2 run directly, with the parameters mappy hides

`mappy` cannot set `-p`, so the same measurement was repeated against the minimap2 binary
in its biocontainer (`quay.io/biocontainers/minimap2`), which is how the pipeline now runs
it (`--align_backend minimap2`):

| flags | d=0 | d=1 | d=2-5 | d=6-20 | d=21-60 | d>60 |
|---|---|---|---|---|---|---|
| `-x asm5 -p 0 -N 1000` | 282/282 | 38/38 | 6/68 | 0/846 | 0/2306 | 0/2940 |
| `-x sr -p 0 -N 1000` | 282/282 | 38/38 | 68/68 | **846/846** | 372/2306 | 0/2940 |
| `-x ava-ont -p 0 -N 1000` | 141/282 | 19/38 | 34/68 | 423/846 | 37/2306 | 0/2940 |
| **no preset, `-k 11 -w 5 -p 0 -N 1000`** | 282/282 | 38/38 | 68/68 | **846/846** | **1658/2306** | **1478/2940** |

**The preset, not `-p`/`-N`, was the binding constraint.** `asm5` is unchanged by
`-p 0 -N 1000` — it assumes ≤5% divergence and its chaining thresholds discard the rest
regardless. Dropping the preset and shrinking the minimizers (`-k 11 -w 5`) recovers
everything out to d=20 and most of d=21-60. Even so it is not complete: at >24% divergence
there are too few shared minimizers to seed a chain, and 1478/2940 of the d>60 pairs come
back. Those are not plausible mis-mapping targets anyway.

Those flags are the shipped defaults for `--align_backend minimap2`, and the resulting `M`
is **identical to edlib's** on the real reference set. The index is built once
(`MINIMAP2_INDEX`, mirroring `MAPSEQ_CLUSTER`) so a large reference DB is not re-indexed
on every all-vs-all pass; `-k`/`-w` live in the `.mmi` and are rejected in
`--minimap2_args`, because minimap2 silently ignores them there.

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
   documented upgrade path, not the default: it is heuristic (finding every d=0/d=1 pair
   is a measurement on one reference set, not a guarantee), it adds a dependency, it
   cannot return the tail from Python, and the reference sets this pipeline builds are
   per-sample genome collections — 81 amplicons here, where the whole operation is 0.01 s.
4. **This changes cost, not accuracy** — and the tail it could have supplied turns out
   not to help. Having the full distance matrix is not the missing ingredient for the
   under-dispersion bias; see
   [the tail experiment](alignment_mismapping.md#does-a-low-identity-tail-help). One
   mechanical note: `build()` bounds distances at `max(tau, 5)`, so a kernel that ever
   does want the tail must ask for `pairwise_distances(..., max_distance=None)`.

## What this does not show

Ambiguity is charged like an edit in the bound (its k-mers are skipped), which is
conservative but untested past the injected-`N` sets. The k-mer size (15) was not tuned —
it is short enough that a 250bp amplicon has k-mers to spare against the bound and long
enough to be specific, and no sweep was run. Memory is O(total k-mers); a minimizer sketch
would trade a weaker bound for less of it, and was not needed here.
