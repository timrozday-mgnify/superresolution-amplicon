# Can pairwise amplicon distance alone predict `M`?

**Yes, for this reference set.** A parameter-free tie-cluster kernel over exact amplicon
duplicates predicts `mean diag(M) = 0.3086`. The measured value from simulate+mapseq is
`0.3064` (trained model) / `0.3093` (the same model re-seeded). The prediction lands
*between the two seeds of the measurement it is trying to reproduce* — with no simulation,
no mapper and no free parameters.

Reproduce with `python dev/amplicon_distance_census.py` (needs `edlib`; ~2 s, no docker).
Raw numbers in `amplicon_distance_census.csv` and `amplicon_distance_census_per_ref.csv`.

This is step 1 of [docs/alignment_mismapping_plan.md](../docs/alignment_mismapping_plan.md).

## Setup

The same 21-genome / 81-amplicon B. uniformis V4 set as
[error_rate_sensitivity.md](error_rate_sensitivity.md). In-silico PCR via the pipeline's
own `extract_v4`, then all-pairs global (Needleman-Wunsch) edit distance with `edlib`.

## The reference set is almost entirely redundant

- 81/97 DB entries amplifiable, 21 genomes, amplicon length 252–254 (spread **2 bases**).
- **73 of 81 references have an exact duplicate** over the amplicon. Only 8 are unique.
- Cluster sizes at d=0: 8 singletons, 4 pairs, 2 triples, 1 quad, 5 quintuples, 5 sextuples.
- Nearest *other* reference: 73 refs at distance 0, 4 at 1, 2 at 3–5, 2 at >10. There is
  essentially no middle ground — references are identical or clearly distinct.

## Predicted mean diagonal

| tau (bases) | mean diag | median | unambiguous refs | mean cluster | vs measured 0.30 |
|---|---|---|---|---|---|
| **0** | **0.3086** | 0.200 | 10% | 4.48 | **+0.002 vs trained, −0.001 vs reseed** |
| 1 | 0.2526 | 0.200 | 5% | 4.95 | −0.05 |
| 2 | 0.2469 | 0.200 | 5% | 5.02 | −0.06 |
| 3 | 0.2469 | 0.200 | 5% | 5.02 | −0.06 |
| 5 | 0.2222 | 0.167 | 2% | 5.79 | −0.08 |

Measured spread for reference: 0.2854 (flat 15%) to 0.3104 (flat 0.0001), trained 0.3064,
trained reseeded 0.3093.

## Reading it

1. **tau = 0 wins, and widening the cluster makes it worse.** Every non-zero tolerance
   undershoots. Mis-mapping here is *exact* redundancy — a one-base difference is already
   enough for mapseq to tell two references apart at these error rates. This is the same
   conclusion the error-rate study reached from the other direction.
2. **The knobs in the plan (tau, softmax beta, the binomial race) have nothing to do.**
   Rung 1 of the kernel ladder is the answer for this DB; rungs 2–4 exist for a set with
   near-misses, and this set has almost none (4 references at distance 1, 0 at distance 2).
3. **Orientation is already normalised.** Minimum forward inter-reference distance 0;
   minimum reverse-complement distance 130. `extract_v4` handles strand, so the estimator
   does not need to align both.
4. **Alignment parameters are unlikely to matter.** With a 2-base length spread and a
   distance distribution that is 90% zeros, no choice of gap penalty or scoring matrix can
   move the argmin sets. The §3 sweep of the plan can be reduced to a confirmation.
5. **The hard pair is exactly one shared copy.** `bacteroides_uniformis` has a single
   amplifiable 16S copy, and it is *identical* to one of `BU_JCM13286_NT5170`'s four. The
   tie-cluster kernel gives that reference a 0.5/0.5 row — which is the whole inference
   problem, stated in one line of arithmetic.

## What this does not show

The mean diagonal is one scalar. Matching it is necessary, not sufficient: the off-diagonal
mass could still be distributed wrongly, and the low-identity tail (mapseq assigning a read
to something 10 bases away) is invisible to a hard cluster. That is step 3 of the plan —
the full `‖M_align − M_sim‖_F` against the 0.645 reseed noise floor, plus the downstream
composition fit. Nothing here yet justifies replacing the simulate+map stage; it justifies
building the estimator.

The caveat from `error_rate_sensitivity.md` also stands, and this census sharpens it: the
confusion in *this* DB is 90% exact duplicates, which is the easiest possible case for a
distance-only `M`. A reference set whose members differ by one or two bases would put the
mass exactly where tau=0 says there is none.
