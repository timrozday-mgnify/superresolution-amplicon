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

## Short and unmerged reads (`--read-len`)

The kernel above compares whole amplicons, which is wrong as soon as a read is shorter
than one: the read sees a window, and references differing *outside* that window are
indistinguishable to the mapper. `build_mismapping_align.py --read-len L` instead scores
every length-`L` window of the source reference against every reference by *infix*
alignment (edlib `HW` — the best placement of the read anywhere in the reference) and
averages the tie clusters over windows. Re-running the whole study at `L = 150`
(`python dev/alignment_mismapping.py --read-len 150`, raw numbers in
`alignment_mismapping_readlen150.csv`):

| candidate | ‖·−M_sim‖_F | beyond one run's noise | mean diag | TV med / max | support J | ρ off-diag |
|---|---|---|---|---|---|---|
| reseed (seed 99) | **0.609** *(the floor)* | 0.430 | 0.2844 | 0.067 / 0.143 | 0.695 | 0.826 |
| trained error model | 0.663 | 0.504 | 0.2869 | 0.070 / 0.147 | 0.704 | 0.848 |
| **alignment tau=0, windowed** | **0.432** | **0.040** | 0.2885 | **0.045 / 0.087** | 0.701 | 0.846 |
| alignment tau=0, *whole-amplicon* | 0.912 | 0.804 | 0.3086 | 0.053 / 0.360 | 0.610 | 0.792 |

1. **The windowed kernel passes and the whole-amplicon one fails.** 0.432 against a 0.609
   floor, versus 0.912 — the pre-fix estimator is the only non-control candidate here that
   lands *above* the noise floor. Its error is systematic (0.804), not sampling.
2. **The bias was in the predicted direction.** The whole-amplicon matrix holds its
   diagonal at 0.3086 while the measurement drops to 0.2844: at 150bp reads there is
   genuinely *more* confusion, and a whole-amplicon distance cannot see it. TV max 0.360
   says a few rows are badly wrong rather than everything being slightly off.
3. **Windowing also fixes the low-identity tail complaint.** Support Jaccard rises to
   0.701 and ρ to 0.846 — at or above the reseed's own 0.695 / 0.826, where the
   whole-amplicon kernel sat at 0.610 / 0.792. 104 overlapping windows generate the
   near-ties a single global comparison has no way to produce.
4. **The cost argument mostly evaporates.** 1.96 s vs 4.8 s — 2×, not 460×. The windowed
   kernel is `n_refs² × n_windows` infix alignments. It is still simpler (no mapper, no
   container, no error model), but speed stops being the reason to choose it.
5. **Everything is harder at 150bp**, as it should be: naive-observed L1 rises to
   0.026–0.031 from 0.022–0.028. Downstream, windowed alignment is again indistinguishable
   from simulate+map (0.0199 / 0.0299 vs 0.0256 / 0.0294) — and note the *whole-amplicon*
   matrix posted the best L1 of all under one generator (0.0234) while being demonstrably
   the wrong matrix. That is §4 of the reading above, in one line: the downstream test
   cannot rank candidates.

## What this does not show

One reference set, and the easiest kind: 73 of its 81 references are exact duplicates.
Point 4 says where the model is thin, and point 5 says the kernel has no knob to absorb a
set that is *near*-identical rather than identical — the binomial-race kernel (plan §4
rung 4) is the designed answer there, and remains unbuilt and untested. Before making this
the pipeline default rather than an option, run this same study on a reference set whose
members differ by one or two bases.
