# Alignment-based mis-mapping estimation

**Status: complete and shipped opt-in** (`--mismapping_method align`). This was the
implementation plan; it now also records what was found, and — more importantly — what the
result does *not* cover.

Steps 1, 2, 3 and 6 were executed. Steps 4 and 5 (the kernel ladder beyond rung 1, and the
alignment-parameter sweeps) were dropped as unjustified once rung 1 passed; §"Limitations
and biases" says when they become necessary again.

| | |
|---|---|
| **Goal** | Replace *simulate reads → run mapseq → tally* with *align the reference amplicons to each other → derive `M`*, so the mis-mapping stage is one Python command with no container, no mapper and no simulation. |
| **Outcome** | `‖M_align − M_sim‖_F = 0.43` against a seed-to-seed noise floor of `0.59`; downstream composition at least as good under both generators; **460×** cheaper (0.011 s vs 5.2 s). |
| **Shipped as** | `--mismapping_method align` (+ `--align_tau`, default 0). Default remains `simulate`. |
| **Evidence** | [census](../dev/amplicon_distance_census.md) (step 1), [equivalence study](../dev/alignment_mismapping.md) (step 3) |

## Why it works, in one line

Confusion in a 16S amplicon reference set is **redundancy**, not sequencing error: 73 of
the 81 references in the B. uniformis V4 set are byte-identical over the amplicon, so `M`
is a property of the references that any mapper must reproduce. That was already visible
in [the error-rate study](../dev/error_rate_sensitivity.md) (mean diagonal ≈0.30 from
0.01% to 15% error); this work made it the estimator.

## What was built

| file | what |
|---|---|
| [`bin/build_mismapping_align.py`](../bin/build_mismapping_align.py) | The estimator. `--amplicons … -o mismapping_matrix.csv`, `--tau`, `--demo`. Writes the CSV `infer_composition.py --build-mismapping` already emits, so it needs no downstream change. |
| [`dev/amplicon_distance_census.py`](../dev/amplicon_distance_census.py) / [`.md`](../dev/amplicon_distance_census.md) | Step 1: does distance alone predict `diag(M)`? |
| [`dev/alignment_mismapping.py`](../dev/alignment_mismapping.py) / [`.md`](../dev/alignment_mismapping.md) | Step 3: the equivalence study. |
| `modules/local/align_mismapping/` + `params.mismapping_method` / `align_tau` | Step 6: replaces `SIMULATE_READS` + `MAPSEQ_CLUSTER_MATRIX` + `MAPSEQ_SIM` + `BUILD_MISMAPPING` (11 tasks → 8) and short-circuits the skiver subworkflow. Both params are in the `MATRIX_KEY` hash and `provenance.json`. |

## Decisions, and what they rest on

**Library: `edlib`** (Myers bit-parallel edit distance, global/`NW` mode). Chosen over
`parasail` (SIMD affine-gap, `_stats_` identity), Biopython's `PairwiseAligner` (kept as a
slow oracle), `pyopal`, and `mappy` (minimap2 — the fallback had no alignment kernel
worked). With a 2-base amplicon length spread and 90% zero distances, nothing edlib
omits — affine gaps, scoring matrices, local alignment — can move an argmin set, so the
knob-bearing options were never needed. **This is a property of the data, not of edlib**:
a set with real length variation would need `parasail` or `HW` (infix) mode.

**Kernel: rung 1, the tie cluster.** `M[a,j] = 1/|{k : d(a,k) ≤ τ}|`, zero free
parameters. Rungs 2–4 of the original ladder (tolerance τ, softmax β, and a binomial
"who wins under `e` errors" race) were **not built**. τ > 0 was tested and failed
decisively — `‖·‖_F` jumps from 0.43 to 2.10, TV max from 0.09 to 0.82 — so there was
nothing to fit and the fit/validate split (§5.4 of the original plan) never applied.

**Criterion: the measurement's own precision.** The floor is
`‖M_sim(seed 1) − M_sim(seed 99)‖_F`. Because that compares two *noisy* matrices,
`floor² ≈ 2·noise²`, so a noiseless estimator that agreed perfectly would still score
`floor/√2 = 0.42`, not 0. Splitting that off puts alignment's systematic difference at
**0.093** — 4.5× below the noise a single simulate+map run contributes.

## Limitations and biases

Ordered by how likely each is to bite. The first three are the reasons the default is
still `simulate`.

### 1. The test set is the easiest possible case, and it is also the set that generated the hypothesis

73/81 references are exact duplicates; 4 sit at distance 1; **none** at distance 2. A
kernel that clusters exact duplicates cannot be much wrong when almost every relationship
in the DB is exact. Worse, this is the same DB whose structure suggested the approach —
hypothesis and test are not independent. `n = 1` reference set, one amplicon (V4
515F/806R), one mapper (mapseq 2.1.1b), one truth composition, one read depth.

**Where it should fail:** a set whose members differ by one or two bases, where a single
sequencing error is exactly what flips an assignment. τ=0 asserts zero confusion there;
τ>0 asserts total confusion; neither is right, and the binomial-race kernel (rung 4)
exists precisely for that regime and remains unbuilt and untested. **Before making
`align` the default, run [`dev/alignment_mismapping.py`](../dev/alignment_mismapping.py)
on such a set.**

### 2. The kernel has a known, directional bias: it emits no low-identity tail

Support Jaccard 0.584 and off-diagonal ρ 0.776 are the two metrics where alignment sits
*below* the reseed floor's own values (0.726 / 0.844) — it lands where the genuinely
different *trained* error model lands (0.594 / 0.781). A hard tie cluster puts **exactly
zero** mass on a reference 10 bases away; mapseq occasionally puts a little there.

The bias has a direction: **`M_align` is over-confident / under-dispersed.** Isolated
references get an exact identity row, i.e. a claim of *zero* confusion where the
measurement shows some. Consequences: the inversion under-corrects for leakage arriving
from outside a cluster, and a genome whose references are all singletons gets no
correction at all. The composition model's `s` scale parameter (`M_eff = (1−s)I + s·M`)
can absorb a *global* miscalibration but not a structurally missing tail. On this DB the
missing mass was too small to move `‖·‖_F` or the fit; on a DB with more distant-but-real
confusion it would not be.

### 3. The observed reads are simulated by the same code family as `M`

Both the "observed" sample and `M_sim` come from `simulate_amplicon_reads.py` (flat or
skiver). Neither `M` has been checked against **real reads with known truth**. Real
amplicon data brings chimeras, off-target amplification, primer bias, PCR error, and
length variation, none of which either method sees. This limitation is inherited from
[`dev/error_rate_sensitivity.md`](../dev/error_rate_sensitivity.md) and is not made worse
by alignment — but it is not fixed by it either, and it caps how much any of these numbers
mean.

### 4. The downstream test has almost no discriminating power

Naive-observed L1 is 0.022–0.028 and *every* non-broken `M` lands at 0.016–0.024, while
run-to-run wobble is ~0.003. The test catches a broken `M` (shuffled control: 0.051, worse
than no correction) and nothing finer. Alignment τ=0 posted the best L1 under both
generators and the best error on the confusable pair — **that ordering is not
statistically meaningful.** The supportable claim is "indistinguishable from, and never
worse than, simulate+map".

### 5. The headline metric is biased toward deterministic estimators

"Below the noise floor" rewards smoothness as such: a noiseless estimator is compared
against one noisy matrix, a reseed against two, so the alignment path gets a `√2` discount
before any correctness enters. That is why the decomposition (§"Criterion") and the
shuffled control are load-bearing. Relatedly, **the mean diagonal — step 1's whole
metric — does not discriminate at all**: the shuffled control scored 0.2915 against a
measured 0.30 while being wrong in every other respect (F = 5.8, support Jaccard 0.024,
ρ ≈ 0). Shuffling preserves the distance multiset and hence the cluster-size distribution.
Step 1 confirmed the mechanism; only step 3 confirmed the matrix.

### 6. Exact-duplicate clustering is discontinuous in its inputs

τ=0 makes cluster membership a *step* function of upstream settings. Change `--fwd_primer`
/ `--rev_primer` / `--primer_mismatches` and amplicon boundaries shift, so clusters can
merge or split abruptly; simulation degrades smoothly instead. Two specific traps:

- **Ambiguity codes.** `edlib` treats `N` as a plain mismatch, so **one `N` in one copy of
  an otherwise-identical pair breaks the cluster** and hands both references identity
  rows — the maximally wrong answer for sequences that are in fact indistinguishable.
  Draft-genome 16S frequently carries ambiguity. This DB has **zero** non-ACGT bases in
  all 81 amplicons, so the failure mode was never exercised here. The fix, if a DB needs
  it, is edlib's `additionalEqualities`.
- **Amplicon extraction.** 16 of 97 DB entries produced no amplicon and are excluded from
  both methods. That bound is unchanged by this work but caps it.

### 7. Short and unmerged reads are silently unsupported

The kernel compares **whole amplicons**. A read shorter than the amplicon sees only a
window, so references differing *outside* that window are indistinguishable to the mapper
— confusion the alignment kernel cannot see and therefore understates. `--sim_read_len`
models this on the simulate path and is meaningless on the align path; the workflow now
logs a warning when both are set, but there is no correct behaviour to fall back to. For
unmerged/short-read data, use `--mismapping_method simulate`.

### 8. High-error long reads are outside the tested regime

The tie cluster is error-model-free, which is the point at Illumina rates and wrong at ONT
/ PacBio-CLR rates where reads genuinely cross a 1–2 base gap. Only `hq-illumina`-like
error was tested. This is the same gap as §1, reached from the error side rather than the
reference side.

### 9. Reported figures carry ~1% run-to-run uncertainty

mapseq is not bit-reproducible across threads: the noise floor itself moved 0.593 → 0.599
between runs of the study, and downstream L1 by ~0.003. No comparison here should be read
at three decimal places.

## What would change the answer

1. **The near-identical reference set** (§1) — the single most valuable next experiment,
   and the gate on making `align` the default. If it fails, build rung 4 (the binomial
   race: for a pair differing at `d` columns and per-base error `e`, the score difference
   is a sum of `d` iid ±1 steps, so `P(j outscores a) = P(S ≤ 0)`; normalise over `j`).
2. **Real reads with known truth** (§3) — a mock community. Would validate both methods,
   not just this one.
3. **A soft tail on the kernel** (§2) — the cheapest fix if support Jaccard becomes the
   binding constraint: rung 3's `M[a,j] ∝ exp(−β·d)` with a large β is a tie cluster with
   a tail, at the cost of one fitted knob and the fit/validate split that avoids.
4. **Ambiguity handling** (§6) — one `additionalEqualities` argument, worth adding the
   first time a DB with `N`s appears rather than pre-emptively.
