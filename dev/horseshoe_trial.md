# Horseshoe shrinkage on the measured matrix (plan Phase 5 regularisation)

**Result: the horseshoe is the right regulariser, but it cannot close the gap on its own.**
It removes the scattered mass the Dirichlet leaves and keeps the relabelled group the hard
gate deletes. What is left sits on a few one-edit pairs that the measured matrix cannot
separate, or that the simulation does not reproduce. No prior fixes either.

    python bin/infer_composition.py --amplicon-dir bundle/reference \
        --mismapping-matrix measured_r0.0025/mismapping_matrix.npz --obs-mseq S01.obs.mseq \
        --sample-id S01 --infer-space v4_group --no-presence --horseshoe --steps 10000 -o OUT
    python dev/horseshoe_trial.py --truth S01_a0.00.515-YF-806BR.truth.tsv \
        --amplicons references/amplicons \
        --alias bacteroides_uniformis_strain2=BU_JCM13286_NT5170 RUN_DIR...

Same setup as `gtdb_measured_matrix.md`: benchmark sample S01 (21 genomes, known
composition), mapped against full GTDB r232, matrix measured at sub rate 0.0025, 259
groups fitted. Model: `lambda_i ~ HalfCauchy(1)`, `tau ~ HalfCauchy(1)`,
`w_i ~ HalfNormal(tau * lambda_i)`, `theta = w / sum(w)`, vi only. Raw rows in
`horseshoe_trial.csv`. The truth is rebuilt from the amplicon files, so the scores differ
from `gtdb_measured_matrix_truth.csv` by at most 0.004.

| Inference | Steps | TV to truth | Mass outside truth | `2acb` (truth 0.064) |
|---|---:|---:|---:|---:|
| none (raw MAPseq labels) | — | **0.205** | 0.175 | 0.000 |
| alignment kernel v2, no gate | 3,000 | 0.218 | 0.181 | 0.000 |
| measured, presence gate | 3,000 | 0.228 | 0.193 | 0.000 |
| measured, Dirichlet, no gate | 3,000 | 0.324 | 0.320 | 0.062 |
| measured, Dirichlet, no gate | 10,000 | 0.462 | 0.460 | 0.063 |
| measured, horseshoe | 3,000 | 0.268 | 0.258 | 0.060 |
| **measured, horseshoe** | **10,000** | **0.233** | **0.209** | **0.065** |

- **Matched steps matter.** With more steps the Dirichlet gets worse (`alpha = 0.5` spreads
  mass as it converges) and the horseshoe gets better. The 3,000-step comparison in
  `gtdb_measured_matrix.md` understated the gap.
- **The scatter is gone.** Groups outside the truth above 0.1%: 38 (Dirichlet) -> 7
  (horseshoe, 3k) -> 4 (horseshoe, 10k). Mass in the 0.1–1% band: 0.074 -> 0.006 -> 0.
- **`2acb` is recovered exactly** (0.0648 against 0.0644) from its secondary leaks alone.
  The hard gate deletes it.
- Fit status `ok`. 22 s and 1.5 GB for 3,000 steps.
- **It converges slower than the Dirichlet**, both here and in the unit test
  (`test_horseshoe_recovers_a_relabelled_group_from_its_secondary_leak`), where 2,000
  steps leave the relabelled source at ~0 and 8,000 recover it. Any default using it
  needs more SVI steps than the current 3,000.

## What the horseshoe cannot fix

At 10k steps, 0.194 of the 0.209 outside-truth mass sits on four groups. The measured rows
(`S · size`, 500 reads per group) explain three of them:

| Outside group | Inferred | Truth group it displaces | Measured rows | Kind |
|---|---:|---|---|---|
| `89c3` | 0.078 | `a3e62` (0.077, never labelled) | `a3e62` -> `89c3` 98.6%; `89c3` -> itself 98.8% | Rows indistinguishable: an exact block |
| `125a` | 0.071 | `87ad` (0.077) | `125a` -> `87ad` 100%; `87ad` -> itself 98%, 2% leak elsewhere | Near-block: decided by a ~2% leak measured from 500 reads |
| `2a9c` | 0.033 | `a49d` (0.077) | `a49d` -> `2a9c` 0% in simulation; real reads land there at 3.2% | The simulation does not reproduce the real relabelling |
| `cb04` | 0.013 | — | 54 edits from `8a3d` | Not explained |

~~Ruled out~~ (wrong; see [the sink section](#after-replacing-row-renormalisation-with-a-sink-label)):
pruning renormalises each row over the kept labels, which discards a source's leak onto
labels that drew no reads. The dropped row mass is small (median 0.4%), but that
measured the wrong thing. A 1% leak on a 6% group is ~50 expected reads against 0, and
removing that evidence changes which near-identical source wins.

Also measured: overdispersion is not absorbing the evidence (`conc_frac` ≈ 1,100, near
multinomial). The mis-mapping scale is, though: `s` = 1.72 (horseshoe) and 1.53
(Dirichlet), so the fit inflates every measured leak by 50–70%. That alone could decide
`87ad`/`125a` against the truth: `87ad`'s 2% leak scaled to ~3.4% is more than the data
shows, and `125a` has no leak at all. It also points at the error-rate calibration step.
This is a hypothesis; it has not been tested with `s` pinned.

All three explained pairs are one edit apart. Inside a block the likelihood is flat, so any
prior only picks a point in it (plan Phase 5). The next steps are those the plan already
lists: report at block resolution, simulate more reads per group, and find out why
simulated `a49d` reads do not follow real ones (read length, primer trimming, or the
error profile).

## After replacing row renormalisation with a sink label

Pruning used to renormalise each kept row over the kept labels. It now sends the dropped
mass to zero-count sink labels (`_subset_mismapping`), which makes the pruned likelihood
equal the full one (`test_pruning_sends_dropped_mass_to_sink_labels_exactly`). Same S01
runs, same settings:

| Inference | Before (renormalised) | After (sink) |
|---|---:|---:|
| alignment kernel v2, fitted decay, no gate | 0.218 | 0.217 |
| measured, presence gate, 3k | 0.228 | 0.243 |
| measured, Dirichlet, no gate, 10k | 0.462 | 0.712 |
| measured, horseshoe, 10k | 0.233 | **0.659** |

The horseshoe still recovers `2acb` (0.065), but its fit check is now `model_misfit`
(group PPC percentile 1.000). `s` drops from 1.72 to 1.38.

**The sink is right; the measured matrix is wrong about leak.** Pushed through the measured
matrix, the truth predicts 1.05% of reads (835 of 79,468) on labels that drew **no** reads.
Each truth group's measured row leaks a median 1.0% (max 2.0%) onto such labels. The
renormalisation had silently removed that disagreement. With the sink it counts, and the fit
resolves it by swapping each truth group for an unobserved neighbour whose measured row has
no such leak:

| Truth group (truth) | Leak to unobserved labels | Replaced by | Replacement's row |
|---|---:|---|---|
| `2fef` (0.064) | 1.8% | `0106` (0.074) | 99.2% -> `2fef` |
| `8a3d` (0.077) | 0.8% | `fe3b` (0.071) | 100% -> `8a3d` |
| `360e` (0.064) | 0.8% | `8ef2` (0.070) | 100% -> `360e` |
| `38ea` (0.064) | 1.6% | `0ebd` (0.069) | 100% -> `38ea` |
| `53d6` (0.064) | 0.8% | `bd95` (0.061) | 99.6% -> `53d6` |

(and five smaller pairs: `a49d`, `42e2`, `e518`, `cf36`, `4c74`). This is the `87ad`/`125a`
case from above, now happening everywhere. Rows are measured from 500 reads, so a 1% leak
against 0% is within noise (SE ~0.45%). The simulation also predicts leak that real reads
do not show, even at a substitution rate of 0.0025. Either one lets noise in the leak
decide between near-identical sources.

Consequences for the method, in order:

1. The leak a measured row predicts onto labels outside the observed set must be calibrated
   before it is used as evidence. That means a lower or fitted error rate, and more reads per
   group.
2. Near-identical sources (rows > 99% onto the same label) are a block. Report them as one,
   rather than letting a noisy leak pick a member.
3. Until then the renormalised result (0.233) looked better only by accident: it hid a
   forward-model misfit, it did not avoid one.
