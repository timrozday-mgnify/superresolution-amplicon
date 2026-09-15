# Was the SC2200627-SC3 GTDB misfit the model's fault or the inference's?

**The inference's.** Under the archived full-GTDB matrix a composition exists that
reproduces every high-depth sample to V4-group TV ≈ 0.006, while the archived posterior sat
at 0.784. On data generated from the matrix itself, the genome-space fit still fails its
own posterior-predictive check; the same fit with one parameter per distinct V4 sequence
passes. This is Evidence §1–2 of the
[GTDB inference recovery plan](../docs/gtdb_inference_recovery_plan.md).

Reproduce with:

    python dev/gtdb_failure_diagnosis.py fit    --bundle B --asv-dir A --out O   # ~1.5 min
    python dev/gtdb_failure_diagnosis.py replay --bundle B --asv-dir A --out O D2
    python dev/gtdb_failure_diagnosis.py replay --bundle B2 --asv-dir A --out O \
        --space v4_group --jobs 3 --infer-args "--taxonomy T"      # all 50, ~10 min

`B` is the archived `results/mismapping/<matrix key>/` (here `44ac592b…`, GTDB r232,
`align/kmer`, `tau = 1`, decay `auto` = 0.005895); `A` is `results/` from
`SC2200627-SC3_results.tar.gz`. Needs ~6 GB RAM. Raw `fit` output in
`gtdb_failure_diagnosis_fit.csv`.

## Observed counts

The GTDB-mapped `.mseq` files are not archived, so each DADA2 ASV is placed on a GTDB V4
sequence by exact match: forward read (before the N padding), trimmed at the reverse-
complemented 806R primer, then 3'-anchored against the GTDB amplicons within ±6 bases of
length. This covers a mean **99.95%** of ASV reads (minimum 99.23%) over 54 samples.

## `fit`: best-achievable fit (50 samples with ≥ 1,000 reads)

Maximum-likelihood mixture weights (EM, 3,000 iterations) over every source that can reach
an observed label:

| Inference space | Kernel | V4-group TV, median (range) |
|---|---|---:|
| Genome (active subset, median 7,252 genomes) | archived (v1) | 0.0062 (0.0009–0.0204) |
| V4 group (median 240 groups) | archived (v1) | 0.0056 (0.0009–0.0096) |
| V4 group | per-distinct, fixed `c` | 0.071 (0.041–0.384) |

The archived fit reached 0.784 in the genome-space model of row 1. The forward model is not
the bottleneck.

The per-distinct row is not evidence against that kernel. Denoised ASVs contain none of the
error reads a kernel predicts as leak, and a per-distinct kernel gives every large group
leak onto each of its many singleton neighbours. The v1 kernel happens to fit this
reconstruction because large groups' rows are nearly pure. Real MAPseq output (plan
Phase 4) is the test for the kernel.

The same pass measures two parts of the mechanism at the SVI starting point:

| | Median | Range |
|---|---:|---:|
| TV of `theta_init = clip(theta_obs, 1e-4)` | 0.281 | 0.226–0.670 |
| Initial mass on floored genomes | 0.397 | 0.277–0.778 |
| Dirichlet prior pseudo-read share (`alpha = 0.5`) | 0.049 | 0.032–0.150 |

These active sets are the reconstruction's; the archive's were ~8.5× larger (median
61,856), because real MAPseq output includes error reads that widen reachability.

## `replay`: forward-model replay (D2, 81,721 reads)

D2's placed counts `y` are pushed through the archived kernel (`y @ K`), and each label's
reads are assigned to a uniformly chosen member reference. MAPseq splits exact ties evenly
(custom run: 569/562/555/534/569 reads over five identical copies). The real
`infer_composition.py` (defaults) and `check_composition_fit.py` then run in genome space,
and with every V4 group as one "genome" (translation weight `1/size`). The table below
used a relabelled bundle for the second; `replay` now runs `--infer-space v4_group`, the
same fit.

| Run | Fitted | Group TV | Group PPC percentile | Per-draw TV (observed / replicated) | Status |
|---|---:|---:|---:|---:|---|
| Genome space | 8,653 genomes | 0.0193 | 1.000 | 0.0351 / 0.0069 | `model_misfit` |
| V4-group space | 339 groups | 0.0071 | 0.988 | 0.0120 / 0.0055 | `ok` |

Both runs take about 75 s together. The V4-group fit passes, but only narrowly (0.988
against a 0.99 tail), with the size-weighted v1 kernel still in place.

Variants tried in genome space before this script existed (plan Evidence §2): a
`1e-4/G` initial floor (0.0168), `--alpha 0.05` (0.0127), `--no-presence` (0.0465),
and `--mode mle` (0.0786) all stayed `model_misfit`; `--alpha 1e-3` produced NaN guide
parameters. The first needs a code change; the others replay with
`--infer-args "--alpha 0.05"` etc.

## `replay`, all 50 high-depth samples: `v4_group`, kernel version 2

Plan Phase 3 acceptance. The bundle is the Phase 2 GTDB r232 rebuild (`kmer`, `tau = 1`,
decay 0.005895, `kernel_version 2`) with the archive's `reference/`; inference defaults plus
`--taxonomy ssu_all_r232_ssu.sr_refs.tax`. Raw rows in `gtdb_replay_v4_group.csv`.

| | Median | Range |
|---|---:|---:|
| `ok` | 30 / 50 | |
| Group TV | 0.0094 | 0.0050–0.0167 |
| Group PPC percentile | 0.983 | 0.918–1.000 |
| Per-draw TV, observed / replicated | 0.0142 / 0.0093 | |
| Groups fitted | 349 | 170–709 |
| Genomes those groups span | 8,776 | 4,646–34,904 |

The mean fit meets the TV criterion (ratio 1.01), but 20 samples fail the PPC, all in the
high tail (0.992–1.000). Their posterior draws disagree with the data more than replicates
disagree with the draws.

The presence gate causes it. Five of the worst misfits, re-run with one change each
(`--infer-args=--no-presence`, `"--infer-args=--alpha 0.05"`); percentile / group TV:

| Sample | Baseline | `--no-presence` | `--alpha 0.05` |
|---|---:|---:|---:|
| B1 | 1.000 / 0.0140 | 0.666 / 0.0047 `ok` | 0.998 / 0.0134 |
| D1 | 1.000 / 0.0125 | 0.662 / 0.0053 `ok` | 0.990 / 0.0121 `ok` |
| F1 | 1.000 / 0.0141 | 0.584 / 0.0062 `ok` | 0.966 / 0.0118 `ok` |
| F6 | 1.000 / 0.0123 | 0.768 / 0.0052 `ok` | 0.960 / 0.0075 `ok` |
| G1 | 1.000 / 0.0136 | 0.878 / 0.0076 `ok` | 0.996 / 0.0132 |

Without the gate the group TV halves and the per-draw observed TV nears the replicated
level (median 0.0132 against 0.0119). A smaller Dirichlet prior only nudges the percentile
off 1.000: every sample stays in the tail and the group TV barely moves.

All 50 again with `--no-presence` (`gtdb_replay_v4_group_nogate.csv`): **49/50 `ok`**,
median group TV 0.0052 (0.0032–0.0138), percentile 0.717 (0.500–0.996), per-draw TV
observed / replicated 0.0108 / 0.0094. The one misfit is C5 at 0.996.

## `real`: the proposed configuration on real GTDB MAPseq output

Plan Phase 4. The archived `SC2200627-SC3_sr-amp_results.tar.gz` holds the real GTDB `.mseq`
for all 54 samples, so no re-mapping was needed. All 50 high-depth samples were run with
`--infer-space v4_group --no-presence --infer-distance-decay --taxonomy`, against the Phase 2
kernel-version-2 bundle. Raw rows in `gtdb_real_mapseq.csv`.

    python dev/gtdb_failure_diagnosis.py real --bundle B2 --asv-dir A --out O \
        --mseq-dir results/mapseq --custom-dir custom/results/composition \
        --custom-reference <custom bundle>/reference --taxonomy T --jobs 4    # ~25 min

Before this could run, one inference defect had to be fixed: MAPseq splits a group's reads
over an arbitrary *subset* of its identical members (20 of 445), which the uniform `1/size`
reference-level split read as evidence against large groups. `infer_composition.py` now
spreads a group's observed total evenly in `v4_group` mode. D2's group TV: 0.624 -> 0.021.

| | Median | Range |
|---|---:|---:|
| `ok` | 0 / 50 | |
| Group TV | 0.0201 | 0.0141–0.0424 |
| Group PPC percentile | 1.000 | all 1.000 |
| Per-draw TV, observed / replicated | 0.0215 / 0.0073 | |
| Fitted decay `c` (built 0.005895) | 0.0001 | 0.0000–0.0004 |
| TV to the ASV exact-match profile | 0.3313 | 0.0903–0.5609 |
| TV to the custom-database run | 0.3133 | 0.0878–0.7182 |
| TV, raw MAPseq labels to ASV | 0.3309 | 0.0853–0.5593 |
| ASV mass on groups MAPseq never reports | 0.239 | 0.012–0.527 |
| Groups fitted | 1,225 | 530–2,087 |
| Inference wall time / peak RSS | 78 s / 1.73 GB | 29–145 s / 1.48–2.57 GB |

Every acceptance criterion failed. The accuracy gap is the forward model, not the fit: TV to
the ASV profile tracks the raw-MAPseq-to-ASV TV with correlation 0.999 (median difference
0.005), so the posterior is reproducing the labels it was given. What the kernel misses:

- **Exact hits relabelled.** All reads matching `v4g_2acb4710c828d339` exactly are reported
  on its one-edit neighbour `v4g_81c3cde1090c2cdc` at identity 0.996 (D2: 0% vs 18%).
  Against a two-sequence database the same reads map to 2acb, so it is a scale effect:

      docker run --rm --platform linux/amd64 -v $PWD:/w -w /w \
          quay.io/biocontainers/mapseq:2.1.1b--hc47f52e_1 mapseq reads.fa db.fasta db.tax

- **IUPAC ties ignored.** `v4g_e0d8cd01617fb6d0` differs from the 2,832-reference
  `v4g_08bafb1f4c88b084` by one `R`, so the ambiguity weight places it at distance 0 and
  sends it 18% of that group's mass; MAPseq sends none. Largest single term in D2's residual.
- **Subset tie-splitting**, above.

The PPC failure has the same cause: group TV is small but the posterior's draws disagree with
the data ~3x more than replicates do, and the fitted decay collapses toward 0 chasing leak
the data does not show.

## Caveats

- `fit` and `replay` observed counts come from DADA2 ASVs, not from GTDB MAPseq output;
  `real` uses the archived MAPseq output itself.
- The reconstruction's active sets are smaller than the archive's, and every genome-space
  defect grows with the active set, so the replay understates them.
