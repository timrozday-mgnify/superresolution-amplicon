# Making GTDB-scale amplicon inference identifiable and testable

**Status:** Phases 0–5 implemented (Phase 5's measurement and horseshoe trial; its inversion is limited by one-edit blocks, not by the prior). **Phase 4 failed every acceptance criterion on real
GTDB MAPseq output** (0/50 `ok`; median group TV to the ASV profile 0.331 against a 0.05
limit): that inference reproduces MAPseq's labels
almost exactly, but those labels are themselves TV 0.33 from exact-match placement because
the alignment kernel does not describe what MAPseq does at GTDB scale. See
[Phase 4](#phase-4--implemented-2026-09-14-acceptance-failed).
Phases 5–6 below are the approved specification for what follows: `M` measured by
simulation on each batch's active groups (Phase 5), then validation (Phase 6). The 2026-09-11 revision replaced the earlier screen/candidate-database
design: a diagnosis against the SC2200627-SC3 archive showed that the archived GTDB failure
came from the *inference*, not the forward model (see [Evidence](#evidence)). That holds —
but it was established against the archived matrix and replayed labels; Phase 4 then found
the forward model wanting against real MAPseq output.

## Executive decision

(Phase 4 qualifies this section: the archived failure is still the inference's, but the
alignment-built kernel does not reproduce real MAPseq labels at GTDB scale, so a measured
matrix is now required — see [Phase 4](#phase-4--implemented-2026-09-14-acceptance-failed).)

The SC2200627-SC3 full-GTDB run did **not** fail because V4 data cannot be described by the
GTDB mis-mapping model. Under the archived matrix a composition exists that reproduces every
high-depth sample's V4-group count distribution to TV ≈ 0.006. The archived posterior's
0.784 came from three defects:

1. **Inference in genome space.** Tens of thousands of genomes share each V4 sequence, so
   the Dirichlet prior, the SVI initial point, and the presence gate act on dimensions the
   likelihood cannot separate. On data generated *from the model itself*, every
   genome-space configuration tried fails the posterior-predictive check; the same fit with
   one parameter per distinct V4 sequence passes.
2. **A size-weighted kernel.** `build_kmer_grouped` weights a neighbouring sequence by
   `size × c^d`, so a V4 sequence carried by many GTDB genomes captures reads in proportion
   to its duplicate count. MAPseq does not behave like that (leak onto a one-edit neighbour
   duplicated 1, 10, and 100 times: 1.40%, 1.54%, 1.92%; kernel: 0.59%, 5.6%, 37%).
3. **A raw-reference release gate.** `check_composition_fit.py` fails a sample on the
   raw-reference PPC, which the data contract below already classifies as diagnostic-only.

Adopt **V4-group inference with a per-distinct-sequence kernel**:

- infer abundance over exact distinct V4 amplicon sequences (V4 groups), with genome and
  taxon labels attached afterwards as annotations;
- weight each distinct neighbouring sequence by `c^d` and split its share evenly across its
  duplicate references;
- gate release on the V4-group posterior-predictive check only;
- validate on real GTDB MAPseq output and mock communities before production use.

The screen database, batch candidate database, candidate re-mapping, and simulated-matrix
calibration suite were **deferred**, each with a stated trigger in
[Conditional work](#conditional-work). Phase 4 fired the simulated-matrix trigger: against
real GTDB MAPseq output the alignment-built kernel does not describe the mapper, so `M` has
to be measured. The other three remain deferred.

## Implementation status

### Phase 0 — implemented 2026-09-11

- `posterior_draws.npz` retains `theta_eff` and fitted `s`, `conc_frac`, and optional `c`.
- `check_composition_fit.py` reconstructs the active subset and emits raw-reference and
  exact-V4-group observed-versus-expected TV and PPC percentiles in `fit_diagnostics.json`.
- `mean_diagonal` is replaced by `mean_kernel_diagonal` and `mean_reference_diagonal`.
- `min_infer_reads = 1000` yields `low_depth` instead of a biological composition; the fit
  check still runs.

### Phase 1 — implemented 2026-09-11

- `fit_status` is decided by the `unique_v4_group` percentile alone; the payload records
  `"gate": "unique_v4_group"`. The test suite covers reads piled on one duplicate member
  (`ok`) and a perturbed group (`model_misfit`).
- The diagnosis is committed as `dev/gtdb_failure_diagnosis.py` (`fit`, `replay`) and
  `dev/mapseq_multiplicity.py`, each with a `.md` of results.
- README "GTDB-scale safety" and the `alignment_mismapping_plan.md` callout no longer
  describe the forward model as refuted.
- Acceptance: the stored D2 replay in V4-group space re-checks as `ok` (group percentile
  0.988; raw 0.34, diagnostic only).

### Phase 2 — implemented 2026-09-11

- `build_kmer_grouped` weights each distinct neighbour by `c^d · amb` once and divides by
  its duplicate count; `tie_cluster_matrix(multiplicity=byte_multiplicity(seqs))`,
  `build`, and `build_sparse` apply the same `1/n(j)`, so all backends agree.
- `sparse_matrix.KERNEL_VERSION = 2` is written by `write_grouped` / `write_matrix`. The
  readers keep their signatures and log a warning for version 1 or a missing field;
  `stored_kernel_version(path)` reads it. `write_mismapping_bundle.py` records it in
  `provenance.json`, and `MATRIX_KEY` hashes `kernel_version: 2`. A test keeps the Groovy
  literal equal to the Python constant.
- Tests: leak `c·amb/(1+c·amb)` for `k ∈ {1, 10, 100}` with and without ambiguity,
  `Σ S·size = 1`, grouped equal to reference-square, `DecayKernel` and
  `_kernel_at_decay` re-decay equal to a direct build, version stored/hashed/warned.
- Acceptance:
  - `dev/mapseq_multiplicity.py`: kernel leak 0.586% at every `k`, ratio 1.00 (MAPseq
    1.41). Passed.
  - `dev/alignment_mismapping.py` (B. uniformis): `‖M_align − M_sim‖_F = 0.461` at
    `tau = 0` against a reseed floor of 0.628 (implied systematic difference 0.126).
    Passed. The recorded version-1 run was 0.430 / 0.593; `M_sim` itself moved between
    runs, and the `tau = 0` diagonal is unchanged at 0.3086. The script's downstream
    composition fit still fails on a sparse `M` in `error_rate_sensitivity.fit`
    (pre-existing, not a Phase 2 check).
  - GTDB r232 rebuild at `kmer`, `tau = 1`: 77 s, 0.80 GB peak RSS. Passed. Row self-mass
    below 0.99: 13,719 rows (version 1: 21,809); below 0.10: 334; 5th percentile 0.977.

### Phase 3 — implemented 2026-09-11; replay acceptance met with the gate off

- `infer_composition.py --infer-space {genome,v4_group}` (default `genome`) and
  `--taxonomy`. `_v4_groups` lives here and names groups `v4g_<16 hex of sha256>`, sorted
  by id. In `v4_group` space `genomes`, `g_of_ref`, and the translation (weight `1/size`)
  are replaced by group space before pruning; fitting and expansion are unchanged. Phase 4
  added one step: each group's observed count is spread evenly over its members first.
- Outputs follow the [Output contract](#output-contract-infer_space--v4_group):
  `inferred_v4_groups.csv`, the split genome table with `v4_group_ids` and `resolution`,
  `infer_space` in `posterior_draws.npz`, and `infer_space`, `n_groups_fitted`,
  `max_genomes_per_active_group`, `taxonomy_sha256` in `inference_diagnostics.csv`. Genome
  space computes the last two whenever `amplicons.fasta` exists, and warns above 100.
  An identifiable genome's `presence_prob` is the maximum over its groups (a lower bound on
  "any present"); a non-identifiable genome's is empty.
- `check_composition_fit.py` reads `infer_space` (absent = `genome`), builds the group
  translation through `ic._v4_groups`, and accepts `v4_group_id` tables.
- Nextflow: `params.infer_space`, `params.taxonomy` (staged path input, `[]` when null),
  optional `v4_groups` emit routed to `CHECK_COMPOSITION_FIT` in `v4_group` mode, published
  under `composition/<id>/`; README parameter rows.
- Tests: `demo_v4` (two genomes sharing an amplicon plus one distinct: two groups, stable
  ids, `not_identifiable` pair with empty intervals, finite genome means summing to 1,
  strict LCA) runs under `--demo` and in pytest, which also round-trips it through the fit
  check; an nf-test stub asserts `inferred_v4_groups.csv`. `normalize_sr_profile.py`
  accepts the v4-mode genome table (checked by hand; it lives in the benchmark repo).
- `dev/gtdb_failure_diagnosis.py replay` runs `--infer-space v4_group` directly, replays
  every high-depth sample when none is named, and takes `--jobs`.
- Acceptance:
  - Custom-database regression, genome space: D2 at the pipeline seed (42) reproduces the
    archived composition to 2e-16. Passed.
  - Forward-model replay, kernel version 2, `v4_group`, all 50 high-depth samples
    (`dev/gtdb_replay_v4_group.csv`): median group TV 0.0094 against median replicated TV
    0.0093 (ratio 1.01 ≤ 1.5). Passed. **`ok` in 30/50 (60%) against ≥ 95%. Failed.**
    Every misfit is in the high tail (percentile 0.992–1.000, median over all 0.983); the
    per-draw observed TV is 0.0142 against 0.0093 replicated. The fit is close in mean but
    its posterior draws are noisier than the data. About 29 s and 1.7 GB per sample.
  - Diagnosis: the presence gate. On five of the worst misfits (B1, D1, F1, F6, G1;
    baseline percentile 1.000) `--no-presence` makes all five `ok` (percentile 0.58–0.88)
    and halves group TV (0.0047–0.0076 against 0.0123–0.0141). `--alpha 0.05` leaves all
    five in the tail (0.960–0.998) with group TV nearly unchanged. This is Evidence §2's
    mechanism surviving the move to group
    space: temperature-1 Concrete noise makes the posterior draws disagree with the data.
  - Same replay with `--no-presence`, all 50 (`dev/gtdb_replay_v4_group_nogate.csv`):
    **`ok` in 49/50 (98%). Passed.** Median group TV 0.0052 against replicated 0.0094
    (ratio 0.56); median percentile 0.717 (0.500–0.996). The one misfit, C5, sits at
    0.996. The Proposed configuration now sets `infer_presence: false` for `v4_group`.
    The pipeline default stays `true`: the presence sweep validated it for genome space.
- `lca` is the strict common prefix specified above. At GTDB scale it is fragile: D2's two
  largest groups are 97–98% *Bacteroides* but report `d__Bacteria`, because a few members
  carry other-phylum lineages. A majority-rule LCA is a candidate follow-up.

### Phase 4 — implemented 2026-09-14; acceptance failed

Step 1 (build the `.mscluster` on HPC and map 54 samples) was **not needed**: the archived
`SC2200627-SC3_sr-amp_results.tar.gz` already contains the real GTDB `.mseq` for all 54
samples, mapped with the pinned MAPseq, the pipeline's read preparation and the same
1,001,241-reference amplicon set. No HPC cost was incurred.

- `dev/gtdb_failure_diagnosis.py real` runs the Proposed configuration (`v4_group`,
  `--no-presence`, `--infer-distance-decay`, `--taxonomy`) on that output and scores each
  sample's group profile against the ASV exact-match profile, the custom-database run
  aggregated to V4 groups, and the Phase 3 replay fit, recording wall time and peak RSS.
  Raw rows in `dev/gtdb_real_mapseq.csv`.
- **An inference defect found and fixed.** MAPseq splits a V4 group's reads evenly over an
  arbitrary *subset* of its identical members (20 of 445; 27 of 1,393), not over all of
  them, so the reference-level likelihood's uniform `1/size` split read a pile-up as
  evidence against large groups. `infer_composition.py` now spreads each group's observed
  total evenly over its members in `v4_group` mode; for a multinomial that is the
  group-label likelihood, and it is the data shape the Phase 3 replay validated. D2's group
  TV went from 0.624 to 0.021. Guarded by
  `test_v4_group_fit_ignores_which_identical_member_mapseq_picked`.

**Acceptance: failed.** All 50 high-depth samples:

| Criterion | Result | Verdict |
|---|---:|---|
| ≥ 95% `ok` | 0 / 50 | Failed |
| Median group TV to the ASV profile ≤ 0.05 | 0.331 (0.090–0.561) | Failed |
| Median group TV to the custom-database run ≤ 0.05 | 0.313 (0.088–0.718) | Failed |
| Per-sample runtime and memory | 78 s median (max 145), 1.73 GB median (max 2.57) | Recorded |

The accuracy failure is **the forward model, not the fit**. Per-sample TV to the ASV
profile tracks the TV between the *raw MAPseq labels* and that profile with correlation
0.999 (median difference 0.005): the posterior reproduces what MAPseq reported. The labels
themselves sit at median TV 0.331 from exact-match placement, and a median 24% of ASV read
mass (max 53%) lands on V4 groups MAPseq never reports in that sample.

Three MAPseq behaviours the kernel does not contain, all measured here:

1. **Subset tie-splitting** (above), which the replay's uniform split hid.
2. **Exact hits relabelled onto a one-edit neighbour.** Every read exactly matching
   `v4g_2acb4710c828d339` (151 references) is reported on `v4g_81c3cde1090c2cdc` at
   identity 0.996; 2acb draws zero reads in D2 while 81c3 takes 18%. Against a two-sequence
   database the same reads map to 2acb, so this is a GTDB-scale search effect, not a
   scoring rule the kernel could reproduce.
3. **IUPAC ties ignored.** `v4g_e0d8cd01617fb6d0` differs from the 2,832-reference
   `v4g_08bafb1f4c88b084` only by an `R`, so the ambiguity weight puts it at distance 0 and
   sends it 18% of that group's mass; MAPseq sends it none. It is the single largest term
   in D2's residual.

The PPC failure has the same root. Group TV is small (median 0.020) but the posterior's own
draws disagree with the data about three times more than replicates do (per-draw observed
0.0215 against 0.0073 replicated), so every sample lands at percentile 1.000, and the fitted
decay collapses to `c ≈ 1e-4` (built 0.0059) trying to buy back leak the data does not show.

**Consequence.** The [Conditional work](#conditional-work) trigger for the simulated-matrix
calibration suite has fired: `M` must be *measured* by simulating reads from the GTDB
amplicons and mapping them with the same MAPseq and database, not built from alignment
distances. [Phase 5](#phase-5--measure-m-by-simulation-on-the-batchs-active-groups) is that
work; Phase 6's validation stays blocked until a measured matrix is scored against the same
three references.

### Phase 5 — implemented 2026-09-14; measurement works, the inversion does not yet

Measurement, per the design below. Nextflow wiring is deliberately *not* done: Phase 4's
lesson is to validate before wiring.

- `bin/select_active_amplicons.py` picks the active groups by direct sequence comparison
  (observed label groups plus everything within `--tau` by the kmer neighbour search).
  SC2200627-SC3: 5,949 of 86,557 groups in 84 s.
- `subspecies_infer.build_mismapping_grouped` measures `S[A, B]` at V4-group level, exposed
  as `infer_composition.py --build-mismapping --build-grouped`. One simulation per distinct
  amplicon covers its duplicates. `--active-amplicons` keeps an identity row for a group the
  simulator had to skip (IUPAC codes: 321 of 5,949 here, carrying 0.0096% of observed reads).
- `sparse_matrix.read_grouped` now accepts all-zero rows — a matrix measured on one batch's
  active groups has a row only where reads were simulated (86,551 of 86,557 rows are empty).
- Tests: group-level rows and row-stochasticity, the identity fallbacks, active-set selection
  including the one-edit neighbour, and the CLI build path.
- Cost and full results: `dev/gtdb_measured_matrix.md`, `dev/gtdb_measured_matrix_truth.csv`.
  Per batch, locally, about 20 minutes plus a one-off `.mscluster` build.

**The measured matrix is a better forward model.** Against a benchmark sample of known
composition mapped to full GTDB, pushing the truth through it reproduces the observed labels
to TV 0.055, against 0.204 for the truth against the labels directly. It captures what the
kernel could not: `v4g_2acb…` sends 88-97% of its reads to `v4g_81c3…`, and the ground truth
confirms `2acb` is present at 6.4% while MAPseq labels it zero.

**The inversion is now the bottleneck.** Sources outnumber observed labels (1,169 against
~240), so the fit is underdetermined:

| Inference | Matrix | TV to truth | Mass outside truth | `2acb` (truth 0.064) |
|---|---|---:|---:|---:|
| none (raw labels) | — | 0.209 | 0.188 | 0.000 |
| vi, no gate | alignment kernel v2 | 0.221 | 0.194 | 0.000 |
| vi, no gate | measured, sub 0.0025 | 0.328 | 0.327 | **0.062** |
| vi + presence gate | measured, sub 0.0025 | 0.231 | 0.206 | 0.000 |

A relabelled group *can* be recovered — `2acb` at 0.0615 against 0.0644 — from its secondary
leaks alone, but only without the gate; the hard gate prefers the single-source explanation
and collapses it to 0.0002. Unregularised, 33% of the mass lands on groups absent from the
truth. This is what the horseshoe trial is for.

**Also established here:** the simulated error rate now matters (it sets the secondary leaks
the recovery depends on). At 0.5% the model over-predicts `2acb`'s leak onto its neighbour
by 2x; 0.25% matches. Calibrating it belongs in the method, not in the defaults.

**Horseshoe trial — done 2026-09-14** (`dev/horseshoe_trial.md`). Added as
`infer_composition.py --horseshoe` (vi only, opt-in, not wired to Nextflow). On S01 at
10,000 steps it gives TV to truth 0.233 (outside-truth mass 0.209) and recovers `2acb` at
0.0648. At the same step count the Dirichlet without the gate gives 0.462, and the gate
gives 0.228 but deletes `2acb`. Groups outside the truth above 0.1% drop from 38 to 4.
It still does not beat the raw labels (0.205). The remaining 0.19 sits on one-edit pairs:
- `a3e62`/`89c3`: indistinguishable measured rows, an exact block.
- `87ad`/`125a`: rows differ only by a ~2% leak measured from 500 reads.
- `a49d` -> `2a9c`: real reads are relabelled onto `2a9c` at 3.2%, simulated reads never.

Shrinkage has done what it can; what is left is identifiability and simulation fidelity.

**Pruning made exact — done 2026-09-14.** Pruning now keeps every reference with a direct
hit. Observed labels with an empty matrix row get an identity row. Mass a kept row sends
to a pruned label goes to a zero-count sink label instead of being renormalised away, so
the pruned likelihood equals the full one. On S01 the horseshoe's TV to truth went from
0.233 to 0.659 (fit check `model_misfit`); the alignment kernel was unaffected (0.217).
The cause is the measured matrix: truth groups' rows leak a median 1.0% onto labels that
drew no reads (835 expected reads, 0 observed). Ten truth groups are swapped for
unobserved neighbours whose 500-read rows show no such leak (`dev/horseshoe_trial.md`).
The earlier 0.233 hid this misfit. Calibrating a measured row's leak (error rate, reads
per group) and reporting near-identical sources as blocks are now prerequisites, not
refinements.

**Error-rate calibration and 5,000 reads per group — done 2026-09-14**
(`dev/error_rate_calibration.md`). The flat model's indel rate was 10× too high. MAPseq
places simulated reads carrying an indel on distant labels that real reads never reach.
Calibrated from S01's own `.mseq` (`dev/error_rate_calibration.py`: substitution 0.0031,
indel 0.000024 each), TV to truth falls from 0.609 to 0.220. Predicted mass on labels
with no reads falls from 0.99% to 0.24%. More reads at the old rates changed nothing
(0.659 -> 0.609).

A flat rate still misses context-specific relabels: `a49d` -> `2a9c` is 0.08% flat,
22.9% trained and ~42% in real reads. The skiver model trained on the batch reproduces
them and gives the best block-level score, 0.107 against 0.106 for the raw labels. Every
fit is still `model_misfit` and `2acb` is not recovered. The measured matrix should use
the trained error model. The next step is block-level reporting.

**Not yet done:** block-level reporting (now the main gap); more reads per group (500 gives
a 1% leak an SE of 0.45%); why simulated `a49d` reads do not follow real ones;
error-rate calibration; Nextflow wiring; the 50-sample re-run once the inversion is fixed
(with the measured matrix and no gate it is 0/50 `ok`, and the ASV-profile comparison is
not a gold standard anyway).

## Evidence

All numbers below were measured on 2026-09-11 against the archive at
`~/Documents/mimicc/fermentor-run-reports_rendered/SC2200627-SC3` (GTDB matrix key
`44ac592b…`, `align/kmer`, `tau=1`, decay `auto` = 0.005895; DADA2 ASVs from
`SC2200627-SC3_results.tar.gz`). The GTDB-mapped `.mseq` files and GTDB posterior draws are
no longer in the archive, so observed V4-group counts were reconstructed by placing ASVs on
GTDB V4 sequences by exact match. Phase 1 commits the scripts.

### Archive facts (unchanged from the previous revision)

| Property | Full GTDB run | Custom-run control |
|---|---:|---:|
| Amplifiable V4 references | 1,001,241 | 79 |
| Genomes | 516,891 | 22 |
| Distinct V4 sequences | 86,557 | 24 |
| Largest exact-V4 group | 99,590 references | 6 |
| Median high-depth raw-reference TV of the archived posterior | 0.9497 | 0.0135 |
| Median high-depth V4-group TV of the archived posterior | 0.7840 | 0.0116 |
| Median active genomes fitted per sample | 61,856 (max 156,163) | ≤ 22 |

### 1. The forward model can fit the data

Exact-match placement covers a mean 99.95% of ASV reads (minimum 99.23%); novel sequence is
negligible in this batch. For each of the 50 samples with ≥ 1,000 ASV reads, the
maximum-likelihood mixture (EM) under the **archived** matrix gives:

| Inference space | Best-achievable V4-group TV, median (range) |
|---|---:|
| Genome (the model the pipeline fits; 516,891 genomes, active subset per sample) | 0.0062 (0.0009–0.0204) |
| V4 group | 0.0056 (0.0009–0.0096) |

The archived fit reached 0.784 in the same model space. The previous revision's
matrix-scale sweep (best raw TV 0.856 at `s = 0`) reused that posterior mean, so it could
not separate a bad forward model from a bad fit.

### 2. Genome-space inference fails on data generated by the model

Sample D2's ASV group counts were pushed through the archived kernel (`y @ K`), and each
read was assigned to a uniformly chosen member reference of its label group. MAPseq splits
exact ties evenly (custom run: 569/562/555/534/569 reads over five identical copies).
Phase 4 showed that this does not hold at GTDB scale: MAPseq splits evenly over a few
members only (20 of 445). The
real `infer_composition.py` and `check_composition_fit.py` were then run on that `.mseq`.
The forward model is correct by construction.

| Run (D2, 81,721 reads, 8,653 genomes fitted) | Group TV | Group PPC percentile | Per-draw TV median (observed / replicated) | Status |
|---|---:|---:|---:|---|
| Defaults | 0.0193 | 1.000 | 0.0351 / 0.0069 | `model_misfit` |
| Initial-point floor `1e-4/G` | 0.0168 | 1.000 | 0.0312 / 0.0068 | `model_misfit` |
| + `alpha 0.05` | 0.0127 | 1.000 | 0.0262 / 0.0057 | `model_misfit` |
| + no presence gate | 0.0465 | 1.000 | 0.0583 / 0.0083 | `model_misfit` |
| + `--mode mle` | 0.0786 | 1.000 | — / 0.0063 | `model_misfit` |
| **"genome" := V4 group** | **0.0071** | 0.988 | 0.0120 / 0.0055 | **`ok`** |

Rows 1 and 6 are reproduced by `dev/gtdb_failure_diagnosis.py replay`; rows 2–5 are
from an earlier ad-hoc run (row 6 was 0.0066 there, with a different random draw).

`alpha = 1e-3` produced NaN guide parameters. Mechanism:

- **Prior mass:** `alpha = 0.5` per genome is 0.5 × 61,856 ≈ 31k pseudo-reads at the
  archive's median active set against ≈ 70k reads. Without the gate this mass spreads over
  all active genomes (row 4).
- **Presence gate:** within a group of indistinguishable genomes the gate follows its 0.01
  prior. The hard threshold in `mle` mode then deletes groups that are present (row 5), and
  temperature-1 Concrete noise makes posterior draws disagree with the data (the
  per-draw column).
- **Initial point:** `theta_init = clip(theta_obs, 1e-4)`. Reads split over thousands of
  members leave most genomes at the floor: 40% of initial mass in the reconstruction, and
  about 6.2 units against 1 unit of signal at 61,856 genomes. Fixing it alone is not
  sufficient (row 2).

The replay touched 8.6k genomes; the archived runs touched a median of 61.9k, where each of
these effects is larger. Group-space inference passed only narrowly (0.988 against a 0.99
tail) with the size-weighted kernel still in place.

### 3. The kernel over-weights duplicated neighbours

MAPseq 2.1.1b (pinned container) was run with 20,000 reads simulated from a GTDB V4
sequence A (flat 0.5% substitution, 0.05% insertion, 0.05% deletion) against a database of
A, a one-substitution neighbour B repeated `k` times, and 200 random GTDB background
sequences:

| B copies `k` | MAPseq reads assigned to B | Built kernel `k·c/(1+k·c)` | Per-distinct kernel `c/(1+c)` |
|---:|---:|---:|---:|
| 1 | 1.40% | 0.59% | 0.59% |
| 10 | 1.54% | 5.57% | 0.59% |
| 100 | 1.92% | 37.1% | 0.59% |

MAPseq's leak depends only weakly on multiplicity. The built kernel is wrong by about 20× at
`k = 100`. The per-distinct kernel has the right shape but under-predicts the magnitude by
about 2.4–3.3×; that is the per-sample error rate that `--infer_distance_decay` fits.

Across the GTDB kernel (86,557 rows):

| Row self-mass | Built kernel | Per-distinct kernel |
|---|---:|---:|
| < 0.99 | 21,809 (25.2%) | — |
| < 0.10 | 3,325 (3.8%) | 223 |
| 5th percentile | 0.22 | 0.97 |

Of the 3,325 rows below 0.1, 3,248 are explained by multiplicity alone and 13 need the
ambiguity weight.

### What this changes

| Previous claim | Status |
|---|---|
| "The current GTDB run is a failed forward model, not merely a difficult optimisation." | Refuted by 1 and 2. |
| Pruning to candidates is required for calibration. | Not needed. Group-space active subsets are a few hundred groups (median 240 source groups per sample). |
| Candidate re-mapping creates one clean observation space. | The full-GTDB MAPseq output already is one observation space. Re-mapping adds a second. |
| V4-equivalent genomes are not identifiable; report blocks and LCA. | Confirmed. This is now the inference space itself, not a post-hoc collapse. |
| PPC must be a release gate. | Confirmed, at V4-group level only. |

## What V4 can and cannot resolve

GTDB species are defined from whole-genome ANI
([Parks et al.](https://academic.oup.com/nar/article/50/D1/D785/6370255)), not from V4
separation. Exact-V4-equivalent genomes cannot be distinguished by this assay; short
variable-region 16S resolves less than full length
([Johnson et al., 2019](https://doi.org/10.1038/s41467-019-13036-1)). MAPseq reports a
confident top hit among ties, but it does not add information absent from the read
([Rodrigues et al., 2017](https://doi.org/10.1093/bioinformatics/btx517)). Therefore:

- the identifiable unit is a distinct V4 sequence, or a block of sequences the kernel cannot
  separate;
- a genome inside a multi-genome V4 group has no identified abundance of its own;
- `presence_prob` must not be used to choose a member of a group.

## Design

### Estimand and inference space

The default estimand is **V4 amplicon-copy abundance**: the fraction of reads originating
from each distinct V4 sequence. A read derives from an rRNA gene copy, not directly from an
organism. The inference space is one parameter per exact distinct V4 amplicon (byte
identity after primer trimming), named `v4g_<first 16 hex of sha256(sequence)>`. That id is
stable across runs, batches, and database versions that contain the same sequence.

Genome, species, and LCA labels are annotations derived from group membership. Organism
abundance requires a validated copy-number model and is out of scope (see
[Conditional work](#conditional-work)).

### Kernel

`M[a, j]` for true source reference `a` and observed label `j`:

```
w_B       = c^d(A, B) · amb(B)       for each distinct sequence B within tau of A (w_A = amb(A))
M[a, j]   = w_{B(j)} / (Σ_B w_B) / size(B(j))
```

A distinct neighbour's share is independent of how many references carry it; within a
sequence the share is split evenly, matching MAPseq's tie behaviour. Row-stochasticity is
`Σ_B S[A, B] · size[B] = 1`, unchanged from the grouped format. `amb(B) = w^k` is the
existing ambiguity weight. The distance-decay re-parameterisation
`M(c) = rownorm(M(c0) · (c/c0)^d)` is unaffected because the `1/size` factor is constant in
`c`. `simulate`-built matrices are measured and unaffected.

### Release gate

`fit_status` is decided by the V4-group PPC percentile, with provisional tails 0.01 and
0.99. Raw-reference TV and percentile remain in `fit_diagnostics.json` as diagnostics only.
Status values: `ok`, `low_depth`, `model_misfit`.

### Output contract (`infer_space = v4_group`)

1. `<id>.inferred_v4_groups.csv` is the space-native result and the input to the fit check.
   Columns: `sample, v4_group_id, n_references, n_genomes, lca, representative_reference,
   observed_rel_abundance, inferred_mean, inferred_lo, inferred_hi, presence_prob,
   fit_status`. `lca` is the longest common prefix of member lineages, or
   `unclassified_v4_group` when `--taxonomy` is absent or the prefix is empty.
2. `<id>.inferred_composition.csv` keeps the benchmark contract: one row per genome and a
   finite `inferred_mean` for every genome (`normalize_sr_profile.py` rejects empty values,
   and `nan` would poison its total). Each group's mass is split over its member references
   and summed per genome. This is a **labelled convention, not an estimate**. Added columns:
   - `v4_group_ids` (`;`-joined);
   - `resolution`: `identifiable` when the genome is the only genome in every group it
     touches, otherwise `not_identifiable`;
   - `inferred_lo` and `inferred_hi` are empty for `not_identifiable` rows.
3. `<id>.posterior_draws.npz` stores the group ids in `genome_ids` and a new scalar
   `infer_space`. Archives without it are read as `genome`.
4. `inference_diagnostics.csv` adds `infer_space`, `n_groups_fitted`, and
   `max_genomes_per_active_group`.

Consumers must use `inferred_v4_groups.csv` plus `fit_diagnostics.json` for interpretation,
and must exclude `low_depth` and `model_misfit` samples from comparative plots.

### Provenance

- Matrices carry `kernel_version` (2 for per-distinct weighting) in the `.npz`, in
  `provenance.json`, and in the `MATRIX_KEY` settings hash, so a size-weighted matrix is
  never silently reused. A matrix without the field is version 1: it still loads for
  reproducibility, with a warning.
- LCA uses the supplied `--taxonomy` file. Its sha256 is recorded in
  `inference_diagnostics.csv`.

## Implementation plan

Phases 1–3 are code changes, in that order. Phase 2 and Phase 3 touch different files and
may be developed in parallel, but Phase 4 needs both.

### Phase 1 — correct the gate and record the evidence

1. `bin/check_composition_fit.py`, `run`: set `fit_status` from the `unique_v4_group`
   percentile only. Add `"gate": "unique_v4_group"` to the payload.
2. `tests/test_composition_fit.py`: add a case where raw-reference counts are concentrated
   on one member of a duplicate group (group totals exact). It must return `ok`; the
   existing perturbed-group case must still return `model_misfit`.
3. Commit the diagnosis as `dev/gtdb_failure_diagnosis.py` and `.md` (ASV placement,
   best-achievable EM fit, forward-model replay), plus `dev/mapseq_multiplicity.py` and
   `.md` (the Evidence §3 experiment). Archive paths are arguments, not constants.
4. Correct the two documents that repeat the refuted claim: README "GTDB-scale safety" and
   the callout in `docs/alignment_mismapping_plan.md`. Both should say the genome-space fit
   is unsuitable at GTDB scale and point here.

**Acceptance:** the unit tests pass. Re-running the Phase 0 check on the D2 replay with the
V4-group configuration gives `ok`.

### Phase 2 — per-distinct-sequence kernel

1. `bin/build_mismapping_align.py`, `build_kmer_grouped`: replace
   `mass = sizes * weights` with per-distinct weights. `per_row = adjacency @ w` (with
   `w = 1` when ambiguity weighting is off), and
   `data = adjacency.data * w[indices] / sizes[indices] / repeat(per_row)`. Keep the
   all-ambiguous fallback, using `adjacency @ 1`.
2. `tie_cluster_matrix` / `build` (reference-square path): divide each column by the number
   of references whose amplicon is byte-identical to that column's before `_normalise`, so
   the minimap2 backend matches the grouped backends.
3. `build_exact_grouped`: unchanged (it has only the self group).
4. `bin/sparse_matrix.py`: `write_grouped` and `write_matrix` gain `kernel_version`.
   `read_grouped` and `read_matrix` return it, or warn on absence. Record it in
   `bin/write_mismapping_bundle.py` / `provenance.json` and add it to the `MATRIX_KEY`
   settings in `modules/local/matrix_key/main.nf`.
5. `DecayKernel` and `check_composition_fit._kernel_at_decay`: no algebraic change; covered
   by a test.

**Tests** (`tests/test_grouped_mismapping.py`):

- sequence A with a one-edit neighbour B duplicated `k ∈ {1, 10, 100}` gives a leak of
  `c·amb/(1 + c·amb)` for every `k`;
- rows satisfy `Σ S[a, b] · size[b] = 1`;
- the grouped and reference-square builds agree on a small set with duplicates;
- `DecayKernel` re-decayed from `c0` to `c` equals a direct build at `c`;
- the matrix key changes with `kernel_version`.

**Acceptance:**

- `dev/alignment_mismapping.py` on the B. uniformis set stays within its seed-to-seed floor;
  record the new `‖M_align − M_sim‖_F`.
- In `dev/mapseq_multiplicity.py`, the kernel leak ratio between `k = 100` and `k = 1`
  lies in [0.5, 2]. MAPseq's measured ratio is 1.37.
- Rebuilding GTDB r232 at `kmer`, `tau = 1` keeps runtime and memory within 1.5× of the
  published 79 s / 0.8 GB.

### Phase 3 — V4-group inference space

1. `bin/infer_composition.py`:
   - add `--infer-space {genome,v4_group}` (default `genome`) and `--taxonomy PATH`
     (MAPseq `.tax`: `header<TAB>lineage`);
   - move `_v4_groups` from `check_composition_fit.py` here as the single implementation,
     returning stable `v4g_` ids;
   - in `run`, when `v4_group`, replace `genomes`, `translation`, and `g_of_ref` with group
     space: translation weight `1/size`, one "genome" per group. Pruning, fitting, and
     expansion then run unchanged;
   - write the Output contract files. The genome table is derived from the group
     posterior mean and the reference-to-genome membership;
   - in `genome` space, if the active subset contains a V4 group spanning more than 100
     genomes, log a warning recommending `v4_group` and record
     `max_genomes_per_active_group`.
2. `bin/check_composition_fit.py`: read `infer_space` from the draws and build the
   translation in that space through the shared helper. `--composition` receives
   `inferred_v4_groups.csv` in `v4_group` mode.
3. Nextflow:
   - `params.infer_space` (default `genome`) and `params.taxonomy` (default `null`) in
     `nextflow.config`;
   - pass `--infer-space` and `--taxonomy` in `conf/modules.config`, staging the taxonomy
     as a path input (`[]` when null);
   - `INFER_COMPOSITION` emits optional `v4_groups`, and the workflow routes that table to
     `CHECK_COMPOSITION_FIT` when `infer_space == 'v4_group'`;
   - publish it under `composition/<id>/`;
   - add the new params to the README parameter tables.
4. Tests:
   - unit: stable group ids; two genomes with byte-identical amplicons plus one distinct
     genome give two groups, the shared genomes marked `not_identifiable`, and genome
     `inferred_mean` values that are finite and sum to 1;
   - `infer_composition.py --demo` passes in both spaces;
   - nf-test stub asserts `inferred_v4_groups.csv` in `v4_group` mode;
   - `normalize_sr_profile.py` accepts the v4-mode genome table.

**Acceptance:**

- Forward-model replay (`dev/gtdb_failure_diagnosis.py`) with the version-2 kernel and
  `v4_group` over all 50 high-depth SC2200627-SC3 samples: at least 95% are `ok`, and the
  median group TV is at most 1.5× the median replicated TV.
- The custom-database run is unchanged in `genome` space (regression on its composition
  and fit status).

### Phase 4 — replay the archive on real GTDB MAPseq output

1. Build the MAPseq `.mscluster` for the GTDB r232 V4 amplicon set once (HPC; record cost).
   Map all 54 SC2200627-SC3 samples with the pinned MAPseq and the pipeline's read
   preparation.
2. Run the pipeline with `infer_space = v4_group`, `kmer`, `tau = 1`, kernel version 2,
   `infer_distance_decay = true`, and `--taxonomy ssu_all_r232_ssu.sr_refs.tax`.
3. Compare against three references:
   - the ASV exact-match group profile (an independent denoiser);
   - the custom-database run, aggregated to V4 groups by exact sequence;
   - the forward-model replay from Phase 3.

**Acceptance:**

- at least 95% of high-depth samples are `ok`;
- median group-level TV to the ASV profile is at most 0.05;
- median group-level TV to the custom-database run is at most 0.05;
- per-sample runtime and memory are recorded.

Any failure is investigated before Phase 5; see [Conditional work](#conditional-work) for
the pre-agreed escalations.

### Phase 5 — measure M by simulation on the batch's active groups

**Decision (2026-09-14).** `M` is measured by simulating reads and mapping them with the
same MAPseq and database the real reads go through — not built from alignment distances.
Simulating the whole GTDB set (1,001,241 references) is unnecessary: only the groups a
batch's reads could have come from need a row, so simulation is restricted to the batch's
**active groups**, and the active set is chosen by **direct sequence comparison** (the
existing kmer/exact-hash neighbour search), not by any mapping.

**Intention for later.** Estimating `M` by direct sequence comparison alone — no
simulation, no mapping — stays the goal: it is what makes a matrix reusable across batches
and cheap at database scale. Phase 4 showed the current alignment kernel does not reproduce
MAPseq at GTDB scale (it relabels exact hits, splits ties over a subset of members, and
ignores IUPAC ties), so a better sequence-comparison estimator is a research problem in its
own right. Establish the rest of the method on a measured matrix first, then use the
measured matrices from real batches as the calibration target for that estimator. Until
then the kernel keeps its existing role: defining the active set, which needs only
reachability, not calibrated leak.

Design:

1. **Order.** The matrix now depends on the batch's observed labels, so it is built after
   `MAPSEQ_OBS` instead of beside it. One matrix per batch (per matrix-key group), not per
   sample.
2. **Active set.** Union the observed label groups over the batch's samples, then add every
   group within `align_tau` of one by the kmer neighbour search — the same "could a read
   from here have landed there" reachability `_active_subset` already uses, applied at
   build time. On SC2200627-SC3 that is a median 1,225 groups per sample and a few thousand
   per batch, against 86,557 distinct V4 sequences.
3. **Simulate per group, not per reference.** Byte-identical references produce identical
   reads, so one representative per active group is simulated (`sim_n_per_ref` reads each).
   A few thousand groups x 500 reads is a few million reads, against 500 million for the
   whole set per reference.
4. **Map against the full database.** The simulated reads go through MAPseq against the
   complete GTDB amplicon set with its `.mscluster`, because the confusion being measured
   includes labels outside the active set (2acb -> 81c3 is exactly that).
5. **Build at group level.** `M[A, B]` is the fraction of reads simulated from group `A`
   that MAPseq labels in group `B`, stored in the grouped format (`S[A, B] = M[A, B] /
   size(B)`) so every member of an active group shares its row and the `v4_group` fit is
   unchanged. A group with no mapped simulated read keeps an identity row. Non-active
   groups have an empty row and prune away.
6. **Provenance.** A measured matrix carries no distance strata, so `--infer-distance-decay`
   does not apply to it; the simulated error rate now sets the leak magnitude directly, and
   the `MATRIX_KEY` settings hash must include the active-set digest, because the matrix is
   no longer a pure function of the reference set.

**Regularisation (decided 2026-09-14, not yet implemented).** Trial **continuous shrinkage
(a horseshoe) on unnormalised weights** — `lambda_i ~ HalfCauchy(1)`, a global `tau`,
`w_i ~ HalfNormal(tau * lambda_i)`, `theta = w / sum(w)` — in place of both the Bernoulli
presence gate and a sparse Dirichlet. Rationale:

- an L1 penalty is meaningless on the simplex (`sum(theta) == 1` makes it constant), and L2
  shrinks towards uniform, which is the wrong direction;
- `alpha << 1` is the simplex-native sparsity knob and does help (D2, measured matrix: TV to
  the ASV profile 0.797 at `alpha 0.5` against 0.359 at `alpha 0.05`), but it is numerically
  fragile — `alpha = 1e-3` produced NaN guide parameters in Phase 3;
- the presence gate's temperature-1 Concrete noise is what fails the posterior-predictive
  check (Phase 3), and a horseshoe is fully reparameterisable, so it buys sparsity without
  that noise.

Regularisation is secondary to identifiability: inside a block of sources the mapper cannot
separate, the likelihood is flat and any prior merely picks a point in it. Shrinkage is for
the *between-block* underdetermination.

**Cost to record.** The GTDB `.mscluster` must be built once on HPC (the archived run's work
directory has been cleaned). Per batch: one simulation, one MAPseq run over a few million
reads, and the build.

**Acceptance:** the Phase 4 comparison, re-run with the measured matrix — at least 95% of
high-depth samples `ok`, median group-level TV to the ASV profile and to the custom-database
run at most 0.05 each, and per-sample cost recorded. The same script (`real`) scores it.

### Phase 6 — validate before production use

1. Mock communities with close GTDB relatives (including one-edit V4 neighbours of large
   duplicate groups), known negatives, and a dynamic range spanning the expected
   abundances.
2. Hold out reads or technical replicates. Freeze the matrix and configuration before
   scoring held-out data; check group-level calibration and held-out deviance.
3. Release the mode only if every positive and negative control passes, and at most 5% of
   assay-valid high-depth benchmark samples are flagged. Freeze or revise these provisional
   thresholds once, from an independent calibration cohort, before interpreting a
   production cohort. Version the benchmark with the GTDB release and MAPseq version.
4. Benchmark follow-up (synthetic-metagenomic-benchmark-pipeline): score `sr_amplicon` at
   V4-group or LCA level in addition to genome level. Genome-level scores against GTDB
   measure the split convention, not the assay.

## Proposed configuration

```yaml
params:
  infer_space: v4_group            # genome | v4_group; GTDB runs must use v4_group
  taxonomy: /path/ssu_all_r232_ssu.sr_refs.tax
  mismapping_method: align
  align_backend: kmer
  align_tau: 1
  align_distance_decay: auto       # starting value; refined per sample below
  infer_distance_decay: true       # MAPseq leak is ~2.4-3.3x the flat-rate c (Evidence §3)
  infer_presence: false            # the gate's Concrete noise fails the PPC (Phase 3)
  min_infer_reads: 1000
```

`alpha` stays at 0.5 per group. Median active set: 240 groups, about 120 pseudo-reads.

## Conditional work

Deferred items, each with the observation that would justify it:

| Work | Trigger |
|---|---|
| Organism-level blocks: merge genomes whose genome→group copy profiles are identical (the exact non-identifiability classes of `θ·T·M`) and infer in that space | A user needs organism rather than amplicon-copy abundance. Needs a validated copy-number model. |
| Near-equivalence blocks from kernel-row distance, calibrated by mocks | Phase 4/5 shows strongly anti-correlated group posteriors (correlation < −0.5) or wide, unstable intervals for one-edit neighbour pairs. |
| Screen database + batch candidate database + candidate re-mapping | MAPseq against full GTDB is too expensive per sample, or active sets exceed memory. This is a cost measure, not a statistical requirement. |
| ~~Simulated-matrix calibration suite (`simulate → MAPseq`)~~ | **Triggered by Phase 4 and adopted as [Phase 5](#phase-5--measure-m-by-simulation-on-the-batchs-active-groups).** |
| Explicit unknown / out-of-database component | Exact-match coverage falls below 98% of reads in a batch (it is 99.95% here). |
| Curated inoculum panel as an exclusive prior | A defined inoculum is to be resolved inside a multi-genome group. It needs experimental validation and a versioned TSV. |

## Non-solutions and guardrails

- Do not infer in genome space against GTDB-scale reference sets. A genome-space fit fails
  its own PPC on correct-by-construction data.
- Do not read a PPC or TV failure as forward-model failure without the best-achievable fit
  (`dev/gtdb_failure_diagnosis.py`): compare the fitted TV with the TV attainable under the
  same matrix.
- Do not choose a genome from a multi-genome V4 group by MAPseq top hit, posterior mass, or
  `presence_prob`. The genome table's split is a labelled convention.
- Do not weight a neighbour by its duplicate count (kernel version 1).
- Do not score against MAPseq top assignments as independent evidence of genome origin.
- Do not equate GTDB's genome-resolved taxonomy with V4 resolution.

## Decisions required before Phase 3

1. **Default `infer_space`.** Recommended: keep `genome` as the default, preserving the
   custom-database and benchmark behaviour, and require `v4_group` for GTDB via the
   warning plus documentation. The alternative is to switch automatically when any active
   group spans more than 100 genomes.
2. **Genome-table convention in `v4_group` mode.** Recommended: the reference-proportional
   split labelled `not_identifiable`, as specified. The alternative is to omit
   non-identifiable genomes, which needs a benchmark change first.
3. **`infer_distance_decay` default for GTDB runs.** Recommended: on, per Evidence §3.

## Caveats on the evidence

- Observed counts were reconstructed from DADA2 ASVs, not from GTDB MAPseq output. The
  forward-model replay covers one sample (D2). Phase 4 removes both limitations.
- The MAPseq multiplicity experiment used one sequence pair, one error model, and 20,000
  reads per database.
- Reconstructed active sets (median 7,252 genomes) are smaller than the archive's (61,856),
  because denoised ASVs lack the error reads that widen reachability. The genome-space
  defects grow with the active set, so the replay understates them.
