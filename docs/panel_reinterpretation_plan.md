# Reinterpreting generic-database MAPseq output with a genome panel

Status: plan, 2026-09-14. Phase 0 done (`dev/panel_reinterpretation.py`,
[results](#phase-0-results)). Phases 1–4 done ([notes](#phase-1-notes),
[notes](#phase-2-notes), [notes](#phase-3-notes), [notes](#phase-4-notes)). Phase 4b (align
kernels, and all 20 sweep samples compared with the 20HM custom database) done
([results](#phase-4b-results)). Phase 5 done ([results](../dev/panel_reinterpretation.md)):
with no presence gate the trained kernel reaches TV 0.0149; deletion passes on all 12 arms;
no-simulation arms still do not beat home labels. Phase 6: Nextflow wiring done
(`panel_references`, `PANEL_PREPARE` → `MAPSEQ_PANEL_HOME` / `SIMULATE_PANEL_READS` →
`MAPSEQ_PANEL_SIM` → `PANEL_KERNEL`, published as `mismapping/panel_<key>/`); the rest
not started.

**Headline (Phase 4b).** On 20 samples, genome TV to truth has these medians:
- 20HM genomes as the MAPseq database: **0.009–0.012**, and 0.010 with MAPseq alone, no
  inference (BU strain split error 0.12, max 0.52).
- GTDB labels with a simulated trained-model panel kernel: 0.019.
- Every alignment panel kernel: 0.039–0.052, no better than home labels alone (0.039).

The custom database is better on every sample. Reinterpretation is a fallback for when
re-mapping against the panel is not possible. It is not a replacement for it.

## Goal

Reads are classified once by MAPseq against a large generic database (GTDB, SILVA). A
user also has a **panel**: a small set of genomes, each with one or more 16S copies, that
the sample is believed to contain (a mock or defined community, an inoculum, an isolate
collection). The goal is to infer the panel's genome abundances from the generic-database
labels, without re-mapping reads against a panel-only database.

The existing method already inverts MAPseq's confusion, but only between members of the
database it mapped against: sources and labels are the same set. The panel case has
sources from one set (panel amplicons) and labels from another (database V4 groups). This
plan makes the confusion matrix rectangular, so both cases are one method, and tests it on
S01 against GTDB r232 with the 20HM panel.

## Why this should work better than GTDB-space inference

GTDB-space inference on S01 is underdetermined. It has 1,169 candidate sources (264 after
pruning) for 49 observed labels (769 distinct hit references), and after the horseshoe, error calibration and exact pruning, its best
TV to truth is 0.22. That only ties the raw labels, and the fit check says `model_misfit`
(`dev/horseshoe_trial.md`, `dev/error_rate_calibration.md`). The panel removes that
problem. It has **26 distinct V4 sources for 22 genomes**, against the same 49 labels.

Facts checked for this plan:

| Fact | Value |
|---|---|
| Panel genomes / distinct V4 amplicons (515F/806R, ≤2 primer mismatches) | 22 / 26 |
| Panel V4 amplicons present byte-identical in GTDB r232 | 26 / 26 |
| V4 amplicons shared between panel genomes | 1 (the *B. uniformis* pair) |
| *B. uniformis* strains | share 1 amplicon; `BU_JCM13286_NT5170` also has 1 of its own |
| S01 truth | 21 present; `bacteroides_uniformis` absent, `strain2` at 5.1% |

The cases GTDB space could not separate should now separate:

- **`2acb` is relabelled onto `81c3` 88–97% of the time.** In GTDB space `81c3` is also a
  candidate source, so the relabel is an exact block. In panel space `81c3` is not a
  source, so reads there must come from `2acb`.
- **`a3e62`/`89c3`** (identical measured rows) is the same story: `89c3` is not in the
  panel.

What is left is the panel's own identifiability (the BU pair, through one copy) and
simulation fidelity.

## The unified model

Every case is described by four things:

| Symbol | Meaning |
|---|---|
| sources `S` | distinct amplicon sequences reads are simulated from, named `v4g_<sha16>` |
| labels `L` | database V4 groups MAPseq reports (a hit's header → its exact-sequence group) |
| `K ∈ R^{S×L}` | measured: fraction of reads simulated from source `s` that MAPseq labels in group `l` (row-stochastic) |
| `h(s) ∈ L` | the **home label**: the group MAPseq assigns the error-free source sequence to |
| `T ∈ R^{G×S}` | genome → source copy weights (the existing compact translation table) |

Forward model, unchanged apart from the diagonal becoming the home entry:

```
r_src = θ · T
r_lab = Σ_s r_src[s] · rownorm((1 − s_scale)·e_{h(s)} + s_scale·K[s])
y ~ DirichletMultinomial(N, conc · r_lab)       over observed labels + sinks
```

The existing cases are specialisations:

| Case | Sources | Labels | Home |
|---|---|---|---|
| custom DB, reference-square (`simulate`, `align`) | references | references | identity |
| GTDB `v4_group`, measured or kernel | active DB groups | all DB groups | identity |
| **panel reinterpretation** (new) | panel distinct amplicons | DB groups | MAPseq of the source sequence |
| panel + measured background (conditional) | panel ∪ DB groups near unexplained labels | DB groups | per row |

Sources carry `v4g_` ids, so a panel amplicon that exists in the database has the same id
as its database group. When the panel *is* the database's active set, the new case reduces
exactly to the existing grouped case. That is the regression test.

### Where the square assumption lives today

These are the only places that need generalising. Each takes a `home` index array, and
`home = arange(n)` restores current behaviour:

- `subspecies_infer._apply_sparse`: `diagonal` → `K[s, h(s)]`; the correction term is
  scattered to `h(s)` instead of added in place.
- `check_composition_fit._apply_mismapping`, `_subset_context`, `_matrix`: same change,
  in numpy.
- `infer_composition._subset_mismapping`: keep all sources and the observed labels. Each
  row's mass on the remaining labels goes to sink labels, as now. The sink proof is
  unchanged: merging zero-count labels is exact.
- `infer_composition._active_subset` / `_identity_rows_for_observed`: with a rectangular
  `K`, sources are pruned by "has mass on an observed label". An empty row stays a
  source no read can come from. It does not get an identity row; that fallback is square-only.
- `DecayKernel`: measured kernels have no distance strata. Refuse `--infer-distance-decay`
  on a rectangular kernel, as already happens for a measured grouped matrix.

The reference-level `group` scatter used by genome-space grouped matrices stays as an
optional source-side factorisation. Its numerics are not touched.

### Out-of-panel reads

An observed label that no panel source reaches (`K[:, l] = 0`) cannot be explained by the
panel. All such labels are merged into one label `U`, and a **background source** `b` with
`K[b, U] = 1` is added. Panel rows' leak onto those labels also lands in `U`. This is
exact: a background source with a free row over the unexplained labels would fit their
internal distribution perfectly, so merging them loses no information. `θ_b` is reported
as `unexplained_fraction`.

Known ceiling: out-of-panel reads that land on labels a panel source *does* reach are
absorbed into panel genomes. Phase 6 exists for that case, and the out-of-panel deletion
test in Phase 5 measures whether it is needed.

## Implementation phases

### Phase 0 — data and baselines (dev only)

`dev/panel_reinterpretation.py prepare|baseline|score`, following the pattern of
`dev/horseshoe_trial.py`.

Inputs, all already local:

- Database: `work/phase5/amplicons.fasta`, `.tax`, `.mscluster` (GTDB r232 V4, 1,001,241
  references), and `work/phase5/reference/` (translation table).
- Panel: `…/sr_amp_param_sweep/export/references/amplicons/*.amplicons.fasta` (22 files,
  primers included; `extract_v4` trims them to the database's insert-only form).
- Observed: `work/phase5_truth/S01.obs.mseq` (79,468 reads, already mapped against full
  GTDB with the pipeline's own read prep).
- Truth: `…/S01_a0.00.515-YF-806BR/S01_a0.00.515-YF-806BR.truth.tsv`,
  `realized_rel_abundance`. The alias `bacteroides_uniformis_strain2=BU_JCM13286_NT5170`
  is still needed.
- Oracle error model:
  `…/error_models/sc2200627/sc2200627.model.pt`.

Baselines, all in genome space against the truth:

- **B0 ceiling:** the custom-database run, `S01…simulate.p01.sr_profile.tsv`, which mapped
  against the panel itself.
- **B1 no confusion model:** `K = e_{h(s)}` (home labels only), EM for `θ`. This shows
  what the measured `K` adds.
- **B2 GTDB space:** the best existing `v4_group` horseshoe fit (TV 0.22), scored in
  source space over the 26 panel groups.

#### Phase 0 results

B1, B2 and `raw` project onto genomes by EM (`home = identity` for B2 and `raw`). Mass on
labels no panel source reaches goes to `background`, and TV is taken over the renormalised
panel genomes. Raw rows: `dev/panel_reinterpretation_baselines.csv`.

| Baseline | Genome TV | `unexplained_fraction` | `bacteroides_uniformis` | `2acb` / `a3e62` genome recovered |
|---|---:|---:|---:|---:|
| B0 custom DB | **0.011** | 0 | 0.0001 | 1.00 / 1.00 |
| B1 home labels only | **0.033** | 0.037 | 0 | 0.99 / 1.05 |
| B2 GTDB-space horseshoe | 0.165 | 0.191 | 0 | 0.00 / 0.00 |
| raw MAPseq labels | 0.164 | 0.175 | 0 | 0 / 0 |

- 5 of 26 sources have a home label that is not themselves: `2acb → 81c3`,
  `a3e62 → 89c3`, `dd9d → 2fef`, and `6b90`/`90a5 → b9af` (all three *Veillonella* copies
  share one home). None of these homes is a different panel genome's only source, so the
  home map alone recovers `2acb` and `a3e62`.
- **B1 is already close.** Almost all of its error is `dorea_formicigenerans` (abs error
  0.031): the `a49d → 2a9c` relabel puts ~42% of its reads on a label with no panel home,
  so they go to `background`. That leak is what `K` has to add.
- B2 projected to panel genomes (0.165) scores below its group-space TV (0.22), because its
  mass on `81c3`/`89c3` becomes `background` rather than error. It is no better than the
  raw labels.
- The BU split cannot be tested on S01 (truth 0/1, every baseline gives 0/1). S10/S19 are
  needed.

### Phase 1 — backend unification (no behaviour change)

1. `subspecies_infer.tally_kernel(sim_mseq, source_of_query, label_of_hit, n_src, n_lab)`
   builds the rectangular count tally. `build_mismapping_grouped` becomes a call to it with
   `source_of_query = group[ref]` and `label_of_hit = group[ref]`, plus its existing
   fallbacks.
2. `sparse_matrix.write_kernel` / `read_kernel` add `format="rectangular"` with:
   `data/indices/indptr/shape`, `source_ids`, `label_ids`, `home`, the DB reference headers
   blob and `label_of_ref`, `db_amplicons_sha256`, per-source `n_simulated`/`n_unmapped`,
   `kernel_version`, and a provenance JSON (error model, rates, reads per source, seed,
   MAPseq version). `label_of_ref` means inference needs no database FASTA, only the
   kernel. `read_grouped` / `read_matrix` stay; a thin adapter presents them as
   `(K, home=arange)`.
3. Generalise the functions listed [above](#where-the-square-assumption-lives-today) to take `home`.

Tests (added to `tests/test_measured_mismapping.py` and `tests/test_composition_fit.py`):
- A rectangular kernel with `S = L` and `home = identity` gives the same log-likelihood and
  fitted `θ` as the grouped path, to 1e-8, on the existing fixtures.
- A rectangular sink subset equals the full rectangular likelihood (mirrors
  `test_pruning_sends_dropped_mass_to_sink_labels_exactly`).
- `_apply_sparse` with a non-identity home and `s_scale = 1` equals `r_src @ K`.
- The whole existing suite passes unchanged.

#### Phase 1 notes

Done: `si.tally_kernel` (and `build_mismapping_grouped` on top of it),
`sm.write_kernel`/`read_kernel`/`home_entries`, a `home` argument on `si._apply_sparse`,
`si._apply_mismapping` (a `(sparse, diagonal, None, home)` tuple) and
`check_composition_fit._apply_mismapping`, and `ic._subset_kernel` (every source kept, the
dropped labels merged into one sink, and a dropped home label mapped to that sink). The
five tests are in `tests/test_measured_mismapping.py`; the suite is 37/37.

Changed from the list above:
- **No `_subset_mismapping(home=…)` overload.** The rectangular subset is a separate
  `_subset_kernel`, because its sinks are labels only while the square form's sinks are
  also sources.
- **Deferred to Phase 3/4**, where a rectangular kernel is first loaded:
  `check_composition_fit._matrix`/`_subset_context`, the `_active_subset` pruning rule,
  refusing `--infer-distance-decay`, and the `(K, home=arange)` adapter. None has a caller
  yet. Until then a rectangular `.npz` passed to `--mismapping-matrix` fails with "invalid
  sparse mis-mapping matrix".
- **The regression test compares `r_obs`, not fitted `θ`.** Equal predictions for every
  `θ` and `s` make it the same model, so the likelihood and the fit are equal too.
- **Clamp difference.** The grouped path clamps `1 − s + s·M[a, a]` per reference; the
  rectangular path clamps `1 − s + s·K[s, h(s)]` per source. They agree whenever neither
  clamp binds, which holds for all `s ≤ 1`. The test uses `s` up to 1.3 with the clamp
  inactive.

### Phase 2 — the mapping step: `bin/build_panel_kernel.py`

Two subcommands, so MAPseq runs outside the script. It is the same container command now,
and later the same `MAPSEQ` process aliased as the observed reads' (per the
measure-don't-model rule).

`prepare --panel-amplicons DIR|FASTA --db-amplicons amplicons.fasta -o OUT`
- Cuts V4 with `extract_v4` using the pipeline's primers. Writes `sources.fasta` (one
  record per distinct amplicon, header `v4g_<sha16>`) and `panel_translation.tsv`
  (genome, source, copy weight).
- Records, per source, whether the sequence is present byte-identical in the database.

Then, externally:
```
mapseq sources.fasta amplicons.fasta amplicons.tax > home.mseq                   # home labels
simulate_amplicon_reads.py --amplicons sources.fasta --error-model trained ... --n-per-ref 5000 -o sim.fasta
mapseq sim.fasta amplicons.fasta amplicons.tax > sim.mseq                        # K
```

`build --prepared OUT --home-mseq home.mseq --sim-mseq sim.mseq -o panel_kernel.npz`
- Tallies `K` with `tally_kernel` and takes `h(s)` from `home.mseq`.
- Writes `panel_sources.tsv`: source, genomes, home label, identity to home, `K[s, h(s)]`,
  top-3 labels, unmapped fraction, and block id (sources whose rows are ≥99% on the same
  label; the rule from `error_rate_calibration.md`).

Tests: a toy database and panel, with hand-written `.mseq` files, covering a relabelled
home, a source absent from the database, a source shared by two genomes, and a
row-stochastic check.

#### Phase 2 notes

Done: `bin/build_panel_kernel.py prepare|build` and
`test_panel_kernel_prepare_and_build` in `tests/test_measured_mismapping.py`. The suite is
38/38.

Changed from the list above:
- **`build` also takes `--db-amplicons`.** It needs the database headers and their group ids
  to write `label_of_ref` and `db_amplicons_sha256`.
- **Two source tables.** `prepare` writes `sources.tsv` (source, genomes, in_db). `build`
  writes the enriched `panel_sources.tsv` next to the kernel.
- **Labels are all database groups**, in sorted `v4g_` order, so a label index means the
  same thing in any kernel built against the same database.
- **Fallback rows.** A source with no labelled simulated read gets a warning and the row
  `e_{h(s)}`. It stays a source, like the square form's identity row. A source with no home
  hit is an error.
- **Provenance** is whatever `--provenance key=value` records, plus the input paths and
  `min_identity`. The script cannot see the simulator's settings, so the caller passes them.
- **Block id** is the first source in the block. A source joins a block through the label
  that holds ≥99% of its row; otherwise it is its own block.

**S01 run** (`work/panel_kernel/`, one subdirectory per arm). `prepare` reproduced Phase 0's
`sources.fasta` and translation byte-for-byte, so Phase 0's `home.mseq` was reused. Each arm
used 5,000 reads per source (130k reads), about 70 s of MAPseq each. Every kernel is 26 sources
× 86,557 GTDB groups, with the same 5 relabelled homes.

| Kernel | Forward TV, truth → observed labels | `2a9c/(a49d+2a9c)` |
|---|---:|---:|
| home labels only (B1's model) | 0.045 | 0.000 |
| trained, seed 0 | **0.025** | 0.230 |
| trained, seed 1 | 0.025 | 0.226 |
| calibrated flat (0.0031 / 0.000024) | 0.044 | 0.001 |
| old flat (0.005 / 0.0005) | 0.056 | 0.002 |
| observed | — | 0.417 |

- Observed mass on labels no panel row reaches: 0.0001, so `U` is nearly empty on S01.
- Seed noise is small: row TV between the seeds has mean 0.0016 and max 0.0078. The
  calibrated flat kernel differs from the trained one by mean 0.014 and max 0.229, and the max
  is `a49d`.
- **The flat models do not produce the `a49d → 2a9c` relabel at all.** It is context-specific.
  The calibrated flat arm's forward TV is no better than home labels only, so in Phase 5 it
  should score close to B1 on `dorea_formicigenerans`, and acceptance 1's "below B1 with the
  calibrated flat model" is at risk. The trained kernel gets about half of the observed leak
  (0.23 of 0.42).

### Phase 3 — panel inference in `infer_composition.py`

`--mismapping-matrix panel_kernel.npz --amplicon-dir PANEL_OUT --obs-mseq S01.obs.mseq`,
detected from `format="rectangular"`:

- Observed counts per label come from `label_of_ref[hit]`. Every hit counts; the old
  "only references in our set" filter does not apply, because labels are the database.
- Build `U` and the background source. Sources are all panel sources (no pruning is needed
  at this size); labels are the observed ones plus sinks plus `U`.
- `T`: panel genome → source copy weights. `θ` covers panel genomes plus `background`.
- Outputs: `inferred_composition.csv` has the panel genomes, with a `resolution` column,
  plus a `background` row. `inference_diagnostics.csv` adds `unexplained_fraction`,
  `n_sources`, `n_labels_observed` and `kernel_format`.
- Genome identifiability: genomes whose rows of `T·K` are within TV 0.01 are flagged
  `not_identifiable` together. That is none in this panel, unless the BU unique copy's row
  collapses onto the shared one.

#### Phase 3 notes

Done: `ic.run_panel`, dispatched from `run` when `sm.is_rectangular(--mismapping-matrix)`,
and `test_panel_inference_recovers_genomes_and_background`. The suite is 39/39.

Changed from the list above:
- **`--amplicon-dir` is the `prepare` directory** and reads `panel_translation.tsv`.
- **One fitted row per (genome, source) pair.** A shared source's kernel row is repeated
  once per genome. The model is unchanged, and the compact `T` keeps one genome per row.
- **"Reached" means a nonzero kernel column or a home label.** Observed labels outside that
  set form `U`. Hits to headers outside the kernel's database are counted as
  `n_foreign_hits` and dropped, not sent to background.
- **`--no-mismapping` swaps `K` for home rows (`e_{h(s)}`)**, the B1 model, and still fits
  on labels.
- **Identifiability** uses the full `T·K` at `s = 1`. A genome is `not_identifiable` if any
  other genome is within TV 0.01. Its interval is still reported; nothing is merged.
- **Diagnostics** also include `n_labels_unexplained`, `observed_unexplained_fraction` and
  `n_foreign_hits`. `unexplained_fraction` is the fitted `θ_b`.
- Refused: `--infer-distance-decay`, `--build-mismapping` and `--infer-space v4_group`.

**S01 smoke run** (`work/panel_infer/`; VI, no gate, α 0.5, 3,000 steps rather than
Phase 5's 10,000). Genome TV is taken over panel genomes renormalised without background.

| Kernel | Genome TV | `unexplained_fraction` | `bacteroides_uniformis` | largest abs error |
|---|---:|---:|---:|---|
| trained, seed 0 | **0.016** | 0.0007 | 0.0018 | `streptococcus_salivarius` 0.005 |
| calibrated flat | 0.035 | 0.0015 | 0.0032 | `dorea_formicigenerans` 0.030 |

49 labels observed, 6–7 unexplained, holding 0.014–0.018% of reads. No genome is flagged
`not_identifiable`. The calibrated flat arm misses B1 (0.033) by the `dorea` leak, as Phase 2
predicted.

### Phase 4 — fit check

`check_composition_fit.py` accepts the rectangular kernel and runs its PPC and group TV
over observed labels + sinks + `U`. Test: exact counts pass and a perturbation is flagged,
mirroring `test_forward_fit_passes_exact_counts_and_flags_a_perturbation`.

#### Phase 4 notes

Done: `check_composition_fit.run` recognises a rectangular kernel and
`test_panel_fit_passes_exact_counts_and_flags_a_perturbation` covers it. The suite is 40/40.

Changed from the list above:
- **One label space for inference and the check.** `ic._panel_context` builds the fitted
  kernel, home, counts over observed labels + sink + `U`, and the compact `T`. `run_panel`
  and the check both call it, so they cannot diverge.
- **The raw and grouped statistics are the same** for a panel fit, because labels are
  already exact-sequence groups. `unique_v4_group` still gates; the JSON adds
  `kernel_format`.
- **The kernel is always applied.** A panel fit always samples `s`, and `--no-mismapping`
  had already swapped `K` for home rows.
- **Fixed in `run_panel`:** `posterior_draws.npz` now includes the `s`/`conc_frac` draws and
  the `infer_distance_decay`/`likelihood`/`use_active_subset`/`kernel_format` fields.
  Without them the check could not read the archive.

**S01** (the same runs as the Phase 3 table, re-run to write the complete archive):

| Kernel | Fit status | PPC percentile | Observed TV (replicated q50) | Labels | Fitted `s` | Largest label residual |
|---|---|---:|---:|---:|---:|---|
| trained, seed 0 | `ok` | 0.55 | 0.022 (0.050) | 45 | 1.36 | `2a9c` 3.2% obs / 2.3% pred, then `a49d` |
| calibrated flat | `ok` | 0.64 | 0.037 (0.084) | 44 | 1.46 | `2a9c` 3.2% obs / 0.01% pred |

- **The PPC is lenient here.** The calibrated flat arm leaves `2a9c` almost completely
  unexplained and still passes. The fitted Dirichlet-multinomial overdispersion is wide
  enough that replicated TV (0.084) is larger than the observed TV. Phase 5 should report the
  per-label residual on `2a9c` next to fit status, and should not rely on `model_misfit`
  alone to catch a missing context-specific relabel.
- **`s` inflates to 1.4–1.5** in both arms. The fit scales up every source's leak to cover
  the under-predicted `a49d → 2a9c`. This is the "per-source leak calibration" trigger in
  Phase 6.

### Phase 4b — align mode, and the 20HM custom-database comparison

The pipeline builds `M` two ways: `simulate` (simulate reads, run MAPseq) and `align`
(`exact`: `tau = 0` duplicate groups; `kmer1`: neighbours within one edit at `c = 0.007`;
`kmer1_latent`: `kmer1` with the decay refitted per sample). Phases 2–4 covered only
`simulate`. This phase adds the rectangular `align` kernel, the latent decay on it, and a
comparison against the pipeline's own 20HM custom-database runs on all 20 sweep samples.

#### Phase 4b implementation

- **`build_panel_kernel.py align`** writes `K[s, l] ∝ w(l)·c^{d(s,l)}` over the GTDB groups
  within `--tau` edits of source `s`. It uses the IUPAC-aware bounded edlib distance and
  ambiguity weight `w` from `build_mismapping_align.py`. The error-free read has an entry of
  weight 1 at distance 0 on the home label: MAPseq's label from `--home-mseq`, or without it
  the source's own group. The source's own group gets no separate entry, so a relabelled
  exact match is not undone. At `tau ≥ 1` the distances are stored as strata.
  - Ceiling: brute force over sources × groups, 26 × 86,557 in about 10 s.
    `pigeonhole_candidates` is the upgrade for large panels.
- **Latent decay on a rectangular kernel:**
  - `sm.write_kernel`/`read_kernel` carry `strata`.
  - `si.DecayKernel(…, home=)` uses the home entry as the diagonal and has label-sized
    columns.
  - `ic._subset_kernel` makes one sink per dropped distance level, as `_subset_mismapping`
    does, so re-decay stays exact through pruning.
  - `run_panel` accepts `--infer-distance-decay`, and the fit check re-decays per draw.
- **Test:** `test_panel_align_kernel_redecays_exactly_and_fits_a_latent_decay`. Suite 41/41.
- **Observed reads:** all 20 sweep samples were mapped against GTDB r232 with the pipeline's
  read prep: `reads_to_fasta.py --paired --min-overlap 20 --primer-mismatches 3 --max-reads
  100000`, seed 0. The output is in `work/panel_obs/`: one MAPseq run, 1.59 M fragments,
  about 25 min under emulation. Re-mapping S01 reproduces the earlier `S01.obs.mseq` to group
  TV 0.006 (MAPseq thread noise).

#### Phase 4b results

`dev/panel_reinterpretation_sweep.py` (raw rows in `dev/panel_reinterpretation_sweep.csv`).
Every arm uses the settings the 20HM `.p01` profiles used: vi, α 0.5, 3,000 steps, presence
prior 0.01, temperature 1.0. Genome TV is over panel genomes renormalised without
`background`.

| Method (20 samples) | TV median | TV max | `dorea` abs err (median) | BU split err (median / max) | `unexplained_fraction` | Fit `ok` |
|---|---:|---:|---:|---:|---:|---:|
| **20HM custom DB, `kmer1_latent`** | **0.0090** | 0.017 | 0.002 | 0.031 / 0.40 | — | — |
| 20HM custom DB, MAPseq only (no inference) | 0.0102 | 0.012 | 0.0003 | 0.124 / 0.52 | — | — |
| 20HM custom DB, `kmer1` | 0.0105 | 0.013 | 0.003 | 0.014 / 0.13 | — | — |
| 20HM custom DB, `exact` | 0.0109 | 0.013 | 0.003 | 0.044 / 0.13 | — | — |
| 20HM custom DB, `simulate` | 0.0118 | 0.015 | 0.003 | 0.013 / 0.19 | — | — |
| panel, simulate trained (seed 0 / seed 1) | 0.0193 / 0.0184 | 0.023 | 0.0005 | 0.013 / 0.22 | 0.0006 | 20/20 |
| panel, home labels only (B1) | 0.0385 | 0.043 | 0.031 | 0.015 / 0.26 | 0.037 | 20/20 |
| panel, align `exact` (MAPseq home) | 0.0387 | 0.043 | 0.031 | 0.017 / 0.27 | 0.037 | 20/20 |
| panel, align `kmer1_latent` (fitted `c` median 0.0009) | 0.0417 | 0.046 | 0.029 | 0.023 / 0.23 | 0.002 | 20/20 |
| panel, simulate calibrated flat | 0.0442 | 0.050 | 0.030 | 0.017 / 0.32 | 0.001 | 20/20 |
| panel, align `kmer1` (`c = 0.007`) | 0.0518 | 0.057 | 0.024 | 0.053 / 0.32 | 0.002 | 20/20 |
| panel, align `exact`, home = own group | 0.2026 | 0.209 | 0.017 | 0.055 / 0.31 | 0.182 | 20/20 |

Paired by sample, the 20HM custom database beats the best panel arm on **20/20** samples
(simulate trained is 0.007 worse than custom `simulate`, median). Simulate trained beats
home labels only on 20/20. Align `exact` beats it on 7/20, with a median difference of
0.0001. Align `kmer1_latent` beats it on 0/20.

Forward check on S01 (truth pushed through `K` at `s = 1`, compared with observed labels):

| Kernel | trained | calibrated flat | align `exact` | home only | align `kmer1`, c 0.0008 | align `kmer1`, c 0.007 | align, home = own group |
|---|---:|---:|---:|---:|---:|---:|---:|
| Forward TV | **0.023** | 0.042 | 0.044 | 0.044 | 0.046 | 0.090 | 0.203 |

What this shows:

1. **Align mode works end to end but adds nothing to home labels.** Align `exact` with the
   MAPseq home is exactly B1: panel sources have no GTDB duplicates, so each row is one
   entry. The home labels come from one MAPseq run on 26 error-free sequences, and they do
   all the work.
2. **Pure alignment (home = own group) fails.** TV is 0.20 and 18% of reads go to
   `background`, because alignment cannot predict MAPseq relabelling exact matches
   (`2acb → 81c3`, `a3e62 → 89c3`, the *Veillonella* copies). So "no mapper at all" is not
   an option for reinterpretation. The MAPseq home run is the minimum.
3. **The `kmer1` leak model is wrong in both directions at GTDB scale.**
   - Too much: panel sources have up to 31 GTDB groups within one edit, and `c = 0.007` per
     neighbour predicts up to 18% leak where MAPseq shows about 2%. Forward TV doubles to
     0.090.
   - Too little: it cannot produce the context-specific `a49d → 2a9c` relabel (42% real).
   - The latent fit drops `c` to about 0.0009, which is back to home labels only.

   On the 20HM database the same kernel works (`kmer1_latent` is the best custom arm),
   because that database has few near neighbours and no GTDB-scale search effects. This is
   the "near-identical reference set" failure case in `alignment_mismapping_plan.md` §1,
   observed at GTDB scale.
4. **Only the trained simulation recovers `dorea_formicigenerans`** (abs error 0.0005 against
   0.03 for every other panel arm). Its remaining 0.007 gap to the custom database is spread
   across genomes (`bacteroides_vulgatus`, `streptococcus_salivarius`). The seed noise floor
   is 0.0009.
5. **The fit check passes every arm on every sample**, including 18% background with home =
   own group. It cannot rank kernels. Report forward TV and `unexplained_fraction` alongside
   it (see Phase 4 notes).
6. **BU strain split:** the panel trained arm (median 0.013, max 0.22) is comparable to the
   custom database arms (0.013–0.044, max 0.13–0.40).
7. **MAPseq alone against the 20HM database beats GTDB + reinterpretation on 20/20 samples**
   (genome TV median 0.010 against 0.019 for simulate trained; the median gap is 0.008).
   MAPseq alone is the `observed_rel_abundance` column of the custom runs, identical across all
   8 matrix × prior runs per sample. It also beats custom `simulate` on 19/20 samples
   (+0.0013) and loses to custom `kmer1_latent` on 14/20 (−0.0017). Its errors are
   concentrated in the BU strain split, which superresolution fixes:
   - The split error grows from 0.12 at `a0.00` to 0.52 at `a1.00`, against ≤0.02 at the ends
     for custom `simulate` and panel trained. MAPseq alone gives a near-fixed split, so it is
     only right in the middle (S10/S11: 0.02).
   - It puts 0.6% on the absent strain at S01/S19/S20, against ≤0.02% for both inference
     arms.
   - BU is a small share of the community, so these errors add little to genome TV.

   Using genome TV alone, a panel user should map against the panel and skip inference. The
   inference steps (custom-database SR, or GTDB reinterpretation) earn their place on the
   strain split, and reinterpretation gets it right (0.013) without re-mapping.

### Phase 5 — write-up and robustness (revised after Phase 4b)

`dev/panel_reinterpretation.md`, building on `dev/panel_reinterpretation_sweep.csv`.
Phase 4b already covers the matrix axis on all 20 samples, S10/S19 included, with the
pipeline's presence gate. What remains is the prior axis, the deletion tests and the
write-up. **The comparison bar is now the 20HM custom database run with the same matrix
method and prior**, not B1.

Arms (3,000 steps as in the pipeline, plus 10,000 for the prior comparison; seed-matched):

| Axis | Values |
|---|---|
| kernel | trained (seed 0 and seed 1 as the noise floor); calibrated flat (sub 0.0031, indel 0.000024); align `exact` and `kmer1_latent` as the no-simulation arms; home labels only (B1). Old flat and align `kmer1` at fixed `c = 0.007` are dropped: Phase 4b shows both are worse than B1. |
| reads per source | 5,000 (26 × 5,000 = 130k reads, about 1 min of MAPseq under emulation plus DB load) |
| prior | Dirichlet α=0.5 no gate; presence gate; horseshoe |
| `s_scale` | free; pinned at 1 |

Scores per arm:
- Genome TV to truth, and the absolute error for each genome.
- `bacteroides_uniformis` mass (truth 0) and BU pair split error.
- Mass on genomes absent from the truth, `unexplained_fraction`, fit status and PPC
  percentile.
- **Forward check:** truth pushed through `K` compared with observed labels. In GTDB space
  the measured matrix gave 0.055.
- The recovered share of `2acb`'s and `a3e62`'s genomes.

Robustness tests that need no new mapping:
- **Out-of-panel deletion:** drop `fusobacterium_nucleatum` (6.4%) and then
  `salinibacter_ruber` (1.3%) from the panel. Pass if ≥80% of the deleted mass goes to
  `background`, rather than to near neighbours in the panel.
- **Strain axis:** map S10 (`a0.42`, both BU strains) and S19 (`a1.00`) against GTDB with
  `reads_to_fasta.py` plus MAPseq (minutes each) and score the BU split. The panel kernel
  is reused.

Acceptance (revised after Phase 4b; status on the 20 samples in brackets):
1. Trained kernel: median genome TV ≤ 0.02 [**met**, 0.019] and within 0.01 of the 20HM
   custom database with the same prior [**met**, +0.007 against custom `simulate`].
   `dorea_formicigenerans` abs error ≤ 0.01 [**met**, 0.0005].
2. No-simulation arms (calibrated flat, align `exact`, align `kmer1_latent`): below B1
   [**not met**: 0.044, 0.039 and 0.042 against 0.0385]. This is a finding, not a bar to
   tune towards. Without the trained error model the panel is only as good as its home
   labels.
3. BU split error ≤ 0.05 in median [**met** by the trained arm, 0.013]. The max (0.22) is
   reported next to the custom database max (0.13–0.40).
4. The seed-replicate `K` changes TV by less than the gap between arms [**met**, 0.0009].
5. `unexplained_fraction` ≤ 1% [**met** for every simulated and `kmer` arm], and the
   deletion test passes [pending].
6. Fit status `ok` [met everywhere, but it does not discriminate; Phase 4b finding 5]. The
   write-up reports forward TV and the `a49d → 2a9c` residual next to it.

A pipeline default needs acceptance 1. Align mode stays available for reinterpretation, but
the write-up documents it as equivalent to home labels only.

#### Phase 5 results

Done: `dev/panel_reinterpretation.md`. Also added: `infer_composition.py --s-sigma` (a tiny
value pins `s`), and prior, `s1`, `10k` and `_del_<genome>` arms plus a
`custom20hm_mapseq_only` baseline in the sweep script. Acceptance on 20 samples:
1. **Met with no gate:** TV 0.0149, +0.003 against custom `simulate`, `dorea` 0.0003.
2. **Not met** under any prior.
3. **Met:** BU split 0.013–0.028. Horseshoe has max 0.039, the best of any arm.
4. **Met.**
5. **Met:** deletion ≥ 0.93 to background on every arm and sample.
6. `ok` everywhere except gate at 10k steps. That arm collapses (TV 0.055) as the gate's KL
   drains presence probabilities, so the gate is not a safe prior for panel fits.

Phase 6's measured-background trigger did not fire. The per-source leak calibration trigger
stands: free `s` (1.4–1.7) still does its job, and pinning it costs 0.002 TV.

### Phase 6 — conditional

| Work | Trigger |
|---|---|
| **Measured background:** add DB-group rows (from `select_active_amplicons.py` on the labels near unexplained or excess reads) to `K`, so non-panel organisms compete for shared labels | The deletion test fails: panel genomes absorb >20% of the deleted mass |
| Per-source leak calibration (fit a per-source home-mass scale) | **Triggered** (Phases 4 and 4b): `s` inflates to 1.4, and every non-trained kernel misses the `a49d → 2a9c` relabel. This is the only route by which a no-simulation kernel could beat B1. |
| Align kernel for reinterpretation beyond `exact`: a per-neighbour leak shaped by MAPseq rather than `c^d` (e.g. rung 4 of `alignment_mismapping_plan.md`, or `c` divided by neighbour count) | Only if a no-simulation arm is required. Phase 4b shows `c^d` over-leaks at GTDB neighbour density and still misses context relabels. |
| Custom-database route in the pipeline for panel samples: map reads against the panel directly when the panel is known in advance | Default recommendation from Phase 4b: 20/20 samples better. Reinterpretation is for reads already mapped against a generic DB with no re-mapping possible. |
| Panel sources absent from the DB (drop their groups from the DB and re-cluster, ~25 min) | Any real panel has a source with no exact DB match |
| Nextflow wiring: `panel_references` samplesheet key; `MAPSEQ_SIM` over panel sources against `--references`; `INFER_COMPOSITION` on the rectangular kernel; publish `panel_sources.tsv` | Phase 5 acceptance met (validate before wiring, as Phase 4 of the GTDB plan taught) |
| SILVA as the generic DB | After GTDB passes; needs only a SILVA V4 `amplicons.fasta`/`.tax`/`.mscluster` |

## Risks

- **Oracle optimism.** S01's reads were simulated with the same skiver model used for the
  trained `K`. The calibrated flat arm is the honest number, and the write-up leads with it.
- **Simulation fidelity.** Relabels depend on context (`a49d → 2a9c`). Under-predicting a
  leak onto a label no other panel source reaches under-estimates that source by roughly the
  missing share, about 1.5% TV here. This shows up in the PPC, not silently.
- **Panel mis-specification** is only partly covered by `U`. The deletion test measures how
  much; Phase 6 is the fix.
- **Identifiability inside the panel** can still collapse: two genomes whose `T·K` rows
  coincide. That case is flagged and not split.

## Files touched

`bin/subspecies_infer.py`, `bin/sparse_matrix.py`, `bin/infer_composition.py`,
`bin/check_composition_fit.py`, new `bin/build_panel_kernel.py` (`prepare`, `build`,
`align`), new `dev/panel_reinterpretation.py` (+ `.md`, `.csv`), new
`dev/panel_reinterpretation_sweep.py` (+ `.csv`), tests in
`tests/test_measured_mismapping.py` and `tests/test_composition_fit.py`, and a README
section once Phase 5 passes. Work directories: `work/panel_kernel/` (one kernel per arm),
`work/panel_obs/` (20 GTDB mappings), `work/panel_sweep/` (fits and fit checks).
