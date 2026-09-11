# Making GTDB-scale amplicon inference identifiable and testable

**Status:** proposed implementation plan.

## Executive decision

The current GTDB run is a failed forward model, not merely a difficult optimisation.
For the 50 high-depth SC2200627-SC3 samples, the median total-variation (TV) distance
between the observed MAPseq reference-count distribution and the distribution generated
from the reported posterior mean was **0.950** with the archived GTDB matrix. It remained
**0.784** after collapsing labels to distinct V4 sequences. The matched custom-database
control was **0.014** and **0.012**, respectively. A TV distance of zero is an exact match.

Adopt a **resolution-aware, candidate-limited GTDB mode**:

1. Create a GTDB screen database from **all** dereplicated V4 sequences, annotated with
   GTDB species representatives and LCA membership.
2. Build a batch-level candidate set from screen-supported V4 groups, recall neighbours,
   curated organisms, and an explicit unknown component.
3. Derive the candidate mis-mapping matrix with the `align/kmer`, `tau=1` kernel and a
   calibrated distance decay; validate that approximation on representative candidate sets
   against simulated MAPseq matrices and known-truth reads.
4. Report abundance at the V4-identifiable block or lowest-common-taxon level; emit a
   genome/species abundance only when the block is a singleton or an external panel makes
   the identity explicit.
5. Make posterior-predictive checking a release gate and retain the fitted parameters
   required to reproduce it.

Candidate pruning is required for computational tractability and calibration, but it does
not add information. Selecting one genome from a large exact-V4 group because it was the
MAPseq top hit must be prohibited: that label is not identified by the assay.

## Evidence from SC2200627-SC3

The input reference order in the full-GTDB archive's extraction index exactly matches the
supplied `ssu_all_r232_ssu` database. A wrong input database is therefore not the cause.

| Property | Full GTDB run | Custom-run control |
|---|---:|---:|
| Amplifiable V4 references | 1,001,241 | 79 |
| Genomes in output | 516,891 | 22 |
| Distinct V4 sequences | 86,557 | 24 |
| References in duplicate-V4 groups | 94.2% | manageable |
| Median high-depth raw-reference TV, matrix scale 1 | **0.9497** | **0.0135** |
| Median high-depth V4-group TV, matrix scale 1 | **0.7840** | **0.0116** |
| Best raw TV while down-weighting matrix from 1 to 0 | **0.8557** at 0 | **0.0135** at 0.988 |

The custom database maps a median of only 21 fewer reads per sample than GTDB (maximum
41 out of about 99,000). Its superior fit is not explained by missing references.

`mean_diagonal = 0.710` is not a valid reference-level health summary for the GTDB run.
It is the unweighted mean over grouped-kernel rows. At original-reference level, where an
individual sequence competes with every duplicate member, the mean self-assignment is
0.080. The largest exact-V4 group has 99,590 members.

The archived matrix used `align/kmer`, `tau=1`, and automatic distance decay. The existing
alignment plan describes this as an opt-in approximation validated on a small,
duplicate-dominated reference set; it does not establish calibration for real reads in a
million-reference database.

## Why full GTDB genome calls cannot work from V4 alone

GTDB is a genome taxonomy: species are defined from whole-genome ANI and selected
representatives, rather than from a claim that a short 16S variable region separates all
genomes. See [Parks et al.](https://academic.oup.com/nar/article/50/D1/D785/6370255).

Using every GTDB assembly as a V4 reference asks reads to distinguish genomes that share
the same sequence. It cannot. Short variable-region 16S has less resolving power than
full-length 16S, and even full length needs explicit treatment of intragenomic copies;
see [Johnson et al., 2019](https://doi.org/10.1038/s41467-019-13036-1). MAPseq is a
reference-based classifier with confidence estimates, not a source of information absent
from the read; see [Rodrigues et al., 2017](https://doi.org/10.1093/bioinformatics/btx517).

Consequences:

- exact-V4-equivalent genomes are not distinguishable by this assay;
- a chosen genome inside such a group is prior-driven unless external evidence narrows it;
- the output must carry an ambiguity block and a lowest-common taxonomy, not a silent
  allocation of mass among members.

## Options

| Option | Benefit | Limitation | Decision |
|---|---|---|---|
| Current full GTDB alignment matrix | None beyond current run | Fails calibration and V4 identifiability | Reject |
| Full GTDB with simulated MAPseq matrix | Tests the alignment approximation | Still cannot distinguish shared V4 genomes; very expensive | Offline calibration experiment only |
| GTDB species representatives | Useful taxonomy/metadata layer; limits a genome panel | Representative-only screening can miss strain-specific V4 variants | Use for annotation, not as the only screen |
| Screen-derived candidate set | Makes a calibrated candidate `M` practical | Cannot select among indistinguishable members | Required with block-level reporting |
| Candidate `align/kmer`, `tau=1` matrix | Fast, reproducible, and scales with candidate sequence groups | Distance kernel is an approximation, not a mapper measurement | Production option after calibration gates pass |
| V4-block/LCA output | Scientifically identifiable calls | Lower apparent resolution | Required output contract |
| Curated inoculum panel + sentinels | Enables justified panel-specific calls | Not discovery; omissions must remain unknown | Recommended for fermentors |
| Full-length 16S, operon, shotgun, or isolate data | Adds discriminatory information | Requires another assay | Only general route to strain claims |

### Why pruning the current `M` is insufficient

`_active_subset` already removes references that cannot reach an observed hit. It still
retains a median 61,856 genomes (maximum 156,163), because supported V4 sequences are
shared by many genomes. Pruning to a raw mapper top-hit ID discards equally plausible
competitors and produces confident but arbitrary calls.

Pruning is valid only when it preserves the complete ambiguity block, or when a documented
external prior, such as a defined inoculum, legitimately limits the block.

## Recommended design

### Reference products

Build and version two products for each GTDB release and primer pair.

1. **Screen DB:** one record per unique V4 sequence from *every* extracted GTDB record,
   annotated with all member genomes, member species representatives, member count, and
   a deterministic lowest common taxonomy (LCA). It identifies supported *sequence
   groups*, including variants found only outside a GTDB species representative.
2. **Candidate DB:** a batch-specific subset of those unique V4 sequence groups. It is
   the production mapper database and the inference basis; its group-to-GTDB membership
   is an annotation/translation table, not an instruction to allocate abundance among
   member genomes.

GTDB species representatives are a useful compact taxonomy layer and may be used in a
documented organism panel. They must not be the only screening sequence set. A supplied
strain panel may add strain genomes and their copies, but this does not alone make a strain
identifiable.

### Candidate selection

Use one candidate DB per biological batch. This makes time points comparable, permits one
reusable alignment matrix, and prevents weak samples from dropping candidates retained by
other samples.

The selector must retain:

- observed screen V4 groups above a configurable read threshold;
- the supported unique V4 groups and the complete GTDB membership/LCA annotation for each;
- groups in a verified edit-distance neighbourhood (initially <=1) as a recall guard;
- panel members, positive controls, and configurable sentinels;
- an `unknown` / out-of-candidate component.

Each manifest row must record `selection_reason` (`observed_group`, `near_neighbour`,
`panel`, or `sentinel`). A hard candidate cap must fail the run unless it lists excluded
groups and marks the result `candidate_incomplete`.

### Candidate alignment matrix

Build the production candidate matrix with `align/kmer`, `tau=1`, using the unique V4
groups as the matrix sources and labels. `--align_distance_decay` must be a calibrated
per-base error probability (normally `auto` from a pre-trained, assay-matched error model)
or a versioned externally measured value. It must never be `1.0`: that allocates the same
weight to a one-edit neighbour as to an exact duplicate and substantially overstates
confusion. Store the matrix with the candidate manifest, group/LCA table, decay source,
alignment settings, hashes, and source-row diagonal distribution.

Simulation remains a validation instrument, not the production route: on a small,
representative set of candidate DBs, compare the alignment rows to `simulate → MAPseq`
rows and then test both against known-truth reads. Any reference ambiguity, short/unmerged
read geometry, or systematic low-identity mapper tail that fails this validation routes
that assay/database to the simulated method instead.

### Resolution-aware inference

Create an **identifiability block** from identical or materially indistinguishable
*true-source row signatures* of the candidate alignment matrix. Here a row is the
probability distribution of candidate-DB MAPseq labels given one true candidate V4 group
(`r_obs = r_true @ M`). Normalise rows to one, calculate their distance in that observation
space, and calibrate the near-equivalence tolerance with mocks at the intended depth. Where
the result is rank-deficient or near-rank-deficient, fit abundance directly on the collapsed
blocks rather than fitting members then summing them.

- Infer and report posterior mass for each block.
- Report species/genome abundance only for singleton blocks or a documented, exclusive
  panel prior that rules out all other block members and has been validated experimentally.
- Otherwise emit `resolution = v4_equivalence_block`, the block abundance, and the LCA.
- Do not use `presence_prob` to choose a member of a non-identifiable block.

### Posterior-predictive gates

Posterior predictive checking compares data with replicated model outcomes and is standard
Bayesian model criticism; see [Wang et al., 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2829996/).

Persist posterior `theta_eff`, `s`, `conc_frac`, and fitted `c` if enabled. At both
raw-reference and unique-V4-group levels, report observed-versus-expected TV, held-out
read log score or deviance, PPC percentile, support outside candidate groups, and
`fit_status`: `ok`, `low_depth`, `candidate_incomplete`, or `model_misfit`.
Calibrate thresholds from posterior replicates, not a universal TV cutoff.

### Non-negotiable data contracts

**One observation space.** After screening, re-map the *real* reads against the exact
candidate DB with the declared MAPseq version, reference headers, trimming, and parameters.
The likelihood and PPC consume this candidate-DB count table only. Screen assignments are
selection evidence, not count observations for a candidate matrix. Persist a
`candidate_reference_id`, `candidate_group_id`, mapper label, read count,
translation-map version, and candidate-manifest hash on every hand-off. The simulated
calibration maps use precisely this same mapper configuration.

**Unknown/out-of-candidate biology.** `unknown` cannot be an unlabeled free abundance
parameter. Every screen-supported group must remain a candidate label, including a group
whose GTDB annotation is unresolved; such mass is reported as `unclassified_v4_group`, not
as a selected taxon. Reads that cannot be assigned to a screen group are counted separately
as `screen_unassigned`; their fraction is a candidate-recall/PPC diagnostic, not silently
absorbed into a selected taxon. A candidate cap that would exclude a supported group fails
`candidate_incomplete` rather than inventing an uncalibrated catch-all row.

**Abundance unit.** The default estimand is *V4 amplicon-copy abundance*, because a read
comes from an rRNA-gene copy rather than directly from an organism. Every candidate row
must record its source genome and copy multiplicity. A genome/species "organism abundance"
may be emitted only from an explicit copy-number model with complete, validated copy counts
and a stated prior; otherwise it is not comparable across taxa. Intragenomic copies that
have the same V4 sequence are one sequence-group source, with multiplicity retained solely
for an optional copy-number correction.

**Matrix and reporting representation.** Store the group-level production matrix plus the
lossless `source_reference -> source_group`, `observed_label -> observed_group`, and
group-membership maps. Group/block predictive checks are the release gate. A
raw-reference residual is diagnostic-only and may be emitted only when those maps permit a
unique, declared expansion; it must never be used to infer a member within a group.

**Alignment, calibration, and taxonomy provenance.** The production matrix key includes
candidate-manifest and group/LCA hashes; aligner/backend version; `tau`; decay value and
its source; sequence normalisation; and ambiguity settings. The validation record includes
mapper and simulator versions, seed, read length, primer/trim handling, quality,
substitution, and indel models, read layout, and all MAPseq options. LCA must use a
versioned GTDB taxonomy table and deterministic rank rule. Block identifiers derive from
the manifest and row-signature hash so that downstream aggregation is stable.

## Implementation plan

### Phase 0 — make failure observable

**Goal:** no completed inference may be interpreted as healthy without a forward-fit
check.

1. In `bin/subspecies_infer.py`, retain posterior draws or summary quantiles for the
   fitted nuisance parameters `s`, `conc_frac`, and `c` (when enabled), alongside
   `theta_eff`.
2. Add `bin/check_composition_fit.py`. Given a composition table, observed MAPseq
   counts, translation map, and the exact matrix bundle, it must reconstruct the active
   subset and calculate raw-reference and unique-V4-group observed-versus-expected
   diagnostics. It must label the result as `model_misfit` when the observed counts are
   implausible under posterior predictive draws.
3. Replace the ambiguous `mean_diagonal` diagnostic in
   `bin/infer_composition.py` with two fields: `mean_kernel_diagonal` and
   `mean_reference_diagonal`. The latter is the average after expanding each V4 group
   back to its member references.
4. Add `--min-infer-reads` (default 1,000) and write `low_depth` rather than a
   biological composition below that threshold.

**Acceptance tests:** the archived custom run has a small grouped residual; the current
full-GTDB archive is flagged `model_misfit`; a synthetic exact-forward dataset passes;
and a deliberately perturbed count vector fails. The check must run for every sample,
including samples that fail the depth gate.

### Phase 1 — build a GTDB screening and candidate stage

**Goal:** reduce the statistical problem before fitting without pretending that V4 has
more resolution than it does.

Add the following products and interfaces.

| Component | Proposed interface | Required output |
|---|---|---|
| Screen database builder | `bin/build_gtdb_screen_db.py` | One V4 sequence per unique group across all GTDB records, mapping to all members, representatives, and group/LCA metadata |
| Candidate selector | `bin/select_gtdb_candidates.py` | Batch candidate manifest, retained V4 groups, unresolved groups, and per-row selection reason |
| Workflow modules | `modules/local/mapseq/screen`, `modules/local/select_gtdb_candidates` | Versioned channels and hashes for the screen result and manifest |
| Configuration | `gtdb_mode = 'screen_candidate'` | Candidate thresholds, panel path, sentinel path, recall radius, and candidate cap |

Start from all GTDB V4 groups, then retain the groups that satisfy the batch-level rules
above. This must be a *recall-oriented* screen:
`candidate_min_reads = 2`, `candidate_neighbour_edits = 1`, mandatory panel/control
members, and an unknown component are sensible initial defaults. Record the screen
assignment confidence and never discard an ambiguous group merely because MAPseq selects
one top reference.

The batch manifest is a pipeline input to all downstream samples. It is also a first-class
provenance object: include GTDB release, source-record and representative selection rules,
screen database hash, normalisation/edit-distance policy, MAPseq settings, all thresholds,
candidate reference and group counts, estimated matrix size, and the candidate list hash.
It must fail `candidate_incomplete` when any group, source reference, or estimated matrix
size cap excludes otherwise selected material.

### Phase 2 — derive and validate the candidate alignment matrix

**Goal:** make the forward model correspond to the actual mapper and limit claims to
what the amplicon can identify.

For `gtdb_mode = 'screen_candidate'`, build the production matrix with
`build_mismapping_align.py --backend kmer --tau 1`. Set
`--distance-decay auto` with an assay-matched pre-trained error model, or supply a
validated fixed decay; refuse `1.0`. Add an explicit `MAPSEQ_OBS_CANDIDATE` step that
maps the real reads to the same candidate DB and produces the only count table passed to
inference.

Retain `SIMULATE → MAPSEQ_SIM → BUILD_MISMAPPING` as a calibration workflow over a small,
frozen candidate-suite—not as a required production operation. The suite must include exact
duplicates, one-edit neighbours, ambiguous references if present, and the expected range of
candidate-set sizes. Compare group-row distributions, not only mean diagonals. A candidate
configuration may use the alignment matrix only when it passes this comparison and the
known-truth/PPC gates in Phase 3.

After the matrix is built, create `bin/build_identifiability_blocks.py`. It should form
blocks from equal or calibrated-near-equal **source rows** of the alignment matrix and
write source-group-to-block, block-member, row-signature, and LCA tables. The inference
output must include block abundance and apply the reporting rule in the preceding section.
A requested genome-level table can still be emitted, but entries from a non-singleton
block must be marked `not_identifiable` and must not be ordered as strain calls.

**Acceptance tests:** an intentionally identical pair is one block; a singleton stays a
singleton; a near-equivalent pair meets the calibrated depth-specific block rule; a
one-edit pair has a non-uniform, calibrated decay weight; changing the candidate manifest
invalidates the matrix cache; real reads are rejected if their candidate manifest hash
differs from `M`; and a small candidate database gives the same result through dense and
grouped matrix paths.

### Phase 3 — validate before production use

**Goal:** establish whether the mode is reliable for this assay, not only whether it
finishes.

1. Construct mock communities containing close GTDB relatives, known negatives, and
   concentrations spanning the expected dynamic range.
2. Hold out a portion of reads or independent technical replicates when enough reads
   are available. Candidate selection must use training reads only; re-map held-out reads
   only after the manifest and matrix have been frozen. Check calibration and predictive
   residuals at group/block level.
3. Compare candidate-mode outputs with the current custom control database. Investigate
   any large disagreement before expanding the reference set.
4. Freeze defaults only after confirming candidate recall, false-positive control,
   block-level calibration, and PPC pass rate. Version the benchmark with the GTDB
   release and mapper version.

For the initial validation protocol, flag a high-depth sample when its calibrated PPC tail
probability is below 0.01 or above 0.99 for either the group-level residual or held-out
deviance. Do not release the mode if any positive/negative control fails, or if more than
5% of assay-valid high-depth benchmark samples are flagged. These are provisional
go/no-go thresholds: freeze or revise them once from an independent calibration cohort,
before a production cohort is interpreted. TV remains a descriptive diagnostic, not the
sole threshold.

## Proposed configuration and output contract

```yaml
params:
  gtdb_mode: screen_candidate       # off | screen_candidate
  gtdb_species_representatives: true
  candidate_scope: batch
  candidate_min_reads: 2
  candidate_neighbour_edits: 1
  candidate_panel: null             # TSV: GTDB accession / reason
  candidate_sentinels: null         # TSV: GTDB accession / reason
  candidate_max_groups: null        # no cap is safest during validation
  candidate_max_source_references: null
  candidate_max_matrix_cells: null
  min_infer_reads: 1000
  production_mismapping: align
  align_backend: kmer
  align_tau: 1
  align_distance_decay: auto         # pre-trained, assay-matched error model required
  align_decay_model: null            # required when auto cannot use validated flat rates
  report_resolution: identifiable_blocks
```

Each sample should receive a machine-readable `fit_diagnostics.json` and the report should
surface: read depth; candidate manifest hash; number of retained groups, references, and
blocks; raw and grouped predictive residuals; PPC result; and `fit_status`. Results marked
`low_depth`, `candidate_incomplete`, or `model_misfit` must not enter comparative abundance
plots by default. The configuration must also state sequence normalisation (orientation,
primer removal, ambiguity policy), the exact edit-distance method (including indels), and
whether the recall neighbourhood was verified against the selected simulation error model.

## Non-solutions and guardrails

- Do not use the current k-mer full-GTDB matrix with a stricter optimiser, more VI steps,
  or a stronger sparsity prior. It is a forward-model failure, not evidence of an
  optimisation failure.
- Do not prune the existing full matrix after inference based on high posterior abundance.
  That is circular and will lock in an arbitrary member of an indistinguishable group.
- Do not score the model against its MAPseq top assignments alone. Those assignments are
  mapper output, not independent evidence that a GTDB genome produced a read.
- Do not equate GTDB's genome-resolved taxonomy with V4 assay resolution. GTDB species
  representatives are useful to control reference multiplicity, but they do not make
  duplicate V4 sequences distinguishable.

## Decision required before coding Phase 1

Implement Phase 0 immediately. For Phase 1, choose the biological candidate panel and
the target reporting level: species representative when singleton, or V4-equivalence
block/LCA everywhere. The second is the scientifically conservative default. The external
panel should be supplied as a versioned TSV so its additions are auditable rather than
hidden in a database snapshot.
