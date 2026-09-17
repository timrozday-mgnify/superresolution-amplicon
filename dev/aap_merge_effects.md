# How AAP's fastp merge treats sequencing errors (AAP plan Phase 0)

Plan: `synthetic-metagenomic-benchmark-pipeline/docs/aap_silva_compat_plan.md`. This note
covers items 0.1–0.6.

## 0.1 fastp behaviour

**Result: the merge corrects little and filters little.** fastp enforces its overlap
mismatch limit only on the first 50 bases of the overlap. Past those, a pair with any number
of mismatches or an indel still merges. The merged read is R1, so R1's substitutions and
indels reach MAPseq intact. R2 affects only the merged read's length, and it only fixes an
R1 base when R1 is at most Q14 and R2 is at least Q30. The plan's effects 1 and 2 are wrong
as written (see below).

Setup: fastp 1.0.1 (`community.wave.seqera.io/library/fastp:1.0.1--c8b87fe62dcc103c`, AAP's
pin) with READS_QC_MERGE's arguments copied from the Nov2025 `fastp.json`. The template is the
most common merged read of `SC2189280-SC3-1-26s000344` (292 bp, 515F to rc(806R)). Mates are
2×310 and read 18 bases into the TruSeq adapters. All bases are Q38 unless a case sets them.
627 pairs, one fastp run, deterministic. Rows in `aap_merge_effects_fastp.csv`.

    python dev/aap_merge_effects.py fastp \
        --merged <aap_outdir>/<run>/qc/<run>.merged.fastq.gz --out dev/aap_merge_effects_fastp.csv

### Which base the merged read keeps

One substitution in one mate, the other mate correct. Same result at fragment positions 50,
150 and 250.

| Error in | Error quality | Correct mate's quality | Merged read |
|---|---|---|---|
| R1 | ≤ Q14 | ≥ Q30 | corrected |
| R1 | ≤ Q14 | ≤ Q24 | error |
| R1 | ≥ Q15 | any | error |
| R2 | ≥ Q30 | ≤ Q14 | error |
| R2 | otherwise | | correct (R1's base) |

The thresholds are fastp's `≥ Q30` against `≤ Q14`, as the plan assumed. The Nov2025 reads
use three quality bins (Q12, Q24, Q38), so correction happens only between a Q12 base and
a Q38 base. An R1 error at Q24 or Q38 is kept whatever R2 says.

### The mismatch limit covers only the first 50 bases

- `k` Q38 substitutions spread over fragment bases 20–272 in R1: every pair merges up to
  `k = 10`, and the merged read carries all `k`. Split between the mates, the pair merges and
  carries R1's share.
- `k` substitutions packed into fragment bases 1–48: merges for `k ≤ 5`, fails for `k ≥ 6`.
  The same `k` in bases 61–108 always merges.

This is consistent with fastp's overlap search stopping its mismatch check after the first
50 compared bases (`complete_compare_require` in `OverlapAnalysis`; not read in the 1.0.1
source here). When the insert is shorter than a read, as here, those 50 bases are the
start of the fragment, i.e. the forward primer and the first ~30 bases of V4.

### Indels

One insertion or deletion in one mate, every 4th fragment position:

| Indel position | In R1 | In R2 |
|---|---|---|
| 8–40 | not merged | not merged |
| 0–4 | merged, carries the indel | merged, identical to the template |
| ≥ 44 | merged, carries the indel, plus a 1-base length error at the 3′ end | merged, R1's sequence with a 1-base length error at the 3′ end |

The merged read's length is set by where R2 starts, and an indel in either mate moves that
by one base. Near the 3′ end, the merged read gains an adapter base or loses the template's
last base. An indel in the first ~42 bases puts more than five mismatches into the first 50
bases of the overlap, so the pair is dropped. Positions 0–4 merge because a one-base shift
of the overlap absorbs the indel.

### Errors in both mates

A substitution, insertion or deletion in both mates at positions 50, 150 or 250 merges, and
the merged read carries it with no other change.

## What this changes in the plan

- **Effect 1** (lower error rate, cut tail): mostly wrong. Correction needs a Q12 base against
  a Q38 base, and the mismatch cap applies to the first 50 bases only. The merged read is R1
  with R1's errors, except for low-quality R1 bases that R2 overrides. Reads with many errors
  after base 50 are not filtered out.
- **Effect 2** (sequencing indels removed): wrong. R1 indels past base ~42 pass unchanged.
  R2 indels past base ~42 become a one-base length error at the 3′ end, which cmsearch
  clipping may remove (0.4). Only indels in bases ~8–40 are removed, together with their
  pair. The indel rate for merged reads is therefore R1's sequencing rate over most of the
  read, not the PCR rate, and the 3′ end has its own length noise.
- **Effect 4** (position): the first ~42 bases are filtered for indels and, above five
  mismatches, for substitutions. The rest of the read is not filtered at all.
- **Decision 3** (merged-read error model): still the right default. The merged read is R1
  plus a low-quality patch and a 3′-end length error, which a skiver model with position
  covariates trained on merged reads should be able to learn (untested). The 0–3 bp 3′
  overhang in real reads is reverse-primer phasing, not this length error (0.3).
- **Phase 3.2** (separate indel decay): the reason given, that indels leak at the PCR rate,
  no longer holds. A separate indel decay may still be needed, but Phase 0.2's measured
  rates must justify it.

## Not measured

- Inserts longer than a read (V3–V4 on 2×250). There the first 50 compared bases are a
  different part of the amplicon.
- Whether `--detect_adapter_for_pe` finds adapters by sequence on a full run and changes the
  3′ end. The synthetic run was not checked for this.

## 0.2 Residual error profile

**Result: in the interior of the amplicon, merged reads carry about 3.4×10⁻⁴ substitutions
per base and about 100× fewer indels.** The rate is the same in all 20 runs (median column
3.0–3.6×10⁻⁴). It does not vary enough along the read to trigger Phase 3.3: the largest
20-column window is at most 2.1× the smallest in every run (trigger > 3). Most indel mass
sits in a few columns that look like variants rather than sequencing error.

    python dev/aap_merge_effects.py profile --genomes ~/Documents/mimicc/data/genomes \
        --runs <Nov2025 amplicon-pipeline-results> --out dev/aap_merge_effects_profile.tsv

Outputs:
- `aap_merge_effects_profile.tsv`: `run pos coverage sub ins del`, one row per fragment
  position (the forward primer is positions 0–18), per run plus `pooled`. The `pooled` rows
  are the plan's `position_error_profile.tsv`.
- `aap_merge_effects_profile.summary.tsv`: one row per run.

### Reference: the 20HM genomes, not the ASVs

The plan said to align to each run's ASVs. Those are DADA2's estimates made from the same
reads, so errors DADA2 absorbed would not be counted, and reads it dropped (36% as chimeras
in one run) would not be seen. Instead, the reference is the V4 amplicons cut from the 22 20HM
genome assemblies plus the three extra assemblies in the same directory (FNPN01,
GCA_001894945.1, GCF_013009555.1), primers excluded: 27
distinct sequences of 252–254 bp. Each distinct merged read is aligned (edlib, infix) to every
amplicon and assigned to the nearest one if that one is unique and at most 4 edits away.

- **C. perfringens has no reference.** The DACTBY01 draft carries no rRNA operon, and the 16S
  table has no entry for it. In the run checked, its ASVs are 36 edits from the nearest
  amplicon, so the cap drops its reads. It is the "unclassified Clostridiales" ASV that dominates some runs. This is why
  only 23–99% of each run's reads are kept (pooled 64.5%).
- Ties between different amplicons at the same distance: 0.03% of reads.
- Columns are fragment positions 44–261 ("interior"). The 5′ bases are where fastp filters
  (0.1). The 3′ bases are ragged next to the reverse primer. The two ends are reported
  separately below.

### Rates

Pooled over 4.5M reads (2.9M kept), cap 4, interior columns:

| | Median column | Mean without outlier columns |
|---|---:|---:|
| Substitution | 3.45×10⁻⁴ | 3.99×10⁻⁴ |
| Insertion | 3.4×10⁻⁷ | 1.7×10⁻⁵ |
| Deletion | 3.4×10⁻⁶ | 2.4×10⁻⁵ |

The substitution median barely moves with the cap: 2.9×10⁻⁴ at cap 1, 3.3–3.6×10⁻⁴ at caps
2–6, and 3.8×10⁻⁴ at cap 10. The mean rises faster (3.2 to 4.8×10⁻⁴) because each higher cap
admits reads with several variant columns. From cap 3 upward, columns 71, 222, 223, 232 and
240 become outliers together, which looks like one strain or rRNA copy that differs from its
assembly at several sites. Column 99 is an outlier at every cap. It reaches 1.5–2.4% in some
runs and 0.5% in others, so it too looks like a variant whose share changes between runs.

By region (cap 4, coverage-weighted mean, median column in brackets):

| Fragment positions | Substitution | Insertion | Deletion |
|---|---|---|---|
| 19–43 (5′, next to the primer) | 8.8×10⁻⁴ (5.1×10⁻⁴) | 9.4×10⁻⁵ (0) | 9.6×10⁻⁵ (5.9×10⁻⁶) |
| 44–261 (interior) | 5.1×10⁻⁴ (3.5×10⁻⁴) | 1.7×10⁻⁵ (3×10⁻⁷) | 2.7×10⁻⁵ (3×10⁻⁶) |
| 262–272 (3′, next to the primer) | 1.0×10⁻³ (8.3×10⁻⁴) | 1.2×10⁻⁵ (9×10⁻⁷) | 7.2×10⁻⁵ (1.2×10⁻⁵) |

(The interior mean here includes outlier columns; the table above excludes them.)

**Indels.** Five interior columns hold 62% of the interior indel mass. The largest are 76–78
and 27–30, where an insertion in one column pairs with a deletion in the next. That pattern is
consistent with a local difference from the reference, not with independent sequencing
indels. The median column, about 4×10⁻⁶ per base for insertions and deletions
together, is the better estimate of the sequencing indel rate. It is about 100× below the
substitution rate.

### What this changes in the plan

- **The interior substitution rate** is 3.4×10⁻⁴, close to the plan's rough 3×10⁻⁴ from ASVs.
  That is 15× below `--flat_sub_rate` (5×10⁻³). For `--sim_error_model flat` on AAP inputs,
  the README starting point is `--sub-rate 3.5e-4` with insertion and deletion rates of
  `2e-6` each. These are per-run numbers.
- **Phase 3.2** (separate indel decay): justified by the numbers, even though 0.1 overturned
  the reason the plan gave. Sequencing indels are ~100× rarer than substitutions in these
  reads, so a shared decay would give one-indel neighbours about 100× too much weight.
- **Phase 3.3** (position-aware decay): not triggered. The window ratio is at most 2.1
  (threshold 3). Skip it.
- **The 5′ and 3′ ends** carry 2–3× the interior substitution rate. A merged-read skiver model
  with `Position(2)` (Phase 2.2) should cover that. The 5′ end also has a higher indel mean
  despite fastp's filtering there, most of it at columns 27–30.

### Limits

- The reference is the DSM strains' assemblies. Where the fermentor strains or rRNA copies
  differ, those differences count as error. The outlier-column exclusion and the medians
  guard against that, but multi-site variants below the outlier threshold remain.
- The rates are PCR plus sequencing together. Errors in both mates cannot be told apart from
  template differences here.
- Base qualities are not used.

## 0.3 Primer-site mix

**Result: the oligos are the original EMP pair with phased spacers, not 515F-Y/806B.** 515F's
`Y` reads C in 99.9% of reads, and 806R's `N` never reads G. So the forward primer is
`GTGCCAGCMGCCGCGGTAA` and the reverse is `GGACTACHVGGGTWTCTAAT`. SR's default primers
still find them, since they are a superset. Both primers carry heterogeneity spacers: 0 and
4–7 bases before the forward primer, and 0–3 bases after the reverse primer (the "overhang").
The mix does not depend on the source genome. It does depend on the reverse spacer, so
Phase 2.1 should draw primer bases jointly with the spacers, not position by position.

    python dev/aap_merge_effects.py primers --genomes ~/Documents/mimicc/data/genomes \
        --runs <Nov2025 amplicon-pipeline-results> --out dev/aap_merge_effects_primer_mix.tsv

Outputs:
- `aap_merge_effects_primer_mix.tsv`: `run genome primer pos code A C G T reads`. `pos` is
  the position in the primer as written, and 806R bases are on the primer's own strand.
  `genome` is `all`, or the 20HM genome whose amplicon equals the read's primer-free
  interior exactly. This is the plan's `primer_mix.tsv`.
- `aap_merge_effects_primer_mix.ends.tsv`: reads by `run spacer overhang fwd_bases
  rev_bases`. `fwd_bases` are 515F positions 3 and 8; `rev_bases` are 806R positions 7, 8
  and 13. Its `pooled` rows are the joint table 2.1 should sample from.

A read counts if both primers match exactly at their fixed positions (degenerate positions
may be any base): the forward primer within the read's first 40 bases, and the reverse
primer's last hit in its final 60. That is 93.1% of 4.5M reads. The forward primer is missing
from 4.7%. In one run checked, the most common such starts are single-base deletions
inside the primer (`TGCCAGCAGCCGCG…`, `GTGCAGCAGCCGCG…`), which look like truncated
synthesis products. The reverse primer is missing from 2.2%. 53% of reads are also assigned
to a genome.

### Mix (pooled)

| Primer | Position | Code | A | C | G | T |
|---|---|---|---:|---:|---:|---:|
| 515F | 3 | Y | 0.000 | 0.999 | 0.000 | 0.001 |
| 515F | 8 | M | 0.618 | 0.382 | 0 | 0 |
| 806R | 7 | N | 0.324 | 0.376 | 0.000 | 0.299 |
| 806R | 8 | V | 0.335 | 0.370 | 0.295 | 0 |
| 806R | 13 | W | 0.483 | 0 | 0 | 0.516 |

Off-code bases are at most 0.1%. Across runs, the major option varies by: 515F-8 A
0.59–0.63, 806R-7 C 0.37–0.39, 806R-8 C 0.36–0.38, and 806R-13 T 0.45–0.53.

**No dependence on the source.** Across the 20 genomes with at least 10,000 assigned reads,
every option's frequency is within 0.02 of the pooled value. The one outlier,
GCF_013009555.1, has 13 reads. The primer bases come from the oligo, as the plan assumed,
and do not follow the template. Uniform draws would be wrong: `Y` is not C/T at 50:50 and `N`
has no G.

### Spacers

- **Forward spacer** (bases before 515F): 0 in 27.0% of reads, 3 in 0.6%, 4 in 19.9%,
  5 in 18.7%, 6 in 18.0%, 7 in 15.7%. The 515F-8 mix barely changes with it (A 0.59–0.63).
- **Overhang** (bases after 806R): 0 in 25.9%, 1 in 28.4%, 2 in 23.8%, 3 in 22.0%. The
  shares match in every run to within 0.01. Each length has its own reverse mix, so each
  is a different oligo:

| Overhang | 806R-7 A / C / T | 806R-8 A / C / G | 806R-13 A / T |
|---|---|---|---|
| 0 | 0.41 / 0.29 / 0.30 | 0.33 / 0.27 / 0.40 | 0.52 / 0.48 |
| 1 | 0.27 / 0.46 / 0.27 | 0.34 / 0.45 / 0.21 | 0.49 / 0.51 |
| 2 | 0.31 / 0.33 / 0.36 | 0.32 / 0.35 / 0.32 | 0.39 / 0.61 |
| 3 | 0.31 / 0.42 / 0.27 | 0.35 / 0.40 / 0.25 | 0.53 / 0.47 |

**Independence.** For each pair of sites, the table below gives the largest gap between their
joint frequency and the product of their marginals:
- The forward and reverse primers are independent of each other, and spacer is independent
  of overhang (≤ 0.005).
- The three reverse sites are not independent: 0.007–0.020 pooled. Within one overhang
  length the gap grows to 0.037–0.045, so positions of one oligo co-vary as well.
- Drawing whole reverse-primer triples from `ends.tsv`, together with the overhang, keeps
  all of this.

## What this changes in the plan (0.3)

- **Phase 2.1:** `--primer-mix` should take the joint `ends.tsv` table: a spacer length,
  forward bases, overhang length and reverse bases drawn together. Per-position frequencies
  would lose the overhang dependence (up to 0.10 on one option) and the co-variation within
  an oligo. The fallback without a file should not be "uniform over the code". For these
  runs it should be the EMP oligos' measured mix, or else refuse.
- **Spacer bases are not recorded yet.** Simulated reads need their content too, and 0.4
  decides whether cmsearch clipping removes them. The 3′ overhang is reverse-primer phasing,
  so it is part of the read structure, not error.
- **0.1's "0–3 bp overhang" note** has its explanation: phasing, not merge length noise.
- **`aap_samplesheet.py` primer check (1.5):** AAP reports `515-YF`/`806BR` for these runs,
  but the reads carry the EMP oligos. The check should compare by compatibility (the reads'
  oligo matches the configured degenerate primer), not by string equality.

## 0.4 cmsearch clip

**Result: clipping matters for MAPseq's top hit, and slightly for SR's V4 groups, so it has to
be emulated.** Clipping to the cmsearch hit changes the top hit for 19.2% of reads. An
unclipped replicate agrees on 100%, so this is not run-to-run noise. Almost all of those
changes (98.8%) are between references with the same V4 amplicon. At the V4-group level SR
uses, 0.19% of reads change group and another 0.04% cannot be checked. Agreement is at least
99.77% and at most 99.81%, just short of the plan's 99.9% threshold. Genus composition moves
by 0.003 (TV).

**Separately, the Nov2025 run did not use this database.** Its execution report shows MAPseq
run against `SSU.fasta` / `slv_ssu_filtered2.txt`, an older MGnify SILVA build that uses the old
phylum names (`Firmicutes`, `Actinobacteria`). AAP 6.1.5's `silva-ssu-138.1`
(`SILVA-SSU.fasta`, `Bacillota`, `Actinomycetota`) is a different reference set. Its labels
are not the Nov2025 labels.

    python dev/aap_merge_effects.py clip --merged <run>.merged.fastq.gz \
        --deoverlapped <run>.tblout.deoverlapped --db SILVA-SSU.fasta --tax SILVA-SSU-tax.txt \
        --work work/aap_clip --out dev/aap_merge_effects_clip.tsv

Setup: `SC2189280-SC3-1-26s000344`, all 517,399 merged reads with a cmsearch hit (all on
`SSU_rRNA_bacteria`, all `+` strand). MAPseq 2.1.1b `--h3ab3c3b_0` (AAP's build, amd64 under
emulation, ~3 min per pass on 10 threads) against `silva-ssu-138.1`, with AAP's arguments
(`-seed 12 -tophits 80 -topotus 40 -outfmt simple`). There are three passes: merged reads,
reads clipped to the table's columns 8–9 as AAP's `esl-sfetch -Cf` does, and merged reads
again. Rows in `aap_merge_effects_clip.tsv`. The `.mseq` files stay in `work/aap_clip/`.

### What the clip removes

The clip keeps both primers. At the 5′ end it starts 0–2 bases before 515F, so it removes the
forward spacer but keeps up to 2 of its bases. At the 3′ end it removes 0–3 bases, usually most
of the 806R overhang (0.3). A read loses 0–10 bases in total. Mean mismatches plus gaps
against the top hit fall from 2.71 to 2.63.

### Effect on MAPseq

| | Merged vs replicate | Merged vs clipped |
|---|---:|---:|
| Same top hit | 100% | 80.76% |
| Same classification string | 100% | 65.91% |
| Same classification, trailing empty ranks ignored | 100% | 95.38% |
| Top hit changes, same V4 amplicon | – | 19.01% |
| Top hit changes, different V4 amplicon | – | 0.19% |
| Top hit changes, a hit without an amplicon | – | 0.04% |

- **Top hits.** 85% of changed top hits have the same mismatch-plus-gap count as before. The
  change is mostly a different pick among equally good references, and SILVA Ref (not NR99)
  holds many references with identical V4 regions. Amplicons were cut with `si.extract_v4`
  (default primers, 3 mismatches) after U→T.
- **Classifications.** Most differences only change depth: `g__;s__` against `g__`, 70k and
  69k reads in the two directions, i.e. whether an empty species rank is printed. Ignoring
  that, 4.6% of reads differ. 3.5% get a deeper or shallower lineage, and 1.1% conflict (for
  example `Collinsella` ↔ `Coriobacterium`, ~2,000 reads each way).
- **Composition.** Genus TV between merged and clipped is 0.003. Both are 0.035 from AAP's
  published counts for this run. That gap comes from the database difference above, not
  from clipping. For example, `Erysipelatoclostridium` (6,040 published reads) does not
  exist in 138.1's taxonomy.

## What this changes in the plan (0.4)

- **Clipping must be emulated.** Observed reads re-mapped by SR under
  `--mapseq_target references` should be clipped as AAP clips them: from the run's
  `.tblout.deoverlapped` when it exists, otherwise by running cmsearch with AAP's arguments.
  Simulated panel reads need the same clip. The cheapest candidate is a fixed rule from
  primer to primer plus the observed 0–2 base flanks. Whether that rule reproduces the real
  clip's V4 groups is not measured yet: compare its `.mseq` against `clipped.mseq` before
  adopting it.
- **Labels should be V4 groups, not top hits.** One in five top hits depends on bases outside
  the amplicon. Any comparison with AAP's `.mseq` (Phase 1 acceptance, 1.5 `mseq:` reuse)
  should match at the V4-group level.
- **Decision 1 needs the right database version.** An existing AAP `.mseq` is only reusable
  if it was made with the same database. The Nov2025 results used the older `SSU.fasta` /
  `slv_ssu_filtered2.txt`, not `silva-ssu-138.1`. `aap_samplesheet.py` (1.5) should check the
  database behind a run's `.mseq` before reusing it: its label is not enough, since both are
  called `SILVA-SSU`. Phase 4's truth sample should map against a single named version
  throughout.

## 0.5 MAPseq build and format

**Result: the two builds are interchangeable; SR's default arguments are not.** With AAP's
arguments, SR's build (`2.1.1b--hc47f52e_1`) and AAP's (`2.1.1b--h3ab3c3b_0`) give the same
output line for line on all 517,399 clipped reads. Only the row order differs, because of
threading. With SR's default arguments (`params.mapseq_args = ''`), the top hit differs from
AAP's for 20.2% of reads, and the V4 group for 2.8%. The columns SR reads are in the same
places in both output formats.

    python dev/aap_merge_effects.py builds --db SILVA-SSU.fasta --tax SILVA-SSU-tax.txt \
        --work work/aap_clip --out dev/aap_merge_effects_builds.tsv

Setup: the clipped reads and the AAP-build `.mseq` from 0.4, mapped again with SR's image,
first with AAP's arguments and then with none. Every file is read through `si.iter_mseq`.
The plan called for 10k reads; this uses the whole run. Rows in `aap_merge_effects_builds.tsv`.

| Build | Arguments | Output columns | Same top hit as AAP | Changed, same V4 | Changed, different V4 | Changed, a hit without an amplicon |
|---|---|---:|---:|---:|---:|---:|
| AAP | AAP | 15 | – | – | – | – |
| SR | AAP | 15 | 100% | 0 | 0 | 0 |
| SR | none (SR default) | 38 | 79.81% | 16.84% | 2.81% | 0.54% |

- **Build.** Sorted, the SR-build file is byte-identical to the AAP-build file.
- **Arguments.** Without arguments MAPseq uses `-tophits 20 -topotus 10` and no fixed seed.
  AAP uses 80, 40 and seed 12. This run does not separate the two causes. An unseeded run
  might also differ between repeats, which is not checked here.
- **Format.** With no `-outfmt`, MAPseq writes the confidences format: 38 columns, with each
  rank and its two confidences as separate columns. `-outfmt simple` writes 15 columns with
  the lineage in one. In both, column 1 is `#query`, 2 is `dbhit` and 4 is `identity`. That
  is all `si.iter_mseq` (fields 0, 1, 3) and `build_panel_kernel._mseq_rows` (the same
  fields) read, so both parsers handle either format. Nothing else in `bin/`, `modules/` or
  `workflows/` parses `.mseq` columns.

## What this changes in the plan (0.5)

- **Build pin:** no change is needed. SR can keep `mapseq_tag = 2.1.1b--hc47f52e_1`.
  Recording both build strings in the kernel id (Risks) is harmless but not required for
  these two.
- **Arguments:** under `--mapseq_target references`, every MAPseq call (MAPSEQ_OBS,
  MAPSEQ_PANEL_HOME, MAPSEQ_PANEL_SIM) must run with AAP's `-seed 12 -tophits 80 -topotus 40`.
  1.3 should set these rather than rely on `params.mapseq_args`, whose default gives a
  different V4 group for 2.8% of reads. `-outfmt simple` is optional for parsing, but
  matches AAP's files and is 17% smaller. The arguments belong in the panel kernel id and
  `provenance.json` too.

## 0.6 Label coverage

**Result: 0.29% of reads (pooled; at most 0.56% in any run) have a top hit on a SILVA
reference with no V4 amplicon, below the plan's 1% trigger. No relaxed extraction pass is
needed.** Those references are not truncated. 16 references take all of these reads, and
all but 5 reads fall on references whose primer site holds an indel that the ungapped
in-silico PCR cannot match. Three references take 98.4% of them.

    python dev/aap_merge_effects.py coverage --runs <Nov2025 amplicon-pipeline-results> \
        --db SILVA-SSU.fasta --tax SILVA-SSU-tax.txt --work work/aap_coverage \
        --out dev/aap_merge_effects_coverage.tsv

Setup: the 20 runs' merged reads with a cmsearch hit (4,497,808), clipped as AAP clips them
(0.4), mapped in one pass with AAP's MAPseq build and arguments against `silva-ssu-138.1`
(~25 min). In-silico PCR (`si.extract_v4`, default primers, 3 mismatches, U→T) runs on the
1,799 distinct top hits only, not the whole database. When it finds no amplicon, the missing
site is tested again with gaps allowed (edlib, at most 3 edits, IUPAC-aware). Outputs:
`aap_merge_effects_coverage.tsv` (per run and pooled) and
`aap_merge_effects_coverage.refs.tsv` (each reference without an amplicon, with its reads
and lineage). The combined `.mseq` (885 MB) is in `work/aap_coverage/`.

| | Pooled | Per run |
|---|---:|---:|
| Top hit has an amplicon | 99.71% | 99.44–99.92% |
| Top hit has an indel in the 806R site | 0.286% | 0.08–0.56% |
| Top hit has an indel in the 515F site | 0.002% | ≤ 0.01% |
| Top hit lacks a primer site | 5 reads | |
| No hit | 0 | 0 |

The runs with the most such reads (`-1`, `-6`) are the ones dominated by *C. perfringens*
(0.2).

| Reference | Reads | Missing site | Lineage |
|---|---:|---|---|
| FJ215343.1.1359 | 11,245 | 806R, one base deleted (`ATTAGATACCC-TGTAGTCC`, 6 mismatches ungapped) | *Clostridium perfringens* |
| FJ983016.1.1511 | 1,092 | 806R, a one-base deletion (7 mismatches ungapped) | *Fusobacterium* |
| HQ787985.1.1445 | 425 | 806R, a one-base deletion (5 mismatches ungapped) | *Bacteroides* |

All three are full-length 16S (1,359–1,511 bp) with an exact 515F site. In FJ215343 the
deletion removes one of the 806R `B`/`N` columns. Whether it is real or a sequencing error in
the reference is not checked here.

## What this changes in the plan (0.6)

- **Phase 2.4:** the trigger (> 1%) is not met, so skip the relaxed extraction pass. The
  `nonamp_<acc>` label columns are still worth adding. Without them these reads (up to 0.56%
  of a run, nearly all from one C. perfringens reference) have no column in the kernel.
  If a relaxed pass is added later, it should allow an indel in the primer site (gapped
  primer match), not "cut at one primer plus the expected length". No reference here is
  truncated.
- **Run cost:** in-silico PCR only on the references reads actually hit took seconds. The
  "hours" of extracting from all 2.1M references (Risks) is only needed for the panel's
  taxon-entry sources and the `align` input, not for label coverage.
