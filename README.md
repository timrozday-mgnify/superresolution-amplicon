# superresolution-amplicon

Error-model-driven **sub-species genome composition inference** from amplicon (e.g.
16S V4) sequencing reads. Given fastq reads and a set of reference sequences, the
pipeline:

1. **Extracts the reference amplicons** by in-silico PCR. They define the database's V4
   groups and the sources reads are simulated from; reads are mapped against the
   reference FASTA itself (see [The MAPseq database](#the-mapseq-database)).
2. **Measures reference-to-reference mis-mapping** — how easily each reference's amplicon
   is confused for another — by **simulating reads from every reference under a
   sequencing error model and mapping them with the same mapper the real reads go
   through** ([mapseq](https://github.com/jfmrod/MAPseq)). The error model defaults to a
   naive flat per-mutation-type one that needs no training at all; the confusion structure
   turns out to be insensitive to it ([why](dev/error_rate_sensitivity.md)). Set
   `--sim_error_model trained` to train one from the reads instead (reference-free, via
   [skiver](https://github.com/timrozday-mgnify/skiver)). `--mismapping_method align`
   replaces this whole stage with one pass of reference-to-reference alignment — no
   mapper, no error model, ~460x cheaper, and on the reference set it was checked against
   indistinguishable from the measurement ([evidence](dev/alignment_mismapping.md)).
3. **Maps the real reads** with mapseq to get the observed per-reference counts.
4. **Infers the true genome composition** with a Bayesian model (Pyro) that inverts the
   mis-mapping to recover latent genome abundances from the observed read signal.

This is a heavy, multi-stage analysis extracted from the
`synthetic-metagenomic-benchmark-pipeline` into a standalone Nextflow (DSL2) pipeline.

## GTDB-scale safety

Against a generic database (GTDB, SILVA) always name a panel (`panel_references` or
`panel_taxa`). A V4 read cannot tell apart the tens of thousands of genomes that share
each GTDB V4 sequence, so without a panel the default, whole-database panel fits a genome
space the reads cannot identify. The whole-database panel is for small reference sets.

## Quick start

```bash
# Build the container (once); build context is the repo root for vendor/skiver.
# On amd64. (Apple Silicon: see "Building on Apple Silicon" below.)
docker build -f containers/skiver/Dockerfile -t ghcr.io/timrozday-mgnify/sra-skiver:latest .

# Run on the bundled tiny fixture.
nextflow run main.nf -profile docker -c tests/nextflow.config \
    --input assets/samplesheet.example.yml \
    --outdir results
```

Outputs land under `results/` (see [Outputs](#outputs)). The image is also built and
published to GHCR by `.github/workflows/build-images.yml` on every push to the default
branch, so on HPC you can just pull it (see [HPC](#hpc-singularity--apptainer)).

> The `-c tests/nextflow.config` above sets `skiver_dump_args = '-c 1'`: the bundled
> fixture is tiny, so skiver's size-based auto-subsampling would otherwise keep too few
> k-mers to train. For small/high-coverage real amplicon data, set `--skiver_dump_args
> '-c 1'` similarly.

## HPC (Singularity / Apptainer)

The pipeline is designed to run on HPC with Singularity/Apptainer. Both profiles pull
the same image; `autoMounts` is enabled.

```bash
# Pre-pull the image to a shared cache (avoids every task re-pulling).
export NXF_SINGULARITY_CACHEDIR=/shared/singularity_cache
singularity pull "$NXF_SINGULARITY_CACHEDIR/sra-skiver-latest.img" \
    docker://ghcr.io/timrozday-mgnify/sra-skiver:latest

nextflow run main.nf -profile singularity \
    --input samplesheet.yml \
    --references db.fasta \
    --outdir results \
    -c your_hpc.config          # executor (slurm/lsf), queue, resource ceilings
```

Use `-profile apptainer` on sites where the binary is `apptainer`. Tune
`--max_cpus/--max_memory/--max_time` (or a site config) to your queue. The
`process_*` labels in `conf/base.config` scale resources on retry.

## Samplesheet

YAML list of samples (or a map with `samples:`). Per sample:

| key | required | description |
|-----|----------|-------------|
| `id` | yes | Unique sample id (output routing). |
| `reads` | yes\* | fastq path or list of paths. |
| `fastq_1` / `fastq_2` | yes\* | Alternative to `reads` (paired-end). Supplying `fastq_2` marks the sample paired: R1/R2 are **merged into one query per fragment** before mapping, so each query spans the amplicon like the references do. |
| `paired` | no | `true` to merge a two-file `reads` list as R1/R2. Not inferred: two files could equally be two single-end runs. |
| `platform` | no | `hq-illumina` \| `lq-illumina` \| `ont` \| `pacbio` (default `hq-illumina`). Sets the skiver error-model context + report notebook. |
| `references` | no | Per-sample reference fasta; overrides `--references`. |
| `error_model` | no | Path to a pre-trained `.pt` model; **skips training** for this sample. |
| `mseq` | no | Path to a mapseq classification of this sample's reads (a previous run's `mapseq/<id>/<id>.obs.mseq.gz`); **skips read mapping** for this sample. It must have been produced against the same reference set — the ids in it are matched to the extracted amplicons — and it carries the read-prep settings it was made with, so `--obs_max_reads`, `--trim_primers` and `--min_pair_overlap` no longer apply to that sample. Mapping is the expensive stage, so this is what makes a parameter sweep over the mis-mapping and inference knobs cheap. |
| `merge_rate` | no | Fraction of input pairs that merged (`bin/aap_samplesheet.py` writes it). Below 0.8 on a `merged: true` row, `--mismapping_method align` is refused: merging lost reads, likely more from some sources than others, and edit distances cannot express that. Use `simulate --sim_read_structure pairs`. |
| `merged` | no | `true` for reads that are already merged pairs, e.g. amplicon-analysis-pipeline's `qc/<id>.merged.fastq.gz`. They are never merged again or primer-trimmed, and the run needs `--trim_primers false` so the simulated reads keep their primers too. Pass `--primer_mix assets/primer_mix_emp_v4.tsv` for AAP's EMP 515F/806R reads, so those primers carry the real oligo mix. `bin/aap_samplesheet.py` writes such rows from an AAP output directory. |
| `panel_references` | no | Genome panel for this sample; overrides `--panel_references`. See [Panel reinterpretation](#panel-reinterpretation). |
| `panel_taxa` | no | Taxon panel entries for this sample; overrides `--panel_taxa`. See [Panel reinterpretation](#panel-reinterpretation). |

\* provide either `reads` or `fastq_1`.

```yaml
- id: sampleA
  reads: [reads/A_R1.fastq.gz, reads/A_R2.fastq.gz]
  platform: hq-illumina
  references: db/refs.fasta
```

### Reference fasta header convention (important)

Genome membership is read from the fasta header as `genome|index|orig` — **the text
before the first `|` is the genome id**. Multiple entries sharing a genome id are that
genome's 16S copies. Example:

```
>Escherichia_coli|0|NR_024570.1
GTGCCAGC...   (a 16S sequence spanning the V4 region)
>Escherichia_coli|1|NR_024571.1
...
>Salmonella_enterica|0|NR_074910.1
...
```

References that do not yield a V4 amplicon under the configured primers are reported
(`refseq_index.csv`) and excluded from the matrix. The genome-space of the inferred
composition is the set of distinct genome ids.

### Build a reference database from genome FASTAs

`bin/build_mapseq_database.py` builds a pipeline-compatible reference FASTA and its
required MAPseq `.tax` sidecar from one FASTA per genome. The input has a `genomes` list;
relative `fasta` paths are resolved relative to this YAML file. `taxonomy` is a
semicolon-delimited lineage. Mixed lineage depths are allowed and are padded with
`unclassified` in the MAPseq metadata.

```yaml
genomes:
  - id: Escherichia_coli
    taxonomy: Bacteria;Pseudomonadota;Gammaproteobacteria;Enterobacterales
    fasta: genomes/E_coli_16S.fasta.gz
  - id: Salmonella_enterica
    taxonomy: Bacteria;Pseudomonadota;Gammaproteobacteria;Enterobacterales
    fasta: genomes/S_enterica_16S.fasta
```

```bash
python bin/build_mapseq_database.py \
    --input genomes.yml \
    --output-prefix db/references

# Use db/references.fasta with the pipeline.
nextflow run main.nf --input samples.yml --references db/references.fasta
```

Each source sequence receives a header such as
`Escherichia_coli|0|original_accession`; copy indices reset for each genome. Run
`python bin/build_mapseq_database.py --demo` for a self-contained check.

#### A generic database from SILVA SSU

SILVA has no genomes, so each sequence is its own reference, `accession|0|accession`.
The builder converts RNA `U` to `T` and pads shallower lineages with `unclassified`. SILVA
has no species rank, so the organism name that ends each lineage becomes one under a
Bacteria/Archaea genus when it is a binomial (`…;Enterocloster;Enterocloster bolteae`, the
genus taken from the lineage), and is dropped otherwise (`sp.`, `uncultured bacterium`).
About a quarter of NR99 sequences get a species. Use Ref NR99. The `.tax` it writes is the
real lineage, so pass it as `--taxonomy`:

```bash
python bin/build_mapseq_database.py \
    --silva-fasta SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz \
    --output-prefix db/silva_138_2_ssu_nr99

nextflow run main.nf --input samples.yml \
    --references db/silva_138_2_ssu_nr99.fasta --taxonomy db/silva_138_2_ssu_nr99.tax \
    --panel_taxa taxa.tsv
```

A SILVA reference is a sequence, not a genome, so inference over it has no biological
reading. Use it through a panel (`panel_references` or `panel_taxa`).

### The MAPseq database

Every MAPseq mapping in a run (observed reads, panel home and simulated reads) goes against
the `references` FASTA itself, not its extracted amplicons. So a database the
amplicon-analysis-pipeline ships is used as is, and its labels are AAP's:

```
SILVA-SSU/138.1/SILVA-SSU.fasta              --references
SILVA-SSU/138.1/SILVA-SSU.fasta.mscluster    MAPseq's clustering, found beside the FASTA
SILVA-SSU/138.1/SILVA-SSU-tax.txt            --taxonomy (AAP's sk__/s__ form reads as SILVA's)
```

- A `<fasta>.mscluster` beside the FASTA is used, never rebuilt. Without one, the run
  clusters the FASTA once, cached under `--amplicon_cache`: seconds for a genome
  collection, hours for 2M sequences.
- The FASTA must be uncompressed: MAPseq cannot read a gzipped database.
- MAPseq is given a generated `references.tax`. The tax file changes only MAPseq's
  taxonomy columns, never the hit, which is all the pipeline reads. Lineages come from
  `--taxonomy`.
- A hit on an entry with no amplicon under the primers is off-target and dropped (0.29% of
  reads against AAP's SILVA, `dev/aap_merge_effects.md` 0.6).

### Inputs from amplicon-analysis-pipeline

SR can profile an amplicon-analysis-pipeline (AAP) run's output instead of the raw reads.
It takes AAP's fastp-merged reads and reuses AAP's MAPseq classification, so nothing is
mapped twice.

```bash
bin/aap_samplesheet.py aap_results --panel panel.fasta -o samplesheet.yml
nextflow run . --input samplesheet.yml --references SILVA-SSU/138.1/SILVA-SSU.fasta \
    --taxonomy SILVA-SSU/138.1/SILVA-SSU-tax.txt --trim_primers false \
    --primer_mix assets/primer_mix_emp_v4.tsv
```

- **Samplesheet.** `bin/aap_samplesheet.py` writes one row per QC-passed run:
  `qc/<id>.merged.fastq.gz` as `reads` with `merged: true`, fastp's `merge_rate`, and
  `taxonomy-summary/<--db-label>/<id>.mseq` as `mseq`. It refuses a run that AAP did not
  call a single 16S V4 amplicon, or whose primers cannot sit on `--fwd_primer`/
  `--rev_primer`'s binding sites. Every row gets the same `--panel`/`--panel-taxa`.
- **Database.** Use the SILVA-SSU directory AAP classified against, as shipped (see
  [The MAPseq database](#the-mapseq-database)). The reused `.mseq` is valid only against
  that same FASTA and `.mscluster`.
- **Read preparation.** Merged reads keep their primers, so the run needs
  `--trim_primers false`, and the kernel simulates untrimmed reads to match. `--primer_mix`
  draws those primers from AAP's measured EMP 515F/806R oligo mix rather than uniformly
  over the degenerate bases.
- **Error model.** Merged reads have their own error profile, not a raw mate's
  ([dev/aap_merge_effects.md](dev/aap_merge_effects.md) 0.1–0.2). Choose one:
  - flat rates measured on merged reads: `--flat_sub_rate 3.4e-4 --flat_ins_rate 4e-6
    --flat_del_rate 4e-6` (20 Nov2025 runs; per-run numbers);
  - `--sim_error_model trained`, which trains one pooled model on the merged reads;
  - `--sim_read_structure pairs`, which simulates raw mates and merges them with AAP's
    fastp. It needs a model trained on raw mates as each row's `error_model`.
- **Align.** Give substitutions and indels separate decays (`--align_distance_decay auto
  --align_indel_decay auto`), since indels are about 100× rarer in merged reads. Align is
  refused on a row whose `merge_rate` is below 0.8.

On a synthetic 16-genome V4 community (2×310, 200k pairs, reinterpreted through a 22-genome
panel), the synthetic-metagenomic-benchmark-pipeline scored species total variation against
truth:

| input | kernel | species TV |
|---|---|---|
| AAP's own MAPseq labels | – | 0.742 |
| AAP merged reads | simulate, merged | 0.0083 |
| AAP merged reads | simulate, pairs (oracle model) | 0.0074 |
| AAP merged reads | align, tau 1 | 0.0188 |
| raw pairs, SR's own merge | simulate | 0.0099 |

Most of AAP's gap is resolution: MAPseq assigns under 5% of its reads to a species.
`merged` is within 0.001 of `pairs`, so it stays the default. Every genome in that
community is its own species, so it does not test a strain split.

## Parameters

Run mode / IO:

| param | default | description |
|-------|---------|-------------|
| `--input` | – | YAML samplesheet (required). |
| `--references` | – | Default reference fasta, also the MAPseq database (per-sample override in samplesheet). A `<fasta>.mscluster` beside it is used as is. |
| `--outdir` | `./results` | Output directory. |
| `--platform` | `hq-illumina` | Default platform (see samplesheet). |
| `--error_model` | – | Global pre-trained model (per-sample override in samplesheet). |
| `--seed` | `42` | Global seed. |

Error-model **architecture** (skiver context model):

| param | default | description |
|-------|---------|-------------|
| `--error_model_candidates` | `AdditiveContext(5),AdditiveContext(7),AdditiveContext(9)` | Candidate context orders; the min-AIC model is kept. |
| `--error_model_components` | – | Force a single component string (skips AIC selection). |

Reference amplicons (in-silico PCR):

| param | default | description |
|-------|---------|-------------|
| `--fwd_primer` / `--rev_primer` | V4 515F / 806R | Amplicon primers. |
| `--primer_mismatches` | `3` | Allowed primer mismatches. |
| `--amplicon_cache` | – | Directory that keeps each reference set's extracted amplicons and, when no `.mscluster` ships beside the FASTA, its mapseq clustering (`storeDir`), keyed by the reference FASTA (path, size, mtime), primers and extractor code. Runs pointed at the same directory reuse them instead of rebuilding — at SILVA/GTDB scale the clustering alone is tens of minutes. The cache is unlocked: warm it with one run before starting several on the same set at once. |
| `--trim_primers` | `true` | Trim primers off observed reads before mapping, and simulate the matrix and panel reads from primer-flanked amplicons trimmed the same way. Set `false` if reads are already primer-trimmed. |
| `--primer_mix` | – | Table (`fwd`, `rev`, `reads`) of concrete primer oligos. Untrimmed simulated reads draw their primers from it, one row per read, because degenerate primer bases come from the oligo mix and not the template. [`assets/primer_mix_emp_v4.tsv`](assets/primer_mix_emp_v4.tsv) is the mix measured in AAP's Nov2025 EMP 515F/806R reads ([dev/aap_merge_effects.md](dev/aap_merge_effects.md), 0.3). Every oligo must fit `--fwd_primer`/`--rev_primer`. Without it, untrimmed reads draw each degenerate base uniformly. |

Mis-mapping — how `M` is built:

| param | default | description |
|-------|---------|-------------|
| `--mismapping_method` | `simulate` | `simulate` (sample errored reads from every panel source and map them with the same mapper the real reads go through — the kernel is *measured*) or `align` (each source aligned to the database labels within `--align_tau`, IUPAC-aware — no read simulation, mapseq, or error model). |
| `--align_tau` | `0` | `align` only: cluster references within this edit distance. `0` (exact duplicate amplicons) beat every larger value tested until `--align_distance_decay` existed; with a decay set, `tau >= 1` is finally softer rather than worse. |
| `--align_distance_decay` | `1.0` | `align` only: a cluster member `d` edits away takes `c**d` of a uniform share. A no-op at `--align_tau 0` (every member is at distance 0), and the whole story at larger `tau`: `c = 1` treats a reference one edit away as an exact duplicate, which is why `tau > 0` used to lose. On the benchmark's two-strain *B. uniformis* V4 set — two exact-duplicate clusters one edit apart, 0.5% flat error — `c = 1` puts `0.2` of a row where the `simulate` measurement puts `0.002`–`0.006`; `c = 0.007` reproduces it, and at that value `kmer --align_tau 1` fits the measured matrix slightly *better* than `exact-hash` does (mean row L1 0.079 vs 0.080). Reaching a distance-1 reference costs one sequencing error at that position, so set `c` to about the per-base error rate. Left at `1` with `tau >= 1` the run warns rather than fails, so a pre-existing matrix stays reproducible. Set it to **`auto`** to measure that rate off an error model instead of guessing it, or leave it and set `--infer_distance_decay` to fit it per sample. |
| `--align_decay_model` | `null` | `--align_distance_decay auto` only: a **pre-trained** skiver `model.pt` whose per-base error rate is measured (by applying it and taking the mean edit rate) instead of the flat rates. Nothing is trained for this — `align` mode never runs the skiver subworkflow. `null` measures `--flat_sub_rate + --flat_ins_rate + --flat_del_rate`, which at the defaults gives `c = 0.0059`. For amplicon-analysis-pipeline merged reads it must be a model of the *merged* reads (`--sim_error_model trained` on `merged: true` rows trains one), not of raw mates. |
| `--align_indel_decay` | `null` | `align` only: weight each indel by this instead of `--align_distance_decay`, so a label `s` substitutions and `i` indels away takes `c_sub**s * c_indel**i`. `auto` measures the model's indel rate, and `--align_distance_decay auto` then measures substitutions only. In amplicon-analysis-pipeline merged reads indels are about 100× rarer than substitutions (4×10⁻⁶ against 3.4×10⁻⁴ per base, [dev/aap_merge_effects.md](dev/aap_merge_effects.md) 0.2), so one decay for both overweights one-indel neighbours. `null` gives indels `--align_distance_decay`. |
| `--max_ambiguous_bases` / `--max_postings` | `4` / `4096` | `align` at `--align_tau >= 1` only. `max_ambiguous_bases` is how many IUPAC positions a pair may carry *between them* before the filter is allowed to miss it; each one costs a block, so raising it shortens the blocks and widens the search. `max_postings` skips a block shared by more than that many distinct amplicons — the conserved windows either side of the variable region, which are quadratic to enumerate. It is the filter's only source of false negatives, and it is not reached on GTDB SSU at either tau. |
| `--align_ambiguity_weight` | `0.3` | `align` only: a cluster member carrying `k` IUPAC ambiguity codes takes `w**k` of a uniform share, since mapseq scores an `N` as a mismatch and prefers a clean duplicate. No-op on reference sets without ambiguity codes; `1` disables. The real penalty varies (0.18–0.97), so this is a compromise — see [the sweep](dev/ambiguity_weight_sweep.md). |

| `--min_pair_overlap` | `20` | Paired samples: shortest mate overlap accepted when merging R1/R2. Unmergeable pairs are dropped and counted; past 20% the run warns that the fragments do not cover the amplicon, and `--mismapping_method simulate --sim_read_len` is the right choice for that sample. |
| `--sim_read_len` | – | Reads are this long, i.e. shorter than the amplicon (unmerged / short reads). **`simulate` only** — `align` builds `M` from whole-reference alignments and refuses this combination rather than understating confusion. Unset = reads span the whole amplicon, correct for merged reads. |

The remaining `simulate` settings below (everything except `--panel_kernel`) are
ignored under `--mismapping_method align`, including `--sim_error_model trained` — nothing
simulates reads, so nothing needs an error model and the skiver subworkflow never runs.

| param | default | description |
|-------|---------|-------------|
| `--sim_error_model` | `flat` | `flat` (constant per-base probability per mutation type; **no training at all**) or `trained` (skiver context model). See [the sensitivity result](dev/error_rate_sensitivity.md) for why `flat` is the default. |
| `--trained_error_model_scope` | `per-sample`, or `pooled` if any row is `merged: true` | With `trained`, fit one model per sample or one pooled model per platform. Explicit sample `error_model` files always win. |
| `--trained_error_model_max_reads` | `1000000` | Pooled mode: deterministic, uniformly sampled FASTQ records per platform (`0` = all). |
| `--sim_read_structure` | `merged` | `merged` simulates each read as one merged read. `pairs` simulates R1/R2 mates of `--sim_mate_len` cycles that run into their TruSeq adapters, and merges them with amplicon-analysis-pipeline's fastp 1.0.1 and its READS_QC_MERGE arguments (`FASTP_MERGE`). Unmerged pairs are dropped, as AAP drops them, and each source's merged share goes to `yield.tsv` in the kernel bundle. Use it to validate `merged`, or where merging loses reads (long amplicons on short reads). Needs `--trim_primers false`; under `trained` every row needs an `error_model` trained on raw mates. No cmsearch clip is applied: the pairs carry no spacer or overhang, so a merged read already spans primer to primer. |
| `--sim_mate_len` | `310` | `pairs`: cycles per mate (the Nov2025 runs are 2×310). |
| `--flat_sub_rate` | `0.005` | Flat model: per-base substitution probability. |
| `--flat_ins_rate` | `0.0005` | Flat model: per-base insertion probability. |
| `--flat_del_rate` | `0.0005` | Flat model: per-base deletion probability. |

For amplicon-analysis-pipeline merged reads, a flat starting point is `--flat_sub_rate 3.4e-4
--flat_ins_rate 4e-6 --flat_del_rate 4e-6`: the pooled interior rates of 20 Nov2025 runs
(2×310 V4), measured against their mock community's genomes
([dev/aap_merge_effects.md](dev/aap_merge_effects.md), 0.2). They are per-run numbers, not
a property of AAP. `--sim_error_model trained` on `merged: true` rows trains on the merged
reads themselves, pooled per platform. Position covariates (skiver's `Position(N)`) are not
among the candidates: the pinned skiver cannot simulate from them.
| `--sim_n_per_ref` | `5000` | Simulated reads per distinct panel V4 source (sampling depth for the kernel). |
| `--panel_kernel` | – | A `mismapping/panel_<key>/` directory published by an earlier run. Skips building the kernel: no panel preparation, error-model training, simulation or panel mapping. Refused unless its `provenance.json` names the same extracted amplicons (`reference_sha256`), MAPseq database (`mapseq_db`), `panel` and `panel_taxa` as every sample's. |

In-silico PCR and the mapseq clustering run once per distinct `references` file,
however many samples name it: 50 samples against one GTDB SSU FASTA extract and
cluster it once (or not at all, when its `.mscluster` ships beside it). `amplicons/<id>_amplicons/` is still published per sample. The
pipeline then fingerprints extracted amplicons and builds each compatible matrix once
per run. Each kernel is published as a bundle under `mismapping/panel_<key>/`, and a later
run can reuse it:

```bash
nextflow run main.nf --input samples.yml --references refs.fasta \
  --panel_kernel results/mismapping/panel_<key>
```

The run refuses the bundle unless it was built against the same extracted amplicons,
MAPseq database (the FASTA's path, size and mtime, and its `.mscluster`) and panel. Its
simulation or alignment settings are not checked, because they are the bundle's own:
the kernel is used as it was built.

Read mapping (mapseq):

| param | default | description |
|-------|---------|-------------|
| `--mapseq_args` | `-seed 12 -tophits 80 -topotus 40 -outfmt simple` | Flags forwarded to **every** mapseq invocation (simulated and real alike — that identity is what makes `M` valid). The default is AAP's, verbatim; a replacement drops all of them. |
| `--mapseq_min_identity` | – | Drop hits below this pairwise identity (off-target background). |
| `--obs_max_reads` | `100000` | Reads mapped per sample (`0` = all). Detecting a low-abundance subspecies is a small-excess subtraction, so depth matters (8k is too few). |

Composition inference (Pyro):

| param | default | description |
|-------|---------|-------------|
| `--infer_mode` | `vi` | `vi` \| `nuts` \| `mle`. Use `vi` (posterior mean): `mle` reports the mode, which collapses a low-abundance subspecies to exactly 0. |
| `--infer_alpha` | `0.5` | Dirichlet prior concentration. |
| `--infer_steps` | `3000` | SVI steps (vi/mle). |
| `--infer_lr` | `0.02` | SVI learning rate. |
| `--infer_num_samples` | `500` | Posterior samples (vi/nuts). |
| `--infer_warmup` | `500` | NUTS warmup. |
| `--min_infer_reads` | `1000` | Minimum mapped reads required to emit a biological composition. Lower-depth samples are zeroed and labelled `low_depth`, but still receive a fit-diagnostics record. |
| `--infer_presence` | `true` | Per-genome Bernoulli presence/absence gate. Not supported with `--infer_mode nuts`. |
| `--infer_presence_prior` | `0.01` | Prior probability a genome is present — the sparsity regulariser. |
| `--infer_presence_temp` | `1.0` | Concrete relaxation temperature for the gate. Below ~1 the gate barely moves off its initialisation and no prior can sparsify it. |
| `--infer_distance_decay` | `false` | Fit the tie-cluster distance decay `c` per sample instead of taking the one the matrix was built with. Needs `--mismapping_method align --align_tau >= 1`: only those builds record the distance behind each nonzero, and at `tau 0` every distance is 0 and `c` cancels. The matrix is **not** rebuilt — `M(c) = rownorm(M(c0) * (c/c0)**d)` — so one build still serves the whole matrix group while each sample fits its own error rate. Reported as `distance_decay` in `inference_diagnostics.csv`. With `--align_indel_decay` set, only the substitution decay is fitted; the indel decay stays at its built value. |
| `--infer_decay_sigma` | `1.2` | Prior width in logs of that decay, centred on the built one. |
| `--taxonomy` | `null` | MAPseq `.tax` (`header<TAB>lineage`) that `--panel_taxa` entries are resolved against. |
| `--infer_horseshoe` | `false` | `vi` only, needs `--infer_presence false`: horseshoe shrinkage on unnormalised weights in place of the Dirichlet (`--infer_alpha` is ignored). |
| `--panel_references` | – | Genome panel (same header convention as `--references`, or a directory of `<genome>.amplicons.fasta`). Per-sample override via the samplesheet. See [Panel reinterpretation](#panel-reinterpretation). |
| `--panel_taxa` | – | TSV `id<TAB>taxon`: panel entries that are taxa. Needs `--taxonomy`. Per-sample override via the samplesheet. See [Taxon entries](#taxon-entries). |
| `--panel_taxon_max_sources` | `200` | Fail a taxon entry that resolves to more database V4 groups than this; every group is a simulated source. |

> **The distance decay, fixed or fitted.** `c` is a property of the *sample* — roughly
> its per-base error rate — but the matrix is built once per reference set and shared by
> every sample in the group, so a fixed `c` has to be right for all of them.
> `--infer_distance_decay` makes it a latent instead, sampled per fit and used to re-decay
> the built matrix in place. It recovers the decay a sample's reads actually experienced
> over a 60x range and cuts composition L1 by 2–9x against using the built value, *where
> the reference set identifies it*: `c` reshapes a row across distance classes, which the
> mis-mapping scale `s` cannot do, so it is identified by rows holding both an exact
> duplicate and a near neighbour, on a set with more references than genomes. Where it is
> not identified the fit stays at the build and tracks the fixed run; where `c` cancels
> outright (every distance 0) the run refuses rather than reporting its prior back. See
> [dev/latent_distance_decay.md](dev/latent_distance_decay.md).

> **Presence/absence.** Abundance and presence are different questions. `theta` is a
> Dirichlet, so every genome in the reference DB gets *some* mass and the output can
> never say a genome simply isn't there. The gate adds a Bernoulli `z_g` per genome
> (`theta_eff ∝ z ⊙ theta`), relaxed to a Concrete distribution so SVI can differentiate
> through it. Its posterior probability is reported per genome as **`presence_prob`** —
> the confidence that the genome is in the sample at all, distinct from the credible
> interval on its abundance — and an absent genome's abundance is shrunk towards zero
> along with it.
>
> On a subset-present truth (8 of 21 genomes) the gate reaches **Jaccard 1.0** at the
> defaults and cuts abundance L1 by a third versus the ungated fit, because zeroing an
> absent genome stops it absorbing mis-mapped reads that belong to its neighbours.
> Thresholding the ungated abundance can match the Jaccard, but only at a cutoff picked by
> hand per dataset. Recall was 1.0 at every setting tried — the gate never deleted a
> genome that was really there, including a 3% sub-species — so `infer_presence_prior`
> trades precision, not recall. See
> [dev/presence_prior_sweep.md](dev/presence_prior_sweep.md) for the numbers (and for why
> the straight-through variant of the gate does not work), and
> `dev/presence_prior_sweep.py` to re-run the sweep on your own reference set.
>
> **Identical amplicons are the exception.** If a genome's only amplicon is byte-identical
> to another genome's 16S copy, "present at 5%" and "absent, neighbour slightly commoner"
> fit the reads equally well: presence is not identifiable and the gate follows the prior.
> The abundance is still recovered. Read a low `presence_prob` on such a genome as "V4
> can't tell", not "not there" — `refseq_index.csv` and the mis-mapping matrix show which
> genomes are in that position.

> **Primer trimming.** Observed reads normally carry the amplification primers and so are
> longer than the primer-trimmed reference amplicons they are mapped against. By default
> the pipeline trims each read to its primer-free amplicon (same primers as the amplicon
> extraction stage, both read orientations) before mapping, so observed and simulated
> reads sit in the same coordinate space. Reads that are already trimmed — or have no
> detectable primer — are left unchanged, so it is safe to leave on; disable with
> `--trim_primers false` only if you have a reason to. The simulated reads get the same
> trim: they are drawn from each amplicon flanked by its primers, so errors a trained
> model puts at the start of a read fall in the primer and are cut away, as they are
> for the observed reads. Without that, 42% of trained reads carried extra 5′ bases,
> and against SILVA NR99 the panel kernel failed the fit check on every sample
> ([dev/panel_silva_sweep.md](dev/panel_silva_sweep.md)).

> **Why the same mapper twice.** `M` is only meaningful if the simulated reads experience
> the confusion the real reads experience, so `MAPSEQ_SIM` and `MAPSEQ_OBS` are the same
> process with the same settings, differing only in which reads they take.

### Panel reinterpretation

A sample with `panel_references` is still mapped against `--references` (e.g. GTDB), but
its composition is inferred over the panel genomes plus a `background` row. The panel's
distinct V4 amplicons are MAPseq'd against that database (each source's home label) and
reads simulated from them are mapped the same way, giving a rectangular kernel (panel
sources × database V4 groups).
Validated on 20 mock samples against GTDB r232 (median genome TV 0.015, see
[dev/panel_reinterpretation.md](dev/panel_reinterpretation.md)) with:

```bash
nextflow run main.nf -profile singularity -c your_hpc.config \
    --input samplesheet.yml --references gtdb_r232_ssu.fasta \
    --panel_references panel.fasta \
    --sim_error_model trained --trained_error_model_scope pooled \
    --infer_presence false --outdir results
```

The flat error model misses context-specific relabels and scores no better than home labels
alone; the presence gate collapses at long runs. Mapping reads directly against the panel
(a panel-only `--references`) was more accurate on every sample; use reinterpretation when
that is not possible. Requires `--mismapping_method simulate`.

#### Taxon entries

A panel entry can be a taxon instead of a genome (`--panel_taxa`, alone or with
`--panel_references`):

```
id	taxon
bacteroides_fragilis	Bacteria;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides;Bacteroides fragilis
streptococcus_salivarius	Streptococcus salivarius
```

`taxon` is a lineage prefix of `--taxonomy`, matched at `;` boundaries, or a bare name that
ends exactly one prefix (zero or several is an error listing the candidates: SILVA has
cross-kingdom homonyms). Its sources are the database V4 groups whose sequences lie under
it, each a free parameter, and the entry's abundance is their sum. A group that is a genome
entry's source stays with the genome, and a sequence counts for the most specific taxon
above it, so `Bacteroides` next to a *B. fragilis* genome means "other *Bacteroides*". A
group whose sequences fall under two unnested taxa is shared by both, which then report
`not_identifiable`.

`inferred_composition.csv` is per entry, with intervals and `presence_prob` from summed
posterior draws; the fitted members (`<entry>::<v4g>`) are in
`<id>.inferred_panel_members.csv`. **Use species taxa only.** Against SILVA NR99 a species
is 1–24 V4 groups, and a panel of 20 species plus two genomes scores genome TV 0.034
(horseshoe) / 0.039 (no gate) with no misfit, against 0.030 for the same genomes as genome
entries; the presence gate misfits on half the samples. A genus is hundreds to thousands of
groups, and inference over more than a few hundred members collapses to a near-uniform
composition (entry TV 0.21–0.35, `model_misfit` on every sample). Both in
[dev/panel_silva_sweep.md](dev/panel_silva_sweep.md).

Containers: `--sra_skiver_tag` (default `latest`), `--mapseq_tag` (default
`2.1.1b--hc47f52e_1`). Resources: `--max_cpus`, `--max_memory`, `--max_time`.

## Outputs

```
results/
  error_models/<id>/
    <id>.model.pt                    trained context error model
    <id>.phred_calibration.json
    <id>.context_model_aic.csv       AIC over candidate architectures
    <id>.error_model_report.html     diagnostic report
  amplicons/<id>_amplicons/
    amplicons.fasta                  extracted reference amplicons (the V4 groups)
    references.tax                   mapseq tax over every reference, for the FASTA
    translation_table.tsv            compact genome->reference table T
    refseq_index.csv                 per-reference amplifiability
  mapseq/<id>/
    <id>.obs.mseq.gz                 mapseq classification of the real reads
  mismapping/
    panel_<key>/                     rectangular kernel (mismapping_matrix.npz),
                                     panel_sources.tsv, panel_translation.tsv, sources.tsv,
                                     provenance.json (what --panel_kernel checks)
  composition/<id>/
    <id>.inferred_composition.csv    inferred vs observed genome abundances
    <id>.inference_diagnostics.csv   kernel ID/path, fit settings and status, background share
    <id>.posterior_draws.npz         theta_eff and fitted nuisance posterior draws
    <id>.fit_diagnostics.json        posterior-predictive forward-fit gate
    <id>.loss_trace.csv              (vi/mle)
    <id>.inferred_panel_members.csv  panel with taxon entries: the fitted per-member table
  pipeline_info/                     trace, report, timeline, dag, software versions
```

`inferred_composition.csv` has one row per genome: `observed_rel_abundance`,
`inferred_mean`, `inferred_lo`/`inferred_hi` (5–95% credible interval for `vi`/`nuts`),
and `presence_prob` (posterior probability the genome is present; empty when
`--infer_presence false`). Call a genome present at `presence_prob >= 0.5`.

`fit_diagnostics.json` is calculated from the real MAPseq counts, the exact panel kernel,
and retained posterior draws. It reports observed-versus-expected total-variation distance
and a posterior-predictive percentile over the fitted labels (observed database V4 groups,
a zero-count sink, and the background label), under `label`.
`fit_status=model_misfit` is a release gate; `low_depth` is not a biological composition.
Treat `fit_diagnostics.json`, rather than the composition CSV alone,
as the result's release status.

## Benchmarking

This pipeline is driven as a nested run by
[synthetic-metagenomic-benchmark-pipeline](https://github.com/timrozday-mgnify/synthetic-metagenomic-benchmark-pipeline)
(`profilers: [sr_amplicon]`), which generates synthetic reads with a known truth and
scores the result. The interface it depends on — keep these stable:

| direction | contract |
|-----------|----------|
| in | `--input` YAML samplesheet of `{id, reads, platform, references}` rows; `references` is one combined FASTA with `genome\|n\|orig` headers. |
| in | `--fwd_primer` / `--rev_primer`, so the reference amplicons are cut from the same region the reads were amplified from. |
| in | `--panel_kernel <dir>`, a `mismapping/panel_<key>/` bundle built by an earlier run of this pipeline against the same amplicons, MAPseq database and panel. |
| in | `--infer_presence`, `--infer_presence_prior`, `--infer_presence_temp`. |
| out | `composition/<id>/<id>.inferred_composition.csv`, with `genome_id` and `inferred_mean` — the columns the benchmark's `normalize_sr_profile.py` reads. |
| out | exactly one `mismapping/panel_<key>/` bundle per run (`mismapping_matrix.npz`, `panel_translation.tsv`, `sources.tsv`, `provenance.json`). The benchmark lifts the directory, and it requires the run to publish **one** bundle. |
| out | a sample whose reads hit no reference is an all-zero composition with `status=no_reference_hits`, not a failure. |

Because the benchmark builds the matrix once per reference set and then supplies it to
every sample, **the mis-mapping mode is a property of the matrix-build run only**. The
modes worth comparing:

| `--mismapping_method` | `--align_tau` | what it costs |
|---|---|---|
| `simulate` | – | Reads simulated from every panel source and mapped with mapseq. The measurement; the most expensive. |
| `align` | `0` | A literal hash join of sources onto byte-identical labels. One pass. |
| `align` | `>= 1` | Also verified neighbours within `tau` (pigeonhole candidates), each discounted by `--align_distance_decay ** d`. Leave the decay at `1` and this mode understates how well the mapper separates near-identical references. |

The benchmark exposes these as `--sr_amplicon_mismapping_method` and
`--sr_amplicon_align_tau`, with
`--sr_amplicon_matrix_args` for the remaining flags
(`--align_distance_decay`, `--align_ambiguity_weight`, `--max_ambiguous_bases`,
`--max_postings`, `--sim_n_per_ref`, …). See its README for how to sweep them.

## Containers

Two images. Read mapping runs in the stock mapseq biocontainer
(`quay.io/biocontainers/mapseq:${mapseq_tag}`, Singularity via depot.galaxyproject.org);
everything else runs in `sra-skiver`: the skiver Rust binary + its Python error-model
library, plus torch (CPU), Pyro, and Quarto. Built from
`containers/skiver/Dockerfile` with **build context = repo root** so the
`vendor/skiver` submodule is visible. Clone with submodules:

```bash
git clone --recurse-submodules <repo>
# or, after a plain clone:
git submodule update --init --recursive
```

**Building on Apple Silicon:** the image builds cleanly on `linux/amd64` (CI runners,
HPC login nodes). It does **not** build natively on `arm64` — `rust-htslib` has an
aarch64 type-signedness bug — and building `amd64` under QEMU on a Mac segfaults `cc1`.
So on a Mac, don't build the self-contained image locally. Instead, retag the already
published `smb-skiver` (which has the skiver binary) as the ref the pipeline expects —
then run under emulation:

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker pull ghcr.io/timrozday-mgnify/smb-skiver:latest
docker tag ghcr.io/timrozday-mgnify/smb-skiver:latest ghcr.io/timrozday-mgnify/sra-skiver:latest
nextflow run main.nf -profile docker -c tests/nextflow.config \
  --input assets/samplesheet.example.yml --outdir results
```

This is verified to run the full pipeline end-to-end on the fixture (slow, emulated).
The mapseq biocontainer is amd64-only too, so it runs emulated on a Mac as well; the
non-skiver Python stages run natively anywhere.

## Testing

```bash
# Structure / wiring (no containers, no compute):
nf-test test --tag stub

# Real end-to-end on the bundled fixture (needs the built image):
nextflow run main.nf -profile docker --input assets/samplesheet.example.yml --outdir results_test

# Script self-checks:
python bin/subspecies_infer.py amplicons --demo   # in-silico PCR, T, M-from-mseq
python bin/simulate_amplicon_reads.py --demo      # fragment sampler + flat error model
python bin/reads_to_fasta.py --demo

# Build a custom database, then verify MAPseq accepts its FASTA/tax pair.
pytest tests/test_build_mapseq_database.py         # needs Docker; pulls the pinned MAPseq image

# Does the flat error rate have to be accurate? (needs docker + mapseq)
python dev/error_rate_sensitivity.py
```

Regenerate the fixture with `python tests/data/generate_fixture.py`.

## Development

- Nextflow DSL2, `>=25`. nf-core-style module/subworkflow layout; validation is
  imperative in `main.nf` (no `nextflow_schema.json`).
- `bin/` python scripts are staged onto `PATH` by Nextflow. `simulate_amplicon_reads.py`
  imports the skiver error-model library from `$SKIVER_SCRIPTS` (set in the image) or the
  `vendor/skiver` checkout; nothing else needs it.
- The mis-mapping + inference engine originates in the benchmark pipeline's
  `reports/scripts/subspecies_infer.py`; here `infer_composition.py` replaces its
  benchmark-coupled `stage_infer` with a standalone per-sample driver.
- Mis-mapping is *measured*, not modelled: the earlier analytic
  `Φ(-E[LLR]/sd[LLR])` matrix and its edlib read scorer were removed when mapping moved
  to mapseq — a model of an aligner is no substitute for the aligner. The simulate →
  map → tally structure mirrors `superresolution-shotgun`
  (`bin/simulate_chunk_reads.py` + `shotgun_infer.py::build_mismapping`).
- `--mismapping_method align` is a deliberate, bounded exception to that rule, and not a
  return to modelling the aligner's scoring. It rests on a measured finding: in a 16S
  amplicon reference set the confusion is *redundancy* — 73 of the 81 references in the
  B. uniformis set are byte-identical over the amplicon — so `M` is a property of the
  references, not of the aligner, and any mapper must produce it
  ([census](dev/amplicon_distance_census.md)). It is validated *against* the measurement
  rather than trusted a priori ([equivalence study](dev/alignment_mismapping.md)), and it
  stays opt-in: a reference set whose members differ by one or two bases is exactly where
  a distance-only `M` is expected to miss, and that case has not been tested.
  [docs/alignment_mismapping_plan.md](docs/alignment_mismapping_plan.md) records the
  design and, in full, its limitations and biases — read it before using `align` on a
  reference set unlike that one, on high-error long reads, or on a reference set carrying
  many IUPAC ambiguity codes (mapseq penalises `N`-bearing references in a way a
  distance-based `M` cannot express — use `simulate` there).
