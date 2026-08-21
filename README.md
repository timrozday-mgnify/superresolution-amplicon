# superresolution-amplicon

Error-model-driven **sub-species genome composition inference** from amplicon (e.g.
16S V4) sequencing reads. Given fastq reads and a set of reference sequences, the
pipeline:

1. **Extracts the reference amplicons** by in-silico PCR, giving the mapseq reference set.
2. **Measures reference-to-reference mis-mapping** — how easily each reference's amplicon
   is confused for another — by **simulating reads from every reference under a
   sequencing error model and mapping them with the same mapper the real reads go
   through** ([mapseq](https://github.com/jfmrod/MAPseq)). The error model defaults to a
   naive flat per-mutation-type one that needs no training at all; the confusion structure
   turns out to be insensitive to it ([why](dev/error_rate_sensitivity.md)). Set
   `--sim_error_model trained` to train one from the reads instead (reference-free, via
   [skiver](https://github.com/timrozday-mgnify/skiver)).
3. **Maps the real reads** with mapseq to get the observed per-reference counts.
4. **Infers the true genome composition** with a Bayesian model (Pyro) that inverts the
   mis-mapping to recover latent genome abundances from the observed read signal.

This is a heavy, multi-stage analysis extracted from the
`synthetic-metagenomic-benchmark-pipeline` into a standalone Nextflow (DSL2) pipeline.

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
| `fastq_1` / `fastq_2` | yes\* | Alternative to `reads` (paired-end). |
| `platform` | no | `hq-illumina` \| `lq-illumina` \| `ont` \| `pacbio` (default `hq-illumina`). Sets the skiver error-model context + report notebook. |
| `references` | no | Per-sample reference fasta; overrides `--references`. |
| `error_model` | no | Path to a pre-trained `.pt` model; **skips training** for this sample. |

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

## Parameters

Run mode / IO:

| param | default | description |
|-------|---------|-------------|
| `--input` | – | YAML samplesheet (required). |
| `--references` | – | Default reference fasta (per-sample override in samplesheet). |
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
| `--trim_primers` | `true` | Trim primers off observed reads before mapping. Set `false` if reads are already primer-trimmed. |

Mis-mapping (simulate reads → map with mapseq):

| param | default | description |
|-------|---------|-------------|
| `--sim_error_model` | `flat` | `flat` (constant per-base probability per mutation type; **no training at all**) or `trained` (skiver context model). See [the sensitivity result](dev/error_rate_sensitivity.md) for why `flat` is the default. |
| `--trained_error_model_scope` | `per-sample` | With `trained`, fit one model per sample or one pooled model per platform. Explicit sample `error_model` files always win. |
| `--trained_error_model_max_reads` | `1000000` | Pooled mode: deterministic, uniformly sampled FASTQ records per platform (`0` = all). |
| `--flat_sub_rate` | `0.005` | Flat model: per-base substitution probability. |
| `--flat_ins_rate` | `0.0005` | Flat model: per-base insertion probability. |
| `--flat_del_rate` | `0.0005` | Flat model: per-base deletion probability. |
| `--sim_n_per_ref` | `500` | Simulated reads per reference (sampling depth for `M`). |
| `--sim_read_len` | – | Draw substrings of this length; unset simulates the whole amplicon (correct for merged reads). |
| `--mismapping_matrix` | – | A previously generated `mismapping_matrix.csv` for the same amplicon reference set. Skips read simulation and simulated-read mapseq; the CSV is checked against the current reference IDs before inference. |

The pipeline fingerprints extracted amplicons and builds each compatible matrix once
per run. Canonical reusable matrices are published under `mismapping/<matrix-key>/`:

```bash
nextflow run main.nf --input samples.yml --references refs.fasta \
  --mismapping_matrix results/mismapping/<matrix-key>/mismapping_matrix.csv
```

Each bundle also contains its reference sidecars and `provenance.json`. The matrix is
tied to the extracted amplicon sequences, simulator, and mapseq settings; reuse it
only with the same reference set and mapper configuration.

For a simulation-only pre-computation outside Nextflow, use the inference utility's
matrix-build mode:

```bash
bin/infer_composition.py --amplicon-dir sample_amplicons \
  --sim-mseq sample.sim.mseq.gz --build-mismapping -o mismapping_only
```

Read mapping (mapseq):

| param | default | description |
|-------|---------|-------------|
| `--mapseq_args` | – | Extra flags forwarded to **every** mapseq invocation (simulated and real alike — that identity is what makes `M` valid). |
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
| `--infer_presence` | `true` | Per-genome Bernoulli presence/absence gate. Not supported with `--infer_mode nuts`. |
| `--infer_presence_prior` | `0.01` | Prior probability a genome is present — the sparsity regulariser. |
| `--infer_presence_temp` | `1.0` | Concrete relaxation temperature for the gate. Below ~1 the gate barely moves off its initialisation and no prior can sparsify it. |

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
> `--trim_primers false` only if you have a reason to.

> **Why the same mapper twice.** `M` is only meaningful if the simulated reads experience
> the confusion the real reads experience, so `MAPSEQ_SIM` and `MAPSEQ_OBS` are the same
> process with the same settings, differing only in which reads they take.

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
    amplicons.fasta                  extracted reference amplicons (mapseq DB)
    amplicons.tax                    mapseq taxonomy sidecar
    translation_table.csv            genome->reference table T
    refseq_index.csv                 per-reference amplifiability
  mapseq/<id>/
    <id>.obs.mseq.gz                 mapseq classification of the real reads
  mismapping/
    groups.tsv                       index of canonical matrix bundles
    <matrix-key>/
      mismapping_matrix.csv          measured reference->reference mis-mapping M
      provenance.json                 simulator, mapper, and member-sample metadata
      samples.tsv                     samples consuming this matrix
      reference/                      amplicons + mapseq/inference sidecars
  composition/<id>/
    <id>.inferred_composition.csv    inferred vs observed genome abundances
    <id>.inference_diagnostics.csv   includes canonical matrix bundle ID/path
    <id>.loss_trace.csv              (vi/mle)
  pipeline_info/                     trace, report, timeline, dag, software versions
```

`inferred_composition.csv` has one row per genome: `observed_rel_abundance`,
`inferred_mean`, `inferred_lo`/`inferred_hi` (5–95% credible interval for `vi`/`nuts`),
and `presence_prob` (posterior probability the genome is present; empty when
`--infer_presence false`). Call a genome present at `presence_prob >= 0.5`.

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
python bin/infer_composition.py --demo            # needs numpy/torch/pyro (use the container)

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
