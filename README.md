# superresolution-amplicon

Error-model-driven **sub-species genome composition inference** from amplicon (e.g.
16S V4) sequencing reads. Given fastq reads and a set of reference sequences, the
pipeline:

1. **Trains a sequencing error model** from the reads (reference-free, via
   [skiver](https://github.com/timrozday-mgnify/skiver)).
2. **Quantifies reference-to-reference mis-mapping** — how easily each reference's
   amplicon is confused for another under that error model — both analytically and by
   **simulation (sampling)**.
3. **Infers the true genome composition** with a Bayesian model (Pyro) that inverts the
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

Mis-mapping (in-silico PCR + sampling):

| param | default | description |
|-------|---------|-------------|
| `--fwd_primer` / `--rev_primer` | V4 515F / 806R | Amplicon primers. |
| `--primer_mismatches` | `3` | Allowed primer mismatches. |
| `--mismap_n_sim` | `500` | Simulated reads per reference (sampling path). |
| `--mismap_sim_bins` | `40` | Score-histogram bins. |
| `--gap_penalty` | `4.0` | Alignment gap penalty for scoring. |

Composition inference (Pyro):

| param | default | description |
|-------|---------|-------------|
| `--infer_mode` | `vi` | `vi` \| `nuts` \| `mle`. |
| `--infer_alpha` | `0.5` | Dirichlet prior concentration. |
| `--infer_steps` | `3000` | SVI steps (vi/mle). |
| `--infer_lr` | `0.02` | SVI learning rate. |
| `--infer_num_samples` | `500` | Posterior samples (vi/nuts). |
| `--infer_warmup` | `500` | NUTS warmup. |
| `--infer_max_reads` | `20000` | Reads scored per sample. |
| `--infer_comp_scale` | – | Composite-likelihood downweight (default = n refs). |

Container: `--sra_skiver_tag` (default `latest`). Resources: `--max_cpus`,
`--max_memory`, `--max_time`.

## Outputs

```
results/
  error_models/<id>/
    <id>.model.pt                    trained context error model
    <id>.phred_calibration.json
    <id>.context_model_aic.csv       AIC over candidate architectures
    <id>.error_model_report.html     diagnostic report
  mismapping/<id>_mismapping/
    mismapping_matrix.csv            analytic reference->reference mis-mapping M
    mismapping_matrix_sim.csv        sampled (simulated) M
    mismapping_compare.csv           analytic-vs-sampled agreement
    translation_table.csv            genome->reference table T
    refseq_index.csv                 per-reference amplifiability
    v4_amplicons.fasta               extracted amplicons
    score_components.npz             per-reference score-distribution tensor D
  composition/<id>/
    <id>.inferred_composition.csv    inferred vs observed genome abundances
    <id>.inference_diagnostics.csv
    <id>.loss_trace.csv              (vi/mle)
  pipeline_info/                     trace, report, timeline, dag, software versions
```

`inferred_composition.csv` has one row per genome: `observed_rel_abundance`,
`inferred_mean`, and `inferred_lo`/`inferred_hi` (5–95% credible interval for
`vi`/`nuts`).

## Containers

One image (`sra-skiver`) serves every process: the skiver Rust binary + its Python
error-model library, plus torch (CPU), Pyro, edlib, and Quarto. Built from
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
So on a Mac, don't build the self-contained image locally. Instead, reuse the already
published `smb-skiver` (which has the skiver binary) and layer `edlib` on top, tagged
as the ref the pipeline expects — then run under emulation:

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
docker pull ghcr.io/timrozday-mgnify/smb-skiver:latest
printf 'FROM ghcr.io/timrozday-mgnify/smb-skiver:latest\nRUN pip install --no-cache-dir edlib\n' \
  | docker build --platform linux/amd64 -t ghcr.io/timrozday-mgnify/sra-skiver:latest -f - .
nextflow run main.nf -profile docker -c tests/nextflow.config \
  --input assets/samplesheet.example.yml --outdir results
```

This is verified to run the full pipeline end-to-end on the fixture (slow, emulated).
The non-skiver stages (`MISMAPPING`, `INFER_COMPOSITION`) are pure Python and run
natively anywhere.

## Testing

```bash
# Structure / wiring (no containers, no compute):
nf-test test --tag stub

# Real end-to-end on the bundled fixture (needs the built image):
nextflow run main.nf -profile docker --input assets/samplesheet.example.yml --outdir results_test

# Inference driver self-check:
python bin/infer_composition.py --demo    # needs numpy/torch/pyro (use the container)
```

Regenerate the fixture with `python tests/data/generate_fixture.py`.

## Development

- Nextflow DSL2, `>=25`. nf-core-style module/subworkflow layout; validation is
  imperative in `main.nf` (no `nextflow_schema.json`).
- `bin/` python scripts are staged onto `PATH` by Nextflow. `subspecies_infer.py` and
  `infer_composition.py` import the skiver error-model library from `$SKIVER_SCRIPTS`
  (set in the image) or the `vendor/skiver` checkout.
- The mis-mapping + inference engine originates in the benchmark pipeline's
  `reports/scripts/subspecies_infer.py`; here `infer_composition.py` replaces its
  benchmark-coupled `stage_infer` with a standalone per-sample driver.
