# Measuring M by simulation on a batch's active groups (plan Phase 5)

Phase 4 showed the alignment-built kernel does not describe MAPseq at GTDB scale. This
measures `M` instead — simulate reads from the groups a batch's reads could have come from,
map them with the same MAPseq against the *full* database, and tally. All of it runs
locally; MAPseq runs in the pinned amd64 container under emulation.

    python bin/select_active_amplicons.py --amplicons reference/amplicons.fasta \
        --obs-mseq results/mapseq/*/*.obs.mseq.gz --tau 1 -o active_amplicons.fasta
    python bin/simulate_amplicon_reads.py --amplicons active_amplicons.fasta \
        --error-model flat --sub-rate 0.0025 --ins-rate 0.00025 --del-rate 0.00025 \
        --n-per-ref 500 --seed 7 -o sim.fasta
    docker run --rm --platform linux/amd64 -v $PWD:/w -w /w \
        quay.io/biocontainers/mapseq:2.1.1b--hc47f52e_1 \
        mapseq sim.fasta amplicons.fasta amplicons.tax -nthreads 8 > sim.mseq
    python bin/infer_composition.py --amplicon-dir reference --sim-mseq sim.mseq \
        --build-mismapping --build-grouped --active-amplicons active_amplicons.fasta -o measured

## Cost (this Mac, docker amd64 emulation)

| Step | SC2200627-SC3 (54 samples, GTDB r232) |
|---|---|
| `.mscluster` for 1,001,241 amplicons (once per database) | ~25 min, 2.1 GB (10 min native, per the archived run's trace) |
| Active-group selection, `tau=1` | 84 s — 5,949 of 86,557 groups (444,846 references) |
| Simulation, 500 reads per active group | 4.2 min, 2,814,000 reads |
| MAPseq of those reads vs the full database | 10.9 min |
| Build the grouped matrix | 8 s |

A single known-truth benchmark sample needed only 264 active groups: the whole chain ran in
about 2 minutes.

## Does it work? Custom database first

Simulated 500 reads per reference for the 22-genome community DB, mapped, built the matrix,
and inferred sample D2 with it: `ok` (group TV 0.0118, PPC percentile 0.864) and a
composition TV 0.0047 from the archived alignment-kernel result. Row-stochastic to 1e-6,
group self-mass median 0.989.

## Against real ground truth (the decisive test)

The DADA2 ASV profile is *not* a gold standard (the independent custom-database run sits a
median TV 0.12 from it). Instead: a benchmark sample with a known composition
(`synthetic-metagenomic-benchmark-pipeline_runs/sr_amp_param_sweep`, S01, 21 genomes,
`realized_rel_abundance`), its reads merged and primer-trimmed by the pipeline's own
`reads_to_fasta.py`, then mapped against **full GTDB** — a GTDB-scale problem whose answer
is known. Truth is carried into V4-group space through the community's copy weights.

Two truth groups are *never labelled by MAPseq*: `v4g_2acb4710c828d339` (6.4% of the truth)
and `v4g_a3e62c1c4b72ed9c` (7.7%). This confirms Phase 4's relabelling against ground truth
rather than against ASVs.

Raw rows in `gtdb_measured_matrix_truth.csv`:

| Inference | Matrix | TV to truth | Mass outside truth | `2acb` (truth 0.064) |
|---|---|---:|---:|---:|
| none (raw MAPseq labels) | — | 0.209 | 0.188 | 0.000 |
| vi, no gate | alignment kernel v2 | 0.221 | 0.194 | 0.000 |
| vi, no gate | measured, sub 0.005 | 0.436 | 0.433 | 0.001 |
| vi + presence gate | measured, sub 0.005 | 0.231 | 0.207 | 0.000 |
| vi, alpha 0.05 | measured, sub 0.005 | 0.238 | 0.215 | 0.000 |
| **vi, no gate** | **measured, sub 0.0025** | **0.328** | **0.327** | **0.062** |
| vi + presence gate | measured, sub 0.0025 | 0.231 | 0.206 | 0.000 |

Three findings:

1. **The measured forward model is right.** Pushing the truth through it reproduces the
   observed labels to TV 0.062 (sub 0.005) or 0.055 (sub 0.0025), against 0.204 for the
   truth compared with the labels directly. It accounts for two thirds of the distortion.
2. **The simulated error rate now matters**, unlike the near-diagonal custom-DB case. At
   sub 0.005 the model over-predicts `2acb`'s secondary leak onto `v4g_c7cf…` by 2x
   (0.0046 predicted against 0.0020 observed); at 0.0025 it predicts 0.0025. The primary
   relabel (`2acb` -> `81c3`, 88-97%) is error-independent.
3. **A relabelled group can be recovered, and sparsity destroys it.** With the calibrated
   matrix and no gate, `2acb` comes back at 0.0615 against a truth of 0.0644 — recovered
   from its secondary leaks alone, since MAPseq never labels it. Turning the presence gate
   on collapses it to 0.0002: a hard gate prefers the single-source explanation ("all of
   `81c3`'s reads came from `81c3`"). `a3e62` is not recovered by any variant.

## What is still wrong

The unregularised fit scatters 33% of its mass onto groups absent from the truth, which is
why its TV is still worse than doing nothing. That is *between-block* underdetermination
(1,169 candidate sources against ~240 observed labels for a SC3 sample), and it is what the
horseshoe trial in the plan is for: continuous shrinkage suppresses small spurious sources
without the hard gate's winner-takes-all behaviour. Also open: more reads per group (500
gives a 1% leak an SE of ~0.45%), calibrating the error rate as a first-class step, and
reporting at identifiable-block resolution.
