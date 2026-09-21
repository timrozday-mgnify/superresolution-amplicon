# MAPseq's candidate funnel, and where an error-free read is labelled

Questions from the benchmark repo's SILVA sweep: does MAPseq reassign reads competitively,
why does it label so many exact-match reads with another V4 group
([mapseq_free_labels.md](mapseq_free_labels.md), finding 2), and can the `align` kernels
be made to match the `simulate` ones on SILVA?

**MAPseq labels each read on its own. What it gets wrong is which references it looks at.**
Its default candidate search often never aligns the read's own group. AAP's flags
(`-seed 12 -tophits 80 -topotus 40`, now the default `--mapseq_args`) mostly fix that. What
remains is a per-read tie-break, which `--align_home_probes` measures and puts into the
align kernel's distance-0 entry.

Scripts are in [`mapseq_candidate_funnel/`](mapseq_candidate_funnel/). Outputs are in
`work/silva/mapseq_probe/`. The reference set is SILVA 138.2 NR99 V4
(`work/silva/amp`, 464,997 amplicons, 274,493 distinct), with MAPseq 2.1.1b.

## How MAPseq labels a read (v2.1.1 source, `seqsearch` and `taxscore`)

1. It counts k-mers shared between the read and each internal cluster's *seed*. The
   clusters are the `.mscluster`, built at 80% shared k-mers. It keeps the top `-topotus`
   clusters (default 10).
2. It takes up to `-otulim` members of each (default 50), shuffles them, keeps the
   `-tophits` with the most shared k-mers (default 20), and aligns only those.
3. The label is the best alignment score. Ties go by the shuffle.

No state passes between reads, so this is not competitive mapping. A measured `M` is
exactly the likelihood of top-hit labels, and the fit's EM/VI over `M` is the competitive
step. Without `-seed`, reruns agree on the reference for 47% of reads and on the sequence
for 97%. With `-seed 12`, a query's label is a fixed function of its sequence: 200
identical copies get one label at 4 or 12 threads.

## Exact-match reads that MAPseq relabels

Of S01's reads that match a SILVA amplicon byte for byte (39,206), MAPseq's defaults label
16,095 with another V4 group. Probe: 1,000 of those and 1,000 that were kept
(`pick_reads.py`, `run.sh`, `score.py`).

| MAPseq flags | own sequence among the aligned candidates | top hit is the own sequence |
|---|---:|---:|
| defaults | 13.6% | 0.6% |
| `-otulim 0` | 16.6% | 3.7% |
| `-otulim 0 -topotus 50 -tophits 200` | 89% | 76% |

The cause is cluster selection, not `-otulim`. Clusters are ranked on their seed's k-mers
alone, so a read matching a non-seed member of a large cluster often never reaches its own
cluster. The own sequence usually has many copies: 38% of the relabelled reads' sequences
have over 200. The largest cluster has 11,891 members. MAPseq does not "prefer
well-annotated singletons", as the earlier note read it. The result depends on database
order.

Aside: 6.4% of the SILVA "amplicons" are over 320 bp, and they are the top hit for 7.5% of
S01's reads. Alignment is local, so a read inside a longer amplicon ties with its exact
match.

## Effect on deconvolution

26 panel V4 sources (`work/panel_kernel/sim_cal.fasta`), 600 simulated reads each
(`run_sim.sh`, `eval_sim.py`). Half of each source's reads measure `M`. Mixtures
(Dirichlet 0.5, 5,000 reads) drawn from the other half are fitted by EM through `M`. The
error is the total-variation distance to the realised composition, over 40 draws.

| MAPseq flags | own-sequence label rate | TV median | TV p90 | 15.6k reads |
|---|---:|---:|---:|---:|
| defaults | 0.62 | 0.0154 | 0.035 | 3.8 s |
| `-otulim 0 -topotus 50` | 0.88 | **0.0086** | **0.013** | 4.1 s |
| `-otulim 0 -topotus 200 -tophits 100` | 0.92 | 0.0092 | 0.013 | 6.9 s |
| defaults, full top-score tie set as the label | 0.52 | 0.0310 | 0.058 | – |
| `topotus 50`, full top-score tie set as the label | 0.75 | 0.0115 | 0.015 | – |

Using the whole set of tied top hits as the label (from `-print_hits`) loses to the single
top hit. It splits reads into many more classes than a finite simulation can estimate, and
`-print_hits` stops at 20 hits. AAP's `-topotus 40` was not in this sweep; it sits between
the rows that were.

## Align kernel against simulate kernel

Each row of a panel kernel (sources × SILVA V4 groups) from the align builder is compared
with the one measured by simulating and mapping (`compare_kernels.py`). The reported value
is the TV distance per row. Against GTDB the two already agree: median 0.003 at tau 0.
Against SILVA they do not:

| MAPseq flags | align kernel's distance-0 entry | tau 0: median / max | tau 1: median / max |
|---|---|---:|---:|
| defaults | the one MAPseq label of the error-free source | 0.149 / 0.718 | 0.195 / 0.678 |
| AAP | the one MAPseq label of the error-free source | 0.043 / 0.375 | 0.040 / 0.380 |
| AAP | **home distribution** (`--home-mseq` of home probes) | **0.019 / 0.138** | **0.028 / 0.135** |

The tau-1 rows use c = 0.001. The distribution rows use e = 0.0031, the simulation's
substitution rate.

What a single label misses: `v4g_6b908c…`'s error-free read ties at score 252 with
`EF185167`, which matches it only through an IUPAC code. Every error-free simulated read
goes there, and reads with any error split about 2:1 back to their own group. So the
distance-0 entry has to be a mixture, not one label:

- **`(1 - e) ** L`**, the chance a read of length `L` carries no error, goes to the
  error-free query's label.
- **The rest** is spread as `K` one-substitution copies landed. One substitution re-draws
  MAPseq's tie-break (an identical copy cannot, under `-seed`). It keeps the exact-score
  ties: a containing or IUPAC-compatible reference loses the same point.

`home_mix.py` fits `e`. At the simulation's own rate the maximum row error falls from 0.375
to 0.097. At `e = 0` it is the single label; at `e = 0.005` it is 0.151.

A measured home replaces **every** distance-0 entry, including the IUPAC-compatible
references that `--align_ambiguity_weight` used to arbitrate. The probes measured MAPseq's
tie-break directly, and keeping both double-counts (tau 0 with the IUPAC partners kept:
maximum 0.673).

## Implementation

- `build_mismapping_align.py --write-home-probes` writes the probes: `<v4g>:e` is
  error-free and `<v4g>:<i>` is one base off. `--home-mseq` reads MAPseq's labels for
  them into each group's distance-0 mass. The grouped backends (`exact-hash` and `kmer`)
  support this. The entries are at distance 0, so `--infer_distance_decay` leaves them
  alone. `e` is the error model's measured per-base rate, the same number as
  `--align_distance_decay auto`.
- In the pipeline, `--align_home_probes K` runs HOME_PROBES, then MAPSEQ_HOME (against the
  set's own database, with `--mapseq_args`), then GROUPED_MISMAPPING. It is off at 0, and
  it is part of the matrix key.
- `build_panel_kernel.py align --home-mseq <probes> --error-rate e` gives the panel align
  kernel the same distribution. Pipeline panel kernels are simulated, so they are
  unaffected.

At SILVA scale, K = 20 is 5,764,353 probes. Mapping them took 34 min at 12 threads (amd64
emulated on arm64). 99.9% hit something, and the exact-hash kernel then builds in 8 s and
0.5 GB. **33,059 of the 274,493 distinct amplicons (12%) send most of their mass to
another group even with AAP's flags.** The unmeasured tau-0 kernel keeps all of that mass
on the group itself.

## Caveats

- The kernel comparison is 26 sources, 600 reads each, and one flat error model. The
  deconvolution numbers are plain EM, with no presence gate or priors. Neither is the full
  `infer_composition.py` fit, so a SILVA sweep arm with `--align_home_probes 20` is the
  real test.
- The residual after the home distribution (tau-1 median 0.028) is the error leak onto
  neighbours, which `c` and `--infer_distance_decay` own.
- Genus accuracy on SILVA may fall. The funnel artefact sent reads out of V4 groups
  shared across genera and into single-genus neighbours
  ([mapseq_free_labels.md](mapseq_free_labels.md), finding 3). Now those reads land in
  their true group, whose lowest common ancestor is above genus.

## Reproduce

Run from the repo root:

    dev/mapseq_candidate_funnel/run.sh        # exact-match probe, needs pick_reads.py output
    dev/mapseq_candidate_funnel/run_sim.sh    # simulated panel reads, several flag sets

The Python scripts read their inputs relative to `work/silva/mapseq_probe/`, so run them
from there as `python ../../../dev/mapseq_candidate_funnel/<script>`:

- `pick_reads.py`, before `run.sh`.
- `score.py default default_rep otulim0 wide`.
- `eval_sim.py sim_default sim_t50 …`.
- `home_mix.py`.

The exception is `compare_kernels.py SIM.npz ALIGN.npz`, which runs from the repo root on
kernels built with `bin/build_panel_kernel.py build|align`.
