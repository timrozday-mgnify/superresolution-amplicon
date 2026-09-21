# Panel-only parity (Phase P.7)

Code: `panel-only-inference` after P.5 (`42067aa`) and P.6 (`f18b1d7`). Baselines:
`dev/panel_only_parity/` (P.0, captured before the square path was removed). Script:
`dev/panel_only_parity.py` (`rescore`, then `compare`). Pipeline runs: `-profile docker -c
tests/nextflow.config`, default `--sim_n_per_ref` (5,000), one run per set and mode:

    --mismapping_method simulate
    --mismapping_method align --align_tau 0
    --mismapping_method align --align_tau 1 --align_distance_decay 0.005

Genome TV is over genomes, `background` dropped and renormalised.

## Result

| set | mode | genome TV vs P.0 | fit (P.0 -> now) | bar (<= 0.02) |
|---|---|---|---|---|
| fixture | simulate | 0.0132 | ok -> ok | met |
| fixture | align tau 0 | 0.0133 | ok -> ok | met |
| fixture | align tau 1 | 0.0133 | ok -> ok | met |
| B. uniformis | simulate | 0.2410 | ok -> ok | **not met** |
| B. uniformis | align tau 0 | 0.2390 | ok -> ok | **not met** |
| B. uniformis | align tau 1 | 0.2394 | ok -> ok | **not met** |

| panel | samples | max abs diff, every entry incl. `background` | fit |
|---|---|---|---|
| 20HM `sim_trained` (re-score) | S01–S20 | 0 | ok -> ok, all 20 |
| SILVA species `sim_calibrated_flat_nogate` (re-score) | S05 | 0 | ok -> ok |

The re-scores reuse each snapshot's stored kernel and observations, so they test the
inference and fit-check code alone. Their results are bit-identical: P.5 and P.6 did not change the panel
numerics. F's dropped non-amplifiable hits cannot show here, because the kernels predate F.

## Why B. uniformis moves: the presence gate collapses

The B. uniformis observation is 1,175 reads over 11 observed V4-group labels (P.0's
square fit saw them over ~50 references). With the default gate (prior 0.01, temperature
1.0) the panel fit settles in a degenerate mode:

| fit | median `conc_frac` | genomes with `presence_prob > 0.5` |
|---|---|---|
| P.0 square, gate on | 0.17 | 8 |
| panel, gate on | 0.014 | 0 |
| panel, gate off | 1.06 | – |

At `conc_frac` 0.014 the Dirichlet-multinomial concentration is ~16 pseudo-reads over
11 labels. The reads then carry almost no weight, and every gate falls back to its 0.01
prior. The same collapse held at seed 0 and at 10,000 steps. It also held when the panel
was cut to the 10 genomes with observed support, which is what the square path's
pruning did, so pruning is not what protected P.0. The horseshoe test noted the same
failure: with few labels, `conc_frac` collapses and the reads stop counting. Grouping labels
by V4 sequence, which panel-only inference always does, gives this sample few labels.

With the gate off (`--no-presence`) the panel fits sit close to P.0 and closer to the
observed composition than P.0 does:

| mode | TV(P.0, panel gate off) | the same, V4-inseparable genomes merged | TV(observed, panel gate off) | TV(observed, P.0) |
|---|---|---|---|---|
| simulate | 0.089 | 0.065 | 0.045 | 0.079 |
| align tau 0 | 0.087 | 0.039 | 0.021 | 0.070 |
| align tau 1 | 0.089 | 0.039 | 0.024 | 0.070 |

"Merged" sums the two B. uniformis strains, and *E. rectale* with *R. intestinalis*.
These are the `not_identifiable` pairs, whose split is decided by the prior. The
remaining gap is P.0's gate: it held the genomes it called present above their
observed share, for example *B. vulgatus* 0.167 against 0.149 observed.

The 20HM samples (79k reads, 54 observed labels) keep their gate. The collapse is a
low-depth, few-label effect of the panel's label space. It is not a code regression,
but it does reach the default settings, since `--infer_presence` defaults to `true`.
The panel sweeps already validated `infer_presence false`.
