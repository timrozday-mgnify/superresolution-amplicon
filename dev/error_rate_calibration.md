# Error-rate calibration and reads per group for the measured matrix (plan Phase 5)

**Result: calibrating the error rate is necessary; more reads alone is not.** The flat model
used so far simulated about 10× too many indels. Simulated reads carrying an indel are placed
by MAPseq on distant labels that real reads never reach, which is what made the fit swap
true groups for unobserved neighbours. At the calibrated rate, TV to truth falls from 0.61
to 0.22. A flat model still gets individual relabels wrong. The skiver model trained on the
batch's own reads reproduces them, and gives the best block-level score, but it only matches
the raw labels at block level. What is left is identifiability inside blocks.

Setup as in `horseshoe_trial.md`: benchmark sample S01 against full GTDB r232, 264 active
groups, horseshoe prior, 10,000 SVI steps, pruning with sink labels. Raw rows in
`error_rate_calibration_scores.csv`.

## Calibration

MAPseq reports mismatches and gaps against each read's hit, so the batch's own output is the
target (`error_rate_calibration.py`):

    python dev/error_rate_calibration.py --amplicons amplicons.fasta --obs-mseq S01.obs.mseq \
        --sim-mseq sim.mseq --sim-sub 0.005 --sim-indel 0.0005 [--relabelled-share 0.14]

| | Observed S01 | Simulated reads on their own group (nominal 0.005 / 0.0005) |
|---|---:|---:|
| Mismatches per base | 0.00313 | 0.00493 (0.986 of nominal) |
| Gaps per base | 0.000042 | 0.00087 (0.869 of nominal) |

Calibrated flat model: `--sub-rate` 0.0032 (0.0026 if the ~14% of reads MAPseq relabels
onto a one-edit neighbour are discounted), and `--ins-rate` and `--del-rate` 0.000024 each.
The earlier 0.00025 indel rate was 10× too high; the 0.0025 substitution rate was about
right. Both substitution values give the same fit (TV 0.2196 against 0.2195).

Two parsing notes: GTDB headers contain `#`, so reading `.mseq` with pandas'
`comment="#"` silently truncates 22–49% of rows (`iter_mseq` is unaffected). Also, the
benchmark reads were simulated with the skiver model trained on the batch, so no true
flat rate exists to validate against. The trained model is the reference instead.

## Why the indel rate mattered

Take the simulated reads from `360e` and `87ad` that MAPseq placed on `6625`, 11 edits from
`87ad`. Every one of them carries a gap, against 10% of the reads that stayed on their own
group, and their identity to the hit is 0.95. An indel in a read sends MAPseq's search to a
distant hit. Real reads have 30× fewer indels, so the matrix predicted leak the data never
shows.

## Results

| Matrix | TV to truth | TV per block | Truth on labels with no reads | `2acb` (0.064) | `s` |
|---|---:|---:|---:|---:|---:|
| raw MAPseq labels | 0.205 | 0.106 | — | 0 | — |
| old rates (0.0025 / 0.00025), 500 reads | 0.659 | 0.057 | 1.05% | 0.065 | 1.38 |
| old rates, 5,000 reads | 0.609 | 0.120 | 0.99% | 0.000 | 1.60 |
| calibrated (0.0031 / 0.000024), 5,000 reads | **0.220** | 0.120 | 0.24% | 0.000 | 1.35 |
| calibrated, `s` pinned at 1 | 0.219 | 0.119 | 0.24% | 0.000 | 1.00 |
| trained skiver model, 5,000 reads | 0.303 | **0.107** | **0.18%** | 0.000 | 1.26 |

Blocks, fixed for every row: a group whose calibrated row sends ≥ 99% of its reads to
another label joins that label's block (148 groups merged). Every fit is still
`model_misfit` (group PPC percentile 1.000); the fit check's group TV is lowest for the
trained model (0.0154 against 0.0183).

- **More reads alone do not help.** The leak was systematic, not noise.
- **The global scale `s` is not what loses `2acb`.** Pinning it changes nothing.
- The 500-read block score (0.057) is not a better matrix. Its wrong leaks happened to
  recover `2acb`, and at group level it is the worst.

## Relabels: flat model versus trained model

| Relabel | Flat, calibrated | Trained | Real S01 reads |
|---|---:|---:|---:|
| `2acb` -> `c7cf` | 4.5% | 3.3% | ≤ 3.2% (all 162 `c7cf` reads, which is not in the truth) |
| `a49d` -> `2a9c` | 0.08% | 22.9% | ~42% (2,527 `2a9c` reads against 3,541 on `a49d`) |

A per-base flat rate cannot produce a context-specific relabel like `a49d` -> `2a9c`; the
trained model gets its direction and most of its size. Its remaining group-level error is
choices inside blocks: `fe3b`/`8a3d` (`fe3b` -> `8a3d` 100%), `2dee`/`42e2`,
`a3e62`/`89c3`, `dd9d`/`2fef`, and `90a5`, `6b90`/`b9af`, plus `2acb`. Even without
`2acb`, the fit already expects 239 reads on `c7cf` against 162 observed; with it, 399.

## Conclusions

1. Calibrate the simulation's error profile from the batch's own reads before measuring
   `M`. Prefer the trained skiver model, which the pipeline already builds per batch, to a
   calibrated flat rate. The flat rate is the fallback, and its indel rate matters most.
2. 5,000 reads per group costs about 8 minutes per batch locally (1.3M reads for 264
   groups, 410 s of MAPseq). Keep it, but it is not a fix on its own.
3. With the error profile right, group-level accuracy is limited by which member of a block
   the fit picks. Block-level reporting is the next step. Even then this sample only
   matches the raw labels (0.107 against 0.106), because `2acb` is not recovered.
