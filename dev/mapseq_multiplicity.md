# Does MAPseq's leak onto a neighbour grow with the neighbour's copy count?

**Barely. The grouped kernel version 1 said it grows linearly, and was wrong by ~20x at
100 copies; version 2 holds it constant.** This is Evidence §3 of the
[GTDB inference recovery plan](../docs/gtdb_inference_recovery_plan.md), and the Phase 2
acceptance check for the per-distinct-sequence kernel.

Reproduce with `python dev/mapseq_multiplicity.py --amplicons <bundle>/reference/amplicons.fasta --out work/mapseq_multiplicity`
(needs docker; ~10 s after the image is pulled). `--out` must be a directory docker can
mount; on macOS the system temp directory mounts empty and MAPseq reports "fasta db not
found". Raw numbers in `mapseq_multiplicity.csv` (the version-2 run).

## Setup

- A: one GTDB r232 V4 sequence (the 1,235th distinct ACGT-only amplicon in the
  SC2200627-SC3 bundle). B: A with its middle base substituted. 200 random GTDB background
  sequences.
- 20,000 reads simulated from A with a flat 0.5% substitution, 0.05% insertion, and 0.05%
  deletion rate.
- Database: A, B repeated `k` times, the background. MAPseq 2.1.1b (pinned biocontainer).
- Kernel: `build_kmer_grouped` on the same database at `tau = 1` and the batch's
  `auto` decay `c = 0.005895`; the leak is the A-row mass on B's group.

## Result

| B copies `k` | MAPseq reads on B | Kernel v1 leak A to B | Kernel v2 leak A to B |
|---|---:|---:|---:|
| 1 | 1.38% | 0.59% | 0.586% |
| 10 | 1.53% | 5.57% | 0.586% |
| 100 | 1.94% | 37.1% | 0.586% |
| **ratio 100 / 1** | **1.41** | **63** | **1.00** |

Version 1 (size-weighted) leaked `k·c / (1 + k·c)`: B's weight was its duplicate count
times `c^d`. MAPseq's top-hit rule does not multiply a neighbour's pull by how many
references carry it. Version 2 (per-distinct, the current `build_kmer_grouped`) gives
`c / (1 + c)` at every `k`. That is the right shape but 2.3 to 3.3x too small; the
per-sample decay (`--infer_distance_decay`) is what closes that gap, not the build-time `c`.

The MAPseq column is the 2026-09-11 version-2 re-run; the version-1 run measured
1.42 / 1.53 / 1.96% (ratio 1.38), within the run-to-run spread below.

**Phase 2 acceptance: passed.** The version-2 ratio is 1.00, inside [0.5, 2].

## Caveats

- One sequence pair, one error model, 20,000 reads. The run-to-run spread in MAPseq's
  leak is about ±0.05 percentage points; the kernel column is deterministic.
- Only a one-substitution neighbour is tested. Indel neighbours and `d = 2` are untested.
