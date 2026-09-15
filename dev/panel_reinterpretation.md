# Panel reinterpretation of GTDB labels: Phase 5 results

Plan: `docs/panel_reinterpretation_plan.md`. Script: `dev/panel_reinterpretation_sweep.py`.
Raw rows: `dev/panel_reinterpretation_sweep.csv` (one row per sample × method). Fits:
`work/panel_sweep/<sample>/<arm>/`.

20 sweep samples (S01–S20, BU strain mix `a0.00`–`a1.00`), 22-genome 20HM panel, reads mapped
once against GTDB r232 V4. Genome TV is over panel genomes renormalised without
`background`. All fits are vi, α 0.5, 3,000 steps unless marked `10k`. "Gate" is the
pipeline's presence gate (prior 0.01, temperature 1.0). `s1` pins the mis-mapping scale at 1
(`--s-sigma 1e-4`).

## Headline

| Method (20 samples) | TV median | TV max | `dorea` abs err | BU split err median / max | fitted `s` |
|---|---:|---:|---:|---:|---:|
| 20HM custom DB, `kmer1_latent` (gate) | **0.0090** | 0.017 | 0.0018 | 0.031 / 0.40 | — |
| 20HM custom DB, MAPseq only, no inference | 0.0102 | 0.012 | 0.0003 | 0.124 / 0.52 | — |
| 20HM custom DB, `simulate` (gate) | 0.0118 | 0.015 | 0.0025 | 0.013 / 0.19 | — |
| **GTDB + panel, trained kernel, no gate** | **0.0149** | 0.017 | 0.0003 | 0.028 / 0.071 | 1.45 |
| GTDB + panel, trained kernel, horseshoe | 0.0178 | 0.020 | 0.0005 | 0.018 / **0.039** | 1.45 |
| GTDB + panel, trained kernel, gate (Phase 4b) | 0.0193 | 0.023 | 0.0005 | 0.013 / 0.215 | 1.70 |
| GTDB + panel, home labels only, horseshoe | 0.0315 | 0.033 | 0.030 | 0.010 / 0.022 | — |
| GTDB + panel, home labels only, no gate | 0.0338 | 0.036 | 0.031 | 0.006 / 0.030 | — |
| GTDB + panel, calibrated flat, no gate | 0.0350 | 0.038 | 0.030 | 0.054 / 0.096 | 1.78 |
| GTDB + panel, align `exact`, no gate | 0.0338 | 0.036 | 0.032 | 0.014 / 0.045 | — |
| GTDB + panel, align `kmer1_latent`, no gate | 0.0458 | 0.048 | 0.026 | 0.069 / 0.150 | — |

1. **Dropping the presence gate is the best prior for reinterpretation.** With the trained
   kernel, no gate beats the gate on 20/20 samples (median −0.0049, TV 0.0149). It is still
   0.003 above custom `simulate` and 0.005 above MAPseq alone against the 20HM database.
2. **Horseshoe gives the best worst-case strain split** of any arm, custom database included
   (BU split max 0.039, against 0.071 with no gate, 0.215 with the gate and 0.19–0.52 for the
   custom database). TV is 0.0178, and it beats the gate on 16/20.
3. **With the gate, 10,000 steps collapses** (TV 0.055, fit check `model_misfit` on 2/20, `s` 2.35).
   The gate's KL keeps pulling present genomes' presence probabilities towards the 0.01
   prior. At 10k they are 0.04–0.64 (0.69–0.97 at 3k), and `s` inflates to cover the noise.
   The gate at 3k steps works only because it stops early. No gate (0.0156) and horseshoe
   (0.0203) at 10k are within 0.001–0.003 of their 3k fits.
4. **Free `s` beats pinned `s` for the trained kernel** (gate +0.0017, no gate +0.0020 when
   pinned). Pinning moves `dorea_formicigenerans` error from 0.0005 to 0.005. The inflated `s`
   (1.4–1.7) is how the fit covers the under-predicted `a49d → 2a9c` relabel. For calibrated
   flat, pinning helps slightly (0.0350 → 0.0341 with no gate), and it changes nothing for
   align `exact`.
5. **Without the trained error model, the panel is still only as good as home labels.**
   Calibrated flat and align `exact` never beat home labels only under the same prior:
   0.0350 / 0.0338 against 0.0338 with no gate; 0.0399 / 0.0348 against 0.0315 with
   horseshoe. The ~0.03 `dorea` error remains in every one of these arms.
6. **Seed noise is below every gap between arms:** 0.0009 (gate), 0.0003 (no gate), 0.0003
   (horseshoe).

## Out-of-panel deletion

The genome is removed from `panel_translation.tsv`; the kernel and reads are unchanged.
`deleted_to_background` = (background with the deletion − background with the full panel) /
the deleted genome's true abundance.

| Kernel, prior | `fusobacterium_nucleatum` (truth 6.5%): median / min | `salinibacter_ruber` (1.3%): median / min |
|---|---:|---:|
| trained, gate | 1.01 / 0.99 | 0.99 / 0.95 |
| trained, no gate | 0.98 / 0.96 | 0.98 / 0.94 |
| calibrated flat, gate | 1.02 / 1.00 | 0.96 / 0.93 |
| calibrated flat, no gate | 0.96 / 0.95 | 0.97 / 0.94 |
| home labels only, gate | 1.01 / 1.00 | 1.28 / 1.25 |
| home labels only, no gate | 0.98 / 0.97 | 1.19 / 1.16 |

All 12 arms pass (≥80% of the deleted mass goes to `background`), on every sample.
Neighbours in the panel absorb almost none of it, so the Phase 6 "measured background" work
is not triggered. For home labels only, deleting `salinibacter_ruber` sends more than its own
mass to background (1.2–1.3×). The fit check still passes every deletion arm (`ok` 20/20),
so it does not flag a missing genome either.

## Acceptance (plan Phase 5)

| # | Criterion | Result |
|---|---|---|
| 1 | Trained kernel: TV ≤ 0.02, within 0.01 of the custom DB, `dorea` ≤ 0.01 | **Met.** No gate: 0.0149, +0.003 against custom `simulate`, `dorea` 0.0003 |
| 2 | No-simulation arms below B1 | **Not met** under any prior (finding 5) |
| 3 | BU split median ≤ 0.05 | **Met.** Gate 0.013, no gate 0.028, horseshoe 0.018 (max 0.039) |
| 4 | Seed replicate below the gap between arms | **Met** (finding 6) |
| 5 | `unexplained_fraction` ≤ 1% and deletion passes | **Met.** Median 0.0004–0.0007 for trained arms; deletion 12/12 |
| 6 | Fit status `ok` | 20/20 on every arm except gate at 10k (18/20). Does not discriminate between kernels |

## Recommendation

- **Panel known before mapping:** map against the panel. MAPseq alone gives TV 0.010, and
  superresolution adds the strain split (BU split 0.013 against 0.12).
- **Reads already mapped against GTDB:** reinterpret with a trained-error-model simulated
  kernel and `--no-presence` (TV 0.015). Use `--horseshoe` when the worst-case strain split
  matters more than TV. Leave `s` free.
- **No trained error model:** reinterpretation adds nothing to home labels only. Align mode
  and calibrated flat simulation stay available, but should be documented as equivalent to
  it.
- The pipeline default of gate + 3,000 steps is fragile for panel fits (finding 3). A
  pipeline default for panel reinterpretation should be no gate.

Caveat (plan Risks): the reads were simulated with the same skiver model as the trained
kernel, so the trained arms are oracle-optimistic. Calibrated flat is the honest
no-oracle number, and it is at home-labels level.
