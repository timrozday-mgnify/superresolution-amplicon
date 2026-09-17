# Panel reinterpretation against SILVA 138.2 NR99: Phase V results

Plan: the benchmark repo's `docs/silva_taxon_panel_plan.md`, Phase V. Scripts:
`dev/panel_silva_sweep.sh` (database, MAPseq, simulation, kernels) and
`dev/panel_silva_sweep.py` (`prepare`, `run`, `score`). Raw rows:
`dev/panel_silva_sweep.csv` (one row per sample × method, 20 samples). Fits:
`work/silva/sweep/<sample>/<arm>/`.

Same 20 samples, 20HM panel and inference settings as `dev/panel_reinterpretation.md`
(vi, α 0.5, 3,000 steps). The GTDB r232 rows are that sweep's own fits, **rescored here**:
truth is counted from the merged reads' names (`<chunk>:<genome>:…`), not from the
benchmark's BAM, because the sweep directory with `truth.tsv` is not on this machine. The
two truths differ slightly: GTDB trained/no gate is TV 0.0127 here against 0.0149 in the
Phase 5 table. Compare rows within this file only.

Entry TV scores every fit over the taxon panel's entries: the *B. uniformis* pair as
genomes and the other 20 genomes rolled up into their 17 SILVA genera. BU split is the
error in strain2's share of the pair, per entry.

## Database

| | GTDB r232 V4 | SILVA 138.2 NR99 V4 |
|---|---:|---:|
| references (amplifiable, 515F/806R, ≤3 mismatches) | 1,001,241 | 464,997 of 510,495 |
| distinct V4 groups | – | 274,493 |
| panel sources present byte-identical | 26 / 26 | 26 / 26 |
| panel sources whose MAPseq home is another group | – | 10 / 26 |
| observed reads with no source-reachable label (home labels only, median) | 0.039 | 0.071 |

## Headline

| Method (20 samples) | genome TV median / max | entry TV median / max | BU split median / max | fit check misfit |
|---|---:|---:|---:|---:|
| GTDB, trained kernel, no gate | **0.0127** / 0.015 | **0.0086** / 0.011 | 0.028 / 0.063 | 0/20 |
| GTDB, trained kernel, horseshoe | 0.0154 / 0.017 | 0.0131 / 0.015 | 0.019 / 0.038 | 0/20 |
| SILVA genome panel, trained kernel, no gate | 0.0808 / 0.091 | 0.0750 / 0.085 | 0.199 / 0.536 | **20/20** |
| SILVA genome panel, trained kernel **length-matched**, no gate | 0.0254 / 0.037 | 0.0213 / 0.033 | 0.125 / 0.375 | 0/20 |
| SILVA genome panel, trained kernel **read-start trimmed**, horseshoe | **0.0236** / 0.027 | 0.0217 / 0.025 | 0.129 / 0.512 | 0/20 |
| SILVA genome panel, trained kernel read-start trimmed, no gate | 0.0251 / 0.031 | 0.0195 / 0.029 | 0.097 / 0.628 | 0/20 |
| SILVA genome panel, calibrated flat, gate | 0.0292 / 0.044 | 0.0268 / 0.041 | 0.059 / 0.920 | 1/20 |
| SILVA genome panel, calibrated flat, no gate | 0.0300 / 0.038 | 0.0291 / 0.037 | 0.096 / 0.630 | 0/20 |
| GTDB, home labels only, no gate | 0.0336 / 0.034 | 0.0336 / 0.034 | 0.007 / 0.025 | 0/20 |
| SILVA genome panel, home labels only, no gate | 0.0482 / 0.050 | 0.0461 / 0.048 | 0.185 / 0.709 | 0/20 |
| SILVA **taxon panel**, calibrated flat, gate | – | 0.2727 / 0.291 | 0.265 / 0.405 | **20/20** |
| SILVA taxon panel, calibrated flat, no gate | – | 0.3504 / 0.357 | 0.275 / 0.720 | 20/20 |
| SILVA taxon panel, calibrated flat, horseshoe | – | 0.3133 / 0.376 | 0.428 / 0.503 | 20/20 |
| SILVA taxon panel, home labels only, horseshoe | – | 0.2066 / 0.225 | 0.445 / 0.487 | 20/20 |

1. **A genome panel over SILVA NR99 works, but about twice as badly as over GTDB** (best
   0.024 against 0.013 genome TV), once the simulated reads are trimmed (finding 2). The
   *B. uniformis* strain split is the main loss: median error 0.10–0.20 against 0.02–0.03.
   The pair shares its home label in SILVA (`not_identifiable` on 20/20 with home labels
   only; it is identifiable on GTDB).
2. **The trained kernel fails on SILVA because simulated reads are not trimmed like the
   observed reads.** 42% of `sim_trained_s0` reads are longer than their source, mostly
   by extra 5′ bases (254–256 bp). The
   observed reads went through primer trimming and are 253 bp in 98% of cases. SILVA NR99 is
   full of one-base near-ties that MAPseq resolves by read context. *Collinsella*: every read
   carries the source's base, and 253 bp reads land on the neighbouring group `0af37f6a`
   (98% of observed reads), but reads with a 5′ insertion land on the source's own group
   73% of the time. At the true composition the kernel is TV 0.14 from the observed labels
   (GTDB 0.018). Keeping only simulated reads as long as their source
   (72,488 of 130,000) brings genome TV from 0.081 to 0.025 and the fit check to 0/20
   misfits. On GTDB the same reads cost nothing.

   97% of the inserted bases sit in the first five read positions, so this is a read-start
   effect, not sequence context. The observed reads were simulated from the same oracle
   model *with* primers and then trimmed, which cuts those bases away. Removing only the
   read-start insertions from all 130,000 trained reads (the effect the fix has) gives genome
   TV 0.024–0.025, 0/20 misfits and fitted `s` back at about 1 (0.6 before).

   **Fixed in the pipeline:** `simulate_amplicon_reads.py --trim-primers` simulates from the
   amplicon flanked by its (concretised) primers and trims each read with
   `trim_read_primers`, as `READS_TO_FASTA` does to the observed reads. `SIMULATE_READS` and
   `SIMULATE_PANEL_READS` pass it whenever `--trim_primers` is on (the default). Not rerun
   with `sc2200627.model.pt` here: the model is not on this machine.
3. **Taxon panels do not work at genus scale.** Entry TV is 0.21–0.35 on every arm,
   worse than the genome panel's home labels alone (0.046), and every fit is
   `model_misfit`. The cause is inference, not the kernel or label sharing:
   - The panel has 8,587 members (17 genera, 15–2,065 V4 groups each; 14 of 17 are above
     the default 200-group cap).
   - The Dirichlet-multinomial concentration collapses (`conc_frac` median 0.0003, against
     0.014 for the genome panel), the likelihood stops constraining θ, and VI returns θ
     close to uniform over members (largest member 0.0004; 1/8,590 = 0.00012). Predicted
     labels are TV 0.89 from the observed ones.
   - It is not the init floor (capping it at 1%: no change), not α (α = 1: 0.361), not the
     step count (20,000 steps: same loss and TV), and not unreachable members (pruning to
     the 2,385 members with mass on an observed label: 0.333).
   - It scales with member count. S01, top-n members per entry by reads on their home
     label: 54 members `conc_frac` 0.011, 498 members 0.0006, 3,967 members 0.0003. The
     54-member fit tracks truth on every entry it kept.
   - MLE is worse (entry TV 0.77): the α < 1 Dirichlet density is unbounded at the
     simplex boundary.
4. **Seven Lachnospiraceae genera are `not_identifiable` on every sample**
   (*Agathobacter, Coprococcus, Dorea, Enterocloster, Lachnoclostridium, Roseburia,
   [Ruminococcus] gnavus group*): NR99 V4 groups whose sequences span these genera are
   shared sources (rule 2). Even with working inference, V4 does not separate these genera
   in SILVA.
5. **SILVA data issues found on the way**, now handled in `build_panel_kernel.py prepare`:
   587 of 9,126 genus groups contain an ambiguous base, and 15 of them get no MAPseq hit at
   all, which fails `PANEL_KERNEL` ("no home label"). Taxon sources now exclude groups with
   any non-ACGT base.

## What follows for the plan

- ~~Fix read prep for simulated reads~~ done (`--trim-primers`, finding 2). Confirm it by
  resimulating the panel with `sc2200627.model.pt` where that model lives.
- Do not start benchmark Phase 4 (`generic_taxa` arm). Taxon entries are correct at the
  prepare/aggregation level (unit tests) but inference does not scale past a few hundred
  members. Options, in order of cost:
  1. Collapse each taxon's members that share a MAPseq home label into one member with a
     read-weighted kernel row (members with the same home are nearly indistinguishable
     anyway).
  2. Keep the fixed-mixture alternative the plan rejected, as a fallback for broad taxa.
  3. Rework the high-dimensional fit (guide init scale, a different θ parameterisation, or
     a collapsed likelihood).
- The 200-group cap is not a cost guard in practice: the fit fails well before simulation
  cost matters. Keep it until option 1 or 3 lands.
- Simulation cost measured: 8,513 taxon sources × 1,000 reads = 8.5M reads, 3 min to
  simulate (8 processes), 32 min to MAPseq (12 threads, emulated amd64 on arm64). Each fit
  takes about 13 s at 8.6k members.
