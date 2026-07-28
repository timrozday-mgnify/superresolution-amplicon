# Does the simulated error rate have to be accurate?

**No.** Across a 500x range of flat error rates the mis-mapping matrix `M` is
statistically indistinguishable from the one built with a trained skiver model, and the
inferred composition is unaffected. `--sim_error_model flat` is now the default and the
skiver training subworkflow does not run unless you ask for it.

Reproduce with `python dev/error_rate_sensitivity.py` (needs docker + torch/pyro, ~15 min).
Raw numbers in `error_rate_sensitivity.csv`.

## Setup

- 81 V4 amplicons over 21 genomes — the real B. uniformis sweep reference set, including
  the two near-identical *B. uniformis* strains (`bacteroides_uniformis`,
  `BU_JCM13286_NT5170`) the pipeline exists to separate.
- Truth: those two at 3% / 17%, the other 19 genomes splitting the remaining 80%.
- 50k observed reads, 300 simulated reads per reference, mapped with mapseq.
- Two **generators** for the observed reads (a trained `additive_7_hq-illumina` preset and
  flat 1%) so the answer isn't an artefact of a simulator agreeing with itself.
- Candidates for building `M`: the trained preset, flat rates 0.0001 → 0.15, and — the
  control that makes the rest readable — **the trained model re-simulated with a different
  seed**.

## Results

| generator | candidate for M | L1 error | err *b. uniformis* | err *BU_JCM13286* | ‖M − M_trained‖_F | mean diag |
|---|---|---|---|---|---|---|
| trained | *(none: naive observed)* | 0.0237 | +0.0065 | −0.0049 | – | – |
| trained | trained | 0.0208 | +0.0027 | −0.0029 | 0.000 | 0.306 |
| trained | flat 0.0001 | 0.0228 | +0.0025 | −0.0038 | 0.626 | 0.310 |
| trained | flat 0.001 | 0.0118 | −0.0006 | −0.0002 | 0.663 | 0.310 |
| trained | flat 0.005 | 0.0202 | +0.0026 | +0.0007 | 0.591 | 0.302 |
| trained | flat 0.01 | 0.0179 | +0.0024 | −0.0003 | 0.550 | 0.303 |
| trained | flat 0.02 | 0.0174 | +0.0006 | +0.0018 | 0.609 | 0.302 |
| trained | flat 0.05 | 0.0263 | +0.0042 | −0.0050 | 0.617 | 0.303 |
| trained | flat 0.15 | 0.0227 | +0.0038 | −0.0008 | **0.896** | 0.285 |
| trained | **trained (reseed)** | 0.0175 | +0.0020 | −0.0025 | **0.645** | 0.309 |

(The flat-1% generator gives the same picture: naive L1 0.0242, every candidate 0.0167–0.0282.)

## Reading it

1. **The noise floor is the whole story.** Re-simulating the *same* trained model with a
   different seed moves `M` by 0.645. Every flat rate from 0.0001 to 0.05 sits at
   0.55–0.66 — at or below that floor. Those matrices are not measurably different from
   the trained one; the differences are Monte-Carlo sampling noise at 300 reads/reference.
2. **Only an absurd rate stands out.** Flat 15% reaches 0.896 and drops the mean diagonal
   from ~0.31 to 0.285 — and even then the composition error barely moves.
3. **Accuracy of the fit is unrelated to accuracy of the rate.** L1 spans 0.012–0.028 with
   no monotonic trend in rate, and the trained model (0.021) is beaten by several flat
   rates. The same model re-seeded varies by ±0.005, which is larger than any
   between-model difference — so none of the ordering is real.
4. **The correction itself does work.** On the hard pair, error drops from +0.0065 (naive
   observed) to +0.0006…+0.0042 with *any* `M`. Inverting mis-mapping matters; where `M`
   came from does not.
5. **Mean diagonal ≈ 0.30 regardless of error rate** — including at 0.01% error, where
   reads are essentially perfect. That is the mechanism: confusion in a 16S amplicon
   reference set is driven by references that are *identical or near-identical over the
   amplicon*, and those collide whether or not a read carries errors. Sequencing error is
   a second-order perturbation on top.

## Caveat

Point 5 is also the limit of the result. This DB's confusion is redundancy-dominated. A
reference set whose members differ by one or two bases across the amplicon — where a
single error is exactly what flips an assignment — could be more rate-sensitive, and the
too-low end (0.0001) would be the risk there, not the too-high end. The flat defaults
(0.5% substitution, 0.05% indel) sit in the middle of the range that behaved identically
here, and `--sim_error_model trained` is still one flag away.
