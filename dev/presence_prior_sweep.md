# What makes a good regularising prior for the presence/absence gate?

`python dev/presence_prior_sweep.py` — 21-genome B. uniformis V4 set, 81 amplicons,
`M` measured with the pipeline default (flat 0.005, mean diagonal **0.31** — a very
confusable reference set). Truth: 8 of 21 genomes present, the rest exactly 0, with the
confusable B. uniformis pair always present and its low member pinned at 3%. 3 replicate
subsets, 50k observed reads each. Presence called at `presence_prob >= 0.5`.

## Result

| method | prior | temp | jaccard | precision | recall | L1 | n_pred |
|---|---|---|---|---|---|---|---|
| **gate** | **0.001–0.1** | **1.0–2.0** | **1.000** | 1.000 | 1.000 | **0.019–0.023** | 8.0 |
| ungated fit (>0.01) | – | – | 1.000 | 1.000 | 1.000 | 0.031 | 8.0 |
| naive observed (>0.01) | – | – | 1.000 | 1.000 | 1.000 | 0.039 | 8.0 |
| gate | 0.001–0.1 | 0.5 | 0.889 | 0.889 | 1.000 | 0.019 | 9.0 |
| ungated fit (>0.001) | – | – | 0.859 | 0.859 | 1.000 | 0.031 | 9.3 |
| gate | 0.5 | 2.0 | 0.830 | 0.830 | 1.000 | 0.028 | 9.7 |
| gate | 0.001–0.1 | 0.1 | 0.755–0.805 | – | 1.000 | 0.034 | 10.0–10.7 |
| gate | 0.5 | 0.1 | 0.662 | 0.662 | 1.000 | 0.037 | 12.3 |
| ungated fit (>0.0001) | – | – | 0.642 | 0.642 | 1.000 | 0.031 | 12.7 |

Full grid in `presence_prior_sweep.csv`.

**Chosen default: `presence_prior = 0.01`, `presence_temp = 1.0`.** Inside the Jaccard-1.0
plateau rather than on its edge, and it never lost the 3% strain in any replicate.

## What the sweep actually showed

**Temperature matters more than the prior probability.** Below ~1.0 the relaxed gate is
too sharp to move far from its initialisation, and no prior — however small — sparsifies
it. Above it, every prior from 0.1 down to 0.001 gives the same answer. That is the
opposite of the expected story, where the prior is the dial and the temperature is a
nuisance parameter.

**Recall was 1.0 in every single configuration.** The gate never deleted a genome that was
really there, including the 3% member of the confusable pair. The whole trade is in
precision: how many absent genomes get called present.

**The gate earns its place on abundance, not just on presence.** At the chosen setting it
matches the best achievable Jaccard *and* cuts L1 by a third versus the ungated fit
(0.020 vs 0.031). Thresholding the ungated abundance can reach Jaccard 1.0 too — but only
at a cutoff (0.01) that has to be picked by hand per dataset, and it does nothing for the
abundance estimate. The gate gets there with no per-dataset tuning and improves the
composition as a side effect, because zeroing an absent genome stops it absorbing
mis-mapped reads that belong to its neighbours.

## The straight-through variant does not work

The gate was first built with `RelaxedBernoulliStraightThrough`, to get hard 0/1 gates in
the forward pass. It scored **Jaccard 0.44** — worse than the naive observed profile at
any cutoff — and, diagnostically, the prior was completely inert: fits at `prior=0.05` and
`prior=1e-12` were identical to three decimals, with 10 of 13 truly-absent genomes sitting
at `p ≈ 0.89–0.97`, i.e. still at their initialisation of `sigmoid(2) = 0.88`.

Cause: Pyro's straight-through `rsample` returns `hard + (soft - soft.detach())`, a fresh
tensor that no longer carries the `_unconstrained` soft value its `log_prob` looks for
(and `.to_event(1)` wouldn't preserve it either). Both model and guide therefore score the
gate at a hard 0 or 1, clamped to `±eps`, where the Concrete log-density is ≈ 470 nats and
dominated by a logits-independent boundary term. The prior's contribution — about 14 nats
even at `1e-6` — is lost in it, and the gradient with respect to the logits is
correspondingly flat.

Dropping to the plain `RelaxedBernoulli` fixed it (0.44 → 1.00). Raising `alpha` to make
the gate more identifiable did **not** (0.44 → 0.47 at best, and L1 blew up to 0.26 at
`alpha=20`) — the problem was the estimator, not the model.

## Known limitation: identical amplicons

When a genome's only amplicon is byte-identical to another genome's 16S copy, "present at
5%" and "absent, neighbour slightly commoner" fit the reads equally well, so presence is
not identifiable and the gate simply follows the prior — `p = 0.03` at the default,
`p = 0.92` at `prior = 0.9` (the case asserted in `infer_composition.py --demo`). The
*abundance* is still recovered correctly in both cases from the shared-copy anchor. Read a
low `presence_prob` on such a genome as "can't tell from V4", not as "not there"; the
`refseq_index.csv` and the mis-mapping matrix show which genomes are in that position.
This does not affect the sweep's B. uniformis pair, which has distinguishable copies.
