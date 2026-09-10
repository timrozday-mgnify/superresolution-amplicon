# Should the distance decay be fitted per sample?

**Yes, where the reference set identifies it — and it says so when it doesn't.** Fitting
`c` recovers the decay a sample's reads actually experienced from a matrix built up to
30x away from it, and cuts composition L1 by **2–9x** against using the built value. On a
reference set that cannot identify it, the fit stays at the build and tracks the fixed
run.

Reproduce with `python dev/latent_distance_decay.py` (needs torch/pyro, ~2 min). Raw
numbers in `latent_distance_decay.csv`.

## Why it is a latent at all

`--align_distance_decay` is a property of the *sample* — roughly its per-base error rate —
but the mis-mapping matrix is built once per reference set and shared by every sample in
the matrix group. A fixed `c` therefore has to be right for all of them at once.

It does not have to be. The tie-cluster kernel is `M[a, j] ∝ u_j · c**d(a, j)` and each
row is normalised, so the built matrix plus the distance behind each of its nonzeros is
enough to rebuild it at any other decay:

    M(c) = rownorm(M(c0) · (c / c0)**d)

`tau >= 1` builds store those distances (`sparse_matrix`'s strata), and
`--infer_distance_decay` samples `c ~ LogNormal(log c0, 1.2)` and rebuilds `M` inside the
model. The expensive stage — alignment, one build for the whole group — is untouched.

**`s` cannot do this.** The existing mis-mapping scale mixes `M` with the identity,
`M_eff = (1-s)I + sM`, which rescales a whole row's off-diagonal mass and leaves every
ratio *within* it alone. `c` is exactly the ratio between an exact duplicate and a
one-edit neighbour. The two are not redundant, and where a row holds only one distance
class they are not separable either — see below.

## Results

Synthetic kernels, 6 genomes, 100k reads, matrix built at `c0 = 0.01`:

| reference set | true c | fitted c | L1 latent | L1 at the built decay |
|---|---|---|---|---|
| identified (14 refs / 6 genomes) | 0.005 | 0.0043 | **0.0063** | 0.0130 |
| identified | 0.02 | 0.0150 | **0.0107** | 0.0155 |
| identified | 0.05 | 0.0472 | **0.0110** | 0.0404 |
| identified | 0.15 | 0.1519 | **0.0117** | 0.1157 |
| identified | 0.3 | 0.2867 | **0.0207** | 0.1872 |
| saturated (6 refs / 6 genomes) | 0.005 | 0.0087 | 0.0103 | 0.0103 |
| saturated | 0.05 | 0.0091 | 0.0071 | 0.0153 |
| saturated | 0.3 | 0.0113 | 0.0717 | 0.0813 |
| duplicates only (no neighbours) | any | *refused* | – | 0.0107 |

## Reading it

1. **Recovery is good over a 60x range.** 0.005 → 0.0043, 0.05 → 0.0472, 0.15 → 0.1519,
   0.3 → 0.2867. The matrix is never rebuilt; only the decay moves.
2. **The gain grows with the error the fixed decay makes.** At `c_true = c0` there is
   nothing to win (and nothing lost: 0.0063 vs 0.0130 is the latent finding the truth
   slightly better than the build's own value). At 30x away, L1 is 9x better.
3. **`saturated` is the honest failure.** With as many references as genomes, `r_obs` has
   exactly the degrees of freedom `theta` does, so the composition can explain any decay
   by itself and the likelihood says nothing about `c`. The fit stays within a factor of 2
   of the build and the composition tracks the fixed run — the prior is doing the work,
   which is the fallback, not a failure to notice. A separate topology check (7 refs, 4
   genomes, but with both one-edit pairs *inside* a single genome) lands in the same
   place: latent L1 0.2014 against fixed 0.1985. Leak between two copies of one genome
   cancels when the reference space collapses to genome space, so what identifies `c` is
   cross-genome leak that differs across distance classes, not the number of near
   neighbours as such.
4. **All-duplicate matrices are refused, not fitted.** If every distance is 0 then `c`
   cancels identically and the ELBO is flat in it; `infer_composition.py` errors out
   rather than reporting a number drawn from its prior. That is the `--align_tau 0` case,
   where the decay was always a no-op.

## When to turn it on

`--infer_distance_decay` needs `--mismapping_method align --align_tau >= 1`. It is worth
it when one matrix serves samples that differ — platforms, read lengths, run quality — or
when you do not want to pick `c` at all. It is not worth it on a reference set whose
clusters are all exact duplicates (there is nothing to fit) or where references barely
outnumber genomes (nothing to fit it *with*): `dev/amplicon_distance_census.py` reports
how many rows of a real set mix distance classes.

If you would rather keep it fixed, `--align_distance_decay auto` measures the per-base
error rate off the error model instead of guessing it — the flat rates, or a pre-trained
skiver model via `--align_decay_model`, with no training run either way. At the pipeline's
flat defaults that gives `c = 0.0059`, against the `0.007` fitted by matching the
simulate+map matrix on the two-strain *B. uniformis* set.
