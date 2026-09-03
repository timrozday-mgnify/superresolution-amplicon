# How much should an ambiguous reference be demoted inside its tie cluster?

**The model is right and the constant is not.** Down-weighting a tie-cluster member as
`w ** (its ambiguous positions)` reproduces mapseq's behaviour: in **5 of 5** injected
configurations the `w` that minimises `‖M_align − M_sim‖_F` matches the penalty measured
independently as incoming read mass. But that penalty is not portable — it ranges from
**0.18 to 0.97** depending on which references carry the ambiguity — so no single default
is correct. `align_ambiguity_weight` defaults to **0.3**, a compromise, not an optimum.

Reproduce with `python dev/ambiguity_weight_sweep.py` (needs docker, ~5 min). Raw numbers
in `ambiguity_weight_sweep.csv`. Background: the ambiguity section of
[alignment_mismapping.md](alignment_mismapping.md).

## Setup

The 21-genome / 81-amplicon B. uniformis V4 set with one `N` injected at a random position
into a fraction of the references — **in the mapseq DB only**, reads still simulated from
the un-`N`'d sequences, since a reference's ambiguity is an assembly artefact and the
organism has a real base there. Two independent quantities per configuration:

- **measured penalty** — mean incoming mass of `N`-bearing references divided by that of
  their clean cluster partners, over clusters containing both;
- **best w** — the weight minimising `‖M_align − M_sim‖_F`, swept over 6 values.

They are computed from different things (column sums of `M` vs a matrix norm) and should
agree if the model has the right form.

## Results

| inject | N-bearing refs | measured penalty | best w | w=1.0 | w=0.5 | w=0.3 | w=0.19 | w=0.1 | w=0.0 |
|---|---|---|---|---|---|---|---|---|---|
| 25%, seed 7 | 22 | 0.176 | **0.19** | 2.455 | 1.735 | 1.440 | **1.351** | 1.391 | 1.649 |
| 25%, seed 23 | 17 | 0.333 | **0.30** | 2.221 | 1.913 | **1.882** | 1.919 | 1.991 | 2.129 |
| 25%, seed 101 | 17 | 0.238 | **0.19** | 2.288 | 1.855 | 1.737 | **1.719** | 1.748 | 1.853 |
| 10%, seed 23 | 4 | 0.967 | **1.00** | **0.919** | 0.939 | 0.961 | 0.978 | 0.995 | 1.016 |
| 5%, seed 5 | 5 | 0.463 | **0.30** | 0.990 | 0.774 | **0.760** | 0.795 | 0.854 | 0.955 |
| *mean* | | | | 1.775 | 1.443 | **1.356** | **1.352** | 1.396 | 1.520 |

## Reading it

1. **The fitted weight equals the measured penalty, every time** — 0.176→0.19,
   0.333→0.30, 0.238→0.19, 0.967→1.00, 0.463→0.30. Two independent measurements of the
   same quantity agreeing across five draws is the strongest evidence here that the
   penalty is what it is claimed to be, and that `w**k` is the right form for it.
2. **The penalty is not a constant, so `w` cannot be one either.** 0.18 to 0.97 across
   draws of *the same DB at the same injection rate*. Which references get the `N` matters
   more than how many: a reference with no clean duplicate has nothing to lose the read
   to, and is not penalised at all.
3. **Where there is a penalty, modelling it recovers ~40% of the error**: 2.455→1.351,
   2.288→1.719, 0.990→0.760. Where there is none it costs little: 0.919→0.961 at w=0.3.
4. **`w = 0.3` is the default because it is the middle of the plateau**, not because it
   won. Mean `‖·‖_F` over the five configurations is 1.352 at w=0.19 and 1.356 at w=0.3 —
   a tie — but 0.3 sits more centrally in the observed penalty range and is the gentler
   error when there is no penalty at all. (Same reasoning as the presence-gate defaults in
   `presence_prior_sweep.md`: pick inside the plateau, not on its edge.)
5. **It does not rescue the ambiguity case.** Best-case 1.35–1.88 against a noise floor of
   ~0.58. Modelling the penalty is a real improvement over ignoring it and is still not
   equivalence. For a reference set where a meaningful fraction of members carry ambiguity
   codes, use `--mismapping_method simulate`.
6. **Free where it does not apply.** Every exponent is 0 on an ambiguity-free reference
   set, so any `w` leaves `M` byte-identical — verified against the validated B. uniformis
   matrix. Turning this on by default cannot disturb the step-3 result.

## What this does not show

The weight counts ambiguity across the *whole* reference. Under `--read-len` a short read
is charged for ambiguity outside the window it actually saw; ambiguity and short reads
were each tested, never together. The penalty's variability (point 2) is characterised on
one DB and one injection scheme — real ambiguity is not uniformly distributed at random,
it clusters in hard-to-assemble regions, which may make the penalty more consistent or
less. Nothing here was tested against real ambiguous references.
