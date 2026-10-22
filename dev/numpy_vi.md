# numpy_vi: the `--mode vi` fit without torch

`bin/numpy_vi.py` is `subspecies_infer._fit`'s default path (rectangular kernel, Dirichlet-
Multinomial, mis-mapping scale `s`, presence gate) in numpy and scipy, for Pyodide.
`infer_composition.py --backend numpy` runs it. The default backend is still pyro.

## Same estimator as pyro

- **Per draw.** Pyro's samplers were fed numpy's noise (`_standard_normal` and
  `RelaxedBernoulli`'s `torch.rand` patched). On a toy panel, `Trace_ELBO`'s gradient
  equals `loss_and_grad`'s to 2e-12 relative over 20 draws, with the presence gate on.
  `tests/test_numpy_vi.py` repeats the check against torch's autograd through sramp's own
  `_apply_mismapping` and pyro's distributions.
- **Objective.** At pyro's fitted parameters for Nov2025 `A_0`, the mean loss over 20,000
  draws was 304.94 ± 0.37 for numpy and 304.61 for pyro (one 20,000-particle estimate).
- **Over seeds.** On a toy panel at lr 0.05, 40 seeds each: g0 0.614 (pyro) and
  0.619 (numpy), each with a spread of about 0.012 per seed. All 14 guide parameters
  are within ±2.2 standard errors.

## Nov2025 (53 samples × 25 genomes, seeds 0–9, defaults: 3000 steps, lr 0.02)

| check | result |
|---|---|
| presence calls where pyro's 10 seeds are unanimous | 1,199; numpy's 10 seeds agree on 1,192 |
| the 7 others | pyro's seeds sit at 0.50–0.74 or are widely spread (`A_0`); one numpy seed crosses 0.5 |
| calls where pyro's own seeds disagree | 126; no backend can match a call pyro itself does not make |
| mean presence spread over seeds | pyro 0.027, numpy 0.022 |
| L1 between the two backends' seed-mean compositions | 0.0076 |
| L1 between two halves (5 seeds) of pyro's own runs | 0.0171 |

M6 asked for "every 0.5 presence call agrees". Pyro does not meet that against itself
on 126 of 1,325 calls, so the check above uses the calls pyro makes on every seed.

## Speed (one sample's fit, Apple M-series)

| implementation | per sample | all 53 samples |
|---|---|---|
| pyro (torch 2.11, 1 thread) | 7.5 s | 400 s |
| numpy_vi, CPython | 0.32 s | 17 s |
| numpy_vi, Pyodide v314.0.7 in Chromium | 0.41 s | 22 s |
| numpy, all samples in one batch (prototype), CPython | – | ~0.8 s |
| numpy, all samples in one batch (prototype), Pyodide | – | ~1.1 s + Adam |
| Rust (statrs, rand_xoshiro; same gradient), native | 0.012 s | 0.63 s |

A step costs about 140 µs, almost all of it Python and numpy call overhead: the arrays
hold 25 genomes and 35 kernel entries. Fitting every sample in one batch shares that
overhead. All samples in a run share the panel's genomes and rows, only their observed
labels differ, and those can be laid end to end. The batched gradient equals the
per-sample one to 2e-12. It lands within 2× of native Rust, so a Rust/wasm component,
with its own toolchain and release, buys little.
