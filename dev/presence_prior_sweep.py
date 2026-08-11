#!/usr/bin/env python
"""What makes a good regularising prior for the presence/absence gate?

The gate adds a per-genome Bernoulli ``z`` to the composition model; its prior
probability is a sparsity regulariser (small => a genome must earn its place against the
prior). Too weak and every genome in the reference DB is "present"; too strong and the
low-abundance sub-species the pipeline exists to detect gets deleted. This picks the
default by measuring both failure modes.

The experiment, on the real 21-genome B. uniformis V4 reference set:

1. Extract the V4 amplicons and measure ``M`` once with the pipeline default (flat 0.005)
   — dev/error_rate_sensitivity.md already showed ``M`` is insensitive to the error model.
2. Build observed samples from a **subset-present** truth: only ``N_PRESENT`` of the 21
   genomes are in the sample, the rest are exactly 0. (The error-rate experiment's truth
   has all 21 present, so it cannot score presence/absence at all.) The confusable pair is
   always present with its low member at 3%.
3. Fit across a grid of (presence_prior, presence_temp), plus two baselines that call
   presence by thresholding an abundance instead: the ungated fit and the naive observed
   profile. Those are what the gate has to beat.
4. Score with Jaccard on the present sets (plus precision/recall/F1), L1 on the
   abundances, and whether the low-abundance member of the pair survived.

Needs docker (the mapseq biocontainer) and torch/pyro. ~10 min on a laptop.

    python dev/presence_prior_sweep.py [--work DIR] [--out CSV]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))
sys.path.insert(0, str(_REPO / "dev"))
import subspecies_infer as si  # noqa: E402
# Reuse the simulate -> mapseq -> M machinery rather than a second copy of it.
from error_rate_sensitivity import (DEFAULT_DB, N_OBS_READS, PAIR,  # noqa: E402
                                    build_M, build_observed, mapseq)

# The error model used to measure M. Any value in this range gives the same M
# (dev/error_rate_sensitivity.md), so use the pipeline default.
SIM_ERROR_RATE = "0.005"

N_PRESENT = 8            # genomes actually in the sample, of 21
LOW_ABUNDANCE = 0.03     # the hard member of the confusable pair
N_REPLICATES = 3         # different present subsets / seeds

PRIORS = [0.5, 0.1, 0.05, 0.01, 0.001]
# Temperature turns out to matter more than the prior probability: it sets how sharply
# the relaxed gate can approach 0/1, and a gate stuck near its initialisation can't be
# pulled down by any prior. Swept over a wider range than the prior for that reason.
TEMPS = [0.1, 0.5, 1.0, 2.0]
# Abundance cutoffs for the baselines that have no presence call of their own.
ABUNDANCE_CUTOFFS = [1e-4, 1e-3, 1e-2]


def draw_truth(genomes: list[str], rng) -> np.ndarray:
    """A composition over ``genomes`` with exact zeros for the absent ones.

    The confusable pair is always present — it is the case the pipeline exists for — with
    its low member pinned at LOW_ABUNDANCE so every replicate poses the same hard
    question. The remaining slots are filled at random and given log-uniform weights, so
    the present set spans a realistic dynamic range rather than being uniformly easy.
    """
    theta = np.zeros(len(genomes))
    idx = {g: i for i, g in enumerate(genomes)}
    others = [g for g in genomes if g not in PAIR]
    chosen = list(rng.choice(others, size=N_PRESENT - len(PAIR), replace=False))

    theta[idx[PAIR[0]]] = LOW_ABUNDANCE
    w = 10.0 ** rng.uniform(-1.5, 0, size=len(chosen) + 1)
    w = w / w.sum() * (1.0 - LOW_ABUNDANCE)
    for g, v in zip([PAIR[1]] + chosen, w):
        theta[idx[g]] = v
    return theta


def fit(M_np, T, ref_rel, total, *, use_presence, prior=None, temp=None,
        use_mismapping=True, seed=0):
    """Return (theta_estimate, presence_prob or None)."""
    import torch
    genomes = list(T.index)
    g_of_ref = np.array([genomes.index(si.genome_of_header(r)) for r in T.columns])
    theta_obs = np.zeros(len(genomes))
    np.add.at(theta_obs, g_of_ref, ref_rel)
    init = torch.tensor(np.clip(theta_obs, 1e-4, None), dtype=torch.float64)
    args = SimpleNamespace(
        mode="vi", lr=0.02, steps=1500, num_samples=300, warmup=0, progress=False,
        use_mismapping=use_mismapping, use_presence=use_presence, seed=seed,
        presence_prior=prior or si.DEFAULT_PRESENCE_PRIOR,
        presence_temp=temp or si.DEFAULT_PRESENCE_TEMP)
    _, point, diag, _ = si._fit(
        "vi", torch.tensor(M_np, dtype=torch.float64),
        torch.tensor(T.to_numpy(), dtype=torch.float64), 0.5, float(total),
        torch.tensor(ref_rel, dtype=torch.float64), "dirichlet_multinomial",
        init / init.sum(), args)
    p = diag.get("presence_prob")
    return point.numpy(), (np.array(p) if p is not None else None), theta_obs


def score(est, pred_present, theta_true, genomes) -> dict:
    """Presence metrics + the abundance error + the low-abundance-pair survival check."""
    m = si.presence_metrics(pred_present, theta_true > 0)
    return dict(**m,
                l1=float(np.abs(est - theta_true).sum()),
                pair_low_called=bool(pred_present[genomes.index(PAIR[0])]),
                pair_low_est=float(est[genomes.index(PAIR[0])]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="reference 16S fasta")
    ap.add_argument("--work", type=Path, default=_REPO / "dev" / "_presence_work")
    ap.add_argument("--out", type=Path, default=_REPO / "dev" / "presence_prior_sweep.csv")
    ap.add_argument("--keep", action="store_true", help="keep the work dir")
    a = ap.parse_args()

    work = a.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    si.stage_amplicons(SimpleNamespace(
        db_fasta=a.db, fwd_primer=si.DEFAULT_FWD_PRIMER, rev_primer=si.DEFAULT_REV_PRIMER,
        primer_mismatches=3, output_dir=work))
    fasta, tax = work / "amplicons.fasta", work / "amplicons.tax"
    recs = si.read_fasta(fasta)
    refseqs = [h for h, _ in recs]
    amps = [s for _, s in recs]
    T = pd.read_csv(work / "translation_table.csv", index_col=0)
    genomes = list(T.index)
    assert list(T.columns) == refseqs
    for g in PAIR:
        assert g in genomes, f"{g} not in the DB; --db is probably the wrong reference set"

    mapseq(fasta, fasta, tax, work / "cluster.mseq")   # cluster once, reused by every call
    M = build_M(SIM_ERROR_RATE, refseqs, amps, work, fasta, tax, seed=1)
    print(f"\n{len(refseqs)} amplicons / {len(genomes)} genomes; "
          f"M mean diagonal {np.diag(M).mean():.4f}; "
          f"{len(PRIORS) * len(TEMPS)} prior settings x {N_REPLICATES} replicates\n")

    rows = []
    for rep in range(N_REPLICATES):
        rng = np.random.default_rng(100 + rep)
        theta_true = draw_truth(genomes, rng)
        r_true = theta_true @ T.to_numpy()
        r_true = r_true / r_true.sum()
        ref_rel = build_observed(SIM_ERROR_RATE, refseqs, amps, r_true, work, fasta, tax,
                                 seed=200 + rep)
        n_obs = N_OBS_READS
        print(f"replicate {rep}: {int((theta_true > 0).sum())} genomes present")

        # Baselines: no gate, presence called by thresholding an abundance.
        ungated, _, theta_obs = fit(M, T, ref_rel, n_obs, use_presence=False, seed=rep)
        for name, est in (("ungated fit", ungated), ("naive observed", theta_obs)):
            for c in ABUNDANCE_CUTOFFS:
                rows.append(dict(replicate=rep, method=f"{name} (>{c:g})",
                                 presence_prior=np.nan, presence_temp=np.nan,
                                 **score(est, est > c, theta_true, genomes)))

        for prior in PRIORS:
            for temp in TEMPS:
                est, p, _ = fit(M, T, ref_rel, n_obs, use_presence=True, prior=prior,
                                temp=temp, seed=rep)
                rows.append(dict(replicate=rep, method="gate",
                                 presence_prior=prior, presence_temp=temp,
                                 **score(est, p >= si.PRESENCE_THRESHOLD, theta_true,
                                         genomes)))
                print(f"  prior={prior:<6g} temp={temp:<4g} "
                      f"jaccard={rows[-1]['jaccard']:.3f} l1={rows[-1]['l1']:.3f} "
                      f"pair_low={'kept' if rows[-1]['pair_low_called'] else 'LOST'}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    summary = (df.groupby(["method", "presence_prior", "presence_temp"], dropna=False)
                 .agg(jaccard=("jaccard", "mean"), precision=("precision", "mean"),
                      recall=("recall", "mean"), l1=("l1", "mean"),
                      n_pred=("n_pred", "mean"), pair_low=("pair_low_called", "mean"))
                 .sort_values("jaccard", ascending=False).reset_index())
    pd.set_option("display.width", 160)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {a.out}")
    # The best *gate* setting that never lost the low-abundance strain: losing it is
    # disqualifying however good the mean Jaccard looks.
    best = summary[(summary.method == "gate") & (summary.pair_low == 1.0)].head(1)
    if len(best):
        print(f"\nbest gate setting keeping the low-abundance strain in every replicate: "
              f"prior={best.presence_prior.iloc[0]:g} temp={best.presence_temp.iloc[0]:g} "
              f"(jaccard {best.jaccard.iloc[0]:.3f}, l1 {best.l1.iloc[0]:.4f})")
    if not a.keep:
        shutil.rmtree(work)


def demo() -> None:
    """Self-check for the parts that don't need docker: truth drawing and scoring."""
    genomes = [f"g{i}" for i in range(10)] + list(PAIR)
    theta = draw_truth(genomes, np.random.default_rng(0))
    assert np.isclose(theta.sum(), 1.0), theta
    assert int((theta > 0).sum()) == N_PRESENT, theta
    assert theta[genomes.index(PAIR[0])] == LOW_ABUNDANCE, theta
    assert theta[genomes.index(PAIR[1])] > 0, theta

    truth = theta > 0
    perfect = si.presence_metrics(truth, truth)
    assert perfect["jaccard"] == 1.0 and perfect["recall"] == 1.0, perfect
    # Dropping the low-abundance strain costs recall but leaves precision perfect —
    # exactly the failure an over-strong prior produces, and why recall is reported.
    dropped = truth.copy()
    dropped[genomes.index(PAIR[0])] = False
    m = si.presence_metrics(dropped, truth)
    assert m["precision"] == 1.0 and m["recall"] < 1.0 and m["jaccard"] < 1.0, m
    # Calling everything present: perfect recall, poor Jaccard — the under-regularised end.
    m = si.presence_metrics(np.ones_like(truth), truth)
    assert m["recall"] == 1.0 and m["jaccard"] == N_PRESENT / len(genomes), m
    print("demo presence_prior_sweep: OK")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
