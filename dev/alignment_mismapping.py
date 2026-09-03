#!/usr/bin/env python
"""Step 3: is the alignment-built ``M`` equivalent to the simulate-and-map one?

``dev/amplicon_distance_census.md`` showed the tie-cluster kernel reproduces the measured
*mean diagonal* of ``M``. That is one scalar. This asks the real question: is the whole
matrix — and the composition it produces — indistinguishable from the measured one?

The criterion is the measurement's own precision, not zero. Re-simulating the *same* error
model with a different seed moves ``M`` by some ``||.||_F``; that is the noise floor, and
``M_align`` passes if it sits at or below it (the method
``dev/error_rate_sensitivity.py`` established).

Reported per candidate:

- ``||M - M_sim||_F`` against the reseed floor, and the mean diagonal.
- Per-row total variation (median / max) — *where* it differs, not just how much.
- Off-diagonal support Jaccard at 1e-3: which pairs are confusable **at all**. Getting the
  support right and the mass wrong is a knob problem; getting the support wrong is the
  wrong model.
- Spearman correlation of the off-diagonal entries.
- Downstream: composition L1 vs truth, and the error on each strain of the confusable
  B. uniformis pair, fitted from the same observed reads.

Candidates: the alignment kernel at tau = 0, 1, 2; the trained error model (a *different*
measurement of the same thing); the reseed (the floor); and a shuffled-distance control
that must fail everything, so a loose criterion can't pass for a result.

Needs docker (the mapseq biocontainer) and torch/pyro. ~10 min on a laptop.

    python dev/alignment_mismapping.py [--work DIR] [--out CSV]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))
sys.path.insert(0, str(_REPO / "vendor" / "skiver" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import subspecies_infer as si  # noqa: E402
import build_mismapping_align as bma  # noqa: E402
# The mapseq-in-docker wrapper, the read simulators and the pyro fit already exist there.
import error_rate_sensitivity as ers  # noqa: E402

# The pipeline's own default error model (nextflow.config flat_sub_rate), used as the
# measurement M_align has to match.
SIM_RATE = "0.005"
SUPPORT_EPS = 1e-3
TAUS = (0, 1, 2)


def compare(M: np.ndarray, ref: np.ndarray) -> dict:
    """Matrix-space diagnostics of ``M`` against the measured ``ref``."""
    n = len(M)
    offdiag = ~np.eye(n, dtype=bool)
    tv = 0.5 * np.abs(M - ref).sum(axis=1)
    sup_m, sup_r = (M > SUPPORT_EPS) & offdiag, (ref > SUPPORT_EPS) & offdiag
    union = (sup_m | sup_r).sum()
    rho = spearmanr(M[offdiag], ref[offdiag]).statistic
    return {
        "frobenius": float(np.linalg.norm(M - ref)),
        "mean_diag": float(np.diag(M).mean()),
        "tv_median": float(np.median(tv)),
        "tv_max": float(tv.max()),
        "tv_worst_row": int(np.argmax(tv)),
        "support_jaccard": float((sup_m & sup_r).sum() / union) if union else 1.0,
        "spearman_offdiag": float(rho),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=ers.DEFAULT_DB)
    ap.add_argument("--work", type=Path, default=_REPO / "dev" / "_alignment_work")
    ap.add_argument("--out", type=Path, default=_REPO / "dev" / "alignment_mismapping.csv")
    ap.add_argument("--keep", action="store_true")
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
    refseqs, amps = [h for h, _ in recs], [s for _, s in recs]
    T = pd.read_csv(work / "translation_table.csv", index_col=0)
    genomes = list(T.index)
    assert list(T.columns) == refseqs

    theta_true = np.full(len(genomes),
                         (1.0 - sum(ers.TRUTH_PAIR.values())) / (len(genomes) - 2))
    for g, v in ers.TRUTH_PAIR.items():
        theta_true[genomes.index(g)] = v
    r_true = theta_true @ T.to_numpy()
    r_true = r_true / r_true.sum()

    ers.mapseq(fasta, fasta, tax, work / "cluster.mseq")   # cluster once, reused by all
    print(f"\n{len(refseqs)} amplicons / {len(genomes)} genomes\n")

    # ── the measurement, and its own precision ───────────────────────────────
    t0 = time.perf_counter()
    M_sim = ers.build_M(SIM_RATE, refseqs, amps, work, fasta, tax, seed=1)
    t_sim = time.perf_counter() - t0
    Ms = {
        f"reseed (flat {SIM_RATE}, seed 99)":
            ers.build_M(SIM_RATE, refseqs, amps, work, fasta, tax, seed=99,
                        tag=f"{SIM_RATE}_reseed"),
        "trained error model":
            ers.build_M("trained", refseqs, amps, work, fasta, tax, seed=1),
    }

    # ── the candidates ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    d = bma.pairwise_distances(amps)
    Ms["alignment tau=0"] = bma.tie_cluster_matrix(d, 0)
    t_align = time.perf_counter() - t0
    for tau in TAUS[1:]:
        Ms[f"alignment tau={tau}"] = bma.tie_cluster_matrix(d, tau)

    # Negative control: the same kernel on shuffled distances. Must fail everything.
    rng = np.random.default_rng(0)
    iu = np.triu_indices(len(d), 1)
    d_shuf = np.zeros_like(d)
    d_shuf[iu] = rng.permutation(d[iu])
    d_shuf = d_shuf + d_shuf.T
    Ms["shuffled control"] = bma.tie_cluster_matrix(d_shuf, 0)

    print(f"cost: simulate+map {t_sim:.1f}s   alignment {t_align:.2f}s "
          f"({t_sim / t_align:.0f}x)\n")

    diag = {name: compare(M, M_sim) for name, M in Ms.items()}
    for name, m in diag.items():
        print(f"  {name:34s} F={m['frobenius']:.3f}  diag={m['mean_diag']:.4f}  "
              f"TV med={m['tv_median']:.3f} max={m['tv_max']:.3f}  "
              f"support J={m['support_jaccard']:.3f}  rho={m['spearman_offdiag']:.3f}")
    floor = diag[f"reseed (flat {SIM_RATE}, seed 99)"]["frobenius"]
    # The floor compares two *noisy* measurements, so it carries sampling noise twice:
    # floor^2 ~ 2*noise^2. A noiseless estimator with no systematic difference would
    # therefore score floor/sqrt(2), not the floor itself — the sharper target. What is
    # left over is the systematic part: bias^2 ~ F^2 - floor^2/2.
    for m in diag.values():
        m["implied_bias"] = float(np.sqrt(max(m["frobenius"] ** 2 - floor ** 2 / 2, 0.0)))
    print(f"\nnoise floor (reseed): ||.||_F = {floor:.3f}; "
          f"noiseless-and-unbiased target = {floor / np.sqrt(2):.3f}")
    for name, m in diag.items():
        print(f"  {name:34s} implied systematic difference {m['implied_bias']:.3f}")
    print()

    # ── downstream: does the composition change? ─────────────────────────────
    rows = []
    for gen in ["trained", "0.01"]:
        ref_rel = ers.build_observed(gen, refseqs, amps, r_true, work, fasta, tax, seed=2)
        theta_obs, _ = ers.fit(np.eye(len(refseqs)), T, genomes, ref_rel, ers.N_OBS_READS,
                               use_mismapping=False)
        rows.append(dict(generator=gen, candidate="(none: naive observed)",
                         l1=float(np.abs(theta_obs - theta_true).sum()),
                         **{f"err_{g}": float(theta_obs[genomes.index(g)]
                                              - theta_true[genomes.index(g)])
                            for g in ers.PAIR}))
        for name, M in {f"simulate+map (flat {SIM_RATE})": M_sim, **Ms}.items():
            _, est = ers.fit(M, T, genomes, ref_rel, ers.N_OBS_READS)
            rows.append(dict(
                generator=gen, candidate=name,
                l1=float(np.abs(est - theta_true).sum()),
                **{f"err_{g}": float(est[genomes.index(g)] - theta_true[genomes.index(g)])
                   for g in ers.PAIR},
                **diag.get(name, {})))
            print(f"  fit gen={gen:8s} cand={name:34s} L1={rows[-1]['l1']:.4f}")

    df = pd.DataFrame(rows)
    df["noise_floor_frobenius"] = floor
    df["seconds_simulate_map"] = t_sim
    df["seconds_alignment"] = t_align
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.drop(columns=[c for c in ("tv_worst_row", "noise_floor_frobenius",
                                              "seconds_simulate_map", "seconds_alignment")
                                  if c in df]).to_string(index=False,
                                                         float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {a.out}")
    if not a.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
