#!/usr/bin/env python
"""Step 1 of the alignment-based mis-mapping plan: can pairwise amplicon distance alone
predict the mis-mapping matrix's diagonal?

``dev/error_rate_sensitivity.md`` found that ``mean diag(M) ~= 0.30`` regardless of the
simulated error rate, and argued the mechanism is reference *redundancy*: the references
that collide are the ones that are identical or near-identical over the amplicon. If that
is right, the tie-cluster kernel — a read from reference ``a`` is assigned uniformly over
the references at minimum distance from ``a`` — predicts ``diag(M)[a] = 1/|cluster(a)|``
with **no free parameters and no simulation**, and its mean should land on 0.30.

This script tests exactly that, and nothing else:

1. In-silico PCR over the reference DB (the pipeline's own ``stage_amplicons`` path).
2. All-pairs global edit distance with edlib.
3. Length spread, nearest-neighbour distance histogram, cluster sizes at d=0.
4. Predicted mean diagonal at tolerance tau = 0,1,2,3,5 vs the measured 0.30.
5. The two plan checks that are cheap here: reverse-complement collisions, and where the
   confusable B. uniformis pair sits.

    python dev/amplicon_distance_census.py [--db-fasta F] [--out CSV]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import edlib
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))
import subspecies_infer as si  # noqa: E402
from build_mismapping_align import pairwise_distances, tie_cluster_matrix  # noqa: E402

DEFAULT_DB = Path.home() / ("Documents/synthetic-metagenomic-benchmark-pipeline_runs/"
                            "subspecies_Buniformis_amplicon_v4/mapseq_db/mapseq_db.fasta")
# The measured value this prediction has to hit (error_rate_sensitivity.md, "mean diag").
MEASURED_MEAN_DIAG = 0.30
TAUS = (0, 1, 2, 3, 5)


def tie_cluster_diagonal(d: np.ndarray, tau: int) -> np.ndarray:
    """``diag(M)`` under the estimator's tie-cluster kernel — the quantity this census
    exists to compare against the measured mean diagonal."""
    return np.diag(tie_cluster_matrix(d, tau))


def revcomp_check(seqs: list[str], d: np.ndarray) -> tuple[int, int]:
    """Min distance to any *other* amplicon, forward vs reverse-complemented.

    ``extract_v4`` already normalises orientation, so the reverse-complement distances
    should be far larger; if they aren't, the census (and the estimator) must align both
    strands."""
    off = d + np.diag(np.full(len(d), 1 << 20))
    fwd = int(off.min())
    rc = min(edlib.align(seqs[i], si.revcomp(seqs[j]), mode="NW", task="distance")[
        "editDistance"] for i in range(len(seqs)) for j in range(len(seqs)) if i != j)
    return fwd, rc


def _selfcheck() -> None:
    """The clustering is the only non-trivial logic here; check it on known input."""
    seqs = ["ACGTACGTAC", "ACGTACGTAC", "ACGTACGTAC",   # 3 identical
            "ACGTACGTAG",                                # 1 base from the trio
            "TTTTTTTTTT"]                                # isolated
    d = pairwise_distances(seqs)
    assert d[0, 1] == 0 and d[0, 3] == 1 and d[0, 4] == 8, d[0]
    diag0 = tie_cluster_diagonal(d, 0)
    assert np.allclose(diag0[:3], 1 / 3) and diag0[3] == 1.0 and diag0[4] == 1.0, diag0
    diag1 = tie_cluster_diagonal(d, 1)
    assert np.allclose(diag1[:4], 1 / 4) and diag1[4] == 1.0, diag1
    print("selfcheck OK: identical trio -> 1/3 diagonals, isolated -> 1.0, tau widens")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-fasta", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=_REPO / "dev" / "amplicon_distance_census.csv")
    ap.add_argument("--fwd-primer", default=si.DEFAULT_FWD_PRIMER)
    ap.add_argument("--rev-primer", default=si.DEFAULT_REV_PRIMER)
    ap.add_argument("--primer-mismatches", type=int, default=3)
    a = ap.parse_args()

    _selfcheck()

    records = si.read_fasta(a.db_fasta)
    refseqs, seqs, genomes = [], [], []
    for header, seq in records:
        amp = si.extract_v4(seq, a.fwd_primer, a.rev_primer, a.primer_mismatches)
        if amp:
            refseqs.append(header)
            seqs.append(amp)
            genomes.append(si.genome_of_header(header))
    print(f"\n{len(refseqs)}/{len(records)} entries amplifiable, "
          f"{len(set(genomes))} genomes")

    lens = np.array([len(s) for s in seqs])
    print(f"amplicon length: min {lens.min()} median {int(np.median(lens))} "
          f"max {lens.max()} (spread {lens.max() - lens.min()})")

    d = pairwise_distances(seqs)

    off = d + np.diag(np.full(len(d), 1 << 20))
    nn = off.min(axis=1)
    print("\nnearest *other* reference, edit distance:")
    for lo, hi in ((0, 0), (1, 1), (2, 2), (3, 5), (6, 10), (11, 10**9)):
        n = int(((nn >= lo) & (nn <= hi)).sum())
        label = f"{lo}" if lo == hi else (f">{lo - 1}" if hi > 10**8 else f"{lo}-{hi}")
        print(f"  {label:>5}: {n:3d} refs  {'#' * int(50 * n / len(nn))}")

    dup = Counter(seqs)
    sizes = Counter(dup.values())
    print("\nexact-duplicate cluster sizes (d = 0):")
    for size in sorted(sizes):
        print(f"  {size} member(s): {sizes[size]:3d} cluster(s) "
              f"({size * sizes[size]:3d} refs)")

    print(f"\npredicted mean diag(M) vs measured {MEASURED_MEAN_DIAG:.2f}:")
    rows = []
    for tau in TAUS:
        diag = tie_cluster_diagonal(d, tau)
        rows.append({"tau": tau, "mean_diag": diag.mean(), "median_diag": np.median(diag),
                     "frac_diag_1": float((diag == 1.0).mean()),
                     "mean_cluster_size": float((1 / diag).mean())})
        print(f"  tau={tau}: mean {diag.mean():.3f}  median {np.median(diag):.3f}  "
              f"unambiguous {100 * (diag == 1.0).mean():.0f}%  "
              f"mean cluster {(1 / diag).mean():.2f}  "
              f"delta {diag.mean() - MEASURED_MEAN_DIAG:+.3f}")

    fwd, rc = revcomp_check(seqs, d)
    print(f"\norientation: min forward inter-ref distance {fwd}, "
          f"min reverse-complement distance {rc} "
          f"({'ok, already normalised' if rc > fwd else 'WARNING: align both strands'})")

    # Where does the confusable pair sit? Genome-level, which is what inference cares about.
    gser = pd.Series(genomes)
    cross = {}
    for i, gi in enumerate(genomes):
        for j in np.flatnonzero(d[i] == 0):
            if genomes[j] != gi:
                cross.setdefault(gi, Counter())[genomes[j]] += 1
    print("\ngenomes sharing an identical amplicon with another genome:")
    if not cross:
        print("  (none)")
    for g, partners in sorted(cross.items()):
        print(f"  {g}: {dict(partners)}")

    pd.DataFrame(rows).to_csv(a.out, index=False)
    per_ref = pd.DataFrame({"refseq": refseqs, "genome": genomes, "amplicon_len": lens,
                            "nn_distance": nn,
                            "cluster_size_tau0": (1 / tie_cluster_diagonal(d, 0)).astype(int)})
    per_ref.to_csv(a.out.with_name(a.out.stem + "_per_ref.csv"), index=False)
    print(f"\n-> {a.out} and {a.out.with_name(a.out.stem + '_per_ref.csv')}")
    print(f"   ({len(gser.unique())} genomes, {len(refseqs)} refs)")


if __name__ == "__main__":
    main()
