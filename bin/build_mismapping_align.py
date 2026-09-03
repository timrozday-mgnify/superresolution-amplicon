#!/usr/bin/env python
"""Build the mis-mapping matrix ``M`` from reference-to-reference alignment alone.

The cheap alternative to the simulate-reads-and-map measurement: instead of sampling
errored reads from every reference and running them through mapseq, align the reference
amplicons to each other and read ``M`` straight off the distances.

The kernel is the *tie cluster*: a read from reference ``a`` is assigned uniformly over
the references within ``--tau`` edit operations of ``a`` (``a`` itself included, at
distance 0), so

    M[a, j] = 1 / |cluster(a)|   if d(a, j) <= tau, else 0

A reference with no neighbour inside ``tau`` gets an identity row — ``r_true`` passes
through uncorrected, the same convention ``subspecies_infer.build_mismapping`` uses for a
reference whose simulated reads all failed to map.

Why this is defensible, and where it stops being so: on the 21-genome B. uniformis V4 set
``tau=0`` reproduces the measured mean diagonal to within the measurement's own seed noise
(0.3086 predicted vs 0.3064/0.3093 measured), because 73 of 81 references are *exact*
duplicates over the amplicon — confusion there is redundancy, not sequencing error. See
dev/amplicon_distance_census.md and dev/error_rate_sensitivity.md. A reference set whose
members differ by one or two bases is the case this kernel is expected to miss; the
printed nearest-neighbour histogram is there so you can see which kind of set you have.

Output is the same CSV ``infer_composition.py --build-mismapping`` writes, so it drops
straight into ``--mismapping-matrix`` / ``--mismapping_matrix``.

Run:
    build_mismapping_align.py --amplicons out/amplicons.fasta -o mismapping_matrix.csv
    build_mismapping_align.py --demo   # self-check
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import edlib
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)

log = logging.getLogger("build_mismapping_align")


def pairwise_distances(seqs: list[str]) -> np.ndarray:
    """Symmetric all-pairs global (Needleman-Wunsch) edit distance, via edlib.

    ponytail: O(n^2) alignments, ~1 us each for a 250bp amplicon — fine for the
    per-sample reference sets this pipeline builds (tens to thousands of amplicons).
    Past ~10k references, group exact duplicates by hash first (at tau=0 that is the whole
    answer and needs no alignment) and align only the cluster representatives.
    Ambiguity codes are plain mismatches here; edlib's `additionalEqualities` would make
    N match everything, if a DB ever turns out to need it.
    """
    n = len(seqs)
    d = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = edlib.align(
                seqs[i], seqs[j], mode="NW", task="distance")["editDistance"]
    return d


def tie_cluster_matrix(d: np.ndarray, tau: int = 0) -> np.ndarray:
    """Row-stochastic ``M`` from a distance matrix: uniform over each tie cluster.

    Self-distance is 0, so every cluster contains its own reference and no row is ever
    empty; an isolated reference gets the identity row for free.
    """
    member = d <= tau
    return member / member.sum(axis=1, keepdims=True)


def build(amplicons: Path, tau: int = 0) -> pd.DataFrame:
    """``M`` as a labelled frame, indexed and columned by the amplicon fasta headers."""
    records = si.read_fasta(amplicons)
    if not records:
        raise SystemExit(f"{amplicons} contains no sequences")
    refseqs = [h for h, _ in records]
    if len(set(refseqs)) != len(refseqs):
        raise SystemExit(f"{amplicons} has duplicate reference ids")
    d = pairwise_distances([s for _, s in records])
    return pd.DataFrame(tie_cluster_matrix(d, tau), index=refseqs, columns=refseqs), d


def summarise(M: pd.DataFrame, d: np.ndarray) -> None:
    """Print the two things that say whether the tie-cluster assumption is safe here."""
    diag = np.diag(M.to_numpy())
    off = d + np.diag(np.full(len(d), 1 << 20))
    nn = off.min(axis=1)
    print(f"nearest other reference: "
          + "  ".join(f"d={lo if lo == hi else f'{lo}-{hi}'}: {int(((nn >= lo) & (nn <= hi)).sum())}"
                      for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 10), (11, 1 << 20))))
    print(f"mean diag(M) {diag.mean():.4f}  median {np.median(diag):.4f}  "
          f"unambiguous {100 * (diag == 1.0).mean():.0f}%  "
          f"mean cluster {(1 / diag).mean():.2f}")


def demo() -> None:
    """Self-check the distance and the kernel on sequences with a known answer."""
    trio = "ACGTACGTACGTACGTACGT"
    seqs = [trio, trio, trio,                  # an identical trio -> 1/3 rows
            trio[:-1] + "A",                   # one base off the trio
            "T" * 20]                          # isolated
    d = pairwise_distances(seqs)
    assert d[0, 1] == 0 and d[0, 3] == 1 and d[0, 4] > 10, d[0]
    assert (d == d.T).all() and (np.diag(d) == 0).all()

    M = tie_cluster_matrix(d, tau=0)
    assert np.allclose(M.sum(axis=1), 1.0), M.sum(axis=1)          # row-stochastic
    assert np.allclose(np.diag(M)[:3], 1 / 3), np.diag(M)          # trio splits 3 ways
    assert np.allclose(M[3], np.eye(5)[3]) and np.allclose(M[4], np.eye(5)[4])  # identity
    M1 = tie_cluster_matrix(d, tau=1)
    assert np.allclose(np.diag(M1)[:4], 1 / 4), np.diag(M1)        # tau pulls #3 in
    assert np.allclose(M1[4], np.eye(5)[4])                        # still isolated

    # End to end through a fasta, and the invariants infer_composition.py checks on load.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "amplicons.fasta"
        fa.write_text("".join(f">ref|{i}|x\n{s}\n" for i, s in enumerate(seqs)))
        frame, _ = build(fa, tau=0)
        frame.to_csv(Path(td) / "M.csv")
        back = pd.read_csv(Path(td) / "M.csv", index_col=0)
        assert list(back.index) == list(back.columns) == [f"ref|{i}|x" for i in range(5)]
        assert np.isfinite(back.to_numpy()).all() and (back.to_numpy() >= 0).all()
        assert np.allclose(back.to_numpy().sum(axis=1), 1.0)
    print("demo OK: identical trio -> 1/3 rows, isolated -> identity, tau widens clusters, "
          "CSV round-trips row-stochastic")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicons", type=Path,
                    help="amplicons.fasta from `subspecies_infer.py amplicons`")
    ap.add_argument("--tau", type=int, default=0,
                    help="cluster references within this edit distance (default 0: exact "
                         "duplicates only, which is what the B. uniformis set wanted)")
    ap.add_argument("-o", "--output", type=Path, help="output mismapping_matrix.csv")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if a.demo:
        return demo()
    for req in ("amplicons", "output"):
        if getattr(a, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (unless --demo)")
    M, d = build(a.amplicons, a.tau)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    M.to_csv(a.output)
    summarise(M, d)
    print(f"build_mismapping_align: {len(M)} references, tau={a.tau} -> {a.output}")


if __name__ == "__main__":
    main()
