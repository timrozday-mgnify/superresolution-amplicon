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

_BASES = "ACGT"


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


def windowed_matrix(seqs: list[str], read_len: int, tau: int = 0,
                    stride: int = 1) -> np.ndarray:
    """``M`` for reads *shorter* than the amplicon.

    A read sees only a window of its source amplicon, so two references that differ
    outside that window are indistinguishable to the mapper — confusion the whole-amplicon
    distance cannot see, and would silently understate. This mirrors what the simulate path
    does with ``--read-len``: draw every length-``read_len`` window of the source reference
    (uniformly, as ``draw_fragment`` does), score it against every reference by *infix*
    alignment (edlib ``HW``: the best placement of the read anywhere in the reference —
    which is the question a mapper asks), cluster the winners, and average over windows.

    A window is a substring of its own source, so its distance to that source is 0 and the
    cluster is never empty. Reduces to the whole-amplicon kernel when ``read_len`` covers
    the amplicon.

    ponytail: n_refs^2 * n_windows infix alignments — seconds for a per-sample reference
    set (81 refs x 104 windows x 81 targets ~ 7 s). ``stride`` subsamples the windows if a
    much larger set ever needs it; the windows are highly redundant, so it costs little.
    """
    n = len(seqs)
    M = np.zeros((n, n), dtype=np.float64)
    for a, seq in enumerate(seqs):
        starts = range(0, max(len(seq) - read_len, 0) + 1, stride)
        n_win = 0
        for p in starts:
            frag = seq[p:p + read_len]
            d = np.fromiter(
                (edlib.align(frag, t, mode="HW", task="distance")["editDistance"]
                 for t in seqs), dtype=np.int32, count=n)
            member = d <= tau
            M[a] += member / member.sum()
            n_win += 1
        M[a] /= n_win
    return M


def build(amplicons: Path, tau: int = 0, read_len: int | None = None,
          stride: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """``M`` as a labelled frame, indexed and columned by the amplicon fasta headers.

    Returns the whole-amplicon distance matrix alongside it, for the summary — under
    ``read_len`` it is a description of the reference set, not the matrix's own input.
    """
    records = si.read_fasta(amplicons)
    if not records:
        raise SystemExit(f"{amplicons} contains no sequences")
    refseqs = [h for h, _ in records]
    if len(set(refseqs)) != len(refseqs):
        raise SystemExit(f"{amplicons} has duplicate reference ids")
    seqs = [s for _, s in records]
    d = pairwise_distances(seqs)
    M = (tie_cluster_matrix(d, tau) if read_len is None
         else windowed_matrix(seqs, read_len, tau, stride))
    return pd.DataFrame(M, index=refseqs, columns=refseqs), d


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
    # Short reads: two references identical over most of their length and differing only
    # at one end. Whole-amplicon distance calls them distinct; a read that never spans the
    # difference cannot tell them apart, and M must say so.
    # Non-repetitive, or infix alignment matches everywhere and the example says nothing.
    rng = np.random.default_rng(0)
    draw = lambda k: "".join(rng.choice(list(_BASES), size=k))
    shared = draw(60)
    pair = [shared + draw(20), shared + draw(20)]   # identical but for the last 20 bases
    d2 = pairwise_distances(pair)
    assert d2[0, 1] > 0
    assert np.allclose(tie_cluster_matrix(d2, tau=0), np.eye(2))       # "never confused"

    W = windowed_matrix(pair, read_len=20)
    assert np.allclose(W.sum(axis=1), 1.0), W
    # 41 of the 61 windows sit entirely inside the shared prefix and cannot separate the
    # two references; the other 20 overlap the difference and resolve it.
    assert np.isclose(W[0, 0], (41 * 0.5 + 20) / 61), W
    assert W[0, 1] > 0.3 and W[1, 0] > 0.3, W
    # A read as long as the amplicon sees everything, so the window kernel collapses back.
    assert np.allclose(windowed_matrix(pair, read_len=len(pair[0])), np.eye(2))
    # Truly identical references stay 50/50 at every read length.
    assert np.allclose(windowed_matrix([shared, shared], read_len=10), 0.5)
    # Striding subsamples the same windows, so it must not move the answer much.
    assert abs(windowed_matrix(pair, read_len=20, stride=5)[0, 0] - W[0, 0]) < 0.05

    print("demo OK: identical trio -> 1/3 rows, isolated -> identity, tau widens clusters, "
          "CSV round-trips row-stochastic; short reads see confusion the whole-amplicon "
          "distance misses, and collapse back to it at full length")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicons", type=Path,
                    help="amplicons.fasta from `subspecies_infer.py amplicons`")
    ap.add_argument("--tau", type=int, default=0,
                    help="cluster references within this edit distance (default 0: exact "
                         "duplicates only, which is what the B. uniformis set wanted)")
    ap.add_argument("--read-len", type=int, default=None,
                    help="reads are this long, i.e. shorter than the amplicon (unmerged / "
                         "short reads). Default: reads span the whole amplicon.")
    ap.add_argument("--window-stride", type=int, default=1,
                    help="--read-len: sample every Nth read window (default 1: all)")
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
    if a.read_len is not None and a.read_len < 1:
        ap.error("--read-len must be positive")
    M, d = build(a.amplicons, a.tau, a.read_len, a.window_stride)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    M.to_csv(a.output)
    summarise(M, d)
    scope = "whole amplicon" if a.read_len is None else f"read-len {a.read_len}"
    print(f"build_mismapping_align: {len(M)} references, tau={a.tau}, {scope} "
          f"-> {a.output}")


if __name__ == "__main__":
    main()
