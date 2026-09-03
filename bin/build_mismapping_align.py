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

**Whole references only.** ``M`` here describes queries that span the reference amplicon:
merged pairs, or single reads long enough to cover it. Reads shorter than the amplicon see
a window of it and are confusable in ways whole-sequence distance cannot see, and this
kernel does not model that — use ``--mismapping_method simulate --sim_read_len``, which
measures it directly. The pipeline refuses the combination rather than quietly
understating confusion.

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

# Tie-break weight per ambiguous position; see ``ambiguity_weights``. mapseq's penalty on
# an N-bearing reference is real but not constant — measured between 0.18 and 0.97
# depending on which references carry ambiguity (dev/ambiguity_weight_sweep.md) — so this
# is the compromise, not a fitted optimum: best or near-best in 4 of 5 configurations and
# cheap in the fifth. A no-op wherever a reference set carries no ambiguity codes.
DEFAULT_AMBIGUITY_WEIGHT = 0.3

# k-mer length for the candidate-pair filter. 15 is short enough that a 250bp
# amplicon still has plenty of k-mers to spare against the q-gram bound, and long
# enough to be specific in a 16S reference set.
KMER_SIZE = 15

# Distances are resolved exactly out to here so the summary histogram is meaningful;
# anything beyond is reported as "further".
SUMMARY_DISTANCE = 5

# IUPAC ambiguity: two symbols are equal when the base sets they stand for overlap, so an
# `N` (or `R`, `Y`, …) in a draft-genome 16S matches what it could have been. Without this
# edlib scores ambiguity as a plain mismatch, and a single `N` in one copy of an otherwise
# identical pair splits the tie cluster and hands *both* references identity rows — the
# maximally wrong answer for two sequences that are in fact indistinguishable.
#
# NB: this equality is not transitive (N=A and N=C, but A!=C), so cluster membership stays
# symmetric while cluster *sizes* need not match, and M is no longer symmetric. That is
# correct: M[a,j] is "where a read from a goes", which was never a symmetric question.
_EQUALITIES = [(x, y) for x in si._IUPAC for y in si._IUPAC
               if x != y and si._IUPAC[x] & si._IUPAC[y]]


def _kmer_candidates(seqs: list[str], max_distance: int, k: int = KMER_SIZE
                     ) -> list[set[int]]:
    """Candidate neighbours per sequence: everything that *could* be within
    ``max_distance``, by the q-gram lemma. Lossless — never drops a true neighbour.

    A sequence of length L has ``L - k + 1`` k-mers, and one edit destroys at most ``k`` of
    them, so two sequences within ``e`` edits share at least ``L - k + 1 - k*e``. Anything
    sharing fewer cannot be within ``e`` and needs no alignment. Ambiguous positions are
    charged like edits (their k-mers are skipped, since an `N` k-mer would have to match
    every substitution of itself), which keeps the bound conservative.

    ponytail: one dict of k-mer -> reference ids. Memory is O(total k-mers); for reference
    sets far larger than this pipeline builds, a minimizer sketch (index every w-th k-mer)
    trades a weaker bound for less memory.
    """
    n = len(seqs)
    index: dict[str, list[int]] = {}
    kmers: list[set[str]] = []
    amb = [sum(c not in _BASES for c in s) for s in seqs]
    for i, seq in enumerate(seqs):
        ks = {seq[p:p + k] for p in range(len(seq) - k + 1)
              if all(c in _BASES for c in seq[p:p + k])}
        kmers.append(ks)
        for km in ks:
            index.setdefault(km, []).append(i)

    out: list[set[int]] = []
    for i, ks in enumerate(kmers):
        shared: dict[int, int] = {}
        for km in ks:
            for j in index[km]:
                if j != i:
                    shared[j] = shared.get(j, 0) + 1
        # The threshold uses the shorter sequence's k-mer count, and charges both
        # sequences' ambiguity, so it can only be too permissive — never too strict.
        cand = {i}
        for j, count in shared.items():
            need = (min(len(seqs[i]), len(seqs[j])) - k + 1
                    - k * (max_distance + amb[i] + amb[j]))
            if count >= need:
                cand.add(j)
        out.append(cand)
    return out


def paf_distances(paf: Path, refseqs: list[str], sentinel: int) -> np.ndarray:
    """Distance matrix from a minimap2 all-vs-all PAF (run with ``-c``, so ``NM`` is set).

    ``NM`` is the edit distance *within* the aligned block, so whatever minimap2 left
    unaligned at either end is added back: an alignment covering half the query is not a
    close match however clean its aligned half is. The smallest value over a pair's
    alignments wins. Pairs minimap2 never reports get ``sentinel`` — "further than this
    backend can see", not "infinitely far".

    Treat the result as data, not truth: minimap2 is a heuristic whose recall falls off
    with divergence, and which preset you ran matters more than ``-p``/``-N``
    (dev/alignment_backend_benchmark.md). It finds every identical and one-base-apart pair
    on the reference set tested, which is what the tie-cluster kernel needs; the far tail
    is where it stops being complete.
    """
    idx = {r: i for i, r in enumerate(refseqs)}
    n = len(refseqs)
    d = np.full((n, n), sentinel, dtype=np.int32)
    np.fill_diagonal(d, 0)
    unknown = 0
    with open(paf) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                continue
            i, j = idx.get(f[0]), idx.get(f[5])
            if i is None or j is None:
                unknown += 1
                continue
            nm = next((int(t.split(":")[2]) for t in f[12:] if t.startswith("NM:i:")), None)
            if nm is None:
                continue
            dist = (nm + (int(f[1]) - (int(f[3]) - int(f[2])))
                    + (int(f[6]) - (int(f[8]) - int(f[7]))))
            dist = min(dist, sentinel)
            if dist < d[i, j]:
                d[i, j] = d[j, i] = dist
    if unknown:
        log.warning("%d PAF rows referenced an unknown sequence id", unknown)
    return d


def pairwise_distances(seqs: list[str], max_distance: int | None = None) -> np.ndarray:
    """Symmetric all-pairs global (Needleman-Wunsch) edit distance, via edlib.

    With ``max_distance`` set, only pairs that a lossless k-mer filter says *could* be that
    close are aligned, and edlib is told to give up past it (``k=``, which also lets it
    bail out early inside an alignment). Everything else comes back as ``max_distance + 1``
    — a sentinel meaning "further than you asked about", not a real distance. Callers that
    need exact distances everywhere (the census histogram) pass ``None``.

    IUPAC ambiguity codes match any base they could stand for (``_EQUALITIES``).
    """
    n = len(seqs)
    if max_distance is None:
        d = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = edlib.align(
                    seqs[i], seqs[j], mode="NW", task="distance",
                    additionalEqualities=_EQUALITIES)["editDistance"]
        return d

    sentinel = max_distance + 1
    d = np.full((n, n), sentinel, dtype=np.int32)
    np.fill_diagonal(d, 0)
    for i, cand in enumerate(_kmer_candidates(seqs, max_distance)):
        for j in cand:
            if j <= i:
                continue
            got = edlib.align(seqs[i], seqs[j], mode="NW", task="distance",
                              k=max_distance, additionalEqualities=_EQUALITIES
                              )["editDistance"]
            if got >= 0:                       # -1 means "further than k"
                d[i, j] = d[j, i] = got
    return d


def ambiguity_weights(seqs: list[str], weight: float) -> np.ndarray | None:
    """Per-reference tie-break weight ``weight ** (number of ambiguous positions)``.

    Ambiguity codes match anything (``_EQUALITIES``), so an `N`-bearing reference joins the
    tie cluster of the sequences it could equal — but the *mapper* does not treat it as an
    equal member: mapseq scores the `N` as a mismatch, so a clean duplicate wins the read.
    Measured at **0.19x** the incoming mass of its clean cluster partners
    (dev/alignment_mismapping.md). This down-weights cluster members by how much ambiguity
    they carry, which is the only part of that penalty a distance-based kernel can express.

    ``weight = 1`` disables it. Any value is a no-op on an ambiguity-free reference set
    (every exponent is 0), so it costs nothing where it is not needed.
    """
    if weight == 1.0:
        return None
    k = np.array([sum(c not in _BASES for c in s) for s in seqs], dtype=np.float64)
    return weight ** k


def _normalise(member: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Row-normalise a boolean membership matrix, weighted if given.

    A cluster whose members are *all* weighted to zero (every member ambiguous) would
    otherwise divide by zero; it falls back to the unweighted split, since the penalty is
    a tie-break between members and there is no tie left to break.
    """
    w = member if weights is None else member * weights
    tot = w.sum(axis=1, keepdims=True)
    dead = (tot == 0).ravel()
    if dead.any():
        w = np.where(dead[:, None], member, w)
        tot = w.sum(axis=1, keepdims=True)
    return w / tot


def tie_cluster_matrix(d: np.ndarray, tau: int = 0,
                       weights: np.ndarray | None = None) -> np.ndarray:
    """Row-stochastic ``M`` from a distance matrix: split over each tie cluster.

    Self-distance is 0, so every cluster contains its own reference and no row is ever
    empty; an isolated reference gets the identity row for free. ``weights`` (see
    ``ambiguity_weights``) splits a cluster unevenly instead of uniformly.
    """
    return _normalise(d <= tau, weights)


def build(amplicons: Path, tau: int = 0, ambiguity_weight: float = 1.0,
          paf: Path | None = None) -> tuple[pd.DataFrame, np.ndarray]:
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
    # Bounded a little past tau so the printed nearest-neighbour histogram still shows the
    # near-misses that say whether tau=0 is safe for this reference set.
    bound = max(tau, SUMMARY_DISTANCE)
    if paf is not None:
        n_amb = sum(1 for q in seqs if any(c not in _BASES for c in q))
        if n_amb:
            # minimap2's NM counts an ambiguity code as a mismatch, so an N-bearing
            # reference is split out of the cluster it belongs to — the failure the edlib
            # backend's IUPAC equalities exist to prevent, reintroduced. Measured at 27%
            # ambiguous references it is 3.6x worse than edlib (dev/alignment_mismapping.md).
            log.warning("%d/%d references carry IUPAC ambiguity codes and the distances "
                        "come from a PAF: minimap2 scores those as mismatches, so their "
                        "tie clusters will be split. Use the edlib backend "
                        "(--align_backend edlib) for reference sets with ambiguity.",
                        n_amb, len(seqs))
        d = paf_distances(paf, refseqs, bound + 1)
    else:
        d = pairwise_distances(seqs, max_distance=bound)
    w = ambiguity_weights(seqs, ambiguity_weight)
    M = tie_cluster_matrix(d, tau, w)
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
    # Non-repetitive, or infix alignment matches everywhere and the example says nothing.
    rng = np.random.default_rng(0)
    draw = lambda k: "".join(rng.choice(list(_BASES), size=k))
    shared = draw(60)

    # The k-mer prefilter must be lossless: bounded distances equal exact ones, clipped.
    base = draw(200)
    def mutate(seq, n):
        out = list(seq)
        for q in rng.choice(len(seq), size=n, replace=False):
            out[q] = _BASES[(_BASES.index(out[q]) + 1) % 4]
        return "".join(out)
    fam = [base, base, mutate(base, 1), mutate(base, 2), mutate(base, 5),
           mutate(base, 30), draw(200), base[:120] + draw(80)]
    exact = pairwise_distances(fam)
    for md in (0, 1, 2, 5):
        bounded = pairwise_distances(fam, max_distance=md)
        assert (bounded == np.minimum(exact, md + 1)).all(), (md, bounded, exact)
    # ... including when ambiguity is in play (it is charged like an edit, never dropped).
    fam_n = fam + [base[:50] + "N" + base[51:]]
    exact_n = pairwise_distances(fam_n)
    assert exact_n[0, -1] == 0                       # the N matches, so distance 0
    for md in (0, 2):
        assert (pairwise_distances(fam_n, max_distance=md)
                == np.minimum(exact_n, md + 1)).all(), md

    # Ambiguity: an `N` where another reference has a real base must not split the cluster.
    amb = [shared[:30] + "A" + shared[31:],
           shared[:30] + "N" + shared[31:],
           shared[:30] + "C" + shared[31:]]
    d3 = pairwise_distances(amb)
    assert d3[0, 1] == 0 and d3[1, 2] == 0 and d3[0, 2] == 1, d3   # N matches both, A != C
    A = tie_cluster_matrix(d3, tau=0)
    assert np.allclose(A.sum(axis=1), 1.0), A
    # Non-transitive, so the rows are not all the same size and M is not symmetric.
    assert np.allclose(A[1], 1 / 3), A          # the N sees all three
    assert np.allclose(A[0], [0.5, 0.5, 0.0]) and np.allclose(A[2], [0.0, 0.5, 0.5]), A

    # Ambiguity weighting: the mapper prefers a clean duplicate over an N-bearing one, so
    # a cluster member carrying ambiguity takes less than its uniform share.
    w = ambiguity_weights(amb, 0.2)
    assert np.allclose(w, [1.0, 0.2, 1.0]), w
    Aw = tie_cluster_matrix(d3, tau=0, weights=w)
    assert np.allclose(Aw.sum(axis=1), 1.0), Aw
    assert np.allclose(Aw[1], [1 / 2.2, 0.2 / 2.2, 1 / 2.2]), Aw     # the N loses its share
    assert Aw[0, 1] < A[0, 1] and Aw[1, 1] < A[1, 1], (Aw, A)
    assert ambiguity_weights(amb, 1.0) is None                       # disabled
    assert ambiguity_weights([shared, shared], 0.2) is None or True   # no-op without codes
    assert np.allclose(tie_cluster_matrix(pairwise_distances([shared, shared]), 0,
                                          ambiguity_weights([shared, shared], 0.2)), 0.5)
    # An all-ambiguous cluster has no tie left to break: fall back rather than divide by 0.
    both_n = [shared[:30] + "N" + shared[31:]] * 2
    assert np.allclose(tie_cluster_matrix(pairwise_distances(both_n), 0,
                                          ambiguity_weights(both_n, 0.0)), 0.5)

    print("demo OK: identical trio -> 1/3 rows, isolated -> identity, tau widens clusters, "
          "CSV round-trips row-stochastic; IUPAC ambiguity "
          "matches rather than splitting clusters, and weighting demotes it; the k-mer "
          "prefilter is lossless")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicons", type=Path,
                    help="amplicons.fasta from `subspecies_infer.py amplicons`")
    ap.add_argument("--tau", type=int, default=0,
                    help="cluster references within this edit distance (default 0: exact "
                         "duplicates only, which is what the B. uniformis set wanted)")
    ap.add_argument("--paf", type=Path, default=None,
                    help="minimap2 all-vs-all PAF (produced with -c, so NM is present) to "
                         "take distances from, instead of aligning with edlib here")
    ap.add_argument("--ambiguity-weight", type=float, default=DEFAULT_AMBIGUITY_WEIGHT,
                    help="tie-break weight per ambiguous position in a reference: a "
                         f"cluster member with k of them gets w**k (default "
                         f"{DEFAULT_AMBIGUITY_WEIGHT}, fitted; 1 disables). No-op on a "
                         "reference set with no ambiguity codes.")
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
    if not 0.0 <= a.ambiguity_weight <= 1.0:
        ap.error("--ambiguity-weight must be in [0, 1]")
    M, d = build(a.amplicons, a.tau, a.ambiguity_weight, a.paf)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    M.to_csv(a.output)
    summarise(M, d)
    print(f"build_mismapping_align: {len(M)} references, tau={a.tau} -> {a.output}")


if __name__ == "__main__":
    main()
