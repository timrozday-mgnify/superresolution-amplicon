#!/usr/bin/env python
"""Build the mis-mapping matrix ``M`` from reference-to-reference alignment alone.

The cheap alternative to the simulate-reads-and-map measurement: instead of sampling
errored reads from every reference and running them through mapseq, align the reference
amplicons to each other and read ``M`` straight off the distances.

The kernel is the *tie cluster*: a read from reference ``a`` is assigned over the
references within ``--tau`` edit operations of ``a`` (``a`` itself included, at
distance 0), in proportion to ``--distance-decay ** d(a, j)``, so

    M[a, j] proportional to c ** d(a, j)   if d(a, j) <= tau, else 0

with ``c = 1`` (the default) giving the uniform split. ``c = 1`` is right at ``tau = 0``,
where every cluster member is at distance 0 and the decay cancels, and wrong at any
larger tau: a reference one edit away is *not* as likely as an exact duplicate. Reaching
it costs one sequencing error at that exact position, so ``c`` is on the order of the
per-base error rate. Measured on the benchmark's two-strain B. uniformis V4 set (two
exact-duplicate clusters 1 edit apart, 0.5% flat error): the simulate+map matrix leaks
0.010 of a row across the gap, which ``c ~= 0.007`` reproduces and ``c = 1`` overstates
by 60x (M[a, j] = 0.2 against a measured 0.002-0.006). Leaving ``c = 1`` at ``tau >= 1``
is warned about, not refused.

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

Backends, in increasing order of what they cost and what they see:

``--backend exact-hash`` (requires ``--tau 0``)
    One streaming pass, no alignment: references are grouped by a digest of the amplicon,
    and each group splits its mass uniformly. Byte identity only — an ``N`` does *not*
    join the group of the sequences it is compatible with, unlike every ``tau >= 1`` path.
``--backend kmer`` (requires ``--tau >= 1``)
    The same duplicate groups, widened by neighbours within ``tau``. Candidates come from
    a pigeonhole block filter over the *distinct* amplicons and are then verified by an
    IUPAC-aware bounded Edlib distance. The filter has no false negatives (see
    ``pigeonhole_candidates``) except at the ``--max-postings`` cap.
``--backend minimap2``
    Index once, align all-vs-all, re-score the PAF CIGARs for IUPAC. Writes the
    reference-square matrix and is the only backend whose cost is quadratic in references.

The first two write the **grouped** matrix (``sparse_matrix.write_grouped``), which is
what makes database scale possible: ``M`` is constant on duplicate groups, so storing it
per *distinct amplicon pair* rather than per reference pair turns GTDB SSU r232 V4 from
24.2 billion nonzeros into 264 thousand — the whole 1,001,241-reference set in 2 s and
0.5 GB at ``tau=0``, 75 s and 0.8 GB at ``tau=1``. Both forms load through
``--mismapping-matrix`` / ``--mismapping_matrix``, and inference never expands either.

Run:
    build_mismapping_align.py --backend exact-hash --tau 0 \\
        --amplicons out/amplicons.fasta -o mismapping_matrix.npz
    build_mismapping_align.py --amplicons out/amplicons.fasta --paf allvsall.paf \\
        -o mismapping_matrix.npz
    build_mismapping_align.py --demo   # self-check
"""
from __future__ import annotations

import argparse
from array import array
from collections.abc import Iterator
from contextlib import nullcontext
import gzip
import hashlib
import logging
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
from scipy import sparse
import edlib

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)
import sparse_matrix as sm  # noqa: E402  (needs sys.path)

log = logging.getLogger("build_mismapping_align")

_BASES = "ACGT"

# Tie-break weight per ambiguous position; see ``ambiguity_weights``. mapseq's penalty on
# an N-bearing reference is real but not constant — measured between 0.18 and 0.97
# depending on which references carry ambiguity (dev/ambiguity_weight_sweep.md) — so this
# is the compromise, not a fitted optimum: best or near-best in 4 of 5 configurations and
# cheap in the fifth. A no-op wherever a reference set carries no ambiguity codes.
DEFAULT_AMBIGUITY_WEIGHT = 0.3

# Distances are resolved exactly out to here so the summary histogram is meaningful;
# anything beyond is reported as "further".
SUMMARY_DISTANCE = 5

# Reported for a reference whose nearest neighbour was never resolved: the grouped
# backends stop looking past ``tau``, so "no neighbour" means "none within tau", not a
# measured distance. It lands in the summary's open final bucket.
UNRESOLVED_DISTANCE = 1 << 20

# A pigeonhole block shared by more than this many distinct amplicons is skipped: those
# are the conserved windows either side of the variable region, where enumerating the
# postings is quadratic and carries no information a rarer block does not already give.
DEFAULT_MAX_POSTINGS = 4096

# Pigeonhole ambiguity budget, counted across the *pair*: with this at 4 the filter is
# exact for any two references carrying four IUPAC positions between them.
DEFAULT_MAX_AMBIGUOUS_BASES = 4

# Postings resolved per slice. Bounds the peak allocation of the candidate expansion,
# which is otherwise proportional to the whole posting list rather than to its pairs.
_POSTING_CHUNK = 1 << 23


# Symbol pairs Edlib must treat as equal on top of literal identity: two IUPAC codes
# match when the base sets they stand for overlap. Passing these to Edlib is what makes
# ambiguity-aware distance run at C speed; the banded Python DP it replaced was ~50x
# slower and is what made an ambiguous reference set the expensive case.
_IUPAC_EQUALITIES = [
    (first, second)
    for first in sorted(si._IUPAC)
    for second in sorted(si._IUPAC)
    if first != second and si._IUPAC[first] & si._IUPAC[second]
]


def bounded_iupac_distance(query: str, target: str, maximum: int) -> int:
    """Return IUPAC-aware edit distance, capped at ``maximum + 1``.

    Args:
        query: First sequence.
        target: Second sequence.
        maximum: Largest distance that needs distinguishing.

    Returns:
        Edit distance when it is at most maximum, otherwise maximum + 1.
    """
    if abs(len(query) - len(target)) > maximum:
        return maximum + 1
    distance = edlib.align(query, target, mode="NW", task="distance", k=maximum,
                           additionalEqualities=_IUPAC_EQUALITIES)["editDistance"]
    return maximum + 1 if distance < 0 else distance


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` pairs without holding the file in memory."""
    header: str | None = None
    parts: list[str] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header, parts = line[1:], []
            else:
                parts.append(line)
    if header is not None:
        yield header, "".join(parts)


def dedup(amplicons: Path, keep_sequences: bool = True
          ) -> tuple[list[str], np.ndarray, list[str]]:
    """Assign every reference to its exact-duplicate group in one streaming pass.

    This is the step that makes the whole thing tractable at database scale: a 16S
    amplicon set is dominated by byte-identical amplicons (GTDB SSU r232 collapses ~9x),
    and both ``M`` and every neighbour search below are constant on those groups.

    Args:
        amplicons: Amplicon FASTA.
        keep_sequences: Retain one sequence per group, needed only by the fuzzy backend.

    Returns:
        Reference headers in file order, each one's group id, and the group
        representatives (empty when ``keep_sequences`` is false).
    """
    refseqs: list[str] = []
    group = array("i")
    # A 128-bit digest instead of the sequence itself: 16 bytes per distinct amplicon
    # rather than ~300, and a collision anywhere in 10^6 sequences has probability ~1e-27.
    seen: dict[bytes, int] = {}
    representatives: list[str] = []
    for header, sequence in iter_fasta(amplicons):
        identifier = seen.get(digest := hashlib.blake2b(sequence.encode(),
                                                        digest_size=16).digest())
        if identifier is None:
            identifier = seen[digest] = len(seen)
            if keep_sequences:
                representatives.append(sequence)
        refseqs.append(header)
        group.append(identifier)
    if not refseqs:
        raise SystemExit(f"{amplicons} contains no sequences")
    if len(set(refseqs)) != len(refseqs):
        raise SystemExit(f"{amplicons} has duplicate reference ids")
    log.info("%d references -> %d distinct amplicons", len(refseqs), len(seen))
    return refseqs, np.frombuffer(group, dtype=np.int32).copy(), representatives


def build_exact_grouped(amplicons: Path
                        ) -> tuple[list[str], sparse.csr_array, np.ndarray, np.ndarray]:
    """Build ``M`` from byte-identical amplicons alone: ``S = diag(1 / group size)``.

    ``--ambiguity-weight`` is deliberately not applied. It demotes a cluster member by how
    much IUPAC ambiguity it carries relative to its partners, and the members of an
    exact-duplicate group carry *the same* ambiguity, so the weights cancel out of the
    normalisation and the split is uniform whatever the weight is.
    """
    refseqs, group, _ = dedup(amplicons, keep_sequences=False)
    sizes = np.bincount(group)
    unique = len(sizes)
    cluster = sparse.csr_array(
        (1.0 / sizes, np.arange(unique, dtype=np.int32),
         np.arange(unique + 1, dtype=np.int32)),
        shape=(unique, unique),
    )
    nearest = np.where(sizes[group] > 1, 0, UNRESOLVED_DISTANCE).astype(np.int32)
    return refseqs, cluster, group, nearest


def _block_pass(sequences: list[str], probes: np.ndarray, tau: int, blocks: int,
                max_postings: int) -> np.ndarray:
    """Return packed ``i * n + j`` pairs sharing one of ``blocks`` displaced blocks.

    Every sequence is indexed; only ``probes`` are searched against that index. Blocks are
    cut on each length's own geometry (``block = length // blocks``) and each probe tries
    the ``2 * tau + 1`` lengths it could pair with, since two sequences within ``tau``
    edits differ in length by at most ``tau``.
    """
    geometry = {length: length // blocks for length in {len(s) for s in sequences}}

    def columns(owners, emit) -> tuple[np.ndarray, np.ndarray]:
        keys, sources = array("q"), array("i")
        for owner in owners:
            for key in emit(sequences[owner]):
                keys.append(key)
                sources.append(int(owner))
        return (np.frombuffer(keys, dtype=np.int64).copy(),
                np.frombuffer(sources, dtype=np.int32).copy())

    # Python's own string hash, not a cryptographic digest: it is randomised per process
    # but index and probe are built in the same one, and a 64-bit collision only proposes a
    # spurious pair, which verification rejects. False positives are free; misses are not.
    def index_blocks(sequence):
        block = geometry[len(sequence)]
        for slot in range(blocks if block else 0):
            yield hash((len(sequence), slot, sequence[slot * block:(slot + 1) * block]))

    def probe_blocks(sequence):
        for length in range(len(sequence) - tau, len(sequence) + tau + 1):
            block = geometry.get(length, 0)
            for slot in range(blocks if block else 0):
                for shift in range(-tau, tau + 1):
                    start = slot * block + shift
                    if 0 <= start <= len(sequence) - block:
                        yield hash((length, slot, sequence[start:start + block]))

    index_keys, index_owners = columns(range(len(sequences)), index_blocks)
    order = np.argsort(index_keys, kind="stable")
    index_keys, index_owners = index_keys[order], index_owners[order]
    probe_keys, probe_owners = columns(probes, probe_blocks)

    low = np.searchsorted(index_keys, probe_keys, "left")
    counts = np.searchsorted(index_keys, probe_keys, "right") - low
    crowded = counts > max_postings
    if crowded.any():
        # The conserved windows either side of the variable region: enumerating their
        # postings is quadratic and they carry nothing a rarer block does not. Dropping
        # them is the one place this filter can miss a pair.
        log.warning("dropped %d of %d block probes over the %d-posting cap",
                    int(crowded.sum()), len(counts), max_postings)
        counts[crowded] = 0
    keep = counts > 0
    low, counts, probe_owners = low[keep], counts[keep], probe_owners[keep]
    log.info("pigeonhole pass: %d blocks, %d probes, %d postings",
             blocks, len(probe_keys), int(counts.sum()))

    # Resolved in slices: the posting list of a whole database is far larger than the
    # distinct pairs it collapses to, and only the slice needs to be resident.
    packed: list[np.ndarray] = []
    cumulative = np.concatenate(([0], np.cumsum(counts)))
    boundaries = np.unique(np.concatenate((
        [0],
        np.searchsorted(cumulative, np.arange(_POSTING_CHUNK, cumulative[-1] + 1,
                                              _POSTING_CHUNK)),
        [len(counts)],
    )))
    for begin, finish in zip(boundaries, boundaries[1:]):
        span = counts[begin:finish]
        total = int(span.sum())
        if not total:
            continue
        offsets = np.arange(total) - np.repeat(np.cumsum(span) - span, span)
        partners = index_owners[np.repeat(low[begin:finish], span) + offsets]
        sources = np.repeat(probe_owners[begin:finish], span)
        lower = np.minimum(sources, partners).astype(np.int64)
        upper = np.maximum(sources, partners).astype(np.int64)
        distinct = lower < upper
        packed.append(np.unique(lower[distinct] * len(sequences) + upper[distinct]))
    return np.unique(np.concatenate(packed)) if packed else np.empty(0, dtype=np.int64)


def pigeonhole_candidates(sequences: list[str], tau: int, max_ambiguous_bases: int,
                          max_postings: int) -> np.ndarray:
    """Return candidate near-duplicate pairs of distinct amplicons, exactly.

    Pigeonhole filter. Cut each sequence into ``P`` disjoint blocks. An aligned pair
    separated by at most ``tau`` edits and ``a`` IUPAC positions can damage at most
    ``tau + a`` of them, so if ``P > tau + a`` at least one block survives as a
    *literally identical* substring of the other sequence, displaced by at most ``tau``.
    The filter therefore has **no false negatives** — unlike a shared-k-mer count, which
    has to materialise every pair it counts.

    Ambiguity is paid for in two passes rather than one, because it is rare (~1% of GTDB
    SSU V4 amplicons) and expensive: budgeting for it globally would need ``P`` blocks so
    short that a conserved window matches thousands of amplicons. The first pass uses
    ``P = tau + 1`` long blocks over the whole set and settles every ambiguity-free pair;
    the second uses ``P = tau + max_ambiguous_bases + 1`` short blocks but probes only
    from the ambiguous sequences, which is enough because a pair this filter could
    otherwise miss has at least one ambiguous member.

    Args:
        sequences: Distinct amplicons, one per duplicate group.
        tau: Edit-distance radius the filter must not miss.
        max_ambiguous_bases: IUPAC positions tolerated across the pair *combined* — two
            references carrying two ambiguity codes each need a budget of four.
        max_postings: Drop a block shared by more than this many sequences.

    Returns:
        ``(m, 2)`` array of ``i < j`` index pairs to verify.
    """
    total = len(sequences)
    packed = _block_pass(sequences, np.arange(total), tau, tau + 1, max_postings)
    ambiguous = np.fromiter(
        (index for index, sequence in enumerate(sequences)
         if not set(sequence).issubset(_BASES)),
        dtype=np.int64,
    )
    log.info("%d of %d distinct amplicons carry IUPAC codes", len(ambiguous), total)
    if len(ambiguous) and max_ambiguous_bases:
        packed = np.union1d(packed, _block_pass(
            sequences, ambiguous, tau, tau + max_ambiguous_bases + 1, max_postings))
    return np.stack(divmod(packed, total), axis=1)


def build_kmer_grouped(
    amplicons: Path,
    tau: int,
    max_ambiguous_bases: int,
    ambiguity_weight: float,
    max_postings: int,
    distance_decay: float = 1.0,
) -> tuple[list[str], sparse.csr_array, np.ndarray, np.ndarray]:
    """Build ``M`` from exact duplicates widened by verified neighbours within ``tau``."""
    if tau < 1:
        raise ValueError("kmer backend requires --tau >= 1; use exact-hash for tau=0")
    refseqs, group, sequences = dedup(amplicons)
    unique = len(sequences)
    pairs = pigeonhole_candidates(sequences, tau, max_ambiguous_bases, max_postings)
    log.info("verifying %d candidate pairs", len(pairs))
    distances = np.fromiter(
        (bounded_iupac_distance(sequences[i], sequences[j], tau) for i, j in pairs),
        dtype=np.int32, count=len(pairs),
    )
    neighbours = pairs[distances <= tau]
    log.info("%d candidate pairs, %d within tau=%d", len(pairs), len(neighbours), tau)

    sizes = np.bincount(group, minlength=unique)
    weights = ambiguity_weights(sequences, ambiguity_weight)
    # ``adjacency`` carries the distance decay, so it is the membership matrix
    # ``tie_cluster_matrix`` normalises densely, not a 0/1 pattern.
    edge = np.float64(distance_decay) ** distances[distances <= tau].astype(np.float64)
    rows = np.concatenate([np.arange(unique), neighbours[:, 0], neighbours[:, 1]])
    columns = np.concatenate([np.arange(unique), neighbours[:, 1], neighbours[:, 0]])
    adjacency = sparse.csr_array(
        (np.concatenate([np.ones(unique), edge, edge]), (rows, columns)),
        shape=(unique, unique))
    adjacency.setdiag(1.0)                        # coo summed the duplicate self-entries
    mass = sizes * (1.0 if weights is None else weights)
    per_row = adjacency @ mass
    width = np.diff(adjacency.indptr)
    share = (adjacency.data if weights is None
             else adjacency.data * weights[adjacency.indices])
    dead = per_row == 0
    if dead.any():
        # An all-ambiguous cluster has no tie left to break; fall back to the plain split
        # rather than divide by zero (same convention as ``_normalise``).
        source_of = np.repeat(np.arange(unique), width)
        share = np.where(dead[source_of], adjacency.data, share)
        per_row = np.where(dead, adjacency @ sizes.astype(np.float64), per_row)
    data = share / np.repeat(per_row, width)
    cluster = sparse.csr_array((data, adjacency.indices, adjacency.indptr),
                               shape=(unique, unique))

    nearest_unique = np.full(unique, UNRESOLVED_DISTANCE, dtype=np.int32)
    for (source, target), distance in zip(neighbours, distances[distances <= tau]):
        nearest_unique[source] = min(nearest_unique[source], distance)
        nearest_unique[target] = min(nearest_unique[target], distance)
    nearest = np.where(sizes[group] > 1, 0, nearest_unique[group]).astype(np.int32)
    return refseqs, cluster, group, nearest


def _cigar_distance(cigar: str, query: str, target: str) -> int:
    """Return an IUPAC-aware edit distance for one PAF CIGAR alignment.

    ``M``, ``=`` and ``X`` consume one base on both sequences. They are deliberately
    re-scored from the sequences instead of using the operation letter: minimap2 emits
    ``M`` for mixed runs and its internal score treats every ambiguity code as a mismatch.
    """
    query_pos = target_pos = distance = 0
    count = ""
    for char in cigar:
        if char.isdigit():
            count += char
            continue
        if not count or char not in "MID=X":
            raise ValueError(f"invalid minimap2 CIGAR: {cigar!r}")
        length = int(count)
        count = ""
        if char in "M=X":
            for query_base, target_base in zip(
                query[query_pos:query_pos + length],
                target[target_pos:target_pos + length],
            ):
                if not si._IUPAC.get(query_base, set()).intersection(
                    si._IUPAC.get(target_base, set())
                ):
                    distance += 1
            query_pos += length
            target_pos += length
        elif char == "I":
            distance += length
            query_pos += length
        else:
            distance += length
            target_pos += length
    if count or query_pos != len(query) or target_pos != len(target):
        raise ValueError(f"CIGAR does not consume its PAF alignment: {cigar!r}")
    return distance


def _open_paf(paf: Path | TextIO):
    """Return a context manager yielding a text PAF stream."""
    if not isinstance(paf, Path):
        return nullcontext(paf)
    opener = gzip.open if paf.suffix == ".gz" else open
    return opener(paf, "rt")


def paf_distances(
    paf: Path | TextIO,
    refseqs: list[str],
    sequences: list[str],
    sentinel: int,
) -> np.ndarray:
    """Return IUPAC-aware distances from minimap2 all-vs-all PAF (``-c`` required).

    The ``cg`` CIGAR is re-scored against the original sequences with IUPAC set overlap.
    Unaligned bases at either end count as edits, so a clean half-length local alignment is
    never mistaken for a close whole-amplicon match. The smallest value over a pair's
    alignments wins. Pairs minimap2 never reports get ``sentinel`` — "further than this
    backend can see", not "infinitely far".

    Treat the result as data, not truth: minimap2 is a heuristic whose recall falls off
    with divergence, and which preset you ran matters more than ``-p``/``-N``
    (dev/alignment_backend_benchmark.md). It finds every identical and one-base-apart pair
    on the reference set tested, which is what the tie-cluster kernel needs; the far tail
    is where it stops being complete.
    """
    idx = {reference: i for i, reference in enumerate(refseqs)}
    n = len(refseqs)
    d = np.full((n, n), sentinel, dtype=np.int32)
    np.fill_diagonal(d, 0)
    unknown = 0
    with _open_paf(paf) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 12:
                continue
            i, j = idx.get(f[0]), idx.get(f[5])
            if i is None or j is None:
                unknown += 1
                continue
            cigar = next((tag[5:] for tag in f[12:] if tag.startswith("cg:Z:")), None)
            if cigar is None:
                log.warning("PAF row has no cg:Z tag; rerun minimap2 with -c")
                continue
            query_start, query_end = int(f[2]), int(f[3])
            target_start, target_end = int(f[7]), int(f[8])
            query = sequences[i][query_start:query_end]
            if f[4] == "-":
                query = si.revcomp(query)
            target = sequences[j][target_start:target_end]
            try:
                aligned_distance = _cigar_distance(cigar, query, target)
            except ValueError as error:
                log.warning("skipping malformed PAF CIGAR for %s -> %s: %s", f[0], f[5], error)
                continue
            dist = (aligned_distance + (int(f[1]) - (query_end - query_start))
                    + (int(f[6]) - (target_end - target_start)))
            dist = min(dist, sentinel)
            if dist < d[i, j]:
                d[i, j] = d[j, i] = dist
    if unknown:
        log.warning("%d PAF rows referenced an unknown sequence id", unknown)
    return d


def ambiguity_weights(seqs: list[str], weight: float) -> np.ndarray | None:
    """Per-reference tie-break weight ``weight ** (number of ambiguous positions)``.

    Ambiguity codes match anything in the PAF re-score, so an `N`-bearing reference joins the
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
    """Row-normalise a membership matrix (boolean, or decayed by distance), weighted if given.

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
                       weights: np.ndarray | None = None,
                       decay: float = 1.0) -> np.ndarray:
    """Row-stochastic ``M`` from a distance matrix: split over each tie cluster.

    Self-distance is 0, so every cluster contains its own reference and no row is ever
    empty; an isolated reference gets the identity row for free. ``weights`` (see
    ``ambiguity_weights``) splits a cluster unevenly instead of uniformly, and ``decay``
    (see the module docstring) discounts a member by ``decay ** d``, which is a no-op at
    ``tau = 0`` and the difference between a useful and a harmful ``tau >= 1``.
    """
    member = np.where(d <= tau, np.float64(decay) ** np.minimum(d, tau), 0.0)
    return _normalise(member, weights)


def build(
    amplicons: Path,
    paf: Path,
    tau: int = 0,
    ambiguity_weight: float = 1.0,
    distance_decay: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray]:
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
    d = paf_distances(paf, refseqs, seqs, bound + 1)
    w = ambiguity_weights(seqs, ambiguity_weight)
    M = tie_cluster_matrix(d, tau, w, distance_decay)
    return pd.DataFrame(M, index=refseqs, columns=refseqs), d


def build_sparse(
    amplicons: Path,
    paf: Path | TextIO,
    tau: int = 0,
    ambiguity_weight: float = 1.0,
    distance_decay: float = 1.0,
) -> tuple[list[str], sparse.csr_array, np.ndarray]:
    """Build the alignment kernel without allocating a reference-square array."""
    records = si.read_fasta(amplicons)
    if not records:
        raise SystemExit(f"{amplicons} contains no sequences")
    refseqs = [header for header, _ in records]
    if len(set(refseqs)) != len(refseqs):
        raise SystemExit(f"{amplicons} has duplicate reference ids")
    sequences = [sequence for _, sequence in records]
    index = {reference: position for position, reference in enumerate(refseqs)}
    n_refs = len(refseqs)
    bound = max(tau, SUMMARY_DISTANCE)
    nearest = np.full(n_refs, bound + 1, dtype=np.int32)
    pairs: dict[tuple[int, int], int] = {}
    unknown = 0
    with _open_paf(paf) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            source, target = index.get(fields[0]), index.get(fields[5])
            if source is None or target is None:
                unknown += 1
                continue
            cigar = next((tag[5:] for tag in fields[12:] if tag.startswith("cg:Z:")), None)
            if cigar is None:
                log.warning("PAF row has no cg:Z tag; rerun minimap2 with -c")
                continue
            query_start, query_end = int(fields[2]), int(fields[3])
            target_start, target_end = int(fields[7]), int(fields[8])
            query = sequences[source][query_start:query_end]
            if fields[4] == "-":
                query = si.revcomp(query)
            target_sequence = sequences[target][target_start:target_end]
            try:
                aligned = _cigar_distance(cigar, query, target_sequence)
            except ValueError as error:
                log.warning("skipping malformed PAF CIGAR for %s -> %s: %s",
                            fields[0], fields[5], error)
                continue
            distance = aligned + (int(fields[1]) - (query_end - query_start)) + (
                int(fields[6]) - (target_end - target_start)
            )
            if source != target and distance <= bound:
                nearest[source] = min(nearest[source], distance)
                nearest[target] = min(nearest[target], distance)
            if distance <= tau:
                key = (min(source, target), max(source, target))
                pairs[key] = min(pairs.get(key, distance), distance)
    if unknown:
        log.warning("%d PAF rows referenced an unknown sequence id", unknown)

    members: list[dict[int, int]] = [{row: 0} for row in range(n_refs)]
    for (source, target), distance in pairs.items():
        members[source][target] = distance
        members[target][source] = distance
    weights = ambiguity_weights(sequences, ambiguity_weight)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for source, targets in enumerate(members):
        target_list = sorted(targets)
        decayed = np.float64(distance_decay) ** np.array(
            [targets[target] for target in target_list], dtype=np.float64)
        weight = decayed if weights is None else decayed * weights[target_list]
        if weight.sum() == 0:
            weight = decayed
        rows.extend([source] * len(target_list))
        columns.extend(target_list)
        values.extend(weight.tolist())
    return refseqs, sparse.csr_array((values, (rows, columns)), shape=(n_refs, n_refs)), nearest


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


def summarise_sparse(M: sparse.csr_array, nearest: np.ndarray,
                     group: np.ndarray | None = None) -> None:
    """Print alignment-kernel diagnostics without densifying its matrix.

    ``group`` switches the diagonal from the reference-square form to the grouped one,
    where ``diag(M)[a] = S[group[a], group[a]]``.
    """
    diag = M.diagonal() if group is None else sm.grouped_diagonal(M, group)
    print("nearest other reference: " + "  ".join(
        f"d={low if low == high else f'{low}-{high}'}: "
        f"{int(((nearest >= low) & (nearest <= high)).sum())}"
        for low, high in ((0, 0), (1, 1), (2, 3), (4, 10), (11, UNRESOLVED_DISTANCE))
    ) + "  (the last bucket is 'no neighbour found', not a measured distance)")
    # Counted from the sparsity pattern, not from 1/diag: ambiguity weighting can drive a
    # heavily-N'd reference's own share to ~1e-37, and its reciprocal is not a cluster size.
    if group is None:
        cluster = np.asarray((M != 0).sum(axis=1)).ravel()
    else:
        pattern = M.copy()
        pattern.data = np.ones_like(pattern.data)
        cluster = (pattern @ np.bincount(group, minlength=M.shape[0]).astype(np.float64))[group]
    print(f"mean diag(M) {diag.mean():.4f}  median {np.median(diag):.4f}  "
          f"unambiguous {100 * (diag == 1.0).mean():.0f}%  "
          f"mean cluster {cluster.mean():.2f}  median {np.median(cluster):.0f}")


def demo() -> None:
    """Self-check the distance and the kernel on sequences with a known answer."""
    trio = "ACGTACGTACGTACGTACGT"
    seqs = [trio, trio, trio,                  # an identical trio -> 1/3 rows
            trio[:-1] + "A",                   # one base off the trio
            "T" * 20]                          # isolated
    d = np.array([
        [0, 0, 0, 1, 6],
        [0, 0, 0, 1, 6],
        [0, 0, 0, 1, 6],
        [1, 1, 1, 0, 6],
        [6, 6, 6, 6, 0],
    ])

    M = tie_cluster_matrix(d, tau=0)
    assert np.allclose(M.sum(axis=1), 1.0), M.sum(axis=1)          # row-stochastic
    assert np.allclose(np.diag(M)[:3], 1 / 3), np.diag(M)          # trio splits 3 ways
    assert np.allclose(M[3], np.eye(5)[3]) and np.allclose(M[4], np.eye(5)[4])  # identity
    M1 = tie_cluster_matrix(d, tau=1)
    assert np.allclose(np.diag(M1)[:4], 1 / 4), np.diag(M1)        # tau pulls #3 in
    assert np.allclose(M1[4], np.eye(5)[4])                        # still isolated

    # Distance decay: the distance-1 neighbour is no longer an equal cluster member.
    Md = tie_cluster_matrix(d, tau=1, decay=0.01)
    assert np.allclose(Md.sum(axis=1), 1.0), Md.sum(axis=1)        # still row-stochastic
    assert np.allclose(Md[:3], tie_cluster_matrix(d, tau=0)[:3], atol=4e-3), Md
    assert Md[0, 3] < M1[0, 3] / 30 and Md[3, 3] > 0.95, Md        # #3 keeps its own mass
    assert np.allclose(tie_cluster_matrix(d, tau=0, decay=0.01),
                       tie_cluster_matrix(d, tau=0))               # no-op at tau=0

    # End to end through a fasta, and the invariants infer_composition.py checks on load.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "amplicons.fasta"
        fa.write_text("".join(f">ref|{i}|x\n{s}\n" for i, s in enumerate(seqs)))
        paf = Path(td) / "allvsall.paf"
        paf.write_text("\n".join([
            "ref|0|x\t20\t0\t20\t+\tref|1|x\t20\t0\t20\t20\t20\t60\tcg:Z:20M",
            "ref|0|x\t20\t0\t20\t+\tref|2|x\t20\t0\t20\t20\t20\t60\tcg:Z:20M",
            "ref|1|x\t20\t0\t20\t+\tref|2|x\t20\t0\t20\t20\t20\t60\tcg:Z:20M",
            "ref|0|x\t20\t0\t20\t+\tref|3|x\t20\t0\t20\t19\t20\t60\tcg:Z:20M",
        ]) + "\n")
        frame, _ = build(fa, paf, tau=0)
        frame.to_csv(Path(td) / "M.csv")
        back = pd.read_csv(Path(td) / "M.csv", index_col=0)
        assert list(back.index) == list(back.columns) == [f"ref|{i}|x" for i in range(5)]
        assert np.isfinite(back.to_numpy()).all() and (back.to_numpy() >= 0).all()
        assert np.allclose(back.to_numpy().sum(axis=1), 1.0)
        # Grouped backends: M[a, j] == S[group[a], group[j]] must equal the dense kernel.
        _, exact_S, exact_group, _ = build_exact_grouped(fa)
        assert np.allclose(exact_S.toarray()[np.ix_(exact_group, exact_group)],
                           tie_cluster_matrix(d, tau=0))
        _, kmer_S, kmer_group, kmer_near = build_kmer_grouped(
            fa, tau=1, max_ambiguous_bases=2, ambiguity_weight=1.0,
            max_postings=DEFAULT_MAX_POSTINGS,
        )
        assert np.allclose(kmer_S.toarray()[np.ix_(kmer_group, kmer_group)], M1)
        assert list(kmer_near) == [0, 0, 0, 1, UNRESOLVED_DISTANCE], kmer_near
        # ... and all three backends agree under decay too, not just at c = 1.
        _, decay_S, decay_group, _ = build_kmer_grouped(
            fa, tau=1, max_ambiguous_bases=2, ambiguity_weight=1.0,
            max_postings=DEFAULT_MAX_POSTINGS, distance_decay=0.01,
        )
        assert np.allclose(decay_S.toarray()[np.ix_(decay_group, decay_group)], Md)
        # The PAF path, on the pairs this PAF actually reports (it omits 1-3 and 2-3, so
        # its clusters are smaller than the hand-written ``d`` above).
        decay_frame = build(fa, paf, tau=1, distance_decay=0.01)[0].to_numpy()
        assert np.allclose(decay_frame.sum(axis=1), 1.0), decay_frame
        assert decay_frame[3, 3] > 0.98 and decay_frame[0, 3] < 0.01, decay_frame
        sm.write_grouped(Path(td) / "g.npz", kmer_S, kmer_group,
                         [f"ref|{i}|x" for i in range(5)])
        back_S, back_group = sm.read_grouped(Path(td) / "g.npz",
                                             [f"ref|{i}|x" for i in range(5)][::-1])
        assert np.allclose(back_S.toarray()[np.ix_(back_group, back_group)], M1[::-1, ::-1])
        assert np.allclose(sm.grouped_diagonal(back_S, back_group), np.diag(M1)[::-1])

    # The pigeonhole filter must not miss a pair inside tau, whatever the block layout.
    near = ["ACGTACGTACGTACGTACGTAAAA", "ACGTACGTACGTACGTACGTAAAC",   # 1 substitution
            "ACGTACGTACGTCGTACGTAAAA", "TTTTTTTTTTTTTTTTTTTTTTTT"]     # 1 deletion; far
    found = {tuple(pair) for pair in pigeonhole_candidates(near, 1, 0, 1 << 30)}
    assert (0, 1) in found and (0, 2) in found and (1, 3) not in found, found
    # IUPAC matches directly from PAF CIGAR columns, unlike minimap2's NM tag.
    amb = ["ACGA", "ACGN", "ACGC"]
    assert _cigar_distance("4M", amb[0], amb[1]) == 0
    assert _cigar_distance("4M", amb[1], amb[2]) == 0
    assert _cigar_distance("4M", amb[0], amb[2]) == 1
    assert _cigar_distance("2M1I2M", "ACGTA", "ACTA") == 1
    assert _cigar_distance("2M1D2M", "ACTA", "ACGTA") == 1
    d3 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
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
    assert np.allclose(tie_cluster_matrix(np.array([[0, 0], [0, 0]]), 0,
                                          ambiguity_weights(["ACGT", "ACGT"], 0.2)), 0.5)
    # An all-ambiguous cluster has no tie left to break: fall back rather than divide by 0.
    both_n = ["ACGN"] * 2
    assert np.allclose(tie_cluster_matrix(np.array([[0, 0], [0, 0]]), 0,
                                          ambiguity_weights(both_n, 0.0)), 0.5)

    print("demo OK: identical trio -> 1/3 rows, isolated -> identity, tau widens clusters "
          "and distance decay discounts what it pulls in, "
          "CSV round-trips row-stochastic; PAF CIGAR re-scoring keeps IUPAC ambiguity "
          "from splitting clusters, and weighting demotes it")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicons", type=Path,
                    help="amplicons.fasta from `subspecies_infer.py amplicons`")
    ap.add_argument("--tau", type=int, default=0,
                    help="cluster references within this edit distance (default 0: exact "
                         "duplicates only, which is what the B. uniformis set wanted)")
    ap.add_argument("--paf", type=Path,
                    help="minimap2 all-vs-all PAF, produced with -c for cg CIGAR tags")
    ap.add_argument("--backend", choices=("minimap2", "kmer", "exact-hash"), default="minimap2")
    ap.add_argument(
        "--max-ambiguous-bases",
        type=int,
        default=DEFAULT_MAX_AMBIGUOUS_BASES,
        help="IUPAC positions a pair may carry between them before the pigeonhole filter "
             "is allowed to miss it; each one costs a block, so raising it shortens the "
             f"blocks and slows the search (default: {DEFAULT_MAX_AMBIGUOUS_BASES})",
    )
    ap.add_argument(
        "--max-postings",
        type=int,
        default=DEFAULT_MAX_POSTINGS,
        help="skip a pigeonhole block shared by more than this many distinct amplicons "
             f"(default: {DEFAULT_MAX_POSTINGS}); the only source of false negatives",
    )
    ap.add_argument("--distance-decay", type=float, default=1.0,
                    help="discount a cluster member by this factor per edit of distance: "
                         "M[a, j] proportional to c ** d(a, j). A no-op at --tau 0 (every "
                         "member is at distance 0). At --tau >= 1 the default of 1 makes a "
                         "reference one edit away as likely as an exact duplicate, which "
                         "overstates confusion by orders of magnitude; set it to about the "
                         "per-base error rate (~0.007 measured at 0.5%% flat error).")
    ap.add_argument("--ambiguity-weight", type=float, default=DEFAULT_AMBIGUITY_WEIGHT,
                    help="tie-break weight per ambiguous position in a reference: a "
                         f"cluster member with k of them gets w**k (default "
                         f"{DEFAULT_AMBIGUITY_WEIGHT}, fitted; 1 disables). No-op on a "
                         "reference set with no ambiguity codes.")
    ap.add_argument("-o", "--output", type=Path, help="output labelled CSR mismapping_matrix.npz")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if a.demo:
        return demo()
    required = ("amplicons", "output") if a.backend != "minimap2" else ("amplicons", "paf", "output")
    for req in required:
        if getattr(a, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (unless --demo)")
    if not 0.0 <= a.ambiguity_weight <= 1.0:
        ap.error("--ambiguity-weight must be in [0, 1]")
    if not 0.0 <= a.distance_decay <= 1.0:
        ap.error("--distance-decay must be in [0, 1]")
    if a.tau >= 1 and a.distance_decay == 1.0:
        log.warning("--tau %d with --distance-decay 1: every reference within tau is "
                    "treated as an exact duplicate, which overstates confusion between "
                    "references a base or two apart. Set --distance-decay to about the "
                    "per-base error rate.", a.tau)
    if a.max_ambiguous_bases < 0 or a.max_postings < 1:
        ap.error("--max-ambiguous-seed-bases must be non-negative and --max-postings >= 1")
    if a.backend == "kmer":
        refseqs, M, group, nearest = build_kmer_grouped(
            a.amplicons, a.tau, a.max_ambiguous_bases, a.ambiguity_weight,
            a.max_postings, a.distance_decay,
        )
    elif a.backend == "exact-hash":
        if a.tau != 0:
            ap.error("--backend exact-hash requires --tau 0")
        refseqs, M, group, nearest = build_exact_grouped(a.amplicons)
    else:
        group = None
        refseqs, M, nearest = build_sparse(a.amplicons, a.paf, a.tau, a.ambiguity_weight,
                                           a.distance_decay)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if group is None:
        sm.write_matrix(a.output, M, refseqs)
    else:
        sm.write_grouped(a.output, M, group, refseqs)
    summarise_sparse(M, nearest, group)
    print(f"build_mismapping_align: {len(refseqs)} references, tau={a.tau}, "
          f"decay={a.distance_decay} -> {a.output}")


if __name__ == "__main__":
    main()
