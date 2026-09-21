#!/usr/bin/env python
"""Sequence-distance helpers for ``build_panel_kernel.py align``.

The alignment kernel weights a database label within ``--tau`` edits of a panel source by
``--distance-decay ** d``. This module finds the candidate pairs (a pigeonhole block
filter with no false negatives), verifies them with an IUPAC-aware bounded Edlib
distance, measures the per-base error rate ``--distance-decay auto`` stands for, and
down-weights ambiguous labels.
"""
from __future__ import annotations

import logging
import sys
from array import array
from pathlib import Path

import edlib
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)

log = logging.getLogger("kernel_align")

_BASES = "ACGT"

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


def pigeonhole_candidates(
    sequences: list[str],
    tau: int,
    max_ambiguous_bases: int,
    max_postings: int,
    probes: np.ndarray | None = None,
) -> np.ndarray:
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
        probes: Sequence indexes to search against all sequences. By default every
            sequence is searched. Ambiguous sequences are also searched when needed to
            retain the filter's recall guarantee for a clean probe and ambiguous target.

    Returns:
        ``(m, 2)`` array of ``i < j`` index pairs to verify.
    """
    total = len(sequences)
    probe_indexes = (np.arange(total, dtype=np.int64) if probes is None
                     else np.unique(np.asarray(probes, dtype=np.int64)))
    if (probe_indexes < 0).any() or (probe_indexes >= total).any():
        raise ValueError("probe indexes must address the supplied sequences")
    packed = _block_pass(sequences, probe_indexes, tau, tau + 1, max_postings)
    ambiguous = np.fromiter(
        (index for index, sequence in enumerate(sequences)
         if not set(sequence).issubset(_BASES)),
        dtype=np.int64,
    )
    log.info("%d of %d distinct amplicons carry IUPAC codes", len(ambiguous), total)
    if len(ambiguous) and max_ambiguous_bases:
        # A clean probe can be near an ambiguous target. Searching the ambiguous targets
        # as well as the requested probes preserves the pigeonhole guarantee in that
        # direction without indexing a second copy of the database.
        packed = np.union1d(packed, _block_pass(
            sequences, ambiguous, tau, tau + max_ambiguous_bases + 1, max_postings))
    return np.stack(divmod(packed, total), axis=1)


def measure_error_rate(error_model: str = "flat", model_pt: Path | None = None,
                       sub_rate: float = 0.005, ins_rate: float = 0.0005,
                       del_rate: float = 0.0005, length: int = 1000, n: int = 200,
                       seed: int = 0) -> float:
    """Mean per-base edit rate of a sequencing-error model, measured by sampling it.

    ``--distance-decay auto`` is this number. Reaching a distance-1 reference costs one
    error at the one position that separates them, so the decay is on the order of the
    per-base error rate — measuring it beats asserting it, and it is the only way to get a
    number out of a *trained* skiver model, whose rate is context-dependent and not a
    parameter anyone can read off.

    Measured rather than summed from the flat rates so both models go through one path:
    at the pipeline's defaults (0.005/0.0005/0.0005) this returns ~0.006, against the
    0.007 fitted by matching the simulate+map matrix on the two-strain B. uniformis set.
    """
    import simulate_amplicon_reads as sim   # local: pulls in skiver for a trained model

    rng = np.random.default_rng(seed)
    sequence = "".join(rng.choice(list(_BASES), size=length))
    reads = sim.sampler(error_model, model_pt, sub_rate, ins_rate, del_rate)(
        [(str(i), sequence, True) for i in range(n)], rng)
    edits = sum(edlib.align(read, sequence, mode="NW", task="distance")["editDistance"]
                for _, read in reads)
    rate = edits / float(n * length)
    log.info("measured per-base error rate %.5f (%s model, %d x %dbp)",
             rate, error_model, n, length)
    return rate


def ambiguity_weights(seqs: list[str], weight: float) -> np.ndarray | None:
    """Per-reference tie-break weight ``weight ** (number of ambiguous positions)``.

    Ambiguity codes match anything in ``bounded_iupac_distance``, so an `N`-bearing label
    is within reach of the sequences it could equal — but the *mapper* does not treat it
    as an equal: mapseq scores the `N` as a mismatch, so a clean duplicate wins the read.
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
