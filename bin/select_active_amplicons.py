#!/usr/bin/env python
"""Pick the V4 groups a batch's reads could have come from, by direct sequence comparison.

Measuring ``M`` by simulation does not need the whole database: only groups a read in this
batch could have originated from need a row. This picks that set *without* consulting the
mapper — a group is active when it is an observed label, or within ``--tau`` edits of one
by the kmer neighbour search, which is the same "could a read from here have landed there"
reachability the inference prunes with. Byte-identical references share a row of ``M``, so
one representative per group is written and simulated from.

    select_active_amplicons.py --amplicons amplicons.fasta --obs-mseq S1.mseq.gz ... \\
        --tau 1 -o active_amplicons.fasta

Feed the output to ``simulate_amplicon_reads.py``, map the reads against the *full*
database (the confusion being measured includes labels outside the active set), and build
the matrix with ``infer_composition.py --build-mismapping --build-grouped``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_mismapping_align as bma
import subspecies_infer as si

log = logging.getLogger("select_active_amplicons")


def active_groups(amplicons: Path, obs_mseq: list[Path], tau: int, min_identity: float | None,
                  max_ambiguous_bases: int, max_postings: int) -> tuple[list[str], np.ndarray,
                                                                       list[str], np.ndarray]:
    """Return ``(refseqs, group, sequences, active)`` for one batch's observed labels."""
    refseqs, group, sequences = bma.dedup(amplicons)
    counts = si.observed_refseq_counts(obs_mseq, refseqs, min_identity)
    position = {r: i for i, r in enumerate(refseqs)}
    observed = np.unique([group[position[r]] for r in counts])
    if not len(observed):
        raise SystemExit("no observed read hits a reference amplicon; nothing to simulate")
    if tau < 1:
        return refseqs, group, sequences, observed
    # Reachability only: the kernel's weights and decay play no part in who is active.
    _, kernel, kernel_group, _, _ = bma.build_kmer_grouped(
        amplicons, tau=tau, max_ambiguous_bases=max_ambiguous_bases,
        ambiguity_weight=bma.DEFAULT_AMBIGUITY_WEIGHT, max_postings=max_postings)
    if not np.array_equal(kernel_group, group):
        raise SystemExit("kmer grouping disagrees with the exact-duplicate grouping")
    return refseqs, group, sequences, np.unique(kernel[:, observed].nonzero()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amplicons", type=Path, required=True)
    parser.add_argument("--obs-mseq", type=Path, nargs="+", required=True,
                        help="the batch's observed mapseq output, one file per sample")
    parser.add_argument("--tau", type=int, default=1,
                        help="widen the active set to groups within this many edits")
    parser.add_argument("--min-identity", type=float, default=None)
    parser.add_argument("--max-ambiguous-bases", type=int,
                        default=bma.DEFAULT_MAX_AMBIGUOUS_BASES)
    parser.add_argument("--max-postings", type=int, default=bma.DEFAULT_MAX_POSTINGS)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    a = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    refseqs, group, sequences, active = active_groups(
        a.amplicons, a.obs_mseq, a.tau, a.min_identity, a.max_ambiguous_bases, a.max_postings)
    representative = np.unique(group, return_index=True)[1]      # first member of each group
    with open(a.output, "w") as handle:
        for g in active:
            handle.write(f">{refseqs[representative[g]]}\n{sequences[g]}\n")
    size = np.bincount(group, minlength=len(sequences))
    log.info("active: %d of %d V4 groups (%d of %d references), tau=%d -> %s",
             len(active), len(sequences), int(size[active].sum()), len(refseqs), a.tau,
             a.output)


if __name__ == "__main__":
    main()
