#!/usr/bin/env python
"""Convert observed fastq reads to the FASTA mapseq expects, optionally primer-trimmed.

mapseq takes FASTA only. Trimming the primers off (``--fwd-primer``/``--rev-primer``,
reusing ``subspecies_infer.trim_read_primers``) puts the reads in the same coordinate
space as the primer-trimmed reference amplicons and as the simulated reads; reads with no
primer found are passed through unchanged. ``--max-reads`` subsamples reservoir-style so
depth is bounded without holding the whole file in memory.

``--paired`` treats the inputs as R1/R2 of the same fragments and **merges each pair into
one query** rather than mapping the mates independently. This matters: a 150bp mate
mapped against a 253bp reference amplicon only sees part of it, so references differing
outside that window are confusable — the whole point of the mis-mapping matrix, and a
confusion the merged fragment does not suffer. A merged pair spans the amplicon, which is
the coverage assumption the alignment-built ``M`` makes. Pairs that do not overlap cannot
be merged; they are reported, and a large count means the amplicon is not covered and the
matrix should come from ``--mismapping_method simulate --sim_read_len`` instead.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

import edlib
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)

log = logging.getLogger("reads_to_fasta")

# Seed length used to place R2 inside R1. Long enough to be unique in an amplicon,
# short enough to survive the error-prone end of a mate.
_SEED = 25


def iter_fastq(path: Path):
    """Yield (name, sequence) from a (gzipped) fastq."""
    opener = gzip.open if str(path).endswith(".gz") else open
    name = None
    with opener(path, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 4 == 0:
                name = line[1:].strip().split()[0]
            elif i % 4 == 1 and name is not None:
                yield name, line.strip().upper()


def merge_pair(r1: str, r2: str, min_overlap: int = 20,
               max_mismatch_frac: float = 0.25) -> str | None:
    """Merge R1 with the reverse complement of R2 on their overlap, or ``None``.

    The mates come from opposite ends of one fragment, so ``rc(R2)`` continues where R1
    ends. A short seed from the start of ``rc(R2)`` is placed in R1 by infix alignment
    (edlib ``HW``); that placement fixes the overlap, which is then checked as a whole
    before the mates are joined. R1's bases win inside the overlap — without qualities
    there is nothing better to break ties with, and R1 is the higher-quality mate on
    Illumina.
    """
    rc = si.revcomp(r2)
    seed = rc[:_SEED]
    if len(r1) < _SEED or len(rc) < _SEED:
        return None
    hit = edlib.align(seed, r1, mode="HW", task="locations")
    if hit["editDistance"] < 0 or not hit["locations"]:
        return None
    start = hit["locations"][0][0]
    overlap = len(r1) - start
    if overlap < min_overlap or overlap > len(rc):
        return None
    a, b = r1[start:], rc[:overlap]
    if sum(x != y for x, y in zip(a, b)) > max_mismatch_frac * overlap:
        return None
    return r1 + rc[overlap:]


def iter_pairs(p1: Path, p2: Path):
    """Yield ``(name, r1, r2)`` from two fastqs read in lockstep."""
    for (n1, s1), (n2, s2) in zip(iter_fastq(p1), iter_fastq(p2)):
        yield n1, s1, s2


def reservoir(items, k: int, rng):
    """Uniform sample of ``k`` items from a stream of unknown length (all if fewer)."""
    keep = []
    for n, item in enumerate(items):
        if n < k:
            keep.append(item)
        else:
            j = int(rng.integers(0, n + 1))
            if j < k:
                keep[j] = item
    return keep


def run(a) -> None:
    rng = np.random.default_rng(a.seed)
    paths = sorted(p for pat in a.reads for p in Path().glob(pat)) if a.glob else a.reads
    if not paths:
        raise SystemExit(f"no read files matched {a.reads}")
    unmerged = 0
    if a.paired:
        if len(paths) != 2:
            raise SystemExit(f"--paired needs exactly two fastq files, got {len(paths)}")
        # Sample *fragments*, so a pair is kept or dropped together rather than split.
        pairs = iter_pairs(*paths)
        pairs = reservoir(pairs, a.max_reads, rng) if a.max_reads else list(pairs)
        reads = []
        for name, r1, r2 in pairs:
            merged = merge_pair(r1, r2, a.min_overlap)
            if merged is None:
                unmerged += 1
                continue
            reads.append((name, merged))
    else:
        stream = ((name, seq) for p in paths for name, seq in iter_fastq(p))
        reads = reservoir(stream, a.max_reads, rng) if a.max_reads else list(stream)

    trim = not a.no_trim_primers and bool(a.fwd_primer and a.rev_primer)
    n = 0
    with open(a.output, "w") as out:
        for name, seq in reads:
            if trim:
                seq = si.trim_read_primers(seq, a.fwd_primer, a.rev_primer,
                                           a.primer_mismatches)
            if not seq:
                continue
            # mapseq keys on the first whitespace token; read names are already tokens.
            out.write(f">{name}_{n}\n{seq}\n")
            n += 1
    log.info("wrote %d reads from %d file(s) -> %s", n, len(paths), a.output)
    unit = "merged fragments" if a.paired else "reads"
    note = ""
    if a.paired:
        total = n + unmerged
        frac = unmerged / total if total else 0.0
        note = f" ({unmerged} pairs did not overlap and were dropped)"
        if frac > 0.2:
            # The alignment-built M assumes a query spans the amplicon. It does not here.
            log.warning("%.0f%% of pairs could not be merged: the fragments do not cover "
                        "the amplicon, so a whole-amplicon mis-mapping matrix will "
                        "understate confusion — use --mismapping_method simulate with "
                        "--sim_read_len for this sample", 100 * frac)
    print(f"reads_to_fasta: {n} {unit} -> {a.output}{note}")


def demo() -> None:
    import tempfile

    rng = np.random.default_rng(0)
    # Reservoir sampling: right size, subset of the input, and roughly uniform coverage.
    sample = reservoir(range(1000), 100, rng)
    assert len(sample) == 100 and set(sample) <= set(range(1000))
    assert len(reservoir(range(10), 100, rng)) == 10          # fewer than k
    counts = np.zeros(10)
    for _ in range(500):
        for v in reservoir(range(10), 3, rng):
            counts[v] += 1
    assert counts.min() > 500 * 3 / 10 * 0.7, counts          # no position starved

    # Pair merging: a fragment split into overlapping mates must come back intact.
    frag = "".join(rng.choice(list("ACGT"), size=253))
    r1, r2 = frag[:150], si.revcomp(frag[103:])
    assert merge_pair(r1, r2) == frag
    # ... with a sequencing error inside the overlap (R1 wins, so the merge equals frag).
    rc_bad = list(si.revcomp(r2))
    rc_bad[30] = "ACGT"[("ACGT".index(rc_bad[30]) + 1) % 4]
    assert merge_pair(r1, si.revcomp("".join(rc_bad))) == frag
    # ... and one *outside* the overlap survives into the merged fragment, as it should.
    rc_bad = list(si.revcomp(r2))
    rc_bad[-5] = "ACGT"[("ACGT".index(rc_bad[-5]) + 1) % 4]
    assert merge_pair(r1, si.revcomp("".join(rc_bad))) == frag[:-5] + rc_bad[-5] + frag[-4:]
    # ... and at other overlaps, including the whole read.
    assert merge_pair(frag[:200], si.revcomp(frag[100:])) == frag
    assert merge_pair(frag[:150], si.revcomp(frag[:150])) == frag[:150]
    # Mates that do not overlap cannot be merged, and must say so rather than guess.
    assert merge_pair(frag[:100], si.revcomp(frag[150:])) is None
    assert merge_pair(frag[:150], "".join(rng.choice(list("ACGT"), size=150))) is None
    assert merge_pair("ACGT", "ACGT") is None                     # shorter than the seed

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fq = td / "r.fastq"
        fq.write_text("@r1 extra\nACGTACGT\n+\nIIIIIIII\n@r2\nTTTTGGGG\n+\nIIIIIIII\n")
        assert list(iter_fastq(fq)) == [("r1", "ACGTACGT"), ("r2", "TTTTGGGG")]
        a = argparse.Namespace(reads=[fq], glob=False, max_reads=0, seed=0,
                               fwd_primer=None, rev_primer=None, no_trim_primers=True,
                               primer_mismatches=3, paired=False, min_overlap=20,
                               output=td / "out.fasta")
        run(a)
        assert dict(si.read_fasta(td / "out.fasta")) == {"r1_0": "ACGTACGT",
                                                         "r2_1": "TTTTGGGG"}

        # Paired end to end: two mates in, one merged fragment out.
        q = "I" * 150
        (td / "p1.fastq").write_text(f"@f1\n{r1}\n+\n{q}\n@f2\n{r1}\n+\n{q}\n")
        (td / "p2.fastq").write_text(f"@f1\n{r2}\n+\n{q}\n@f2\n{r2}\n+\n{q}\n")
        a = argparse.Namespace(reads=[td / "p1.fastq", td / "p2.fastq"], glob=False,
                               max_reads=0, seed=0, fwd_primer=None, rev_primer=None,
                               no_trim_primers=True, primer_mismatches=3, paired=True,
                               min_overlap=20, output=td / "paired.fasta")
        run(a)
        got = dict(si.read_fasta(td / "paired.fasta"))
        assert list(got.values()) == [frag, frag], got                # merged, not 4 mates
    print("demo OK: fastq parsed, reservoir uniform, fasta written, pairs merged into "
          "whole fragments")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reads", nargs="*", type=Path, help="fastq(.gz) files (or globs with --glob)")
    ap.add_argument("--glob", action="store_true", help="treat the positional args as globs")
    ap.add_argument("--fwd-primer", default=si.DEFAULT_FWD_PRIMER)
    ap.add_argument("--rev-primer", default=si.DEFAULT_REV_PRIMER)
    ap.add_argument("--primer-mismatches", type=int, default=3)
    ap.add_argument("--no-trim-primers", action="store_true",
                    help="disable primer trimming (reads are already primer-trimmed)")
    ap.add_argument("--paired", action="store_true",
                    help="the two inputs are R1/R2: merge each pair into one query")
    ap.add_argument("--min-overlap", type=int, default=20,
                    help="--paired: shortest mate overlap accepted for a merge")
    ap.add_argument("--max-reads", type=int, default=0,
                    help="0 = all (fragments, not mates, under --paired)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", type=Path, help="output FASTA (mapseq input)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if a.demo:
        return demo()
    if not a.reads or a.output is None:
        ap.error("read files and --output are required (unless --demo)")
    run(a)


if __name__ == "__main__":
    main()
