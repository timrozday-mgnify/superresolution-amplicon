#!/usr/bin/env python
"""Convert observed fastq reads to the FASTA mapseq expects, optionally primer-trimmed.

mapseq takes FASTA only. Trimming the primers off (``--fwd-primer``/``--rev-primer``,
reusing ``subspecies_infer.trim_read_primers``) puts the reads in the same coordinate
space as the primer-trimmed reference amplicons and as the simulated reads; reads with no
primer found are passed through unchanged. ``--max-reads`` subsamples reservoir-style so
depth is bounded without holding the whole file in memory.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)

log = logging.getLogger("reads_to_fasta")


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
    print(f"reads_to_fasta: {n} reads -> {a.output}")


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

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fq = td / "r.fastq"
        fq.write_text("@r1 extra\nACGTACGT\n+\nIIIIIIII\n@r2\nTTTTGGGG\n+\nIIIIIIII\n")
        assert list(iter_fastq(fq)) == [("r1", "ACGTACGT"), ("r2", "TTTTGGGG")]
        a = argparse.Namespace(reads=[fq], glob=False, max_reads=0, seed=0,
                               fwd_primer=None, rev_primer=None, no_trim_primers=True,
                               primer_mismatches=3, output=td / "out.fasta")
        run(a)
        assert dict(si.read_fasta(td / "out.fasta")) == {"r1_0": "ACGTACGT",
                                                         "r2_1": "TTTTGGGG"}
    print("demo OK: fastq parsed, reservoir uniform, fasta written")


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
    ap.add_argument("--max-reads", type=int, default=0, help="0 = all reads")
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
