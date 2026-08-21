#!/usr/bin/env python3
"""Create a deterministic, shuffled reservoir sample of FASTQ records."""

from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path
from collections.abc import Iterable, Iterator


FastqRecord = tuple[str, str, str, str]


def _open_fastq(path: Path):
    """Return a text handle for a plain or gzipped FASTQ file."""
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def read_records(paths: Iterable[Path]) -> Iterator[FastqRecord]:
    """Yield validated FASTQ records from paths."""
    for path in paths:
        with _open_fastq(path) as handle:
            while True:
                record = tuple(handle.readline().rstrip("\n") for _ in range(4))
                if not record[0]:
                    break
                if len(record) != 4 or not record[2].startswith("+"):
                    raise ValueError(f"Invalid FASTQ record in {path}")
                yield record  # type: ignore[misc]


def reservoir_sample(
    records: Iterable[FastqRecord], maximum: int, rng: random.Random
) -> list[FastqRecord]:
    """Return a uniform sample of up to maximum records, or all records when zero."""
    selected: list[FastqRecord] = []
    if maximum == 0:
        selected.extend(records)
    else:
        for count, record in enumerate(records, start=1):
            if len(selected) < maximum:
                selected.append(record)
                continue
            replacement = rng.randrange(count)
            if replacement < maximum:
                selected[replacement] = record
    rng.shuffle(selected)
    return selected


def main() -> None:
    """Parse arguments, sample FASTQ records, and write a gzipped pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--max-reads", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_reads < 0:
        parser.error("--max-reads must be zero or positive")

    selected = reservoir_sample(read_records(args.input), args.max_reads, random.Random(args.seed))
    with gzip.open(args.output, "wt") as handle:
        for record in selected:
            handle.write("\n".join(record) + "\n")


if __name__ == "__main__":
    main()
