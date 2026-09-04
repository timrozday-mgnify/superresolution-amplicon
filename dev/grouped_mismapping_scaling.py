#!/usr/bin/env python
"""Time and measure the grouped mis-mapping backends over prefixes of an amplicon set.

Unlike ``benchmark_mismapping_scaling.py`` this starts from an *already extracted*
``amplicons.fasta``, so the curve is the matrix builder alone rather than the builder plus
a re-run of in-silico PCR at every point.

    dev/grouped_mismapping_scaling.py out/amplicons.fasta results/grouped_scaling
"""
from __future__ import annotations

import csv
import resource
import subprocess
import sys
import time
from pathlib import Path

BACKENDS = (("exact-hash", 0), ("kmer", 1), ("kmer", 2))
SIZES = (16_000, 32_000, 64_000, 125_000, 250_000, 500_000)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    """Return ``(header, sequence)`` for a two-line-per-record amplicon FASTA."""
    with path.open() as handle:
        return [(line[1:].strip(), next(handle).strip())
                for line in handle if line.startswith(">")]


def main() -> None:
    """Run every backend at every prefix size and write ``scaling.csv``."""
    source, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    records = read_pairs(source)
    rows: list[dict[str, object]] = []
    for size in [n for n in SIZES if n < len(records)] + [len(records)]:
        subset = outdir / f"prefix_{size}.fasta"
        subset.write_text("".join(f">{h}\n{s}\n" for h, s in records[:size]))
        for backend, tau in BACKENDS:
            # ru_maxrss over RUSAGE_CHILDREN is a running high-water mark, so each row is
            # an upper bound on its own run rather than that run's exact peak.
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            start = time.perf_counter()
            subprocess.run(
                [sys.executable, "bin/build_mismapping_align.py", "--backend", backend,
                 "--tau", str(tau), "--amplicons", str(subset),
                 "-o", str(outdir / "matrix.npz")],
                check=True, capture_output=True,
            )
            elapsed = time.perf_counter() - start
            peak = max(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, before)
            rows.append({
                "references": size, "backend": backend, "tau": tau,
                "seconds": round(elapsed, 2), "peak_child_rss_gb": round(peak / 1e9, 2),
                "matrix_mb": round((outdir / "matrix.npz").stat().st_size / 1e6, 1),
            })
            print(rows[-1], flush=True)
        subset.unlink()
    with (outdir / "scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
