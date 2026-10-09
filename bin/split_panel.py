#!/usr/bin/env python3
"""Split a panel FASTA into one FASTA per genome, ``<out>/<genome>.fasta``.

    split_panel.py custom.fasta -o genomes/

A genome is the header's id before the first ``|`` (the ``genome|index|orig`` convention).
Each file holds that genome's lines byte for byte, in panel order, so its sha256 -- what a
one-genome bundle's provenance ``panel`` records -- can be recomputed from the panel FASTA
alone. ``build_panel_kernel.py combine`` does exactly that to find each genome's bundle.

Stdlib only, so it runs on a cluster head node with no environment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def genome_chunks(panel: Path) -> dict[str, bytes]:
    """Genome id -> its header and sequence lines, each ending in a newline."""
    lines = panel.read_bytes().split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    chunks: dict[str, list[bytes]] = defaultdict(list)
    genome = None
    for line in lines:
        if line.startswith(b">"):
            genome = line[1:].decode().split()[0].split("|", 1)[0]
            if genome in ("", ".", "..") or "/" in genome:
                raise SystemExit(f"{panel}: genome id {genome!r} cannot name a file")
        elif genome is None:
            raise SystemExit(f"{panel}: sequence before the first header")
        chunks[genome].append(line + b"\n")
    if not chunks:
        raise SystemExit(f"{panel}: no records")
    return {g: b"".join(c) for g, c in chunks.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panel", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    chunks = genome_chunks(a.panel)
    for genome, chunk in chunks.items():
        (a.out / f"{genome}.fasta").write_bytes(chunk)
    print(f"{len(chunks)} genome(s) from {a.panel} -> {a.out}")


if __name__ == "__main__":
    main()
