#!/usr/bin/env python
"""Generate a tiny V4-amplicon fixture: refs.fasta + sample.fastq.gz.

Two genomes x two 16S copies, real 515F/806R primer flanks so `extract_v4` recovers
the payload, headers in the `genome|index|orig` convention. Reads are errored draws
from the amplicons at a planted composition, enough coverage for skiver to train.
"""
from __future__ import annotations

import gzip
import random
from pathlib import Path

random.seed(7)
HERE = Path(__file__).resolve().parent

FWD = "GTGYCAGCMGCCGCGGTAA"     # 515F
REV = "GGACTACNVGGGTWTCTAAT"    # 806R
IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "Y": "CT", "M": "AC",
         "N": "ACGT", "V": "ACG", "W": "AT", "R": "AG", "K": "GT",
         "S": "GC", "B": "CGT", "D": "AGT", "H": "ACT"}


def concrete(primer: str) -> str:
    return "".join(sorted(IUPAC[c])[0] for c in primer)


def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def rand_payload(n: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(n))


def mutate(s: str, k: int) -> str:
    """Return s with k random substitutions (distinct positions)."""
    b = list(s)
    for p in random.sample(range(len(s)), k):
        b[p] = random.choice([x for x in "ACGT" if x != b[p]])
    return "".join(b)


# Two genomes; each has a base V4 payload, its 16S copies differ by a couple of bases.
base = {"genomeA": rand_payload(230), "genomeB": rand_payload(230)}
# genomeB is a near-relative of genomeA (sub-species-like): 6% divergence.
base["genomeB"] = mutate(base["genomeA"], 14)

refs: list[tuple[str, str]] = []   # (header, full amplicon incl primers)
payloads: dict[str, str] = {}      # header -> V4 payload
for g, payload in base.items():
    for copy in range(2):
        v4 = mutate(payload, random.randint(0, 2))     # intra-genome copy variation
        header = f"{g}|{copy}|{g}_16S_{copy}"
        amp = concrete(FWD) + v4 + revcomp(concrete(REV))
        refs.append((header, amp))
        payloads[header] = v4

with open(HERE / "refs.fasta", "w") as fh:
    for h, seq in refs:
        fh.write(f">{h}\n{seq}\n")

# Reads: planted genome composition 0.7 / 0.3, ~1500 reads, 1% substitution error.
comp = {"genomeA": 0.7, "genomeB": 0.3}
headers_of = {g: [h for h, _ in refs if h.startswith(g + "|")] for g in comp}
n_reads = 1500
with gzip.open(HERE / "sample.fastq.gz", "wt") as fh:
    for i in range(n_reads):
        g = "genomeA" if random.random() < comp["genomeA"] else "genomeB"
        h = random.choice(headers_of[g])
        amp = dict(refs)[h]
        n_err = sum(1 for _ in amp if random.random() < 0.01)
        read = mutate(amp, min(n_err, len(amp)))
        qual = "I" * len(read)
        fh.write(f"@read{i}_{g}\n{read}\n+\n{qual}\n")

print(f"wrote {len(refs)} refs and {n_reads} reads to {HERE}")
