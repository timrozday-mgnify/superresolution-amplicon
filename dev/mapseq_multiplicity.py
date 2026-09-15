#!/usr/bin/env python
"""Does MAPseq's leak onto a one-edit neighbour grow with that neighbour's copy count?

Evidence §3 of ``docs/gtdb_inference_recovery_plan.md``. Reads simulated from one GTDB V4
sequence A (flat 0.5% substitution, 0.05% insertion, 0.05% deletion) are mapped with the
pinned MAPseq against A, a one-substitution neighbour B repeated ``k`` times, and 200
random background sequences. The same database is built with the kmer backend (``tau=1``)
and the kernel's leak A -> B is set beside MAPseq's at each ``k``.

Phase 2 acceptance: the kernel's leak ratio ``k=100 / k=1`` lies in [0.5, 2]
(MAPseq: 1.37). Kernel version 1 weights B by ``k`` and fails it by ~60x.

    python dev/mapseq_multiplicity.py --amplicons reference/amplicons.fasta --out DIR

Needs docker. ~3 min.
"""
from __future__ import annotations

import argparse
import collections
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))

import build_mismapping_align as bma  # noqa: E402  (needs sys.path)
import subspecies_infer as si  # noqa: E402  (needs sys.path)

MAPSEQ = "quay.io/biocontainers/mapseq:2.1.1b--hc47f52e_1"


def mutate(s: str, rnd: random.Random, sub=0.005, ins=0.0005, dele=0.0005) -> str:
    out = []
    for ch in s:
        r = rnd.random()
        if r < dele:
            continue
        if r < dele + sub:
            ch = rnd.choice([b for b in "ACGT" if b != ch])
        out.append(ch)
        if rnd.random() < ins:
            out.append(rnd.choice("ACGT"))
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amplicons", type=Path, required=True, help="GTDB V4 amplicons.fasta")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decay", type=float, default=0.005895,
                        help="kernel distance decay c (SC2200627-SC3's 'auto' value)")
    parser.add_argument("--reads", type=int, default=20000)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--container", default=MAPSEQ)
    args = parser.parse_args()
    rnd = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    seqs, seen = [], set()
    for _, s in si.read_fasta(args.amplicons):
        if len(seqs) >= 400000:
            break
        if s not in seen and set(s) <= set("ACGT"):
            seen.add(s)
            seqs.append(s)
    A = seqs[1234]
    i = len(A) // 2
    B = A[:i] + {"A": "C", "C": "G", "G": "T", "T": "A"}[A[i]] + A[i + 1:]
    background = rnd.sample(seqs, 200)
    (args.out / "reads.fa").write_text(
        "".join(f">r{n}\n{mutate(A, rnd)}\n" for n in range(args.reads)))

    rows = []
    for k in args.copies:
        heads = (["A|0|A"] + [f"B{j}|0|B{j}" for j in range(k)]
                 + [f"bg{j}|0|bg{j}" for j in range(len(background))])
        db = args.out / f"db{k}.fasta"
        db.write_text("".join(f">{h}\n{s}\n" for h, s in zip(heads, [A] + [B] * k + background)))
        si.write_mapseq_tax(heads, args.out / f"db{k}.tax")
        mapseq = subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{args.out}:/w",
             "-w", "/w", args.container, "mapseq", "reads.fa", db.name, f"db{k}.tax"],
            capture_output=True, text=True)
        if mapseq.returncode:
            raise SystemExit(f"mapseq failed; --out must be a directory docker can mount "
                             f"(not /tmp on macOS):\n{mapseq.stderr[-1000:]}")
        hits = collections.Counter(line.split("\t")[1][0] for line in mapseq.stdout.splitlines()
                                   if line and line[0] != "#")
        refseqs, kernel, group, _, _ = bma.build_kmer_grouped(
            db, tau=1, max_ambiguous_bases=bma.DEFAULT_MAX_AMBIGUOUS_BASES,
            ambiguity_weight=bma.DEFAULT_AMBIGUITY_WEIGHT,
            max_postings=bma.DEFAULT_MAX_POSTINGS, distance_decay=args.decay)
        size = np.bincount(group)
        a, b = group[refseqs.index("A|0|A")], group[refseqs.index("B0|0|B0")]
        rows.append({"copies": k, "mapped": sum(hits.values()),
                     "mapseq_leak": hits["B"] / sum(hits.values()),
                     "kernel_leak": float(kernel[a, b] * size[b])})
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "mapseq_multiplicity.csv", index=False)
    first, last = df.iloc[0], df.iloc[-1]
    print(df.to_string(index=False))
    print(f"leak ratio k={int(last.copies)}/k={int(first.copies)}: "
          f"MAPseq {last.mapseq_leak / first.mapseq_leak:.2f}, "
          f"kernel {last.kernel_leak / first.kernel_leak:.2f} (Phase 2 accepts [0.5, 2])")


if __name__ == "__main__":
    main()
