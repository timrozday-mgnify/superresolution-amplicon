#!/usr/bin/env python
"""Which all-vs-all alignment backend should build ``M``?

The tie-cluster kernel needs, for every reference, the set of references within ``tau``
edit operations. Done naively that is n^2 alignments. Two ways to avoid most of them:

**A lossless k-mer filter** (what ``build_mismapping_align.py`` ships). By the q-gram
lemma a sequence of length L has ``L-k+1`` k-mers and one edit destroys at most ``k``, so
two sequences within ``e`` edits share at least ``L-k+1-k*e``. Anything sharing fewer
needs no alignment at all, and the survivors are aligned with edlib's ``k=`` bound so it
gives up early. Provably drops nothing.

**minimap2 via ``mappy``** — a real minimizer index. Heuristic, so this measures whether
it actually finds every true distance-0 pair, and what it costs.

Reference sets larger than the pipeline builds are synthesised from the real amplicons in
the same duplicate-heavy shape, since that is what makes the problem hard.

    python dev/alignment_backend_benchmark.py [--out CSV]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import subspecies_infer as si  # noqa: E402
import build_mismapping_align as bma  # noqa: E402
import error_rate_sensitivity as ers  # noqa: E402

SIZES = (500, 2000, 8000)
NAIVE_LIMIT = 2000          # above this the n^2 baseline is projected, not run
PRESETS = ("asm5", "asm10", "sr", "map-ont")


def synthesise(seqs: list[str], n: int, rng) -> list[str]:
    """A larger reference set shaped like a real one: clusters of near-identical copies."""
    out: list[str] = []
    while len(out) < n:
        base = seqs[rng.integers(0, len(seqs))]
        for _ in range(int(rng.integers(1, 7))):
            t = list(base)
            for q in rng.choice(len(t), size=int(rng.integers(0, 4)), replace=False):
                t[q] = "ACGT"[rng.integers(0, 4)]
            out.append("".join(t))
    return out[:n]


def mappy_clusters(seqs: list[str], preset: str, work: Path):
    """minimap2's candidate neighbours per sequence, and what it cost."""
    import mappy

    fa = work / f"{preset}.fasta"
    with open(fa, "w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">r{i}\n{s}\n")
    t = time.perf_counter()
    al = mappy.Aligner(str(fa), preset=preset, best_n=500)
    got = [{int(h.ctg[1:]) for h in al.map(s)} | {i} for i, s in enumerate(seqs)]
    return got, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=ers.DEFAULT_DB)
    ap.add_argument("--out", type=Path,
                    default=_REPO / "dev" / "alignment_backend_benchmark.csv")
    a = ap.parse_args()

    work = Path(tempfile.mkdtemp())
    try:
        si.stage_amplicons(SimpleNamespace(
            db_fasta=a.db, fwd_primer=si.DEFAULT_FWD_PRIMER,
            rev_primer=si.DEFAULT_REV_PRIMER, primer_mismatches=3, output_dir=work))
        real = [s for _, s in si.read_fasta(work / "amplicons.fasta")]

        # 1. Does minimap2 find every true distance-0 pair on the real reference set?
        exact = bma.pairwise_distances(real)
        truth = [set(np.flatnonzero(exact[i] == 0)) for i in range(len(real))]
        rows = []
        for preset in PRESETS:
            got, secs = mappy_clusters(real, preset, work)
            rows.append({
                "test": "mappy recall (real 81-ref set)", "backend": f"mappy {preset}",
                "n": len(real), "seconds": secs,
                "recall": float(np.mean([len(g & t) / len(t) for g, t in zip(got, truth)])),
                "precision": float(np.mean([len(g & t) / len(g) for g, t in zip(got, truth)])),
                "exact_clusters": int(sum(g == t for g, t in zip(got, truth))),
            })
            print(f"  {preset:8s} recall {rows[-1]['recall']:.3f} "
                  f"precision {rows[-1]['precision']:.3f} "
                  f"exact {rows[-1]['exact_clusters']}/{len(real)}  {secs:.3f}s")

        # 2. Losslessness of the shipped filter, then cost against n.
        for md in (0, 1, 5):
            assert (bma.pairwise_distances(real, max_distance=md)
                    == np.minimum(exact, md + 1)).all(), md
        print("  k-mer filter == exact distances (clipped) on the real set")

        rng = np.random.default_rng(0)
        print(f"\n{'n':>7} {'naive':>10} {'filtered':>10} {'speedup':>8} {'mappy asm5':>11}")
        for n in SIZES:
            ss = synthesise(real, n, rng)
            if n <= NAIVE_LIMIT:
                t = time.perf_counter(); bma.pairwise_distances(ss)
                naive = time.perf_counter() - t
            else:
                naive = float("nan")
            t = time.perf_counter(); bma.pairwise_distances(ss, max_distance=0)
            filt = time.perf_counter() - t
            _, mp = mappy_clusters(ss, "asm5", work)
            rows.append({"test": "cost vs n", "backend": "edlib all-pairs", "n": n,
                         "seconds": naive})
            rows.append({"test": "cost vs n", "backend": "k-mer filter + edlib", "n": n,
                         "seconds": filt})
            rows.append({"test": "cost vs n", "backend": "mappy asm5 (candidates only)",
                         "n": n, "seconds": mp})
            print(f"{n:>7} {naive:>10.2f} {filt:>10.2f} "
                  f"{naive / filt if naive == naive else float('nan'):>8.1f} {mp:>10.2f}s")

        df = pd.DataFrame(rows)
        df.to_csv(a.out, index=False)
        print(f"\n-> {a.out}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
