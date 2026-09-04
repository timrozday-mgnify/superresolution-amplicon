#!/usr/bin/env python
"""How much should an ambiguous reference be demoted inside its tie cluster?

``dev/alignment_mismapping.md`` found that mapseq is not ambiguity-agnostic: an N-bearing
reference receives a fraction of the reads its clean cluster partners get, because the N
scores as a mismatch and the clean duplicate wins. ``build_mismapping_align.py`` can model
that by down-weighting a cluster member as ``w ** (its ambiguous positions)``.

The question is whether ``w`` is a portable constant. For a range of injected-ambiguity
configurations this measures the penalty directly (incoming mass of N-bearing references
relative to their clean cluster partners) and, independently, fits ``w`` by minimising
``||M_align - M_sim||_F``. If the model is right the two should agree.

Needs docker (the mapseq biocontainer). ~5 min.

    python dev/ambiguity_weight_sweep.py [--out CSV]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))
sys.path.insert(0, str(_REPO / "vendor" / "skiver" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import subspecies_infer as si  # noqa: E402
import build_mismapping_align as bma  # noqa: E402
import error_rate_sensitivity as ers  # noqa: E402

WEIGHTS = (1.0, 0.5, 0.3, 0.19, 0.1, 0.0)
# Injection rate and draw. Same rate / different seed is the fit-vs-holdout pair; the low
# rates check that a weight fitted at high ambiguity does not hurt at low ambiguity.
CONFIGS = ((0.25, 7), (0.25, 23), (0.25, 101), (0.10, 23), (0.05, 5))


def run_config(rate: float, seed: int, work: Path) -> dict:
    """Inject Ns into the DB (not the reads), measure M with mapseq, sweep ``w``."""
    si.stage_amplicons(SimpleNamespace(
        db_fasta=ers.DEFAULT_DB, fwd_primer=si.DEFAULT_FWD_PRIMER,
        rev_primer=si.DEFAULT_REV_PRIMER, primer_mismatches=3, output_dir=work))
    fasta, tax = work / "amplicons.fasta", work / "amplicons.tax"
    recs = si.read_fasta(fasta)
    refseqs, amps = [h for h, _ in recs], [s for _, s in recs]

    rng = np.random.default_rng(seed)
    hit = rng.random(len(amps)) < rate
    db = list(amps)
    for i in np.flatnonzero(hit):
        p = int(rng.integers(0, len(db[i])))
        db[i] = db[i][:p] + "N" + db[i][p + 1:]
    with open(fasta, "w") as fh:                       # the DB carries the ambiguity ...
        for h, s in zip(refseqs, db):
            fh.write(f">{h}\n{s}\n")
    ers.mapseq(fasta, fasta, tax, work / "cluster.mseq")
    M = ers.build_M("0.005", refseqs, amps, work, fasta, tax, seed=1)   # ... reads do not

    d, d_clean = bma.pairwise_distances(db), bma.pairwise_distances(amps)

    # The penalty, measured: incoming mass of N-bearing references relative to their clean
    # partners, over clusters (of the un-N'd sequences) that contain both.
    col = M.sum(axis=0)
    mixed = [(col[cl][hit[cl]].mean(), col[cl][~hit[cl]].mean())
             for cl in (np.flatnonzero(d_clean[i] == 0) for i in range(len(amps)))
             if len(cl) > 1 and hit[cl].any() and (~hit[cl]).any()]
    mixed = np.array(mixed)
    ratio = float(mixed[:, 0].mean() / mixed[:, 1].mean()) if len(mixed) else float("nan")

    fs = {w: float(np.linalg.norm(
        bma.tie_cluster_matrix(d, 0, bma.ambiguity_weights(db, w)) - M)) for w in WEIGHTS}
    return {"inject_rate": rate, "inject_seed": seed, "n_ambiguous_refs": int(hit.sum()),
            "measured_mass_ratio": ratio, "best_w": min(fs, key=fs.get),
            **{f"frobenius_w{w}": v for w, v in fs.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=_REPO / "dev" / "ambiguity_weight_sweep.csv")
    a = ap.parse_args()

    rows = []
    for n, (rate, seed) in enumerate(CONFIGS):
        work = _REPO / "dev" / f"_ambiguity_work{n}"    # a fresh dir per config: mapseq
        shutil.rmtree(work, ignore_errors=True)         # caches .mscluster next to the DB
        work.mkdir(parents=True)
        try:
            rows.append(run_config(rate, seed, work))
        finally:
            shutil.rmtree(work, ignore_errors=True)
        r = rows[-1]
        print(f"  rate={rate} seed={seed}: {r['n_ambiguous_refs']:2d} N-bearing refs, "
              f"measured penalty {r['measured_mass_ratio']:.2f}, best w={r['best_w']}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
