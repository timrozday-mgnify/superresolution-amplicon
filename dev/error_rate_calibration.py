"""Calibrate the flat simulation error rates against a batch's own MAPseq output.

MAPseq reports mismatches and gaps against each read's hit. The observed per-base rates
are the target. The simulator's nominal rates map onto the rates MAPseq reports through
ratios measured on simulated reads that hit their own V4 group (reads relabelled onto a
neighbour carry extra mismatches that are not sequencing error):

    python dev/error_rate_calibration.py --amplicons amplicons.fasta \
        --obs-mseq S01.obs.mseq --sim-mseq sim.mseq --sim-sub 0.005 --sim-indel 0.0005

Observed rates include relabelled reads too, so the calibrated substitution rate is an
upper bound. ``--relabelled-share`` subtracts one mismatch per read for that share.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import subspecies_infer as si  # noqa: E402

COLUMNS = ["query", "dbhit", "bitscore", "identity", "matches", "mismatches", "gaps"]


def load(path: Path) -> pd.DataFrame:
    # Not pandas' comment="#": GTDB headers contain '#', which would truncate rows.
    rows = [line.rstrip("\n").split("\t")[:7] for line in open(path) if not line.startswith("#")]
    table = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("matches", "mismatches", "gaps"):
        table[column] = pd.to_numeric(table[column])
    return table


def per_base(table: pd.DataFrame) -> tuple[float, float, float]:
    """Mismatches and gaps per aligned base, and the median aligned length."""
    aligned = table.matches + table.mismatches
    return (table.mismatches.sum() / aligned.sum(), table.gaps.sum() / aligned.sum(),
            float(aligned.median()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicons", type=Path, required=True)
    ap.add_argument("--obs-mseq", type=Path, required=True)
    ap.add_argument("--sim-mseq", type=Path, required=True)
    ap.add_argument("--sim-sub", type=float, required=True, help="nominal sub rate simulated")
    ap.add_argument("--sim-indel", type=float, required=True,
                    help="nominal rate simulated for insertions and for deletions, each")
    ap.add_argument("--relabelled-share", type=float, default=0.0)
    a = ap.parse_args()

    group = {h: hashlib.sha256(s.encode()).hexdigest()[:16] for h, s in si.read_fasta(a.amplicons)}
    sim = load(a.sim_mseq)
    own = sim["query"].str.rsplit(":", n=1).str[0].map(group) == sim.dbhit.map(group)
    sim_mismatch, sim_gap, _ = per_base(sim[own])
    sub_ratio, gap_ratio = sim_mismatch / a.sim_sub, sim_gap / (2 * a.sim_indel)

    obs_mismatch, obs_gap, length = per_base(load(a.obs_mseq))
    corrected = max(obs_mismatch - a.relabelled_share / length, 0.0)
    print(f"simulated reads on their own group: {own.mean():.2f} of {len(sim)}; "
          f"reported/nominal: substitutions {sub_ratio:.3f}, indels {gap_ratio:.3f}")
    print(f"observed: {obs_mismatch:.5f} mismatches/base, {obs_gap:.6f} gaps/base "
          f"(median aligned length {length:.0f})")
    print(f"calibrated flat model: --sub-rate {corrected / sub_ratio:.5f} "
          f"--ins-rate {obs_gap / gap_ratio / 2:.6f} --del-rate {obs_gap / gap_ratio / 2:.6f}")


if __name__ == "__main__":
    main()
