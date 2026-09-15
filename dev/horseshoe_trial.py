"""Score V4-group fits against a known-truth benchmark sample (plan Phase 5 regularisation).

Truth genomes are carried into V4-group space through their amplicon copies (each copy an
equal share of its genome's reads), then each run's ``inferred_v4_groups.csv`` is scored:

    python dev/horseshoe_trial.py --truth S01.truth.tsv --amplicons references/amplicons \
        --alias bacteroides_uniformis_strain2=BU_JCM13286_NT5170 RUN_DIR... > out.csv
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import subspecies_infer as si  # noqa: E402

WATCH = {"v4g_2acb": "v4g_2acb4710c828d339", "v4g_a3e62": "v4g_a3e62c1c4b72ed9c"}


def truth_groups(truth: Path, amplicons: Path, alias: dict[str, str]) -> pd.Series:
    table = pd.read_csv(truth, sep="\t")
    mass: dict[str, float] = {}
    for genome, share in zip(table.genome_id, table.realized_rel_abundance):
        if share <= 0:
            continue
        name = alias.get(genome, genome)
        copies = [si.extract_v4(s, si.DEFAULT_FWD_PRIMER, si.DEFAULT_REV_PRIMER, 2)
                  for _, s in si.read_fasta(amplicons / f"{name}.amplicons.fasta")]
        copies = [c for c in copies if c]
        if not copies:
            raise SystemExit(f"{genome}: no amplifiable copy")
        for c in copies:
            gid = "v4g_" + hashlib.sha256(c.encode()).hexdigest()[:16]
            mass[gid] = mass.get(gid, 0.0) + share / len(copies)
    t = pd.Series(mass)
    return t / t.sum()


def score(profile: pd.Series, truth: pd.Series) -> dict:
    p = profile[profile > 0]
    p = p / p.sum()
    joined = pd.concat([p, truth], axis=1, keys=["p", "t"]).fillna(0.0)
    return {"tv_truth": 0.5 * (joined.p - joined.t).abs().sum(),
            "mass_outside_truth": float(p[~p.index.isin(truth.index)].sum()),
            **{k: float(p.get(v, 0.0)) for k, v in WATCH.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--amplicons", type=Path, required=True)
    ap.add_argument("--alias", action="append", default=[],
                    help="truth_genome=amplicon_file_stem (repeatable)")
    ap.add_argument("runs", type=Path, nargs="+")
    a = ap.parse_args()
    truth = truth_groups(a.truth, a.amplicons, dict(x.split("=", 1) for x in a.alias))
    rows = []
    for i, run in enumerate(a.runs):
        groups = pd.read_csv(run / "inferred_v4_groups.csv", index_col="v4_group_id")
        if i == 0:
            rows.append({"run": "raw MAPseq labels",
                         **score(groups.observed_rel_abundance, truth)})
        rows.append({"run": run.name, **score(groups.inferred_mean, truth)})
    pd.DataFrame(rows).to_csv(sys.stdout, index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
