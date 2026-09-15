"""Panel reinterpretation of generic-database MAPseq labels: data and baselines (plan Phase 0).

    python dev/panel_reinterpretation.py prepare --panel-amplicons references/amplicons \
        --db-amplicons work/phase5/amplicons.fasta --obs-mseq S01.obs.mseq \
        --alias bacteroides_uniformis_strain2=BU_JCM13286_NT5170 -o PREP
    mapseq PREP/sources.fasta amplicons.fasta amplicons.tax > PREP/home.mseq
    python dev/panel_reinterpretation.py baseline --prepared PREP \
        --db-amplicons amplicons.fasta --custom-profile S01...simulate.p01.sr_profile.tsv \
        --gtdb-run work/phase5_truth/hs_sub0.0031_indel0.000024_n5000
    python dev/panel_reinterpretation.py score --prepared PREP --truth S01...truth.tsv \
        PREP/B*.csv > scores.csv

``prepare`` writes the panel's distinct V4 sources (``v4g_<sha16>``), the genome -> source
copy weights and the observed S01 label counts. ``baseline`` writes genome compositions
(``genome_id,rel_abundance``, with a ``background`` row for mass the panel cannot explain):

- B0: the custom-database run, which mapped against the panel itself.
- B1: no confusion model. Each source emits only its home label (MAPseq of its own
  sequence) and EM fits the genomes to the observed label counts.
- B2: the best GTDB-space ``v4_group`` fit, projected onto panel genomes by the same EM
  with ``home = identity``. ``raw`` projects the raw MAPseq labels the same way.
"""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import subspecies_infer as si  # noqa: E402
from horseshoe_trial import WATCH  # noqa: E402

UNEXPLAINED = "U"
BACKGROUND = "background"


def v4g(seq: str) -> str:
    return "v4g_" + hashlib.sha256(seq.encode()).hexdigest()[:16]


def db_labels(db_amplicons: Path) -> dict[str, str]:
    """Database header -> exact-sequence group id."""
    return {h: v4g(s) for h, s in si.read_fasta(db_amplicons)}


def label_counts(mseq: Path, label: dict[str, str]) -> pd.Series:
    return pd.Series(Counter(label[hit] for _, hit in si.iter_mseq(mseq)), dtype=float)


def em(A: np.ndarray, y: np.ndarray, iters: int = 20000, tol: float = 1e-12) -> np.ndarray:
    """Maximum-likelihood mixture weights for multinomial counts ``y`` over rows of ``A``."""
    theta = np.full(A.shape[0], 1.0 / A.shape[0])
    for _ in range(iters):
        pred = theta @ A
        resp = np.divide(y, pred, out=np.zeros_like(y), where=pred > 0)
        new = theta * (A @ resp) / y.sum()
        if np.abs(new - theta).max() < tol:
            return new
        theta = new
    return theta


def project(translation: pd.DataFrame, home: dict[str, str], y: pd.Series) -> pd.Series:
    """EM genome composition when each source emits only ``home[source]``.

    Counts on labels no source reaches go to one label ``U`` owned by a background source.
    """
    tr = translation.assign(label=translation.source.map(home))
    A = tr.pivot_table(index="genome_id", columns="label", values="weight",
                       aggfunc="sum", fill_value=0.0)
    A = A.div(A.sum(axis=1), axis=0)
    A[UNEXPLAINED] = 0.0
    A.loc[BACKGROUND] = 0.0
    A.loc[BACKGROUND, UNEXPLAINED] = 1.0
    yy = y.reindex(A.columns, fill_value=0.0)
    yy[UNEXPLAINED] = y[~y.index.isin(A.columns)].sum()
    return pd.Series(em(A.to_numpy(), yy.to_numpy(dtype=float)), index=A.index)


def prepare(a) -> None:
    stem_to_genome = {v: k for k, v in (x.split("=", 1) for x in a.alias)}
    rows = []
    for path in sorted(a.panel_amplicons.glob("*.amplicons.fasta")):
        genome = stem_to_genome.get(path.name.removesuffix(".amplicons.fasta"),
                                    path.name.removesuffix(".amplicons.fasta"))
        copies = [si.extract_v4(s, si.DEFAULT_FWD_PRIMER, si.DEFAULT_REV_PRIMER, 2)
                  for _, s in si.read_fasta(path)]
        copies = [c for c in copies if c]
        if not copies:
            raise SystemExit(f"{genome}: no amplifiable copy")
        for seq, n in Counter(copies).items():
            rows.append({"genome_id": genome, "source": v4g(seq), "weight": n / len(copies),
                         "seq": seq})
    tr = pd.DataFrame(rows)
    label = db_labels(a.db_amplicons)
    in_db = set(label.values())
    a.out.mkdir(parents=True, exist_ok=True)
    tr[["genome_id", "source", "weight"]].to_csv(a.out / "panel_translation.tsv", sep="\t",
                                                 index=False)
    sources = tr.groupby("source").agg(seq=("seq", "first"),
                                       genomes=("genome_id", ";".join)).reset_index()
    sources["in_db"] = sources.source.isin(in_db)
    sources[["source", "genomes", "in_db"]].to_csv(a.out / "panel_sources.tsv", sep="\t",
                                                   index=False)
    with open(a.out / "sources.fasta", "w") as fh:
        fh.writelines(f">{s}\n{q}\n" for s, q in zip(sources.source, sources.seq))
    obs = label_counts(a.obs_mseq, label)
    obs.rename_axis("label").rename("count").to_csv(a.out / "obs_labels.tsv", sep="\t")
    shared = sources.genomes.str.contains(";").sum()
    print(f"{tr.genome_id.nunique()} genomes, {len(sources)} distinct sources, "
          f"{sources.in_db.sum()} in DB, {shared} shared; {int(obs.sum())} observed reads "
          f"on {len(obs)} labels", file=sys.stderr)


def baseline(a) -> None:
    tr = pd.read_csv(a.prepared / "panel_translation.tsv", sep="\t")
    obs = pd.read_csv(a.prepared / "obs_labels.tsv", sep="\t", index_col="label")["count"]
    label = db_labels(a.db_amplicons)
    home = {q: label[hit] for q, hit in si.iter_mseq(a.prepared / "home.mseq")}
    missing = set(tr.source) - set(home)
    if missing:
        raise SystemExit(f"no home label for {sorted(missing)}")
    pd.Series(home).rename_axis("source").rename("home").to_csv(
        a.prepared / "home_labels.tsv", sep="\t")
    identity = {s: s for s in tr.source}
    groups = pd.read_csv(a.gtdb_run / "inferred_v4_groups.csv", index_col="v4_group_id")
    b0 = pd.read_csv(a.custom_profile, sep="\t", index_col="genome_id").predicted_rel_abundance
    out = {"B0_custom_db": b0,
           "B1_home_only": project(tr, home, obs),
           "B2_gtdb_space": project(tr, identity, groups.inferred_mean),
           "raw_labels": project(tr, identity, obs)}
    for name, comp in out.items():
        comp.rename_axis("genome_id").rename("rel_abundance").to_csv(
            a.prepared / f"{name}.csv")
    moved = sum(h != s for s, h in home.items())
    print(f"{moved}/{len(home)} sources have a home label other than themselves",
          file=sys.stderr)


def score(a) -> None:
    tr = pd.read_csv(a.prepared / "panel_translation.tsv", sep="\t")
    truth = pd.read_csv(a.truth, sep="\t", index_col="genome_id").realized_rel_abundance
    truth = truth / truth.sum()
    bu = ["bacteroides_uniformis", "bacteroides_uniformis_strain2"]
    watch = {k: tr.genome_id[tr.source == v].unique() for k, v in WATCH.items()}
    rows = []
    for path in a.compositions:
        p = pd.read_csv(path, index_col="genome_id").rel_abundance
        unexplained = float(p.get(BACKGROUND, 0.0)) / p.sum()
        q = p.drop(BACKGROUND, errors="ignore")
        q = q / q.sum()
        j = pd.concat([q, truth], axis=1, keys=["p", "t"]).fillna(0.0)
        split = lambda s: s[bu[1]] / s[bu].sum() if s[bu].sum() > 0 else np.nan  # noqa: E731
        rows.append({"run": path.stem,
                     "tv_truth": 0.5 * (j.p - j.t).abs().sum(),
                     "mass_absent_from_truth": j.p[j.t == 0].sum(),
                     "bacteroides_uniformis": j.p[bu[0]],
                     "bu_split_error": abs(split(j.p) - split(j.t)),
                     "unexplained_fraction": unexplained,
                     **{f"recovered_{k}": j.p[g].sum() / j.t[g].sum() for k, g in watch.items()},
                     **{f"abs_err_{g}": abs(j.p[g] - j.t[g]) for g in j.index}})
    pd.DataFrame(rows).to_csv(sys.stdout, index=False, float_format="%.4f")


def selfcheck() -> None:
    A = np.array([[1.0, 0, 0], [0.5, 0.5, 0], [0, 0, 1.0]])
    theta = np.array([0.2, 0.5, 0.3])
    assert np.allclose(em(A, theta @ A * 1e6), theta, atol=1e-6)
    tr = pd.DataFrame({"genome_id": ["g1", "g2", "g2"], "source": ["s1", "s2", "s3"],
                       "weight": [1.0, 0.5, 0.5]})
    # s3's home is s1: g2 is only identifiable through s2, and the stray label goes to U.
    p = project(tr, {"s1": "s1", "s2": "s2", "s3": "s1"},
                pd.Series({"s1": 450.0, "s2": 250.0, "x": 300.0}))
    assert np.allclose(p[["g1", "g2", BACKGROUND]], [0.2, 0.5, 0.3], atol=1e-6), p
    print("selfcheck ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--panel-amplicons", type=Path, required=True)
    p.add_argument("--db-amplicons", type=Path, required=True)
    p.add_argument("--obs-mseq", type=Path, required=True)
    p.add_argument("--alias", action="append", default=[],
                   help="genome_id=amplicon_file_stem (repeatable)")
    p.add_argument("-o", "--out", type=Path, required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--prepared", type=Path, required=True)
    b.add_argument("--db-amplicons", type=Path, required=True)
    b.add_argument("--custom-profile", type=Path, required=True)
    b.add_argument("--gtdb-run", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--prepared", type=Path, required=True)
    s.add_argument("--truth", type=Path, required=True)
    s.add_argument("compositions", type=Path, nargs="+")
    sub.add_parser("selfcheck")
    a = ap.parse_args()
    {"prepare": prepare, "baseline": baseline, "score": score,
     "selfcheck": lambda _: selfcheck()}[a.cmd](a)


if __name__ == "__main__":
    main()
