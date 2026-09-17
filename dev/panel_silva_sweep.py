"""Panel reinterpretation against SILVA 138.2 NR99 V4, genome and taxon panels (Phase V of
the benchmark repo's docs/silva_taxon_panel_plan.md). Driven by dev/panel_silva_sweep.sh.

    python dev/panel_silva_sweep.py prepare --silva work/silva
    python dev/panel_silva_sweep.py run --silva work/silva --jobs 6
    python dev/panel_silva_sweep.py score --silva work/silva --gtdb-sweep work/panel_sweep \
        > dev/panel_silva_sweep.csv

``prepare`` writes two prepared panel directories against the SILVA amplicons:

- ``genome/``: the 22-genome 20HM panel of the GTDB sweep (its translation and sources).
- ``taxa/``: the *B. uniformis* pair as genomes and the other 20 genomes as their 17 SILVA
  genera (``taxa/panel_taxa.tsv``). ``taxa/sim_sources.fasta`` holds the taxon sources the
  GTDB sweep never simulated.
- ``species/``: the pair as genomes and the other 20 genomes as their SILVA species, over
  ``species/silva_nr99_species.tax`` (a species rank from SILVA's organism names). Its sources
  are a subset of ``taxa/``'s, so they reuse its simulated reads.

``run`` fits every arm on every ``obs/S*.obs.mseq`` with the GTDB sweep's inference settings.
``score`` compares each fit, and the GTDB sweep's fits of the same arms, with truth counted
from the read names (``<chunk>:<genome>:...``; merged reads only, so it differs slightly from
the benchmark's BAM truth, and the GTDB fits are rescored with it). Genome arms are scored
per genome and per taxon entry; taxon arms per entry only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mapseq_database as bmd  # noqa: E402
import build_panel_kernel as bpk  # noqa: E402
import infer_composition as ic  # noqa: E402
import subspecies_infer as si  # noqa: E402
from panel_reinterpretation_sweep import BU_PAIR, INFER, PRIORS, scores  # noqa: E402

GTDB_PANEL = Path("work/panel_kernel")
LACHNO = "Bacteria;Bacillota;Clostridia;Lachnospirales;Lachnospiraceae;"
# entry -> (SILVA taxon, panel genomes it stands for). Genera are the majority SILVA genus of
# the sequences sharing each genome's V4 source; bare names are used where unique.
TAXA = {
    "bacteroides": ("Bacteria;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides",
                    ["bacteroides_fragilis", "bacteroides_thetaiotaomicron",
                     "bacteroides_vulgatus"]),
    "enterocloster": (LACHNO + "Enterocloster", ["clostridium_bolteae"]),
    "thomasclavelia": ("Thomasclavelia", ["clostridium_ramosum"]),
    "lachnoclostridium": (LACHNO + "Lachnoclostridium", ["clostridium_saccharolyticum"]),
    "collinsella": ("Collinsella", ["collinsella_aerofaciens"]),
    "coprococcus": (LACHNO + "Coprococcus", ["coprococcus_comes"]),
    "dorea": (LACHNO + "Dorea", ["dorea_formicigenerans"]),
    "eggerthella": ("Eggerthella", ["eggerthella_lenta"]),
    "agathobacter": (LACHNO + "Agathobacter", ["eubacterium_rectale"]),
    "fusobacterium": ("Fusobacterium", ["fusobacterium_nucleatum"]),
    "methanobrevibacter": ("Methanobrevibacter", ["methanobrevibacter_smithii"]),
    "parabacteroides": ("Parabacteroides", ["parabacteroides_merdae"]),
    "roseburia": (LACHNO + "Roseburia", ["roseburia_intestinalis"]),
    "ruminococcus_gnavus_group": (LACHNO + "[Ruminococcus] gnavus group",
                                  ["ruminococcus_gnavus"]),
    "salinibacter": ("Salinibacter", ["salinibacter_ruber"]),
    "streptococcus": ("Bacteria;Bacillota;Bacilli;Lactobacillales;Streptococcaceae;Streptococcus",
                      ["streptococcus_parasanguinis", "streptococcus_salivarius"]),
    "veillonella": ("Veillonella", ["veillonella_parvula"]),
}
ENTRY_OF_GENOME = {g: e for e, (_, gs) in TAXA.items() for g in gs}
# Species panel: the same 20 genomes as their SILVA species (entry id = genome id, so it scores
# per genome). SILVA has no species rank; species/silva_nr99_species.tax appends one to
# genus-level Bacteria/Archaea lineages (build_mapseq_database.parse_silva_header's rule).
SPECIES = {g: s for g, s in {
    "bacteroides_fragilis": "Bacteroides fragilis",
    "bacteroides_thetaiotaomicron": "Bacteroides thetaiotaomicron",
    "bacteroides_vulgatus": "Bacteroides vulgatus",
    "clostridium_bolteae": "Enterocloster bolteae",
    "clostridium_ramosum": "Thomasclavelia ramosum",
    "clostridium_saccharolyticum": "Lachnoclostridium saccharolyticum",
    "collinsella_aerofaciens": "Collinsella aerofaciens",
    "coprococcus_comes": "Coprococcus comes",
    "dorea_formicigenerans": "Dorea formicigenerans",
    "eggerthella_lenta": "Eggerthella lenta",
    "eubacterium_rectale": "Agathobacter rectale",
    "fusobacterium_nucleatum": "Fusobacterium nucleatum",
    "methanobrevibacter_smithii": "Methanobrevibacter smithii",
    "parabacteroides_merdae": "Parabacteroides merdae",
    "roseburia_intestinalis": "Roseburia intestinalis",
    "ruminococcus_gnavus": "[Ruminococcus] gnavus group gnavus",
    "salinibacter_ruber": "Salinibacter ruber",
    "streptococcus_parasanguinis": "Streptococcus parasanguinis",
    "streptococcus_salivarius": "Streptococcus salivarius",
    "veillonella_parvula": "Veillonella parvula",
}.items()}

# arm -> (panel dir, kernel, extra flags). Kernels live at <silva>/<panel>/<kernel>.npz.
ARMS = {}
for prior, flags in PRIORS.items():
    ARMS[f"genome_sim_trained{prior}"] = ("genome", "trained_s0", flags)
    ARMS[f"genome_sim_calibrated_flat{prior}"] = ("genome", "cal", flags)
    # Diagnostic: only simulated reads as long as their source. The trained reads carry
    # untrimmed 5' insertions the primer-trimmed observed reads lack, and on SILVA's
    # one-base near-ties MAPseq places such reads differently.
    ARMS[f"genome_sim_trained_lenmatch{prior}"] = ("genome", "trained_s0_lenmatch", flags)
    # The fix emulated: the trained reads with only their read-start insertions removed,
    # which is what simulating from primer-flanked amplicons and trimming does.
    ARMS[f"genome_sim_trained_starttrim{prior}"] = ("genome", "trained_s0_starttrim", flags)
    ARMS[f"genome_home_labels_only{prior}"] = ("genome", "trained_s0", ["--no-mismapping"] + flags)
    ARMS[f"taxa_sim_calibrated_flat{prior}"] = ("taxa", "cal", flags)
    ARMS[f"taxa_home_labels_only{prior}"] = ("taxa", "cal", ["--no-mismapping"] + flags)
    ARMS[f"species_sim_calibrated_flat{prior}"] = ("species", "cal", flags)
    ARMS[f"species_home_labels_only{prior}"] = ("species", "cal", ["--no-mismapping"] + flags)
# The GTDB sweep's fits of the genome arms, for the same-truth comparison.
GTDB_ARMS = {f"gtdb_{kind}{prior}": f"{kind}{prior}"
             for kind in ["sim_trained", "sim_calibrated_flat", "home_labels_only"]
             for prior in PRIORS}


def species_taxonomy(silva_fasta: Path, out: Path) -> None:
    """Unpadded ``.tax`` over the SILVA FASTA, with build_mapseq_database's species rank."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as tax:
        for header, _ in bmd.iter_fasta(silva_fasta):
            accession, lineage = bmd.parse_silva_header(header)
            tax.write(f"{accession}|0|{accession}\t{';'.join(lineage)}\n")


def taxon_panel(out: Path, entries: dict[str, str], lineage_of: dict[str, str], db, bu,
                sequence: dict[str, str], simulated: set[str], write) -> None:
    """A prepared panel of the *B. uniformis* genomes plus taxon entries (id -> taxon)."""
    headers, label_ids, label_of_ref, label_seqs = db
    members = bpk.taxon_members(list(entries.items()), headers, label_ids, label_of_ref,
                                lineage_of,
                                set(bu.source) | {g for g, s in zip(label_ids, label_seqs)
                                                  if set(s) - set("ACGT")},
                                max_sources=10**9)
    sequence.update(zip(label_ids, label_seqs))
    rows = [{"genome_id": f"{e}{bpk.ENTRY_SEP}{g}", "source": g, "weight": 1.0}
            for e, groups in members.items() for g in groups]
    table = pd.concat([bu, pd.DataFrame(rows)], ignore_index=True)
    write(out, table)
    pd.DataFrame({"id": list(entries), "taxon": list(entries.values())}).to_csv(
        out / "panel_taxa.tsv", sep="\t", index=False)
    pd.DataFrame([{"entry": e, "n_groups": len(g)} for e, g in members.items()]).to_csv(
        out / "entry_groups.tsv", sep="\t", index=False)
    new = sorted(set(table.source) - simulated)
    with open(out / "sim_sources.fasta", "w") as fh:
        fh.writelines(f">{s}\n{sequence[s]}\n" for s in new)
    print(f"{out}: {len(new)} taxon sources to simulate", file=sys.stderr)


def prepare(a) -> None:
    headers, label_ids, label_of_ref, label_seqs = bpk.db_groups(a.silva / "amp" / "amplicons.fasta")
    in_db = set(label_ids)
    translation = pd.read_csv(GTDB_PANEL / "panel_translation.tsv", sep="\t")
    sequence = dict(si.read_fasta(GTDB_PANEL / "sources.fasta"))

    def write(out: Path, table: pd.DataFrame) -> None:
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "panel_translation.tsv", sep="\t", index=False)
        sources = (table.groupby("source").genome_id.agg(";".join).rename("genomes")
                   .reset_index())
        sources["in_db"] = sources.source.isin(in_db)
        sources.to_csv(out / "sources.tsv", sep="\t", index=False)
        with open(out / "sources.fasta", "w") as fh:
            fh.writelines(f">{s}\n{sequence[s]}\n" for s in sources.source)
        print(f"{out}: {table.genome_id.nunique()} members, {len(sources)} sources, "
              f"{sources.in_db.sum()} in SILVA", file=sys.stderr)

    write(a.silva / "genome", translation)

    bu = translation[translation.genome_id.isin(BU_PAIR)]
    db = (headers, label_ids, label_of_ref, label_seqs)
    simulated = set(translation.source)
    if not (a.silva / "taxa" / "sources.tsv").exists():
        taxon_panel(a.silva / "taxa", {e: t for e, (t, _) in TAXA.items()},
                    ic._read_taxonomy(a.silva / "db" / "silva_nr99.tax"), db, bu, sequence,
                    simulated, write)
    # Species sources already simulated for the genus panel reuse those reads.
    simulated |= set(pd.read_csv(a.silva / "taxa" / "sources.tsv", sep="\t").source)
    species_tax = a.silva / "species" / "silva_nr99_species.tax"
    if not species_tax.exists():
        species_taxonomy(a.silva / "SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz", species_tax)
    taxon_panel(a.silva / "species", SPECIES, ic._read_taxonomy(species_tax), db, bu, sequence,
                simulated, write)


def one(obs: Path, arm: str, a) -> str:
    sample = obs.name.removesuffix(".obs.mseq")
    panel, kernel, extra = ARMS[arm]
    out = a.silva / "sweep" / sample / arm
    if (out / "fit.json").exists():
        return f"{sample} {arm}: done"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    common = ["--obs-mseq", str(obs), "--amplicon-dir", str(a.silva / panel),
              "--mismapping-matrix", str(a.silva / panel / f"{kernel}.npz")]
    subprocess.run([sys.executable, BIN / "infer_composition.py", *common, *INFER, *extra,
                    "--sample-id", sample, "-o", str(out)], check=True, env=env,
                   capture_output=True)
    subprocess.run([sys.executable, BIN / "check_composition_fit.py", *common,
                    "--composition", str(out / "inferred_composition.csv"),
                    "--posterior-draws", str(out / "posterior_draws.npz"),
                    "-o", str(out / "fit.json")], check=True, env=env, capture_output=True)
    draws = out / "posterior_draws.npz"
    (out / "fitted_s.json").write_text(json.dumps(float(np.load(draws)["s"].mean())))
    if panel in ("taxa", "species"):
        draws.unlink()          # ~70 MB each at 9k members; the scores need only mean s
    return f"{sample} {arm}: ok"


def run(a) -> None:
    jobs = [(obs, arm) for obs in sorted((a.silva / "obs").glob("S*.obs.mseq"))
            for arm in ARMS if a.arms is None or arm in a.arms]
    with ThreadPoolExecutor(a.jobs) as pool:
        for message in pool.map(lambda job: one(*job, a), jobs):
            print(message, file=sys.stderr, flush=True)


def truth_of(sample: str) -> pd.Series:
    with open(f"work/panel_obs/{sample}.fasta") as fh:
        counts = Counter(line.split(":")[1] for line in fh if line.startswith(">"))
    truth = pd.Series(counts, dtype=float)
    return truth / truth.sum()


def to_entries(genomes: pd.Series) -> pd.Series:
    return genomes.groupby(lambda g: ENTRY_OF_GENOME.get(g, g)).sum()


def entry_scores(pred: pd.Series, truth: pd.Series) -> dict:
    pred = pred / pred.sum()
    j = pd.concat([pred, truth], axis=1, keys=["p", "t"]).fillna(0.0)
    both_t, both_p = j.t[list(BU_PAIR)].sum(), j.p[list(BU_PAIR)].sum()
    split = (abs(j.p[BU_PAIR[1]] / both_p - j.t[BU_PAIR[1]] / both_t)
             if both_t > 0.01 and both_p > 0 else float("nan"))
    err = (j.p - j.t).abs()
    return {"entry_tv": 0.5 * err.sum(), "entry_max_abs_error": err.max(),
            "entry_worst": err.idxmax(), "entry_bu_split_error": split}


def score(a) -> None:
    rows = []
    for obs in sorted((a.silva / "obs").glob("S*.obs.mseq")):
        sample = obs.name.removesuffix(".obs.mseq")
        truth = truth_of(sample)
        fits = [(f"silva_{arm}", a.silva / "sweep" / sample / arm) for arm in ARMS]
        fits += [(arm, a.gtdb_sweep / sample / gtdb) for arm, gtdb in GTDB_ARMS.items()]
        for method, out in fits:
            if not (out / "fit.json").exists():
                continue
            comp = pd.read_csv(out / "inferred_composition.csv", index_col="genome_id")
            diag = pd.read_csv(out / "inference_diagnostics.csv").iloc[0]
            fit = json.loads((out / "fit.json").read_text())
            s_file = out / "fitted_s.json"
            fitted_s = (json.loads(s_file.read_text()) if s_file.exists()
                        else float(np.load(out / "posterior_draws.npz")["s"].mean()))
            pred = comp.inferred_mean.drop("background")
            row = {"sample": sample, "method": method,
                   "database": "gtdb_r232" if method.startswith("gtdb") else "silva_138_2_nr99",
                   "panel": next((p for p in ("taxa", "species") if f"_{p}_" in method),
                                 "genome"),
                   "unexplained_fraction": diag.unexplained_fraction,
                   "n_members": diag.n_genomes, "fit_status": fit["fit_status"],
                   "ppc_percentile": (fit["unique_v4_group"] or {}).get(
                       "posterior_predictive_percentile"),
                   "fitted_s": fitted_s,
                   "n_not_identifiable": int((comp.resolution == "not_identifiable").sum()),
                   "not_identifiable": ";".join(comp.index[comp.resolution == "not_identifiable"])}
            if row["panel"] in ("genome", "species"):
                row.update(scores(pred, truth))
                pred = to_entries(pred)
            row.update(entry_scores(pred, to_entries(truth)))
            rows.append(row)
    pd.DataFrame(rows).to_csv(sys.stdout, index=False, float_format="%.5f")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["prepare", "run", "score"]:
        p = sub.add_parser(name)
        p.add_argument("--silva", type=Path, default=Path("work/silva"))
        if name == "run":
            p.add_argument("--jobs", type=int, default=6)
            p.add_argument("--arms", nargs="*", default=None)
        if name == "score":
            p.add_argument("--gtdb-sweep", type=Path, default=Path("work/panel_sweep"))
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
