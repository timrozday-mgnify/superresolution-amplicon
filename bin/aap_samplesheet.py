#!/usr/bin/env python
"""Write a superresolution-amplicon samplesheet from an amplicon-analysis-pipeline outdir.

Each QC-passed run becomes one row: its fastp-merged reads (``qc/<id>.merged.fastq.gz``)
as ``reads:`` with ``merged: true``, the fraction of pairs fastp merged as
``merge_rate``, and AAP's own MAPseq classification against ``--db-label`` as ``mseq:``
when AAP wrote one, so SR reuses it instead of mapping again. Every row gets the same
panel (``--panel`` and/or ``--panel-taxa``): against a generic database like SILVA the
panel is what abundances are reported over.

A run is refused, and nothing is written, when AAP did not call it a single 16S V4
amplicon, or when the primers AAP identified cannot sit on the same binding sites as
``--fwd-primer``/``--rev-primer`` (primers are global in SR). Primers that differ but
are compatible (a few bases longer or shorter, a different degeneracy) are logged.

The merged reads keep their primers, so the SR run needs ``--trim_primers false``.

Example:
    aap_samplesheet.py aap_results --panel panel.fasta -o samplesheet.yml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)

log = logging.getLogger("aap_samplesheet")

# reads_to_fasta.py warns at the same merge rate: below it, merged reads are a biased
# subset of the fragments and the kernel should come from simulated pairs.
MIN_MERGE_RATE = 0.8


def primers_compatible(a: str, b: str, max_shift: int = 4, min_overlap: int = 15) -> bool:
    """Whether IUPAC primers ``a`` and ``b`` can bind the same site.

    True when, at some offset of at most ``max_shift`` bases, they overlap by at least
    ``min_overlap`` positions and every overlapping pair of codes shares a base.
    """
    for shift in range(-max_shift, max_shift + 1):
        pairs = [(a[i], b[i - shift]) for i in range(len(a)) if 0 <= i - shift < len(b)]
        if len(pairs) >= min_overlap and all(
                si._IUPAC.get(x, set("ACGT")) & si._IUPAC.get(y, set("ACGT"))
                for x, y in pairs):
            return True
    return False


def first_fasta_seq(path: Path) -> str | None:
    """The first sequence in a FASTA, or ``None`` if it has none."""
    records = list(si.read_fasta(path)) if path.exists() else []
    return records[0][1].upper() if records else None


def passed_runs(outdir: Path) -> list[str]:
    """Run ids from ``qc_passed_runs.csv`` (``<id>,<status>`` with no header)."""
    lines = (outdir / "qc_passed_runs.csv").read_text().splitlines()
    return [line.split(",")[0].strip() for line in lines if line.strip()]


def check_run(run: Path, fwd: str, rev: str) -> list[str]:
    """Reasons to refuse ``run``; empty if SR can use it."""
    rid = run.name
    problems = []
    region = run / "amplified-region-inference" / f"{rid}.tsv"
    if not region.exists():
        problems.append(f"{rid}: no {region.relative_to(run.parent)}")
    else:
        rows = [line.split("\t") for line in region.read_text().splitlines()[1:] if line.strip()]
        calls = sorted({(r[3], r[4]) for r in rows if len(r) >= 5})
        if calls != [("16S", "V4")]:
            problems.append(f"{rid}: amplified region {calls or 'none'}, not a single 16S V4")
    for name, want in (("fwd", fwd), ("rev", rev)):
        got = first_fasta_seq(run / "primer-identification" / f"{name}_primers.fasta")
        if got is None:
            problems.append(f"{rid}: AAP identified no {name} primer")
        elif not primers_compatible(got, want):
            problems.append(f"{rid}: {name} primer {got} does not bind where {want} does")
        elif got != want:
            log.info("%s: %s primer %s is compatible with %s", rid, name, got, want)
    return problems


def merge_rate(fastp_json: Path) -> float:
    """Merged reads over input pairs, from fastp's JSON report."""
    report = json.loads(fastp_json.read_text())
    pairs = report["read1_before_filtering"]["total_reads"]
    return report["merged_and_filtered"]["total_reads"] / pairs if pairs else 0.0


def find_mseq(run: Path, db_label: str) -> Path | None:
    """AAP's MAPseq output for ``run`` against ``db_label``, if it wrote one."""
    hits = sorted((run / "taxonomy-summary" / db_label).glob("*.mseq*"))
    if len(hits) > 1:
        raise SystemExit(f"{run.name}: more than one .mseq under taxonomy-summary/{db_label}")
    return hits[0] if hits else None


def build_rows(a) -> list[dict]:
    outdir = a.aap_outdir.resolve()
    fwd, rev = a.fwd_primer.upper(), a.rev_primer.upper()
    passed = set(passed_runs(outdir))
    runs = sorted(p for p in outdir.iterdir() if (p / "qc").is_dir())
    for run in runs:
        if run.name not in passed:
            log.info("%s: not in qc_passed_runs.csv, skipped", run.name)
    runs = [r for r in runs if r.name in passed]
    problems = [p for run in runs for p in check_run(run, fwd, rev)]
    if problems:
        raise SystemExit("refusing to write a samplesheet:\n  " + "\n  ".join(problems))
    if not runs:
        raise SystemExit(f"no QC-passed runs under {outdir}")

    rows = []
    for run in runs:
        rid = run.name
        rate = merge_rate(run / "qc" / f"{rid}.fastp.json")
        if rate < MIN_MERGE_RATE:
            log.warning("%s: fastp merged only %.0f%% of pairs; the merged reads are a biased "
                        "subset of the fragments", rid, 100 * rate)
        row = {"id": rid, "reads": str(run / "qc" / f"{rid}.merged.fastq.gz"),
               "merged": True, "merge_rate": round(rate, 4)}
        mseq = find_mseq(run, a.db_label)
        if mseq:
            row["mseq"] = str(mseq)
        if a.panel:
            row["panel_references"] = str(a.panel.resolve())
        if a.panel_taxa:
            row["panel_taxa"] = str(a.panel_taxa.resolve())
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aap_outdir", type=Path, help="amplicon-analysis-pipeline --outdir")
    ap.add_argument("--db-label", default="SILVA-SSU",
                    help="AAP database label whose .mseq to reuse (taxonomy-summary/<label>/)")
    ap.add_argument("--panel", type=Path, help="genome panel FASTA (panel_references)")
    ap.add_argument("--panel-taxa", type=Path, help="taxon panel TSV (panel_taxa)")
    ap.add_argument("--fwd-primer", default=si.DEFAULT_FWD_PRIMER)
    ap.add_argument("--rev-primer", default=si.DEFAULT_REV_PRIMER)
    ap.add_argument("-o", "--output", type=Path, help="samplesheet YAML (default stdout)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not (a.panel or a.panel_taxa):
        ap.error("name a panel: --panel and/or --panel-taxa")
    text = yaml.safe_dump(build_rows(a), sort_keys=False)
    if a.output:
        a.output.write_text(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
