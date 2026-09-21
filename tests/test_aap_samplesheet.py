"""aap_samplesheet.py turns an amplicon-analysis-pipeline outdir into merged-read rows."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import aap_samplesheet as aap  # noqa: E402  (needs local bin directory)

AAP = ROOT / "tests" / "data" / "aap"
PANEL = ROOT / "tests" / "data" / "refs.fasta"


def _rows(outdir: Path) -> list[dict]:
    a = aap.argparse.Namespace(aap_outdir=outdir, db_label="SILVA-SSU", panel=PANEL,
                               panel_taxa=None, fwd_primer=aap.si.DEFAULT_FWD_PRIMER,
                               rev_primer=aap.si.DEFAULT_REV_PRIMER)
    return aap.build_rows(a)


def test_fixture_matches_the_stub_samplesheet():
    # run_c failed QC; run_a has AAP's .mseq, run_b does not and has compatible primers.
    got = yaml.safe_load(yaml.safe_dump(_rows(AAP)).replace(f"{ROOT}/", ""))
    want = yaml.safe_load((ROOT / "tests" / "data" / "samplesheet.aap.yml").read_text())
    assert got == want


def test_primer_compatibility():
    fwd, rev = aap.si.DEFAULT_FWD_PRIMER, aap.si.DEFAULT_REV_PRIMER
    assert aap.primers_compatible(fwd, fwd)
    assert aap.primers_compatible("CCAGCAGCCGCGGTAATACG", fwd)     # shifted 3, extended 3'
    assert aap.primers_compatible("GGACTACHVGGGTWTCTAAT", rev)     # H within N
    assert not aap.primers_compatible(rev, fwd)
    assert not aap.primers_compatible("GTGACAGCAGCCGCGGTAA", "GTGCCAGCAGCCGCGGTAA")


@pytest.mark.parametrize("break_run, message", [
    (lambda run: (run / "primer-identification" / "rev_primers.fasta")
        .write_text(f">r\n{aap.si.DEFAULT_FWD_PRIMER}\n"), "does not bind"),
    (lambda run: (run / "primer-identification" / "fwd_primers.fasta").unlink(),
        "no fwd primer"),
    (lambda run: (run / "amplified-region-inference" / "run_a.tsv").write_text(
        "Run\tE\tM\tMarker gene\tVariable region\nrun_a\te\tm\t16S\tV4\nrun_a\te\tm\t16S\tV3\n"),
        "not a single 16S V4"),
])
def test_refuses_runs_sr_cannot_use(tmp_path, break_run, message):
    outdir = tmp_path / "aap"
    shutil.copytree(AAP, outdir)
    break_run(outdir / "run_a")
    with pytest.raises(SystemExit, match=message):
        _rows(outdir)


def test_needs_a_panel():
    with pytest.raises(SystemExit):
        aap.main([str(AAP)])
