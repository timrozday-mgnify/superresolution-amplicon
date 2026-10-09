"""Per-genome panel bundles: combining them equals building the whole panel at once."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import sparse_matrix as sm  # noqa: E402  (needs local bin directory)
import split_panel  # noqa: E402  (needs local bin directory)

FWD, REV = "GTGCCAGCAGCCGCGGTAA", "GGACTACAAGGGTATCTAAT"
COMP = str.maketrans("ACGT", "TGCA")


def _seq(rng: random.Random, n: int = 80) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutate(seq: str, at: int) -> str:
    return seq[:at] + ("A" if seq[at] != "A" else "C") + seq[at + 1:]


def _bpk(*args) -> None:
    subprocess.run([sys.executable, str(ROOT / "bin" / "build_panel_kernel.py"), *map(str, args)],
                   check=True)


def _bundle(tmp: Path, name: str, panel: Path, db: Path, *,
            weights: Path | None = None) -> Path | None:
    """prepare + align a panel the way PANEL_PREPARE and PANEL_ALIGN do, with the
    provenance.json the workflow writes; ``None`` where nothing amplifies, which the
    workflow drops before aligning."""
    prepared, out = tmp / f"{name}_prepared", tmp / name
    _bpk("prepare", "--panel-amplicons", panel, "--db-amplicons", db, "--fwd-primer", FWD,
         "--rev-primer", REV, "--max-mismatch", 0, "--allow-empty", "-o", prepared,
         *(["--panel-weights", weights] if weights else []))
    if not (prepared / "sources.fasta").stat().st_size:
        return None
    out.mkdir()
    for f in ("panel_translation.tsv", "sources.tsv"):
        (out / f).write_bytes((prepared / f).read_bytes())
    _bpk("align", "--prepared", prepared, "--db-amplicons", db, "--tau", 1,
         "--distance-decay", 0.01, "-o", out / "mismapping_matrix.npz")
    (out / "provenance.json").write_text(json.dumps({
        "matrix_key": f"panel_{name}", "panel": hashlib.sha256(panel.read_bytes()).hexdigest(),
        "panel_taxa": None, "panel_weights": None, "reference_sha256": "db",
        "fwd_primer": FWD, "rev_primer": REV, "primer_mismatches": 0, "samples": [name]}))
    return out


@pytest.fixture
def panel(tmp_path):
    rng = random.Random(1)
    v1, v3, other = _seq(rng), _seq(rng), _seq(rng)
    v2 = _mutate(v1, 40)                       # one edit from v1: a neighbour and a strata entry
    amp = lambda v: "TTT" + FWD + v + REV.translate(COMP)[::-1] + "GGG"  # noqa: E731
    fasta = tmp_path / "panel.fasta"
    fasta.write_text(f">ga|0|x\n{amp(v1)}\n>ga|1|x\n{amp(v2)}\n>gb|0|x\n{amp(v2)}\n"
                     f">gb|1|x\n{amp(v3)}\n>gc|0|x\n{_seq(rng, 200)}\n")
    db = tmp_path / "db_amplicons.fasta"
    db.write_text(f">r1\n{v1}\n>r2\n{v2}\n>r3\n{v3}\n>r4\n{other}\n>r5\n{v3}\n")
    return fasta, db


def _part_bundles(tmp_path, fasta, db):
    genomes = tmp_path / "genomes"
    subprocess.run([sys.executable, str(ROOT / "bin" / "split_panel.py"), fasta, "-o", genomes],
                   check=True)
    bundles = {g.stem: _bundle(tmp_path, g.stem, g, db) for g in sorted(genomes.glob("*.fasta"))}
    assert bundles.pop("gc") is None          # no primer site: nothing to build
    return list(bundles.values())


def _table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df.sort_values(list(df.columns[:2])).reset_index(drop=True)


def test_combined_parts_equal_the_whole_panel(tmp_path, panel):
    fasta, db = panel
    whole = _bundle(tmp_path, "whole", fasta, db)
    parts = _part_bundles(tmp_path, fasta, db)
    out = tmp_path / "combined"
    _bpk("combine", "--panel", fasta, "--parts", *parts, "-o", out)

    a, b = (sm.read_kernel(d / "mismapping_matrix.npz") for d in (whole, out))
    assert a.source_ids == b.source_ids and a.label_ids == b.label_ids
    np.testing.assert_array_equal(a.home, b.home)
    np.testing.assert_allclose(a.kernel.toarray(), b.kernel.toarray())
    np.testing.assert_array_equal(a.strata[0].toarray(), b.strata[0].toarray())
    assert (sorted(zip(*(x.tolist() for x in a.neighbours)))
            == sorted(zip(*(x.tolist() for x in b.neighbours))))
    for f in ("panel_translation.tsv", "sources.tsv"):
        pd.testing.assert_frame_equal(_table(whole / f), _table(out / f))
    prov = json.loads((out / "provenance.json").read_text())
    # What --panel_kernel checks against the panel FASTA it is run with.
    assert prov["panel"] == hashlib.sha256(fasta.read_bytes()).hexdigest()
    assert prov["samples"] == ["ga", "gb"]
    assert (out / "panel_unamplifiable.txt").read_text() == "gc\n"


def test_a_panel_is_any_subset_of_the_parts(tmp_path, panel):
    fasta, db = panel
    parts = _part_bundles(tmp_path, fasta, db)
    sub = tmp_path / "sub.fasta"
    sub.write_bytes(split_panel.genome_chunks(fasta)["gb"])
    out = tmp_path / "combined"
    _bpk("combine", "--panel", sub, "--parts", *parts, "-o", out)
    assert sm.read_kernel(out / "mismapping_matrix.npz").kernel.shape[0] == 2


def test_weights_apply_at_combine(tmp_path, panel):
    fasta, db = panel
    parts = _part_bundles(tmp_path, fasta, db)
    translation = pd.read_csv(tmp_path / "ga" / "panel_translation.tsv", sep="\t")
    weights = tmp_path / "weights.tsv"
    weights.write_text("genome_id\tsource\tweight\n"
                       + "".join(f"ga\t{s}\t{w}\n" for s, w in zip(translation.source, (3, 0))))
    whole = _bundle(tmp_path, "whole", fasta, db, weights=weights)
    out = tmp_path / "combined"
    _bpk("combine", "--panel", fasta, "--parts", *parts, "--panel-weights", weights, "-o", out)
    pd.testing.assert_frame_equal(_table(whole / "panel_translation.tsv"),
                                  _table(out / "panel_translation.tsv"))
    assert json.loads((out / "provenance.json").read_text())["panel_weights"] == \
        hashlib.sha256(weights.read_bytes()).hexdigest()


def test_an_amplifiable_genome_without_a_part_fails(tmp_path, panel):
    fasta, db = panel
    parts = [p for p in _part_bundles(tmp_path, fasta, db) if p.name == "ga"]
    with pytest.raises(subprocess.CalledProcessError):
        _bpk("combine", "--panel", fasta, "--parts", *parts, "-o", tmp_path / "combined")
