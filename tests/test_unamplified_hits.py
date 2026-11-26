"""Hits on database entries the primers do not amplify (27F-1492R against SILVA).

MAPseq maps against the whole database, so reads land on entries with no amplicon. Those
used to be dropped from the kernel and counted as background in the sample; now each such
entry is a label, so both the kernel and the sample count them for their source.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import build_panel_kernel as bpk  # noqa: E402  (needs local bin directory)
import infer_composition as ic  # noqa: E402  (needs local bin directory)
import sparse_matrix as sm  # noqa: E402  (needs local bin directory)

A = "ACGTACGTAGGCTTAGCTGGACCTTGACAGTTCCAGGTACCGATTGCAAGCTTACGGATCCGTTAACGGTA"
C = "TTGCAAGCTTACGGATCCGTTAACGGTAACGTACGTAGGCTTAGCTGGACCTTGACAGTTCCAGGTACCGA"
TRUNC = "GQ898710.1.1465"          # a database entry with no amplicon under the primers
UNAMP = bpk.UNAMPLIFIED + TRUNC


def _mseq(path: Path, rows) -> Path:
    path.write_text("".join(f"{q}\t{hit}\t250\t1.0\n" for q, hit in rows))
    return path


def _prepared(tmp_path: Path):
    a, c = bpk.v4g(A), bpk.v4g(C)
    prep = tmp_path / "prep"
    prep.mkdir()
    pd.DataFrame({"source": [a, c], "genomes": ["gX", "gY"], "in_db": True}).to_csv(
        prep / "sources.tsv", sep="\t", index=False)
    (prep / "sources.fasta").write_text(f">{a}\n{A}\n>{c}\n{C}\n")
    pd.DataFrame({"genome_id": ["gX", "gY"], "source": [a, c], "weight": 1.0}).to_csv(
        prep / "panel_translation.tsv", sep="\t", index=False)
    db = tmp_path / "amplicons.fasta"             # A has two references, C one
    db.write_text(f">r1\n{A}\n>r2\n{A}\n>r3\n{C}\n")
    # a's error-free read hits the unamplified entry exactly, as B. uniformis's does
    # HQ776863.1.1451: its own group's two references do not keep it.
    home = _mseq(tmp_path / "home.mseq", [(a, TRUNC), (c, "r3")])
    return prep, db, home, a, c


def test_build_measures_unamplified_hits_and_inference_counts_them(tmp_path):
    prep, db, home, a, c = _prepared(tmp_path)
    sim = _mseq(tmp_path / "sim.mseq", [(f"{a}:{i}", "r1") for i in range(6)]
                + [(f"{a}:{i}", TRUNC) for i in range(6, 10)]
                + [(f"{c}:{i}", "r3") for i in range(10)])
    kernel = prep / "panel" / "mismapping_matrix.npz"
    kernel.parent.mkdir()
    bpk.build(SimpleNamespace(prepared=prep, db_amplicons=db, home_mseq=home, sim_mseq=sim,
                              min_identity=None, provenance=[], out=kernel))
    k = sm.read_kernel(kernel)
    row = dict(zip(k.label_ids, k.kernel.toarray()[k.source_ids.index(a)]))
    # Before: the 4 truncated hits were dropped and A took the whole row.
    assert np.isclose(row[bpk.v4g(A)], 0.6) and np.isclose(row[UNAMP], 0.4)
    assert k.label_ids[k.home[k.source_ids.index(a)]] == UNAMP
    assert dict(zip(k.ref_headers, k.label_of_ref))[TRUNC] == k.label_ids.index(UNAMP)

    # gX and gY at 0.5 each; gX's reads split 3:2 over r1 and the truncated entry.
    obs = _mseq(tmp_path / "obs.mseq", [(f"x{i}", "r1") for i in range(300)]
                + [(f"t{i}", TRUNC) for i in range(200)]
                + [(f"y{i}", "r3") for i in range(500)])
    out = tmp_path / "out"
    ic.run(SimpleNamespace(
        amplicon_dir=prep, mismapping_matrix=kernel, sim_mseq=None, obs_mseq=[obs],
        min_identity=None, sample_id="S", mode="mle", alpha=1.0, steps=3000, lr=0.02,
        num_samples=10, warmup=0, no_mismapping=False, no_presence=True,
        presence_prior=0.5, presence_temp=0.5, seed=0, min_infer_reads=1, output_dir=out))
    got = pd.read_csv(out / "inferred_composition.csv", index_col="genome_id").inferred_mean
    assert pd.read_csv(out / "inference_diagnostics.csv").n_unlabelled_hits[0] == 0
    assert abs(got.gX - 0.5) < 0.02 and got.background < 0.02, got   # was gX 0.3, bg 0.2


def test_align_homes_a_source_on_its_unamplified_hit(tmp_path):
    """The align kernel (the browser's path) puts the error-free read where MAPseq does."""
    prep, db, home, a, c = _prepared(tmp_path)
    kernel = prep / "panel" / "mismapping_matrix.npz"
    kernel.parent.mkdir()
    bpk.align(SimpleNamespace(prepared=prep, db_amplicons=db, home_mseq=home, tau=1,
                              distance_decay="0.1", ambiguity_weight=0.3, out=kernel))
    k = sm.read_kernel(kernel)
    homes = {s: k.label_ids[h] for s, h in zip(k.source_ids, k.home)}
    assert homes == {a: UNAMP, c: bpk.v4g(C)}
    assert np.isclose(k.kernel.toarray()[k.source_ids.index(a), k.label_ids.index(UNAMP)], 1.0)


def test_v4_style_hits_add_no_labels(tmp_path):
    """Every hit amplifiable (the MiMiCC V4 case): the label space is the database's."""
    headers, labels, label_of_ref = bpk._add_unamplified(
        [_mseq(tmp_path / "h.mseq", [("s", "r1"), ("t", "")])], ["r1", "r2"], ["l0"],
        np.array([0, 0]))
    assert headers == ["r1", "r2"] and labels == ["l0"] and label_of_ref.tolist() == [0, 0]
