"""A sample whose reads hit no reference amplicon must be a reported result, not a
crash: `infer_composition.py` writes the same files a normal run does, all-zero and
flagged `status=no_reference_hits`, and exits 0.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import infer_composition  # noqa: E402  (needs sys.path)
import sparse_matrix as sm  # noqa: E402  (needs sys.path)


def test_no_reference_hits_writes_a_zero_composition(tmp_path: Path) -> None:
    refs = ["g1|0|A", "g2|0|B"]
    amplicon_dir = tmp_path / "amplicons"
    amplicon_dir.mkdir()
    pd.DataFrame({"genome_id": ["g1", "g2"], "refseq": refs, "weight": [1.0, 1.0]}).to_csv(
        amplicon_dir / "translation_table.tsv", sep="\t", index=False)
    matrix = tmp_path / "mismapping_matrix.npz"
    sm.write_matrix(matrix, sparse.eye(2, format="csr"), refs)
    obs = tmp_path / "obs.mseq"          # no hits at all
    obs.write_text("")

    out = tmp_path / "out"
    infer_composition.run(SimpleNamespace(
        amplicon_dir=amplicon_dir, mismapping_matrix=matrix, mismapping_matrix_path=str(matrix),
        mismapping_group_id="grp", sim_mseq=None, obs_mseq=[obs], min_identity=None,
        build_mismapping=False, sample_id="S1", output_dir=out, mode="vi", alpha=0.5,
        no_mismapping=False, no_presence=False, presence_prior=0.5, presence_temp=0.1,
        num_samples=10, warmup=10, steps=10, lr=0.02, seed=0,
    ))

    comp = list(csv.DictReader((out / "inferred_composition.csv").open()))
    assert [r["genome_id"] for r in comp] == ["g1", "g2"]
    assert all(float(r["inferred_mean"]) == 0.0 for r in comp), comp
    assert all(float(r["observed_rel_abundance"]) == 0.0 for r in comp), comp

    diag = list(csv.DictReader((out / "inference_diagnostics.csv").open()))
    assert len(diag) == 1 and diag[0]["status"] == "no_reference_hits", diag
    assert int(diag[0]["n_reads"]) == 0, diag
    assert not (out / "mismapping_matrix.csv").exists()
