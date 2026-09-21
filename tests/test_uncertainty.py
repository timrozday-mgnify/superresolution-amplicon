"""Posterior ambiguity outputs: a two-way and a three-way tie are found, a resolved genome
is not, and the LCA table cuts tied genomes back to the genus they share.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import infer_composition  # noqa: E402  (needs sys.path)


def test_ties_are_reported_and_cut_to_lca(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_draws, n_reads = 2000, 2000
    base = rng.dirichlet(n_reads * np.array([0.4, 0.3, 0.3]), n_draws)   # [AB, C, DEF]
    p = rng.uniform(size=n_draws)                                        # A:B unidentified
    q = rng.dirichlet(np.ones(3), n_draws)                               # D:E:F unidentified
    draws = np.column_stack([base[:, 0] * p, base[:, 0] * (1 - p), base[:, 1],
                             base[:, [2]] * q, np.zeros(n_draws)])       # Z pruned
    ids = ["A", "B", "C", "D", "E", "F", "Z"]
    genus = ["G1", "G1", "G2", "G3", "G3", "G3"]
    lineage_of = {i: f"Bacteria;{g};{ids[i]}" for i, g in enumerate(genus)}
    tax = tmp_path / "refs.tax"
    tax.write_text("#cutoff: 0.00:0 0.80:0 0.80:0\n#name: refdb\n#levels: Kingdom Genus Genome\n")

    a = SimpleNamespace(sample_id="S", output_dir=tmp_path, taxonomy=tax)
    infer_composition._uncertainty_outputs(a, ids, draws, n_reads, lineage_of)

    pairs = pd.read_csv(tmp_path / "ambiguity_pairs.csv")
    assert {frozenset(r) for r in zip(pairs.id_a, pairs.id_b)} == {
        frozenset("AB"), frozenset("DE"), frozenset("DF"), frozenset("EF")}, pairs

    sets = pd.read_csv(tmp_path / "ambiguity_sets.csv").set_index("members")
    assert set(sets.index) == {"A;B", "D;E;F"}, sets
    assert sets.loc["D;E;F", "lca"] == "Bacteria;G3" and sets.loc["D;E;F", "gain"] > 0.9, sets

    lca = pd.read_csv(tmp_path / "lca_composition.csv").set_index("lca")
    assert lca["members"].to_dict() == {
        "Bacteria;G1": "A;B", "Bacteria;G2;C": "C", "Bacteria;G3": "D;E;F"}, lca
    assert lca.loc["Bacteria;G1", "rank"] == "Genus", lca
    assert abs(lca["inferred_mean"].sum() - 1.0) < 1e-9, lca
