"""Posterior-predictive fit check on a panel kernel."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import check_composition_fit as ccf  # noqa: E402  (needs local bin directory)
import sparse_matrix as sm  # noqa: E402  (needs local bin directory)


def _write_mseq(path: Path, references: list[str], counts: np.ndarray) -> None:
    """Write MAPseq rows with the requested counts in reference order."""
    with path.open("w") as handle:
        for reference, count in zip(references, counts):
            for index in range(int(count)):
                handle.write(f"read_{reference}_{index}\t{reference}\t100\t0.99\n")


def _fit_args(
    amplicon_dir: Path,
    matrix: Path,
    observation: Path,
    composition: Path,
    posterior: Path,
    output: Path,
) -> SimpleNamespace:
    """Return stable direct-call arguments for the fit-check script."""
    return SimpleNamespace(
        composition=composition,
        posterior_draws=posterior,
        obs_mseq=[observation],
        amplicon_dir=amplicon_dir,
        mismapping_matrix=matrix,
        min_identity=None,
        min_infer_reads=1000,
        ppc_draws=200,
        ppc_low_tail=0.01,
        ppc_high_tail=0.99,
        seed=11,
        output=output,
    )


def test_panel_fit_passes_exact_counts_and_flags_a_perturbation(tmp_path: Path) -> None:
    """The rectangular-kernel check runs over observed labels + U: gA/gC share source a,
    gB's source leaks, and reads on g5 (no source reaches it) are background."""
    headers = ["g1|0|A", "g2|0|A", "g3|0|B", "g4|0|C", "g5|0|D"]
    matrix = tmp_path / "panel_kernel.npz"
    sm.write_kernel(matrix, sparse.csr_array(np.array([[0.7, 0.3, 0.0, 0.0],
                                                      [0.0, 0.1, 0.9, 0.0]])),
                    ["a", "b"], ["l0", "l1", "l2", "l3"], np.array([0, 2]),
                    ref_headers=headers, label_of_ref=np.array([0, 0, 1, 2, 3]))
    prep = tmp_path / "prep"
    prep.mkdir()
    pd.DataFrame({"genome_id": ["gA", "gB", "gC"], "source": ["a", "b", "a"],
                  "weight": 1.0}).to_csv(prep / "panel_translation.tsv", sep="\t", index=False)
    genomes = ["gA", "gB", "gC", "background"]
    theta = np.array([0.25, 0.40, 0.25, 0.10])
    composition = tmp_path / "inferred_composition.csv"
    pd.DataFrame({"sample": "S1", "genome_id": genomes, "inferred_mean": theta,
                  "fit_status": "ok"}).to_csv(composition, index=False)
    posterior = tmp_path / "posterior_draws.npz"
    np.savez_compressed(
        posterior, genome_ids=np.asarray(genomes), theta_eff=np.tile(theta, (200, 1)),
        s=np.ones(200), conc_frac=np.full(200, 1000.0), use_mismapping=np.asarray(True),
        infer_distance_decay=np.asarray(False), likelihood=np.asarray("dirichlet_multinomial"),
        use_active_subset=np.asarray(False), n_reads=np.asarray(10000), fit_status=np.asarray("ok"))

    observation = tmp_path / "observed.mseq"
    labelled = ["g1|0|A", "g3|0|B", "g4|0|C", "g5|0|D"]     # l0, l1, l2, U
    exact = np.random.default_rng(7).multinomial(10000, [0.35, 0.19, 0.36, 0.10])
    _write_mseq(observation, labelled, exact)
    passed = tmp_path / "passed.json"
    ccf.run(_fit_args(prep, matrix, observation, composition, posterior, passed))
    diagnostics = json.loads(passed.read_text())
    assert diagnostics["fit_status"] == "ok", diagnostics
    assert diagnostics["kernel_format"] == "rectangular", diagnostics
    assert diagnostics["gate"] == "label" and diagnostics["label"]["n_labels"] == 4, diagnostics

    _write_mseq(observation, labelled, exact + np.array([-1500, 0, 1500, 0]))
    failed = tmp_path / "failed.json"
    ccf.run(_fit_args(prep, matrix, observation, composition, posterior, failed))
    diagnostics = json.loads(failed.read_text())
    assert diagnostics["fit_status"] == "model_misfit", diagnostics
