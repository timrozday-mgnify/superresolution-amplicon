"""Forward-fit and depth-gate tests for GTDB Phase 0 diagnostics."""
from __future__ import annotations

import csv
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
import infer_composition as ic  # noqa: E402  (needs local bin directory)
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


def test_forward_fit_passes_exact_counts_and_flags_a_perturbation(tmp_path: Path) -> None:
    """Check exact-forward PPC at reference and unique-V4-group levels."""
    references = ["g1|0|A", "g2|0|A", "g3|0|B"]
    amplicon_dir = tmp_path / "amplicons"
    amplicon_dir.mkdir()
    pd.DataFrame(
        {
            "genome_id": ["g1", "g2", "g3"],
            "refseq": references,
            "weight": [1.0, 1.0, 1.0],
        }
    ).to_csv(amplicon_dir / "translation_table.tsv", sep="\t", index=False)
    (amplicon_dir / "amplicons.fasta").write_text(
        ">g1|0|A\nACGT\n>g2|0|A\nACGT\n>g3|0|B\nTGCA\n"
    )

    matrix = tmp_path / "mismapping_matrix.npz"
    sm.write_grouped(
        matrix,
        sparse.csr_array(np.array([[0.5, 0.0], [0.0, 1.0]])),
        np.array([0, 0, 1], dtype=np.int32),
        references,
    )
    theta = np.array([0.30, 0.50, 0.20])
    composition = tmp_path / "inferred_composition.csv"
    pd.DataFrame(
        {
            "sample": "S1",
            "genome_id": ["g1", "g2", "g3"],
            "inferred_mean": theta,
            "fit_status": "ok",
        }
    ).to_csv(composition, index=False)
    posterior = tmp_path / "posterior_draws.npz"
    np.savez_compressed(
        posterior,
        format_version=np.asarray("1"),
        genome_ids=np.asarray(["g1", "g2", "g3"]),
        theta_eff=np.tile(theta, (200, 1)),
        s=np.ones(200),
        conc_frac=np.full(200, 1000.0),
        use_mismapping=np.asarray(True),
        infer_distance_decay=np.asarray(False),
        likelihood=np.asarray("dirichlet_multinomial"),
        use_active_subset=np.asarray(False),
        n_reads=np.asarray(10000, dtype=np.int64),
        fit_status=np.asarray("ok"),
    )

    observation = tmp_path / "observed.mseq"
    exact_counts = np.random.default_rng(7).multinomial(10000, [0.4, 0.4, 0.2])
    _write_mseq(observation, references, exact_counts)
    passed = tmp_path / "passed.json"
    ccf.run(_fit_args(amplicon_dir, matrix, observation, composition, posterior, passed))
    diagnostics = json.loads(passed.read_text())
    assert diagnostics["fit_status"] == "ok", diagnostics
    assert diagnostics["raw_reference"]["n_labels"] == 3, diagnostics
    assert diagnostics["unique_v4_group"]["n_labels"] == 2, diagnostics
    assert diagnostics["gate"] == "unique_v4_group", diagnostics

    # Reads piled onto one member of the duplicate group: group totals are exact, so only
    # the diagnostic raw-reference check may complain.
    tied = exact_counts.copy()
    tied[:2] = [tied[:2].sum(), 0]
    _write_mseq(observation, references, tied)
    split = tmp_path / "split.json"
    ccf.run(_fit_args(amplicon_dir, matrix, observation, composition, posterior, split))
    diagnostics = json.loads(split.read_text())
    assert diagnostics["raw_reference"]["posterior_predictive_percentile"] > 0.99, diagnostics
    assert diagnostics["fit_status"] == "ok", diagnostics

    _write_mseq(observation, references, np.array([6000, 0, 4000]))
    failed = tmp_path / "failed.json"
    ccf.run(_fit_args(amplicon_dir, matrix, observation, composition, posterior, failed))
    diagnostics = json.loads(failed.read_text())
    assert diagnostics["fit_status"] == "model_misfit", diagnostics
    assert diagnostics["unique_v4_group"]["posterior_predictive_percentile"] > 0.99


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
    assert diagnostics["unique_v4_group"]["n_labels"] == 4, diagnostics

    _write_mseq(observation, labelled, exact + np.array([-1500, 0, 1500, 0]))
    failed = tmp_path / "failed.json"
    ccf.run(_fit_args(prep, matrix, observation, composition, posterior, failed))
    diagnostics = json.loads(failed.read_text())
    assert diagnostics["fit_status"] == "model_misfit", diagnostics


def test_v4_group_fit_is_checked_in_group_space(tmp_path: Path) -> None:
    """A V4-group-space fit round-trips through the fit check on its group table."""
    ic.demo_v4(tmp_path)
    fitted = tmp_path / "v4"
    output = tmp_path / "fit.json"
    ccf.run(_fit_args(tmp_path / "amp", tmp_path / "M.npz", tmp_path / "obs.mseq",
                      fitted / "inferred_v4_groups.csv", fitted / "posterior_draws.npz",
                      output))
    diagnostics = json.loads(output.read_text())
    assert diagnostics["infer_space"] == "v4_group", diagnostics
    assert diagnostics["unique_v4_group"]["n_labels"] == 2, diagnostics
    assert diagnostics["fit_status"] == "ok", diagnostics


def test_v4_group_fit_ignores_which_identical_member_mapseq_picked(tmp_path: Path) -> None:
    """MAPseq piles a group's reads on a few of its identical members; only totals count.

    Group X is carried by ten genomes, Y by one; reads are 50/50 but every X read sits on
    one member. A reference-level likelihood with a 1/size split would starve X.
    """
    references = [f"a{i}|0|x" for i in range(10)] + ["c|0|y"]
    amplicon_dir = tmp_path / "amplicons"
    amplicon_dir.mkdir()
    pd.DataFrame({"genome_id": [r.split("|")[0] for r in references], "refseq": references,
                  "weight": 1.0}).to_csv(amplicon_dir / "translation_table.tsv", sep="\t",
                                         index=False)
    (amplicon_dir / "amplicons.fasta").write_text(
        "".join(f">{r}\n{'ACGTACGT' if r.endswith('x') else 'TTGCAAGC'}\n" for r in references))
    matrix = tmp_path / "M.npz"
    sm.write_matrix(matrix, sparse.csr_array(sparse.block_diag(
        [np.full((10, 10), 0.1), np.ones((1, 1))]).toarray()), references)
    observation = tmp_path / "obs.mseq"
    _write_mseq(observation, references, np.array([5000] + [0] * 9 + [5000]))

    ic.run(SimpleNamespace(
        amplicon_dir=amplicon_dir, mismapping_matrix=matrix, sim_mseq=None,
        obs_mseq=[observation], min_identity=None, sample_id="S1", mode="vi", alpha=0.5,
        steps=500, lr=0.05, num_samples=100, warmup=0, no_mismapping=False, seed=0,
        no_presence=True, presence_prior=0.01, presence_temp=1.0, no_prune=False,
        infer_space="v4_group", taxonomy=None, output_dir=tmp_path / "out"))
    groups = pd.read_csv(tmp_path / "out" / "inferred_v4_groups.csv").set_index("n_references")
    assert abs(groups.loc[10, "inferred_mean"] - 0.5) < 0.03, groups


def test_horseshoe_recovers_a_relabelled_group_from_its_secondary_leak(tmp_path: Path) -> None:
    """The Phase 5 case: MAPseq never labels A (90% -> B, 10% -> C), so only C's reads
    betray it. C as a source would also leak onto E, which drew no reads, so A is the only
    consistent explanation; the horseshoe must keep A and shrink C. E leaks a little onto D
    so pruning keeps E's zero-count label: a pruned label's leak is renormalised away.
    Twenty pure groups at exact counts pin the overdispersion, as a real sample's many
    labels do; with five labels ``conc_frac`` collapses and C's reads carry no weight."""
    n_pure = 20
    names = list("abcde") + [f"p{i}" for i in range(n_pure)]
    references = [f"{n}|0|{n}" for n in names]
    amplicon_dir = tmp_path / "amplicons"
    amplicon_dir.mkdir()
    pd.DataFrame({"genome_id": names, "refseq": references, "weight": 1.0}).to_csv(
        amplicon_dir / "translation_table.tsv", sep="\t", index=False)
    (amplicon_dir / "amplicons.fasta").write_text(
        "".join(f">{r}\n{format(i, '010b').replace('0', 'A').replace('1', 'C')}\n"
                for i, r in enumerate(references)))
    M = np.eye(len(names))
    M[:5, :5] = [[0, .9, .1, 0, 0], [0, 1, 0, 0, 0], [0, 0, .5, 0, .5], [0, 0, 0, 1, 0],
                 [0, 0, 0, .1, .9]]
    matrix = tmp_path / "M.npz"
    sm.write_matrix(matrix, sparse.csr_array(M), references)
    observation = tmp_path / "obs.mseq"
    _write_mseq(observation, references, np.array([0, 5700, 300, 4000, 0] + [500] * n_pure))

    ic.run(SimpleNamespace(
        amplicon_dir=amplicon_dir, mismapping_matrix=matrix, sim_mseq=None,
        obs_mseq=[observation], min_identity=None, sample_id="S1", mode="vi", alpha=0.5,
        # The horseshoe converges slower than the Dirichlet: 2,000 steps leaves A at ~0.
        steps=8000, lr=0.05, num_samples=100, warmup=0, no_mismapping=False, seed=0,
        no_presence=True, presence_prior=0.01, presence_temp=1.0, no_prune=False,
        horseshoe=True, infer_space="v4_group", taxonomy=None, output_dir=tmp_path / "out"))
    groups = pd.read_csv(tmp_path / "out" / "inferred_v4_groups.csv").set_index(
        "representative_reference")
    # Truth: A 0.15, B 0.15, D 0.20, pure groups 0.50 (A's 3,000 reads of 20,000).
    assert abs(groups.loc["a|0|a", "inferred_mean"] - 0.15) < 0.03, groups
    assert groups.loc["c|0|c", "inferred_mean"] < 0.01, groups


def test_depth_gate_writes_low_depth_instead_of_a_composition(tmp_path: Path) -> None:
    """Do not fit or report a biological composition below the configured depth gate."""
    references = ["g1|0|A", "g2|0|B"]
    amplicon_dir = tmp_path / "amplicons"
    amplicon_dir.mkdir()
    pd.DataFrame(
        {
            "genome_id": ["g1", "g2"],
            "refseq": references,
            "weight": [1.0, 1.0],
        }
    ).to_csv(amplicon_dir / "translation_table.tsv", sep="\t", index=False)
    matrix = tmp_path / "mismapping_matrix.npz"
    sm.write_matrix(matrix, sparse.eye(2, format="csr"), references)
    observation = tmp_path / "observed.mseq"
    _write_mseq(observation, references, np.array([15, 15]))

    output = tmp_path / "out"
    ic.run(
        SimpleNamespace(
            amplicon_dir=amplicon_dir,
            mismapping_matrix=matrix,
            mismapping_matrix_path=str(matrix),
            mismapping_group_id="group",
            sim_mseq=None,
            obs_mseq=[observation],
            min_identity=None,
            build_mismapping=False,
            sample_id="S1",
            output_dir=output,
            mode="vi",
            alpha=0.5,
            no_mismapping=False,
            no_presence=False,
            presence_prior=0.5,
            presence_temp=0.1,
            num_samples=10,
            warmup=10,
            steps=10,
            lr=0.02,
            seed=0,
            min_infer_reads=1000,
        )
    )

    rows = list(csv.DictReader((output / "inferred_composition.csv").open()))
    assert {row["fit_status"] for row in rows} == {"low_depth"}
    assert all(float(row["inferred_mean"]) == 0.0 for row in rows)
    inference = next(csv.DictReader((output / "inference_diagnostics.csv").open()))
    assert inference["status"] == "low_depth"
    assert inference["fit_status"] == "low_depth"
    with np.load(output / "posterior_draws.npz", allow_pickle=False) as posterior:
        assert posterior["theta_eff"].shape == (0, 2)


def test_pruning_never_drops_a_reference_with_a_direct_hit() -> None:
    """A hit reference is kept even when M gives it no route to any observed label (an
    empty measured row nothing maps to), and its genome's other copies come with it. A
    genome with no hit and no route to one is still pruned."""
    # refs: a (g0, hit), b (g1, hit, empty row), b2 (g1, no hit), d (g2, no hit)
    M = sparse.csr_array(np.array([[1., 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    obs, g_of_ref = np.array([5., 3, 0, 0]), np.array([0, 1, 1, 2])
    for group in (None, np.arange(4)):
        ref_idx, gen_idx = ic._active_subset(M, group, obs, g_of_ref)
        assert ref_idx.tolist() == [0, 1, 2] and gen_idx.tolist() == [0, 1], (group, ref_idx)


def test_an_observed_label_with_an_empty_matrix_row_gets_an_identity_row(tmp_path: Path) -> None:
    """A grouped matrix from another batch has no row for group b, which drew 3,000 of
    8,000 reads. Without a fallback no source can produce them and the fit distorts
    (0.46 / 0.54 against 0.625 / 0.375); with an identity row it matches them."""
    # One copy per genome: a genome with an unobserved second copy would contradict the
    # data in genome space on its own (the pruning test covers that closure).
    references = ["g0|0|a", "g1|0|b", "g2|0|d"]
    amplicon_dir = tmp_path / "amplicons"
    amplicon_dir.mkdir()
    pd.DataFrame({"genome_id": [r.split("|")[0] for r in references], "refseq": references,
                  "weight": 1.0}).to_csv(amplicon_dir / "translation_table.tsv", sep="\t",
                                         index=False)
    (amplicon_dir / "amplicons.fasta").write_text(
        "".join(f">{r}\n{s}\n" for r, s in zip(references, ["AAAA", "CCCC", "TTTT"])))
    matrix = tmp_path / "M.npz"
    sm.write_grouped(matrix, sparse.csr_array(np.diag([1., 0, 1])),
                     np.arange(3, dtype=np.int32), references)
    observation = tmp_path / "obs.mseq"
    _write_mseq(observation, references, np.array([5000, 3000, 0]))

    for space, table, key in (("genome", "inferred_composition.csv", "genome_id"),
                              ("v4_group", "inferred_v4_groups.csv", "representative_reference")):
        ic.run(SimpleNamespace(
            amplicon_dir=amplicon_dir, mismapping_matrix=matrix, sim_mseq=None,
            obs_mseq=[observation], min_identity=None, sample_id="S1", mode="vi", alpha=0.5,
            steps=500, lr=0.05, num_samples=100, warmup=0, no_mismapping=False, seed=0,
            no_presence=True, presence_prior=0.01, presence_temp=1.0, no_prune=False,
            infer_space=space, taxonomy=None, output_dir=tmp_path / space))
        fitted = pd.read_csv(tmp_path / space / table).set_index(key)["inferred_mean"]
        first = "g0" if space == "genome" else "g0|0|a"
        assert abs(fitted[first] - 0.625) < 0.03, (space, fitted)

    output = tmp_path / "fit.json"
    ccf.run(_fit_args(amplicon_dir, matrix, observation, tmp_path / "v4_group" / table,
                      tmp_path / "v4_group" / "posterior_draws.npz", output))
    assert json.loads(output.read_text())["fit_status"] == "ok", output.read_text()


def test_pruning_sends_dropped_mass_to_sink_labels_exactly() -> None:
    """On every kept label the pruned model predicts what the full model predicts, and its
    sink labels hold exactly the rest: grouped and reference-square, at the built decay
    and re-decayed, with one group dropped whole and one kept only in part."""
    import torch
    import subspecies_infer as si

    rng = np.random.default_rng(3)
    group = np.array([0, 0, 1, 2, 3, 4, 4])
    n = group.max() + 1
    sizes = np.bincount(group).astype(float)
    mask = rng.random((n, n)) < 0.6
    np.fill_diagonal(mask, True)
    distance = np.where(mask, rng.integers(1, 3, (n, n)), 0)
    np.fill_diagonal(distance, 0)
    raw = np.where(mask, rng.random((n, n)) * 0.5 ** distance, 0.0)
    kernel = raw / (raw @ sizes)[:, None]
    ref_idx = np.array([0, 1, 2, 5])      # groups 2 and 3 dropped, group 4 kept in part

    r_true = np.zeros(len(group))
    r_true[ref_idx] = rng.random(len(ref_idx))
    r_true /= r_true.sum()

    def with_distances(dense_values, dense_distance):
        matrix = sparse.csr_array(dense_values)
        rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        return matrix, sparse.csr_array(
            (dense_distance[rows, matrix.indices].astype(float), matrix.indices,
             matrix.indptr), shape=matrix.shape)

    def predict(matrix, distances, matrix_group, proportions, decay):
        decay_kernel = si.DecayKernel(matrix, distances, 0.5, matrix_group)
        return si._apply_mismapping(torch.tensor(proportions), decay_kernel,
                                    torch.tensor(0.8), torch.tensor(decay)).numpy()

    square = np.ix_(group, group)
    for matrix_group, values, dist in ((group, kernel, distance),
                                       (None, kernel[square], distance[square])):
        matrix, distances = with_distances(values, dist)
        sub, sub_group, sub_strata, n_sink = ic._subset_mismapping(
            matrix, matrix_group, ref_idx, (distances, 0.5))
        assert n_sink >= 1
        for decay in (0.5, 0.2):
            full = predict(matrix, distances, matrix_group, r_true, decay)
            pruned = predict(sub, sub_strata[0], sub_group,
                             np.r_[r_true[ref_idx], np.zeros(n_sink)], decay)
            assert np.allclose(pruned[:len(ref_idx)], full[ref_idx]), (pruned, full)
            assert np.isclose(pruned[len(ref_idx):].sum(), 1.0 - full[ref_idx].sum())
