"""Measuring M by simulation at V4-group level, on a batch's active groups only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import pandas as pd  # noqa: E402  (needs local bin directory)

import check_composition_fit as fit  # noqa: E402  (needs local bin directory)
import infer_composition as ic  # noqa: E402  (needs local bin directory)
import select_active_amplicons as saa  # noqa: E402  (needs local bin directory)
import sparse_matrix as sm  # noqa: E402  (needs local bin directory)
import subspecies_infer as si  # noqa: E402  (needs local bin directory)

# Two byte-identical amplicons (group A), one distinct (B), one unobserved and far (C).
A = "ACGTACGTAGGCTTAGCTGGACCTTGACAGTTCCAGGTACCGATTGCAAGCTTACGGATCCGTTAACGGTA"
B = A[:40] + "T" + A[41:]                      # one substitution from A
C = "TTGCAAGCTTACGGATCCGTTAACGGTAACGTACGTAGGCTTAGCTGGACCTTGACAGTTCCAGGTACCGA"
REFS = ["g1|0|A", "g2|0|A", "g3|0|B", "g4|0|C"]
SEQS = [A, A, B, C]


def _amplicons(path: Path) -> Path:
    path.write_text("".join(f">{r}\n{s}\n" for r, s in zip(REFS, SEQS)))
    return path


def _write_mseq(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text("".join(f"{q}\t{hit}\t250\t1.0\n" for q, hit in rows))
    return path


def test_measured_matrix_rows_are_groups_and_stay_row_stochastic(tmp_path: Path) -> None:
    """A group's row is measured once and shared; which duplicate mapseq named is ignored."""
    group = np.array([0, 0, 1, 2])
    # Reads simulated from group A's representative: 90% labelled on its *other* duplicate,
    # 10% leaked to B. Nothing was simulated from C.
    rows = [(f"g1|0|A:{i}", "g2|0|A") for i in range(90)]
    rows += [(f"g1|0|A:{i}", "g3|0|B") for i in range(90, 100)]
    rows += [(f"g3|0|B:{i}", "g3|0|B") for i in range(100)]
    mseq = _write_mseq(tmp_path / "sim.mseq", rows)

    S = si.build_mismapping_grouped([mseq], REFS, group).toarray()
    size = np.bincount(group).astype(float)
    assert np.allclose(S[0] * size, [0.9, 0.1, 0.0]), S[0]
    assert np.allclose(S[1] * size, [0.0, 1.0, 0.0]), S[1]
    assert not S[2].any(), "an unsimulated group must stay empty so inference prunes it"
    assert np.allclose((S[:2] * size).sum(axis=1), 1.0)


def test_a_group_whose_simulated_reads_all_failed_keeps_an_identity_row(tmp_path: Path) -> None:
    group = np.array([0, 0, 1, 2])
    mseq = _write_mseq(tmp_path / "sim.mseq", [(f"g1|0|A:{i}", "unknown|0|X") for i in range(10)])
    S = si.build_mismapping_grouped([mseq], REFS, group).toarray()
    assert np.allclose(S[0] * np.bincount(group), [1.0, 0.0, 0.0]), S[0]


def test_an_active_group_the_simulator_skipped_keeps_an_identity_row(tmp_path: Path) -> None:
    """A group with IUPAC codes draws no simulated read; it must stay a possible source."""
    group = np.array([0, 0, 1, 2])
    mseq = _write_mseq(tmp_path / "sim.mseq", [(f"g1|0|A:{i}", "g1|0|A") for i in range(10)])
    S = si.build_mismapping_grouped([mseq], REFS, group, active=np.array([0, 2])).toarray()
    assert np.allclose(S[2] * np.bincount(group), [0.0, 0.0, 1.0]), S[2]
    assert not S[1].any(), "a group outside the active set still prunes away"


def test_active_groups_take_observed_labels_and_their_neighbours(tmp_path: Path) -> None:
    """Direct sequence comparison, not the mapper: B is active because it is one edit from A."""
    amplicons = _amplicons(tmp_path / "amplicons.fasta")
    obs = _write_mseq(tmp_path / "obs.mseq", [(f"read{i}", "g2|0|A") for i in range(50)])

    refs, group, sequences, active = saa.active_groups(
        amplicons, [obs], tau=1, min_identity=None, max_ambiguous_bases=4, max_postings=4096)
    assert [sequences[g] for g in sorted(active)] == [A, B], "A observed, B its one-edit neighbour"

    _, _, _, exact = saa.active_groups(
        amplicons, [obs], tau=0, min_identity=None, max_ambiguous_bases=4, max_postings=4096)
    assert [sequences[g] for g in exact] == [A]


def test_build_grouped_cli_path_writes_a_grouped_matrix(tmp_path: Path) -> None:
    """--build-mismapping --build-grouped writes the measured kernel, active rows included."""
    amplicon_dir = tmp_path / "amp"
    amplicon_dir.mkdir()
    _amplicons(amplicon_dir / "amplicons.fasta")
    pd.DataFrame({"genome_id": [r.split("|")[0] for r in REFS], "refseq": REFS,
                  "weight": 1.0}).to_csv(amplicon_dir / "translation_table.tsv", sep="\t",
                                         index=False)
    # Only group A was simulated from; C is active but the simulator skipped it.
    active = tmp_path / "active.fasta"
    active.write_text(f">g1|0|A\n{A}\n>g4|0|C\n{C}\n")
    mseq = _write_mseq(tmp_path / "sim.mseq", [(f"g1|0|A:{i}", "g2|0|A") for i in range(100)])

    ic.run(SimpleNamespace(
        amplicon_dir=amplicon_dir, sim_mseq=[mseq], mismapping_matrix=None,
        min_identity=None, build_mismapping=True, build_grouped=True,
        active_amplicons=active, output_dir=tmp_path / "out"))

    written = tmp_path / "out" / "mismapping_matrix.npz"
    assert sm.is_grouped(written)
    M, group, strata = sm.read_grouped(written, REFS, strata=True)
    assert strata is None, "a measured matrix has no distances to re-decay"
    size = np.bincount(group).astype(float)
    rows = (M @ size)
    a, c = group[REFS.index("g1|0|A")], group[REFS.index("g4|0|C")]
    assert np.isclose(rows[a], 1.0) and np.isclose(rows[c], 1.0), rows
    assert np.isclose(rows[group[REFS.index("g3|0|B")]], 0.0), "inactive group stays empty"


# ── Rectangular kernels: sources (panel amplicons) x labels (database groups) ──


def test_tally_kernel_counts_panel_sources_against_database_labels(tmp_path: Path) -> None:
    rows = [("src_x:0", "g1|0|A"), ("src_x:1", "g2|0|A"), ("src_x:2", "g3|0|B"),
            ("src_y:0", "not|in|db")]
    mseq = _write_mseq(tmp_path / "sim.mseq", rows)
    label = dict(zip(REFS, [0, 0, 1, 2]))
    counts, seen = si.tally_kernel([mseq], {"src_x": 0, "src_y": 1}, label, 2, 3)
    assert counts.toarray().tolist() == [[2, 1, 0], [0, 0, 0]]
    assert seen.tolist() == [True, True], "a source with only unlabelled hits was still seen"


def test_rectangular_kernel_round_trips_and_rejects_a_bad_home(tmp_path: Path) -> None:
    K = sparse.csr_array(np.array([[0.9, 0.1, 0.0], [0.0, 0.0, 0.0]]))
    path = tmp_path / "kernel.npz"
    sm.write_kernel(path, K, ["s0", "s1"], ["l0", "l1", "l2"], np.array([0, 2]),
                    ref_headers=REFS, label_of_ref=np.array([0, 0, 1, 2]),
                    n_simulated=np.array([100, 0]), provenance={"seed": 7})
    k = sm.read_kernel(path)
    assert np.array_equal(k.kernel.toarray(), K.toarray()) and k.home.tolist() == [0, 2]
    assert k.ref_headers == REFS and k.label_of_ref.tolist() == [0, 0, 1, 2]
    assert k.n_simulated.tolist() == [100, 0] and k.n_unmapped is None
    assert k.provenance == {"seed": 7} and not sm.is_grouped(path)
    assert sm.home_entries(k.kernel, k.home).tolist() == [0.9, 0.0]

    sm.write_kernel(path, K, ["s0", "s1"], ["l0", "l1", "l2"], np.array([0, 3]))
    with pytest.raises(ValueError, match="out of range"):
        sm.read_kernel(path)


def test_rectangular_projection_sends_unscaled_mass_to_each_home_label() -> None:
    """With a non-identity home, torch and numpy projections equal r @ ((1-s)E + sK)."""
    import torch

    rng = np.random.default_rng(0)
    K = rng.random((3, 5)) * (rng.random((3, 5)) < 0.6) + np.eye(3, 5)
    K /= K.sum(axis=1, keepdims=True)
    home = np.array([4, 0, 4])              # two sources share a home; none is "itself"
    E = np.zeros_like(K)
    E[np.arange(3), home] = 1.0
    r = rng.random(3)
    r /= r.sum()
    csr = sparse.csr_array(K)
    M = (torch.tensor(K).to_sparse(), torch.tensor(sm.home_entries(csr, home)), None,
         torch.tensor(home))
    for s in (1.0, 0.4):
        dense = r @ ((1.0 - s) * E + s * K)
        got = si._apply_mismapping(torch.tensor(r), M, torch.tensor(s)).numpy()
        assert np.allclose(got, dense, atol=1e-12), (s, got, dense)
        assert np.allclose(fit._apply_mismapping(r, csr, None, s, home), dense, atol=1e-12)
    assert np.allclose(r @ K, si._apply_mismapping(torch.tensor(r), M, torch.tensor(1.0)).numpy())


def test_rectangular_kernel_over_groups_reproduces_the_grouped_path() -> None:
    """Sources = labels = groups with home = identity is the grouped model summed over each
    group's members: a panel equal to the active set reduces to the existing case. The
    model is the same function of theta and s, so the likelihood and fit are too."""
    import torch

    group = np.array([0, 0, 1, 2, 3, 3, 3])
    n = int(group.max()) + 1
    size = np.bincount(group).astype(float)
    rng = np.random.default_rng(1)
    raw = rng.random((n, n)) * (rng.random((n, n)) < 0.7)
    np.fill_diagonal(raw, 6.0)              # keeps 1 - s + s*S[g, g] >= 0 at s = 1.3
    S = raw / (raw @ size)[:, None]
    K = S * size                            # row-stochastic over groups
    grouped = (torch.tensor(S).to_sparse(),
               torch.tensor(sm.grouped_diagonal(sparse.csr_array(S), group)),
               torch.tensor(group))
    rect = (torch.tensor(K).to_sparse(), torch.tensor(np.diag(K).copy()), None,
            torch.arange(n))
    r_ref = rng.random(len(group))
    r_ref /= r_ref.sum()
    for s in (0.5, 1.0, 1.3):
        by_ref = si._apply_mismapping(torch.tensor(r_ref), grouped, torch.tensor(s)).numpy()
        by_group = si._apply_mismapping(torch.tensor(np.bincount(group, r_ref)), rect,
                                        torch.tensor(s)).numpy()
        assert np.allclose(np.bincount(group, by_ref), by_group, atol=1e-8), s


def test_rectangular_pruning_sends_dropped_labels_to_one_sink_exactly() -> None:
    """Kept labels predict what the full kernel predicts and the sink holds the rest,
    including the unscaled mass of a source whose home label was dropped."""
    import torch

    rng = np.random.default_rng(4)
    K = rng.random((3, 6))
    K /= K.sum(axis=1, keepdims=True)
    home = np.array([1, 5, 1])
    labels = np.array([0, 1, 3])            # drops label 5, source 1's home
    r = rng.random(3)
    r /= r.sum()

    def predict(kernel, h):
        M = (torch.tensor(kernel.toarray()).to_sparse(),
             torch.tensor(sm.home_entries(kernel, h)), None, torch.tensor(h))
        return si._apply_mismapping(torch.tensor(r), M, torch.tensor(0.7)).numpy()

    full = predict(sparse.csr_array(K), home)
    sub, sub_home, n_sink, _ = ic._subset_kernel(sparse.csr_array(K), home, labels)
    assert n_sink == 1 and sub_home.tolist() == [1, 3, 1]
    assert np.allclose(sub.sum(axis=1), 1.0)
    pruned = predict(sub, sub_home)
    assert np.allclose(pruned[:3], full[labels], atol=1e-12)
    assert np.isclose(pruned[3], np.delete(full, labels).sum(), atol=1e-12)


def test_panel_inference_recovers_genomes_and_background(tmp_path: Path) -> None:
    """gA/gC share source a (not separable), gB's source is relabelled off itself, and
    reads on a label no source reaches become background."""
    K = sparse.csr_array(np.array([[0.7, 0.3, 0.0, 0.0, 0.0],
                                   [0.0, 0.1, 0.9, 0.0, 0.0]]))
    kernel = tmp_path / "panel_kernel.npz"
    headers = REFS + ["g5|0|D", "g6|0|E"]
    sm.write_kernel(kernel, K, ["a", "b"], [f"l{i}" for i in range(5)], np.array([0, 2]),
                    ref_headers=headers, label_of_ref=np.array([0, 0, 1, 2, 3, 4]))
    prep = tmp_path / "prep"
    prep.mkdir()
    pd.DataFrame({"genome_id": ["gA", "gB", "gC"], "source": ["a", "b", "a"],
                  "weight": 1.0}).to_csv(prep / "panel_translation.tsv", sep="\t", index=False)
    # Truth gA+gC 0.5, gB 0.4, background 0.1 (all on g5|0|D) -> labels at s = 1.
    per_label = {"g1|0|A": 3500, "g3|0|B": 1900, "g4|0|C": 3600, "g5|0|D": 1000}
    obs = _write_mseq(tmp_path / "obs.mseq", [(f"r{h}{i}", h) for h, n in per_label.items()
                                              for i in range(n)])
    out = tmp_path / "out"
    ic.run(SimpleNamespace(
        amplicon_dir=prep, mismapping_matrix=kernel, sim_mseq=None, obs_mseq=[obs],
        min_identity=None, sample_id="S", mode="mle", alpha=1.0, steps=3000, lr=0.02,
        num_samples=10, warmup=0, no_mismapping=False, no_presence=True,
        presence_prior=0.5, presence_temp=0.5, seed=0, min_infer_reads=1, output_dir=out))

    comp = pd.read_csv(out / "inferred_composition.csv", index_col="genome_id")
    got = comp.inferred_mean
    assert abs(got.gA + got.gC - 0.5) < 0.02 and abs(got.gB - 0.4) < 0.02, got
    assert abs(got.background - 0.1) < 0.02, got
    assert comp.resolution.to_dict() == {"gA": "not_identifiable", "gB": "identifiable",
                                         "gC": "not_identifiable", "background": "background"}
    d = pd.read_csv(out / "inference_diagnostics.csv").iloc[0]
    assert d.kernel_format == "rectangular" and d.n_sources == 2
    assert d.n_labels_observed == 4 and d.n_labels_unexplained == 1
    assert np.isclose(d.observed_unexplained_fraction, 0.1)
    assert np.isclose(d.unexplained_fraction, got.background)


def test_panel_align_kernel_redecays_exactly_and_fits_a_latent_decay(tmp_path: Path) -> None:
    """align: the error-free read lands on the home label and a neighbour within tau gets
    c ** d. The stored distances re-decay the rectangular kernel exactly, through pruning
    to a subset of labels, and a latent-decay panel fit passes its fit check."""
    import torch
    import build_panel_kernel as bpk

    D = C[:30] + ("A" if C[30] != "A" else "T") + C[31:]   # 1 edit from C, not in the DB
    a, d = bpk.v4g(A), bpk.v4g(D)
    prep = tmp_path / "prep"
    prep.mkdir()
    pd.DataFrame({"source": [a, d], "genomes": ["gX", "gY"], "in_db": [True, False]}).to_csv(
        prep / "sources.tsv", sep="\t", index=False)
    (prep / "sources.fasta").write_text(f">{a}\n{A}\n>{d}\n{D}\n")
    pd.DataFrame({"genome_id": ["gX", "gY"], "source": [a, d], "weight": 1.0}).to_csv(
        prep / "panel_translation.tsv", sep="\t", index=False)
    db = _amplicons(tmp_path / "amplicons.fasta")
    kernel = prep / "align_kernel.npz"
    args = dict(prepared=prep, db_amplicons=db, tau=1, distance_decay=0.1,
                ambiguity_weight=0.3, out=kernel)
    with pytest.raises(SystemExit, match="no home label"):
        bpk.align(SimpleNamespace(home_mseq=None, **args))       # d is not in the database
    home = _write_mseq(tmp_path / "home.mseq", [(a, "g1|0|A"), (d, "g4|0|C")])
    bpk.align(SimpleNamespace(home_mseq=home, **args))

    k = sm.read_kernel(kernel)
    lab = [k.label_ids.index(bpk.v4g(g)) for g in (A, B, C)]
    K, dist = k.kernel.toarray(), k.strata[0].toarray()
    assert np.allclose(K[0, lab], [1 / 1.1, 0.1 / 1.1, 0]) and np.allclose(K[1, lab], [0, 0, 1])
    assert dist[0, lab[1]] == 1 and k.strata[1] == 0.1 and k.home.tolist() == [lab[0], lab[2]]

    r, s, c = torch.tensor([0.6, 0.4]), torch.tensor(0.8), torch.tensor(0.3)
    redecayed = np.where(K > 0, K * (0.3 / 0.1) ** dist, 0.0)
    redecayed /= redecayed.sum(axis=1, keepdims=True)
    E = np.zeros_like(K)
    E[[0, 1], k.home] = 1.0
    full = si._apply_mismapping(r, si.DecayKernel(k.kernel, k.strata[0], 0.1, home=k.home),
                                s, c).numpy()
    assert np.allclose(full, r.numpy() @ ((1 - 0.8) * E + 0.8 * redecayed))
    keep = np.array([lab[0], lab[2]])                           # drops B, a's d=1 neighbour
    sub, sub_home, n_sink, sub_d = ic._subset_kernel(k.kernel, k.home, keep, k.strata[0])
    pruned = si._apply_mismapping(r, si.DecayKernel(sub, sub_d, 0.1, home=sub_home), s, c).numpy()
    assert n_sink == 1 and np.allclose(pruned[:2], full[keep])
    assert np.isclose(pruned[2:].sum(), 1.0 - full[keep].sum())

    # Reads drawn at theta (0.6, 0.4), s = 1, c = 0.1; the latent fit and its check.
    obs = _write_mseq(tmp_path / "obs.mseq", [(f"r{h}{i}", h) for h, n in
                                              (("g1|0|A", 5455), ("g3|0|B", 545),
                                               ("g4|0|C", 4000)) for i in range(n)])
    out = tmp_path / "out"
    ic.run(SimpleNamespace(
        amplicon_dir=prep, mismapping_matrix=kernel, sim_mseq=None, obs_mseq=[obs],
        min_identity=None, sample_id="S", mode="vi", alpha=0.5, steps=1500, lr=0.05,
        num_samples=200, warmup=0, no_mismapping=False, no_presence=True, presence_prior=0.5,
        presence_temp=0.5, seed=0, min_infer_reads=1, infer_distance_decay=True,
        output_dir=out))
    comp = pd.read_csv(out / "inferred_composition.csv", index_col="genome_id").inferred_mean
    assert abs(comp.gX - 0.6) < 0.03 and abs(comp.gY - 0.4) < 0.03, comp
    diag = pd.read_csv(out / "inference_diagnostics.csv").iloc[0]
    assert diag.infer_distance_decay and diag.kernel_method == "align" and diag.distance_decay > 0
    fit_json = tmp_path / "fit.json"
    fit.run(SimpleNamespace(
        composition=out / "inferred_composition.csv", posterior_draws=out / "posterior_draws.npz",
        obs_mseq=[obs], amplicon_dir=prep, mismapping_matrix=kernel, min_identity=None,
        min_infer_reads=1, ppc_draws=200, ppc_low_tail=0.01, ppc_high_tail=0.99, seed=1,
        output=fit_json))
    import json
    assert json.loads(fit_json.read_text())["fit_status"] == "ok", fit_json.read_text()


def test_panel_align_candidate_probes_and_auto_decay(monkeypatch) -> None:
    """Limit candidate search to panel sources without losing an ambiguous target."""
    import build_mismapping_align as bma
    import build_panel_kernel as bpk

    sequences = [A, B, C, C[:-1] + "N"]
    pairs = {tuple(pair) for pair in bma.pigeonhole_candidates(
        sequences, tau=1, max_ambiguous_bases=4, max_postings=1 << 30,
        probes=np.array([0, 2], dtype=np.int64),
    )}
    assert (0, 1) in pairs and (2, 3) in pairs

    monkeypatch.setattr(bma, "measure_error_rate", lambda *args: 0.013)
    assert bpk._align_decay(SimpleNamespace(
        distance_decay="auto", model_pt=None, flat_sub_rate=0.005,
        flat_ins_rate=0.0005, flat_del_rate=0.0005,
    ), bma) == 0.013


def test_panel_kernel_prepare_and_build(tmp_path: Path) -> None:
    """Panel sources against database labels: a shared copy, a copy absent from the
    database whose home is relabelled, an unhit read, and row-stochastic rows."""
    import build_panel_kernel as bpk

    D = C[:30] + ("A" if C[30] != "A" else "T") + C[31:]   # not in the database
    fwd, tail = "GTGCCAGCAGCCGCGGTAA", si.revcomp("GGACTACAAGGGTATCTAAT")
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "gX.amplicons.fasta").write_text(f">x1\n{fwd}{A}{tail}\n>x2\n{fwd}{D}{tail}\n")
    (panel / "gY.amplicons.fasta").write_text(f">y1\n{fwd}{A}{tail}\n>y2\n{A}\n")  # y2 has no primers
    db = _amplicons(tmp_path / "amplicons.fasta")
    prep = tmp_path / "prep"
    bpk.prepare(SimpleNamespace(panel_amplicons=panel, db_amplicons=db, out=prep, alias=[],
                                panel_taxa=None, db_taxonomy=None, max_taxon_sources=200,
                                fwd_primer=si.DEFAULT_FWD_PRIMER,
                                rev_primer=si.DEFAULT_REV_PRIMER, max_mismatch=2))

    a, d = bpk.v4g(A), bpk.v4g(D)
    tr = pd.read_csv(prep / "panel_translation.tsv", sep="\t")
    assert sorted(map(tuple, tr.to_numpy().tolist())) == [("gX", a, 0.5), ("gX", d, 0.5),
                                                          ("gY", a, 1.0)]
    sources = pd.read_csv(prep / "sources.tsv", sep="\t", index_col="source")
    assert sources.genomes[a] == "gX;gY" and sources.in_db[a] and not sources.in_db[d]
    assert dict(si.read_fasta(prep / "sources.fasta")) == {a: A, d: D}

    home = _write_mseq(tmp_path / "home.mseq", [(a, "g1|0|A"), (d, "g4|0|C")])
    sim = _write_mseq(tmp_path / "sim.mseq",
                      [(f"{a}:{i}", "g2|0|A") for i in range(8)]
                      + [(f"{a}:{i}", "g3|0|B") for i in (8, 9)]
                      + [(f"{d}:{i}", "g4|0|C") for i in range(3)]
                      + [(f"{d}:{i}", "g3|0|B") for i in range(3, 9)])
    with open(sim, "a") as fh:
        fh.write(f"{d}:9\t\t\t\n")                          # unhit
    out = prep / "panel_kernel.npz"
    bpk.build(SimpleNamespace(prepared=prep, db_amplicons=db, home_mseq=home, sim_mseq=sim,
                              min_identity=None, provenance=["seed=7"], out=out))

    k = sm.read_kernel(out)                                  # validates row-stochastic rows
    lab = {g: k.label_ids.index(bpk.v4g(g)) for g in (A, B, C)}
    rows = {s: k.kernel.toarray()[k.source_ids.index(s)] for s in (a, d)}
    assert np.isclose(rows[a][lab[A]], 0.8) and np.isclose(rows[a][lab[B]], 0.2)
    assert np.isclose(rows[d][lab[C]], 1 / 3) and np.isclose(rows[d][lab[B]], 2 / 3)
    assert k.home[k.source_ids.index(d)] == lab[C], "home is MAPseq's label, not the source"
    assert k.n_simulated.tolist() == [10, 10] and sorted(k.n_unmapped.tolist()) == [0, 1]
    assert k.label_of_ref.tolist() == [lab[A], lab[A], lab[B], lab[C]]
    assert k.provenance["seed"] == "7"
    table = pd.read_csv(prep / "panel_sources.tsv", sep="\t", index_col="source")
    assert np.isclose(table.home_mass[a], 0.8) and np.isclose(table.home_mass[d], 1 / 3, atol=1e-4)
    assert np.isclose(table.unmapped_fraction[d], 0.1) and table.block[a] != table.block[d]
