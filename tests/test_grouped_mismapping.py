"""The grouped mis-mapping matrix must be the reference-square one, without building it.

``M[a, j] == S[group[a], group[j]]``, so every consumer — the file round trip, the
diagonal, and the ``r_true @ M_eff`` product Pyro differentiates through — has to agree
with the dense form it replaces.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import build_mismapping_align as bma  # noqa: E402  (needs sys.path)
import sparse_matrix as sm  # noqa: E402  (needs sys.path)
import subspecies_infer as si  # noqa: E402  (needs sys.path)

SEQUENCES = [
    "ACGTACGTACGTTTGACCAGGTACG",
    "ACGTACGTACGTTTGACCAGGTACG",          # exact duplicate of the first
    "ACGTACGTACGTTTGACCAGGTACA",          # one substitution away
    "ACGTACGTACGTTTGACCAGGTACN",          # ambiguous, compatible with both above
    "TTTTTTTTTTTTTTTTTTTTTTTTT",          # isolated
]


def _fasta(tmp_path: Path) -> Path:
    path = tmp_path / "amplicons.fasta"
    path.write_text("".join(f">r{i}\n{s}\n" for i, s in enumerate(SEQUENCES)))
    return path


def _dense(tau: int, weight: float, literal: bool = False) -> np.ndarray:
    """The reference-square kernel this is all meant to reproduce.

    ``literal`` is the exact-hash convention: byte identity, so an ``N`` does *not* join
    the group of the sequences it is compatible with. That is a deliberate difference
    from every tau >= 1 path, which is IUPAC-aware.
    """
    distances = np.array([
        [0 if a == b else 1 if literal else bma.bounded_iupac_distance(a, b, tau)
         for b in SEQUENCES] for a in SEQUENCES])
    return bma.tie_cluster_matrix(distances, tau=0 if literal else tau,
                                  weights=bma.ambiguity_weights(SEQUENCES, weight),
                                  multiplicity=bma.byte_multiplicity(SEQUENCES))


@pytest.mark.parametrize("tau,weight", [(0, 1.0), (1, 1.0), (1, 0.3), (2, 0.3)])
def test_grouped_matches_the_dense_tie_cluster(tmp_path: Path, tau: int, weight: float) -> None:
    fasta = _fasta(tmp_path)
    if tau:
        _, cluster, group, _, _ = bma.build_kmer_grouped(fasta, tau, 4, weight, 1 << 30)
    else:
        _, cluster, group, _ = bma.build_exact_grouped(fasta)
    expanded = cluster.toarray()[np.ix_(group, group)]
    assert np.allclose(expanded, _dense(tau, weight, literal=not tau))
    assert np.allclose(expanded.sum(axis=1), 1.0)


@pytest.mark.parametrize("weight", [1.0, 0.3])
@pytest.mark.parametrize("copies", [1, 10, 100])
def test_leak_onto_a_neighbour_ignores_its_copy_number(
        tmp_path: Path, copies: int, weight: float) -> None:
    """Kernel version 2: MAPseq's leak onto a one-edit neighbour barely moves with how many
    references carry it (dev/mapseq_multiplicity.md), so the kernel's must not move at all.
    """
    source = SEQUENCES[0]
    neighbour = source[:-1] + "A"                               # one substitution
    if weight != 1.0:
        neighbour = "N" + neighbour[1:]                         # still one edit, amb = w
    fasta = tmp_path / "a.fasta"
    fasta.write_text(f">a\n{source}\n" + "".join(f">b{i}\n{neighbour}\n" for i in range(copies)))
    refs, cluster, group, _, _ = bma.build_kmer_grouped(fasta, 1, 4, weight, 1 << 30, 0.01)
    sizes = np.bincount(group).astype(np.float64)
    a, b = group[refs.index("a")], group[refs.index("b0")]
    assert np.isclose(cluster[a, b] * sizes[b], 0.01 * weight / (1 + 0.01 * weight))
    assert np.allclose(cluster @ sizes, 1.0)                    # sum_b S[a, b] size[b] = 1


def test_redecaying_a_built_kernel_equals_building_at_that_decay(tmp_path: Path) -> None:
    """``M(c) = rownorm(M(c0) * (c / c0) ** d)`` survives the ``1 / size`` factor."""
    torch = pytest.importorskip("torch")
    import check_composition_fit as ccf                         # noqa: PLC0415

    fasta = _fasta(tmp_path)
    _, built, group, _, distances = bma.build_kmer_grouped(fasta, 1, 4, 0.3, 1 << 30, 0.05)
    _, direct, _, _, _ = bma.build_kmer_grouped(fasta, 1, 4, 0.3, 1 << 30, 0.005)
    redecayed = ccf._kernel_at_decay(built, group, (distances, 0.05), 0.005)
    assert np.allclose(redecayed.toarray(), direct.toarray())
    values, _ = si.DecayKernel(built, distances, 0.05, group).kernel(
        torch.tensor(0.005, dtype=torch.float64))
    assert np.allclose(values.numpy(), direct.data)


def test_kernel_version_is_stored_hashed_and_warned_about(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A stale kernel must change MATRIX_KEY, and a file that predates versioning warns."""
    key_module = (ROOT / "modules" / "local" / "matrix_key" / "main.nf").read_text()
    assert f"kernel_version: {sm.KERNEL_VERSION}," in key_module

    path = tmp_path / "m.npz"
    sm.write_grouped(path, sparse.csr_array(np.eye(1)), np.zeros(1, dtype=np.int32), ["r"])
    assert sm.stored_kernel_version(path) == sm.KERNEL_VERSION
    with np.load(path) as archive:
        legacy = {key: archive[key] for key in archive.files if key != "kernel_version"}
    np.savez(path, **legacy)
    assert sm.stored_kernel_version(path) == 1
    with caplog.at_level("WARNING"):
        sm.read_grouped(path, ["r"])
    assert "kernel_version 1" in caplog.text


def test_round_trip_reorders_and_rejects_a_foreign_reference_set(tmp_path: Path) -> None:
    refs = [f"r{i}" for i in range(len(SEQUENCES))]
    _, cluster, group, _, _ = bma.build_kmer_grouped(_fasta(tmp_path), 1, 4, 1.0, 1 << 30)
    path = tmp_path / "m.npz"
    sm.write_grouped(path, cluster, group, refs)
    assert sm.is_grouped(path)

    reversed_refs = refs[::-1]
    back, back_group = sm.read_grouped(path, reversed_refs)
    dense = _dense(1, 1.0)[::-1, ::-1]
    assert np.allclose(back.toarray()[np.ix_(back_group, back_group)], dense)
    assert np.allclose(sm.grouped_diagonal(back, back_group), np.diag(dense))
    with pytest.raises(ValueError):
        sm.read_grouped(path, refs[:-1] + ["elsewhere"])


def test_apply_mismapping_agrees_with_the_dense_product() -> None:
    torch = pytest.importorskip("torch")
    dense = _dense(1, 0.3)
    unique, group = np.unique(np.array([0, 0, 1, 2, 3]), return_inverse=True)
    assert len(unique) == 4
    cluster = sparse.csr_array(dense[[0, 2, 3, 4]][:, [0, 2, 3, 4]])
    coo = torch.sparse_coo_tensor(
        torch.tensor(np.vstack(cluster.nonzero()), dtype=torch.long),
        torch.tensor(cluster.data, dtype=torch.float64), size=cluster.shape).coalesce()
    grouped = (coo, torch.tensor(sm.grouped_diagonal(cluster, group), dtype=torch.float64),
               torch.tensor(group, dtype=torch.long))

    r_true = torch.tensor([0.4, 0.1, 0.2, 0.25, 0.05], dtype=torch.float64)
    reference = torch.tensor(dense, dtype=torch.float64)
    for scale in (0.0, 0.5, 1.0, 1.7):
        s = torch.tensor(scale, dtype=torch.float64)
        assert torch.allclose(si._apply_mismapping(r_true, grouped, s),
                              si._apply_mismapping(r_true, reference, s)), scale


def test_infer_composition_loads_a_grouped_matrix(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    import infer_composition                                    # noqa: PLC0415

    refs = ["g1|0|A", "g1|1|A", "g2|0|B"]
    amplicon_dir = tmp_path / "amplicons"
    amplicon_dir.mkdir()
    import pandas as pd                                         # noqa: PLC0415
    pd.DataFrame({"genome_id": ["g1", "g1", "g2"], "refseq": refs,
                  "weight": [0.5, 0.5, 1.0]}).to_csv(
        amplicon_dir / "translation_table.tsv", sep="\t", index=False)
    matrix = tmp_path / "mismapping_matrix.npz"
    # g1's two copies share an amplicon; g2 has its own.
    sm.write_grouped(matrix, sparse.csr_array(np.array([[0.5, 0.0], [0.0, 1.0]])),
                     np.array([0, 0, 1], dtype=np.int32), refs)
    observed = tmp_path / "obs.mseq"
    observed.write_text("\n".join(f"read{i}\t{refs[i % 3]}\t100\t0.99" for i in range(30)) + "\n")

    out = tmp_path / "out"
    from types import SimpleNamespace                           # noqa: PLC0415
    infer_composition.run(SimpleNamespace(
        amplicon_dir=amplicon_dir, mismapping_matrix=matrix,
        mismapping_matrix_path=str(matrix), mismapping_group_id="grp", sim_mseq=None,
        obs_mseq=[observed], min_identity=None, build_mismapping=False, sample_id="S1",
        output_dir=out, mode="vi", alpha=0.5, no_mismapping=False, no_presence=False,
        presence_prior=0.5, presence_temp=0.1, num_samples=10, warmup=10, steps=20,
        lr=0.02, seed=0,
    ))
    import csv                                                  # noqa: PLC0415
    rows = list(csv.DictReader((out / "inferred_composition.csv").open()))
    assert [row["genome_id"] for row in rows] == ["g1", "g2"]
    assert abs(sum(float(row["inferred_mean"]) for row in rows) - 1.0) < 1e-6, rows


def test_end_to_end_from_reference_fasta_to_composition(tmp_path: Path) -> None:
    """The contract the benchmark pipeline consumes: refs -> amplicons -> M -> composition.

    `BUILD_SUPERRESOLUTION_MISMAPPING` runs the matrix build once per reference set and
    hands the file to every inference run through `--mismapping_matrix`. Both grouped
    backends have to survive that hand-off with no mapper in between.
    """
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    import csv                                                  # noqa: PLC0415
    from types import SimpleNamespace                           # noqa: PLC0415

    import infer_composition                                    # noqa: PLC0415

    amplicon_dir = tmp_path / "amplicons"
    si.stage_amplicons(SimpleNamespace(
        db_fasta=ROOT / "tests" / "data" / "refs.fasta", output_dir=amplicon_dir,
        fwd_primer=si.DEFAULT_FWD_PRIMER, rev_primer=si.DEFAULT_REV_PRIMER,
        primer_mismatches=3,
    ))
    refseqs = [header for header, _ in si.read_fasta(amplicon_dir / "amplicons.fasta")]
    assert refseqs, "the fixture must amplify"

    for backend, tau in (("exact-hash", 0), ("kmer", 1)):
        matrix = tmp_path / f"{backend}.npz"
        if tau:
            refs, cluster, group, _, _ = bma.build_kmer_grouped(
                amplicon_dir / "amplicons.fasta", tau, 4, 0.3, 4096)
        else:
            refs, cluster, group, _ = bma.build_exact_grouped(amplicon_dir / "amplicons.fasta")
        sm.write_grouped(matrix, cluster, group, refs)

        observed = tmp_path / f"{backend}.mseq"
        observed.write_text("\n".join(
            f"read{i}\t{refseqs[i % len(refseqs)]}\t100\t0.99" for i in range(40)) + "\n")
        out = tmp_path / f"out_{backend}"
        infer_composition.run(SimpleNamespace(
            amplicon_dir=amplicon_dir, mismapping_matrix=matrix,
            mismapping_matrix_path=str(matrix), mismapping_group_id="grp", sim_mseq=None,
            obs_mseq=[observed], min_identity=None, build_mismapping=False,
            sample_id=f"S_{backend}", output_dir=out, mode="vi", alpha=0.5,
            no_mismapping=False, no_presence=False, presence_prior=0.5, presence_temp=1.0,
            num_samples=10, warmup=10, steps=20, lr=0.02, seed=0,
        ))
        # The exact columns normalize_sr_profile.py reads out of this file.
        rows = list(csv.DictReader((out / "inferred_composition.csv").open()))
        assert rows and {"genome_id", "inferred_mean"} <= set(rows[0]), rows[:1]
        assert abs(sum(float(row["inferred_mean"]) for row in rows) - 1.0) < 1e-6, rows


def test_pruning_drops_unreachable_genomes_from_a_grouped_fit(tmp_path: Path) -> None:
    """A genome no read can reach is pruned out of the fit but still reported at zero.

    This is the database-scale case: the reference set is GTDB while a sample touches a
    few hundred of its amplicons, and the rest only cost the fit a dimension. `g1`'s two
    copies share an amplicon with nothing else, `g2` has its own, and `ghost` is never
    observed — so `ghost` leaves the fit while `g1`/`g2` are unaffected by its absence.
    """
    pytest.importorskip("torch")
    pytest.importorskip("pyro")
    import csv                                                  # noqa: PLC0415
    from types import SimpleNamespace                           # noqa: PLC0415

    import pandas as pd                                         # noqa: PLC0415

    import infer_composition                                    # noqa: PLC0415

    refs = ["g1|0|A", "g1|1|A", "g2|0|B", "ghost|0|C"]
    amplicon_dir = tmp_path / "amplicons"
    amplicon_dir.mkdir()
    pd.DataFrame({"genome_id": ["g1", "g1", "g2", "ghost"], "refseq": refs,
                  "weight": [0.5, 0.5, 1.0, 1.0]}).to_csv(
        amplicon_dir / "translation_table.tsv", sep="\t", index=False)
    matrix = tmp_path / "mismapping_matrix.npz"
    sm.write_grouped(matrix, sparse.csr_array(np.diag([0.5, 1.0, 1.0])),
                     np.array([0, 0, 1, 2], dtype=np.int32), refs)
    observed = tmp_path / "obs.mseq"
    observed.write_text("\n".join(
        f"read{i}\t{refs[i % 3]}\t100\t0.99" for i in range(3000)) + "\n")

    # A well-powered fit on purpose. Pruning is not a bit-identical transformation: the
    # Dirichlet prior is over the genomes being fitted, so dropping one redistributes
    # prior mass over the survivors. That matters when the likelihood is flat and a
    # handful of reads is all there is; it washes out as soon as the data says anything,
    # which is the regime this runs in.
    out = tmp_path / "out"
    arguments = SimpleNamespace(
        amplicon_dir=amplicon_dir, mismapping_matrix=matrix,
        mismapping_matrix_path=str(matrix), mismapping_group_id="grp", sim_mseq=None,
        obs_mseq=[observed], min_identity=None, build_mismapping=False, sample_id="S1",
        output_dir=out, mode="vi", alpha=0.5, no_mismapping=False, no_presence=False,
        presence_prior=0.5, presence_temp=0.1, num_samples=100, warmup=10, steps=500,
        lr=0.05, seed=0, no_prune=False,
    )
    infer_composition.run(arguments)

    rows = {row["genome_id"]: row for row in
            csv.DictReader((out / "inferred_composition.csv").open())}
    assert set(rows) == {"g1", "g2", "ghost"}, rows
    assert float(rows["ghost"]["inferred_mean"]) == 0.0, rows
    assert float(rows["ghost"]["presence_prob"]) == 0.0, rows
    assert abs(sum(float(row["inferred_mean"]) for row in rows.values()) - 1.0) < 1e-6, rows

    diagnostics = next(csv.DictReader((out / "inference_diagnostics.csv").open()))
    assert (diagnostics["n_genomes_fitted"], diagnostics["n_genomes"]) == ("2", "3")
    assert diagnostics["n_refs_fitted"] == "3", diagnostics

    # --no-prune must reach the same answer, or this is a different model rather than a
    # cheaper route to the same one.
    arguments.no_prune, arguments.output_dir = True, tmp_path / "out_full"
    infer_composition.run(arguments)
    full = {row["genome_id"]: row for row in
            csv.DictReader((tmp_path / "out_full" / "inferred_composition.csv").open())}
    # A few percent, not equality: see the prior note above. A pruning bug that dropped
    # a genome the reads do reach moves a survivor by a third of the composition, not by
    # a percent, so this still catches one.
    for genome in ("g1", "g2"):
        assert abs(float(rows[genome]["inferred_mean"])
                   - float(full[genome]["inferred_mean"])) < 0.05, (rows, full)


def _home_mseq(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "home.mseq"
    path.write_text("#query\tdbhit\tbitscore\tidentity\n"
                    + "".join(f"{q}\t{hit}\t25\t1\n" for q, hit in rows))
    return path


def test_home_probes_are_the_amplicon_and_single_substitutions(tmp_path: Path) -> None:
    out = tmp_path / "probes.fasta"
    assert bma.write_home_probes(SEQUENCES[:1], out, 5) == 6
    records = list(bma.iter_fasta(out))
    key = bma.v4g(SEQUENCES[0])
    assert records[0] == (f"{key}:e", SEQUENCES[0])
    for i, (header, sequence) in enumerate(records[1:]):
        assert header == f"{key}:{i}"
        assert sum(a != b for a, b in zip(sequence, SEQUENCES[0])) == 1


def test_home_distribution_replaces_the_distance_zero_mass(tmp_path: Path) -> None:
    """Group 0 (r0, r1) is sent to r2 when error-free and 3:1 to itself:r2 with one error."""
    fasta = _fasta(tmp_path)
    g0, g2 = bma.v4g(SEQUENCES[0]), bma.v4g(SEQUENCES[2])
    mseq = _home_mseq(tmp_path, [(f"{g0}:e", "r2"), (f"{g0}:0", "r0"), (f"{g0}:1", "r1"),
                                 (f"{g0}:2", "r0"), (f"{g0}:3", "r2"),
                                 (g2, "r2")])               # suffix-less: error-free
    e = 0.01
    free = (1 - e) ** len(SEQUENCES[0])
    _, cluster, group, _ = bma.build_exact_grouped(fasta, mseq, e)
    M = cluster.toarray()[np.ix_(group, group)]
    np.testing.assert_allclose(M.sum(axis=1), 1.0)
    np.testing.assert_allclose(M[0], [(1 - free) * 3 / 8, (1 - free) * 3 / 8,
                                      free + (1 - free) / 4, 0, 0])
    np.testing.assert_allclose(M[2], [0, 0, 1, 0, 0])       # its own home
    np.testing.assert_allclose(M[4], [0, 0, 0, 0, 1])       # unmeasured: the self entry

    # tau 1: the homed row drops its distance-0 IUPAC partner (r3) and keeps its
    # distance-1 neighbour only where the home does not already cover it.
    _, cluster, group, _, strata = bma.build_kmer_grouped(fasta, 1, 4, 0.3, 1 << 30, 0.01,
                                                          mseq, e)
    M = cluster.toarray()[np.ix_(group, group)]
    np.testing.assert_allclose(M.sum(axis=1), 1.0)
    assert M[0, 3] == 0
    home_share = np.array([(1 - free) * 3 / 8, (1 - free) * 3 / 8, free + (1 - free) / 4])
    np.testing.assert_allclose(M[0, :3], home_share)
    assert strata.toarray()[group[0], group[2]] == 0         # re-decay leaves it alone
    base = bma.build_kmer_grouped(fasta, 1, 4, 0.3, 1 << 30, 0.01)[1].toarray()
    np.testing.assert_allclose(cluster.toarray()[group[4]], base[group[4]])
