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

    fasta = _fasta(tmp_path)
    _, built, group, _, distances = bma.build_kmer_grouped(fasta, 1, 4, 0.3, 1 << 30, 0.05)
    _, direct, _, _, _ = bma.build_kmer_grouped(fasta, 1, 4, 0.3, 1 << 30, 0.005)
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
