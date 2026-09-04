"""Read, write, and validate labelled sparse mis-mapping matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse


def write_matrix(path: Path, matrix: sparse.csr_array, refseqs: list[str]) -> None:
    """Write a row-stochastic CSR matrix and its ordered reference identifiers."""
    matrix = matrix.tocsr()
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
        refseqs=np.asarray(refseqs, dtype=np.str_),
    )


def read_matrix(path: Path, refseqs: list[str]) -> sparse.csr_array:
    """Load and reorder a labelled CSR matrix, checking its stochastic invariants."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            stored_refs = archive["refseqs"].astype(str).tolist()
            matrix = sparse.csr_array(
                (archive["data"], archive["indices"], archive["indptr"]),
                shape=tuple(archive["shape"]),
            )
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid sparse mis-mapping matrix {path}") from exc
    if len(set(stored_refs)) != len(stored_refs):
        raise ValueError(f"mis-mapping matrix {path} has duplicate reference IDs")
    if set(stored_refs) != set(refseqs) or matrix.shape != (len(refseqs), len(refseqs)):
        raise ValueError(f"mis-mapping matrix {path} does not match this reference set")
    position = {reference: index for index, reference in enumerate(stored_refs)}
    order = np.asarray([position[reference] for reference in refseqs])
    matrix = matrix[order][:, order].tocsr()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f"mis-mapping matrix {path} must contain finite, non-negative values")
    if not np.allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"mis-mapping matrix {path} must be row-stochastic")
    return matrix


def matrix_diagonal_mean(matrix: sparse.csr_array) -> float:
    """Return the mean diagonal probability without densifying the matrix."""
    return float(matrix.diagonal().mean())


# ── Grouped form ──────────────────────────────────────────────────────────────
#
# A tie-cluster ``M`` is constant on exact-duplicate groups: two references with the same
# amplicon are interchangeable, so every row and every column of ``M`` that belongs to the
# same duplicate group is identical. That makes ``M`` fully described by
#
#     M[a, j] = S[group[a], group[j]]
#
# with ``S`` over the *unique* amplicons. The dense-CSR form stores one entry per
# reference pair inside a cluster, i.e. sum-of-squares of the group sizes — 1.1M GTDB SSU
# references collapse to ~10^9 nonzeros there and ~10^5 here. Row-stochasticity becomes
# ``sum_b S[a, b] * size[b] == 1``.


def write_grouped(path: Path, cluster: sparse.csr_array, group: np.ndarray,
                  refseqs: list[str]) -> None:
    """Write ``M`` as a unique-amplicon kernel plus each reference's group id."""
    cluster = cluster.tocsr()
    np.savez_compressed(
        path,
        format=np.asarray("grouped"),
        data=cluster.data,
        indices=cluster.indices,
        indptr=cluster.indptr,
        shape=np.asarray(cluster.shape, dtype=np.int64),
        group=np.asarray(group, dtype=np.int32),
        # Headers as one blob: a fixed-width numpy unicode array pads every id to the
        # longest one, which is ~100x larger for a million references.
        refseqs=np.frombuffer("\n".join(refseqs).encode(), dtype=np.uint8),
    )


def read_grouped(path: Path, refseqs: list[str]) -> tuple[sparse.csr_array, np.ndarray]:
    """Load a grouped matrix and return ``(S, group)`` in the caller's reference order."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["format"]) != "grouped":
                raise ValueError(f"{path} is not a grouped mis-mapping matrix")
            stored_refs = archive["refseqs"].tobytes().decode().split("\n")
            group = archive["group"]
            cluster = sparse.csr_array(
                (archive["data"], archive["indices"], archive["indptr"]),
                shape=tuple(archive["shape"]),
            )
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid grouped mis-mapping matrix {path}") from exc
    position = {reference: index for index, reference in enumerate(stored_refs)}
    if len(position) != len(stored_refs):
        raise ValueError(f"mis-mapping matrix {path} has duplicate reference IDs")
    if position.keys() != set(refseqs) or len(group) != len(refseqs):
        raise ValueError(f"mis-mapping matrix {path} does not match this reference set")
    group = group[np.fromiter((position[r] for r in refseqs), dtype=np.int64, count=len(refseqs))]
    if not np.isfinite(cluster.data).all() or (cluster.data < 0).any():
        raise ValueError(f"mis-mapping matrix {path} must contain finite, non-negative values")
    sizes = np.bincount(group, minlength=cluster.shape[0]).astype(np.float64)
    if not np.allclose(cluster @ sizes, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"mis-mapping matrix {path} must be row-stochastic")
    return cluster, group


def is_grouped(path: Path) -> bool:
    """Return whether path holds the grouped form rather than a reference-square CSR."""
    with np.load(path, allow_pickle=False) as archive:
        return "format" in archive and str(archive["format"]) == "grouped"


def grouped_diagonal(cluster: sparse.csr_array, group: np.ndarray) -> np.ndarray:
    """Return ``diag(M)`` — each reference's own share of its cluster."""
    return cluster.diagonal()[group]
