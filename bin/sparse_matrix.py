"""Read, write, and validate labelled sparse mis-mapping matrices.

Optionally a matrix carries its **distance strata**: the edit distance ``d(a, j)`` behind
every nonzero, plus the ``--distance-decay`` ``c0`` it was built with. The tie-cluster
kernel is ``M[a, j] proportional to u_j * c ** d(a, j)`` and the row normalisation kills
any per-row constant, so those two are enough to rebuild the matrix at *any* other decay
without the ambiguity weights, the group sizes or the sequences:

    M(c) = rownorm(M(c0) * (c / c0) ** d)

which is what lets ``infer_composition.py --infer-distance-decay`` fit ``c`` per sample
against a matrix built once (see ``subspecies_infer.DecayKernel``). Only ``tau >= 1``
builds store them — at ``tau = 0`` every distance is 0 and ``c`` cancels exactly.

Every matrix also carries ``kernel_version``. Version 2 weights a neighbouring *distinct*
amplicon once and splits its share over its duplicates; version 1 weighted it by its
duplicate count, which over-predicts leakage onto a sequence many references share by
up to ~20x at GTDB scale (docs/gtdb_inference_recovery_plan.md, Evidence §3). A file
without the field is version 1: it still loads, with a warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

log = logging.getLogger("sparse_matrix")

# Bump whenever the alignment kernel's definition changes; MATRIX_KEY hashes it too
# (modules/local/matrix_key/main.nf), so a stale matrix is never silently reused.
KERNEL_VERSION = 2


def _stored_version(archive) -> int:
    return int(archive["kernel_version"]) if "kernel_version" in archive else 1


def _warn_if_stale(path: Path, version: int) -> None:
    if version < KERNEL_VERSION:
        log.warning("mis-mapping matrix %s has kernel_version %d (current %d): a version-1 "
                    "alignment kernel weights neighbours by duplicate count; rebuild it",
                    path, version, KERNEL_VERSION)


def stored_kernel_version(path: Path) -> int:
    """Return the stored kernel version of a matrix archive (1 when absent)."""
    with np.load(path, allow_pickle=False) as archive:
        return _stored_version(archive)


def _strata_arrays(matrix: sparse.csr_array, strata) -> dict:
    """Validate ``(distances, decay)`` against ``matrix`` and return the arrays to store."""
    if strata is None:
        return {}
    distances, decay = strata
    distances = distances.tocsr()
    if distances.shape != matrix.shape or distances.nnz != matrix.nnz or not np.array_equal(
            distances.indices, matrix.indices) or not np.array_equal(
            distances.indptr, matrix.indptr):
        raise ValueError("distance strata must share the matrix's sparsity pattern")
    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("stored distance decay must be in (0, 1]")
    return {"distances": distances.data.astype(np.int16),
            "decay": np.float64(decay)}


def _read_strata(archive, pattern: sparse.csr_array):
    """Rebuild the stored distance strata as a matrix of ``d + 1``, or ``None`` if absent.

    ``d + 1`` rather than ``d`` so no value is ever a structural zero: the strata then ride
    on the matrix's own sparsity pattern and reorder (and subset) through the same scipy
    permutation, instead of needing index arithmetic of their own.
    """
    if "distances" not in archive:
        return None
    return sparse.csr_array(
        (archive["distances"].astype(np.float64) + 1.0, pattern.indices, pattern.indptr),
        shape=pattern.shape,
    ), float(archive["decay"])


def _reorder_strata(strata, order: np.ndarray):
    """Permute strata alongside their matrix, and drop the ``+ 1`` offset."""
    if strata is None:
        return None
    distances, decay = strata
    distances = distances[order][:, order].tocsr()
    distances.data -= 1.0
    return distances, decay


def write_matrix(path: Path, matrix: sparse.csr_array, refseqs: list[str],
                 strata=None, kernel_version: int = KERNEL_VERSION) -> None:
    """Write a row-stochastic CSR matrix and its ordered reference identifiers.

    ``strata`` is an optional ``(distances, decay)`` pair — see the module docstring.
    """
    matrix = matrix.tocsr()
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
        refseqs=np.asarray(refseqs, dtype=np.str_),
        kernel_version=np.int64(kernel_version),
        **_strata_arrays(matrix, strata),
    )


def read_matrix(path: Path, refseqs: list[str], strata: bool = False):
    """Load and reorder a labelled CSR matrix, checking its stochastic invariants.

    With ``strata`` the return is ``(matrix, (distances, decay))``, the second element
    ``None`` when the matrix was built without them.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            stored_refs = archive["refseqs"].astype(str).tolist()
            matrix = sparse.csr_array(
                (archive["data"], archive["indices"], archive["indptr"]),
                shape=tuple(archive["shape"]),
            )
            stored_strata = _read_strata(archive, matrix)
            version = _stored_version(archive)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid sparse mis-mapping matrix {path}") from exc
    _warn_if_stale(path, version)
    if len(set(stored_refs)) != len(stored_refs):
        raise ValueError(f"mis-mapping matrix {path} has duplicate reference IDs")
    if set(stored_refs) != set(refseqs) or matrix.shape != (len(refseqs), len(refseqs)):
        raise ValueError(f"mis-mapping matrix {path} does not match this reference set")
    position = {reference: index for index, reference in enumerate(stored_refs)}
    order = np.asarray([position[reference] for reference in refseqs])
    stored_strata = _reorder_strata(stored_strata, order)
    matrix = matrix[order][:, order].tocsr()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f"mis-mapping matrix {path} must contain finite, non-negative values")
    if not np.allclose(np.asarray(matrix.sum(axis=1)).ravel(), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"mis-mapping matrix {path} must be row-stochastic")
    return (matrix, stored_strata) if strata else matrix


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
                  refseqs: list[str], strata=None,
                  kernel_version: int = KERNEL_VERSION) -> None:
    """Write ``M`` as a unique-amplicon kernel plus each reference's group id.

    ``strata`` is an optional ``(distances, decay)`` pair over the *kernel*, i.e. over the
    same distinct amplicons as ``cluster`` — see the module docstring.
    """
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
        kernel_version=np.int64(kernel_version),
        **_strata_arrays(cluster, strata),
    )


def read_grouped(path: Path, refseqs: list[str], strata: bool = False):
    """Load a grouped matrix and return ``(S, group)`` in the caller's reference order.

    With ``strata`` the return is ``(S, group, (distances, decay))``, the third element
    ``None`` when the matrix was built without them. ``S`` and the distances are over the
    distinct amplicons, whose order is the stored one either way — only ``group`` is
    permuted into the caller's reference order.
    """
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
            stored_strata = _read_strata(archive, cluster)
            version = _stored_version(archive)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid grouped mis-mapping matrix {path}") from exc
    _warn_if_stale(path, version)
    position = {reference: index for index, reference in enumerate(stored_refs)}
    if len(position) != len(stored_refs):
        raise ValueError(f"mis-mapping matrix {path} has duplicate reference IDs")
    if position.keys() != set(refseqs) or len(group) != len(refseqs):
        raise ValueError(f"mis-mapping matrix {path} does not match this reference set")
    group = group[np.fromiter((position[r] for r in refseqs), dtype=np.int64, count=len(refseqs))]
    if not np.isfinite(cluster.data).all() or (cluster.data < 0).any():
        raise ValueError(f"mis-mapping matrix {path} must contain finite, non-negative values")
    sizes = np.bincount(group, minlength=cluster.shape[0]).astype(np.float64)
    # An all-zero row is allowed and means "no read can come from this group": a matrix
    # measured on one batch's active groups has a row only where reads were simulated,
    # and inference prunes the rest. Every other row must still be row-stochastic.
    rows = cluster @ sizes
    if not np.allclose(rows[rows > 0], 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"mis-mapping matrix {path} must be row-stochastic")
    if not strata:
        return cluster, group
    # The kernel keeps its stored order (only ``group`` is remapped), so the strata need
    # no permutation of their own — just the ``+ 1`` offset removed.
    if stored_strata is not None:
        stored_strata = (stored_strata[0].copy(), stored_strata[1])
        stored_strata[0].data -= 1.0
    return cluster, group, stored_strata


def is_grouped(path: Path) -> bool:
    """Return whether path holds the grouped form rather than a reference-square CSR."""
    with np.load(path, allow_pickle=False) as archive:
        return "format" in archive and str(archive["format"]) == "grouped"


def is_rectangular(path: Path) -> bool:
    """Return whether path holds a sources x labels kernel from ``write_kernel``."""
    with np.load(path, allow_pickle=False) as archive:
        return "format" in archive and str(archive["format"]) == "rectangular"


def grouped_diagonal(cluster: sparse.csr_array, group: np.ndarray) -> np.ndarray:
    """Return ``diag(M)`` — each reference's own share of its cluster."""
    return cluster.diagonal()[group]


# ── Rectangular form ──────────────────────────────────────────────────────────
#
# Sources and labels are different sets: rows are the distinct amplicons reads were
# simulated from (a genome panel), columns the database exact-sequence groups MAPseq
# labels them with. ``K[s, l]`` is the fraction of source ``s``'s classified reads labelled
# ``l``, and ``home[s]``, the label MAPseq gives the error-free source sequence, takes the
# diagonal's part in ``rownorm((1-s) e_home + s K)``. ``label_of_ref`` maps every database
# header to its label, so inference needs the kernel and not the database FASTA.


def write_kernel(path: Path, kernel: sparse.csr_array, source_ids: list[str],
                 label_ids: list[str], home: np.ndarray, *, ref_headers: list[str] = (),
                 label_of_ref: np.ndarray | None = None, db_amplicons_sha256: str = "",
                 n_simulated: np.ndarray | None = None, n_unmapped: np.ndarray | None = None,
                 provenance: dict | None = None, strata=None,
                 kernel_version: int = KERNEL_VERSION) -> None:
    """Write a sources x labels kernel with each source's home label.

    ``n_simulated``/``n_unmapped`` are per source and optional; ``provenance`` is any
    JSON-serialisable record of the build (error model, rates, reads per source, seed).
    ``strata`` is an alignment kernel's ``(distances, decay)``, as for the square forms.
    """
    kernel = kernel.tocsr()
    per_source = {"n_simulated": n_simulated, "n_unmapped": n_unmapped}
    np.savez_compressed(
        path,
        format=np.asarray("rectangular"),
        data=kernel.data,
        indices=kernel.indices,
        indptr=kernel.indptr,
        shape=np.asarray(kernel.shape, dtype=np.int64),
        source_ids=np.asarray(source_ids, dtype=np.str_),
        label_ids=np.asarray(label_ids, dtype=np.str_),
        home=np.asarray(home, dtype=np.int64),
        ref_headers=np.frombuffer("\n".join(ref_headers).encode(), dtype=np.uint8),
        label_of_ref=np.asarray([] if label_of_ref is None else label_of_ref, dtype=np.int32),
        db_amplicons_sha256=np.asarray(db_amplicons_sha256),
        provenance=np.asarray(json.dumps(provenance or {})),
        kernel_version=np.int64(kernel_version),
        **_strata_arrays(kernel, strata),
        **{k: np.asarray(v, dtype=np.int64) for k, v in per_source.items() if v is not None},
    )


def read_kernel(path: Path) -> SimpleNamespace:
    """Load and validate a rectangular kernel written by ``write_kernel``.

    An all-zero row is allowed, as in the grouped form: a source no read can come from.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "format" not in archive or str(archive["format"]) != "rectangular":
                raise ValueError(f"{path} is not a rectangular mis-mapping kernel")
            headers = archive["ref_headers"].tobytes().decode()
            kernel = sparse.csr_array((archive["data"], archive["indices"], archive["indptr"]),
                                      shape=tuple(archive["shape"]))
            k = SimpleNamespace(
                kernel=kernel, strata=_read_strata(archive, kernel),
                source_ids=archive["source_ids"].astype(str).tolist(),
                label_ids=archive["label_ids"].astype(str).tolist(),
                home=archive["home"].astype(np.int64),
                ref_headers=headers.split("\n") if headers else [],
                label_of_ref=archive["label_of_ref"].astype(np.int64),
                db_amplicons_sha256=str(archive["db_amplicons_sha256"]),
                provenance=json.loads(str(archive["provenance"])),
                kernel_version=_stored_version(archive),
                **{f: archive[f] if f in archive else None
                   for f in ("n_simulated", "n_unmapped")},
            )
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid rectangular mis-mapping kernel {path}") from exc
    if k.strata is not None:
        k.strata[0].data -= 1.0
    n_src, n_lab = k.kernel.shape
    if len(k.source_ids) != n_src or len(k.label_ids) != n_lab or len(k.home) != n_src:
        raise ValueError(f"mis-mapping kernel {path}: ids or home do not match its shape")
    if len(set(k.source_ids)) != n_src or len(set(k.label_ids)) != n_lab:
        raise ValueError(f"mis-mapping kernel {path} has duplicate source or label IDs")
    if (len(k.label_of_ref) != len(k.ref_headers)
            or ((k.home < 0) | (k.home >= n_lab)).any()
            or ((k.label_of_ref < 0) | (k.label_of_ref >= n_lab)).any()):
        raise ValueError(f"mis-mapping kernel {path} has a label index out of range")
    if not np.isfinite(k.kernel.data).all() or (k.kernel.data < 0).any():
        raise ValueError(f"mis-mapping kernel {path} must contain finite, non-negative values")
    rows = np.asarray(k.kernel.sum(axis=1)).ravel()
    if not np.allclose(rows[rows > 0], 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"mis-mapping kernel {path} must be row-stochastic")
    return k


def home_entries(kernel: sparse.csr_array, home: np.ndarray) -> np.ndarray:
    """Return ``K[s, home[s]]`` per source: the rectangular kernel's diagonal."""
    coo = kernel.tocoo()
    at_home = coo.col == np.asarray(home)[coo.row]
    entries = np.zeros(kernel.shape[0])
    entries[coo.row[at_home]] = coo.data[at_home]
    return entries
