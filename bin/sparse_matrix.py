"""Read, write, and validate the labelled sparse panel kernel (sources x labels).

Optionally a kernel carries its **distance strata**: the edit distance ``d(s, l)`` behind
every nonzero, plus the ``--distance-decay`` ``c0`` it was built with. The alignment
kernel is ``K[s, l] proportional to u_l * c ** d(s, l)`` and the row normalisation kills
any per-row constant, so those two are enough to rebuild the kernel at *any* other decay
without the ambiguity weights or the sequences:

    K(c) = rownorm(K(c0) * (c / c0) ** d)

which is what lets ``infer_composition.py --infer-distance-decay`` fit ``c`` per sample
against a kernel built once (see ``subspecies_infer.DecayKernel``). Only ``tau >= 1``
builds store them — at ``tau = 0`` every distance is 0 and ``c`` cancels exactly.

Every kernel also carries ``kernel_version``; a file without the field reads as version 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

# Bump whenever the alignment kernel's definition changes; MATRIX_KEY hashes it too
# (modules/local/matrix_key/main.nf), so a stale kernel is never silently reused.
KERNEL_VERSION = 2


def _stored_version(archive) -> int:
    return int(archive["kernel_version"]) if "kernel_version" in archive else 1


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
    ``strata`` is an alignment kernel's ``(distances, decay)``.
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

    An all-zero row is allowed: a source no read can come from.
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
