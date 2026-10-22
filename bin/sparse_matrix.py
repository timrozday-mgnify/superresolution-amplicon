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

import functools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

# Bump whenever the alignment kernel's definition changes; MATRIX_KEY hashes it too
# (modules/local/matrix_key/main.nf), so a stale kernel is never silently reused.
# 4: the label space is compacted to the columns the kernel gives a value to (_compact),
#    and the database's full accession list moved to a shared db_refs file.
KERNEL_VERSION = 4


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


def _compact(kernel: sparse.csr_array, label_ids: list[str], home: np.ndarray,
             ref_headers, label_of_ref, strata, neighbours):
    """Restrict the label space to the columns this kernel gives a value to.

    Those are its nonzeros, every source's home label and every neighbour label. A label
    with none of the three carries no mass and no distance, so inference merges it into a
    zero-count sink -- which ``infer_composition._subset_kernel`` already does, exactly,
    for every label a sample has no read on. Dropping them here is the same merge done
    once at build time instead of once per sample, and over a whole database it is almost
    everything: a one-genome part of SILVA-SSU keeps 278 of 463,499 columns.

    A database accession survives only where its label does. Whether a hit on one of the
    others is in this database at all -- an unexplained read rather than a foreign one --
    is what the shared ``db_refs`` file answers; see ``write_db_refs``.
    """
    n_lab = kernel.shape[1]
    home = np.asarray(home, dtype=np.int64)
    label_of_ref = np.asarray([] if label_of_ref is None else label_of_ref, dtype=np.int64)
    # read_kernel reports an index out of range; compacting one would raise IndexError here
    # instead, and a kernel this broken has no label space worth compacting.
    for idx in (home, label_of_ref):
        if len(idx) and (idx.min() < 0 or idx.max() >= n_lab):
            return kernel, label_ids, home, ref_headers, label_of_ref, strata, neighbours
    keep = np.zeros(n_lab, dtype=bool)
    keep[kernel.indices] = True
    keep[home] = True
    if neighbours is not None:
        keep[np.asarray(neighbours[1], dtype=np.int64)] = True
    kept = np.flatnonzero(keep)
    if len(kept) == n_lab:
        return kernel, label_ids, home, ref_headers, label_of_ref, strata, neighbours
    position = np.full(n_lab, -1, dtype=np.int64)
    position[kept] = np.arange(len(kept))
    shape = (kernel.shape[0], len(kept))
    # Renumbering the columns keeps the pattern, so the strata ride along on it unchanged.
    compacted = sparse.csr_array((kernel.data, position[kernel.indices], kernel.indptr),
                                 shape=shape)
    if strata is not None:
        distances = strata[0].tocsr()
        strata = (sparse.csr_array((distances.data, position[distances.indices],
                                    distances.indptr), shape=shape), strata[1])
    if len(label_of_ref):
        on_kept = keep[label_of_ref]
        ref_headers = [h for h, k in zip(ref_headers, on_kept) if k]
        label_of_ref = position[label_of_ref[on_kept]]
    if neighbours is not None:
        source, label, distance = (np.asarray(v, dtype=np.int64) for v in neighbours)
        neighbours = (source, position[label], distance)
    return (compacted, [label_ids[i] for i in kept], position[home], ref_headers,
            label_of_ref, strata, neighbours)


DB_REFS_PREFIX = "db_refs_"


def db_refs_name(db_amplicons_sha256: str) -> str:
    """The file name holding the accessions of the database with this digest."""
    return f"{DB_REFS_PREFIX}{db_amplicons_sha256[:16]}.npz"


def write_db_refs(directory: Path, db_amplicons_sha256: str, ref_headers) -> Path:
    """Write the database's amplifiable accessions, shared by every kernel built on it.

    A compacted kernel (``_compact``) keeps only the accessions whose label it has a value
    for, so it can no longer tell a hit on another entry of its own database -- an
    unexplained read, which is background mass -- from a hit on some other database, which
    is a wrong pairing and must not be counted at all. This file is what answers that, and
    it is a property of the extraction (the database and the primers) alone: one copy
    serves every kernel built from the same amplicons, which is what its digest names.
    """
    path = Path(directory) / db_refs_name(db_amplicons_sha256)
    np.savez_compressed(
        path,
        format=np.asarray("db_refs"),
        db_amplicons_sha256=np.asarray(db_amplicons_sha256),
        ref_headers=np.frombuffer("\n".join(ref_headers).encode(), dtype=np.uint8),
    )
    return path


def read_db_refs(path: Path, db_amplicons_sha256: str = "") -> frozenset[str]:
    """Load the accession set written by ``write_db_refs``, checking whose database it is.

    Each file is read once per process and the set shared: it holds millions of
    accessions, and ``combine`` reads one per part, which made 24 parts peak at 4.8 GB.
    """
    try:
        stat = Path(path).stat()
        stored, refs = _load_db_refs(str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"invalid database accession file {path}") from exc
    if db_amplicons_sha256 and stored != db_amplicons_sha256:
        raise ValueError(f"{path} holds database {stored[:12]}, not {db_amplicons_sha256[:12]}")
    return refs


@functools.lru_cache(maxsize=4)
def _load_db_refs(path: str, _mtime_ns: int, _size: int) -> tuple[str, frozenset[str]]:
    """``(digest, accessions)`` of a db_refs file; keyed by its stat, so a rewrite is reread."""
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["format"]) != "db_refs":
            raise ValueError("not a db_refs file")
        stored = str(archive["db_amplicons_sha256"])
        headers = archive["ref_headers"].tobytes().decode()
    return stored, frozenset(headers.split("\n")) if headers else frozenset()


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
                 neighbours: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                 kernel_version: int = KERNEL_VERSION) -> None:
    """Write a sources x labels kernel with each source's home label.

    ``n_simulated``/``n_unmapped`` are per source and optional; ``provenance`` is any
    JSON-serialisable record of the build (error model, rates, reads per source, seed).
    ``strata`` is an alignment kernel's ``(distances, decay)``. ``neighbours`` is
    ``(source, label, distance)``: every database label a few edits from each source,
    which the presence evidence needs to expect sequencing-error spill-over.
    """
    kernel = kernel.tocsr()
    (kernel, label_ids, home, ref_headers, label_of_ref, strata,
     neighbours) = _compact(kernel, list(label_ids), home, list(ref_headers), label_of_ref,
                            strata, neighbours)
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
        **({} if neighbours is None else {
            f"neighbour_{name}": np.asarray(v, dtype=np.int64)
            for name, v in zip(("source", "label", "distance"), neighbours)}),
    )


def read_kernel(path: Path, db_refs: Path | None = None) -> SimpleNamespace:
    """Load and validate a rectangular kernel written by ``write_kernel``.

    An all-zero row is allowed: a source no read can come from.

    ``k.db_refs`` is the database's full accession set when one is to hand: ``db_refs``
    names it, else ``db_refs_<digest>.npz`` beside the kernel, else it is ``None`` and a
    hit the kernel does not label cannot be told from a foreign one (``write_db_refs``).
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
                neighbours=(tuple(archive[f"neighbour_{n}"].astype(np.int64)
                                  for n in ("source", "label", "distance"))
                            if "neighbour_source" in archive else None),
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
    # Beside the kernel is a combined bundle's own copy; one level up is the published
    # mismapping/ directory, where the extraction puts the one copy its parts share.
    name = db_refs_name(k.db_amplicons_sha256)
    found = next((p for p in (Path(path).parent / name, Path(path).parent.parent / name)
                  if p.exists()), None)
    k.db_refs = (read_db_refs(db_refs, k.db_amplicons_sha256) if db_refs
                 else read_db_refs(found, k.db_amplicons_sha256) if found else None)
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
