#!/usr/bin/env python
"""Benchmark sparse mis-mapping matrix generation on a reference FASTA.

Each point uses a deterministic prefix of the supplied reference database, extracts
amplicons, builds the selected matrix backend, and writes the sparse matrix. Read
simulation, MAPseq, and minimap2 are not used.

The default sizes are 1,000 followed by ten doublings, then the complete supplied
database. A point that reaches ``--timeout-seconds`` is terminated and recorded as a
timeout; later, larger points are not attempted.
"""
from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
MINIMAP2_IMAGE = "quay.io/biocontainers/minimap2:2.28--he4a0461_3"
DEFAULT_SIZES = tuple(1_000 * 2**step for step in range(11))
sys.path.insert(0, str(REPOSITORY / "bin"))
import build_mismapping_align as bma  # noqa: E402
import sparse_matrix as sm  # noqa: E402


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield complete FASTA headers and sequences from a two-line or wrapped FASTA.

    Args:
        path: Input FASTA path.

    Yields:
        Header without ``>`` and its concatenated sequence.

    Raises:
        ValueError: If the FASTA structure is invalid.
    """
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if header is not None:
                    if not sequence_parts:
                        raise ValueError(f"{path}: record {header!r} has no sequence")
                    yield header, "".join(sequence_parts)
                header = text[1:]
                sequence_parts = []
            elif header is None:
                raise ValueError(f"{path}:{line_number}: sequence precedes the first header")
            else:
                sequence_parts.append(text)
    if header is None:
        raise ValueError(f"{path}: FASTA contains no records")
    if not sequence_parts:
        raise ValueError(f"{path}: record {header!r} has no sequence")
    yield header, "".join(sequence_parts)


def write_prefix(source: Path, count: int, destination: Path) -> None:
    """Write the first ``count`` FASTA records from source to destination.

    Args:
        source: Complete reference FASTA.
        count: Number of records to write.
        destination: FASTA output path.

    Raises:
        ValueError: If source has fewer than count records.
    """
    written = 0
    with destination.open("w") as handle:
        for header, sequence in iter_fasta(source):
            handle.write(f">{header}\n{sequence}\n")
            written += 1
            if written == count:
                return
    raise ValueError(f"{source} has {written} records; cannot select {count}")


def run_phase(
    command: Sequence[str],
    log_handle: object,
    timeout_seconds: float,
) -> float:
    """Run command, forwarding output to a log, and return elapsed seconds.

    Args:
        command: Command and arguments to run.
        log_handle: Open text file receiving command output.
        timeout_seconds: Maximum allowed wall time for this phase.

    Returns:
        Elapsed wall-clock seconds.

    Raises:
        subprocess.TimeoutExpired: If command exceeds timeout_seconds.
        subprocess.CalledProcessError: If command exits unsuccessfully.
    """
    start = time.perf_counter()
    subprocess.run(
        command,
        check=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
    )
    return time.perf_counter() - start


def run_streaming_alignment(
    command: Sequence[str],
    amplicons: Path,
    matrix: Path,
    log_handle: object,
    timeout_seconds: float,
) -> float:
    """Build a matrix from minimap2's PAF stream without writing a PAF file.

    Args:
        command: Minimap2 Docker command and arguments.
        amplicons: Reference amplicons FASTA.
        matrix: Destination for the labelled sparse matrix.
        log_handle: Open text file receiving minimap2 diagnostics.
        timeout_seconds: Maximum allowed wall time for the alignment.

    Returns:
        Elapsed wall-clock seconds.

    Raises:
        subprocess.TimeoutExpired: If minimap2 exceeds timeout_seconds.
        subprocess.CalledProcessError: If minimap2 exits unsuccessfully.
    """
    start = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log_handle)
    try:
        if process.stdout is None:
            raise RuntimeError("minimap2 stdout pipe was not created")
        paf_stream = io.TextIOWrapper(process.stdout)
        refseqs, mismapping, nearest = bma.build_sparse(
            amplicons, paf_stream, tau=0, ambiguity_weight=0.3
        )
        if time.perf_counter() - start > timeout_seconds:
            process.kill()
            raise subprocess.TimeoutExpired(command, timeout_seconds)
        return_code = process.wait(timeout=max(0, timeout_seconds - (time.perf_counter() - start)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    sm.write_matrix(matrix, mismapping, refseqs)
    bma.summarise_sparse(mismapping, nearest)
    return time.perf_counter() - start


def remaining_seconds(start: float, timeout_seconds: float) -> float:
    """Return remaining time for a benchmark point or raise TimeoutExpired."""
    remaining = timeout_seconds - (time.perf_counter() - start)
    if remaining <= 0:
        raise subprocess.TimeoutExpired("benchmark point", timeout_seconds)
    return remaining


def docker_command(case_directory: Path, *arguments: str) -> list[str]:
    """Return a minimap2 Docker command mounted at /data."""
    return [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--volume",
        f"{case_directory.resolve()}:/data",
        MINIMAP2_IMAGE,
        "minimap2",
        *arguments,
    ]


def write_results(rows: Sequence[dict[str, object]], destination: Path) -> None:
    """Write benchmark rows as a CSV file.

    Args:
        rows: Completed or timed-out benchmark rows.
        destination: Output CSV path.
    """
    fields = [
        "references",
        "status",
        "amplicon_references",
        "extract_seconds",
        "index_seconds",
        "align_seconds",
        "matrix_seconds",
        "total_seconds",
        "log",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path: Path) -> list[dict[str, object]]:
    """Return existing benchmark rows, or an empty list when no results exist."""
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_sizes(value: str, full_size: int, include_full: bool) -> tuple[int, ...]:
    """Parse comma-separated benchmark sizes and optionally append the full size.

    Args:
        value: Comma-separated positive integer sizes.
        full_size: Number of records in the complete database.
        include_full: Whether to include full_size even when not explicitly requested.

    Returns:
        Sorted, unique sizes no greater than full_size.
    """
    try:
        sizes = {int(part) for part in value.split(",")}
    except ValueError as error:
        raise ValueError("--sizes must be comma-separated integers") from error
    if not sizes or min(sizes) <= 0:
        raise ValueError("--sizes must contain positive integers")
    if max(sizes) > full_size:
        raise ValueError(f"--sizes includes a value above the {full_size} source records")
    if include_full:
        sizes.add(full_size)
    return tuple(sorted(sizes))


def count_records(path: Path) -> int:
    """Return the number of FASTA records in path."""
    return sum(1 for _ in iter_fasta(path))


def main() -> None:
    """Parse arguments and run the alignment-only scaling benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--sizes",
        default=",".join(str(size) for size in DEFAULT_SIZES),
        help="comma-separated source-reference counts; full size is always appended",
    )
    parser.add_argument(
        "--include-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="append the complete database size (default: true)",
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--backend", choices=("kmer", "exact-hash"), default="kmer")
    parser.add_argument("--tau", type=int, default=1)
    parser.add_argument("--max-ambiguous-bases", type=int, default=4)
    parser.add_argument("--max-postings", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=3_600)
    arguments = parser.parse_args()
    if arguments.threads <= 0:
        parser.error("--threads must be positive")
    if arguments.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if arguments.max_ambiguous_bases < 0 or arguments.max_postings < 1:
        parser.error("--max-ambiguous-bases must be non-negative and --max-postings >= 1")
    if arguments.backend == "kmer" and arguments.tau < 1:
        parser.error("--tau must be positive for the kmer backend")
    if arguments.backend == "exact-hash" and arguments.tau != 0:
        parser.error("--backend exact-hash requires --tau 0")
    if not arguments.references.is_file():
        parser.error(f"--references is not a file: {arguments.references}")

    full_size = count_records(arguments.references)
    try:
        sizes = parse_sizes(arguments.sizes, full_size, arguments.include_full)
    except ValueError as error:
        parser.error(str(error))
    arguments.outdir.mkdir(parents=True, exist_ok=True)
    logs_directory = arguments.outdir / "logs"
    logs_directory.mkdir(exist_ok=True)
    results_path = arguments.outdir / "scaling.csv"
    rows = read_results(results_path)
    completed_sizes = {int(row["references"]) for row in rows}

    for size in sizes:
        if size in completed_sizes:
            continue
        case_directory = arguments.outdir / f"case_{size}"
        case_directory.mkdir(exist_ok=True)
        subset = case_directory / "references.fasta"
        amplicon_directory = case_directory / "amplicons"
        amplicons = amplicon_directory / "amplicons.fasta"
        matrix = case_directory / "mismapping_matrix.npz"
        log_path = logs_directory / f"{size}.log"
        row: dict[str, object] = {
            "references": size,
            "status": "complete",
            "amplicon_references": "",
            "extract_seconds": "",
            "index_seconds": "",
            "align_seconds": "",
            "matrix_seconds": "",
            "total_seconds": "",
            "log": log_path.relative_to(arguments.outdir),
        }
        point_start = time.perf_counter()
        try:
            with log_path.open("w") as log_handle:
                log_handle.write(f"references={size}\n")
                write_prefix(arguments.references, size, subset)
                row["extract_seconds"] = run_phase(
                    [
                        sys.executable,
                        str(REPOSITORY / "bin" / "subspecies_infer.py"),
                        "amplicons",
                        "--db-fasta",
                        str(subset),
                        "--output-dir",
                        str(amplicon_directory),
                    ],
                    log_handle,
                    remaining_seconds(point_start, arguments.timeout_seconds),
                )
                row["amplicon_references"] = count_records(amplicons)
                row["index_seconds"] = 0.0
                row["align_seconds"] = run_phase(
                    [
                        sys.executable,
                        str(REPOSITORY / "bin" / "build_mismapping_align.py"),
                        "--backend", arguments.backend, "--amplicons", str(amplicons),
                        "--tau", str(arguments.tau),
                        "--max-ambiguous-bases", str(arguments.max_ambiguous_bases),
                        "--max-postings", str(arguments.max_postings),
                        "--ambiguity-weight", "0.3", "--output", str(matrix),
                    ],
                    log_handle,
                    remaining_seconds(point_start, arguments.timeout_seconds),
                )
                row["matrix_seconds"] = arguments.backend
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
        except subprocess.CalledProcessError as error:
            row["status"] = f"failed:{error.returncode}"
        finally:
            row["total_seconds"] = time.perf_counter() - point_start
            rows.append(row)
            write_results(rows, results_path)
            shutil.rmtree(case_directory, ignore_errors=True)
        print(
            f"n={size} status={row['status']} total={float(row['total_seconds']):.1f}s "
            f"amplicons={row['amplicon_references']}"
        )
        if row["status"] != "complete":
            break


if __name__ == "__main__":
    main()
