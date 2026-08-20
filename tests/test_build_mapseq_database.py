"""Integration tests for the MAPseq database builder."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "bin" / "build_mapseq_database.py"
MAPSEQ_IMAGE = "quay.io/biocontainers/mapseq:2.1.1b--hc47f52e_1"


def _docker_is_available() -> bool:
    """Return whether the Docker CLI can contact its daemon."""
    docker = shutil.which("docker")
    if docker is None:
        return False
    try:
        result = subprocess.run(
            [docker, "info"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_is_available(),
    reason="requires a running Docker daemon to invoke MAPseq",
)


def _read_fasta(path: Path) -> dict[str, str]:
    """Return FASTA records keyed by first header token."""
    records: dict[str, str] = {}
    header: str | None = None
    sequence_parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records[header] = "".join(sequence_parts)
            header = line[1:].split()[0]
            sequence_parts = []
        else:
            sequence_parts.append(line)
    if header is not None:
        records[header] = "".join(sequence_parts)
    return records


def _write_fasta(path: Path, records: dict[str, str]) -> None:
    """Write FASTA records in dictionary insertion order."""
    with path.open("w") as output_handle:
        for header, sequence in records.items():
            output_handle.write(f">{header}\n{sequence}\n")


def test_built_database_is_accepted_by_mapseq(tmp_path: Path) -> None:
    """Build a database and classify an identical query with MAPseq in its pinned image."""
    records = _read_fasta(ROOT / "tests" / "data" / "refs.fasta")
    genome_a = dict(list(records.items())[:2])
    genome_b = dict(list(records.items())[2:])
    _write_fasta(tmp_path / "genome_a.fasta", genome_a)
    _write_fasta(tmp_path / "genome_b.fasta", genome_b)
    (tmp_path / "genomes.yml").write_text(
        "genomes:\n"
        "  - id: genomeA\n"
        "    taxonomy: Bacteria;Example\n"
        "    fasta: genome_a.fasta\n"
        "  - id: genomeB\n"
        "    taxonomy: Bacteria;Example;Other\n"
        "    fasta: genome_b.fasta\n"
    )
    query_id, query_sequence = next(iter(genome_a.items()))
    _write_fasta(tmp_path / "query.fasta", {"query": query_sequence})

    subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--input",
            "genomes.yml",
            "--output-prefix",
            "references",
        ],
        check=True,
        cwd=tmp_path,
    )
    container = subprocess.run(
        [
            "docker",
            "create",
            "--platform",
            "linux/amd64",
            "--entrypoint",
            "/bin/sh",
            MAPSEQ_IMAGE,
            "-c",
            "sleep 600",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    ).stdout.strip()
    try:
        subprocess.run(["docker", "start", container], check=True, capture_output=True, text=True)
        subprocess.run(
            ["docker", "cp", f"{tmp_path}/.", f"{container}:/data"],
            check=True,
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "mapseq",
                "/data/query.fasta",
                "/data/references.fasta",
                "/data/references.tax",
                "-nthreads",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            check=False,
            capture_output=True,
            text=True,
        )

    hits = [line.split("\t") for line in result.stdout.splitlines() if line and not line.startswith("#")]
    assert hits
    assert hits[0][1] == f"genomeA|0|{query_id}"
