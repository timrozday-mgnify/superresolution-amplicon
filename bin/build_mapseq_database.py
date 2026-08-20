#!/usr/bin/env python
"""Build a MAPseq reference FASTA and taxonomy sidecar from a genome YAML file.

The input YAML must contain a ``genomes`` list. Each entry supplies a genome ``id``,
semicolon-delimited ``taxonomy``, and source ``fasta`` path. Each source FASTA record is
written to ``<output-prefix>.fasta`` as ``genome|copy_index|original_id`` and receives a
matching record in ``<output-prefix>.tax``.

Example:
    build_mapseq_database.py --input genomes.yml --output-prefix db/references
"""
from __future__ import annotations

import argparse
import gzip
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

_TAX_NAME = "#name: refdb\n"
_UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Genome:
    """Describe one genome's source FASTA and taxonomy."""

    identifier: str
    taxonomy: tuple[str, ...]
    fasta: Path


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield source FASTA records from a plain-text or gzip-compressed file.

    Args:
        path: FASTA file to parse.

    Yields:
        Tuples of the first header token and uppercased sequence.

    Raises:
        ValueError: If the FASTA structure, header, or sequence is invalid.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    header: str | None = None
    sequence_parts: list[str] = []
    record_count = 0

    with opener(path, "rt") as fasta_handle:
        for line_number, line in enumerate(fasta_handle, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if header is not None:
                    if not sequence_parts:
                        raise ValueError(f"{path}: record {header!r} has no sequence")
                    record_count += 1
                    yield header, "".join(sequence_parts).upper()
                header_text = text[1:].strip()
                if not header_text:
                    raise ValueError(f"{path}:{line_number}: FASTA header is empty")
                header = header_text.split()[0]
                sequence_parts = []
            elif header is None:
                raise ValueError(
                    f"{path}:{line_number}: sequence appears before the first FASTA header"
                )
            else:
                sequence_parts.append(text)

    if header is not None:
        if not sequence_parts:
            raise ValueError(f"{path}: record {header!r} has no sequence")
        record_count += 1
        yield header, "".join(sequence_parts).upper()
    if not record_count:
        raise ValueError(f"{path}: FASTA contains no records")


def parse_taxonomy(value: object, entry_number: int) -> tuple[str, ...]:
    """Validate and split a semicolon-delimited taxonomy lineage.

    Args:
        value: YAML taxonomy value.
        entry_number: One-based YAML entry number for error messages.

    Returns:
        Lineage fields with surrounding whitespace removed.

    Raises:
        ValueError: If the taxonomy is not a non-empty, well-formed string.
    """
    if not isinstance(value, str):
        raise ValueError(f"genomes[{entry_number}].taxonomy must be a string")
    if "\t" in value or "\n" in value or "\r" in value:
        raise ValueError(f"genomes[{entry_number}].taxonomy cannot contain tabs or newlines")
    taxonomy = tuple(part.strip() for part in value.split(";"))
    if not taxonomy or any(not part for part in taxonomy):
        raise ValueError(
            f"genomes[{entry_number}].taxonomy must be a non-empty semicolon-delimited lineage"
        )
    return taxonomy


def parse_genomes(input_path: Path) -> list[Genome]:
    """Load and validate genome entries from an input YAML file.

    Args:
        input_path: YAML configuration path.

    Returns:
        Validated genome entries, in YAML order.

    Raises:
        ValueError: If the YAML structure or any genome entry is invalid.
    """
    try:
        with input_path.open() as input_handle:
            document = yaml.safe_load(input_handle)
    except OSError as exc:
        raise ValueError(f"cannot read YAML input {input_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"cannot parse YAML input {input_path}: {exc}") from exc

    if not isinstance(document, Mapping):
        raise ValueError("input YAML must be a mapping with a 'genomes' key")
    entries = document.get("genomes")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ValueError("input YAML 'genomes' must be a non-empty list")

    genomes: list[Genome] = []
    seen_identifiers: set[str] = set()
    for entry_number, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise ValueError(f"genomes[{entry_number}] must be a mapping")
        missing = {"id", "taxonomy", "fasta"}.difference(entry)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"genomes[{entry_number}] is missing required key(s): {missing_names}")
        identifier = entry["id"]
        fasta_value = entry["fasta"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"genomes[{entry_number}].id must be a non-empty string")
        if "|" in identifier or any(character.isspace() for character in identifier):
            raise ValueError(f"genomes[{entry_number}].id cannot contain '|' or whitespace")
        if identifier in seen_identifiers:
            raise ValueError(f"duplicate genome id: {identifier!r}")
        if not isinstance(fasta_value, str) or not fasta_value.strip():
            raise ValueError(f"genomes[{entry_number}].fasta must be a non-empty path string")
        fasta = Path(fasta_value)
        if not fasta.is_absolute():
            fasta = input_path.parent / fasta
        if not fasta.is_file():
            raise ValueError(
                f"genomes[{entry_number}].fasta does not exist or is not a file: {fasta}"
            )

        genomes.append(Genome(identifier, parse_taxonomy(entry["taxonomy"], entry_number), fasta))
        seen_identifiers.add(identifier)
    return genomes


def output_paths(prefix: Path) -> tuple[Path, Path]:
    """Return output FASTA and taxonomy paths for a prefix."""
    return Path(f"{prefix}.fasta"), Path(f"{prefix}.tax")


def taxonomy_cutoffs(level_count: int) -> str:
    """Return a MAPseq cutoff header with one neutral cutoff per taxonomy level."""
    return "#cutoff: " + " ".join("0.00:0.00" for _ in range(level_count)) + "\n"


def _temporary_path(directory: Path, suffix: str) -> Path:
    """Create an empty temporary path in a destination directory."""
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=".build_mapseq_database_",
        suffix=suffix,
    )
    os.close(descriptor)
    Path(name).unlink()
    return Path(name)


def build_database(genomes: Sequence[Genome], output_prefix: Path) -> tuple[Path, Path, int]:
    """Build MAPseq FASTA and taxonomy files from validated genome entries.

    Args:
        genomes: Validated genomes in output order.
        output_prefix: Prefix for the generated files.

    Returns:
        Output FASTA path, taxonomy path, and number of reference records written.

    Raises:
        ValueError: If a source FASTA emits a duplicate generated reference ID.
    """
    fasta_path, tax_path = output_paths(output_prefix)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    max_taxonomy_depth = max(len(genome.taxonomy) for genome in genomes)
    levels = [f"Taxonomy_{index}" for index in range(1, max_taxonomy_depth + 1)] + ["Copy"]
    fasta_temp = _temporary_path(fasta_path.parent, ".fasta")
    tax_temp = _temporary_path(tax_path.parent, ".tax")
    seen_references: set[str] = set()
    record_count = 0

    complete = False
    try:
        with fasta_temp.open("w") as fasta_handle, tax_temp.open("w") as tax_handle:
            tax_handle.write(taxonomy_cutoffs(len(levels)))
            tax_handle.write(_TAX_NAME)
            tax_handle.write(f"#levels: {' '.join(levels)}\n")
            for genome in genomes:
                padded_taxonomy = genome.taxonomy + (_UNCLASSIFIED,) * (
                    max_taxonomy_depth - len(genome.taxonomy)
                )
                for copy_index, (source_header, sequence) in enumerate(iter_fasta(genome.fasta)):
                    reference_id = f"{genome.identifier}|{copy_index}|{source_header}"
                    if reference_id in seen_references:
                        raise ValueError(f"duplicate generated reference ID: {reference_id!r}")
                    seen_references.add(reference_id)
                    fasta_handle.write(f">{reference_id}\n{sequence}\n")
                    taxonomy = ";".join((*padded_taxonomy, reference_id))
                    tax_handle.write(f"{reference_id}\t{taxonomy}\n")
                    record_count += 1
        fasta_temp.replace(fasta_path)
        tax_temp.replace(tax_path)
        complete = True
    finally:
        if not complete:
            fasta_temp.unlink(missing_ok=True)
            tax_temp.unlink(missing_ok=True)
    return fasta_path, tax_path, record_count


def _require(condition: bool, message: str) -> None:
    """Raise RuntimeError unless a demo condition is true."""
    if not condition:
        raise RuntimeError(message)


def _expect_value_error(callback: object) -> None:
    """Run a no-argument callback and require it to raise ValueError."""
    if not callable(callback):
        raise TypeError("demo callback must be callable")
    try:
        callback()
    except ValueError:
        return
    raise RuntimeError("expected ValueError")


def demo() -> None:
    """Run self-contained successful and invalid-input checks."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        (root / "alpha.fasta").write_text(">alpha one\nacgtn\n>alpha_two description\nrysw\n")
        with gzip.open(root / "beta.fasta.gz", "wt") as output_handle:
            output_handle.write(">beta\nNNac\n")
        (root / "genomes.yml").write_text(
            "genomes:\n"
            "  - id: alpha\n"
            "    taxonomy: Bacteria;Alpha\n"
            "    fasta: alpha.fasta\n"
            "  - id: beta\n"
            "    taxonomy: Bacteria;Beta;Species\n"
            "    fasta: beta.fasta.gz\n"
        )
        genomes = parse_genomes(root / "genomes.yml")
        fasta_path, tax_path, record_count = build_database(genomes, root / "db" / "references")
        _require(record_count == 3, "demo wrote the wrong record count")
        _require(
            fasta_path.read_text().splitlines()
            == [
                ">alpha|0|alpha",
                "ACGTN",
                ">alpha|1|alpha_two",
                "RYSW",
                ">beta|0|beta",
                "NNAC",
            ],
            "demo FASTA headers or sequences are wrong",
        )
        tax_lines = tax_path.read_text().splitlines()
        _require(tax_lines[2] == "#levels: Taxonomy_1 Taxonomy_2 Taxonomy_3 Copy", "bad levels")
        _require(tax_lines[3].endswith("Bacteria;Alpha;unclassified;alpha|0|alpha"), "bad padding")
        _require(tax_lines[5].endswith("Bacteria;Beta;Species;beta|0|beta"), "bad taxonomy")

        (root / "duplicate.yml").write_text(
            "genomes:\n- id: alpha\n  taxonomy: Bacteria\n  fasta: alpha.fasta\n"
            "- id: alpha\n  taxonomy: Bacteria\n  fasta: alpha.fasta\n"
        )
        (root / "empty.fasta").write_text("")
        (root / "empty.yml").write_text(
            "genomes:\n- id: empty\n  taxonomy: Bacteria\n  fasta: empty.fasta\n"
        )
        (root / "missing_fasta.yml").write_text(
            "genomes:\n- id: missing\n  taxonomy: Bacteria\n  fasta: absent.fasta\n"
        )
        _expect_value_error(lambda: parse_genomes(root / "duplicate.yml"))
        _expect_value_error(lambda: parse_taxonomy("Bacteria;;Species", 1))
        _expect_value_error(lambda: parse_genomes(root / "missing_fasta.yml"))
        empty_output = root / "empty_database"
        _expect_value_error(
            lambda: build_database(parse_genomes(root / "empty.yml"), empty_output)
        )
        empty_fasta, empty_tax = output_paths(empty_output)
        _require(not empty_fasta.exists() and not empty_tax.exists(), "partial empty-FASTA output")
    print("demo OK: built MAPseq FASTA/tax and validated invalid inputs")


def main() -> None:
    """Parse CLI arguments and build a MAPseq database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="YAML file containing a genomes list")
    parser.add_argument(
        "-o",
        "--output-prefix",
        type=Path,
        help="output prefix for .fasta and .tax",
    )
    parser.add_argument("--demo", action="store_true", help="run self-contained validation checks")
    arguments = parser.parse_args()
    if arguments.demo:
        demo()
        return
    if arguments.input is None or arguments.output_prefix is None:
        parser.error("--input and --output-prefix are required unless --demo is used")
    try:
        genomes = parse_genomes(arguments.input)
        fasta_path, tax_path, record_count = build_database(genomes, arguments.output_prefix)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"mapseq database: {record_count} references -> {fasta_path}, {tax_path}")


if __name__ == "__main__":
    main()
