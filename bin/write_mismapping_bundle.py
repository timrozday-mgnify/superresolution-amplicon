#!/usr/bin/env python3
"""Write a self-contained, canonical mismapping-matrix bundle."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import shutil
from pathlib import Path

import numpy as np
REFERENCE_FILES = (
    "amplicons.fasta",
    "amplicons.tax",
    "translation_table.tsv",
    "refseq_index.csv",
)


def main() -> None:
    """Copy matrix inputs and write bundle provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--amplicon-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance-base64", required=True)
    parser.add_argument("--members-base64", required=True)
    parser.add_argument("--group-record", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    reference_dir = output_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "mismapping_matrix.npz"
    if args.matrix.suffix == ".npz":
        shutil.copy2(args.matrix, matrix_path)
    else:
        with args.matrix.open(newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows or len(rows[0]) < 2:
            raise SystemExit("legacy mis-mapping CSV is empty or malformed")
        refs = rows[0][1:]
        if [row[0] for row in rows[1:]] != refs:
            raise SystemExit("legacy mis-mapping CSV must have identical row and column IDs")
        data: list[float] = []
        indices: list[int] = []
        indptr = [0]
        for row in rows[1:]:
            if len(row) != len(refs) + 1:
                raise SystemExit("legacy mis-mapping CSV has inconsistent row widths")
            for column, value in enumerate(row[1:]):
                probability = float(value)
                if probability:
                    data.append(probability)
                    indices.append(column)
            indptr.append(len(data))
        np.savez_compressed(
            matrix_path,
            data=np.asarray(data, dtype=np.float64),
            indices=np.asarray(indices, dtype=np.int64),
            indptr=np.asarray(indptr, dtype=np.int64),
            shape=np.asarray((len(refs), len(refs)), dtype=np.int64),
            refseqs=np.asarray(refs, dtype=np.str_),
        )
    for name in REFERENCE_FILES:
        shutil.copy2(args.amplicon_dir / name, reference_dir / name)

    provenance = json.loads(base64.b64decode(args.provenance_base64))
    members = json.loads(base64.b64decode(args.members_base64))
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    with (output_dir / "samples.tsv").open("w") as handle:
        handle.write("sample_id\tplatform\n")
        for member in members:
            handle.write(f"{member['id']}\t{member['platform']}\n")
    args.group_record.write_text(
        "matrix_key\treference_sha256\tmodel_scope\tsource\tmatrix_path\tsamples\n"
        f"{provenance['matrix_key']}\t{provenance['reference_sha256']}\t"
        f"{provenance['model_scope']}\t{provenance['source']}\t"
        f"mismapping/{provenance['matrix_key']}/mismapping_matrix.npz\t"
        f"{','.join(member['id'] for member in members)}\n"
    )


if __name__ == "__main__":
    main()
