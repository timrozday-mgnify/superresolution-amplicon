#!/usr/bin/env python3
"""Write a self-contained, canonical mismapping-matrix bundle."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path


REFERENCE_FILES = (
    "amplicons.fasta",
    "amplicons.tax",
    "translation_table.csv",
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
    shutil.copy2(args.matrix, output_dir / "mismapping_matrix.csv")
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
        f"mismapping/{provenance['matrix_key']}/mismapping_matrix.csv\t"
        f"{','.join(member['id'] for member in members)}\n"
    )


if __name__ == "__main__":
    main()
