"""In-silico PCR over references: strand-agnostic, and tolerant of a partial 16S."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import build_panel_kernel as bpk  # noqa: E402
import subspecies_infer as si  # noqa: E402

FWD, REV = "GTGYCAGCMGCCGCGGTAA", "GGACTACNVGGGTWTCTAAT"


def _ssu(seed: int = 0) -> tuple[str, str]:
    """(plus-strand 16S carrying both primer sites, its V4 amplicon)."""
    rng = random.Random(seed)
    flank = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    v4 = flank(250)
    return flank(300) + "GTGCCAGCAGCCGCGGTAA" + v4 + si.revcomp("GGACTACCAGGGTATCTAAT") + flank(300), v4


def test_both_orientations_give_the_same_amplicon() -> None:
    seq, v4 = _ssu()
    assert si.extract_v4(seq, FWD, REV, 3) == v4
    assert si.extract_v4(si.revcomp(seq), FWD, REV, 3) == v4


def test_unamplifiable_is_none() -> None:
    rng = random.Random(1)
    assert si.extract_v4("".join(rng.choice("ACGT") for _ in range(800)), FWD, REV, 3) is None


def _panel(tmp_path: Path, records: dict[str, str]) -> Path:
    path = tmp_path / "panel.fasta"
    path.write_text("".join(f">{h}\n{s}\n" for h, s in records.items()))
    return path


def test_partial_copy_drops_the_genome_instead_of_failing(tmp_path: Path) -> None:
    good, v4 = _ssu()
    partial = good[400:]                     # truncated past the forward primer site
    panel = _panel(tmp_path, {"gA|0|full": good, "gB|0|partial": partial,
                              "gB|1|minus": si.revcomp(good)})
    copies, dropped = bpk.panel_copies(panel, FWD, REV, 3, {})
    assert dropped == []                     # gB is rescued by its minus-strand copy
    assert copies == {"gA": [v4], "gB": [v4]}

    panel = _panel(tmp_path, {"gA|0|full": good, "gB|0|partial": partial})
    copies, dropped = bpk.panel_copies(panel, FWD, REV, 3, {})
    assert dropped == ["gB"] and set(copies) == {"gA"}


def test_all_unamplifiable_still_fails(tmp_path: Path) -> None:
    good, _ = _ssu()
    panel = _panel(tmp_path, {"gA|0|partial": good[400:]})
    with pytest.raises(SystemExit):
        bpk.panel_copies(panel, FWD, REV, 3, {})


def test_rna_reference_is_folded_to_dna(tmp_path: Path, caplog) -> None:
    """SILVA ships RNA; U must fold to T or no primer ever matches."""
    seq, v4 = _ssu()
    rna = _panel(tmp_path, {"gA|0|silva": seq.replace("T", "U")})
    with caplog.at_level("WARNING"):
        records = si.read_fasta(rna)
    assert "RNA" in caplog.text
    assert si.extract_v4(records[0][1], FWD, REV, 3) == v4


def test_batched_extraction_matches_the_per_sequence_scan() -> None:
    """The vectorised whole-DB path must be the per-sequence scan, only faster."""
    rng = random.Random(7)
    seqs = []
    for seed in range(12):
        good, _ = _ssu(seed)
        seqs += [good,                                     # plus strand
                 si.revcomp(good),                         # minus strand
                 good[400:],                               # truncated past the fwd site
                 good[:100] + "N" * 30 + good[130:],       # ambiguous bases
                 "".join(rng.choice("ACGT") for _ in range(rng.randint(200, 1800)))]
    expected = [si.extract_v4(s, FWD, REV, 3) for s in seqs]
    assert any(expected) and not all(expected), "fixture must mix hits and misses"
    assert si.extract_v4_all(seqs, FWD, REV, 3, progress=False) == expected
    assert si.extract_v4_all(seqs, FWD, REV, 3, threads=2, progress=False) == expected
    # Batch boundaries must not change any call.
    assert [a for b in si._batches(seqs, max_cells=4000) for a in
            si.extract_v4_batch(b, FWD, REV, 3)] == expected


def test_stage_amplicons_demo():
    """T over amplifiable entries, references.tax over all of them, bare-accession genomes."""
    si.demo_amplicons()
