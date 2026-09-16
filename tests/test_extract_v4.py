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
