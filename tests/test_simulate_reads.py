"""Simulated reads get the observed reads' primer trim, so read-start errors fall away."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import simulate_amplicon_reads as sar  # noqa: E402  (needs local bin directory)
import subspecies_infer as si  # noqa: E402  (needs local bin directory)

AMPLICON = "TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGCGGACGCTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGATACTGGGTGTCTTGAGTACAGTAGAGGCAGGCGGAATTCGTGG"


def _simulate(tmp_path: Path, trim: bool) -> list[str]:
    amplicons = tmp_path / "amplicons.fasta"
    amplicons.write_text(f">ref|0|ref\n{AMPLICON}\n")
    out = tmp_path / f"sim_{trim}.fasta"
    a = SimpleNamespace(amplicons=amplicons, output=out, n_per_ref=50, read_len=None, seed=0,
                        error_model="flat", model_pt=None, sub_rate=0.0, ins_rate=0.0,
                        del_rate=0.0, use_vi=False, trim_primers=trim,
                        fwd_primer=si.DEFAULT_FWD_PRIMER, rev_primer=si.DEFAULT_REV_PRIMER,
                        primer_mismatches=3)
    # A model that prepends two bases to every read: a read-start error.
    real_sampler = sar.sampler
    sar.sampler = lambda *args: lambda records, rng: [
        (name, "GA" + seq) for name, seq, _ in records]
    try:
        sar.run(a)
    finally:
        sar.sampler = real_sampler
    return [s for _, s in si.read_fasta(out)]


def test_primer_trim_removes_read_start_errors(tmp_path: Path) -> None:
    untrimmed = _simulate(tmp_path, trim=False)
    assert set(untrimmed) == {"GA" + AMPLICON}
    trimmed = _simulate(tmp_path, trim=True)
    assert len(trimmed) == 50 and set(trimmed) == {AMPLICON}
