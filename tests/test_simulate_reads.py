"""Simulated reads are prepared like the observed ones: primer-trimmed, or carrying primers
drawn from the oligo mix."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import simulate_amplicon_reads as sar  # noqa: E402  (needs local bin directory)
import subspecies_infer as si  # noqa: E402  (needs local bin directory)

AMPLICON = "TACGGAGGATCCGAGCGTTATCCGGATTTATTGGGTTTAAAGGGAGCGTAGGCGGACGCTTAAGTCAGTTGTGAAAGTTTGCGGCTCAACCGTAAAATTGCAGTTGATACTGGGTGTCTTGAGTACAGTAGAGGCAGGCGGAATTCGTGG"


def _simulate(tmp_path: Path, trim: bool, primer_mix: Path | None = None,
              n: int = 50) -> list[str]:
    amplicons = tmp_path / "amplicons.fasta"
    amplicons.write_text(f">ref|0|ref\n{AMPLICON}\n")
    out = tmp_path / f"sim_{trim}.fasta"
    a = SimpleNamespace(amplicons=amplicons, output=out, n_per_ref=n, read_len=None, seed=0,
                        error_model="flat", model_pt=None, sub_rate=0.0, ins_rate=0.0,
                        del_rate=0.0, use_vi=False, trim_primers=trim,
                        fwd_primer=si.DEFAULT_FWD_PRIMER, rev_primer=si.DEFAULT_REV_PRIMER,
                        primer_mismatches=3, primer_mix=primer_mix)
    # A model that prepends two bases to every read: a read-start error.
    real_sampler = sar.sampler
    sar.sampler = lambda *args, **kw: lambda records, rng: [
        (name, "GA" + seq) for name, seq, _ in records]
    try:
        sar.run(a)
    finally:
        sar.sampler = real_sampler
    return [s for _, s in si.read_fasta(out)]


def test_primer_trim_removes_read_start_errors(tmp_path: Path) -> None:
    untrimmed = _simulate(tmp_path, trim=False)
    assert all(r.startswith("GA") and AMPLICON in r for r in untrimmed)
    trimmed = _simulate(tmp_path, trim=True)
    assert len(trimmed) == 50 and set(trimmed) == {AMPLICON}


MIX = ROOT / "assets" / "primer_mix_emp_v4.tsv"


def _oligos(read: str) -> tuple[str, str]:
    """(fwd, rev) oligos of an error-free simulated read, rev on its own strand."""
    body = read[2:]   # the fake model's "GA"
    lf, lr = len(si.DEFAULT_FWD_PRIMER), len(si.DEFAULT_REV_PRIMER)
    assert body[lf:len(body) - lr] == AMPLICON
    return body[:lf], si.revcomp(body[len(body) - lr:])


def test_untrimmed_reads_carry_the_measured_oligo_mix(tmp_path: Path) -> None:
    reads = _simulate(tmp_path, trim=False, primer_mix=MIX, n=4000)
    fwd, rev = zip(*map(_oligos, reads))
    # 515F position 8 (M) is A in 0.618 of reads; 806R position 7 (N) is never G.
    assert abs(sum(f[8] == "A" for f in fwd) / len(fwd) - 0.618) < 0.03
    assert not any(r[7] == "G" for r in rev)
    pairs, _ = sar.read_primer_mix(MIX, si.DEFAULT_FWD_PRIMER, si.DEFAULT_REV_PRIMER)
    assert set(zip(fwd, rev)) <= set(pairs)


def test_uniform_fallback_draws_every_option(tmp_path: Path) -> None:
    reads = _simulate(tmp_path, trim=False, n=400)
    fwd, rev = zip(*map(_oligos, reads))
    assert {f[3] for f in fwd} == {"C", "T"} and {r[7] for r in rev} == set("ACGT")


def test_mix_for_another_primer_is_refused(tmp_path: Path) -> None:
    try:
        sar.read_primer_mix(MIX, "CCTACGGGNGGCWGCAG", si.DEFAULT_REV_PRIMER)
    except ValueError as e:
        assert "not an instance" in str(e)
    else:
        raise AssertionError("a V3 primer accepted a V4 mix")


def test_pairs_read_the_fragment_from_both_ends_into_the_adapter(tmp_path: Path) -> None:
    import gzip
    amplicons = tmp_path / "amplicons.fasta"
    amplicons.write_text(f">ref|0|ref\n{AMPLICON}\n")
    a = SimpleNamespace(amplicons=amplicons, output=tmp_path / "sim", n_per_ref=20,
                        read_len=None, seed=0, error_model="flat", model_pt=None,
                        sub_rate=0.0, ins_rate=0.0, del_rate=0.0, use_vi=False,
                        trim_primers=False, fwd_primer=si.DEFAULT_FWD_PRIMER,
                        rev_primer=si.DEFAULT_REV_PRIMER, primer_mismatches=3,
                        primer_mix=MIX, mate_len=310)
    sar.run(a)
    mates = [gzip.open(tmp_path / f"sim_{m}.fastq.gz", "rt").read().splitlines()
             for m in (1, 2)]
    for r1, r2 in zip(zip(*[iter(mates[0])] * 4), zip(*[iter(mates[1])] * 4)):
        assert r1[0] == r2[0] and r1[0].startswith("@ref|0|ref:")
        assert len(r1[1]) == len(r2[1]) == len(r1[3]) == 310
        fragment = r1[1][:r1[1].index(sar._ADAPTER_R1)]
        assert AMPLICON in fragment
        assert r2[1].startswith(si.revcomp(fragment) + sar._ADAPTER_R2)


def test_pairs_refuse_primer_trimming(tmp_path: Path) -> None:
    import pytest
    a = SimpleNamespace(amplicons=None, output=tmp_path / "sim", n_per_ref=1, read_len=None,
                        seed=0, error_model="flat", model_pt=None, sub_rate=0.0,
                        ins_rate=0.0, del_rate=0.0, use_vi=False, trim_primers=True,
                        fwd_primer=si.DEFAULT_FWD_PRIMER, rev_primer=si.DEFAULT_REV_PRIMER,
                        primer_mismatches=3, primer_mix=None, mate_len=310)
    with pytest.raises(ValueError, match="keep their primers"):
        sar.run(a)
