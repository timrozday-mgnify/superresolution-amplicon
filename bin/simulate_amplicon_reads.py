#!/usr/bin/env python
"""Simulate errored reads from each reference amplicon, for mis-mapping characterisation.

For every reference in ``amplicons.fasta`` we emit ``--n-per-ref`` reads carrying
sequencing error, named ``{reference header}:{i}`` so the *true source* survives mapping.
Mapping these with the same mapper used for the real reads (mapseq) and tallying where
each lands gives the mis-mapping matrix ``M`` — measured rather than modelled. Ported from
superresolution-shotgun's ``simulate_chunk_reads.py``; output is FASTA because that is
what mapseq reads.

Two error models (``--error-model``):

``trained``
    The skiver context error model (``lib.error_application.apply_batch``) — the same
    sampler the shotgun pipeline uses, context-dependent and platform-calibrated.

``flat``
    A naive constant per-base probability for each mutation type (``--sub-rate``,
    ``--ins-rate``, ``--del-rate``), no training required. ``M`` only needs enough
    variation in the simulated reads to expose which references collide, so an accurate
    error model may be unnecessary; this is the cheap alternative (and drops the whole
    skiver training subworkflow when it suffices).

By default a read is the whole amplicon (correct for merged V4 reads); ``--read-len``
draws a uniform random substring of that length instead, for unmerged/short reads.

``--trim-primers`` simulates each read from the amplicon flanked by its primers and then
trims them off with ``trim_read_primers``, as ``reads_to_fasta.py`` does to the observed
reads. Without it, an error the model places at the start of a read lands inside the
amplicon: 42% of the reads a trained model simulated from bare amplicons carried 1-3 extra
5' bases, and against SILVA NR99's one-base near-ties MAPseq labels those reads differently
from the trimmed observed reads (dev/panel_silva_sweep.md).

Without ``--trim-primers`` the reads keep their primers, as observed reads that were never
trimmed do (``merged: true`` rows from amplicon-analysis-pipeline). The primer bases come
from the oligo mix, not the template, so ``--primer-mix`` draws each read's pair of
concrete oligos from a measured table (``assets/primer_mix_emp_v4.tsv``). Without the
table, untrimmed reads draw each degenerate position uniformly over its code's options.
ponytail: the forward spacer and 3' overhang are not simulated. AAP's cmsearch clip removes
most of both, keeping at most 2 spacer bases and some overhang (dev/aap_merge_effects.md,
0.4); simulate them if the kernel fit shows those few bases matter.

``--mate-len`` writes read pairs instead (``<output>_1.fastq.gz``/``_2``): ``--mate-len``
cycles of R1 over the fragment, and of R2 over its reverse complement (the model's reverse
strand), each running on into its TruSeq adapter. Run through AAP's own fastp merge, these
give merged reads whose errors went through the same merge as the real ones (plan 2.3).
ponytail: without a Phred calibration, trained qualities come from the context error
rate, so fastp's low-against-high quality correction almost never fires. It corrected
9e-5 per base in the Nov2025 run; pass skiver's calibration if that turns out to matter.
"""
from __future__ import annotations

import argparse
import gzip
import logging
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)

# skiver error-model library (ErrorModel.load, apply_batch). In the container it lives at
# $SKIVER_SCRIPTS (/opt/skiver/scripts); from a dev checkout it's the vendored submodule.
_SKIVER_LIB = Path(os.environ.get(
    "SKIVER_SCRIPTS", _HERE.parent / "vendor" / "skiver" / "scripts"))

log = logging.getLogger("simulate_amplicon_reads")

_BASES = "ACGT"
# What a mate reads past the fragment (dev/aap_merge_effects.py). The poly-A tail only
# keeps a mate full length when the fragment plus adapter is shorter than a read.
_ADAPTER_R1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
_ADAPTER_R2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTA"
_FLAT_QUAL = chr(33 + 38)


def draw_fragment(seq: str, read_len: int | None, rng) -> str:
    """A read-length substring drawn uniformly from ``seq`` (whole seq if unset/shorter)."""
    if not read_len or len(seq) <= read_len:
        return seq
    start = int(rng.integers(0, len(seq) - read_len + 1))
    return seq[start: start + read_len]


def apply_flat(seq: str, rng, sub_rate: float, ins_rate: float, del_rate: float) -> str:
    """Mutate ``seq`` with a constant per-base probability of each mutation type.

    Deletion and substitution are drawn from one uniform per base (they are mutually
    exclusive); insertions are drawn independently after the base is emitted, so runs of
    inserts are possible but geometrically rare.
    ponytail: per-base Python loop — fine at n_refs x n_per_ref amplicons; vectorise if
    that ever stops being true.
    """
    out = []
    for base in seq:
        r = rng.random()
        if r < del_rate:
            continue
        if r < del_rate + sub_rate and base in _BASES:
            base = _BASES[(_BASES.index(base) + 1 + int(rng.integers(0, 3))) % 4]
        out.append(base)
        while rng.random() < ins_rate:
            out.append(_BASES[int(rng.integers(0, 4))])
    return "".join(out)


def sampler(error_model: str, model_pt=None, sub_rate: float = 0.005,
            ins_rate: float = 0.0005, del_rate: float = 0.0005, use_vi: bool = False,
            quality: bool = False):
    """Return ``f(records, rng) -> [(name, sequence)]`` for either error model, or
    ``[(name, sequence, quality)]`` with ``quality`` (flat reads are Q38 throughout).

    ``records`` are ``(name, sequence, True)`` triples, the shape skiver's ``apply_batch``
    takes. Both models are reached through one call so callers that only want to
    *characterise* a model (``kernel_align.measure_error_rate``) do not have to
    re-implement the loading.
    """
    if error_model != "trained":
        log.info("flat error model: sub=%g ins=%g del=%g", sub_rate, ins_rate, del_rate)
        flat = lambda seq, rng: apply_flat(seq, rng, sub_rate, ins_rate, del_rate)  # noqa: E731
        if quality:
            return lambda records, rng: [
                (name, m, _FLAT_QUAL * len(m)) for name, seq, _ in records
                for m in [flat(seq, rng)]]
        return lambda records, rng: [(name, flat(seq, rng)) for name, seq, _ in records]
    if str(_SKIVER_LIB) not in sys.path:
        sys.path.insert(0, str(_SKIVER_LIB))
    from lib.error_application import ErrorModel, apply_batch  # noqa: E402
    log.info("loading error model %s", model_pt)
    model = ErrorModel.load(model_pt, use_vi=use_vi)
    if quality:
        return lambda records, rng: [
            (r.name, r.sequence, r.quality)
            for r in apply_batch(model, records, rng, emit_quality=True)]
    return lambda records, rng: [
        (r.name, r.sequence) for r in apply_batch(model, records, rng, emit_quality=False)]


def flank(seq: str, fwd: str, rev: str) -> str:
    """``seq`` between its primers, degenerate codes resolved to their first ACGT option."""
    concrete = lambda p: "".join(sorted(si._IUPAC[c])[0] for c in p)  # noqa: E731
    return concrete(fwd) + seq + concrete(si.revcomp(rev))


def read_primer_mix(path: Path, fwd: str, rev: str) -> tuple[list[tuple[str, str]], np.ndarray]:
    """``([(fwd oligo, rev oligo)], probabilities)`` from a ``fwd rev reads`` table.

    Refuses an oligo that is not a concrete instance of the configured primer: the table
    was measured for one primer pair and means nothing for another."""
    pairs, weights = [], []
    with open(path) as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh if not line.startswith("#")]
    for f, r, n in rows[1:]:
        for oligo, primer in ((f, fwd), (r, rev)):
            if len(oligo) != len(primer) or any(
                    b not in si._IUPAC[c] for b, c in zip(oligo, primer)):
                raise ValueError(f"{path}: oligo {oligo} is not an instance of primer {primer}")
        pairs.append((f, r))
        weights.append(float(n))
    if not pairs:
        raise ValueError(f"{path}: no primer pairs")
    w = np.asarray(weights)
    return pairs, w / w.sum()


def uniform_primer_mix(fwd: str, rev: str):
    """``f(rng, n) -> n oligo pairs``, each degenerate position uniform over its options."""
    options = lambda p: [sorted(si._IUPAC[c]) for c in p]  # noqa: E731
    fo, ro = options(fwd), options(rev)
    one = lambda rng, opts: "".join(o[int(rng.integers(len(o)))] for o in opts)  # noqa: E731
    return lambda rng, n: [(one(rng, fo), one(rng, ro)) for _ in range(n)]


def mate_templates(fragment: str, mate_len: int) -> tuple[str, str]:
    """What R1 and R2 read before errors: ``mate_len`` cycles into the adapter, plus a few
    bases so a deletion still leaves a full-length read."""
    n = mate_len + 10
    return ((fragment + _ADAPTER_R1 + "A" * n)[:n],
            (si.revcomp(fragment) + _ADAPTER_R2 + "A" * n)[:n])


def run(a) -> None:
    rng = np.random.default_rng(a.seed)
    mate_len = getattr(a, "mate_len", None)
    apply_error = sampler(a.error_model, a.model_pt, a.sub_rate, a.ins_rate, a.del_rate,
                          a.use_vi, quality=bool(mate_len))
    trim = getattr(a, "trim_primers", False)
    if mate_len and trim:
        raise ValueError("--mate-len simulates AAP merged reads, which keep their primers: "
                         "drop --trim-primers")
    mix = getattr(a, "primer_mix", None)
    if mix:
        pairs, p = read_primer_mix(mix, a.fwd_primer, a.rev_primer)
        draw_oligos = lambda rng, n: [pairs[i] for i in rng.choice(len(pairs), size=n, p=p)]  # noqa: E731
    elif not trim:
        log.warning("untrimmed reads without --primer-mix: degenerate primer bases are drawn "
                    "uniformly, which is not what a real oligo mix looks like")
        draw_oligos = uniform_primer_mix(a.fwd_primer, a.rev_primer)
    else:
        draw_oligos = None   # trimmed off anyway: the first option is as good as any

    if mate_len:
        return run_pairs(a, rng, apply_error, draw_oligos, mate_len)
    n_reads = 0
    with open(a.output, "w") as out:
        # Stream reference-by-reference so a large DB never holds all reads in memory.
        for header, seq in si.read_fasta(a.amplicons):
            if draw_oligos:
                frags = [draw_fragment(f + seq + si.revcomp(r), a.read_len, rng)
                         for f, r in draw_oligos(rng, a.n_per_ref)]
            else:
                if trim:
                    seq = flank(seq, a.fwd_primer, a.rev_primer)
                frags = [draw_fragment(seq, a.read_len, rng) for _ in range(a.n_per_ref)]
            recs = [(f"{header}:{i}", f, True) for i, f in enumerate(frags)
                    if f and set(f) <= set(_BASES)]
            if not recs:
                log.warning("reference %s produced no usable fragment (non-ACGT?)", header)
                continue
            for name, sequence in apply_error(recs, rng):
                if trim:
                    sequence = si.trim_read_primers(sequence, a.fwd_primer, a.rev_primer,
                                                    a.primer_mismatches)
                if not sequence:
                    continue
                out.write(f">{name}\n{sequence}\n")
                n_reads += 1
    log.info("wrote %d simulated reads -> %s", n_reads, a.output)
    print(f"simulate_amplicon_reads: {n_reads} reads -> {a.output}")


def run_pairs(a, rng, apply_error, draw_oligos, mate_len: int) -> None:
    """Write ``<output>_1.fastq.gz``/``_2``; both mates of a pair share a name."""
    n_pairs = 0
    with gzip.open(f"{a.output}_1.fastq.gz", "wt") as o1, \
            gzip.open(f"{a.output}_2.fastq.gz", "wt") as o2:
        for header, seq in si.read_fasta(a.amplicons):
            if not set(seq) <= set(_BASES):
                log.warning("reference %s produced no usable fragment (non-ACGT?)", header)
                continue
            r1s, r2s = [], []
            for i, (f, r) in enumerate(draw_oligos(rng, a.n_per_ref)):
                t1, t2 = mate_templates(f + seq + si.revcomp(r), mate_len)
                r1s.append((f"{header}:{i}", t1, True))
                r2s.append((f"{header}:{i}", t2, False))
            for (name, s1, q1), (_, s2, q2) in zip(apply_error(r1s, rng), apply_error(r2s, rng)):
                o1.write(f"@{name}\n{s1[:mate_len]}\n+\n{q1[:mate_len]}\n")
                o2.write(f"@{name}\n{s2[:mate_len]}\n+\n{q2[:mate_len]}\n")
                n_pairs += 1
    log.info("wrote %d simulated pairs -> %s_{1,2}.fastq.gz", n_pairs, a.output)
    print(f"simulate_amplicon_reads: {n_pairs} pairs -> {a.output}_{{1,2}}.fastq.gz")


def demo() -> None:
    """Self-check the fragment sampler and the flat error model (the trained sampler is
    skiver's own, and tested there)."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list(_BASES), size=1000))   # random -> fragments are unique
    assert draw_fragment(seq, None, rng) == seq
    assert draw_fragment("ACGTACGT", 100, rng) == "ACGTACGT"      # shorter than read_len
    frags = [draw_fragment(seq, 100, rng) for _ in range(500)]
    assert all(len(f) == 100 for f in frags)
    starts = [seq.index(f) for f in frags]
    assert min(starts) < 50 and max(starts) > len(seq) - 150, (min(starts), max(starts))

    # Flat model: zero rates are a no-op; realistic rates mutate a measurable fraction.
    assert apply_flat(seq, rng, 0.0, 0.0, 0.0) == seq
    subs = [apply_flat(seq, rng, 0.01, 0.0, 0.0) for _ in range(20)]
    diffs = np.mean([sum(x != y for x, y in zip(seq, s)) / len(seq) for s in subs])
    assert 0.005 < diffs < 0.02, diffs                # ~1% substituted (3/4 of draws differ)
    dels = [apply_flat(seq, rng, 0.0, 0.0, 0.02) for _ in range(20)]
    assert 0.97 < np.mean([len(d) for d in dels]) / len(seq) < 0.99
    ins = [apply_flat(seq, rng, 0.0, 0.02, 0.0) for _ in range(20)]
    assert 1.01 < np.mean([len(i) for i in ins]) / len(seq) < 1.03
    print("demo OK: fragments read-length & edge-covering; flat rates hit their targets")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicons", type=Path, help="amplicons.fasta from `subspecies_infer.py amplicons`")
    ap.add_argument("--error-model", choices=["trained", "flat"], default="trained")
    ap.add_argument("--model-pt", type=Path, help="trained skiver error model (.pt)")
    ap.add_argument("--use-vi", action="store_true", help="use the model's VI posterior mean")
    ap.add_argument("--sub-rate", type=float, default=0.005, help="flat model: per-base substitution")
    ap.add_argument("--ins-rate", type=float, default=0.0005, help="flat model: per-base insertion")
    ap.add_argument("--del-rate", type=float, default=0.0005, help="flat model: per-base deletion")
    ap.add_argument("--read-len", type=int, default=None,
                    help="draw substrings of this length (default: the whole amplicon)")
    ap.add_argument("--n-per-ref", type=int, default=500,
                    help="simulated reads per reference (mis-mapping sampling depth)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trim-primers", action="store_true",
                    help="simulate from primer-flanked amplicons and trim the primers off "
                         "each read, as the observed reads are")
    ap.add_argument("--fwd-primer", default=si.DEFAULT_FWD_PRIMER)
    ap.add_argument("--rev-primer", default=si.DEFAULT_REV_PRIMER)
    ap.add_argument("--primer-mismatches", type=int, default=3)
    ap.add_argument("--primer-mix", type=Path, default=None,
                    help="fwd/rev/reads table of concrete primer oligos to draw each read's "
                         "primers from (default: uniform over the codes when untrimmed)")
    ap.add_argument("--mate-len", type=int, default=None,
                    help="write read pairs of this many cycles (<output>_1/_2.fastq.gz) "
                         "instead of FASTA, for AAP's fastp merge")
    ap.add_argument("-o", "--output", type=Path,
                    help="output FASTA (mapseq input), or the pair prefix with --mate-len")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if a.demo:
        return demo()
    for req in ("amplicons", "output"):
        if getattr(a, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (unless --demo)")
    if a.error_model == "trained" and a.model_pt is None:
        ap.error("--model-pt is required for --error-model trained")
    run(a)


if __name__ == "__main__":
    main()
