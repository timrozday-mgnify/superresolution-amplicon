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
"""
from __future__ import annotations

import argparse
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


def run(a) -> None:
    rng = np.random.default_rng(a.seed)
    apply_batch = model = None
    if a.error_model == "trained":
        if str(_SKIVER_LIB) not in sys.path:
            sys.path.insert(0, str(_SKIVER_LIB))
        from lib.error_application import ErrorModel, apply_batch  # noqa: E402
        log.info("loading error model %s", a.model_pt)
        model = ErrorModel.load(a.model_pt, use_vi=a.use_vi)
    else:
        log.info("flat error model: sub=%g ins=%g del=%g",
                 a.sub_rate, a.ins_rate, a.del_rate)

    n_reads = 0
    with open(a.output, "w") as out:
        # Stream reference-by-reference so a large DB never holds all reads in memory.
        for header, seq in si.read_fasta(a.amplicons):
            frags = [draw_fragment(seq, a.read_len, rng) for _ in range(a.n_per_ref)]
            recs = [(f"{header}:{i}", f, True) for i, f in enumerate(frags)
                    if f and set(f) <= set(_BASES)]
            if not recs:
                log.warning("reference %s produced no usable fragment (non-ACGT?)", header)
                continue
            if apply_batch is not None:
                reads = [(r.name, r.sequence) for r in apply_batch(model, recs, rng,
                                                                   emit_quality=False)]
            else:
                reads = [(name, apply_flat(f, rng, a.sub_rate, a.ins_rate, a.del_rate))
                         for name, f, _ in recs]
            for name, sequence in reads:
                if not sequence:
                    continue
                out.write(f">{name}\n{sequence}\n")
                n_reads += 1
    log.info("wrote %d simulated reads -> %s", n_reads, a.output)
    print(f"simulate_amplicon_reads: {n_reads} reads -> {a.output}")


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
    ap.add_argument("-o", "--output", type=Path, help="output FASTA (mapseq input)")
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
