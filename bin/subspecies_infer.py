#!/usr/bin/env python
"""Error-model-driven sub-species composition inference for amplicon runs.

Two stages, run from the reports Taskfile (basicpython env):

``mismapping``
    Estimate the reference-to-reference mis-mapping matrix ``M`` **analytically
    from the trained skiver error model** (no read simulation / re-mapping), plus
    the deterministic genome->reference translation table ``T``. For each pair of
    V4 reference amplicons (A, B) we model mapseq as a max-likelihood assigner and
    form the per-read log-likelihood ratio ``LLR(A,B)=logP(r|A)-logP(r|B)`` for a
    read ``r`` drawn from A's error model. Positions are independent, so
    ``E[LLR]`` (= per-column KL of the emission distributions) and ``Var[LLR]`` are
    closed-form; ``P(A mis-assigned to B) ~= Phi(-E[LLR]/sd[LLR])``.

``infer``
    Bayesian inference (Pyro: NUTS / VI / MLE) on the latent *true genome*
    composition ``theta``. Generative model:
    ``theta ~ Dirichlet(alpha)`` -> ``r_true = theta @ T`` (per reference) ->
    ``r_obs = r_true @ M`` (per reference, observed) -> likelihood on the observed
    per-reference mseq abundance. Started from the observed composition.

Both stages carry a ``--demo`` self-check that needs no run data.

Note: inference uses **Pyro**, not NumPyro, because loading the skiver error model
(`lib.error_application` -> `lib.context_error_models`) already imports pyro, so a
NumPyro+JAX stack would be redundant weight in the same env.

Run:
    subspecies_infer.py mismapping --db-fasta db.fasta --model-pt m.pt -o out/
    subspecies_infer.py infer --run-dir RUN --mismap-dir out/ --mode vi -o out/
    subspecies_infer.py mismapping --demo   # self-check
    subspecies_infer.py infer --demo        # self-check
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("subspecies_infer")

_HERE = Path(__file__).resolve().parent
# skiver error-model library (ErrorModel.load, _logits_for_reference, _masked_probs).
# In the container it lives at $SKIVER_SCRIPTS (/opt/skiver/scripts); when run from a
# checkout it's the vendored submodule. superresolution-amplicon: bin/ layout, so the
# env var is the primary resolution and the relative path is a dev-checkout fallback.
_SKIVER_LIB = Path(os.environ.get(
    "SKIVER_SCRIPTS", _HERE.parent / "vendor" / "skiver" / "scripts"))

# 515-YF / 806BR (V4). Overridable on the CLI.
DEFAULT_FWD_PRIMER = "GTGYCAGCMGCCGCGGTAA"
DEFAULT_REV_PRIMER = "GGACTACNVGGGTWTCTAAT"

_IUPAC = {
    "A": set("A"), "C": set("C"), "G": set("G"), "T": set("T"),
    "R": set("AG"), "Y": set("CT"), "S": set("GC"), "W": set("AT"),
    "K": set("GT"), "M": set("AC"), "B": set("CGT"), "D": set("AGT"),
    "H": set("ACT"), "V": set("ACG"), "N": set("ACGT"),
}
_COMP = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


# ── FASTA / headers / primers ────────────────────────────────────────────────


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Return ``[(header, sequence)]``; header is the full id line (sans '>')."""
    opener = gzip.open if str(path).endswith(".gz") else open
    records: list[tuple[str, str]] = []
    header, chunks = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:].strip().split()[0], []
            else:
                chunks.append(line.strip().upper())
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def genome_of_header(header: str) -> str:
    """DB entry header ``genome|index|orig`` -> genome id (before first '|')."""
    return header.split("|", 1)[0]


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def _find_primer(seq: str, primer: str, max_mismatch: int, start: int = 0) -> int | None:
    """Earliest index >= ``start`` where ``primer`` (IUPAC) matches ``seq`` within
    ``max_mismatch`` mismatches, else ``None``."""
    lp = len(primer)
    for i in range(start, len(seq) - lp + 1):
        mm = 0
        for j in range(lp):
            if seq[i + j] not in _IUPAC.get(primer[j], set(seq[i + j])):
                mm += 1
                if mm > max_mismatch:
                    break
        if mm <= max_mismatch:
            return i
    return None


def extract_v4(seq: str, fwd: str, rev: str, max_mismatch: int) -> str | None:
    """In-silico PCR: return the amplicon between the forward primer and the
    reverse-complement of the reverse primer, or ``None`` if not amplifiable.

    ponytail: exact-window IUPAC scan with a mismatch budget; AmpliconHunter may
    amplify a few edge cases this misses, which just drops them from the estimate.
    """
    f = _find_primer(seq, fwd, max_mismatch)
    if f is None:
        return None
    amp_start = f + len(fwd)
    rc = revcomp(rev)
    r = _find_primer(seq, rc, max_mismatch, start=amp_start)
    if r is None:
        return None
    amplicon = seq[amp_start:r]
    return amplicon or None


# ── Error-model emission distributions & mis-mapping matrix ───────────────────


def _load_error_model(model_pt: Path, use_vi: bool):
    if str(_SKIVER_LIB) not in sys.path:
        sys.path.insert(0, str(_SKIVER_LIB))
    from lib.error_application import ErrorModel  # noqa: E402  (needs sys.path)

    return ErrorModel.load(model_pt, use_vi=use_vi)


def emission_distribution(model, seq: str) -> np.ndarray:
    """Per-position emitted-base distribution ``[L, 4]`` for a reference amplicon.

    Uses the error model's masked per-position error-type probabilities: emitting
    the true base is the match probability, emitting another base is that
    substitution probability. Insertion/deletion mass is dropped and the four
    base probabilities renormalised (indels are handled at alignment gaps).
    """
    if str(_SKIVER_LIB) not in sys.path:
        sys.path.insert(0, str(_SKIVER_LIB))
    from lib.error_application import _masked_probs  # noqa: E402
    from lib.error_application import _CHAR_TO_IDX, _ERR_SUB_START

    raw = np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)
    ref_idx = _CHAR_TO_IDX[raw].astype(np.int64)
    probs = _masked_probs(model._logits_for_reference(ref_idx, True), ref_idx)  # [L,10]
    emit = np.empty((ref_idx.shape[0], 4), dtype=np.float64)
    for b in range(4):
        emit[:, b] = np.where(ref_idx == b, probs[:, 0], probs[:, _ERR_SUB_START + b])
    emit /= emit.sum(axis=1, keepdims=True)
    return emit


_C2I_CACHE = None


def _char_to_idx() -> np.ndarray:
    """Cached ASCII->base-index (ACGT->0123) table from the skiver error model, so the
    read scorer maps observed bases with the same convention as ``emission_distribution``."""
    global _C2I_CACHE
    if _C2I_CACHE is None:
        if str(_SKIVER_LIB) not in sys.path:
            sys.path.insert(0, str(_SKIVER_LIB))
        from lib.error_application import _CHAR_TO_IDX  # noqa: E402
        _C2I_CACHE = _CHAR_TO_IDX
    return _C2I_CACHE


def _oriented_score(read: str, read_idx: np.ndarray, ref_emit: np.ndarray,
                    ref_seq: str, gap_penalty: float) -> float:
    """Length-normalised log P(read | ref) for one read orientation. Aligns the ref
    into the read (edlib infix, so read primer/overhang bases are free) and sums the
    per-column emission log-prob; each indel column costs ``gap_penalty``."""
    import math

    total = 0.0
    n = 0
    for ai, bj in _align_columns(ref_seq, read, mode="HW"):
        if ai is None or bj is None:
            total -= gap_penalty
        else:
            total += math.log(ref_emit[ai, read_idx[bj]] + 1e-12)
        n += 1
    return total / n if n else float("-inf")


def read_score(read: str, ref_emit: np.ndarray, ref_seq: str, gap_penalty: float) -> float:
    """Best-orientation length-normalised raw log-likelihood of ``read`` under ``ref``.

    Raw (not softmax'd across refs): keeps absolute fit, so off-target reads score low
    under every reference and self-sort into the low-score background. ``ref_emit`` is
    the precomputed ``emission_distribution(model, ref_seq)``.
    """
    c2i = _char_to_idx()
    rev = revcomp(read)
    fi = c2i[np.frombuffer(read.encode("ascii", "replace"), dtype=np.uint8)]
    ri = c2i[np.frombuffer(rev.encode("ascii", "replace"), dtype=np.uint8)]
    return max(_oriented_score(read, fi, ref_emit, ref_seq, gap_penalty),
               _oriented_score(rev, ri, ref_emit, ref_seq, gap_penalty))


def _align_columns(a: str, b: str, mode: str = "NW") -> list[tuple[int | None, int | None]]:
    """Align ``a`` (query) to ``b`` (target) with edlib; return aligned columns as
    ``(ai, bj)`` where a gap is ``None``. ``mode="HW"`` (infix) leaves target (``b``)
    prefix/suffix overhangs free — used to align a reference into a read whose primer
    bases overhang the amplicon."""
    import edlib

    res = edlib.align(a, b, task="path", mode=mode)
    cols: list[tuple[int | None, int | None]] = []
    ai = bj = 0
    for length, op in _parse_cigar(res["cigar"]):
        for _ in range(length):
            if op in "=X":
                cols.append((ai, bj)); ai += 1; bj += 1
            elif op == "I":  # base in query (a), gap in target (b)
                cols.append((ai, None)); ai += 1
            else:            # 'D': base in target (b), gap in query (a)
                cols.append((None, bj)); bj += 1
    return cols


def _parse_cigar(cigar: str) -> list[tuple[int, str]]:
    out, num = [], ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            out.append((int(num), ch)); num = ""
    return out


def _llr_moments(emit_a: np.ndarray, emit_b: np.ndarray, seq_a: str, seq_b: str,
                 gap_penalty: float) -> tuple[float, float]:
    """Mean and variance of ``LLR(A,B)`` for a read drawn from A, aligned A->B.

    Match/mismatch columns contribute the KL divergence (mean) and second-moment
    of the per-base log-ratio under A's emission distribution. Gap columns add a
    fixed ``gap_penalty`` favouring A (near-identical 16S rarely differ by indels).
    """
    mean = var = 0.0
    for ai, bj in _align_columns(seq_a, seq_b):
        if ai is None or bj is None:
            mean += gap_penalty  # deterministic; no variance contribution
            continue
        pa = emit_a[ai]
        pb = np.clip(emit_b[bj], 1e-9, None)
        logratio = np.log(np.clip(pa, 1e-9, None)) - np.log(pb)  # [4]
        e = float((pa * logratio).sum())               # KL(pa||pb) >= 0
        e2 = float((pa * logratio ** 2).sum())
        mean += e
        var += max(e2 - e * e, 0.0)
    return mean, var


def build_mismapping_matrix(emits: list[np.ndarray], seqs: list[str],
                            gap_penalty: float, progress: bool = True) -> np.ndarray:
    """Row-stochastic mis-mapping matrix ``M[a,b]=P(assign a->b)``.

    Off-diagonal weight ``Phi(-E[LLR]/sd[LLR])`` (probability a read from A scores
    higher under B than under A); the row is then normalised with a unit diagonal.
    """
    from scipy.stats import norm

    n = len(seqs)
    M = np.eye(n, dtype=np.float64)
    rows = _progress(range(n), total=n, desc=f"mismapping {n}x{n} refs",
                     enabled=progress, unit="ref")
    for a in rows:
        for b in range(n):
            if a == b:
                continue
            mean, var = _llr_moments(emits[a], emits[b], seqs[a], seqs[b], gap_penalty)
            sd = np.sqrt(var)
            if sd < 1e-9:
                M[a, b] = 1.0 if mean < 1e-9 else 0.0  # identical -> full confusion
            else:
                M[a, b] = float(norm.cdf(-mean / sd))
    M /= M.sum(axis=1, keepdims=True)
    return M


def _progress(iterable, *, total=None, desc="", enabled=True, unit="it", leave=False):
    """tqdm progress bar when enabled, else the plain iterable (tqdm ships with pyro)."""
    if not enabled:
        return iterable
    from tqdm.auto import tqdm
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave)


def stage_mismapping(args) -> None:
    log.info("loading DB fasta %s", args.db_fasta)
    records = read_fasta(args.db_fasta)
    log.info("loading error model %s", args.model_pt)
    model = _load_error_model(args.model_pt, args.use_vi)
    log.info("extracting V4 amplicons (primers %s / %s, <=%d mismatches) from %d entries",
             args.fwd_primer, args.rev_primer, args.primer_mismatches, len(records))

    refseqs: list[str] = []          # DB entry headers with a valid V4
    v4: list[str] = []
    genomes_of: list[str] = []
    idx_rows: list[dict] = []
    for header, seq in records:
        amp = extract_v4(seq, args.fwd_primer, args.rev_primer, args.primer_mismatches)
        amplifiable = amp is not None
        idx_rows.append({"refseq": header, "genome": genome_of_header(header),
                         "v4_len": len(amp) if amp else 0, "amplifiable": amplifiable})
        if amplifiable:
            refseqs.append(header)
            v4.append(amp)
            genomes_of.append(genome_of_header(header))
    if not refseqs:
        raise SystemExit("no reference produced a V4 amplicon; check primers / DB fasta")
    log.info("%d/%d entries amplifiable; computing mis-mapping via '%s'",
             len(refseqs), len(records), args.method)

    emits = [emission_distribution(model, s) for s in v4]
    t0 = time.time()
    if args.method == "simulate":
        M = _mismapping_by_simulation(model, refseqs, v4, args)
    else:
        M = build_mismapping_matrix(emits, v4, args.gap_penalty, progress=args.progress)
    log.info("mis-mapping matrix computed in %.1fs", time.time() - t0)

    genomes = sorted(set(genomes_of))
    T = np.zeros((len(genomes), len(refseqs)), dtype=np.float64)
    g_idx = {g: i for i, g in enumerate(genomes)}
    for j, g in enumerate(genomes_of):
        T[g_idx[g], j] = 1.0
    # Rows sum to 1: the within-genome distribution of a genome's amplicon reads over
    # its 16S copies (uniform). This makes the latent theta the true *read-space* genome
    # composition (r_true genome-marginal = theta), directly comparable to the read-space
    # realized truth and observed profile — no spurious copy-number reweighting.
    T = T / T.sum(axis=1, keepdims=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(M, index=refseqs, columns=refseqs).to_csv(out / "mismapping_matrix.csv")
    pd.DataFrame(T, index=genomes, columns=refseqs).to_csv(out / "translation_table.csv")
    pd.DataFrame(idx_rows).to_csv(out / "refseq_index.csv", index=False)
    # Reference V4 amplicons — needed by `infer --score-hist` to score reads.
    with open(out / "v4_amplicons.fasta", "w") as fh:
        for h, s in zip(refseqs, v4):
            fh.write(f">{h}\n{s}\n")

    if args.score_components:
        log.info("simulating score components (%d reads/ref, %d bins)", args.n_sim, args.sim_bins)
        t0 = time.time()
        D, bin_edges, M_sim = simulate_score_components(
            model, v4, emits, args.gap_penalty, args.n_sim, args.sim_bins, args.seed,
            progress=args.progress)
        log.info("score components computed in %.1fs", time.time() - t0)
        np.savez_compressed(out / "score_components.npz", D=D, bin_edges=bin_edges,
                            refseqs=np.array(refseqs, dtype=object))
        pd.DataFrame(M_sim, index=refseqs, columns=refseqs).to_csv(
            out / "mismapping_matrix_sim.csv")
        diff = np.abs(M - M_sim)
        diag_a, diag_s = np.diag(M), np.diag(M_sim)
        pd.DataFrame([{
            "max_abs_diff": float(diff.max()),
            "frobenius": float(np.linalg.norm(M - M_sim)),
            "diag_corr": float(np.corrcoef(diag_a, diag_s)[0, 1]) if len(diag_a) > 1 else 1.0,
        }]).to_csv(out / "mismapping_compare.csv", index=False)

    print(f"mismapping: {len(refseqs)} amplifiable refs / {len(records)} DB entries, "
          f"{len(genomes)} genomes -> {out}")


def _mismapping_by_simulation(model, refseqs, v4, args) -> np.ndarray:
    """Cross-check M: simulate N errored V4 reads per ref, identity-assign via edlib."""
    import edlib
    from lib.error_application import apply_batch

    rng = np.random.default_rng(args.seed)
    n = len(refseqs)
    M = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        reads = apply_batch(model, [(f"r{a}_{i}", v4[a], True) for i in range(args.n_sim)],
                            rng, emit_quality=False)
        for rec in reads:
            best, best_d = a, None
            for b in range(n):
                d = edlib.align(rec.sequence, v4[b], task="distance")["editDistance"]
                if best_d is None or d < best_d:
                    best, best_d = b, d
            M[a, best] += 1.0
    M /= M.sum(axis=1, keepdims=True)
    return M


def simulate_score_components(model, v4: list[str], emits: list[np.ndarray],
                              gap_penalty: float, n_sim: int, n_bins: int, seed: int,
                              progress: bool = True):
    """Simulate errored reads from each reference and score them against every reference.

    Returns ``(D, bin_edges, M_sim)``:
    - ``D[a, j, :]`` — the score histogram (normalised to sum 1) of reads truly from ``a``
      scored against ``j``. This is the mis-mapping matrix ``M`` generalised from a matrix
      of means to a tensor of full score distributions; it is the model's per-reference
      mixture component.
    - ``bin_edges`` — shared score bins (from the pooled simulated-score range), reused by
      ``observed_score_histograms`` so observed and predicted histograms align. Observed
      off-target reads scoring below the range pile into bin 0 (the background).
    - ``M_sim[a, j]`` — argmax confusion (fraction of ``a``'s reads whose best-scoring ref
      is ``j``), row-stochastic; the empirical counterpart of the analytic ``M`` for the
      "is the closed-form sufficient?" check.
    """
    from lib.error_application import apply_batch

    rng = np.random.default_rng(seed)
    n = len(v4)
    # Simulated reads are generated forward from v4[a]; score forward orientation only.
    c2i = _char_to_idx()
    scores = np.empty((n, n_sim, n), dtype=np.float64)  # [source a, read, target j]
    rows = _progress(range(n), total=n, desc=f"score-components {n} refs",
                     enabled=progress, unit="ref")
    for a in rows:
        reads = apply_batch(model, [(f"r{a}_{i}", v4[a], True) for i in range(n_sim)],
                            rng, emit_quality=False)
        for i, rec in enumerate(reads):
            ridx = c2i[np.frombuffer(rec.sequence.encode("ascii", "replace"), dtype=np.uint8)]
            for j in range(n):
                scores[a, i, j] = _oriented_score(rec.sequence, ridx, emits[j], v4[j],
                                                  gap_penalty)

    lo, hi = np.percentile(scores, [0.1, 99.9])
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    D = np.zeros((n, n, n_bins), dtype=np.float64)
    for a in range(n):
        for j in range(n):
            D[a, j], _ = np.histogram(scores[a, :, j], bins=bin_edges)
    D += 1e-6                                   # avoid zero-prob bins under the mixture
    D /= D.sum(axis=2, keepdims=True)

    best = scores.argmax(axis=2)                # [n, n_sim] best-scoring ref per read
    M_sim = np.zeros((n, n), dtype=np.float64)
    for a in range(n):
        for j in best[a]:
            M_sim[a, j] += 1.0
    M_sim /= M_sim.sum(axis=1, keepdims=True)
    return D, bin_edges, M_sim


# ── Observed abundances from mseq ─────────────────────────────────────────────


def observed_refseq_counts(cell_dir: Path, mseq_glob: str, refseqs: list[str]) -> Counter:
    """Per-reference observed read counts from a cell's mseq files (dbhit == full
    DB header). Only counts hits to references in our amplifiable set."""
    keep = set(refseqs)
    counts: Counter = Counter()
    for path in cell_dir.glob(mseq_glob):
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2 or not fields[1]:
                    continue
                dbhit = fields[1]
                if dbhit in keep:
                    counts[dbhit] += 1
    return counts


def _iter_fastq(path: Path):
    """Yield uppercased read sequences from a (optionally gzipped) FASTQ."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        while True:
            if not fh.readline():          # header (or EOF)
                break
            seq = fh.readline().strip().upper()
            fh.readline()                  # '+'
            fh.readline()                  # qual
            if seq:
                yield seq


def _score_bin(score: float, bin_edges: np.ndarray) -> int:
    """Bin index of ``score`` on ``bin_edges``, clamped to the valid range."""
    k = int(np.searchsorted(bin_edges, score, side="right")) - 1
    return min(max(k, 0), bin_edges.shape[0] - 2)


def observed_score_histograms(cell_dir: Path, read_glob: str, v4_seqs: list[str],
                              emits: list[np.ndarray], bin_edges: np.ndarray,
                              gap_penalty: float, max_reads: int, rng) -> tuple[np.ndarray, int]:
    """Per-reference score histograms ``H[S,K]`` and read count ``N`` for one cell.

    For each read, score it against every reference (best of both orientations, decided
    once per read against ref 0 — strand is global across the same-strand 16S refs) and
    tally the score into that reference's histogram on the shared ``bin_edges``. Reads are
    subsampled to ``max_reads`` (a distribution estimate needs far fewer than the full
    depth, and the scorer is O(reads x refs)). ``ponytail:`` per-read orientation detected
    once via ref 0; large community DBs would also want a top-K ref prefilter.
    """
    S, K = len(v4_seqs), bin_edges.shape[0] - 1
    c2i = _char_to_idx()
    reads = [seq for path in sorted(cell_dir.glob(read_glob)) for seq in _iter_fastq(path)]
    if max_reads and len(reads) > max_reads:
        reads = [reads[i] for i in rng.choice(len(reads), size=max_reads, replace=False)]

    H = np.zeros((S, K), dtype=np.float64)
    for seq in reads:
        rev = revcomp(seq)
        fi = c2i[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
        ri = c2i[np.frombuffer(rev.encode("ascii", "replace"), dtype=np.uint8)]
        if _oriented_score(rev, ri, emits[0], v4_seqs[0], gap_penalty) > \
                _oriented_score(seq, fi, emits[0], v4_seqs[0], gap_penalty):
            rd, ridx = rev, ri
        else:
            rd, ridx = seq, fi
        for j in range(S):
            sc = _oriented_score(rd, ridx, emits[j], v4_seqs[j], gap_penalty)
            H[j, _score_bin(sc, bin_edges)] += 1.0
    return H, len(reads)


# ── Pyro inference ────────────────────────────────────────────────────────────


def composition_model(M, T, alpha, N, y_obs=None, likelihood="dirichlet_multinomial",
                      use_mismapping=True, s_sigma=0.3, od_loc=1.1, od_scale=1.0,
                      comp_scale=None):
    """Generative model of the observed per-reference counts.

    ``likelihood="score_hist"`` is the per-reference score-distribution model: ``M`` carries
    the simulated component tensor ``D[S,S,K]`` and ``y_obs`` the observed score histograms
    ``H[S,K]``. Each reference's histogram is Multinomial over ``K`` score bins with
    probabilities ``p[j] = Σ_a r_true[a]·D[a,j]``. Because every read feeds all ``S``
    histograms, these are a *composite* likelihood — downweighted by ``1/comp_scale``
    (default ``S``) so the posterior isn't spuriously tight.

    ``theta`` is the true **read-space** genome composition; ``T`` rows sum to 1
    (within-genome copy distribution) so ``r_true = theta@T`` has genome-marginal
    ``theta``. ``theta ~ Dirichlet`` -> ``r_true = theta@T``. When ``use_mismapping`` a sampled
    scalar ``s`` scales the mis-mapping matrix (``M_eff = (1-s)I + s*M``,
    ``s~LogNormal`` centred at 1) which then acts on ``r_true`` to give
    ``r_obs = r_true@M_eff``; otherwise ``r_obs = r_true`` (no correction — the
    baseline control). Counts follow a Dirichlet-Multinomial whose concentration is
    ``conc_frac*N*r_obs`` — overdispersion parameterised as a *fraction of N* so the
    prior is sample-size-invariant and centred near the multinomial limit rather than
    discarding read-count precision — or plain Multinomial.
    """
    import pyro
    import pyro.distributions as dist
    import torch

    dt = M.dtype
    G, Sdim = T.shape[0], M.shape[0]
    theta = pyro.sample("theta", dist.Dirichlet(alpha * torch.ones(G, dtype=dt)))

    r_true = theta @ T
    r_true = r_true / r_true.sum()

    if likelihood == "score_hist":
        import pyro.poutine as poutine

        D, H_obs = M, y_obs                       # M := D[S,S,K], y_obs := H[S,K]
        S = D.shape[0]
        p = torch.einsum("a,ajk->jk", r_true, D)  # [S,K] predicted per-ref bin probs
        p = p / p.sum(-1, keepdim=True)
        total = int(round(float(H_obs[0].sum()))) if H_obs is not None else int(N)
        c = float(comp_scale) if comp_scale else float(S)   # composite downweight
        with poutine.scale(scale=1.0 / c), pyro.plate("refs", S):
            pyro.sample("H", dist.Multinomial(total_count=total, probs=p), obs=H_obs)
        return

    if use_mismapping:
        # Mis-mapping scale applied to M before it acts on r_true.
        s = pyro.sample("s", dist.LogNormal(torch.tensor(0.0, dtype=dt),
                                            torch.tensor(s_sigma, dtype=dt)))
        M_eff = (1.0 - s) * torch.eye(Sdim, dtype=dt) + s * M
        # A large s can push a diagonal negative for very confusable refs; keep M_eff
        # a valid stochastic matrix. ponytail: clamp+renorm only bites in s's upper tail.
        M_eff = torch.clamp(M_eff, min=0.0)
        M_eff = M_eff / M_eff.sum(-1, keepdim=True)
        r_obs = r_true @ M_eff
        r_obs = r_obs / r_obs.sum()
    else:
        r_obs = r_true

    if y_obs is None:
        counts, total = None, int(N)
    else:
        counts = torch.round(y_obs * N).to(torch.long)
        total = int(counts.sum())   # obs must sum to total_count (rounding drifts ±1)
    if likelihood == "multinomial":
        pyro.sample("y", dist.Multinomial(total_count=total, probs=r_obs), obs=counts)
    else:
        # Overdispersion as a fraction of N (sample-size-invariant, centred near the
        # multinomial limit): concentration = conc_frac * N * r_obs.
        conc_frac = pyro.sample("conc_frac", dist.LogNormal(torch.tensor(od_loc, dtype=dt),
                                                            torch.tensor(od_scale, dtype=dt)))
        conc = conc_frac * float(N) * r_obs + 1e-6
        pyro.sample("y", dist.DirichletMultinomial(concentration=conc, total_count=total),
                    obs=counts)


def _fit(mode, M, T, alpha, N, y_obs, likelihood, theta_init, args, desc=""):
    """Return (theta_samples[K,G] or None, theta_point[G], diagnostics dict, loss_trace)."""
    import pyro
    import torch
    from pyro.infer.autoguide.initialization import init_to_value

    use_mm = getattr(args, "use_mismapping", True) and likelihood != "score_hist"
    show = getattr(args, "progress", True)
    mk = {"y_obs": y_obs, "likelihood": likelihood, "use_mismapping": use_mm,
          "comp_scale": getattr(args, "comp_scale", None)}

    pyro.clear_param_store()
    init = {"theta": theta_init}
    if use_mm:
        init["s"] = torch.tensor(1.0, dtype=M.dtype)
    if likelihood == "dirichlet_multinomial":
        init["conc_frac"] = torch.tensor(3.0, dtype=M.dtype)  # ~exp(od_loc), prior median

    def _summ(samples):
        mean = samples.mean(0)
        lo = torch.quantile(samples, 0.05, dim=0)
        hi = torch.quantile(samples, 0.95, dim=0)
        return mean, lo, hi

    if mode == "nuts":
        from pyro.infer import MCMC, NUTS
        log.debug("%s NUTS: %d warmup + %d samples", desc, args.warmup, args.num_samples)
        kernel = NUTS(composition_model, init_strategy=init_to_value(values=init))
        mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.warmup,
                    disable_progbar=not show)
        mcmc.run(M, T, alpha, N, **mk)
        s = mcmc.get_samples()["theta"]
        diag = mcmc.diagnostics().get("theta", {})
        rhat = np.atleast_1d(np.asarray(diag.get("r_hat", np.nan))).astype(float)
        ess = np.atleast_1d(np.asarray(diag.get("n_eff", np.nan))).astype(float)
        mean, lo, hi = _summ(s)
        return s, mean, {"r_hat": rhat.tolist(), "n_eff": ess.tolist(),
                         "max_r_hat": float(np.nanmax(rhat))}, None

    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.infer.autoguide import AutoNormal, AutoDelta
    from pyro.optim import Adam

    guide_cls = AutoDelta if mode == "mle" else AutoNormal
    guide = guide_cls(composition_model, init_loc_fn=init_to_value(values=init))
    svi = SVI(composition_model, guide, Adam({"lr": args.lr}), Trace_ELBO())
    log.debug("%s %s: %d SVI steps (lr=%g)", desc, mode.upper(), args.steps, args.lr)
    losses = []
    bar = _progress(range(args.steps), total=args.steps, desc=f"{desc} {mode}",
                    enabled=show, unit="step")
    every = max(1, args.steps // 20)
    for step in bar:
        loss = float(svi.step(M, T, alpha, N, **mk))
        losses.append(loss)
        if step % every == 0:
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss:.1f}")
            log.debug("%s %s step %d/%d loss=%.3f", desc, mode, step, args.steps, loss)
    if mode == "mle":
        point = guide.median()["theta"].detach()
        return None, point, {"final_loss": losses[-1]}, losses
    pred = Predictive(composition_model, guide=guide, num_samples=args.num_samples,
                      return_sites=["theta"])
    s = pred(M, T, alpha, N, **{**mk, "y_obs": None})["theta"].squeeze()
    mean, lo, hi = _summ(s)
    return s, mean, {"final_loss": losses[-1]}, losses


def stage_infer(args) -> None:
    import torch

    _bp = _load_module("benchmark_preprocess", _HERE / "benchmark_preprocess.py")

    mm_dir = args.mismap_dir
    T_df = pd.read_csv(mm_dir / "translation_table.csv", index_col=0)
    genomes = list(T_df.index)

    score_hist = getattr(args, "score_hist", False)
    if score_hist:
        # Observed variable = per-reference score histograms; "M" positional := the
        # simulated component tensor D[S,S,K]. Need the model + ref amplicons to score reads.
        npz = np.load(mm_dir / "score_components.npz", allow_pickle=True)
        D, bin_edges = npz["D"], npz["bin_edges"]
        refseqs = [str(r) for r in npz["refseqs"]]
        model = _load_error_model(args.model_pt, args.use_vi)
        v4_map = dict(read_fasta(mm_dir / "v4_amplicons.fasta"))
        v4_seqs = [v4_map[r] for r in refseqs]
        emits = [emission_distribution(model, s) for s in v4_seqs]
        M = torch.tensor(D, dtype=torch.float64)
        likelihood, use_mm_eff = "score_hist", False
        rng = np.random.default_rng(args.seed)
    else:
        M_df = pd.read_csv(mm_dir / "mismapping_matrix.csv", index_col=0)
        refseqs = list(M_df.columns)
        M = torch.tensor(M_df.to_numpy(), dtype=torch.float64)
        likelihood, use_mm_eff = args.likelihood, args.use_mismapping

    T = torch.tensor(T_df.loc[genomes, refseqs].to_numpy(), dtype=torch.float64)
    refseq_genome = {r: genome_of_header(r) for r in refseqs}
    g_of_ref = np.array([genomes.index(refseq_genome[r]) for r in refseqs])

    pipeline_root = args.run_dir / args.pipeline_dir
    cells = _bp.find_cells(pipeline_root)
    if args.assay:
        # M is amplicon-specific: only apply it to cells from the matching assay.
        cells = [c for c in cells if c["assay"] == args.assay]
    if not cells:
        raise SystemExit(f"no cells under {pipeline_root}"
                         + (f" for assay {args.assay}" if args.assay else ""))

    log.info("inference: %d cell(s), %d genomes, %d references, mode=%s likelihood=%s%s",
             len(cells), len(genomes), len(refseqs), args.mode, likelihood,
             "" if use_mm_eff else " (no mis-mapping correction)")

    comp_rows: list[dict] = []
    diag_rows: list[dict] = []
    loss_rows: list[dict] = []
    for ci, cell in enumerate(cells, 1):
        cname = f"{cell['sample']}.{cell['assay']}/{cell['depth']}"
        if score_hist:
            H_obs, total = observed_score_histograms(
                cell["dir"], args.read_glob, v4_seqs, emits, bin_edges,
                args.gap_penalty, args.max_reads, rng)
            if total == 0:
                log.info("[%d/%d] %s: no reads, skipping", ci, len(cells), cname)
                continue
            obs_for_fit = torch.tensor(H_obs, dtype=torch.float64)
            # Per-ref proxy for init/baseline: upper-half score-mass (reads scoring well
            # against ref j) — a rough hard-assignment stand-in, not the inference itself.
            w = H_obs[:, H_obs.shape[1] // 2:].sum(1)
            ref_rel = w / w.sum() if w.sum() > 0 else np.full(len(refseqs), 1.0 / len(refseqs))
        else:
            counts = observed_refseq_counts(cell["dir"], args.mseq_glob, refseqs)
            total = sum(counts.values())
            if total == 0:
                log.info("[%d/%d] %s: no reads, skipping", ci, len(cells), cname)
                continue
            y = np.array([counts.get(r, 0) for r in refseqs], dtype=np.float64)
            ref_rel = y / total
            obs_for_fit = torch.tensor(ref_rel, dtype=torch.float64)
        log.info("[%d/%d] %s: %d reads", ci, len(cells), cname, total)

        # Observed genome composition (collapse ref->genome): baseline + init.
        theta_obs = np.zeros(len(genomes))
        np.add.at(theta_obs, g_of_ref, ref_rel)
        theta_init = torch.tensor(np.clip(theta_obs, 1e-4, None), dtype=torch.float64)
        theta_init = theta_init / theta_init.sum()

        truth = _bp.read_truth(next(iter(cell["dir"].glob(_bp.TRUTH_GLOB))))
        truth_map = truth.set_index("genome_id")["realized_rel_abundance"].to_dict()

        samples, point, diag, losses = _fit(
            args.mode, M, T, args.alpha, float(total), obs_for_fit, likelihood,
            theta_init, args, desc=f"[{ci}/{len(cells)}] {cname}")
        inferred = point.numpy()
        lo = hi = [np.nan] * len(genomes)
        if samples is not None:
            lo = torch.quantile(samples, 0.05, dim=0).numpy()
            hi = torch.quantile(samples, 0.95, dim=0).numpy()

        tag = {"sample": cell["sample"], "assay": cell["assay"], "depth": cell["depth"],
               "sweep_x": cell["sweep_x"]}
        for i, g in enumerate(genomes):
            comp_rows.append({**tag, "genome_id": g,
                              "observed_rel_abundance": theta_obs[i],
                              "inferred_mean": float(inferred[i]),
                              "inferred_lo": float(lo[i]), "inferred_hi": float(hi[i]),
                              "truth_rel_abundance": float(truth_map.get(g, 0.0))})
        diag_rows.append({**tag, "mode": args.mode, "likelihood": likelihood,
                          "use_mismapping": use_mm_eff, "n_reads": int(total),
                          **{k: json.dumps(v) for k, v in diag.items()}})
        if losses is not None:
            for step, loss in enumerate(losses):
                loss_rows.append({**tag, "step": step, "loss": loss})

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comp_rows).to_csv(out / "inferred_composition.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(out / "inference_diagnostics.csv", index=False)
    if loss_rows:
        pd.DataFrame(loss_rows).to_csv(out / "loss_trace.csv", index=False)
    print(f"infer[{args.mode}]: {len(cells)} cells, {len(genomes)} genomes -> {out}")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Demos ─────────────────────────────────────────────────────────────────────


def demo_mismapping() -> None:
    # Primer extraction: concretise the degenerate primers to ACGT (real refs are
    # ACGT), flank a payload, and recover it.
    def _concrete(p):  # first ACGT option per IUPAC code
        return "".join(sorted(_IUPAC[c])[0] for c in p)
    payload = "ACGTACGTAA" * 3
    seq = ("TTT" + _concrete(DEFAULT_FWD_PRIMER) + payload
           + _concrete(revcomp(DEFAULT_REV_PRIMER)) + "GGG")
    assert extract_v4(seq, DEFAULT_FWD_PRIMER, DEFAULT_REV_PRIMER, 2) == payload

    sys.path.insert(0, str(_SKIVER_LIB))
    from lib.error_application import ErrorModel
    from lib.encoding import NUM_ERROR_TYPES
    # Context-free model: match likely, subs less likely, no indels.
    logits = np.full((4, NUM_ERROR_TYPES), -6.0, dtype=np.float32)
    logits[:, 0] = 3.0                        # match
    for r in range(4):
        for b in range(4):
            if b != r:
                logits[r, 1 + b] = -1.0       # substitutions
    model = ErrorModel.from_spec("BaseContext(1)", {"logits": logits})

    a = "ACGTACGTACGTACGT"
    b = a[:8] + "T" + a[9:]                    # 1 diff
    c = "TGCATGCATGCATGCA"                      # very different
    emits = [emission_distribution(model, s) for s in (a, b, c)]
    M = build_mismapping_matrix(emits, [a, b, c], gap_penalty=4.0)
    assert np.allclose(M.sum(1), 1.0), M.sum(1)
    assert (np.diag(M) > 0.5).all(), M
    assert M[0, 1] > M[0, 2], M            # near-identical confused more than distant
    # identical refs -> ~half mass each
    Mid = build_mismapping_matrix([emits[0], emits[0]], [a, a], gap_penalty=4.0)
    assert abs(Mid[0, 1] - 0.5) < 1e-6, Mid

    # read_score: a read equal to ref a scores higher under a than under distant c.
    assert read_score(a, emits[0], a, 4.0) > read_score(a, emits[2], c, 4.0)

    # simulated score components: shapes, normalisation, and argmax confusion structure.
    D, edges, M_sim = simulate_score_components(model, [a, b, c], emits, gap_penalty=4.0,
                                                n_sim=120, n_bins=16, seed=0, progress=False)
    assert D.shape == (3, 3, 16) and edges.shape == (17,)
    assert np.allclose(D.sum(2), 1.0), D.sum(2)
    assert np.allclose(M_sim.sum(1), 1.0) and (np.diag(M_sim) > 0.5).all(), M_sim
    # a-reads' score distribution against near-identical b overlaps a's self-distribution
    # far more than against distant c (histogram intersection).
    ov = lambda p, q: float(np.minimum(p, q).sum())
    assert ov(D[0, 0], D[0, 1]) > ov(D[0, 0], D[0, 2]), D[0]
    print("demo mismapping: OK")


def demo_infer() -> None:
    import torch
    torch.manual_seed(0)

    # 3 genomes, genome 1 has two near-identical copies (refseqs 1 & 2) that map to
    # each other; genome 0 -> refseq 0, genome 2 -> refseq 3.
    genomes = ["g0", "g1", "g2"]
    T = torch.tensor([[1, 0, 0, 0],
                      [0, 1, 1, 0],
                      [0, 0, 0, 1]], dtype=torch.float64)
    M = torch.tensor([[0.95, 0.03, 0.01, 0.01],
                      [0.02, 0.55, 0.40, 0.03],   # copies 1<->2 heavily confused
                      [0.02, 0.42, 0.53, 0.03],
                      [0.01, 0.02, 0.02, 0.95]], dtype=torch.float64)
    theta_true = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    r_true = theta_true @ T
    r_obs = (r_true / r_true.sum()) @ M
    r_obs = r_obs / r_obs.sum()

    # Naive observed genome composition (collapse refseq->genome by membership).
    memb = T.argmax(0)
    obs_genome = torch.zeros(3, dtype=torch.float64)
    for s in range(4):
        obs_genome[memb[s]] += r_obs[s]

    args = argparse.Namespace(mode="vi", num_samples=200, warmup=0, steps=1200, lr=0.05,
                              progress=False)
    _, point, diag, losses = _fit("vi", M, T, 0.5, 5000.0, r_obs, "dirichlet_multinomial",
                                  obs_genome / obs_genome.sum(), args)
    inferred = point.numpy()
    err_naive = float(abs(obs_genome / obs_genome.sum() - theta_true).sum())
    err_inf = float(np.abs(inferred - theta_true.numpy()).sum())
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    assert err_inf < err_naive, (err_inf, err_naive)
    print(f"demo infer: L1 naive={err_naive:.3f} -> inferred={err_inf:.3f} OK")

    # ── score-histogram likelihood ───────────────────────────────────────────
    # Same 3-genome / 4-ref setup. Build synthetic per-source score components D[a,j,K]
    # (reads from a score high against confusable refs) and observed histograms H = N·(r_true@D).
    K, S = 10, 4
    close = np.array([[1.0, 0.1, 0.1, 0.1],
                      [0.1, 1.0, 0.9, 0.1],    # refs 1<->2 confusable copies
                      [0.1, 0.9, 1.0, 0.1],
                      [0.1, 0.1, 0.1, 1.0]])
    D = np.full((S, S, K), 1e-3)
    for aa in range(S):
        for jj in range(S):
            D[aa, jj, int(round(close[aa, jj] * (K - 1)))] += 1.0
    D /= D.sum(2, keepdims=True)
    r_true_np = (theta_true @ T).numpy(); r_true_np /= r_true_np.sum()
    p_hist = np.einsum("a,ajk->jk", r_true_np, D)          # [S,K]
    Nreads = 5000
    H_obs = np.round(p_hist / p_hist.sum(1, keepdims=True) * Nreads)

    w = H_obs[:, K // 2:].sum(1); ref_rel = w / w.sum()
    theta_obs2 = np.zeros(3)
    np.add.at(theta_obs2, T.argmax(0).numpy(), ref_rel)
    theta_obs2 /= theta_obs2.sum()

    args2 = argparse.Namespace(mode="vi", num_samples=200, warmup=0, steps=1500, lr=0.05,
                               progress=False, comp_scale=None)
    _, point2, _, losses2 = _fit("vi", torch.tensor(D), T, 0.5, float(Nreads),
                                 torch.tensor(H_obs), "score_hist",
                                 torch.tensor(np.clip(theta_obs2, 1e-4, None)) /
                                 np.clip(theta_obs2, 1e-4, None).sum(), args2)
    err_naive2 = float(np.abs(theta_obs2 - theta_true.numpy()).sum())
    err_inf2 = float(np.abs(point2.numpy() - theta_true.numpy()).sum())
    assert losses2[-1] < losses2[0], (losses2[0], losses2[-1])
    assert err_inf2 <= err_naive2 + 1e-3, (err_inf2, err_naive2)
    print(f"demo infer score_hist: L1 naive={err_naive2:.3f} -> inferred={err_inf2:.3f} OK")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _add_common(p) -> None:
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level step logging")
    p.add_argument("--no-progress", dest="progress", action="store_false",
                   help="disable tqdm progress bars (e.g. non-interactive runs)")
    p.set_defaults(progress=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mismapping", help="estimate M and T from the error model")
    m.add_argument("--db-fasta", type=Path)
    m.add_argument("--model-pt", type=Path)
    m.add_argument("--use-vi", action="store_true", help="use the model's VI posterior mean")
    m.add_argument("--fwd-primer", default=DEFAULT_FWD_PRIMER)
    m.add_argument("--rev-primer", default=DEFAULT_REV_PRIMER)
    m.add_argument("--primer-mismatches", type=int, default=3)
    m.add_argument("--method", choices=["likelihood", "simulate"], default="likelihood")
    m.add_argument("--gap-penalty", type=float, default=4.0,
                   help="LLR added per alignment gap column (likelihood method)")
    m.add_argument("--n-sim", type=int, default=500,
                   help="reads/ref for --method simulate and for --score-components")
    m.add_argument("--score-components", action="store_true",
                   help="also simulate per-reference score-distribution components D[S,S,K] "
                        "(-> score_components.npz) for `infer --score-hist`")
    m.add_argument("--sim-bins", type=int, default=40, help="score-histogram bins K")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("-o", "--output-dir", type=Path)
    m.add_argument("--demo", action="store_true")
    _add_common(m)

    inf = sub.add_parser("infer", help="Bayesian inference of true genome composition")
    inf.add_argument("--run-dir", type=Path)
    inf.add_argument("--pipeline-dir", default="results/subspecies_v4_sweep")
    inf.add_argument("--mismap-dir", type=Path, help="dir with mismapping_matrix.csv etc.")
    inf.add_argument("--mseq-glob", default="profiling/aap/*/taxonomy-summary/*/*.mseq.gz")
    inf.add_argument("--assay", default=None,
                     help="only infer cells from this assay (M is amplicon-specific), "
                          "e.g. 515-YF-806BR")
    inf.add_argument("--mode", choices=["nuts", "vi", "mle"], default="vi")
    # multinomial is the default: with deep amplicon data the read count is highly
    # informative, and Dirichlet-Multinomial overdispersion tends to absorb the
    # mis-mapping-correction signal (use it only when counts are genuinely overdispersed).
    inf.add_argument("--likelihood", choices=["multinomial", "dirichlet_multinomial"],
                     default="multinomial")
    inf.add_argument("--no-mismapping", dest="use_mismapping", action="store_false",
                     help="baseline control: infer without the mis-mapping correction "
                          "(r_obs = r_true, no s term)")
    inf.set_defaults(use_mismapping=True)
    # Score-histogram path: observed variable = per-reference alignment-score distributions,
    # fit against simulated components (score_components.npz from `mismapping --score-components`).
    inf.add_argument("--score-hist", action="store_true",
                     help="use per-reference score-distribution likelihood instead of mseq counts")
    inf.add_argument("--model-pt", type=Path, help="error model (required for --score-hist)")
    inf.add_argument("--use-vi", action="store_true", help="use the model's VI posterior mean")
    inf.add_argument("--read-glob", default="*_R1.fastq.gz",
                     help="per-cell read file glob (--score-hist)")
    inf.add_argument("--max-reads", type=int, default=20000,
                     help="subsample reads/cell for the score histograms (--score-hist)")
    inf.add_argument("--gap-penalty", type=float, default=4.0,
                     help="log-prob penalty per alignment gap when scoring reads (--score-hist)")
    inf.add_argument("--comp-scale", type=float, default=None,
                     help="composite-likelihood downweight c (default = #references)")
    inf.add_argument("--seed", type=int, default=0)
    inf.add_argument("--alpha", type=float, default=0.5, help="Dirichlet prior concentration")
    inf.add_argument("--num-samples", type=int, default=500)
    inf.add_argument("--warmup", type=int, default=500)
    inf.add_argument("--steps", type=int, default=3000, help="SVI steps (vi/mle)")
    inf.add_argument("--lr", type=float, default=0.02)
    inf.add_argument("-o", "--output-dir", type=Path)
    inf.add_argument("--demo", action="store_true")
    _add_common(inf)

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "mismapping":
        if args.demo:
            return demo_mismapping()
        for req in ("db_fasta", "model_pt", "output_dir"):
            if getattr(args, req) is None:
                ap.error(f"--{req.replace('_', '-')} required")
        stage_mismapping(args)
    else:
        if args.demo:
            return demo_infer()
        req = ["run_dir", "mismap_dir", "output_dir"]
        if args.score_hist:
            req.append("model_pt")
        for r in req:
            if getattr(args, r) is None:
                ap.error(f"--{r.replace('_', '-')} required")
        stage_infer(args)


if __name__ == "__main__":
    main()
