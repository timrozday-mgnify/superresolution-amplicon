#!/usr/bin/env python
"""Error-model-driven sub-species composition inference for amplicon runs.

Library + the ``amplicons`` CLI stage. The pipeline is:

``amplicons`` (this CLI)
    In-silico PCR over the reference DB: extract each entry's amplicon between the
    primers and write the mapseq reference set (``amplicons.fasta`` + ``amplicons.tax``)
    plus the deterministic genome->reference translation table ``T``.

*simulate -> map* (``simulate_amplicon_reads.py`` + mapseq, driven by Nextflow)
    Reads are sampled from each reference amplicon under a sequencing error model and
    mapped with **the same mapper used for the real reads**, so the mis-mapping matrix
    ``M[a,j] = P(a read truly from a is assigned to j)`` is measured, not modelled.

``infer_composition.py``
    Bayesian inference (Pyro: NUTS / VI / MLE) on the latent *true genome* composition
    ``theta``: ``theta ~ Dirichlet(alpha)`` -> ``r_true = theta @ T`` ->
    ``r_obs = r_true @ M`` -> Dirichlet-Multinomial on the observed per-reference mapseq
    counts. Started from the observed composition.

Note: inference uses **Pyro**, not NumPyro, because the skiver error model
(``lib.error_application``) already imports pyro, so a NumPyro+JAX stack would be
redundant weight in the same env.

Run:
    subspecies_infer.py amplicons --db-fasta db.fasta -o out/
    subspecies_infer.py amplicons --demo   # self-check
"""
from __future__ import annotations

import argparse
import gzip
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("subspecies_infer")

# 515-YF / 806BR (V4). Overridable on the CLI.
DEFAULT_FWD_PRIMER = "GTGYCAGCMGCCGCGGTAA"
DEFAULT_REV_PRIMER = "GGACTACNVGGGTWTCTAAT"

# Presence/absence gate. The prior probability is the sparsity regulariser (small =>
# a genome must earn its place); the temperature controls how hard the Concrete
# relaxation is. Both chosen by Jaccard against a subset-present truth on the 21-genome
# B. uniformis V4 set — see dev/presence_prior_sweep.md. Temperature turned out to matter
# more than the prior: below ~1 the gate barely moves and no prior can sparsify it. Both
# defaults sit inside the plateau of settings scoring Jaccard 1.0 rather than on its edge.
DEFAULT_PRESENCE_PRIOR = 0.01
DEFAULT_PRESENCE_TEMP = 1.0
# A gate above this is called "present" (Jaccard / n_present / the hard MLE gate).
PRESENCE_THRESHOLD = 0.5

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


def trim_read_primers(seq: str, fwd: str, rev: str, max_mismatch: int) -> str:
    """Return ``seq`` cut down to its primer-free amplicon, trying both orientations.

    Observed amplicon reads carry the primers (and are longer than the primer-trimmed
    reference amplicons they are mapped against); putting them in the same coordinate
    space as ``extract_v4``'s references keeps the observed and simulated reads
    comparable. If no primer pair is found in either orientation (reads already trimmed,
    or an off-target read), the read is returned unchanged."""
    amp = extract_v4(seq, fwd, rev, max_mismatch)
    if amp is None:
        amp = extract_v4(revcomp(seq), fwd, rev, max_mismatch)
    return amp if amp else seq


def _progress(iterable, *, total=None, desc="", enabled=True, unit="it", leave=False):
    """tqdm progress bar when enabled, else the plain iterable (tqdm ships with pyro)."""
    if not enabled:
        return iterable
    from tqdm.auto import tqdm
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave)


# ── mapseq reference set (in-silico PCR) ─────────────────────────────────────

# mapseq refuses a custom reference without a taxonomy sidecar, but only field 2 of the
# .mseq (the top-hit reference id) is used downstream, so the taxonomy is a formality:
# one synthetic 3-level lineage per reference. Cutoffs/levels format per the MAPseq
# README (ported from synthetic-metagenomic-benchmark-pipeline/bin/build_mapseq_refs.py).
_TAX_HEADER = ("#cutoff: 0.00:0.08 0.80:0.35 0.95:0.05\n"
               "#name: refdb\n"
               "#levels: Kingdom Genome Copy\n")


def write_mapseq_tax(headers: list[str], path: Path) -> None:
    """Write the mapseq ``.tax`` sidecar: ``<header>\\tBacteria;<genome>;<header>``."""
    with open(path, "w") as fh:
        fh.write(_TAX_HEADER)
        for h in headers:
            fh.write(f"{h}\tBacteria;{genome_of_header(h)};{h}\n")


def stage_amplicons(args) -> None:
    """In-silico PCR over the DB -> mapseq reference set + translation table ``T``."""
    log.info("loading DB fasta %s", args.db_fasta)
    records = read_fasta(args.db_fasta)
    log.info("extracting amplicons (primers %s / %s, <=%d mismatches) from %d entries",
             args.fwd_primer, args.rev_primer, args.primer_mismatches, len(records))

    refseqs: list[str] = []          # DB entry headers with a valid amplicon
    amplicons: list[str] = []
    genomes_of: list[str] = []
    idx_rows: list[dict] = []
    for header, seq in records:
        amp = extract_v4(seq, args.fwd_primer, args.rev_primer, args.primer_mismatches)
        amplifiable = amp is not None
        idx_rows.append({"refseq": header, "genome": genome_of_header(header),
                         "amplicon_len": len(amp) if amp else 0, "amplifiable": amplifiable})
        if amplifiable:
            refseqs.append(header)
            amplicons.append(amp)
            genomes_of.append(genome_of_header(header))
    if not refseqs:
        raise SystemExit("no reference produced an amplicon; check primers / DB fasta")
    log.info("%d/%d entries amplifiable", len(refseqs), len(records))

    genomes = sorted(set(genomes_of))
    T = np.zeros((len(genomes), len(refseqs)), dtype=np.float64)
    g_idx = {g: i for i, g in enumerate(genomes)}
    for j, g in enumerate(genomes_of):
        T[g_idx[g], j] = 1.0
    # Rows sum to 1: the within-genome distribution of a genome's amplicon reads over
    # its 16S copies (uniform). This makes the latent theta the true *read-space* genome
    # composition (r_true genome-marginal = theta), directly comparable to the read-space
    # realized truth and observed profile — no spurious copy-number reweighting.
    # NB: copy-count (unnormalized) rows were tested on the B.uniformis V4 sweep and
    # OVERCORRECT — uni flips from ~18% under to ~150% over; the mismapping inversion
    # already recovers genome space, so the residual offset is leakage, not copy number.
    T = T / T.sum(axis=1, keepdims=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(T, index=genomes, columns=refseqs).to_csv(out / "translation_table.csv")
    pd.DataFrame(idx_rows).to_csv(out / "refseq_index.csv", index=False)
    with open(out / "amplicons.fasta", "w") as fh:
        for h, s in zip(refseqs, amplicons):
            fh.write(f">{h}\n{s}\n")
    write_mapseq_tax(refseqs, out / "amplicons.tax")

    print(f"amplicons: {len(refseqs)} amplifiable refs / {len(records)} DB entries, "
          f"{len(genomes)} genomes -> {out}")


# ── mapseq (.mseq) parsing ────────────────────────────────────────────────────

# .mseq fields (MAPseq README): 1 query id, 2 top-hit reference id, 3 alignment score,
# 4 pairwise identity, ... Comment lines start with '#'.
_MSEQ_QUERY, _MSEQ_HIT, _MSEQ_IDENTITY = 0, 1, 3


def iter_mseq(path: Path, min_identity: float | None = None):
    """Yield ``(query_id, hit_id)`` per classified read, skipping unhit/low-identity rows."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) <= _MSEQ_HIT or not f[_MSEQ_HIT]:
                continue
            if min_identity is not None:
                try:
                    if float(f[_MSEQ_IDENTITY]) < min_identity:
                        continue
                except (IndexError, ValueError):
                    continue
            yield f[_MSEQ_QUERY], f[_MSEQ_HIT]


def observed_refseq_counts(mseq_paths, refseqs: list[str],
                           min_identity: float | None = None) -> Counter:
    """Per-reference observed read counts from mapseq output (field 2 == DB header).

    Only hits to references in our amplifiable set are counted; anything else is
    off-target background."""
    keep = set(refseqs)
    counts: Counter = Counter()
    for path in mseq_paths:
        for _, hit in iter_mseq(path, min_identity):
            if hit in keep:
                counts[hit] += 1
    return counts


def build_mismapping(sim_mseq_paths, refseqs: list[str],
                     min_identity: float | None = None) -> np.ndarray:
    """``M[a,j]`` = fraction of reads simulated from reference ``a`` that mapseq assigns
    to reference ``j``; row-stochastic.

    The simulator names each read ``<source header>:<i>``, so the true source survives
    mapping (same trick as superresolution-shotgun's chunk ids). A reference whose reads
    all failed to map gets an identity row, i.e. ``r_true`` passes through uncorrected.
    """
    idx = {r: i for i, r in enumerate(refseqs)}
    n = len(refseqs)
    counts = np.zeros((n, n), dtype=np.float64)
    unknown = 0
    for path in sim_mseq_paths:
        for query, hit in iter_mseq(path, min_identity):
            a = idx.get(query.rsplit(":", 1)[0])
            j = idx.get(hit)
            if a is None or j is None:
                unknown += 1
                continue
            counts[a, j] += 1.0
    if unknown:
        log.warning("%d simulated hits with an unrecognised source/target reference", unknown)
    tot = counts.sum(axis=1)
    empty = tot == 0
    if empty.any():
        log.warning("%d reference(s) had no mapped simulated read; using an identity row",
                    int(empty.sum()))
        counts[empty] = np.eye(n)[empty]
        tot = counts.sum(axis=1)
    return counts / tot[:, None]


# ── Pyro inference ────────────────────────────────────────────────────────────


def composition_model(M, T, alpha, N, y_obs=None, likelihood="dirichlet_multinomial",
                      use_mismapping=True, s_sigma=0.3, od_loc=1.1, od_scale=1.0,
                      use_presence=True, presence_prior=DEFAULT_PRESENCE_PRIOR,
                      presence_temp=DEFAULT_PRESENCE_TEMP):
    """Generative model of the observed per-reference counts.

    ``theta`` is the true **read-space** genome composition; ``T`` rows sum to 1
    (within-genome copy distribution) so ``r_true = theta@T`` has genome-marginal
    ``theta``. ``theta ~ Dirichlet`` -> ``r_true = theta@T``. When ``use_presence`` a
    per-genome Bernoulli gate ``z`` multiplies ``theta`` before renormalising; its
    posterior probability is the reported presence confidence and an absent genome's
    abundance is shrunk towards zero by it. When ``use_mismapping`` a
    sampled scalar ``s`` scales the mis-mapping matrix (``M_eff = (1-s)I + s*M``,
    ``s~LogNormal`` centred at 1) which then acts on ``r_true`` to give
    ``r_obs = r_true@M_eff``; otherwise ``r_obs = r_true`` (no correction — the
    baseline control). Counts follow a Dirichlet-Multinomial whose concentration is
    ``conc_frac*N*r_obs`` — overdispersion parameterised as a *fraction of N* so the
    prior is sample-size-invariant and centred near the multinomial limit rather than
    discarding read-count precision — or plain Multinomial.

    The gated composition is exposed as the deterministic site ``theta_eff`` (equal to
    ``theta`` when the gate is off), which is what downstream code reads.
    """
    import pyro
    import pyro.distributions as dist
    import torch

    dt = M.dtype
    G, Sdim = T.shape[0], M.shape[0]
    theta = pyro.sample("theta", dist.Dirichlet(alpha * torch.ones(G, dtype=dt)))

    if use_presence:
        # Bernoulli presence per genome, relaxed to a Concrete distribution so SVI can
        # differentiate through it. ``presence_prior`` << 0.5 is the regulariser: the KL
        # pulls every gate towards 0 and the likelihood has to pay for each genome it
        # keeps.
        #
        # NB: the straight-through variant (RelaxedBernoulliStraightThrough) was tried
        # first, for hard 0/1 gates. It does not work here: the hard value loses the soft
        # sample that carries the gradient, so both model and guide score the gate at the
        # clamped boundary where the Concrete density is ~470 nats and almost insensitive
        # to the logits. The gates never leave their initialisation and the prior is inert
        # (identical fits at 0.05 and 1e-12). See dev/presence_prior_sweep.md.
        z = pyro.sample("z", dist.RelaxedBernoulli(
            temperature=torch.tensor(presence_temp, dtype=dt),
            probs=presence_prior * torch.ones(G, dtype=dt)).to_event(1))
        theta = z * theta
        theta = theta / theta.sum()
    theta = pyro.deterministic("theta_eff", theta)

    r_true = theta @ T
    r_true = r_true / r_true.sum()

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


def _presence_guide(G, temp, dt, init_logit=2.0):
    """Mean-field guide for the presence gate: a learnable per-genome Bernoulli.

    ``z`` gets its own site rather than an autoguide entry so the variational family
    matches the prior's: an AutoNormal over the Concrete's unit-interval support would
    fit a sigmoid-normal instead, and ``sigmoid(z_logits)`` would no longer be a
    Bernoulli probability. As written the learned ``sigmoid(z_logits)`` *is* the
    posterior presence probability we report. Initialised at ~0.88 ("start with
    everything present") so the prior has to earn the zeros off the data.
    """
    import pyro
    import pyro.distributions as dist
    import torch

    def guide(*a, **kw):
        logits = pyro.param("z_logits", lambda: torch.full((G,), init_logit, dtype=dt))
        pyro.sample("z", dist.RelaxedBernoulli(
            temperature=torch.tensor(temp, dtype=dt), logits=logits).to_event(1))
    return guide


def _presence_probs(G):
    """The fitted presence probabilities, or all-ones when the gate is off."""
    import pyro
    import torch
    if "z_logits" not in pyro.get_param_store():
        return torch.ones(G, dtype=torch.float64)
    return torch.sigmoid(pyro.param("z_logits").detach())


def _fit(mode, M, T, alpha, N, y_obs, likelihood, theta_init, args, desc=""):
    """Return (theta_samples[K,G] or None, theta_point[G], diagnostics dict, loss_trace).

    ``theta`` here is the *gated* composition (the ``theta_eff`` site). When the presence
    gate is on the diagnostics carry ``presence_prob`` (per-genome posterior probability
    of presence) and ``n_present``.
    """
    import pyro
    import torch
    from pyro.infer.autoguide.initialization import init_to_value

    use_mm = getattr(args, "use_mismapping", True)
    use_presence = getattr(args, "use_presence", True)
    p_prior = getattr(args, "presence_prior", DEFAULT_PRESENCE_PRIOR)
    p_temp = getattr(args, "presence_temp", DEFAULT_PRESENCE_TEMP)
    show = getattr(args, "progress", True)
    G = T.shape[0]
    mk = {"y_obs": y_obs, "likelihood": likelihood, "use_mismapping": use_mm,
          "use_presence": use_presence, "presence_prior": p_prior, "presence_temp": p_temp}

    pyro.set_rng_seed(int(getattr(args, "seed", 0) or 0))
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

    def _presence_diag():
        if not use_presence:
            return {}
        p = _presence_probs(G)
        return {"presence_prob": p.tolist(),
                "n_present": int((p >= PRESENCE_THRESHOLD).sum())}

    if mode == "nuts":
        if use_presence:
            # A Concrete density is stiff for HMC and not worth the step-size tuning.
            raise SystemExit("presence gate is not supported in nuts mode; "
                             "use --mode vi or --no-presence")
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

    from pyro import poutine
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.infer.autoguide import AutoNormal, AutoDelta, AutoGuideList
    from pyro.optim import Adam

    guide_cls = AutoDelta if mode == "mle" else AutoNormal
    guide = AutoGuideList(composition_model)
    guide.append(guide_cls(poutine.block(composition_model, hide=["z"]),
                           init_loc_fn=init_to_value(values=init)))
    if use_presence:
        guide.append(_presence_guide(G, p_temp, M.dtype))
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
    diag = {"final_loss": losses[-1], **_presence_diag()}
    if mode == "mle":
        # AutoDelta.median() has no deterministic sites, so apply the hard gate here.
        point = guide.median()["theta"].detach()
        if use_presence:
            point = point * (_presence_probs(G) >= PRESENCE_THRESHOLD).to(point.dtype)
            point = point / point.sum().clamp(min=1e-12)
        return None, point, diag, losses
    pred = Predictive(composition_model, guide=guide, num_samples=args.num_samples,
                      return_sites=["theta_eff"])
    s = pred(M, T, alpha, N, **{**mk, "y_obs": None})["theta_eff"].squeeze()
    mean, lo, hi = _summ(s)
    return s, mean, diag, losses


# ── Presence/absence scoring ──────────────────────────────────────────────────


def presence_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Set-overlap metrics for two boolean presence vectors.

    Jaccard is the headline number (intersection over union of the present sets); it is
    1.0 for a perfect call and, unlike accuracy, is not inflated by the many genomes both
    sides agree are absent. Precision/recall are reported alongside because the two
    failure modes differ in cost: a missed low-abundance sub-species (recall) is the one
    this pipeline exists to avoid.
    """
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = int((pred & truth).sum())
    union = int((pred | truth).sum())
    prec = tp / pred.sum() if pred.sum() else 0.0
    rec = tp / truth.sum() if truth.sum() else 0.0
    return {"jaccard": tp / union if union else 1.0,
            "precision": float(prec), "recall": float(rec),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "n_pred": int(pred.sum())}


# ── Demos ─────────────────────────────────────────────────────────────────────


def demo_amplicons() -> None:
    import tempfile

    # Primer extraction: concretise the degenerate primers to ACGT (real refs are
    # ACGT), flank a payload, and recover it.
    def _concrete(p):  # first ACGT option per IUPAC code
        return "".join(sorted(_IUPAC[c])[0] for c in p)
    payload = "ACGTACGTAA" * 3
    seq = ("TTT" + _concrete(DEFAULT_FWD_PRIMER) + payload
           + _concrete(revcomp(DEFAULT_REV_PRIMER)) + "GGG")
    assert extract_v4(seq, DEFAULT_FWD_PRIMER, DEFAULT_REV_PRIMER, 2) == payload
    # A read carrying the primers trims back to the same amplicon, either orientation.
    assert trim_read_primers(seq, DEFAULT_FWD_PRIMER, DEFAULT_REV_PRIMER, 2) == payload
    assert trim_read_primers(revcomp(seq), DEFAULT_FWD_PRIMER, DEFAULT_REV_PRIMER, 2) == payload

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        db = td / "db.fasta"
        headers = ["gA|0|x", "gA|1|y", "gB|0|z"]
        with open(db, "w") as fh:
            for i, h in enumerate(headers):
                fh.write(f">{h}\n{seq[:-3] + 'ACG'[i]}\n")   # all amplifiable
        args = argparse.Namespace(
            db_fasta=db, fwd_primer=DEFAULT_FWD_PRIMER, rev_primer=DEFAULT_REV_PRIMER,
            primer_mismatches=2, output_dir=td / "out")
        stage_amplicons(args)
        T = pd.read_csv(td / "out" / "translation_table.csv", index_col=0)
        assert list(T.index) == ["gA", "gB"] and list(T.columns) == headers, T
        assert np.allclose(T.to_numpy().sum(1), 1.0), T          # rows sum to 1
        assert np.allclose(T.loc["gA"].to_numpy(), [0.5, 0.5, 0.0]), T
        tax = (td / "out" / "amplicons.tax").read_text().splitlines()
        assert tax[0].startswith("#cutoff:") and len(tax) == 3 + len(headers), tax
        assert tax[3] == "gA|0|x\tBacteria;gA;gA|0|x", tax[3]
        assert dict(read_fasta(td / "out" / "amplicons.fasta"))["gA|0|x"] == payload

        # build_mismapping: source ref from the read name, target from mseq field 2.
        mseq = td / "sim.mseq"
        rows = ([("gA|0|x", "gA|0|x")] * 7 + [("gA|0|x", "gA|1|y")] * 3   # copies confused
                + [("gA|1|y", "gA|1|y")] * 10
                + [("gB|0|z", "gB|0|z")] * 10)
        with open(mseq, "w") as fh:
            fh.write("# comment\n")
            for i, (src, hit) in enumerate(rows):
                fh.write(f"{src}:{i}\t{hit}\t500\t0.99\n")
        M = build_mismapping([mseq], headers)
        assert np.allclose(M.sum(1), 1.0), M
        assert np.allclose(M[0], [0.7, 0.3, 0.0]), M
        assert np.allclose(np.diag(M)[1:], 1.0), M
        # identity threshold drops everything -> identity rows, not a crash
        assert np.allclose(build_mismapping([mseq], headers, min_identity=1.5), np.eye(3))
        counts = observed_refseq_counts([mseq], headers)
        assert counts["gA|1|y"] == 13 and counts["gB|0|z"] == 10, counts
    print("demo amplicons: OK")


def demo_infer() -> None:
    import torch
    torch.manual_seed(0)

    # 3 genomes, genome 1 has two near-identical copies (refseqs 1 & 2) that map to
    # each other; genome 0 -> refseq 0, genome 2 -> refseq 3.
    T = torch.tensor([[1, 0, 0, 0],
                      [0, .5, .5, 0],
                      [0, 0, 0, 1]], dtype=torch.float64)
    # Leakage is *across* genomes as well as between genome 1's two copies: confusion
    # within a genome cancels when the naive estimate collapses refseq->genome, so a
    # within-genome-only M leaves nothing for the inversion to beat.
    M = torch.tensor([[0.80, 0.12, 0.05, 0.03],
                      [0.08, 0.50, 0.38, 0.04],   # copies 1<->2 heavily confused
                      [0.06, 0.40, 0.50, 0.04],
                      [0.04, 0.06, 0.05, 0.85]], dtype=torch.float64)
    theta_true = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    r_true = theta_true @ T
    r_obs = (r_true / r_true.sum()) @ M
    r_obs = r_obs / r_obs.sum()

    # Naive observed genome composition (collapse refseq->genome by membership).
    memb = T.argmax(0)
    obs_genome = torch.zeros(3, dtype=torch.float64)
    for s in range(4):
        obs_genome[memb[s]] += r_obs[s]

    # Gate off: this case is testing the mis-mapping inversion in isolation.
    args = argparse.Namespace(mode="vi", num_samples=500, warmup=0, steps=2000, lr=0.05,
                              progress=False, use_presence=False, seed=0)
    _, point, diag, losses = _fit("vi", M, T, 0.5, 5000.0, r_obs, "dirichlet_multinomial",
                                  obs_genome / obs_genome.sum(), args)
    inferred = point.numpy()
    err_naive = float(abs(obs_genome / obs_genome.sum() - theta_true).sum())
    err_inf = float(np.abs(inferred - theta_true.numpy()).sum())
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    assert err_inf < err_naive, (err_inf, err_naive)
    assert "presence_prob" not in diag, diag
    print(f"demo infer: L1 naive={err_naive:.3f} -> inferred={err_inf:.3f} OK")


def demo_presence() -> None:
    """The gate must call a genome that contributes no reads absent, and keep the rest.

    4 genomes over 4 refseqs; genome 3 is truly absent. Mis-mapping leaks a little signal
    onto its reference, so the ungated fit gives it a non-zero abundance and has no way
    to say it isn't there. The gate has to (a) drop it from the present set and (b) shrink
    its abundance well below the ungated tail.
    """
    import torch

    T = torch.eye(4, dtype=torch.float64)
    M = torch.eye(4, dtype=torch.float64) * 0.80 + 0.05   # 5% leakage per off-diagonal
    theta_true = np.array([0.5, 0.3, 0.2, 0.0])
    r_obs = torch.tensor(theta_true, dtype=torch.float64) @ M
    r_obs = r_obs / r_obs.sum()

    def fit(**kw):
        args = argparse.Namespace(mode="vi", num_samples=500, warmup=0, steps=1500,
                                  lr=0.05, progress=False, seed=0,
                                  presence_prior=DEFAULT_PRESENCE_PRIOR,
                                  presence_temp=DEFAULT_PRESENCE_TEMP, **kw)
        return _fit("vi", M, T, 0.5, 20000.0, r_obs, "dirichlet_multinomial",
                    torch.tensor([.25] * 4, dtype=torch.float64), args)

    _, ungated, off_diag, _ = fit(use_presence=False)
    _, point, diag, _ = fit(use_presence=True)
    p = np.array(diag["presence_prob"])
    m = presence_metrics(p >= PRESENCE_THRESHOLD, theta_true > 0)
    assert "presence_prob" not in off_diag, off_diag
    assert m["jaccard"] == 1.0, (p, m)                 # absent genome dropped, rest kept
    assert diag["n_present"] == 3, diag
    # The reported mean still marginalises over presence uncertainty, so it is ~p*theta,
    # not a hard zero — but it must be a small fraction of the ungated tail.
    assert point.numpy()[3] < 0.25 * float(ungated.numpy()[3]), (point, ungated)
    print(f"demo presence: probs={np.round(p, 3)} jaccard={m['jaccard']:.2f} "
          f"absent abundance {ungated.numpy()[3]:.4f} -> {point.numpy()[3]:.4f} OK")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("amplicons", help="in-silico PCR -> mapseq reference set + T")
    m.add_argument("--db-fasta", type=Path)
    m.add_argument("--fwd-primer", default=DEFAULT_FWD_PRIMER)
    m.add_argument("--rev-primer", default=DEFAULT_REV_PRIMER)
    m.add_argument("--primer-mismatches", type=int, default=3)
    m.add_argument("-o", "--output-dir", type=Path)
    m.add_argument("--demo", action="store_true")
    m.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level step logging")

    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.demo:
        demo_amplicons()
        demo_infer()
        return demo_presence()
    for req in ("db_fasta", "output_dir"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} required")
    stage_amplicons(args)


if __name__ == "__main__":
    main()
