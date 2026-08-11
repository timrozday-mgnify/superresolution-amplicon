#!/usr/bin/env python
"""Does the simulated error rate have to be *accurate*, or just big enough?

The mis-mapping matrix ``M`` exists to say which references get confused for which. If
that structure is insensitive to the error model, the whole skiver training subworkflow
can be replaced by ``--sim_error_model flat`` with a made-up rate.

The experiment, on the real 21-genome B. uniformis V4 reference set (two near-identical
B. uniformis strains — the case the pipeline exists for):

1. Extract the V4 amplicons (the pipeline's own ``stage_amplicons``).
2. Build one "observed" sample from a known composition under a **generator** error
   model, mapped with mapseq. Two generators (a trained skiver preset and flat 1%) so a
   conclusion can't be an artefact of the simulator agreeing with itself.
3. For each **candidate** error model — the trained preset and flat rates spanning 50x —
   simulate the mis-mapping reads, map them with the same mapseq, build ``M``, and fit.
4. Report L1 error vs truth, the error on each strain of the confusable pair, and how far
   each ``M`` is from the generator's own ``M``.

Needs docker (the mapseq biocontainer) and torch/pyro. ~10 min on a laptop.

    python dev/error_rate_sensitivity.py [--work DIR] [--out CSV]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))
sys.path.insert(0, str(_REPO / "vendor" / "skiver" / "scripts"))
import simulate_amplicon_reads as sim  # noqa: E402
import subspecies_infer as si  # noqa: E402

DEFAULT_DB = Path.home() / ("Documents/synthetic-metagenomic-benchmark-pipeline_runs/"
                            "subspecies_Buniformis_amplicon_v4/mapseq_db/mapseq_db.fasta")
DEFAULT_MODEL = Path.home() / "Documents/skiver/context_error_models/additive_7_hq-illumina.pt"
MAPSEQ_IMAGE = "quay.io/biocontainers/mapseq:2.1.1b--hc47f52e_1"

# The confusable pair: two B. uniformis strains sharing 16S copies.
PAIR = ("bacteroides_uniformis", "BU_JCM13286_NT5170")
# Truth: the low-abundance member of the pair is the hard part.
TRUTH_PAIR = {PAIR[0]: 0.03, PAIR[1]: 0.17}

N_OBS_READS = 50_000
N_PER_REF = 300

# Candidate error models used to build M. Flat rates span 1500x around a plausible
# Illumina substitution rate; indels are set to a tenth of the substitution rate.
FLAT_RATES = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.15]


def mapseq(query: Path, fasta: Path, tax: Path, out: Path, threads: int = 4) -> None:
    """Run mapseq in its biocontainer over the shared work directory."""
    work = query.parent.resolve()
    cmd = ["docker", "run", "--rm", "--platform", "linux/amd64",
           "-v", f"{work}:/d", "-w", "/d", MAPSEQ_IMAGE,
           "mapseq", query.name, fasta.name, tax.name, "-nthreads", str(threads)]
    with open(out, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.DEVNULL, check=True)


def write_reads(path: Path, reads) -> None:
    with open(path, "w") as fh:
        for name, seq in reads:
            fh.write(f">{name}\n{seq}\n")


def make_generator(spec, rng):
    """Return ``mutate(seq) -> seq`` for a model spec ('trained' or a flat rate)."""
    if spec == "trained":
        from lib.error_application import ErrorModel, apply_batch
        model = ErrorModel.load(DEFAULT_MODEL, use_vi=False)

        def mutate_batch(recs):
            return [(r.name, r.sequence) for r in apply_batch(model, recs, rng,
                                                              emit_quality=False)]
    else:
        sub = float(spec)

        def mutate_batch(recs):
            return [(n, sim.apply_flat(s, rng, sub, sub / 10, sub / 10)) for n, s, _ in recs]
    return mutate_batch


def build_M(spec, refseqs, amps, work: Path, fasta: Path, tax: Path, seed: int,
            tag: str | None = None) -> np.ndarray:
    """Simulate N_PER_REF reads per reference under ``spec``, map, tally -> M."""
    rng = np.random.default_rng(seed)
    mutate = make_generator(spec, rng)
    tag = f"sim_{tag or spec}".replace(".", "p")
    reads = []
    for header, seq in zip(refseqs, amps):
        recs = [(f"{header}:{i}", seq, True) for i in range(N_PER_REF)]
        reads.extend(mutate(recs))
    write_reads(work / f"{tag}.fasta", reads)
    mapseq(work / f"{tag}.fasta", fasta, tax, work / f"{tag}.mseq")
    return si.build_mismapping([work / f"{tag}.mseq"], refseqs)


def build_observed(spec, refseqs, amps, r_true, work: Path, fasta: Path, tax: Path,
                   seed: int) -> np.ndarray:
    """Sample N_OBS_READS reads from the true composition under ``spec``, map, count."""
    rng = np.random.default_rng(seed)
    mutate = make_generator(spec, rng)
    tag = f"obs_{spec}".replace(".", "p")
    src = rng.choice(len(refseqs), size=N_OBS_READS, p=r_true)
    recs = [(f"obs{i}", amps[j], True) for i, j in enumerate(src)]
    write_reads(work / f"{tag}.fasta", mutate(recs))
    mapseq(work / f"{tag}.fasta", fasta, tax, work / f"{tag}.mseq")
    counts = si.observed_refseq_counts([work / f"{tag}.mseq"], refseqs)
    obs = np.array([counts.get(r, 0) for r in refseqs], dtype=float)
    return obs / obs.sum()


def fit(M_np, T, genomes, ref_rel, total, use_mismapping=True):
    import torch
    g_of_ref = np.array([genomes.index(si.genome_of_header(r)) for r in T.columns])
    theta_obs = np.zeros(len(genomes))
    np.add.at(theta_obs, g_of_ref, ref_rel)
    init = torch.tensor(np.clip(theta_obs, 1e-4, None), dtype=torch.float64)
    # Gate off: this experiment is about M, and all 21 genomes are present in its truth,
    # so a presence gate would only add noise. See dev/presence_prior_sweep.py.
    args = SimpleNamespace(mode="vi", lr=0.02, steps=1500, num_samples=200, warmup=0,
                           progress=False, use_mismapping=use_mismapping,
                           use_presence=False, seed=0)
    _, point, _, _ = si._fit(
        "vi", torch.tensor(M_np, dtype=torch.float64),
        torch.tensor(T.to_numpy(), dtype=torch.float64), 0.5, float(total),
        torch.tensor(ref_rel, dtype=torch.float64), "dirichlet_multinomial",
        init / init.sum(), args)
    return theta_obs, point.numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="reference 16S fasta")
    ap.add_argument("--work", type=Path, default=_REPO / "dev" / "_sensitivity_work")
    ap.add_argument("--out", type=Path, default=_REPO / "dev" / "error_rate_sensitivity.csv")
    ap.add_argument("--keep", action="store_true", help="keep the work dir")
    a = ap.parse_args()

    work = a.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1. In-silico PCR -> the mapseq reference set (the pipeline's own stage).
    si.stage_amplicons(SimpleNamespace(
        db_fasta=a.db, fwd_primer=si.DEFAULT_FWD_PRIMER, rev_primer=si.DEFAULT_REV_PRIMER,
        primer_mismatches=3, output_dir=work))
    fasta, tax = work / "amplicons.fasta", work / "amplicons.tax"
    recs = si.read_fasta(fasta)
    refseqs = [h for h, _ in recs]
    amps = [s for _, s in recs]
    T = pd.read_csv(work / "translation_table.csv", index_col=0)
    genomes = list(T.index)
    assert list(T.columns) == refseqs

    # 2. Truth: the confusable pair at 3% / 17%, the rest split evenly.
    theta_true = np.full(len(genomes), (1.0 - sum(TRUTH_PAIR.values())) / (len(genomes) - 2))
    for g, v in TRUTH_PAIR.items():
        theta_true[genomes.index(g)] = v
    r_true = theta_true @ T.to_numpy()
    r_true = r_true / r_true.sum()

    # Cluster once so every mapseq call reuses it (and is therefore identical).
    mapseq(fasta, fasta, tax, work / "cluster.mseq")

    candidates = ["trained"] + [str(r) for r in FLAT_RATES]
    print(f"\n{len(refseqs)} amplicons / {len(genomes)} genomes; "
          f"{len(candidates)} candidate error models\n")

    # 3. M per candidate (generator-independent), observed per generator.
    Ms = {}
    for c in candidates:
        Ms[c] = build_M(c, refseqs, amps, work, fasta, tax, seed=1)
        print(f"  M[{c}]: mean diagonal {np.diag(Ms[c]).mean():.4f}")
    # Noise floor: the same model re-simulated with a different seed. Any cross-model
    # m_dist at this level means the models are statistically indistinguishable.
    Ms["trained (reseed)"] = build_M("trained", refseqs, amps, work, fasta, tax,
                                     seed=99, tag="trained_reseed")
    candidates.append("trained (reseed)")

    rows = []
    for gen in ["trained", "0.01"]:
        ref_rel = build_observed(gen, refseqs, amps, r_true, work, fasta, tax, seed=2)
        theta_obs, _ = fit(np.eye(len(refseqs)), T, genomes, ref_rel, N_OBS_READS,
                           use_mismapping=False)
        rows.append(dict(generator=gen, candidate="(none: naive observed)",
                         l1=float(np.abs(theta_obs - theta_true).sum()),
                         **{f"err_{g}": float(theta_obs[genomes.index(g)] - theta_true[genomes.index(g)])
                            for g in PAIR},
                         m_dist=np.nan, mean_diag=np.nan))
        for c in candidates:
            _, est = fit(Ms[c], T, genomes, ref_rel, N_OBS_READS)
            rows.append(dict(
                generator=gen, candidate=c,
                l1=float(np.abs(est - theta_true).sum()),
                **{f"err_{g}": float(est[genomes.index(g)] - theta_true[genomes.index(g)])
                   for g in PAIR},
                # Distance from the trained model's M. The "trained (reseed)" row is the
                # simulation-noise floor: anything at that level is indistinguishable.
                m_dist=float(np.linalg.norm(Ms[c] - Ms["trained"])),
                mean_diag=float(np.diag(Ms[c]).mean())))
            print(f"  fit gen={gen:8s} cand={c:8s} L1={rows[-1]['l1']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 160)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {a.out}")
    if not a.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
