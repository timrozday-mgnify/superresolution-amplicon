#!/usr/bin/env python
"""Standalone per-sample genome-composition inference for superresolution-amplicon.

Both observed signal and mis-mapping come from **mapseq**: the observed per-reference
read counts are the top hits of the real reads, and the mis-mapping matrix ``M`` is
measured by mapping simulated reads (whose names carry their source reference) through
the same mapper. Feeding both into the ``dirichlet_multinomial`` likelihood inverts the
confusion between (near-)identical references rather than ignoring it. Fitting defaults
to VI (posterior mean): the mode/MLE collapses weak components (e.g. a low-abundance
subspecies) to exactly 0, the mean does not.

    infer_composition.py \
        --amplicon-dir AMPLICON_DIR --sim-mseq sim.mseq --obs-mseq obs.mseq \
        --sample-id S1 --mode vi -o out/
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
import sys
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)

log = logging.getLogger("infer_composition")


def _fit_args(a) -> SimpleNamespace:
    """The attribute bag _fit / composition_model read off ``args``."""
    return SimpleNamespace(
        mode=a.mode, lr=a.lr, steps=a.steps, num_samples=a.num_samples,
        warmup=a.warmup, progress=False, use_mismapping=not a.no_mismapping,
    )


def run(a) -> None:
    import torch

    T_df = pd.read_csv(a.amplicon_dir / "translation_table.csv", index_col=0)
    genomes = list(T_df.index)
    refseqs = list(T_df.columns)
    T = torch.tensor(T_df.to_numpy(), dtype=torch.float64)
    g_of_ref = np.array([genomes.index(si.genome_of_header(r)) for r in refseqs])

    # M measured from the simulated reads, mapped by the same mapper as the real reads.
    M_np = si.build_mismapping(a.sim_mseq, refseqs, a.min_identity)
    M = torch.tensor(M_np, dtype=torch.float64)

    counts = si.observed_refseq_counts(a.obs_mseq, refseqs, a.min_identity)
    obs = np.array([counts.get(r, 0) for r in refseqs], dtype=np.float64)
    total = int(obs.sum())
    if total == 0:
        raise SystemExit(f"no reads in {a.obs_mseq} hit a reference amplicon")
    ref_rel = obs / total

    log.info("sample %s: %d mapped reads, %d refs, %d genomes, mode=%s",
             a.sample_id, total, len(refseqs), len(genomes), a.mode)

    # Observed per-ref signal collapsed to genome-space -> observed composition (init + baseline).
    theta_obs = np.zeros(len(genomes))
    np.add.at(theta_obs, g_of_ref, ref_rel)
    theta_init = torch.tensor(np.clip(theta_obs, 1e-4, None), dtype=torch.float64)
    theta_init = theta_init / theta_init.sum()

    fa = _fit_args(a)
    samples, point, diag, losses = si._fit(
        a.mode, M, T, a.alpha, float(total), torch.tensor(ref_rel, dtype=torch.float64),
        "dirichlet_multinomial", theta_init, fa, desc=a.sample_id)
    inferred = point.numpy()
    lo = hi = [np.nan] * len(genomes)
    if samples is not None:
        lo = torch.quantile(samples, 0.05, dim=0).numpy()
        hi = torch.quantile(samples, 0.95, dim=0).numpy()

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(M_np, index=refseqs, columns=refseqs).to_csv(out / "mismapping_matrix.csv")
    comp = pd.DataFrame([{
        "sample": a.sample_id, "genome_id": g,
        "observed_rel_abundance": float(theta_obs[i]),
        "inferred_mean": float(inferred[i]),
        "inferred_lo": float(lo[i]), "inferred_hi": float(hi[i]),
    } for i, g in enumerate(genomes)])
    comp.to_csv(out / "inferred_composition.csv", index=False)
    pd.DataFrame([{
        "sample": a.sample_id, "mode": a.mode, "likelihood": "dirichlet_multinomial",
        "use_mismapping": not a.no_mismapping, "min_identity": a.min_identity,
        "n_reads": int(total), "mean_diagonal": float(np.diag(M_np).mean()),
        **{k: json.dumps(v) for k, v in diag.items()},
    }]).to_csv(out / "inference_diagnostics.csv", index=False)
    if losses is not None:
        pd.DataFrame([{"sample": a.sample_id, "step": s, "loss": l}
                      for s, l in enumerate(losses)]).to_csv(out / "loss_trace.csv", index=False)
    print(f"infer[{a.mode}]: sample {a.sample_id}, {len(genomes)} genomes -> {out}")


def demo() -> None:
    """Self-check: the mis-mapping matrix recovers a genome whose only amplicon is
    byte-identical to a copy of another genome (the B. uniformis identifiability case).

    Genomes [uni, strain2]; refs [uni|0(A), strain2|0(A), strain2|1(B)] — refs 0,1 are
    identical (A), ref 2 is distinct (B). ``M`` fully confuses the identical pair; ``T``
    encodes strain2's two equal copies. The observed counts collapse the identical pair
    (mapseq can't tell 0 from 1), yet fitting through ``M`` + ``T`` must invert the shared
    ref-2 anchor to back out uni's small excess — the naive observed massively
    overestimates uni.
    """
    import tempfile
    import torch

    M = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    T = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=torch.float64)  # uni / strain2
    theta_true = np.array([0.05, 0.95])
    r_true = torch.tensor(theta_true, dtype=torch.float64) @ T
    r_obs = (r_true @ M)                                   # observed per-ref fractions
    theta_obs = np.array([float(r_obs[0]), float(r_obs[1] + r_obs[2])])   # naive collapse
    assert theta_obs[0] > 0.2, theta_obs                  # naive overestimates uni (~0.26)
    theta_init = torch.tensor(theta_obs, dtype=torch.float64)

    fa = SimpleNamespace(mode="mle", lr=0.05, steps=3000, num_samples=1,
                         warmup=0, progress=False, use_mismapping=True)
    _, point, _, _ = si._fit("mle", M, T, 0.5, 5000.0, r_obs, "dirichlet_multinomial",
                             theta_init, fa, desc="demo")
    est = point.numpy()
    assert np.allclose(est.sum(), 1.0, atol=1e-3), est
    assert np.abs(est - theta_true).max() < 0.03, (est, theta_true, theta_obs)
    print("demo OK:", np.round(est, 4), "~", theta_true, "(naive uni was", round(theta_obs[0], 3), ")")

    # End-to-end on synthetic mseq files: the same case, driven through run().
    refs = ["uni|0|A", "strain2|0|A", "strain2|1|B"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        amp = td / "amp"
        amp.mkdir()
        pd.DataFrame(T.numpy(), index=["uni", "strain2"], columns=refs).to_csv(
            amp / "translation_table.csv")
        # Simulated reads: identical refs 0/1 split 50:50, ref 2 self-hits.
        with open(td / "sim.mseq", "w") as fh:
            for a_i, hits in enumerate([[0, 1], [0, 1], [2]]):
                for i in range(100):
                    fh.write(f"{refs[a_i]}:{i}\t{refs[hits[i % len(hits)]]}\t500\t0.99\n")
        with open(td / "obs.mseq", "w") as fh:
            for j, frac in enumerate(r_obs.numpy()):
                for i in range(int(round(frac * 10000))):
                    fh.write(f"read{j}_{i}\t{refs[j]}\t500\t0.99\n")
        args = argparse.Namespace(
            amplicon_dir=amp, sim_mseq=[td / "sim.mseq"], obs_mseq=[td / "obs.mseq"],
            min_identity=None, sample_id="demo", mode="mle", alpha=0.5, steps=3000,
            lr=0.05, num_samples=1, warmup=0, no_mismapping=False, seed=0,
            output_dir=td / "out")
        run(args)
        got = pd.read_csv(td / "out" / "inferred_composition.csv").set_index("genome_id")
        assert abs(got.loc["uni", "inferred_mean"] - 0.05) < 0.03, got
        assert got.loc["uni", "observed_rel_abundance"] > 0.2, got   # naive is wrong
        Mio = pd.read_csv(td / "out" / "mismapping_matrix.csv", index_col=0)
        assert np.allclose(Mio.loc[refs[0]].to_numpy(), [0.5, 0.5, 0.0]), Mio
    print("end-to-end mseq demo OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicon-dir", type=Path,
                    help="dir with translation_table.csv (from `subspecies_infer.py amplicons`)")
    ap.add_argument("--sim-mseq", type=Path, nargs="+",
                    help="mapseq output for the simulated reads (source ref in the read name)")
    ap.add_argument("--obs-mseq", type=Path, nargs="+",
                    help="mapseq output for this sample's real reads")
    ap.add_argument("--min-identity", type=float, default=None,
                    help="drop mapseq hits below this pairwise identity (off-target background)")
    ap.add_argument("--sample-id", default="sample")
    ap.add_argument("--mode", choices=["nuts", "vi", "mle"], default="vi")
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet prior concentration")
    ap.add_argument("--no-mismapping", action="store_true",
                    help="disable the mis-mapping correction (r_obs=r_true baseline)")
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=3000, help="SVI steps (vi/mle)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output-dir", type=Path)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if a.demo:
        return demo()
    for req in ("amplicon_dir", "sim_mseq", "obs_mseq", "output_dir"):
        if getattr(a, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (unless --demo)")
    run(a)


if __name__ == "__main__":
    main()
