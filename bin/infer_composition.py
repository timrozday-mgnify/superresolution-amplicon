#!/usr/bin/env python
"""Standalone per-sample genome-composition inference for superresolution-amplicon.

Thin driver over ``subspecies_infer.py``: the benchmark's ``stage_infer`` is coupled
to a ``find_cells``/``read_truth`` directory harness, so instead of reusing it we call
the un-coupled math directly for a single sample. Observed signal is the score-hist
path (reads scored against the reference V4 amplicons with the trained error model —
no external aligner).

    infer_composition.py \
        --mismap-dir MISMAP_DIR --model-pt model.pt --reads-dir READS_DIR \
        --sample-id S1 --mode vi -o out/

ponytail: standalone replacement for subspecies_infer's stage_infer; if an aligner is
ever wired in, add an mseq-count path here (see observed_refseq_counts).
"""
from __future__ import annotations

import argparse
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
        warmup=a.warmup, comp_scale=a.comp_scale, progress=False,
        # score_hist ignores use_mismapping, but _fit reads it.
        use_mismapping=False,
    )


def run(a) -> None:
    import torch

    mm = a.mismap_dir
    npz = np.load(mm / "score_components.npz", allow_pickle=True)
    D, bin_edges = npz["D"], npz["bin_edges"]
    refseqs = [str(r) for r in npz["refseqs"]]
    model = si._load_error_model(a.model_pt, a.use_vi)
    v4_map = dict(si.read_fasta(mm / "v4_amplicons.fasta"))
    v4_seqs = [v4_map[r] for r in refseqs]
    emits = [si.emission_distribution(model, s) for s in v4_seqs]
    M = torch.tensor(D, dtype=torch.float64)

    T_df = pd.read_csv(mm / "translation_table.csv", index_col=0)
    genomes = list(T_df.index)
    T = torch.tensor(T_df.loc[genomes, refseqs].to_numpy(), dtype=torch.float64)
    g_of_ref = np.array([genomes.index(si.genome_of_header(r)) for r in refseqs])

    rng = np.random.default_rng(a.seed)
    reads_dir = a.reads_dir or Path(".")
    H_obs, total = si.observed_score_histograms(
        reads_dir, a.read_glob, v4_seqs, emits, bin_edges, a.gap_penalty, a.max_reads, rng)
    if total == 0:
        raise SystemExit(f"no reads matched {a.read_glob} under {reads_dir}")
    log.info("sample %s: %d reads, %d refs, %d genomes, mode=%s",
             a.sample_id, total, len(refseqs), len(genomes), a.mode)

    # Per-ref proxy (upper-half score mass) -> observed genome composition (init + baseline).
    w = H_obs[:, H_obs.shape[1] // 2:].sum(1)
    ref_rel = w / w.sum() if w.sum() > 0 else np.full(len(refseqs), 1.0 / len(refseqs))
    theta_obs = np.zeros(len(genomes))
    np.add.at(theta_obs, g_of_ref, ref_rel)
    theta_init = torch.tensor(np.clip(theta_obs, 1e-4, None), dtype=torch.float64)
    theta_init = theta_init / theta_init.sum()

    fa = _fit_args(a)
    samples, point, diag, losses = si._fit(
        a.mode, M, T, a.alpha, float(total),
        torch.tensor(H_obs, dtype=torch.float64), "score_hist", theta_init, fa,
        desc=a.sample_id)
    inferred = point.numpy()
    lo = hi = [np.nan] * len(genomes)
    if samples is not None:
        lo = torch.quantile(samples, 0.05, dim=0).numpy()
        hi = torch.quantile(samples, 0.95, dim=0).numpy()

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.DataFrame([{
        "sample": a.sample_id, "genome_id": g,
        "observed_rel_abundance": float(theta_obs[i]),
        "inferred_mean": float(inferred[i]),
        "inferred_lo": float(lo[i]), "inferred_hi": float(hi[i]),
    } for i, g in enumerate(genomes)])
    comp.to_csv(out / "inferred_composition.csv", index=False)
    pd.DataFrame([{
        "sample": a.sample_id, "mode": a.mode, "likelihood": "score_hist",
        "n_reads": int(total), **{k: si.json.dumps(v) for k, v in diag.items()},
    }]).to_csv(out / "inference_diagnostics.csv", index=False)
    if losses is not None:
        pd.DataFrame([{"sample": a.sample_id, "step": s, "loss": l}
                      for s, l in enumerate(losses)]).to_csv(out / "loss_trace.csv", index=False)
    print(f"infer[{a.mode}]: sample {a.sample_id}, {len(genomes)} genomes -> {out}")


def demo() -> None:
    """Self-check: fit histograms generated at a known theta and recover it.

    Two refs / two genomes (T=I). Build distinct per-ref score components D and set the
    observed H exactly to the model's prediction at theta_true — MLE must invert it.
    """
    import torch

    K, S = 6, 2
    D = np.zeros((S, S, K))
    # ref a scores high (upper bins) against itself, low (lower bins) against the other.
    hi = np.array([0.02, 0.03, 0.05, 0.15, 0.35, 0.40])
    lo = hi[::-1]
    for a in range(S):
        for j in range(S):
            D[a, j] = hi if a == j else lo
    T = torch.eye(S, dtype=torch.float64)
    theta_true = np.array([0.7, 0.3])
    p = np.einsum("a,ajk->jk", theta_true, D)      # [S,K]
    p = p / p.sum(-1, keepdims=True)
    total = 5000
    H = np.round(p * total)
    theta_init = torch.tensor([0.5, 0.5], dtype=torch.float64)
    fa = SimpleNamespace(mode="mle", lr=0.05, steps=1500, num_samples=1,
                         warmup=0, comp_scale=None, progress=False, use_mismapping=False)
    _, point, _, _ = si._fit("mle", torch.tensor(D), T, 0.5, float(total),
                             torch.tensor(H), "score_hist", theta_init, fa, desc="demo")
    est = point.numpy()
    assert np.allclose(est.sum(), 1.0, atol=1e-3), est
    assert np.abs(est - theta_true).max() < 0.1, (est, theta_true)
    print("demo OK:", np.round(est, 3), "~", theta_true)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mismap-dir", type=Path, help="dir with score_components.npz etc.")
    ap.add_argument("--model-pt", type=Path, help="trained error model (.pt)")
    ap.add_argument("--reads-dir", type=Path, help="dir of the sample's fastq(s)")
    ap.add_argument("--read-glob", default="*.fastq.gz", help="reads glob within --reads-dir")
    ap.add_argument("--sample-id", default="sample")
    ap.add_argument("--use-vi", action="store_true", help="use the model's VI posterior mean")
    ap.add_argument("--mode", choices=["nuts", "vi", "mle"], default="vi")
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet prior concentration")
    ap.add_argument("--gap-penalty", type=float, default=4.0)
    ap.add_argument("--max-reads", type=int, default=20000)
    ap.add_argument("--comp-scale", type=float, default=None,
                    help="composite-likelihood downweight (default = n_refs)")
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
        demo()
        return
    for req in ("mismap_dir", "model_pt", "output_dir"):
        if getattr(a, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (unless --demo)")
    run(a)


if __name__ == "__main__":
    main()
