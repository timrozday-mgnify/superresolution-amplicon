"""Panel reinterpretation of GTDB labels on the 20 sweep samples, against the 20HM custom DB.

    python dev/panel_reinterpretation_sweep.py run --obs-dir work/panel_obs \
        --kernel-dir work/panel_kernel -o work/panel_sweep --jobs 6
    python dev/panel_reinterpretation_sweep.py score --sweep-dir SWEEP_RESULTS \
        -o work/panel_sweep > dev/panel_reinterpretation_sweep.csv

``run`` fits every arm on every ``<sample>.obs.mseq`` with the pipeline's inference settings
(vi, alpha 0.5, 3,000 steps, presence prior 0.01, temperature 1.0) and runs the fit check.
Phase 5 arms vary the prior (``_nogate``, ``_horseshoe``), pin ``s`` at 1 (``_s1``), run
10,000 steps (``_10k``), or drop one genome from the panel (``_del_<genome>``).
``score`` compares each arm and each 20HM custom-database profile (``.<matrix>.p01``) with the
sample's ``realized_rel_abundance`` and prints one row per (sample, method).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BIN = Path(__file__).resolve().parents[1] / "bin"

# arm -> (kernel subdirectory, extra infer_composition flags, genome dropped from the panel)
ARMS = {
    "sim_trained": ("trained_s0", [], None),
    "sim_trained_seed1": ("trained_s1", [], None),
    "sim_calibrated_flat": ("cal", [], None),
    "align_exact": ("align_exact", [], None),
    "align_exact_self_home": ("align_exact_self", [], None),
    "align_kmer1": ("align_kmer1", [], None),
    "align_kmer1_latent": ("align_kmer1", ["--infer-distance-decay"], None),
    "home_labels_only": ("trained_s0", ["--no-mismapping"], None),
}
PRIORS = {"": [], "_nogate": ["--no-presence"], "_horseshoe": ["--horseshoe", "--no-presence"]}
PIN_S = ["--s-sigma", "1e-4"]
DELETED = ["fusobacterium_nucleatum", "salinibacter_ruber"]
for base in ["sim_trained", "sim_trained_seed1", "sim_calibrated_flat", "align_exact",
             "align_kmer1_latent", "home_labels_only"]:
    kernel, flags, _ = ARMS[base]
    for prior, prior_flags in PRIORS.items():
        ARMS[base + prior] = (kernel, flags + prior_flags, None)
        # s has no effect on home rows, so home labels only has no pinned arm.
        if base != "home_labels_only":
            ARMS[base + prior + "_s1"] = (kernel, flags + prior_flags + PIN_S, None)
for prior, prior_flags in PRIORS.items():
    ARMS["sim_trained" + prior + "_10k"] = ("trained_s0", prior_flags + ["--steps", "10000"],
                                            None)
for base in ["sim_trained", "sim_calibrated_flat", "home_labels_only"]:
    kernel, flags, _ = ARMS[base]
    for prior in ["", "_nogate"]:
        for genome in DELETED:
            ARMS[f"{base}{prior}_del_{genome}"] = (kernel, flags + PRIORS[prior], genome)

CUSTOM = ["simulate", "exact", "kmer1", "kmer1_latent"]
INFER = ["--mode", "vi", "--alpha", "0.5", "--steps", "3000", "--presence-prior", "0.01",
         "--presence-temp", "1.0"]
BU_PAIR = ("bacteroides_uniformis", "bacteroides_uniformis_strain2")


def panel_dir(a, genome: str | None) -> Path:
    """The kernel directory, or a copy of its translation without ``genome``."""
    if genome is None:
        return a.kernel_dir
    out = a.out / "panels" / genome
    if not (out / "panel_translation.tsv").exists():
        out.mkdir(parents=True, exist_ok=True)
        table = pd.read_csv(a.kernel_dir / "panel_translation.tsv", sep="\t")
        table[table.genome_id != genome].to_csv(out / "panel_translation.tsv", sep="\t",
                                                index=False)
    return out


def one(obs: Path, arm: str, a) -> str:
    sample = obs.name.removesuffix(".obs.mseq")
    kernel_dir, extra, deleted = ARMS[arm]
    kernel = a.kernel_dir / kernel_dir / "panel_kernel.npz"
    out = a.out / sample / arm
    if (out / "fit.json").exists():
        return f"{sample} {arm}: done"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    common = ["--obs-mseq", str(obs), "--amplicon-dir", str(panel_dir(a, deleted)),
              "--mismapping-matrix", str(kernel)]
    # Later flags win in argparse, so an arm's --steps overrides INFER's.
    subprocess.run([sys.executable, BIN / "infer_composition.py", *common, *INFER, *extra,
                    "--sample-id", sample, "-o", str(out)], check=True, env=env,
                   capture_output=True)
    subprocess.run([sys.executable, BIN / "check_composition_fit.py", *common,
                    "--composition", str(out / "inferred_composition.csv"),
                    "--posterior-draws", str(out / "posterior_draws.npz"),
                    "-o", str(out / "fit.json")], check=True, env=env, capture_output=True)
    return f"{sample} {arm}: ok"


def run(a) -> None:
    for genome in DELETED:
        panel_dir(a, genome)            # before the threads, so no two jobs write it
    jobs = [(obs, arm) for obs in sorted(a.obs_dir.glob("S*.obs.mseq")) for arm in ARMS]
    with ThreadPoolExecutor(a.jobs) as pool:
        for message in pool.map(lambda job: one(*job, a), jobs):
            print(message, file=sys.stderr, flush=True)


def scores(pred: pd.Series, truth: pd.Series) -> dict:
    """Genome TV over renormalised panel genomes, BU false mass and BU split error."""
    pred = pred / pred.sum()
    j = pd.concat([pred, truth], axis=1, keys=["p", "t"]).fillna(0.0)
    err = (j.p - j.t).abs()
    both_t, both_p = j.t[list(BU_PAIR)].sum(), j.p[list(BU_PAIR)].sum()
    split = (abs(j.p[BU_PAIR[1]] / both_p - j.t[BU_PAIR[1]] / both_t)
             if both_t > 0.01 and both_p > 0 else float("nan"))
    return {"tv": 0.5 * err.sum(), "max_abs_error": err.max(), "worst_genome": err.idxmax(),
            "dorea_abs_error": err.get("dorea_formicigenerans", 0.0),
            "mass_absent_from_truth": j.p[j.t == 0].sum(), "bu_split_error": split}


def score(a) -> None:
    rows = []
    for sample_dir in sorted(a.sweep_dir.glob("S*_*")):
        sample = sample_dir.name.split("_")[0]
        truth = pd.read_csv(sample_dir / f"{sample_dir.name}.truth.tsv", sep="\t",
                            index_col="genome_id").realized_rel_abundance
        truth = truth / truth.sum()
        base = {"sample": sample, "alpha": sample_dir.name.split("_")[1]}
        for matrix in CUSTOM:
            profile = pd.read_csv(sample_dir / f"{sample_dir.name}.{matrix}.p01.sr_profile.tsv",
                                  sep="\t", index_col="genome_id").predicted_rel_abundance
            rows.append({**base, "method": f"custom20hm_{matrix}", **scores(profile, truth)})
        # MAPseq only against the 20HM DB: reads per genome's references, no inference.
        observed = pd.read_csv(sample_dir / "profiling" / "sr" /
                               f"{sample_dir.name}.simulate.p01.inferred_composition.csv",
                               index_col="genome_id").observed_rel_abundance
        rows.append({**base, "method": "custom20hm_mapseq_only", **scores(observed, truth)})
        background = {}
        for arm, (_, _, deleted) in ARMS.items():
            out = a.out / sample / arm
            if not (out / "fit.json").exists():
                continue
            comp = pd.read_csv(out / "inferred_composition.csv", index_col="genome_id")
            diag = pd.read_csv(out / "inference_diagnostics.csv").iloc[0]
            fit = json.loads((out / "fit.json").read_text())
            row = {**base, "method": f"panel_{arm}",
                   "unexplained_fraction": diag.unexplained_fraction,
                   "fit_status": fit["fit_status"],
                   "ppc_percentile": (fit["unique_v4_group"] or {}).get(
                       "posterior_predictive_percentile"),
                   "fitted_distance_decay": diag.get("distance_decay"),
                   "fitted_s": float(np.load(out / "posterior_draws.npz")["s"].mean())}
            if deleted is None:
                background[arm] = diag.unexplained_fraction
                row.update(scores(comp.inferred_mean.drop("background"), truth))
            else:
                # The full-panel arm's background is the baseline the deleted mass adds to.
                full = arm.split("_del_")[0]
                row.update(scores(comp.inferred_mean.drop("background"),
                                  truth.drop(deleted)))
                row["deleted_truth"] = truth[deleted]
                row["deleted_to_background"] = ((diag.unexplained_fraction - background[full])
                                                / truth[deleted] if truth[deleted] > 0
                                                else float("nan"))
            rows.append(row)
    pd.DataFrame(rows).to_csv(sys.stdout, index=False, float_format="%.5f")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--obs-dir", type=Path, required=True)
    r.add_argument("--kernel-dir", type=Path, required=True)
    r.add_argument("--jobs", type=int, default=6)
    r.add_argument("-o", "--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--sweep-dir", type=Path, required=True)
    s.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    {"run": run, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
