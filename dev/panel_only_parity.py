#!/usr/bin/env python
"""Phase P.7 parity: panel-only inference against the P.0 snapshots.

    python dev/panel_only_parity.py rescore      # 20HM S01-S20 and SILVA S05, inference only
    python dev/panel_only_parity.py compare      # prints the tables in panel_only_parity.md
    python dev/panel_only_parity.py depth P0_CHECKOUT 1 2 4 8 ...   # B. uniformis depth sweep

``rescore`` reruns ``infer_composition.py`` + ``check_composition_fit.py`` on the stored
observations and kernels the snapshots were made from (``work/panel_obs``,
``work/panel_kernel``, ``work/silva``), with the snapshots' flags. ``compare`` also reads
the pipeline runs of the fixture and B. uniformis set, one per mode, from
``work/panel_only_parity_p7/<set>_<mode>/`` (see panel_only_parity.md for the commands).
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BIN, SNAP, WORK = ROOT / "bin", ROOT / "dev" / "panel_only_parity", ROOT / "work"
OUT = WORK / "panel_only_parity_p7"
INFER = ["--mode", "vi", "--alpha", "0.5", "--steps", "3000", "--presence-prior", "0.01",
         "--presence-temp", "1.0"]
SAMPLES = [f"S{i:02d}" for i in range(1, 21)]
MODES = {"simulate": "simulate", "align_exact_hash": "align_tau0", "align_kmer_tau1": "align_tau1"}
SETS = {"fixture": "demo", "buniformis": "b_uniformis"}


def _infer(obs: Path, prep: Path, kernel: Path, out: Path, sample: str, extra=()) -> None:
    common = ["--obs-mseq", str(obs), "--amplicon-dir", str(prep),
              "--mismapping-matrix", str(kernel)]
    subprocess.run([sys.executable, BIN / "infer_composition.py", *common, *INFER, *extra,
                    "--sample-id", sample, "-o", str(out)], check=True, capture_output=True)
    subprocess.run([sys.executable, BIN / "check_composition_fit.py", *common,
                    "--composition", str(out / "inferred_composition.csv"),
                    "--posterior-draws", str(out / "posterior_draws.npz"),
                    "-o", str(out / "fit.json")], check=True, capture_output=True)


def rescore() -> None:
    jobs = [(WORK / "panel_obs" / f"{s}.obs.mseq", WORK / "panel_kernel",
             WORK / "panel_kernel" / "trained_s0" / "panel_kernel.npz",
             OUT / "panel_20hm" / s, s, [])
            for s in SAMPLES]
    silva = WORK / "silva"
    jobs.append((silva / "obs" / "S05.obs.mseq", silva / "species", silva / "species" / "cal.npz",
                 OUT / "silva_species_s05", "S05", ["--no-presence"]))
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda job: _infer(*job), jobs))


def _genomes(path: Path) -> pd.Series:
    """Inferred means over genomes, background dropped and renormalised."""
    means = pd.read_csv(path).set_index("genome_id")["inferred_mean"].drop("background",
                                                                           errors="ignore")
    return means / means.sum()


def _tv(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, fill_value=0.0)
    return 0.5 * float((a - b).abs().sum())


def _fit(path: Path) -> str:
    return json.loads(path.read_text())["fit_status"]


def compare() -> None:
    print("| set | mode | genome TV vs P.0 | max abs diff | fit (P.0 -> now) |")
    print("|---|---|---|---|---|")
    for folder, sample in SETS.items():
        for snap_mode, mode in MODES.items():
            old = SNAP / folder / snap_mode / "composition" / sample
            new = OUT / f"{folder}_{mode}" / "composition" / sample
            if not new.exists():
                print(f"| {folder} | {mode} | not run | | |")
                continue
            a = _genomes(old / f"{sample}.inferred_composition.csv")
            b = _genomes(new / f"{sample}.inferred_composition.csv")
            diff = a.align(b, fill_value=0.0)
            print(f"| {folder} | {mode} | {_tv(a, b):.4f} | "
                  f"{(diff[0] - diff[1]).abs().max():.4f} | "
                  f"{_fit(old / f'{sample}.fit_diagnostics.json')} -> "
                  f"{_fit(new / f'{sample}.fit_diagnostics.json')} |")

    print("\n| panel | sample | max abs diff (entries incl. background) | fit (P.0 -> now) |")
    print("|---|---|---|---|")
    pairs = [("20HM sim_trained", s, SNAP / "panel_20hm" / "sim_trained" / s,
              OUT / "panel_20hm" / s) for s in SAMPLES]
    pairs.append(("SILVA species", "S05", SNAP / "silva_species_s05" / "sim_calibrated_flat_nogate",
                  OUT / "silva_species_s05"))
    for name, sample, old, new in pairs:
        a = pd.read_csv(old / "inferred_composition.csv").set_index("genome_id")["inferred_mean"]
        b = pd.read_csv(new / "inferred_composition.csv").set_index("genome_id")["inferred_mean"]
        print(f"| {name} | {sample} | {(a - b).abs().max():.2e} | "
              f"{_fit(old / 'fit.json')} -> {_fit(new / 'fit.json')} |")


# The pipeline's INFER_COMPOSITION flags (conf/modules.config defaults, --seed 42).
PIPELINE_INFER = ["--seed", "42", "--mode", "vi", "--alpha", "0.5", "--steps", "3000",
                  "--lr", "0.02", "--num-samples", "500", "--warmup", "500",
                  "--min-infer-reads", "1000", "--presence-prior", "0.01",
                  "--presence-temp", "1.0"]
OLD_RUNS = {"simulate": "buniformis_simulate", "align_tau0": "buniformis_exact_hash",
            "align_tau1": "buniformis_kmer_tau1"}


def _task_dir(root: Path, needle: str) -> Path:
    """The INFER_COMPOSITION work dir under ``root`` whose script mentions ``needle``."""
    for script in root.rglob(".command.sh"):
        text = script.read_text()
        if "infer_composition.py" in text and "--obs-mseq" in text and needle in text:
            return script.parent
    raise SystemExit(f"no inference task for {needle} under {root}")


def depth(p0_checkout: Path, multiples: list[int]) -> None:
    """Scale the B. uniformis observation k-fold (same composition, k x the reads) and fit
    it with the P.0 square code and the panel code, gate on, the pipeline's flags."""
    print("| mode | x reads | reads | TV(P.0 code, panel code) | present P.0 / panel "
          "| conc_frac P.0 / panel |")
    print("|---|---|---|---|---|---|")
    observed = WORK / "panel_only_parity" / "buniformis" / "observed.mseq"
    rows = [line for line in observed.read_text().splitlines() if line and not line.startswith("#")]
    for mode, old_run in OLD_RUNS.items():
        old_task = _task_dir(WORK / "panel_only_parity" / old_run, "b_uniformis")
        key = next((OUT / f"buniformis_{mode}" / "mismapping").glob("panel_*")).name
        new_task = _task_dir(OUT / "work", key)
        for k in multiples:
            base = OUT / "depth" / mode / f"x{k}"
            base.mkdir(parents=True, exist_ok=True)
            obs = base / "observed.mseq"
            obs.write_text("".join(f"r{i}_{line}\n" for i in range(k) for line in rows))
            fits = {}
            for name, code, task, amplicons, extra in (
                    ("old", p0_checkout, old_task, "refs_0970e0d6aa57_amplicons",
                     ["--infer-space", "genome"]),
                    ("new", ROOT, new_task, key, [])):
                out = base / name
                subprocess.run([sys.executable, code / "bin" / "infer_composition.py",
                                "--amplicon-dir", str(task / amplicons),
                                "--mismapping-matrix", str(task / "mismapping_matrix.npz"),
                                "--obs-mseq", str(obs), "--sample-id", "b", "-o", str(out),
                                *PIPELINE_INFER, *extra], check=True, capture_output=True)
                comp = pd.read_csv(out / "inferred_composition.csv")
                with np.load(out / "posterior_draws.npz") as draws:
                    conc = float(np.median(draws["conc_frac"]))
                fits[name] = (_genomes(out / "inferred_composition.csv"),
                              int((comp.presence_prob > 0.5).sum()), conc)
            (a, pa, ca), (b, pb, cb) = fits["old"], fits["new"]
            print(f"| {mode} | {k} | {k * len(rows)} | {_tv(a, b):.4f} | {pa} / {pb} "
                  f"| {ca:.3f} / {cb:.3f} |", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "depth":
        depth(Path(sys.argv[2]), [int(k) for k in sys.argv[3:]])
    else:
        {"rescore": rescore, "compare": compare}[sys.argv[1]]()
