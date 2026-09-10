#!/usr/bin/env python
"""Can the distance decay ``c`` be fitted per sample instead of chosen at build time?

``--align_distance_decay`` is a property of the *sample* (roughly its per-base error
rate), but it is baked into a matrix built once and shared by every sample in a matrix
group. ``--infer_distance_decay`` makes it a latent instead: the build records the
distance behind each nonzero, so inference can re-decay the matrix,
``M(c) = rownorm(M(c0) * (c / c0) ** d)``, without rebuilding anything.

Two questions, and the second is the one that decides whether the knob is honest:

1. Does the fit recover the decay the reads actually experienced, from a matrix built
   somewhere else?
2. When does it *not*? ``c`` reshapes a row across distance classes; ``s`` (the existing
   mis-mapping scale) rescales the whole off-diagonal. So ``c`` is identified only where
   rows hold both an exact duplicate and a near neighbour — and only where there are more
   references than genomes, or ``theta`` absorbs the difference by itself.

Synthetic kernels, no mapper and no aligner: the reference *topology* is the variable
under test, so it is built directly. Needs torch/pyro. ~2 min.

    python dev/latent_distance_decay.py [--out CSV]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))

import infer_composition as ic  # noqa: E402  (needs sys.path)
import sparse_matrix as sm  # noqa: E402  (needs sys.path)

BUILT_DECAY = 0.01          # what the shared matrix was built with
TRUE_DECAYS = (0.005, 0.02, 0.05, 0.15, 0.3)


def topology(shape: str):
    """``(owner, amplicon, distance, pattern, truth)`` for one reference-set shape.

    ``identified``
        14 references over 6 genomes and 10 distinct amplicons, four pairs one edit
        apart, several genomes multi-copy. Rows carry duplicates *and* neighbours, and
        references outnumber genomes: both conditions met.
    ``saturated``
        6 references over 6 genomes, three pairs one edit apart. Every row still mixes
        distance classes, but ``r_obs`` has exactly as many degrees of freedom as
        ``theta``, so the composition can explain any decay on its own.
    ``duplicates_only``
        14 references over 6 genomes, no neighbours at all — every cluster is exact
        duplicates. ``c`` cancels: this is the ``--align_tau 0`` case in disguise.
    """
    if shape == "identified":
        owner = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 3]
        amplicon = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6]
        edges = [(0, 1), (2, 3), (4, 5), (6, 7)]
        unique = 10
    elif shape == "saturated":
        owner = [0, 1, 2, 3, 4, 5]
        amplicon = [0, 1, 2, 3, 4, 5]
        edges = [(0, 1), (2, 3), (4, 5)]
        unique = 6
    elif shape == "duplicates_only":
        owner = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 3]
        amplicon = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6]
        edges = []
        unique = 10
    else:
        raise ValueError(shape)

    distance = np.zeros((unique, unique))
    pattern = np.eye(unique, dtype=bool)
    for i, j in edges:
        distance[i, j] = distance[j, i] = 1.0
        pattern[i, j] = pattern[j, i] = True
    truth = np.array([0.30, 0.25, 0.20, 0.12, 0.09, 0.04])
    return np.array(owner), np.array(amplicon), distance, pattern, truth


def kernel(pattern, distance, sizes, c):
    """The grouped tie-cluster kernel a build at decay ``c`` would write."""
    raw = np.where(pattern, c ** distance, 0.0)
    return sparse.csr_array(raw / (raw @ sizes)[:, None])


def run_config(shape: str, c_true: float, seed: int, reads: int) -> dict:
    """Fit one sample twice — decay latent, and fixed at the build's — and score both."""
    owner, group, distance, pattern, truth = topology(shape)
    sizes = np.bincount(group, minlength=pattern.shape[0]).astype(np.float64)
    refs = [f"g{genome}|{i}|x" for i, genome in enumerate(owner)]
    copies = np.bincount(owner)

    distances = sparse.csr_array(np.where(pattern, distance + 1.0, 0.0))
    distances.data -= 1.0            # d, with the zeros kept as explicit entries

    with tempfile.TemporaryDirectory() as temporary:
        td = Path(temporary)
        amplicon_dir = td / "amp"
        amplicon_dir.mkdir()
        pd.DataFrame({"genome_id": [f"g{genome}" for genome in owner], "refseq": refs,
                      "weight": [1.0 / copies[genome] for genome in owner]}).to_csv(
            amplicon_dir / "translation_table.tsv", sep="\t", index=False)

        # Reads pass through the matrix this sample actually experienced.
        r_true = np.array([truth[genome] / copies[genome] for genome in owner])
        M_true = kernel(pattern, distance, sizes, c_true).toarray()[np.ix_(group, group)]
        r_obs = r_true @ M_true
        counts = (reads * r_obs / r_obs.sum()).round().astype(int)
        with open(td / "obs.mseq", "w") as handle:
            for reference, n in enumerate(counts):
                for i in range(n):
                    handle.write(f"read{reference}_{i}\t{refs[reference]}\t500\t0.99\n")

        # ...but the matrix was built once, for the whole group, somewhere else.
        sm.write_grouped(td / "M.npz", kernel(pattern, distance, sizes, BUILT_DECAY),
                         group, refs, (distances, BUILT_DECAY))

        def fit(latent: bool, out: Path) -> np.ndarray:
            ic.run(SimpleNamespace(
                amplicon_dir=amplicon_dir, sim_mseq=None, mismapping_matrix=td / "M.npz",
                obs_mseq=[td / "obs.mseq"], min_identity=None, sample_id=shape, mode="vi",
                alpha=0.5, steps=2000, lr=0.05, num_samples=200, warmup=0,
                no_mismapping=False, seed=seed, no_presence=True, presence_prior=0.01,
                presence_temp=1.0, no_prune=False, infer_distance_decay=latent,
                decay_sigma=1.2, output_dir=out))
            return pd.read_csv(out / "inferred_composition.csv").set_index(
                "genome_id").loc[[f"g{i}" for i in range(len(truth))], "inferred_mean"]

        # A matrix with nothing but distance-0 clusters is refused rather than fitted:
        # ``c`` cancels there, so the likelihood is exactly flat in it.
        try:
            latent = fit(True, td / "latent")
            fitted = float(pd.read_csv(td / "latent" / "inference_diagnostics.csv"
                                       ).iloc[0]["distance_decay"])
        except SystemExit as refused:
            latent, fitted = None, float("nan")
            print(f"  {shape:16s} c_true={c_true:<6g} refused: {refused}")
        fixed = fit(False, td / "fixed")

    return {
        "shape": shape, "n_refs": len(refs), "n_genomes": len(truth),
        "built_decay": BUILT_DECAY, "true_decay": c_true, "fitted_decay": fitted,
        "l1_latent": (float("nan") if latent is None
                      else float(np.abs(latent.to_numpy() - truth).sum())),
        "l1_fixed": float(np.abs(fixed.to_numpy() - truth).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=_REPO / "dev" / "latent_distance_decay.csv")
    parser.add_argument("--reads", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for shape in ("identified", "saturated", "duplicates_only"):
        for c_true in TRUE_DECAYS:
            rows.append(run_config(shape, c_true, args.seed, args.reads))
            row = rows[-1]
            if np.isfinite(row["fitted_decay"]):
                print(f"  {shape:16s} c_true={c_true:<6g} fitted={row['fitted_decay']:.4f}"
                      f"  L1 {row['l1_latent']:.4f} vs {row['l1_fixed']:.4f} fixed")

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    print("\n" + frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
