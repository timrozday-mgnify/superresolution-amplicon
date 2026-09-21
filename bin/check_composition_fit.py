#!/usr/bin/env python
"""Check posterior-predictive fit for a panel composition inference.

The check consumes the exact observations, panel translation and panel kernel used for
inference. It rebuilds the fitted label space (observed labels, at most one sink, and the
background label ``U``), projects each retained posterior draw into it, and compares the
observed counts with posterior-predictive replicates.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import infer_composition as ic  # noqa: E402  (needs local bin directory)
import sparse_matrix as sm  # noqa: E402  (needs local bin directory)

log = logging.getLogger("check_composition_fit")


def _read_composition(path: Path) -> tuple[str, list[str], np.ndarray, str | None]:
    """Read and validate the composition table used as the reporting contract.

    Args:
        path: Path to ``inferred_composition.csv``.

    Returns:
        Sample identifier, panel entry identifiers, inferred means, and optional fit
        status.
    """
    table = pd.read_csv(path)
    id_column = "genome_id"
    required = {"sample", id_column, "inferred_mean"}
    if not required <= set(table.columns):
        raise SystemExit(f"{path} must contain {', '.join(sorted(required))}")
    if table[id_column].duplicated().any() or table["sample"].nunique() != 1:
        raise SystemExit(f"{path} must have one composition row per {id_column} for one sample")
    try:
        means = table["inferred_mean"].to_numpy(dtype=np.float64)
    except ValueError as exc:
        raise SystemExit(f"{path} has non-numeric inferred_mean values") from exc
    if not np.isfinite(means).all() or (means < 0).any():
        raise SystemExit(f"{path} has invalid inferred_mean values")
    status = str(table["fit_status"].iloc[0]) if "fit_status" in table else None
    return str(table["sample"].iloc[0]), table[id_column].tolist(), means, status


def _read_posterior_draws(
    path: Path,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, object]]:
    """Load retained posterior draws and their inference context.

    Args:
        path: Path to ``posterior_draws.npz`` emitted by inference.

    Returns:
        Genome identifiers, posterior draw arrays, and scalar metadata.
    """
    required = {
        "genome_ids",
        "theta_eff",
        "use_mismapping",
        "infer_distance_decay",
        "likelihood",
        "n_reads",
        "fit_status",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = required - set(archive.files)
            if missing:
                raise SystemExit(
                    f"{path} is missing posterior-draw fields: {', '.join(sorted(missing))}"
                )
            draws = {
                name: archive[name].astype(np.float64)
                for name in ("theta_eff", "s", "conc_frac", "c")
                if name in archive.files
            }
            metadata = {
                "use_mismapping": bool(archive["use_mismapping"].item()),
                "infer_distance_decay": bool(archive["infer_distance_decay"].item()),
                "likelihood": str(archive["likelihood"].item()),
                "n_reads": int(archive["n_reads"].item()),
                "fit_status": str(archive["fit_status"].item()),
            }
            genome_ids = archive["genome_ids"].astype(str).tolist()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid posterior draw archive {path}") from exc

    theta = draws["theta_eff"]
    if theta.ndim != 2 or theta.shape[1] != len(genome_ids):
        raise SystemExit(f"{path} has a theta_eff array that does not match genome_ids")
    return genome_ids, draws, metadata


def _kernel_at_decay(
    matrix: sparse.csr_array,
    strata: tuple[sparse.csr_array, float] | None,
    decay: float | None,
) -> sparse.csr_array:
    """Return the kernel at one posterior distance-decay draw.

    Args:
        matrix: Stored kernel at its calibrated decay.
        strata: Stored distance strata and calibrated decay.
        decay: Posterior decay draw, or None when no decay is fitted.

    Returns:
        Row-stochastic kernel at the requested decay.
    """
    if decay is None:
        return matrix
    if strata is None:
        raise SystemExit("posterior contains c but the mis-mapping matrix has no distance strata")
    distances, built_decay = strata
    if not 0.0 < decay <= 1.0:
        raise SystemExit("posterior distance-decay draws must be in (0, 1]")
    kernel = matrix.copy().tocsr()
    kernel.data *= (decay / built_decay) ** distances.data
    row_sums = np.asarray(kernel.sum(axis=1)).ravel()
    if (row_sums <= 0).any():
        raise SystemExit("re-decayed matrix has an empty row")
    kernel.data /= np.repeat(row_sums, np.diff(kernel.indptr))
    return kernel


def _apply_mismapping(
    true_proportions: np.ndarray,
    matrix: sparse.csr_array,
    scale: float,
    home: np.ndarray,
) -> np.ndarray:
    """Apply the inference model's scaled kernel without expansion.

    Args:
        true_proportions: True-source proportions in source order.
        matrix: Sources x labels kernel.
        scale: Posterior mis-mapping scale draw.
        home: Each source's home label.

    Returns:
        Predicted MAPseq label proportions.
    """
    if scale < 0:
        raise SystemExit("posterior s draws must be non-negative")
    diagonal = sm.home_entries(matrix, home)
    diagonal_effective = np.maximum(1.0 - scale + scale * diagonal, 0.0)
    row_sums = diagonal_effective + scale * (1.0 - diagonal)
    mapped = np.asarray((true_proportions * (scale / row_sums)) @ matrix).ravel()
    correction = true_proportions * ((diagonal_effective - scale * diagonal) / row_sums)
    predicted = mapped + np.bincount(home, correction, minlength=matrix.shape[1])
    if not np.isfinite(predicted).all() or (predicted < -1e-12).any():
        raise SystemExit("mis-mapping projection produced invalid probabilities")
    predicted = np.clip(predicted, 0.0, None)
    return predicted / predicted.sum()


def _tv_distance(observed: np.ndarray, expected: np.ndarray) -> float:
    """Return total-variation distance between two non-negative count vectors.

    Args:
        observed: Observed count or probability vector.
        expected: Expected count or probability vector.

    Returns:
        Total-variation distance after normalisation.
    """
    observed_total, expected_total = observed.sum(), expected.sum()
    if observed_total <= 0 or expected_total <= 0:
        return float("nan")
    return float(0.5 * np.abs(observed / observed_total - expected / expected_total).sum())


def _quantiles(values: np.ndarray) -> dict[str, float]:
    """Return stable summary quantiles for a non-empty vector.

    Args:
        values: Numeric vector.

    Returns:
        Named five, fifty, and ninety-five percent quantiles.
    """
    return {
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _posterior_predictive_diagnostics(
    observations: np.ndarray,
    predicted: np.ndarray,
    concentration_fraction: np.ndarray | None,
    likelihood: str,
    rng: np.random.Generator,
) -> dict[str, float | dict[str, float]]:
    """Calculate the predictive residual over the fitted labels from posterior draws.

    Args:
        observations: Observed counts over the fitted labels.
        predicted: One predicted label distribution per posterior draw.
        concentration_fraction: Dirichlet-multinomial concentration draws, if fitted.
        likelihood: Inference likelihood name.
        rng: Random generator for posterior-predictive replicates.

    Returns:
        The label-space diagnostic mapping.
    """
    total = int(observations.sum())
    observed = np.empty(len(predicted), dtype=np.float64)
    replicated_tv = np.empty(len(predicted), dtype=np.float64)
    for draw_index, probabilities in enumerate(predicted):
        if likelihood == "dirichlet_multinomial":
            if concentration_fraction is None:
                raise SystemExit("posterior draws lack conc_frac for a Dirichlet-multinomial fit")
            concentration = concentration_fraction[draw_index] * total * probabilities + 1e-6
            replicated = rng.multinomial(total, rng.dirichlet(concentration))
        elif likelihood == "multinomial":
            replicated = rng.multinomial(total, probabilities)
        else:
            raise SystemExit(f"unsupported likelihood in posterior draw archive: {likelihood}")
        observed[draw_index] = _tv_distance(observations, probabilities)
        replicated_tv[draw_index] = _tv_distance(replicated, probabilities)
    return {
        "n_labels": len(observations),
        "observed_expected_tv": _tv_distance(observations, predicted.mean(axis=0)),
        "observed_statistic": _quantiles(observed),
        "replicated_statistic": _quantiles(replicated_tv),
        "posterior_predictive_percentile": float(np.mean(replicated_tv <= observed)),
    }


def run(args: argparse.Namespace) -> None:
    """Run the fit check and write one machine-readable diagnostics document.

    Args:
        args: Parsed command-line arguments.
    """
    sample, composition_genomes, composition_means, composition_status = _read_composition(
        args.composition
    )
    posterior_genomes, draws, metadata = _read_posterior_draws(args.posterior_draws)
    panel = ic._panel_context(args.mismapping_matrix, args.amplicon_dir, args.obs_mseq,
                              args.min_identity, not metadata["use_mismapping"])
    observations = panel.y
    # The composition is reported per panel entry; the draws stay per fitted member.
    to_entry = np.eye(len(panel.entries))[panel.entry_of]
    if composition_genomes != panel.entries or posterior_genomes != panel.genomes:
        raise SystemExit("composition table and posterior draws must use the inference order")
    total = int(observations.sum())
    if total != metadata["n_reads"]:
        raise SystemExit(
            "observed MAPseq count total differs from the count used for inference; "
            "check the exact observation table and identity threshold"
        )
    if args.min_infer_reads < 1:
        raise SystemExit("--min-infer-reads must be at least 1")

    payload: dict[str, object] = {
        "sample": sample,
        "kernel_format": "rectangular",
        "n_reads": total,
        "min_infer_reads": args.min_infer_reads,
        "matrix_path": str(args.mismapping_matrix),
        "posterior_draw_path": str(args.posterior_draws),
        "label": None,
    }
    low_depth = total < args.min_infer_reads or metadata["fit_status"] == "low_depth"
    if low_depth:
        payload["fit_status"] = "low_depth"
        payload["reason"] = "insufficient mapped reads for biological composition inference"
    elif composition_status == "low_depth":
        payload["fit_status"] = "low_depth"
        payload["reason"] = "composition table is depth-gated"
    else:
        theta = draws["theta_eff"]
        if not len(theta):
            raise SystemExit("high-depth inference has no retained theta_eff posterior draws")
        if not np.isfinite(theta).all() or (theta < 0).any():
            raise SystemExit("posterior theta_eff draws must be finite and non-negative")
        if not np.allclose(theta.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
            raise SystemExit("posterior theta_eff draws must each sum to one")

        available_draws = len(theta)
        draw_count = args.ppc_draws
        rng = np.random.default_rng(args.seed)
        if available_draws >= draw_count:
            draw_indices = np.linspace(0, available_draws - 1, draw_count, dtype=np.int64)
        else:
            # MLE retains one fitted point rather than a posterior sample. Reusing that
            # point for independently drawn replicated outcomes still gives a calibrated
            # sampling check instead of a degenerate percentile of exactly zero or one.
            draw_indices = rng.integers(available_draws, size=draw_count, dtype=np.int64)
        if not np.allclose(composition_means, theta.mean(axis=0) @ to_entry,
                           rtol=1e-6, atol=1e-8):
            raise SystemExit("composition inferred_mean values do not match retained theta_eff draws")
        theta = theta[draw_indices]
        # A panel fit always samples s; --no-mismapping swapped K for home rows instead.
        if "s" not in draws:
            raise SystemExit("posterior draws lack s")
        scales = draws["s"][draw_indices]
        decays = None
        if metadata["infer_distance_decay"]:
            if "c" not in draws:
                raise SystemExit("posterior draws lack c for a fitted distance decay")
            decays = draws["c"][draw_indices]
        predicted = np.empty((draw_count, len(observations)), dtype=np.float64)
        for draw_index in range(draw_count):
            true_proportions = theta[draw_index, panel.genome_of_row] * panel.weight
            true_proportions = true_proportions / true_proportions.sum()
            kernel = _kernel_at_decay(
                panel.fitted, panel.strata,
                None if decays is None else float(decays[draw_index]),
            )
            predicted[draw_index] = _apply_mismapping(
                true_proportions, kernel, float(scales[draw_index]), panel.home
            )

        if metadata["likelihood"] == "dirichlet_multinomial" and "conc_frac" not in draws:
            raise SystemExit("posterior draws lack conc_frac for a Dirichlet-multinomial fit")
        concentration = (
            draws["conc_frac"][draw_indices]
            if metadata["likelihood"] == "dirichlet_multinomial"
            else None
        )
        label = _posterior_predictive_diagnostics(
            observations, predicted, concentration, str(metadata["likelihood"]), rng
        )
        payload["posterior_draw_count"] = draw_count
        payload["label"] = label
        payload["gate"] = "label"
        percentile = float(label["posterior_predictive_percentile"])
        payload["fit_status"] = (
            "ok" if args.ppc_low_tail <= percentile <= args.ppc_high_tail else "model_misfit"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"fit[{payload['fit_status']}]: sample {sample}, {total} mapped reads -> {args.output}")


def main() -> None:
    """Parse command-line arguments and run the composition fit check."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--posterior-draws", type=Path, required=True)
    parser.add_argument("--obs-mseq", type=Path, nargs="+", required=True)
    parser.add_argument("--amplicon-dir", type=Path, required=True)
    parser.add_argument("--mismapping-matrix", type=Path, required=True)
    parser.add_argument("--min-identity", type=float, default=None)
    parser.add_argument("--min-infer-reads", type=int, default=1000)
    parser.add_argument("--ppc-draws", type=int, default=500)
    parser.add_argument("--ppc-low-tail", type=float, default=0.01)
    parser.add_argument("--ppc-high-tail", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    if args.ppc_draws < 1:
        parser.error("--ppc-draws must be at least 1")
    if not 0.0 <= args.ppc_low_tail < args.ppc_high_tail <= 1.0:
        parser.error("PPC tails must satisfy 0 <= low < high <= 1")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
