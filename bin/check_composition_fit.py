#!/usr/bin/env python
"""Check posterior-predictive fit for an amplicon composition inference.

The check consumes the exact observations, translation table, and mis-mapping matrix used
for inference. It reconstructs the fitted active subset, projects each retained posterior
draw into MAPseq observation space, and compares observed counts with posterior-predictive
replicates at both reference and unique-V4-sequence resolution.
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
import subspecies_infer as si  # noqa: E402  (needs local bin directory)

log = logging.getLogger("check_composition_fit")


def _translation_weights(
    amplicon_dir: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Return genomes, references, reference genome indices, and copy weights.

    Args:
        amplicon_dir: Directory containing the inference translation table.

    Returns:
        Genome identifiers, reference identifiers, reference-to-genome indices, and
        per-reference within-genome weights.
    """
    genomes, references, translation = ic._translation(amplicon_dir)
    if isinstance(translation, tuple):
        genome_of_reference, weights, _ = translation
        return genomes, references, genome_of_reference, weights

    owners = np.flatnonzero(translation > 0)
    if len(owners) != len(references):
        raise SystemExit(
            "legacy translation table must assign every reference to exactly one genome"
        )
    owner_rows, owner_columns = np.nonzero(translation > 0)
    if len(np.unique(owner_columns)) != len(references):
        raise SystemExit(
            "legacy translation table must assign every reference to exactly one genome"
        )
    order = np.argsort(owner_columns, kind="stable")
    return genomes, references, owner_rows[order], translation[owner_rows[order], owner_columns[order]]


def _matrix(
    matrix_path: Path,
    references: list[str],
) -> tuple[sparse.csr_array, np.ndarray | None, tuple[sparse.csr_array, float] | None]:
    """Load the labelled sparse matrix in translation-table reference order.

    Args:
        matrix_path: Path to a dense or grouped sparse matrix archive.
        references: Required reference order.

    Returns:
        Matrix, optional grouped-kernel labels, and optional distance strata.
    """
    if matrix_path.suffix != ".npz":
        raise SystemExit("--mismapping-matrix must be the exact sparse .npz matrix bundle entry")
    try:
        if sm.is_grouped(matrix_path):
            return sm.read_grouped(matrix_path, references, strata=True)
        matrix, strata = sm.read_matrix(matrix_path, references, strata=True)
        return matrix, None, strata
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _read_composition(path: Path) -> tuple[str, list[str], np.ndarray, str | None]:
    """Read and validate the composition table used as the reporting contract.

    Args:
        path: Path to ``inferred_composition.csv``, or ``inferred_v4_groups.csv`` for a
            V4-group-space fit.

    Returns:
        Sample identifier, genome or V4-group identifiers, inferred means, and optional
        fit status.
    """
    table = pd.read_csv(path)
    id_column = "v4_group_id" if "v4_group_id" in table else "genome_id"
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
        "use_active_subset",
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
                "use_active_subset": bool(archive["use_active_subset"].item()),
                "n_reads": int(archive["n_reads"].item()),
                "fit_status": str(archive["fit_status"].item()),
                # Archives written before V4-group inference existed are genome space.
                "infer_space": (str(archive["infer_space"].item())
                                if "infer_space" in archive.files else "genome"),
            }
            genome_ids = archive["genome_ids"].astype(str).tolist()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid posterior draw archive {path}") from exc

    theta = draws["theta_eff"]
    if theta.ndim != 2 or theta.shape[1] != len(genome_ids):
        raise SystemExit(f"{path} has a theta_eff array that does not match genome_ids")
    return genome_ids, draws, metadata


def _subset_context(
    matrix: sparse.csr_array,
    matrix_group: np.ndarray | None,
    strata: tuple[sparse.csr_array, float] | None,
    references: list[str],
    reference_genomes: np.ndarray,
    weights: np.ndarray,
    observations: np.ndarray,
    v4_group: np.ndarray,
    use_active_subset: bool,
) -> tuple[
    sparse.csr_array,
    np.ndarray | None,
    tuple[sparse.csr_array, float] | None,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Reconstruct the matrix and translation subset used during inference.

    Args:
        matrix: Full reference or grouped-kernel matrix.
        matrix_group: Optional matrix duplicate-group labels.
        strata: Optional distance strata paired with matrix.
        references: Full reference identifiers.
        reference_genomes: Full reference-to-genome indices.
        weights: Full per-reference copy weights.
        observations: Full observed reference counts.
        v4_group: Full exact V4 group labels.
        use_active_subset: Whether the inference used active-subset conditioning.

    Returns:
        The fitted matrix context, local reference-genome indices, V4 labels, and global
        genome indices represented by the fitted context.
    """
    if not use_active_subset:
        genome_indices = np.arange(int(reference_genomes.max()) + 1, dtype=np.int64)
        reference_indices = np.arange(len(references), dtype=np.int64)
        return (
            matrix,
            matrix_group,
            strata,
            reference_genomes,
            weights,
            v4_group,
            genome_indices,
            reference_indices,
        )

    reference_indices, genome_indices = ic._active_subset(
        matrix, matrix_group, observations, reference_genomes
    )
    genome_positions = np.full(int(reference_genomes.max()) + 1, -1, dtype=np.int64)
    genome_positions[genome_indices] = np.arange(len(genome_indices))
    matrix, matrix_group, strata, n_sink = ic._subset_mismapping(
        matrix, matrix_group, reference_indices, strata
    )
    # Sink labels, as in inference: zero weight, no reads, one V4 label each.
    local_v4 = np.unique(v4_group[reference_indices], return_inverse=True)[1].astype(np.int64)
    return (
        matrix,
        matrix_group,
        strata,
        np.concatenate([genome_positions[reference_genomes[reference_indices]],
                        np.zeros(n_sink, dtype=np.int64)]),
        np.concatenate([weights[reference_indices], np.zeros(n_sink)]),
        np.concatenate([local_v4, local_v4.max(initial=-1) + 1 + np.arange(n_sink)]),
        genome_indices,
        reference_indices,
    )


def _kernel_at_decay(
    matrix: sparse.csr_array,
    matrix_group: np.ndarray | None,
    strata: tuple[sparse.csr_array, float] | None,
    decay: float | None,
) -> sparse.csr_array:
    """Return the kernel at one posterior distance-decay draw.

    Args:
        matrix: Stored kernel at its calibrated decay.
        matrix_group: Optional grouped-kernel labels.
        strata: Stored distance strata and calibrated decay.
        decay: Posterior decay draw, or None when no decay is fitted.

    Returns:
        Row-stochastic reference or grouped kernel at the requested decay.
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
    sizes = (
        np.ones(kernel.shape[1], dtype=np.float64)      # labels; square unless rectangular
        if matrix_group is None
        else np.bincount(matrix_group, minlength=kernel.shape[0]).astype(np.float64)
    )
    row_sums = np.asarray(kernel @ sizes).ravel()
    if (row_sums <= 0).any():
        raise SystemExit("re-decayed matrix has an empty row")
    kernel.data /= np.repeat(row_sums, np.diff(kernel.indptr))
    return kernel


def _apply_mismapping(
    true_proportions: np.ndarray,
    matrix: sparse.csr_array,
    matrix_group: np.ndarray | None,
    scale: float,
    home: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the inference model's scaled confusion matrix without expansion.

    Args:
        true_proportions: True-source proportions in reference (or source) order.
        matrix: Reference-square, grouped, or rectangular sources x labels kernel.
        matrix_group: Optional grouped-kernel labels.
        scale: Posterior mis-mapping scale draw.
        home: Each source's home label, for a rectangular kernel only.

    Returns:
        Predicted MAPseq reference-label (or label) proportions.
    """
    if scale < 0:
        raise SystemExit("posterior s draws must be non-negative")
    if home is not None:
        diagonal = sm.home_entries(matrix, home)
    elif matrix_group is None:
        diagonal = matrix.diagonal()
    else:
        diagonal = matrix.diagonal()[matrix_group]
    diagonal_effective = np.maximum(1.0 - scale + scale * diagonal, 0.0)
    row_sums = diagonal_effective + scale * (1.0 - diagonal)
    scaled_mass = true_proportions * (scale / row_sums)
    if matrix_group is None:
        mapped = np.asarray(scaled_mass @ matrix).ravel()
    else:
        pooled = np.bincount(matrix_group, weights=scaled_mass, minlength=matrix.shape[0])
        mapped = np.asarray(pooled @ matrix).ravel()[matrix_group]
    correction = true_proportions * ((diagonal_effective - scale * diagonal) / row_sums)
    predicted = (mapped + correction if home is None
                 else mapped + np.bincount(home, correction, minlength=matrix.shape[1]))
    if not np.isfinite(predicted).all() or (predicted < -1e-12).any():
        raise SystemExit("mis-mapping projection produced invalid probabilities")
    predicted = np.clip(predicted, 0.0, None)
    return predicted / predicted.sum()


def _group_sum(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Sum one reference-space vector into unique-V4 groups.

    Args:
        values: Values in reference order.
        groups: Exact V4 group identifiers in the same order.

    Returns:
        Group-summed values in deterministic integer group order.
    """
    return np.bincount(groups, weights=values, minlength=int(groups.max()) + 1)


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
    groups: np.ndarray,
    concentration_fraction: np.ndarray | None,
    likelihood: str,
    rng: np.random.Generator,
) -> tuple[dict[str, float | dict[str, float]], dict[str, float | dict[str, float]]]:
    """Calculate raw and grouped predictive residuals from posterior draws.

    Args:
        observations: Observed MAPseq reference counts.
        predicted: One predicted reference distribution per posterior draw.
        groups: Exact V4 group identifiers in reference order.
        concentration_fraction: Dirichlet-multinomial concentration draws, if fitted.
        likelihood: Inference likelihood name.
        rng: Random generator for posterior-predictive replicates.

    Returns:
        Raw-reference and unique-V4-group diagnostic mappings.
    """
    total = int(observations.sum())
    observed_group = _group_sum(observations, groups)
    mean_predicted = predicted.mean(axis=0)
    mean_group_predicted = _group_sum(mean_predicted, groups)
    raw_observed = np.empty(len(predicted), dtype=np.float64)
    grouped_observed = np.empty(len(predicted), dtype=np.float64)
    raw_replicated = np.empty(len(predicted), dtype=np.float64)
    grouped_replicated = np.empty(len(predicted), dtype=np.float64)

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
        grouped_probabilities = _group_sum(probabilities, groups)
        grouped_replicated_counts = _group_sum(replicated, groups)
        raw_observed[draw_index] = _tv_distance(observations, probabilities)
        grouped_observed[draw_index] = _tv_distance(observed_group, grouped_probabilities)
        raw_replicated[draw_index] = _tv_distance(replicated, probabilities)
        grouped_replicated[draw_index] = _tv_distance(
            grouped_replicated_counts, grouped_probabilities
        )

    def diagnostic(
        observed_statistic: np.ndarray,
        replicated_statistic: np.ndarray,
        observed_expected_tv: float,
        labels: int,
    ) -> dict[str, float | dict[str, float]]:
        percentile = float(np.mean(replicated_statistic <= observed_statistic))
        return {
            "n_labels": labels,
            "observed_expected_tv": observed_expected_tv,
            "observed_statistic": _quantiles(observed_statistic),
            "replicated_statistic": _quantiles(replicated_statistic),
            "posterior_predictive_percentile": percentile,
        }

    return (
        diagnostic(raw_observed, raw_replicated, _tv_distance(observations, mean_predicted),
                   len(observations)),
        diagnostic(grouped_observed, grouped_replicated,
                   _tv_distance(observed_group, mean_group_predicted), len(observed_group)),
    )


def run(args: argparse.Namespace) -> None:
    """Run the fit check and write one machine-readable diagnostics document.

    Args:
        args: Parsed command-line arguments.
    """
    sample, composition_genomes, composition_means, composition_status = _read_composition(
        args.composition
    )
    posterior_genomes, draws, metadata = _read_posterior_draws(args.posterior_draws)
    # A panel kernel (sources x database labels): inference and check share one label space,
    # observed labels + sink + U, so the raw and grouped statistics coincide.
    rectangular = (args.mismapping_matrix.suffix == ".npz"
                   and sm.is_rectangular(args.mismapping_matrix))
    if rectangular:
        panel = ic._panel_context(args.mismapping_matrix, args.amplicon_dir, args.obs_mseq,
                                  args.min_identity, not metadata["use_mismapping"])
        genomes, observations = panel.genomes, panel.y
    else:
        genomes, references, reference_genomes, weights = _translation_weights(args.amplicon_dir)
        if metadata["infer_space"] == "v4_group":
            # The same translation inference fitted: one "genome" per exact V4 group.
            genomes, reference_genomes = ic._v4_groups(args.amplicon_dir, references)
            weights = 1.0 / np.bincount(reference_genomes)[reference_genomes]
    if composition_genomes != genomes or posterior_genomes != genomes:
        raise SystemExit("composition table and posterior draws must use the inference-space order")

    if not rectangular:
        matrix, matrix_group, strata = _matrix(args.mismapping_matrix, references)
        counts = si.observed_refseq_counts(args.obs_mseq, references, args.min_identity)
        observations = np.asarray([counts.get(reference, 0) for reference in references],
                                  dtype=np.float64)
        # The same identity-row fallback inference applied, so both see one matrix.
        matrix = ic._identity_rows_for_observed(matrix, matrix_group, observations)
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
        "infer_space": metadata["infer_space"],
        "kernel_format": "rectangular" if rectangular else "square",
        "n_reads": total,
        "min_infer_reads": args.min_infer_reads,
        "matrix_path": str(args.mismapping_matrix),
        "posterior_draw_path": str(args.posterior_draws),
        "raw_reference": None,
        "unique_v4_group": None,
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

        home = None
        if rectangular:
            matrix, home, matrix_group, strata = panel.fitted, panel.home, None, panel.strata
            local_reference_genomes, weights = panel.genome_of_row, panel.weight
            v4_group = np.arange(len(observations))
            genome_indices = np.arange(len(genomes))
        else:
            v4_group = (reference_genomes if metadata["infer_space"] == "v4_group"
                        else ic._v4_groups(args.amplicon_dir, references)[1])
            (
                matrix,
                matrix_group,
                strata,
                local_reference_genomes,
                weights,
                v4_group,
                genome_indices,
                reference_indices,
            ) = _subset_context(
                matrix,
                matrix_group,
                strata,
                references,
                reference_genomes,
                weights,
                observations,
                v4_group,
                metadata["use_active_subset"],
            )
            # Sink labels (the tail beyond the real references) drew no reads.
            observations = np.concatenate(
                [observations[reference_indices],
                 np.zeros(len(weights) - len(reference_indices))]
            )
        # A panel fit always samples s; --no-mismapping swapped K for home rows instead.
        use_kernel = metadata["use_mismapping"] or rectangular

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
        if not np.allclose(composition_means, theta.mean(axis=0), rtol=1e-6, atol=1e-8):
            raise SystemExit("composition inferred_mean values do not match retained theta_eff draws")
        theta = theta[draw_indices][:, genome_indices]
        if use_kernel and "s" not in draws:
            raise SystemExit("posterior draws lack s for a fit that used mis-mapping")
        scales = (
            draws["s"][draw_indices]
            if use_kernel
            else np.ones(draw_count, dtype=np.float64)
        )
        decays = None
        if metadata["infer_distance_decay"]:
            if "c" not in draws:
                raise SystemExit("posterior draws lack c for a fitted distance decay")
            decays = draws["c"][draw_indices]
        predicted = np.empty((draw_count, len(observations)), dtype=np.float64)
        for draw_index in range(draw_count):
            true_proportions = theta[draw_index, local_reference_genomes] * weights
            true_proportions = true_proportions / true_proportions.sum()
            if use_kernel:
                kernel = _kernel_at_decay(
                    matrix,
                    matrix_group,
                    strata,
                    None if decays is None else float(decays[draw_index]),
                )
                predicted[draw_index] = _apply_mismapping(
                    true_proportions, kernel, matrix_group, float(scales[draw_index]), home
                )
            else:
                predicted[draw_index] = true_proportions

        if metadata["likelihood"] == "dirichlet_multinomial" and "conc_frac" not in draws:
            raise SystemExit("posterior draws lack conc_frac for a Dirichlet-multinomial fit")
        concentration = (
            draws["conc_frac"][draw_indices]
            if metadata["likelihood"] == "dirichlet_multinomial"
            else None
        )
        raw, grouped = _posterior_predictive_diagnostics(
            observations,
            predicted,
            v4_group,
            concentration,
            str(metadata["likelihood"]),
            rng,
        )
        payload["posterior_draw_count"] = draw_count
        payload["raw_reference"] = raw
        payload["unique_v4_group"] = grouped
        # Only the V4-group check gates release: MAPseq's split of reads across
        # byte-identical references is not information the model claims to explain.
        payload["gate"] = "unique_v4_group"
        percentile = float(grouped["posterior_predictive_percentile"])
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
