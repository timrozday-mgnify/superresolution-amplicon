#!/usr/bin/env python
"""Standalone per-sample genome-composition inference for superresolution-amplicon.

Both observed signal and mis-mapping come from **mapseq**: the observed per-reference
read counts are the top hits of the real reads, and the mis-mapping matrix ``M`` is
measured by mapping simulated reads (whose names carry their source reference) through
the same mapper. Feeding both into the ``dirichlet_multinomial`` likelihood inverts the
confusion between (near-)identical references rather than ignoring it. Fitting defaults
to VI (posterior mean): the mode/MLE collapses weak components (e.g. a low-abundance
subspecies) to exactly 0, the mean does not.

A per-genome Bernoulli presence gate (on by default) answers the separate question of
whether a genome is in the sample at all. Its prior probability is a sparsity
regulariser, and its posterior probability is reported per genome as ``presence_prob`` —
the confidence in the presence call, not in the abundance.

    infer_composition.py \
        --amplicon-dir AMPLICON_DIR --sim-mseq sim.mseq --obs-mseq obs.mseq \
        --sample-id S1 --mode vi -o out/

To pre-compute a reusable matrix, omit ``--obs-mseq`` and use
``--build-mismapping``. Later inference can use the labelled CSR ``.npz`` through
``--mismapping-matrix`` instead of ``--sim-mseq``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse

_HERE = Path(__file__).resolve().parent
import sys
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import subspecies_infer as si  # noqa: E402  (needs sys.path)
import sparse_matrix as sm  # noqa: E402  (needs sys.path)

log = logging.getLogger("infer_composition")


def _fit_args(a) -> SimpleNamespace:
    """The attribute bag _fit / composition_model read off ``args``."""
    return SimpleNamespace(
        mode=a.mode, lr=a.lr, steps=a.steps, num_samples=a.num_samples,
        warmup=a.warmup, progress=False, use_mismapping=not a.no_mismapping,
        use_presence=not a.no_presence, presence_prior=a.presence_prior,
        presence_temp=a.presence_temp, seed=a.seed,
        decay_sigma=getattr(a, "decay_sigma", si.DEFAULT_DECAY_SIGMA),
    )


def _mismapping_matrix(a, refseqs: list[str]):
    """Return ``(M, group, strata)``: a measured or pre-computed matrix in reference order.

    ``M`` is a reference-square CSR with ``group`` ``None``, or the unique-amplicon kernel
    the duplicate-collapsing backends write for database-scale reference sets. ``strata``
    is the ``(distances, decay)`` pair a ``tau >= 1`` alignment build stores, or ``None``
    for every other source — only a build that recorded its distances can be re-decayed.
    """
    if getattr(a, "mismapping_matrix", None) is None:
        return si.build_mismapping(a.sim_mseq, refseqs, a.min_identity), None, None

    if a.mismapping_matrix.suffix == ".npz":
        try:
            if sm.is_grouped(a.mismapping_matrix):
                matrix, group, strata = sm.read_grouped(a.mismapping_matrix, refseqs,
                                                        strata=True)
                return matrix, group, strata
            matrix, strata = sm.read_matrix(a.mismapping_matrix, refseqs, strata=True)
            return matrix, None, strata
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    matrix = pd.read_csv(a.mismapping_matrix, index_col=0)
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise SystemExit(f"mis-mapping matrix {a.mismapping_matrix} has duplicate reference IDs")
    if set(matrix.index) != set(refseqs) or set(matrix.columns) != set(refseqs):
        raise SystemExit(
            f"mis-mapping matrix {a.mismapping_matrix} must have exactly the reference IDs "
            "in the translation table as both rows and columns"
        )
    matrix = matrix.loc[refseqs, refseqs]
    try:
        values = matrix.to_numpy(dtype=np.float64)
    except ValueError as exc:
        raise SystemExit(f"mis-mapping matrix {a.mismapping_matrix} contains non-numeric values") from exc
    if not np.isfinite(values).all() or (values < 0).any():
        raise SystemExit(f"mis-mapping matrix {a.mismapping_matrix} must contain finite, non-negative values")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise SystemExit(f"mis-mapping matrix {a.mismapping_matrix} must be row-stochastic")
    return sparse.csr_array(values), None, None


def _translation(amplicon_dir: Path):
    """Return compact genome-to-reference weights, with CSV support for old bundles."""
    compact = amplicon_dir / "translation_table.tsv"
    if compact.exists():
        table = pd.read_csv(compact, sep="\t")
        required = {"genome_id", "refseq", "weight"}
        if set(table.columns) != required or table["refseq"].duplicated().any():
            raise SystemExit(f"invalid compact translation table {compact}")
        genomes = sorted(table["genome_id"].unique())
        genome_index = {genome: index for index, genome in enumerate(genomes)}
        refs = table["refseq"].tolist()
        return genomes, refs, (
            np.asarray([genome_index[genome] for genome in table["genome_id"]], dtype=np.int64),
            table["weight"].to_numpy(dtype=np.float64),
            len(genomes),
        )
    legacy = pd.read_csv(amplicon_dir / "translation_table.csv", index_col=0)
    return list(legacy.index), list(legacy.columns), legacy.to_numpy(dtype=np.float64)


def _active_subset(M_sparse: sparse.csr_array, group: np.ndarray | None,
                   obs: np.ndarray, g_of_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference and genome indices worth fitting, given the observed counts.

    A reference is reachable when the mis-mapping matrix gives it a route to a reference
    that was actually observed: ``M[a, j] > 0`` for some ``j`` with a read on it. That is
    the "could this reference have produced one of these reads" question, and it answers
    the byte-identity case for free — identical amplicons share a row of ``M``, so a
    zero-count twin of a hit reference is reachable through the tie mapseq broke
    arbitrarily. The fuzzy backends widen it further, to the near-neighbours they let
    reads leak across, which is exactly the set they say a read might have come from.

    Then take the genomes owning that set, and every reference copy those genomes own:
    ``T`` is the within-genome copy distribution and has to stay complete, or a kept
    genome quietly loses the mass of its dropped copies.

    What is left out is dead weight — a reference no observed read can be traced to, and
    a genome none of whose copies were seen. Each is an empty row, an empty column, and a
    genome dimension whose likelihood is flat. Note the closure is one round, not a fixed
    point: a kept genome's unobserved copy may itself tie to a reference outside the set,
    and the mass leaking there is absorbed by the renormalisation in
    ``_subset_mismapping`` rather than pulling another genome in.
    """
    hit = np.flatnonzero(obs > 0)
    if group is None:
        reachable = np.zeros(len(obs), dtype=bool)
        reachable[M_sparse[:, hit].nonzero()[0]] = True
    else:
        # Grouped form: M[a, j] == M_sparse[group[a], group[j]], so reachability is a
        # question about the unique-amplicon kernel, scattered back over references.
        sources = np.unique(M_sparse[:, np.unique(group[hit])].nonzero()[0])
        reachable = np.isin(group, sources)
    gen_idx = np.unique(g_of_ref[reachable])
    return np.flatnonzero(np.isin(g_of_ref, gen_idx)), gen_idx


def _subset_translation(translation, ref_idx: np.ndarray, gen_idx: np.ndarray,
                        gen_pos: np.ndarray):
    """Restrict ``T`` to the kept genomes x kept references.

    Rows stay normalised without rescaling: every copy of a kept genome is kept.
    """
    if isinstance(translation, tuple):
        genome_of_row, weight, _ = translation
        return gen_pos[genome_of_row[ref_idx]], weight[ref_idx], len(gen_idx)
    return translation[np.ix_(gen_idx, ref_idx)]


def _renormalise_rows(matrix: sparse.csr_array, row_sums: np.ndarray) -> sparse.csr_array:
    """Scale each row by ``1 / row_sum``, leaving an all-zero row alone."""
    matrix = matrix.tocsr()
    divisor = np.where(row_sums > 0, row_sums, 1.0)
    matrix.data = matrix.data / np.repeat(divisor, np.diff(matrix.indptr))
    return matrix


def _subset_strata(distances: sparse.csr_array, idx: np.ndarray) -> sparse.csr_array:
    """Subset the distances with their matrix, keeping distance-0 entries explicit."""
    distances = distances.copy()
    distances.data += 1.0
    distances = distances[idx][:, idx].tocsr()
    distances.data -= 1.0
    return distances


def _subset_mismapping(M_sparse: sparse.csr_array, group: np.ndarray | None,
                       ref_idx: np.ndarray):
    """Restrict ``M`` to ``ref_idx`` and renormalise it back to row-stochastic.

    Dropping columns drops the mass a kept reference mis-maps onto a pruned one, so the
    rows no longer sum to 1 — and the sparse path in ``_apply_mismapping`` derives its
    off-diagonal mass as ``1 - diagonal`` rather than summing. Rescaling is the usual
    conditioning ("of the reads that landed inside the kept set..."), and the mass it
    redistributes is small by construction: a pruned reference drew no reads.
    """
    if group is None:
        sub = M_sparse[ref_idx][:, ref_idx].tocsr()
        return _renormalise_rows(sub, np.asarray(sub.sum(axis=1)).ravel()), None
    # Grouped form: subset the unique-amplicon kernel, not the reference square.
    keep_groups = np.unique(group[ref_idx])
    sub = M_sparse[keep_groups][:, keep_groups].tocsr()
    position = np.full(M_sparse.shape[0], -1, dtype=np.int64)
    position[keep_groups] = np.arange(len(keep_groups))
    new_group = position[group[ref_idx]]
    sizes = np.bincount(new_group, minlength=len(keep_groups)).astype(np.float64)
    return _renormalise_rows(sub, sub @ sizes), new_group


def run(a) -> None:
    import torch

    genomes, refseqs, translation = _translation(a.amplicon_dir)
    # dict, not genomes.index(): a linear scan per reference is O(n_refs x n_genomes),
    # which at database scale (GTDB SSU: ~1e5 of each) costs hours before inference starts.
    genome_index = {genome: index for index, genome in enumerate(genomes)}
    g_of_ref = np.array([genome_index[si.genome_of_header(r)] for r in refseqs])

    # M is either measured from simulated mapseq output or loaded from a prior run.
    M_sparse, group, strata = _mismapping_matrix(a, refseqs)

    # The decay is only a latent where the matrix says what distance each nonzero came
    # from, and only *identified* where those distances differ within a row. Refusing is
    # better than silently fitting a parameter the likelihood is flat in.
    infer_decay = getattr(a, "infer_distance_decay", False)
    if infer_decay:
        if strata is None:
            raise SystemExit(
                "--infer-distance-decay needs a mis-mapping matrix built with distance "
                "strata: --mismapping_method align at --align_tau >= 1 (a simulate-built, "
                "CSV or tau=0 matrix carries no distances to re-decay)")
        if not strata[0].data.any():
            raise SystemExit(
                "--infer-distance-decay: every distance in this matrix is 0, so the decay "
                "cancels and the fit would be flat in it. Build at --align_tau >= 1.")

    if getattr(a, "build_mismapping", False):
        a.output_dir.mkdir(parents=True, exist_ok=True)
        out = a.output_dir / "mismapping_matrix.npz"
        if group is None:
            sm.write_matrix(out, M_sparse, refseqs)
        else:
            sm.write_grouped(out, M_sparse, group, refseqs)
        print(f"mismapping: {len(refseqs)} references -> {a.output_dir}")
        return

    # Of the whole matrix, not the pruned one: a property of the reference set, so it
    # stays comparable between samples that prune to different sizes.
    mean_diagonal = sm.matrix_diagonal_mean(M_sparse)

    counts = si.observed_refseq_counts(a.obs_mseq, refseqs, a.min_identity)
    obs = np.array([counts.get(r, 0) for r in refseqs], dtype=np.float64)
    total = int(obs.sum())
    # A sample whose reads hit no reference amplicon is a result, not an error: it is
    # reported as an all-zero composition flagged `status=no_reference_hits` (with
    # `n_reads=0`) in the same files a normal run writes, so a multi-sample run is not
    # taken down by one empty sample and the outcome is unambiguous downstream.
    no_hits = total == 0
    if no_hits:
        log.warning("no reads in %s hit a reference amplicon; emitting a zero composition",
                    a.obs_mseq)
    log.info("sample %s: %d mapped reads, %d refs, %d genomes, mode=%s",
             a.sample_id, total, len(refseqs), len(genomes), a.mode)

    # Fit only what the reads can speak to. The reference set is the whole database
    # (GTDB SSU: ~1e5 amplicons over ~1e5 genomes) while a sample touches a few hundred
    # of them; the rest is an empty row, an empty column and a flat genome dimension.
    # `all_genomes` keeps the output contract — every genome in the set is reported, a
    # pruned one at zero.
    all_genomes, gen_idx = genomes, None
    if not no_hits and not getattr(a, "no_prune", False):
        ref_idx, gen_idx = _active_subset(M_sparse, group, obs, g_of_ref)
        if len(gen_idx) == len(genomes):
            gen_idx = None                       # nothing to prune; keep the fast path
        else:
            log.info("pruned to %d/%d references and %d/%d genomes with observed support",
                     len(ref_idx), len(refseqs), len(gen_idx), len(genomes))
            gen_pos = np.full(len(genomes), -1, dtype=np.int64)
            gen_pos[gen_idx] = np.arange(len(gen_idx))
            translation = _subset_translation(translation, ref_idx, gen_idx, gen_pos)
            kernel_idx = (ref_idx if group is None else np.unique(group[ref_idx]))
            M_sparse, group = _subset_mismapping(M_sparse, group, ref_idx)
            if strata is not None:
                strata = (_subset_strata(strata[0], kernel_idx), strata[1])
            refseqs = [refseqs[i] for i in ref_idx]
            genomes = [genomes[i] for i in gen_idx]
            g_of_ref = gen_pos[g_of_ref[ref_idx]]
            obs = obs[ref_idx]

    if isinstance(translation, tuple):
        T = (
            torch.tensor(translation[0], dtype=torch.long),
            torch.tensor(translation[1], dtype=torch.float64),
            translation[2],
        )
    else:
        T = torch.tensor(translation, dtype=torch.float64)
    if infer_decay:
        # M becomes a function of the sampled decay instead of a fixed matrix.
        M = si.DecayKernel(M_sparse, strata[0], strata[1], group)
    else:
        coo = torch.sparse_coo_tensor(
            torch.tensor(np.vstack(M_sparse.nonzero()), dtype=torch.long),
            torch.tensor(M_sparse.data, dtype=torch.float64),
            size=M_sparse.shape,
        ).coalesce()
        if group is None:
            M = (coo, torch.tensor(M_sparse.diagonal(), dtype=torch.float64))
        else:
            M = (coo,
                 torch.tensor(sm.grouped_diagonal(M_sparse, group), dtype=torch.float64),
                 torch.tensor(group, dtype=torch.long))

    ref_rel = obs / total if total else obs

    # Observed per-ref signal collapsed to genome-space -> observed composition (init + baseline).
    theta_obs = np.zeros(len(genomes))
    np.add.at(theta_obs, g_of_ref, ref_rel)

    lo = hi = np.full(len(genomes), np.nan)
    if no_hits:
        samples, losses, diag = None, None, {}
        inferred = np.zeros(len(genomes))
    else:
        theta_init = torch.tensor(np.clip(theta_obs, 1e-4, None), dtype=torch.float64)
        theta_init = theta_init / theta_init.sum()

        fa = _fit_args(a)
        samples, point, diag, losses = si._fit(
            a.mode, M, T, a.alpha, float(total), torch.tensor(ref_rel, dtype=torch.float64),
            "dirichlet_multinomial", theta_init, fa, desc=a.sample_id)
        inferred = point.numpy()
        if samples is not None:
            lo = torch.quantile(samples, 0.05, dim=0).numpy()
            hi = torch.quantile(samples, 0.95, dim=0).numpy()
    # Posterior probability that each genome is present at all (the Bernoulli gate);
    # NaN when the gate is off, so "no call" is never confused with "called absent".
    presence = np.array(diag.pop("presence_prob", [np.nan] * len(genomes)))

    # Back to the full genome list. A pruned genome had no read pointing at it under any
    # resolution of the ties, so it is reported absent rather than omitted — downstream
    # scoring compares against a truth over the whole reference set.
    def expand(values, fill):
        if gen_idx is None:
            return np.asarray(values, dtype=np.float64)
        full = np.full(len(all_genomes), fill, dtype=np.float64)
        full[gen_idx] = values
        return full

    interval_fill = np.nan if samples is None else 0.0
    theta_obs, inferred = expand(theta_obs, 0.0), expand(inferred, 0.0)
    lo, hi = expand(lo, interval_fill), expand(hi, interval_fill)
    presence = expand(presence, np.nan if a.no_presence else 0.0)

    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.DataFrame([{
        "sample": a.sample_id, "genome_id": g,
        "observed_rel_abundance": float(theta_obs[i]),
        "inferred_mean": float(inferred[i]),
        "inferred_lo": float(lo[i]), "inferred_hi": float(hi[i]),
        "presence_prob": float(presence[i]),
    } for i, g in enumerate(all_genomes)])
    comp.to_csv(out / "inferred_composition.csv", index=False)
    pd.DataFrame([{
        "sample": a.sample_id, "mode": a.mode, "likelihood": "dirichlet_multinomial",
        "use_mismapping": not a.no_mismapping, "min_identity": a.min_identity,
        "use_presence": not a.no_presence, "presence_prior": a.presence_prior,
        "presence_temp": a.presence_temp,
        "infer_distance_decay": infer_decay,
        "built_distance_decay": None if strata is None else strata[1],
        "mismapping_group_id": getattr(a, "mismapping_group_id", None),
        "mismapping_matrix_path": getattr(a, "mismapping_matrix_path", None),
        "n_reads": int(total), "mean_diagonal": mean_diagonal,
        "n_refs_fitted": len(refseqs), "n_genomes_fitted": len(genomes),
        "n_genomes": len(all_genomes),
        "status": "no_reference_hits" if no_hits else "ok",
        **{k: json.dumps(v) for k, v in diag.items()},
    }]).to_csv(out / "inference_diagnostics.csv", index=False)
    if losses is not None:
        pd.DataFrame([{"sample": a.sample_id, "step": s, "loss": l}
                      for s, l in enumerate(losses)]).to_csv(out / "loss_trace.csv", index=False)
    print(f"infer[{a.mode}]: sample {a.sample_id}, {len(all_genomes)} genomes -> {out}")


def demo_prune() -> None:
    """Self-check: pruning drops only genomes no read can reach, and the grouped
    subset stays row-stochastic.

    Genomes [uni, strain2, ghost]; refs [uni|0(A), strain2|0(A), strain2|1(B),
    ghost|0(C)]. Every read maps to strain2's two refs — none to uni|0, none to ghost|0.
    ``ghost`` must be pruned (no read reaches C under any assignment) and reported at
    zero, while ``uni`` must survive: its amplicon is byte-identical to strain2|0, so
    mapseq's tie-break is the only reason it shows no counts — and ``M``, built here from
    the simulated reads, is where that tie is recorded. Pruning uni would delete exactly
    the case the mis-mapping correction exists for.
    """
    import tempfile

    refs = ["uni|0|A", "strain2|0|A", "strain2|1|B", "ghost|0|C"]
    with tempfile.TemporaryDirectory() as temporary:
        td = Path(temporary)
        amp = td / "amp"
        amp.mkdir()
        pd.DataFrame({
            "genome_id": ["uni", "strain2", "strain2", "ghost"],
            "refseq": refs,
            "weight": [1.0, 0.5, 0.5, 1.0],
        }).to_csv(amp / "translation_table.tsv", sep="\t", index=False)
        # Simulated reads: the identical pair splits 50:50, the distinct refs self-hit.
        with open(td / "sim.mseq", "w") as fh:
            for source, hits in enumerate([[0, 1], [0, 1], [2], [3]]):
                for i in range(100):
                    fh.write(f"{refs[source]}:{i}\t{refs[hits[i % len(hits)]]}\t500\t0.99\n")
        # Observed: strain2's refs only. uni|0 is invisible behind the tie; ghost|0 is
        # genuinely absent.
        with open(td / "obs.mseq", "w") as fh:
            for j, n in [(1, 3000), (2, 3000)]:
                for i in range(n):
                    fh.write(f"read{j}_{i}\t{refs[j]}\t500\t0.99\n")

        args = argparse.Namespace(
            amplicon_dir=amp, sim_mseq=[td / "sim.mseq"], obs_mseq=[td / "obs.mseq"],
            min_identity=None, sample_id="prune", mode="vi", alpha=0.5, steps=500,
            lr=0.05, num_samples=100, warmup=0, no_mismapping=False, seed=0,
            no_presence=True, presence_prior=si.DEFAULT_PRESENCE_PRIOR,
            presence_temp=si.DEFAULT_PRESENCE_TEMP, no_prune=False,
            output_dir=td / "out")
        run(args)
        got = pd.read_csv(td / "out" / "inferred_composition.csv").set_index("genome_id")
        diag = pd.read_csv(td / "out" / "inference_diagnostics.csv").iloc[0]

        assert set(got.index) == {"uni", "strain2", "ghost"}, got    # full set reported
        assert diag["n_genomes_fitted"] == 2 and diag["n_genomes"] == 3, diag
        assert diag["n_refs_fitted"] == 3, diag                      # C dropped, A/A/B kept
        assert got.loc["ghost", "inferred_mean"] == 0.0, got
        assert got.loc["uni", "inferred_mean"] > 0.0, got            # survived the tie
        assert abs(got.loc[["uni", "strain2"], "inferred_mean"].sum() - 1.0) < 1e-6, got

        # Same fit without pruning: the survivors land in the same place. Not bit
        # identical — the Dirichlet prior is over the fitted genomes, so dropping one
        # redistributes a little prior mass — but far inside the gap a real pruning bug
        # would open.
        args.no_prune, args.output_dir = True, td / "out_full"
        run(args)
        full = pd.read_csv(td / "out_full" / "inferred_composition.csv").set_index("genome_id")
        assert (got["inferred_mean"] - full["inferred_mean"]).abs().max() < 0.05, (got, full)

    # Grouped matrices are subset through their unique-amplicon kernel; the result has to
    # stay row-stochastic over the references it keeps, or _apply_mismapping's
    # `1 - diagonal` off-diagonal mass is wrong.
    group = np.array([0, 0, 1, 2])
    cluster = sparse.csr_array(np.array([[0.5, 0.0, 0.0], [0.1, 0.8, 0.1], [0.0, 0.2, 0.8]]))
    kept = np.array([0, 1, 2])                       # drop ghost|0, the only member of 2
    sub, sub_group = _subset_mismapping(cluster, group, kept)
    sizes = np.bincount(sub_group).astype(float)
    assert np.allclose(sub @ sizes, 1.0), (sub.toarray(), sizes)
    assert sub_group.tolist() == [0, 0, 1], sub_group
    print("prune demo OK")


def demo_decay() -> None:
    """Self-check the latent distance decay end to end, through a stored matrix.

    14 references over 6 genomes and 10 distinct amplicons, four pairs of which are one
    edit apart. Two things about that shape matter:

    * a row holding an exact duplicate *and* a one-edit neighbour is the only place ``c``
      changes anything ``s`` cannot — ``s`` rescales a row's off-diagonal mass, ``c``
      reshapes it across distance classes;
    * there are more references than genomes, so ``theta`` cannot absorb the difference.
      On a set where it can (references ~ genomes), ``c`` is not identified and the fit
      leans on its prior — which is the honest failure mode, not a bug.
    """
    import tempfile

    owner = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 3]     # genome of each reference
    amplicon = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6]  # distinct amplicon it carries
    refs = [f"g{genome}|{i}|x" for i, genome in enumerate(owner)]
    group = np.array(amplicon)
    sizes = np.bincount(group, minlength=10).astype(np.float64)
    truth = np.array([0.30, 0.25, 0.20, 0.12, 0.09, 0.04])
    c0, c_true = 0.01, 0.3

    distance = np.zeros((10, 10))
    pattern = np.eye(10, dtype=bool)
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:          # one-edit neighbours
        distance[i, j] = distance[j, i] = 1.0
        pattern[i, j] = pattern[j, i] = True

    def kernel(c):
        raw = np.where(pattern, c ** distance, 0.0)
        return sparse.csr_array(raw / (raw @ sizes)[:, None])

    distances = sparse.csr_array(np.where(pattern, distance + 1.0, 0.0))
    distances.data -= 1.0                     # zeros stay explicit: M's own pattern

    with tempfile.TemporaryDirectory() as temporary:
        td = Path(temporary)
        amp = td / "amp"
        amp.mkdir()
        pd.DataFrame({
            "genome_id": [f"g{genome}" for genome in owner], "refseq": refs,
            "weight": [1.0 / owner.count(genome) for genome in owner],
        }).to_csv(amp / "translation_table.tsv", sep="\t", index=False)

        # Observed reads drawn through the matrix the *sample* experienced, which is not
        # the one the matrix was built with: 30x the leak the build assumed.
        r_true = np.zeros(len(refs))
        for reference, genome in enumerate(owner):
            r_true[reference] = truth[genome] / owner.count(genome)
        r_obs = r_true @ kernel(c_true).toarray()[np.ix_(group, group)]
        counts = (100000 * r_obs / r_obs.sum()).round().astype(int)
        with open(td / "obs.mseq", "w") as fh:
            for reference, n in enumerate(counts):
                for i in range(n):
                    fh.write(f"read{reference}_{i}\t{refs[reference]}\t500\t0.99\n")

        sm.write_grouped(td / "M.npz", kernel(c0), group, refs, (distances, c0))
        sm.write_grouped(td / "plain.npz", kernel(c0), group, refs)

        def args(**over):
            return argparse.Namespace(**{
                "amplicon_dir": amp, "sim_mseq": None, "mismapping_matrix": td / "M.npz",
                "obs_mseq": [td / "obs.mseq"], "min_identity": None,
                "sample_id": "decay", "mode": "vi", "alpha": 0.5, "steps": 1500,
                "lr": 0.05, "num_samples": 200, "warmup": 0, "no_mismapping": False,
                "seed": 0, "no_presence": True,
                "presence_prior": si.DEFAULT_PRESENCE_PRIOR,
                "presence_temp": si.DEFAULT_PRESENCE_TEMP, "no_prune": False,
                "infer_distance_decay": True,
                "decay_sigma": si.DEFAULT_DECAY_SIGMA, "output_dir": td / "out", **over})

        def composition(where):
            return pd.read_csv(where / "inferred_composition.csv").set_index(
                "genome_id").loc[[f"g{i}" for i in range(6)], "inferred_mean"].to_numpy()

        # The fit must recover the decay the reads actually experienced, from a matrix
        # built 30x away from it, and beat the fixed decay on composition for doing so.
        run(args())
        fitted = float(pd.read_csv(td / "out" / "inference_diagnostics.csv"
                                   ).iloc[0]["distance_decay"])
        assert 0.5 * c_true < fitted < 2.0 * c_true, fitted
        latent_l1 = float(np.abs(composition(td / "out") - truth).sum())

        run(args(infer_distance_decay=False, output_dir=td / "fixed"))
        fixed_l1 = float(np.abs(composition(td / "fixed") - truth).sum())
        assert latent_l1 < fixed_l1 / 2, (latent_l1, fixed_l1)

        # Pinned to the built decay (a near-zero prior width) it must reproduce the fixed
        # path instead — same matrix, same answer, however it got there.
        run(args(decay_sigma=1e-4, output_dir=td / "pinned"))
        pinned = pd.read_csv(td / "pinned" / "inference_diagnostics.csv").iloc[0]
        assert abs(float(pinned["distance_decay"]) - c0) < 1e-3, pinned["distance_decay"]
        assert pinned["built_distance_decay"] == c0, pinned
        assert np.abs(composition(td / "pinned") - composition(td / "fixed")).max() < 0.02

        # A matrix that never recorded its distances cannot be re-decayed, and saying so
        # beats fitting a parameter nothing in the likelihood constrains.
        try:
            run(args(mismapping_matrix=td / "plain.npz", output_dir=td / "nope"))
        except SystemExit as exc:
            assert "distance strata" in str(exc), exc
        else:
            raise AssertionError("--infer-distance-decay must refuse a strata-less matrix")

    print(f"decay demo OK: recovered c={fitted:.3f} (built at {c0}, sample at {c_true}), "
          f"L1 {latent_l1:.4f} vs {fixed_l1:.4f} at the built decay")


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

    # Gate off: this half is testing the mis-mapping inversion in isolation.
    fa = SimpleNamespace(mode="mle", lr=0.05, steps=3000, num_samples=1, warmup=0,
                         progress=False, use_mismapping=True, use_presence=False, seed=0)
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
        pd.DataFrame({
            "genome_id": ["uni", "strain2", "strain2"],
            "refseq": refs,
            "weight": [1.0, 0.5, 0.5],
        }).to_csv(amp / "translation_table.tsv", sep="\t", index=False)
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
            min_identity=None, sample_id="demo", mode="vi", alpha=0.5, steps=2000,
            lr=0.05, num_samples=300, warmup=0, no_mismapping=False, seed=0,
            no_presence=True, presence_prior=si.DEFAULT_PRESENCE_PRIOR,
            presence_temp=si.DEFAULT_PRESENCE_TEMP, output_dir=td / "out")
        run(args)
        got = pd.read_csv(td / "out" / "inferred_composition.csv").set_index("genome_id")
        assert abs(got.loc["uni", "inferred_mean"] - 0.05) < 0.03, got
        assert got.loc["uni", "observed_rel_abundance"] > 0.2, got   # naive is wrong
        assert got["presence_prob"].isna().all(), got                # no call when gate off
        assert not (td / "out" / "mismapping_matrix.npz").exists()

        # Same sample with the gate on. This is the *worst* case for a presence call:
        # uni's only amplicon is byte-identical to a copy of strain2, so "uni present at
        # 5%" and "uni absent, strain2 slightly commoner" fit the reads equally well.
        # Presence is therefore not identifiable and the prior decides — strictly at the
        # default, present under a permissive prior. Abundance stays recoverable either
        # way (the ref-2 anchor still pins it), which is the point worth guarding: a low
        # presence_prob here means "can't tell", not "the abundance is wrong".
        def gated(prior, out):
            args.no_presence, args.presence_prior, args.output_dir = False, prior, td / out
            run(args)
            return pd.read_csv(td / out / "inferred_composition.csv").set_index("genome_id")

        strict = gated(si.DEFAULT_PRESENCE_PRIOR, "out_gated")
        loose = gated(0.9, "out_gated_loose")
        assert strict.loc["strain2", "presence_prob"] > si.PRESENCE_THRESHOLD, strict
        assert strict.loc["uni", "presence_prob"] < si.PRESENCE_THRESHOLD, strict
        assert loose.loc["uni", "presence_prob"] > si.PRESENCE_THRESHOLD, loose
        assert abs(loose.loc["uni", "inferred_mean"] - 0.05) < 0.03, loose
        print(f"gated (identical amplicon): p(uni) = "
              f"{strict.loc['uni', 'presence_prob']:.3f} at prior "
              f"{si.DEFAULT_PRESENCE_PRIOR:g} -> {loose.loc['uni', 'presence_prob']:.3f} "
              f"at prior 0.9")
    print("end-to-end mseq demo OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicon-dir", type=Path,
                    help="dir with translation_table.tsv (from `subspecies_infer.py amplicons`)")
    ap.add_argument("--sim-mseq", type=Path, nargs="+",
                    help="mapseq output for simulated reads (required unless --mismapping-matrix is used)")
    ap.add_argument("--mismapping-matrix", type=Path,
                    help="pre-computed row-stochastic CSV matrix; skips simulated-read mapping")
    ap.add_argument("--build-mismapping", action="store_true",
                    help="write labelled CSR mismapping_matrix.npz from --sim-mseq and exit")
    ap.add_argument("--obs-mseq", type=Path, nargs="+",
                    help="mapseq output for this sample's real reads")
    ap.add_argument("--min-identity", type=float, default=None,
                    help="drop mapseq hits below this pairwise identity (off-target background)")
    ap.add_argument("--sample-id", default="sample")
    ap.add_argument("--mismapping-group-id", default=None,
                    help="canonical matrix bundle identifier for diagnostics")
    ap.add_argument("--mismapping-matrix-path", default=None,
                    help="output-relative canonical matrix path for diagnostics")
    ap.add_argument("--mode", choices=["nuts", "vi", "mle"], default="vi")
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet prior concentration")
    ap.add_argument("--no-mismapping", action="store_true",
                    help="disable the mis-mapping correction (r_obs=r_true baseline)")
    ap.add_argument("--no-presence", action="store_true",
                    help="disable the presence/absence gate (every genome always present)")
    ap.add_argument("--presence-prior", type=float, default=si.DEFAULT_PRESENCE_PRIOR,
                    help="prior probability a genome is present; the sparsity regulariser "
                         "(smaller => stronger pull towards absent)")
    ap.add_argument("--presence-temp", type=float, default=si.DEFAULT_PRESENCE_TEMP,
                    help="Concrete relaxation temperature for the presence gate")
    ap.add_argument("--infer-distance-decay", action="store_true",
                    help="fit the tie-cluster distance decay c per sample instead of "
                         "taking the one the matrix was built with. Needs an alignment "
                         "matrix built at --align_tau >= 1 (it carries the distance behind "
                         "each nonzero); the matrix itself is not rebuilt, so one build "
                         "serves samples whose error rates differ.")
    ap.add_argument("--decay-sigma", type=float, default=si.DEFAULT_DECAY_SIGMA,
                    help="prior width in logs of that decay, centred on the built one")
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=3000, help="SVI steps (vi/mle)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output-dir", type=Path)
    ap.add_argument("--no-prune", action="store_true",
                    help="fit every genome in the reference set, not just those with "
                         "observed support (slow at database scale)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if a.demo:
        demo()
        demo_prune()
        return demo_decay()
    if a.sim_mseq and a.mismapping_matrix:
        ap.error("--sim-mseq and --mismapping-matrix are mutually exclusive")
    required = ["amplicon_dir", "output_dir"]
    if a.mismapping_matrix is None:
        required.append("sim_mseq")
    if not a.build_mismapping:
        required.append("obs_mseq")
    for req in required:
        if getattr(a, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (unless --demo)")
    run(a)


if __name__ == "__main__":
    main()
