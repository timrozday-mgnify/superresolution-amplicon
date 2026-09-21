#!/usr/bin/env python
"""Standalone per-sample panel-composition inference for superresolution-amplicon.

The observed signal is the MAPseq top hit of each real read, labelled with its database
V4 group. The panel kernel ``K`` (panel sources x database labels, from
``build_panel_kernel.py``) says where a read from each panel source lands. Feeding both
into the ``dirichlet_multinomial`` likelihood inverts the confusion between (near-)
identical sources rather than ignoring it. Fitting defaults to VI (posterior mean): the
mode/MLE collapses weak components (e.g. a low-abundance subspecies) to exactly 0, the
mean does not.

A per-genome Bernoulli presence gate (on by default) answers the separate question of
whether a genome is in the sample at all. Its prior probability is a sparsity
regulariser, and its posterior probability is reported per genome as ``presence_prob`` —
the confidence in the presence call, not in the abundance.

    infer_composition.py \
        --amplicon-dir PANEL_PREP_DIR --mismapping-matrix mismapping_matrix.npz \
        --obs-mseq obs.mseq --sample-id S1 --mode vi -o out/
"""
from __future__ import annotations

import argparse
import json
import logging
import re
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
        use_horseshoe=getattr(a, "horseshoe", False),
        s_sigma=getattr(a, "s_sigma", 0.3),
    )


_RANK_PREFIX = re.compile(r"^[a-z]+__")


def _plain_lineage(lineage: str) -> str:
    """An AAP-form lineage (``sk__Bacteria;k__;...;s__Genus_species``) in the plain form
    SILVA's own .tax uses: prefixes dropped, the species as ``Genus species``, an empty
    rank ``unclassified``. A lineage without ``__`` is returned unchanged."""
    if "__" not in lineage:
        return lineage
    ranks = []
    for rank in lineage.split(";"):
        name = _RANK_PREFIX.sub("", rank.strip())
        if rank.strip().startswith("s__"):
            name = name.replace("_", " ")
        ranks.append(name or "unclassified")
    return ";".join(ranks)


def _read_taxonomy(path: Path) -> dict[str, str]:
    """MAPseq ``.tax`` (``header<TAB>lineage``, ``#`` comments) -> header: lineage.

    Lineages stay unsplit: GTDB has ~1e6 of them and only fitted groups need ranks.
    AAP's SILVA-SSU-tax.txt reads as the plain SILVA form (_plain_lineage), so panel taxa
    and ``lca`` use bare names whichever tax file the database ships.
    """
    lineages = {}
    with open(path) as handle:
        for line in handle:
            if line.startswith("#") or "\t" not in line:
                continue
            header, lineage = line.rstrip("\n").split("\t", 1)
            lineages[header] = _plain_lineage(lineage)
    return lineages


def _lca(lineages: list[str]) -> str:
    """Longest common rank prefix, or ``unclassified_v4_group`` when there is none."""
    prefix = []
    for ranks in zip(*(lineage.split(";") for lineage in lineages)):
        if len(set(ranks)) != 1:
            break
        prefix.append(ranks[0])
    return ";".join(prefix) if lineages and prefix else "unclassified_v4_group"


def _tax_header(path: Path) -> tuple[list[str], list[float]]:
    """MAPseq ``.tax`` ``#levels:`` names and ``#cutoff:`` combined-confidence cutoffs."""
    levels, cutoffs = [], []
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            if key == "levels":
                levels = value.split()
            elif key == "cutoff":
                cutoffs = [float(pair.split(":")[0]) for pair in value.split()]
    return levels, cutoffs


# ponytail: fixed; a clade is confident when its summed draws stay within ±50% of its mean.
_LCA_TOLERANCE = 0.5


def _uncertainty_outputs(a, ids: list[str], draws: np.ndarray, n_reads: int,
                         lineage_of: dict[int, str] | None = None) -> None:
    """Write ``ambiguity_pairs.csv``, ``ambiguity_sets.csv`` and ``lca_composition.csv``.

    Ambiguity is posterior covariance the reads do not explain. Entities the data cannot
    tell apart trade mass across draws, so their sum is tighter than its members:
    ``Var(a) + Var(b) - Var(a + b) = -2 Cov(a, b)``. The simplex alone contributes
    ``-theta_a theta_b / (N + 1)`` (a Dirichlet posterior over ``N`` reads), removed first.
    What remains is ``ambiguous_sd`` (abundance units); over the members' summed variance
    it is ``gain`` (1 = only the sum is known, 0 = members resolved).

    Pairs flatten that to entity pairs, and a set's ambiguous variance is the sum of its
    pairs'. Sets merge whichever two clusters have the highest gain *between their sums*
    until none reaches ``--ambiguity-min-gain``: a k-way tie has pairwise gain 1/(k-1) but
    merged gain 1, so sets recover ties the pair list dilutes. Rows below the gain or
    ``--ambiguity-min-sd`` are dropped as noise.

    The LCA table follows MAPseq's taxonomy report: each rank of an entity's lineage gets a
    confidence (here the posterior probability that the clade's summed abundance is within
    ±50% of its mean), the lineage is cut at the first rank below that level's ``#cutoff:``
    (or ``--lca-cutoff``), and entities are summed per cut lineage. Needs a taxonomy.
    """
    keep = np.flatnonzero(draws.any(axis=0)) if len(draws) else np.empty(0, dtype=np.int64)
    if len(draws) < 2 or len(keep) == 0:
        return                              # mle, no hits, low depth: no posterior spread
    X, names = draws[:, keep], [ids[i] for i in keep]
    mean = X.mean(axis=0)
    cov = np.atleast_2d(np.cov(X, rowvar=False))
    var = np.diag(cov).copy()
    excess = -(cov + np.outer(mean, mean) / (n_reads + 1))
    min_gain = getattr(a, "ambiguity_min_gain", 0.2)
    min_sd = getattr(a, "ambiguity_min_sd", 1e-3)
    out = a.output_dir

    def summary(columns):
        total = X[:, columns].sum(axis=1)
        lo, hi = np.quantile(total, [0.05, 0.95])
        return {"inferred_mean": float(total.mean()), "inferred_lo": float(lo),
                "inferred_hi": float(hi)}

    ia, ib = np.triu_indices(len(keep), 1)
    pair_var, pair_denominator = 2 * excess[ia, ib], var[ia] + var[ib]
    pair_gain = np.divide(pair_var, pair_denominator, out=np.zeros_like(pair_var),
                          where=pair_denominator > 0)
    pair_sd = np.sqrt(np.clip(pair_var, 0, None))
    hit = (pair_gain >= min_gain) & (pair_sd >= min_sd)
    pd.DataFrame({
        "sample": a.sample_id, "id_a": [names[i] for i in ia[hit]],
        "id_b": [names[i] for i in ib[hit]], "mean_a": mean[ia[hit]],
        "mean_b": mean[ib[hit]], "ambiguous_sd": pair_sd[hit], "gain": pair_gain[hit],
    }).sort_values("ambiguous_sd", ascending=False).to_csv(out / "ambiguity_pairs.csv",
                                                           index=False)

    # ponytail: O(K^3) greedy merge over active entities; fine to a few thousand.
    clusters = [[i] for i in range(len(keep))]
    S, E = cov.copy(), excess.copy()          # per cluster: blocks summed over members
    while len(clusters) > 1:
        v = np.diag(S)
        denominator = v[:, None] + v[None, :]
        gain = np.divide(2 * E, denominator, out=np.zeros_like(E), where=denominator > 0)
        np.fill_diagonal(gain, -np.inf)
        p, q = sorted(np.unravel_index(np.argmax(gain), gain.shape))
        if gain[p, q] < min_gain:
            break
        for block in (S, E):
            block[p] += block[q]
            block[:, p] += block[:, q]
        S, E = (np.delete(np.delete(block, q, 0), q, 1) for block in (S, E))
        clusters[p] += clusters.pop(q)
    rows = []
    for members in clusters:
        m = np.sort(members)
        block = excess[np.ix_(m, m)]
        ambiguous = block.sum() - np.trace(block)
        if len(m) < 2 or np.sqrt(max(ambiguous, 0.0)) < min_sd:
            continue
        rows.append({
            "sample": a.sample_id, "n_members": len(m),
            "members": ";".join(names[i] for i in m),
            "lca": (_lca([lineage_of[keep[i]] for i in m if keep[i] in lineage_of])
                    if lineage_of is not None else None),
            **summary(m), "member_sd": float(np.sqrt(var[m].sum())),
            "ambiguous_sd": float(np.sqrt(ambiguous)), "gain": float(ambiguous / var[m].sum()),
        })
    rows.sort(key=lambda row: -row["ambiguous_sd"])
    pd.DataFrame([{"set_id": f"set_{n}", **row} for n, row in enumerate(rows)],
                 columns=["set_id", "sample", "n_members", "members", "lca", "inferred_mean",
                          "inferred_lo", "inferred_hi", "member_sd", "ambiguous_sd", "gain"]
                 ).to_csv(out / "ambiguity_sets.csv", index=False)

    if lineage_of is None:
        return
    levels, header_cutoffs = _tax_header(a.taxonomy)
    lca_cutoff = getattr(a, "lca_cutoff", None)

    def cutoff(depth):
        if lca_cutoff is not None:
            return lca_cutoff
        if any(header_cutoffs) and depth < len(header_cutoffs):
            return header_cutoffs[depth]
        return 0.5                            # an all-zero #cutoff: would never cut

    paths, rank_names = [], {}
    for column, (k, name) in enumerate(zip(keep, names)):
        lineage = lineage_of.get(k, "")
        ranks = [] if lineage in ("", "unclassified_v4_group") else lineage.split(";")
        for depth in range(len(ranks)):
            rank_names[tuple(ranks[:depth + 1])] = (levels[depth] if depth < len(levels)
                                                    else f"rank_{depth + 1}")
        if not ranks or ranks[-1] != name:    # the fitted entity is the leaf
            ranks.append(name)
            rank_names[tuple(ranks)] = "genome"
        paths.append(tuple(ranks))
    clade = {}
    for column, path in enumerate(paths):
        for depth in range(1, len(path) + 1):
            clade[path[:depth]] = clade.get(path[:depth], 0) + X[:, column]
    confidence = {}
    for node, total in clade.items():
        mu = total.mean()
        confidence[node] = float(np.mean(np.abs(total - mu) <= _LCA_TOLERANCE * mu)) if mu else 0.0

    assigned = {}
    for column, path in enumerate(paths):
        depth = 0
        while depth < len(path) and confidence[path[:depth + 1]] >= cutoff(depth):
            depth += 1
        assigned.setdefault(path[:depth], []).append(column)
    pd.DataFrame([{
        "sample": a.sample_id, "lca": ";".join(node) or "unassigned",
        "rank": rank_names.get(node, "root"), "n_members": len(columns),
        "members": ";".join(names[i] for i in columns), **summary(columns),
        "confidence": confidence.get(node, 1.0),
    } for node, columns in assigned.items()],
        columns=["sample", "lca", "rank", "n_members", "members", "inferred_mean",
                 "inferred_lo", "inferred_hi", "confidence"],
    ).sort_values("inferred_mean", ascending=False).to_csv(out / "lca_composition.csv",
                                                           index=False)


def _subset_kernel(K: sparse.csr_array, home: np.ndarray, labels: np.ndarray,
                   distances: sparse.csr_array | None = None):
    """Restrict a rectangular kernel to ``labels``, merging the other labels into sinks.

    Every source is kept. A
    row's mass on the dropped labels goes to zero-count sinks after the kept labels, and
    so does a source's unscaled mass when its home label is dropped (its home becomes the
    distance-0 sink). Merging zero-count labels is exact, so the likelihood is unchanged.
    Sinks are labels only: sources are not labels here, so no row is added for them.

    There is one sink per distinct distance among the dropped entries, so a
    ``DecayKernel`` re-decays them exactly; without ``distances`` (sharing ``K``'s
    pattern) there is at most one.

    Returns ``(K, home, n_sink, distances)``.
    """
    K = K.tocsr()
    coo = K.tocoo()
    labels = np.asarray(labels)
    k = len(labels)
    position = np.full(K.shape[1], -1, dtype=np.int64)
    position[labels] = np.arange(k)
    if distances is None:
        d = np.zeros(coo.nnz)
    else:
        d = sparse.csr_array((distances.tocsr().data + 1.0, K.indices, K.indptr),
                             shape=K.shape).tocoo().data - 1.0
    inside = position[coo.col] >= 0
    new_home = position[np.asarray(home)]
    lost_d = d[~inside]
    levels = np.unique(np.concatenate([lost_d, [0.0] if (new_home < 0).any() else []]))
    n_sink, width = len(levels), max(len(levels), 1)
    pair, pair_of = np.unique(coo.row[~inside].astype(np.int64) * width
                              + np.searchsorted(levels, lost_d), return_inverse=True)
    mass = np.bincount(pair_of, coo.data[~inside], minlength=len(pair))
    if n_sink:
        new_home[new_home < 0] = k + np.searchsorted(levels, 0.0)
    r = np.concatenate([coo.row[inside], pair // width])
    c = np.concatenate([position[coo.col[inside]], k + pair % width])
    shape = (K.shape[0], k + n_sink)
    matrix = sparse.csr_array((np.concatenate([coo.data[inside], mass]), (r, c)), shape=shape)
    new_distances = None
    if distances is not None:
        new_distances = sparse.csr_array(
            (np.concatenate([d[inside], levels[pair % width]]) + 1.0, (r, c)), shape=shape)
        new_distances.data -= 1.0
    return matrix, new_home, n_sink, new_distances


IDENTIFIABLE_TV = 0.01   # genomes whose label distributions are closer than this collide


def _panel_identifiability(K: sparse.csr_array, genome_of_row: np.ndarray,
                           weight: np.ndarray, entry_of: np.ndarray) -> np.ndarray:
    """Per entry, whether every member's label distribution ``T·K`` is TV >= 0.01 from
    every member of every *other* entry. Members of one taxon entry may collide freely:
    only their sum is reported.

    A taxon panel has thousands of members, so pairs are not enumerated. TV(a, b) < 0.01
    needs ``|a_l - b_l| < 0.02`` on every label, in particular on ``a``'s largest one, so
    only members with more than ``max(a) - 0.02`` there are compared with ``a``.
    """
    n_members = len(entry_of)
    n_entries = int(entry_of.max()) + 1 if n_members else 0
    rows = sparse.csr_array(sparse.csr_array(
        (weight, (genome_of_row, np.arange(len(weight)))),
        shape=(n_members, K.shape[0])) @ K)
    rows = sparse.csr_array(sparse.diags_array(1.0 / rows.sum(axis=1)) @ rows)
    top_label = np.asarray(rows.argmax(axis=1)).ravel()
    top_mass = rows.max(axis=1).toarray().ravel()
    at_top = rows[:, top_label].tocoo()       # [j, i] = member j's mass on i's top label
    near = at_top.data > top_mass[at_top.col] - 2 * IDENTIFIABLE_TV
    j, i = at_top.row[near], at_top.col[near]
    other = entry_of[j] != entry_of[i]
    j, i = j[other], i[other]
    ok = np.ones(n_entries, dtype=bool)
    if len(i):
        tv = 0.5 * np.asarray(abs(rows[i] - rows[j]).sum(axis=1)).ravel()
        close = tv < IDENTIFIABLE_TV
        ok[np.unique(entry_of[np.concatenate([i[close], j[close]])])] = False
    return ok


def _panel_entries(members: list[str]) -> tuple[list[str], np.ndarray]:
    """Entry ids in first-seen order and each member's entry index. A taxon member is
    ``<entry>::<v4g>`` (``build_panel_kernel.py``); a genome is its own entry."""
    entry_ids = list(dict.fromkeys(m.split("::", 1)[0] for m in members))
    position = {e: i for i, e in enumerate(entry_ids)}
    return entry_ids, np.array([position[m.split("::", 1)[0]] for m in members], dtype=np.int64)


def run(a) -> None:
    """Infer panel genome abundances from generic-database labels with a rectangular kernel.

    ``--amplicon-dir`` is a ``build_panel_kernel.py prepare`` directory
    (``panel_translation.tsv``). Every classified read counts, labelled by its hit's group.
    Observed labels no panel source reaches (no kernel mass, not a home) merge into one
    label ``U`` with a background source ``b``, ``K[b, U] = 1``: a free background row over
    them would fit their split exactly, so the merge loses nothing. The remaining labels
    merge into one zero-count sink (``_subset_kernel``). ``theta`` is over the panel genomes
    plus ``background``, whose mass is reported as ``unexplained_fraction``.
    """
    import torch

    c = _panel_context(a.mismapping_matrix, a.amplicon_dir, a.obs_mseq, a.min_identity,
                       a.no_mismapping)
    fitted, fitted_home, y, genome_of_row, weight = (c.fitted, c.home, c.y, c.genome_of_row,
                                                     c.weight)
    G, S, u, total = len(c.genomes) - 1, c.n_rows, fitted.shape[1] - 1, int(c.y.sum())
    infer_decay = getattr(a, "infer_distance_decay", False)
    if infer_decay and (c.strata is None or not c.strata[0].data.any()):
        raise SystemExit("--infer-distance-decay needs a panel kernel from "
                         "`build_panel_kernel.py align --tau >= 1`: a simulated or tau=0 "
                         "kernel has no distances to re-decay")

    # Observed composition (init and baseline): each home label's reads split evenly over
    # the (genome, source) rows it is home to; the rest is background.
    rel = y / total if total else y
    per_home = np.bincount(fitted_home[:S], minlength=u + 1)
    theta_obs = np.bincount(genome_of_row[:S], rel[fitted_home[:S]] / per_home[fitted_home[:S]],
                            minlength=G + 1)
    theta_obs[G] = 1.0 - theta_obs[:G].sum() if total else 0.0

    min_infer_reads = int(getattr(a, "min_infer_reads", 1))
    if min_infer_reads < 1:
        raise SystemExit("--min-infer-reads must be at least 1")
    # A sample whose reads hit no database label is a result, not an error: an all-zero
    # composition with status=no_reference_hits, so one empty sample cannot take down a run.
    low_depth, no_hits = total < min_infer_reads, total == 0
    if no_hits:
        log.warning("no reads in %s hit the database; emitting a zero composition", a.obs_mseq)
    genomes = c.genomes
    lo = hi = presence = np.full(G + 1, np.nan)
    draws, losses, diag, other_draws = np.empty((0, G + 1)), None, {}, {}
    if no_hits or low_depth:
        inferred, theta_obs = np.zeros(G + 1), np.zeros(G + 1)
    else:
        if infer_decay:
            M = si.DecayKernel(fitted, c.strata[0], c.strata[1], home=fitted_home)
        else:
            coo = fitted.tocoo()
            M = (torch.sparse_coo_tensor(
                    torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long),
                    torch.tensor(coo.data, dtype=torch.float64), size=fitted.shape).coalesce(),
                 torch.tensor(sm.home_entries(fitted, fitted_home), dtype=torch.float64),
                 None, torch.tensor(fitted_home, dtype=torch.long))
        T = (torch.tensor(genome_of_row, dtype=torch.long),
             torch.tensor(weight, dtype=torch.float64), G + 1)
        init = torch.tensor(np.clip(theta_obs, 1e-4, None), dtype=torch.float64)
        fa = _fit_args(a)
        fa.use_mismapping = True             # --no-mismapping already replaced K above
        samples, point, diag, losses = si._fit(
            a.mode, M, T, a.alpha, float(total), torch.tensor(y / total, dtype=torch.float64),
            "dirichlet_multinomial", init / init.sum(), fa, desc=a.sample_id)
        other_draws = diag.pop("_posterior_draws")
        draws = np.asarray(other_draws.pop("theta_eff"), dtype=np.float64)
        inferred = point.numpy()
        if samples is not None:
            lo = torch.quantile(samples, 0.05, dim=0).numpy()
            hi = torch.quantile(samples, 0.95, dim=0).numpy()
        presence = np.array(diag.pop("presence_prob", [np.nan] * (G + 1)))

    fit_status = "low_depth" if low_depth else "ok"
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    # Report per entry. Interval ends come from summed draws, not summed member intervals;
    # members are independent under the mean-field presence guide, so an entry is present
    # unless every member is absent.
    to_entry = np.eye(len(c.entries))[c.entry_of]                  # members x entries
    entry_draws = draws @ to_entry
    entry_lo = entry_hi = np.full(len(c.entries), np.nan)
    if len(entry_draws) > 1:
        entry_lo, entry_hi = np.quantile(entry_draws, [0.05, 0.95], axis=0)
    absent = np.ones(len(c.entries))
    np.multiply.at(absent, c.entry_of, 1.0 - presence)
    _uncertainty_outputs(a, list(c.entries), entry_draws, total)   # panel ids carry no lineage
    pd.DataFrame({
        "sample": a.sample_id, "genome_id": c.entries,
        "observed_rel_abundance": theta_obs @ to_entry, "inferred_mean": inferred @ to_entry,
        "inferred_lo": entry_lo, "inferred_hi": entry_hi, "presence_prob": 1.0 - absent,
        "fit_status": fit_status,
        "resolution": [("identifiable" if ok else "not_identifiable") for ok in c.identifiable]
                      + ["background"],
    }).to_csv(out / "inferred_composition.csv", index=False)
    if len(c.entries) < len(genomes):
        pd.DataFrame({
            "sample": a.sample_id, "genome_id": genomes,
            "entry": np.asarray(c.entries)[c.entry_of],
            "observed_rel_abundance": theta_obs, "inferred_mean": inferred,
            "inferred_lo": lo, "inferred_hi": hi, "presence_prob": presence,
        }).to_csv(out / "inferred_panel_members.csv", index=False)
    np.savez_compressed(
        out / "posterior_draws.npz", format_version=np.asarray("1"),
        kernel_format=np.asarray("rectangular"),
        genome_ids=np.asarray(genomes), theta_eff=draws,
        use_mismapping=np.asarray(not a.no_mismapping), infer_distance_decay=np.asarray(infer_decay),
        likelihood=np.asarray("dirichlet_multinomial"),
        n_reads=np.asarray(total, dtype=np.int64),
        min_infer_reads=np.asarray(min_infer_reads, dtype=np.int64),
        fit_status=np.asarray(fit_status), **other_draws)
    pd.DataFrame([{
        "sample": a.sample_id, "mode": a.mode, "likelihood": "dirichlet_multinomial",
        "use_mismapping": not a.no_mismapping, "min_identity": a.min_identity,
        "use_presence": not a.no_presence, "presence_prior": a.presence_prior,
        "presence_temp": a.presence_temp, "horseshoe": getattr(a, "horseshoe", False),
        "s_sigma": getattr(a, "s_sigma", 0.3), "infer_distance_decay": infer_decay,
        "built_distance_decay": None if c.strata is None else c.strata[1],
        "kernel_method": c.method,
        "mismapping_group_id": getattr(a, "mismapping_group_id", None),
        "mismapping_matrix_path": getattr(a, "mismapping_matrix_path", None),
        "kernel_format": "rectangular", "n_sources": c.n_sources,
        "n_labels_observed": c.n_labels_observed,
        "n_labels_unexplained": c.n_labels_unexplained,
        "unexplained_fraction": float(inferred[G]),
        "observed_unexplained_fraction": float(y[-1] / total) if total else 0.0,
        "n_reads": total, "n_foreign_hits": c.n_foreign_hits, "min_infer_reads": min_infer_reads,
        "n_genomes": G, "n_entries": len(c.entries) - 1, "status": "no_reference_hits" if no_hits else fit_status,
        "fit_status": fit_status, "posterior_draw_count": len(draws),
        **{key: json.dumps(v) for key, v in diag.items()},
    }]).to_csv(out / "inference_diagnostics.csv", index=False)
    if losses is not None:
        pd.DataFrame({"sample": a.sample_id, "step": range(len(losses)),
                      "loss": losses}).to_csv(out / "loss_trace.csv", index=False)
    print(f"infer[{a.mode}]: sample {a.sample_id}, {G} panel genomes + background -> {out}")


def _panel_context(kernel_path: Path, amplicon_dir: Path, obs_mseq, min_identity,
                   no_mismapping: bool) -> SimpleNamespace:
    """The fitted label space of a panel run, shared by inference and the fit check.

    Returns ``fitted`` (rows: one per (genome, source) pair, then background; columns:
    explained observed labels, at most one sink, then ``U``), ``home``, counts ``y`` over
    those columns, ``genome_of_row``/``weight`` (compact ``T``), ``genomes`` (fitted panel
    members then ``background``), ``entries`` (reported ids, ``background`` last) with each
    member's ``entry_of`` index, per-entry ``identifiable``, and counts for diagnostics.
    """
    try:
        k = sm.read_kernel(kernel_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    K, home = k.kernel, k.home
    distances = None if k.strata is None else k.strata[0]
    if no_mismapping:
        # The uncorrected baseline: every read is labelled with its source's home (B1).
        K = sparse.csr_array((np.ones(len(home)), (np.arange(len(home)), home)), shape=K.shape)
        distances = None

    table = pd.read_csv(amplicon_dir / "panel_translation.tsv", sep="\t")
    missing = sorted(set(table.source) - set(k.source_ids))
    if missing:
        raise SystemExit(f"panel sources not in the kernel: {', '.join(missing)}")
    panel_genomes = sorted(table.genome_id.unique())
    G = len(panel_genomes)
    # One fitted row per (genome, source) pair, so the compact T keeps one genome per row.
    # A source shared by two genomes gets its kernel row twice; the model is unchanged.
    rows = table.source.map({s: i for i, s in enumerate(k.source_ids)}).to_numpy()
    if distances is not None:
        # d + 1 so distance-0 entries survive the row selection on K's own pattern.
        distances = sparse.csr_array((distances.data + 1.0, K.indices, K.indptr),
                                     shape=K.shape)[rows]
        distances.data -= 1.0
    K, home, S = K[rows], home[rows], len(rows)
    gpos = {g: i for i, g in enumerate(panel_genomes)}
    genome_of_row = np.append(table.genome_id.map(gpos).to_numpy(), G)   # + background
    weight = np.append(table.weight.to_numpy(dtype=np.float64), 1.0)
    entries, entry_of = _panel_entries(panel_genomes + ["background"])
    identifiable = _panel_identifiability(K, genome_of_row[:S], weight[:S], entry_of[:G])

    # Observed reads per database label. A hit outside the kernel's database is a
    # different database, not background, so it is counted and warned about.
    label_of_hit = dict(zip(k.ref_headers, k.label_of_ref.tolist()))
    counts = np.zeros(K.shape[1])
    foreign = 0
    for path in obs_mseq:
        for _, hit in si.iter_mseq(path, min_identity):
            label = label_of_hit.get(hit)
            if label is None:
                foreign += 1
            else:
                counts[label] += 1
    if foreign:
        log.warning("%d observed hit(s) are not in the kernel's database (%s); ignored",
                    foreign, k.db_amplicons_sha256[:12] or "unknown sha256")
    observed = np.flatnonzero(counts)
    reached = np.zeros(K.shape[1], dtype=bool)
    reached[K.nonzero()[1]] = True
    reached[home] = True
    explained, unexplained = observed[reached[observed]], observed[~reached[observed]]

    sub, sub_home, n_sink, sub_distances = _subset_kernel(K, home, explained, distances)
    u = sub.shape[1]
    coo = sub.tocoo()
    r, col = np.append(coo.row, S), np.append(coo.col, u)       # + background row onto U
    fitted = sparse.csr_array((np.append(coo.data, 1.0), (r, col)), shape=(S + 1, u + 1))
    strata = None
    if sub_distances is not None:
        fitted_distances = sparse.csr_array((np.append(sub_distances.tocoo().data, 0.0) + 1.0,
                                             (r, col)), shape=fitted.shape)
        fitted_distances.data -= 1.0
        strata = (fitted_distances, k.strata[1])
    fitted_home = np.append(sub_home, u)
    y = np.concatenate([counts[explained], np.zeros(n_sink), [counts[unexplained].sum()]])
    return SimpleNamespace(
        fitted=fitted, home=fitted_home, y=y, genome_of_row=genome_of_row, weight=weight,
        strata=strata, method=k.provenance.get("method", "simulate"),
        genomes=panel_genomes + ["background"], entries=entries, entry_of=entry_of,
        identifiable=identifiable, n_rows=S,
        n_sources=len(set(rows)), n_labels_observed=len(observed),
        n_labels_unexplained=len(unexplained), n_foreign_hits=foreign)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amplicon-dir", type=Path, required=True,
                    help="`build_panel_kernel.py prepare` dir (panel_translation.tsv)")
    ap.add_argument("--mismapping-matrix", type=Path, required=True,
                    help="the panel kernel .npz (`build_panel_kernel.py build|align`)")
    ap.add_argument("--obs-mseq", type=Path, nargs="+", required=True,
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
    ap.add_argument("--horseshoe", action="store_true",
                    help="vi only: horseshoe shrinkage on unnormalised weights in place of "
                         "the Dirichlet prior (--alpha is ignored); pair with --no-presence")
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
    ap.add_argument("--taxonomy", type=Path,
                    help="MAPseq .tax (header<TAB>lineage) for lca_composition.csv")
    ap.add_argument("--lca-cutoff", type=float, default=None,
                    help="confidence below which lca_composition.csv cuts a lineage, at "
                         "every rank; default: the .tax #cutoff: combined values, or 0.5 "
                         "when those are all 0")
    ap.add_argument("--ambiguity-min-gain", type=float, default=0.2,
                    help="noise floor for ambiguity_pairs/sets.csv: minimum fraction of the "
                         "members' variance that only their sum resolves")
    ap.add_argument("--ambiguity-min-sd", type=float, default=1e-3,
                    help="noise floor for ambiguity_pairs/sets.csv: minimum ambiguous sd, "
                         "in relative abundance")
    ap.add_argument("--num-samples", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--s-sigma", type=float, default=0.3,
                    help="log-normal sigma of the mis-mapping scale s around 1; a tiny value "
                         "(e.g. 1e-4) pins s at 1")
    ap.add_argument("--steps", type=int, default=3000, help="SVI steps (vi/mle)")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--min-infer-reads", type=int, default=1000,
                    help="minimum mapped reads required before fitting a biological composition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output-dir", type=Path, required=True)
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run(a)


if __name__ == "__main__":
    main()
