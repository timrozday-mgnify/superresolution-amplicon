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
import hashlib
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


def _mismapping_matrix(a, refseqs: list[str]):
    """Return ``(M, group, strata)``: a measured or pre-computed matrix in reference order.

    ``M`` is a reference-square CSR with ``group`` ``None``, or the unique-amplicon kernel
    the duplicate-collapsing backends write for database-scale reference sets. ``strata``
    is the ``(distances, decay)`` pair a ``tau >= 1`` alignment build stores, or ``None``
    for every other source — only a build that recorded its distances can be re-decayed.
    """
    if getattr(a, "mismapping_matrix", None) is None:
        if getattr(a, "build_grouped", False):
            # Measured at V4-group level: one simulation per distinct amplicon stands in
            # for its duplicates, and the row is shared by every member of the group.
            v4_ids, ref_group = _v4_groups(a.amplicon_dir, refseqs)
            active = None
            if getattr(a, "active_amplicons", None):
                position = {i: k for k, i in enumerate(v4_ids)}
                active = np.array(sorted({
                    position["v4g_" + hashlib.sha256(sequence.encode()).hexdigest()[:16]]
                    for _, sequence in si.read_fasta(a.active_amplicons)}), dtype=np.int64)
            return (si.build_mismapping_grouped(a.sim_mseq, refseqs, ref_group,
                                                a.min_identity, active),
                    ref_group, None)
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


def _v4_groups(amplicon_dir: Path, references: list[str]) -> tuple[list[str], np.ndarray]:
    """Return the exact-V4 group ids and each reference's group index.

    A group is one byte-identical amplicon, named ``v4g_`` + the first 16 hex digits of its
    sha256, so the id is the same in any run, batch or database holding that sequence.
    Groups are indexed in sorted id order.
    """
    records = dict(si.read_fasta(amplicon_dir / "amplicons.fasta"))
    if len(records) != len(references) or set(records) != set(references):
        raise SystemExit("amplicons.fasta does not match the translation-table references")
    digest = {sequence: "v4g_" + hashlib.sha256(sequence.encode()).hexdigest()[:16]
              for sequence in set(records.values())}
    if len(set(digest.values())) != len(digest):
        raise SystemExit("two distinct amplicons share a v4g_ id; lengthen the digest")
    ids, group = np.unique(np.asarray([digest[records[r]] for r in references]),
                           return_inverse=True)
    return ids.tolist(), group.astype(np.int64)


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
        if not ranks or ranks[-1] != name:    # v4 groups: the fitted entity is the leaf
            ranks.append(name)
            rank_names[tuple(ranks)] = getattr(a, "infer_space", "genome")
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


def _v4_outputs(a, v4_ids, ref_group, refseqs, member_genomes, member_g_of_ref,
                theta_obs, inferred, lo, hi, presence, theta_draws, fit_status, lineages):
    """Write ``inferred_v4_groups.csv`` and the genome table derived from it.

    The genome table splits each group's mass evenly over its member references and sums
    per genome. That is a labelled convention, not an estimate: a genome sharing a group
    with another genome is ``not_identifiable`` and gets no interval or presence call.
    """
    size = np.bincount(ref_group, minlength=len(v4_ids))
    split = sparse.csr_array((1.0 / size[ref_group], (member_g_of_ref, ref_group)),
                             shape=(len(member_genomes), len(v4_ids)))
    n_genomes = np.bincount(split.nonzero()[1], minlength=len(v4_ids))
    representative = np.unique(ref_group, return_index=True)[1]         # first member

    members = [[] for _ in v4_ids]
    for reference, group in zip(refseqs, ref_group):
        if reference in lineages:
            members[group].append(lineages[reference])

    pd.DataFrame({
        "sample": a.sample_id, "v4_group_id": v4_ids, "n_references": size,
        "n_genomes": n_genomes, "lca": [_lca(m) for m in members],
        "representative_reference": [refseqs[i] for i in representative],
        "observed_rel_abundance": theta_obs, "inferred_mean": inferred,
        "inferred_lo": lo, "inferred_hi": hi, "presence_prob": presence,
        "fit_status": fit_status,
    }).to_csv(a.output_dir / "inferred_v4_groups.csv", index=False)

    genome_mean, genome_obs = split @ inferred, split @ theta_obs
    shared = n_genomes > 1
    genome_ids, resolution = [], []
    genome_lo = np.full(len(member_genomes), np.nan)
    genome_hi, genome_presence = genome_lo.copy(), genome_lo.copy()
    for g in range(len(member_genomes)):
        groups = split.indices[split.indptr[g]:split.indptr[g + 1]]
        genome_ids.append(";".join(v4_ids[k] for k in groups))
        if shared[groups].any():
            resolution.append("not_identifiable")
            continue
        resolution.append("identifiable")
        # The genome owns these groups outright, so its draws are their summed draws.
        # Intervals exist only where the group fit drew samples (not mle, not low depth).
        if not np.isnan(lo).all():
            genome_lo[g], genome_hi[g] = (
                np.quantile(theta_draws[:, groups].sum(axis=1), [0.05, 0.95])
                if genome_mean[g] > 0 else (0.0, 0.0))
        genome_presence[g] = presence[groups].max()      # lower bound on P(any present)

    return pd.DataFrame({
        "sample": a.sample_id, "genome_id": member_genomes,
        "observed_rel_abundance": genome_obs, "inferred_mean": genome_mean,
        "inferred_lo": genome_lo, "inferred_hi": genome_hi,
        "presence_prob": genome_presence, "fit_status": fit_status,
        "v4_group_ids": genome_ids, "resolution": resolution,
    })


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
    A reference with a direct hit is kept regardless of ``M``.

    Then take the genomes owning that set, and every reference copy those genomes own:
    ``T`` is the within-genome copy distribution and has to stay complete, or a kept
    genome quietly loses the mass of its dropped copies.

    What is left out is dead weight — a reference no observed read can be traced to, and
    a genome none of whose copies were seen. Each is an empty row, an empty column, and a
    genome dimension whose likelihood is flat. Note the closure is one round, not a fixed
    point: a kept genome's unobserved copy may itself tie to a reference outside the set,
    and the mass leaking there goes to ``_subset_mismapping``'s zero-count sink labels
    rather than pulling another genome in.
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
    # A reference with a direct hit is always kept, whatever M says about it: a measured
    # row can be empty or send every read elsewhere, and dropping it would silently drop
    # its reads from the likelihood. Its genome then comes back with it.
    reachable[hit] = True
    gen_idx = np.unique(g_of_ref[reachable])
    return np.flatnonzero(np.isin(g_of_ref, gen_idx)), gen_idx


def _identity_rows_for_observed(M_sparse: sparse.csr_array, group: np.ndarray | None,
                                obs: np.ndarray) -> sparse.csr_array:
    """Give every observed label whose row of ``M`` is empty an identity row.

    A measured matrix built for another batch has empty rows for groups it never
    simulated. Kept by pruning, such a label's reads could come from no source and would
    distort the rest of the fit. An identity row ("reads here came from here") is the same
    fallback ``build_mismapping_grouped`` uses for a group the simulator skipped. In the
    grouped format the diagonal is ``1 / size`` so ``sum_B S[A, B] * size(B) == 1``.
    """
    M_sparse = M_sparse.tocsr()
    labels = np.flatnonzero(obs > 0)
    if group is not None:
        labels = np.unique(group[labels])
    empty = labels[np.diff(M_sparse.indptr)[labels] == 0]
    if not len(empty):
        return M_sparse
    log.warning("%d observed label(s) have no row in the mis-mapping matrix; "
                "using an identity row", len(empty))
    size = (np.ones(M_sparse.shape[0]) if group is None
            else np.bincount(group, minlength=M_sparse.shape[0]).astype(np.float64))
    # ponytail: no distance stratum is added; alignment kernels always carry a self entry,
    # so an empty row only occurs in measured matrices, which have no strata.
    return (M_sparse + sparse.csr_array((1.0 / size[empty], (empty, empty)),
                                        shape=M_sparse.shape)).tocsr()


def _subset_translation(translation, ref_idx: np.ndarray, gen_idx: np.ndarray,
                        gen_pos: np.ndarray, n_sink: int = 0):
    """Restrict ``T`` to the kept genomes x kept references, plus ``n_sink`` sink columns.

    Rows stay normalised without rescaling: every copy of a kept genome is kept. Sink
    columns carry no weight (in the compact form they nominally belong to genome 0).
    """
    if isinstance(translation, tuple):
        genome_of_row, weight, _ = translation
        return (np.concatenate([gen_pos[genome_of_row[ref_idx]],
                                np.zeros(n_sink, dtype=np.int64)]),
                np.concatenate([weight[ref_idx], np.zeros(n_sink)]), len(gen_idx))
    return np.hstack([translation[np.ix_(gen_idx, ref_idx)],
                      np.zeros((len(gen_idx), n_sink))])


def _subset_mismapping(M_sparse: sparse.csr_array, group: np.ndarray | None,
                       ref_idx: np.ndarray, strata=None):
    """Restrict ``M`` to ``ref_idx``, sending the mass it drops to zero-count sink labels.

    A kept source's leak onto a pruned label, or onto a pruned member of a kept group, is
    evidence against that source: the label drew no reads. Renormalising the row over the
    kept labels would throw that evidence away and favour sources that leak outside the
    kept set. The dropped mass goes instead to sink labels appended after the kept ones,
    each with no reads. Merging zero-count labels is exact for the multinomial and the
    Dirichlet-multinomial, so the pruned likelihood equals the full one and every row
    still sums to 1 (which ``_apply_mismapping``'s ``1 - diagonal`` relies on).

    There is one sink per distinct distance among the dropped entries, so a
    ``DecayKernel`` re-decays the sinks exactly as it would the entries they replace;
    without ``strata`` there is at most one. A sink is a label of size 1 with an identity
    row; no genome sends it reads.

    Returns ``(matrix, group, strata, n_sink)``, with ``group`` and ``strata`` extended over
    the sinks (``group`` stays ``None`` for a reference-square matrix).
    """
    M_sparse = M_sparse.tocsr()
    n = M_sparse.shape[0]
    ref_idx = np.asarray(ref_idx)
    if group is None:
        keep = ref_idx
        size_full, size_kept = np.ones(n), np.zeros(n)
        size_kept[keep] = 1.0
    else:
        # Grouped form: subset the unique-amplicon kernel, not the reference square.
        keep = np.unique(group[ref_idx])
        size_full = np.bincount(group, minlength=n).astype(np.float64)
        size_kept = np.bincount(group[ref_idx], minlength=n).astype(np.float64)
    k = len(keep)
    position = np.full(n, -1, dtype=np.int64)
    position[keep] = np.arange(k)

    rows = M_sparse[keep].tocoo()
    source, target = rows.row.astype(np.int64), rows.col.astype(np.int64)
    if strata is None:
        distance = np.zeros(len(rows.data))
    else:
        shifted = strata[0].tocsr().copy()
        shifted.data = shifted.data + 1.0          # keep distance-0 entries explicit
        distance = shifted[keep].tocoo().data - 1.0
        if len(distance) != len(rows.data):
            raise ValueError("distance strata must share the matrix's sparsity pattern")
    inside = position[target] >= 0
    lost = rows.data * (size_full[target] - size_kept[target])
    sinking = lost > 0
    levels, level = np.unique(distance[sinking], return_inverse=True)
    n_sink = len(levels)
    width = max(n_sink, 1)
    pair, pair_of = np.unique(source[sinking] * width + level, return_inverse=True)
    sink_mass = np.bincount(pair_of, lost[sinking], minlength=len(pair))
    diagonal = k + np.arange(n_sink)

    r = np.concatenate([source[inside], pair // width, diagonal])
    c = np.concatenate([position[target[inside]], k + pair % width, diagonal])
    shape = (k + n_sink, k + n_sink)
    matrix = sparse.csr_array(
        (np.concatenate([rows.data[inside], sink_mass, np.ones(n_sink)]), (r, c)), shape=shape)
    new_strata = None
    if strata is not None:
        d = np.concatenate([distance[inside], levels[pair % width], np.zeros(n_sink)])
        distances = sparse.csr_array((d + 1.0, (r, c)), shape=shape)
        distances.data -= 1.0
        new_strata = (distances, strata[1])
    new_group = None if group is None else np.concatenate([position[group[ref_idx]], diagonal])
    return matrix, new_group, new_strata, n_sink


def _subset_kernel(K: sparse.csr_array, home: np.ndarray, labels: np.ndarray,
                   distances: sparse.csr_array | None = None):
    """Restrict a rectangular kernel to ``labels``, merging the other labels into sinks.

    The sources x labels counterpart of ``_subset_mismapping``. Every source is kept. A
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


def run_panel(a) -> None:
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

    if getattr(a, "build_mismapping", False) or getattr(a, "infer_space", "genome") != "genome":
        raise SystemExit("a rectangular panel kernel supports only genome-space inference")
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
    low_depth, no_hits = total < min_infer_reads, total == 0
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
        infer_space=np.asarray("genome"), kernel_format=np.asarray("rectangular"),
        genome_ids=np.asarray(genomes), theta_eff=draws,
        use_mismapping=np.asarray(not a.no_mismapping), infer_distance_decay=np.asarray(infer_decay),
        likelihood=np.asarray("dirichlet_multinomial"), use_active_subset=np.asarray(False),
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


def run(a) -> None:
    import torch

    if (getattr(a, "mismapping_matrix", None) is not None
            and a.mismapping_matrix.suffix == ".npz" and sm.is_rectangular(a.mismapping_matrix)):
        return run_panel(a)

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

    # Exact-V4 groups: the inference space in v4_group mode, and in genome mode the
    # diagnostic for how many genomes one fitted sequence spans (only where
    # amplicons.fasta exists; hand-built reference sets may have none).
    infer_space = getattr(a, "infer_space", "genome")
    all_refseqs, member_genomes, member_g_of_ref = refseqs, genomes, g_of_ref
    v4_ids = ref_group = None
    if infer_space == "v4_group" or (a.amplicon_dir / "amplicons.fasta").exists():
        v4_ids, ref_group = _v4_groups(a.amplicon_dir, refseqs)
        genomes_per_group = np.bincount(np.unique(np.c_[ref_group, g_of_ref], axis=0)[:, 0],
                                        minlength=len(v4_ids))
    if infer_space == "v4_group":
        # One parameter per distinct V4 sequence. Genomes sharing one are not separable by
        # any read, so fitting them apart lets the prior, the initial point and the
        # presence gate decide instead (docs/gtdb_inference_recovery_plan.md, Evidence §2).
        size = np.bincount(ref_group)
        genomes, g_of_ref = v4_ids, ref_group
        translation = (ref_group, 1.0 / size[ref_group], len(v4_ids))

    # Of the whole matrix, not the pruned one: properties of the reference set, so they
    # stay comparable between samples that prune to different sizes. The kernel average
    # gives every unique V4 group equal weight; the reference average restores every
    # group member and is the diagnostic relevant to a reference-level report.
    mean_kernel_diagonal = sm.matrix_diagonal_mean(M_sparse)
    mean_reference_diagonal = (
        mean_kernel_diagonal if group is None
        else float(sm.grouped_diagonal(M_sparse, group).mean())
    )

    counts = si.observed_refseq_counts(a.obs_mseq, refseqs, a.min_identity)
    obs = np.array([counts.get(r, 0) for r in refseqs], dtype=np.float64)
    total = int(obs.sum())
    if infer_space == "v4_group":
        # MAPseq splits a group's reads evenly over an arbitrary *subset* of its identical
        # members (SC2200627-SC3 on GTDB: 20 of 445), which the uniform 1/size split above
        # reads as evidence against large groups. Only the group total is information, so
        # spread it evenly; for a multinomial this is the group-label likelihood.
        # ponytail: exact for kernel-built M (uniform within a group); a simulate-built M
        # that measured MAPseq's pile-up would need its columns collapsed too.
        obs = (np.bincount(ref_group, obs, minlength=len(v4_ids)) / size)[ref_group]
    M_sparse = _identity_rows_for_observed(M_sparse, group, obs)
    min_infer_reads = int(getattr(a, "min_infer_reads", 1))
    if min_infer_reads < 1:
        raise SystemExit("--min-infer-reads must be at least 1")
    low_depth = total < min_infer_reads
    # A sample whose reads hit no reference amplicon is a result, not an error: it is
    # reported as an all-zero composition flagged `status=no_reference_hits` (with
    # `n_reads=0`) in the same files a normal run writes, so a multi-sample run is not
    # taken down by one empty sample and the outcome is unambiguous downstream.
    no_hits = total == 0
    if no_hits:
        log.warning("no reads in %s hit a reference amplicon; emitting a zero composition",
                    a.obs_mseq)
    elif low_depth:
        log.info("sample %s has %d mapped reads; below --min-infer-reads %d",
                 a.sample_id, total, min_infer_reads)
    log.info("sample %s: %d mapped reads, %d refs, %d genomes, mode=%s",
             a.sample_id, total, len(refseqs), len(genomes), a.mode)

    # Fit only what the reads can speak to. The reference set is the whole database
    # (GTDB SSU: ~1e5 amplicons over ~1e5 genomes) while a sample touches a few hundred
    # of them; the rest is an empty row, an empty column and a flat genome dimension.
    # `all_genomes` keeps the output contract — every genome in the set is reported, a
    # pruned one at zero.
    all_genomes, gen_idx = genomes, None
    fitted_refs = np.arange(len(refseqs))
    if not no_hits and not getattr(a, "no_prune", False):
        ref_idx, gen_idx = _active_subset(M_sparse, group, obs, g_of_ref)
        if len(gen_idx) == len(genomes):
            gen_idx = None                       # nothing to prune; keep the fast path
        else:
            log.info("pruned to %d/%d references and %d/%d genomes with observed support",
                     len(ref_idx), len(refseqs), len(gen_idx), len(genomes))
            gen_pos = np.full(len(genomes), -1, dtype=np.int64)
            gen_pos[gen_idx] = np.arange(len(gen_idx))
            M_sparse, group, strata, n_sink = _subset_mismapping(M_sparse, group, ref_idx,
                                                                 strata)
            # The sink labels close the fitted reference arrays: no reads, and no genome
            # sends them mass (weight 0). `refseqs` keeps only the real references.
            translation = _subset_translation(translation, ref_idx, gen_idx, gen_pos, n_sink)
            refseqs = [refseqs[i] for i in ref_idx]
            genomes = [genomes[i] for i in gen_idx]
            g_of_ref = np.concatenate([gen_pos[g_of_ref[ref_idx]],
                                       np.zeros(n_sink, dtype=np.int64)])
            obs = np.concatenate([obs[ref_idx], np.zeros(n_sink)])
            fitted_refs = ref_idx

    n_groups_fitted = max_genomes_per_active_group = None
    if v4_ids is not None:
        active_groups = np.unique(ref_group[fitted_refs])
        n_groups_fitted = len(active_groups)
        max_genomes_per_active_group = int(genomes_per_group[active_groups].max())
        if infer_space == "genome" and max_genomes_per_active_group > 100:
            log.warning("sample %s: one fitted V4 sequence spans %d genomes; genome space "
                        "cannot separate them. Use --infer-space v4_group.",
                        a.sample_id, max_genomes_per_active_group)

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
    posterior_draws = {}
    if no_hits or low_depth:
        samples, losses, diag = None, None, {}
        inferred = np.zeros(len(genomes))
        theta_obs = np.zeros(len(genomes))
    else:
        theta_init = torch.tensor(np.clip(theta_obs, 1e-4, None), dtype=torch.float64)
        theta_init = theta_init / theta_init.sum()

        fa = _fit_args(a)
        samples, point, diag, losses = si._fit(
            a.mode, M, T, a.alpha, float(total), torch.tensor(ref_rel, dtype=torch.float64),
            "dirichlet_multinomial", theta_init, fa, desc=a.sample_id)
        posterior_draws = diag.pop("_posterior_draws")
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

    if posterior_draws:
        theta_draws = np.asarray(posterior_draws.pop("theta_eff"), dtype=np.float64)
        if gen_idx is None:
            full_theta_draws = theta_draws
        else:
            full_theta_draws = np.zeros((len(theta_draws), len(all_genomes)), dtype=np.float64)
            full_theta_draws[:, gen_idx] = theta_draws
    else:
        full_theta_draws = np.empty((0, len(all_genomes)), dtype=np.float64)

    fit_status = "low_depth" if low_depth else "ok"
    status = "no_reference_hits" if no_hits else fit_status
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    lineages = _read_taxonomy(a.taxonomy) if getattr(a, "taxonomy", None) else None
    lineage_of = None
    if lineages is not None and len(full_theta_draws) > 1:
        entity_of_ref = ref_group if infer_space == "v4_group" else member_g_of_ref
        active, grouped = full_theta_draws.any(axis=0), {}
        for reference, entity in zip(all_refseqs, entity_of_ref):
            if active[entity] and reference in lineages:
                grouped.setdefault(entity, []).append(lineages[reference])
        lineage_of = {entity: _lca(m) for entity, m in grouped.items()}
    _uncertainty_outputs(a, list(all_genomes), full_theta_draws, total, lineage_of)
    if infer_space == "v4_group":
        comp = _v4_outputs(a, v4_ids, ref_group, all_refseqs, member_genomes,
                           member_g_of_ref, theta_obs, inferred, lo, hi, presence,
                           full_theta_draws, fit_status, lineages or {})
    else:
        comp = pd.DataFrame([{
            "sample": a.sample_id, "genome_id": g,
            "observed_rel_abundance": float(theta_obs[i]),
            "inferred_mean": float(inferred[i]),
            "inferred_lo": float(lo[i]), "inferred_hi": float(hi[i]),
            "presence_prob": float(presence[i]),
            "fit_status": fit_status,
        } for i, g in enumerate(all_genomes)])
    comp.to_csv(out / "inferred_composition.csv", index=False)
    np.savez_compressed(
        out / "posterior_draws.npz",
        format_version=np.asarray("1"),
        infer_space=np.asarray(infer_space),
        genome_ids=np.asarray(all_genomes),
        theta_eff=full_theta_draws,
        use_mismapping=np.asarray(not a.no_mismapping),
        infer_distance_decay=np.asarray(infer_decay),
        likelihood=np.asarray("dirichlet_multinomial"),
        use_active_subset=np.asarray(gen_idx is not None),
        n_reads=np.asarray(total, dtype=np.int64),
        min_infer_reads=np.asarray(min_infer_reads, dtype=np.int64),
        fit_status=np.asarray(fit_status),
        **posterior_draws,
    )
    pd.DataFrame([{
        "sample": a.sample_id, "mode": a.mode, "likelihood": "dirichlet_multinomial",
        "use_mismapping": not a.no_mismapping, "min_identity": a.min_identity,
        "use_presence": not a.no_presence, "presence_prior": a.presence_prior,
        "presence_temp": a.presence_temp,
        "horseshoe": getattr(a, "horseshoe", False),
        "infer_distance_decay": infer_decay,
        "built_distance_decay": None if strata is None else strata[1],
        "mismapping_group_id": getattr(a, "mismapping_group_id", None),
        "mismapping_matrix_path": getattr(a, "mismapping_matrix_path", None),
        "n_reads": int(total), "min_infer_reads": min_infer_reads,
        "mean_kernel_diagonal": mean_kernel_diagonal,
        "mean_reference_diagonal": mean_reference_diagonal,
        "infer_space": infer_space,
        "taxonomy_sha256": (hashlib.sha256(Path(a.taxonomy).read_bytes()).hexdigest()
                            if getattr(a, "taxonomy", None) else None),
        "n_refs_fitted": len(refseqs),
        "n_genomes_fitted": len(np.unique(member_g_of_ref[fitted_refs])),
        "n_genomes": len(member_genomes),
        "n_groups_fitted": n_groups_fitted,
        "max_genomes_per_active_group": max_genomes_per_active_group,
        "status": status, "fit_status": fit_status,
        "posterior_draw_count": len(full_theta_draws),
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
    # stay row-stochastic over the references it keeps plus the sink, or
    # _apply_mismapping's `1 - diagonal` off-diagonal mass is wrong.
    group = np.array([0, 0, 1, 2])
    cluster = sparse.csr_array(np.array([[0.5, 0.0, 0.0], [0.05, 0.8, 0.1], [0.0, 0.2, 0.8]]))
    kept = np.array([0, 1, 2])                       # drop ghost|0, the only member of 2
    sub, sub_group, _, n_sink = _subset_mismapping(cluster, group, kept)
    sizes = np.bincount(sub_group).astype(float)
    assert np.allclose(sub @ sizes, 1.0), (sub.toarray(), sizes)
    assert n_sink == 1 and sub_group.tolist() == [0, 0, 1, 2], sub_group
    assert np.isclose(sub[1, 2], 0.1), sub.toarray()  # group 1's leak onto ghost, kept
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


def demo_v4(workdir: Path) -> None:
    """Self-check the V4-group space: genomes a and b share amplicon X, c alone carries Y.

    Two groups are fitted. a and b are ``not_identifiable`` and get the same labelled split
    of X with no interval; the genome table still has a finite mean for every genome and
    sums to 1. Genome space on the same set records the widest fitted group instead.
    """
    refs, seqs = ["a|0|x", "b|0|x", "c|0|y"], ["ACGTACGT", "ACGTACGT", "TTGCAAGC"]
    amp = workdir / "amp"
    amp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"genome_id": ["a", "b", "c"], "refseq": refs, "weight": 1.0}).to_csv(
        amp / "translation_table.tsv", sep="\t", index=False)
    (amp / "amplicons.fasta").write_text("".join(f">{r}\n{s}\n" for r, s in zip(refs, seqs)))
    (workdir / "refs.tax").write_text("#name: refdb\na|0|x\tBacteria;P1;s__a;a\n"
                                      "b|0|x\tBacteria;P1;s__b;b\nc|0|y\tBacteria;P2;s__c;c\n")
    sm.write_matrix(workdir / "M.npz", sparse.csr_array(
        np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])), refs)
    with open(workdir / "obs.mseq", "w") as fh:
        for j, n in enumerate([3000, 3000, 4000]):
            for i in range(n):
                fh.write(f"read{j}_{i}\t{refs[j]}\t500\t0.99\n")

    args = argparse.Namespace(
        amplicon_dir=amp, sim_mseq=None, mismapping_matrix=workdir / "M.npz",
        obs_mseq=[workdir / "obs.mseq"], min_identity=None, sample_id="v4", mode="vi",
        alpha=0.5, steps=500, lr=0.05, num_samples=100, warmup=0, no_mismapping=False,
        seed=0, no_presence=False, presence_prior=si.DEFAULT_PRESENCE_PRIOR,
        presence_temp=si.DEFAULT_PRESENCE_TEMP, no_prune=False, infer_space="v4_group",
        taxonomy=workdir / "refs.tax", output_dir=workdir / "v4")
    run(args)
    groups = pd.read_csv(workdir / "v4" / "inferred_v4_groups.csv").set_index("v4_group_id")
    x = "v4g_" + hashlib.sha256(b"ACGTACGT").hexdigest()[:16]
    assert len(groups) == 2 and x in groups.index, groups
    assert groups.loc[x, "n_genomes"] == 2 and groups.loc[x, "lca"] == "Bacteria;P1", groups
    assert abs(groups.loc[x, "inferred_mean"] - 0.6) < 0.03, groups

    genomes = pd.read_csv(workdir / "v4" / "inferred_composition.csv").set_index("genome_id")
    assert genomes["resolution"].to_dict() == {
        "a": "not_identifiable", "b": "not_identifiable", "c": "identifiable"}, genomes
    assert np.isfinite(genomes["inferred_mean"]).all(), genomes
    assert abs(genomes["inferred_mean"].sum() - 1.0) < 1e-6, genomes
    assert genomes.loc["a", "inferred_mean"] == genomes.loc["b", "inferred_mean"], genomes
    assert genomes.loc[["a", "b"], "inferred_lo"].isna().all(), genomes
    assert np.isfinite(genomes.loc["c", ["inferred_lo", "inferred_hi"]].astype(float)).all()
    with np.load(workdir / "v4" / "posterior_draws.npz") as posterior:
        assert str(posterior["infer_space"]) == "v4_group"
        assert posterior["genome_ids"].tolist() == groups.index.tolist()

    run(argparse.Namespace(**{**vars(args), "infer_space": "genome",
                              "output_dir": workdir / "genome"}))
    diag = pd.read_csv(workdir / "genome" / "inference_diagnostics.csv").iloc[0]
    assert diag["infer_space"] == "genome" and diag["max_genomes_per_active_group"] == 2, diag
    assert diag["n_groups_fitted"] == 2 and diag["n_genomes_fitted"] == 3, diag
    print("v4-group demo OK")


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
    ap.add_argument("--active-amplicons", type=Path, default=None,
                    help="--build-grouped: the FASTA simulated from, so a group the "
                         "simulator skipped (IUPAC codes) keeps an identity row instead "
                         "of becoming a source no read can come from")
    ap.add_argument("--build-grouped", action="store_true",
                    help="measure that matrix per distinct V4 amplicon instead of per "
                         "reference: rows are groups, so one simulated read set per "
                         "distinct sequence covers its duplicates. Needs amplicons.fasta")
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
    ap.add_argument("--infer-space", choices=["genome", "v4_group"], default="genome",
                    help="fit one parameter per genome, or per exact distinct V4 amplicon. "
                         "Use v4_group against GTDB-scale sets, where thousands of genomes "
                         "share each V4 sequence; it also writes inferred_v4_groups.csv")
    ap.add_argument("--taxonomy", type=Path,
                    help="MAPseq .tax (header<TAB>lineage) for the v4_group lca column and "
                         "lca_composition.csv")
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
        import tempfile

        demo()
        demo_prune()
        demo_decay()
        with tempfile.TemporaryDirectory() as temporary:
            return demo_v4(Path(temporary))
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
