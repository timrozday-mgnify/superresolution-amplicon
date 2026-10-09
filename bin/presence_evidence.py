"""Evidence that each panel member is present, from the reads on its own amplicons.

The inference's presence gate is no such measure: its probability tracks abundance
(Nov2025: a member with a median 2,350 reads on amplicons nothing else in the panel
makes got 0.47). This module asks the question directly, per sample, in five steps.

1. Greedy claims. Members go most-read first. Each claims reads on its amplicon labels
   in its copy-weight proportions, the claim estimated from the labels no earlier member
   has taken (its *free* labels), so a label two members share is the more abundant one's
   and only what the other makes on its own counts for it. Members with identical label
   weights are one indistinguishable group.
2. Spill-over. A claimed member's sequencing errors land on labels one or two edits from
   its amplicons. The rate per edit distance is calibrated per sample, as a high quantile
   of reads on non-panel neighbour labels over the reads of the one member they are near.
3. Leftover test. A member's reads on its free labels against the reads expected there
   from every other member's spill-over plus a small floor: a one-sided Poisson p-value,
   and the detection limit (fewest reads that would be significant), so "absent" and
   "below detection" are told apart.
4. Copy ratios. Real presence arrives in the member's copy ratios: the total variation
   between its free labels' observed and expected shares.
5. Bayes factor. Fit every member's reads (amplicons plus spill-over) by Poisson EM and
   refit with the member removed. Against that fit, average the likelihood of adding the
   member back over a log-uniform prior on its reads (1 read to the sample's total),
   deflated by the fit's overdispersion, and combine with a prior presence probability.
   A plain drop-one likelihood ratio can only find evidence *for* presence -- a member
   whose amplicons hold nothing but spill-over scores 0 and sits at the prior -- whereas
   the averaged likelihood also pays for the reads a present member would have made.

Evidence only: sramp's fit stays the source of abundances.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import special, stats

DEFAULT_ALPHA = 1e-3      # leftover p-value below which a member is called present
DEFAULT_PRIOR = 0.5       # prior presence probability for step 5
DEFAULT_FLOOR = 0.5       # reads expected on any label from nothing (barcode hopping, chimeras)
# Spill-over per read of the parent, used when a sample has too few clean neighbour pairs
# to calibrate on. Nov2025's one-edit neighbours held 0.02-2.5% of their parent's reads.
DEFAULT_RATES = {1: 0.02, 2: 0.002}
RATE_QUANTILE = 0.95
MIN_PARENT_READS = 1000   # neighbours of a member claiming fewer are too noisy to calibrate on
MIN_PAIRS = 10
EM_STEPS = 2000


def _member_groups(table: pd.DataFrame, home_of_source: dict[str, int]):
    """Member groups with identical label weights, as ``{group: (members, {label: w})}``."""
    weights: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for g, s, w in zip(table.genome_id, table.source, table.weight):
        weights[g][home_of_source[s]] += float(w)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for g, w in weights.items():
        groups[tuple(sorted((label, round(v, 9)) for label, v in w.items()))].append(g)
    return {"+".join(sorted(m)): (sorted(m), dict(key)) for key, m in groups.items()}


def _greedy(counts: np.ndarray, groups: dict):
    """Step 1: claims most-read first; returns ``{group: (claim, free labels, own reads)}``.

    The order ranks each group by the claim its unshared labels support -- all its labels
    only if it has none -- so a rare strain that also carries an abundant member's
    amplicon cannot outrank that member on the reads of the label they share.
    """
    owners = defaultdict(int)
    for _, w in groups.values():
        for label in w:
            owners[label] += 1

    def first_claim(w):
        own = [label for label in w if owners[label] == 1] or list(w)
        return sum(counts[label] for label in own) / sum(w[label] for label in own)

    total = {g: first_claim(w) for g, (_, w) in groups.items()}
    remaining = {label: float(counts[label]) for _, w in groups.values() for label in w}
    taken: set[int] = set()
    claims = {}
    for g in sorted(groups, key=lambda g: (-total[g], g)):
        w = groups[g][1]
        free = [label for label in w if label not in taken]
        own = sum(remaining[label] for label in free)
        claim = own / sum(w[label] for label in free) if free else 0.0
        for label, wl in w.items():
            remaining[label] -= min(remaining[label], claim * wl)
        taken |= set(w)
        claims[g] = (claim, free, own)
    return claims


def _spill_rates(counts, source_reads, source_neighbours, panel_labels):
    """Step 2: per-distance spill rate, a high quantile over clean neighbour pairs.

    A pair is (amplicon, non-panel label within reach of that amplicon only); its ratio is
    the label's reads over the reads the amplicon should carry (every claiming member's
    share of it summed, so an amplicon two members share still calibrates).
    """
    near = defaultdict(set)
    for s, label, d in source_neighbours:
        near[label].add(s)
    ratios = defaultdict(list)
    for s, label, d in source_neighbours:
        parent = source_reads.get(s, 0.0)
        if label in panel_labels or len(near[label]) != 1 or parent < MIN_PARENT_READS:
            continue
        ratios[d].append(counts[label] / parent)
    rates, n_pairs = {}, {}
    for d, default in DEFAULT_RATES.items():
        n_pairs[d] = len(ratios[d])
        rates[d] = (float(np.quantile(ratios[d], RATE_QUANTILE)) if len(ratios[d]) >= MIN_PAIRS
                    else default)
    return rates, n_pairs


def _em(y: np.ndarray, A: np.ndarray, floor: float, keep: np.ndarray, init: np.ndarray):
    """Poisson EM for ``y ~ Poisson(A @ n + floor)``, ``n >= 0``, with ``n[~keep] = 0``."""
    n = np.where(keep, np.maximum(init, 1.0), 0.0)
    colsum = A.sum(axis=0)
    for _ in range(EM_STEPS):
        mu = A @ n + floor
        n = np.where(keep & (colsum > 0), n * (A.T @ (y / mu)) / np.where(colsum > 0, colsum, 1),
                     0.0)
    mu = A @ n + floor
    return n, float(np.sum(special.xlogy(y, mu) - mu)), mu


def _log_bayes_factor(y, column, mu_rest, phi, total, points: int = 80) -> float:
    """log BF of "present with a log-uniform number of reads" over "absent", others fixed
    at their fit without the member; log-likelihood differences deflated by ``phi``."""
    ll0 = np.sum(special.xlogy(y, mu_rest) - mu_rest)
    grid = np.geomspace(1.0, total, points)
    mu = mu_rest[:, None] + column[:, None] * grid[None, :]
    ll = np.sum(special.xlogy(y[:, None], mu) - mu, axis=0)
    return float(special.logsumexp((ll - ll0) / phi) - np.log(points))


def evaluate(sample: str, counts: np.ndarray, table: pd.DataFrame,
             home_of_source: dict[str, int], neighbours, *, alpha: float = DEFAULT_ALPHA,
             prior: float = DEFAULT_PRIOR, floor: float = DEFAULT_FLOOR):
    """Per-member presence evidence for one sample.

    ``counts``: reads per database label. ``table``: the panel translation (genome_id,
    source, weight). ``home_of_source``: each source's home label. ``neighbours``: the
    kernel's ``(source index, label, distance)`` arrays, source indexes into
    ``source_ids``, passed as ``(source_ids, rows, labels, distances)``.

    Returns ``(evidence, calibration)``: one row per member, and the sample's spill rates.
    """
    counts = np.asarray(counts, dtype=np.float64)
    groups = _member_groups(table, home_of_source)
    group_of = {m: g for g, (members, _) in groups.items() for m in members}
    panel_labels = {label for _, w in groups.values() for label in w}
    claims = _greedy(counts, groups)

    source_ids, n_rows, n_labels, n_dist = neighbours
    weight_of = defaultdict(float)                     # (group, source) -> weight
    for g, s, w in zip(table.genome_id, table.source, table.weight):
        weight_of[(group_of[g], s)] = float(w)         # identical within a group
    source_groups = defaultdict(set)
    for (g, s) in weight_of:
        source_groups[s].add(g)
    source_neighbours = [(source_ids[row], int(label), int(d))
                         for row, label, d in zip(n_rows, n_labels, n_dist)]
    neighbour_rows = [(g, label, d, weight_of[(g, src)])
                      for src, label, d in source_neighbours for g in source_groups.get(src, ())]
    source_reads = defaultdict(float)
    for (g, src), w in weight_of.items():
        source_reads[src] += claims[g][0] * w
    rates, n_pairs = _spill_rates(counts, source_reads, source_neighbours, panel_labels)

    # Expected spill per label, by the group it comes from.
    spill = defaultdict(lambda: defaultdict(float))    # label -> group -> reads
    for g, label, d, w in neighbour_rows:
        spill[label][g] += rates[d] * claims[g][0] * w

    # Step 5's model: the panel's own labels, one column per group, each carrying its
    # amplicons plus its spill-over onto them. Non-panel neighbour labels set the rates
    # (step 2) but are not fitted: their spill varies tenfold around the quantile, and
    # that misfit would inflate the overdispersion that deflates every member's evidence.
    names = sorted(groups)
    labels = sorted(panel_labels)
    row_of = {label: i for i, label in enumerate(labels)}
    A = np.zeros((len(labels), len(names)))
    for j, g in enumerate(names):
        for label, w in groups[g][1].items():
            A[row_of[label], j] += w
    for g, label, d, w in neighbour_rows:
        if label in row_of:
            A[row_of[label], names.index(g)] += rates[d] * w
    y = counts[labels]
    init = np.array([claims[g][0] for g in names])
    every = np.ones(len(names), dtype=bool)
    n_fit, _, mu = _em(y, A, floor, every, init)
    dof = max(1, len(labels) - int((n_fit > 0).sum()))
    phi = max(1.0, float(np.sum((y - mu) ** 2 / mu)) / dof)

    rows = []
    for j, g in enumerate(names):
        members, w = groups[g]
        claim, free, own = claims[g]
        noise = sum(floor + sum(v for h, v in spill[label].items() if h != g) for label in free)
        if free:
            p = float(stats.poisson.sf(own - 1, noise))
            lod = int(stats.poisson.isf(alpha, noise)) + 1
            lod_relab = lod / sum(w[label] for label in free) / max(counts.sum(), 1.0)
        else:
            p, lod, lod_relab = np.nan, np.nan, np.nan
        tv = np.nan
        if len(free) >= 2 and own > 0:
            expected = np.array([w[label] for label in free])
            observed = counts[free]
            tv = 0.5 * float(np.abs(observed / observed.sum() - expected / expected.sum()).sum())
        keep = every.copy()
        keep[j] = False
        rest, _, mu_rest = _em(y, A, floor, keep, init)
        log_bf = _log_bayes_factor(y, A[:, j], mu_rest, phi, max(counts.sum(), 2.0))
        posterior = float(special.expit(log_bf + special.logit(prior)))
        status = ("not_identifiable" if not free
                  else "present" if p < alpha else "not_detected")
        for m in members:
            rows.append({
                "sample": sample, "genome_id": m, "group": g, "group_size": len(members),
                "claimed_reads": claim, "free_labels": len(free), "own_reads": own,
                "expected_noise": noise, "p_value": p, "detection_limit_reads": lod,
                "detection_limit_relab": lod_relab, "copy_ratio_tv": tv,
                "log_bayes_factor": log_bf, "overdispersion": phi,
                "presence_posterior": posterior, "status": status})
    calibration = {"sample": sample, **{f"spill_rate_d{d}": r for d, r in rates.items()},
                   **{f"spill_pairs_d{d}": n for d, n in n_pairs.items()}}
    return pd.DataFrame(rows), calibration
