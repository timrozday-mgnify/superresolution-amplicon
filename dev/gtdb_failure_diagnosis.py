#!/usr/bin/env python
# UNMAINTAINED: imports code removed with the square mis-mapping path (panel-only
# pipeline). Kept as the record behind its .md; it no longer runs against bin/.
"""Was a failed GTDB-scale fit the forward model's fault, or the inference's?

Evidence §1–2 of ``docs/gtdb_inference_recovery_plan.md``. Both subcommands take an
archived matrix bundle and the batch's DADA2 ASVs. The GTDB-mapped ``.mseq`` files are not
archived, so observed V4-group counts are reconstructed by placing each ASV on a GTDB V4
sequence by exact match.

``fit``     Best-achievable fit. Per sample: exact-match coverage, then the
            maximum-likelihood mixture (EM) under the matrix, in genome space and in
            V4-group space, plus V4-group space under the per-distinct-sequence kernel.
            Small TVs mean the forward model *can* describe the data, so a failed
            posterior is an inference problem.
``replay``  Forward-model replay. Draw a sample's MAPseq labels from the matrix itself
            (``y @ K``; each label's reads split evenly over its member references, as
            MAPseq splits exact ties), then run the real ``infer_composition.py`` and
            ``check_composition_fit.py`` in each ``--space`` (``genome``, and
            ``--infer-space v4_group``). The model is right by construction, so a misfit
            is the inference's fault. With no samples named, every sample with
            ``--min-reads`` placed reads is replayed.
``blocks``  What these labels can identify at all: sources the mapper sends to the same
            labels form one block, and the fit is scored at that resolution too.
``real``    Plan Phases 4-5. Run the proposed GTDB configuration (``v4_group``, no presence
            gate, and the decay fit when the matrix carries distances) on the batch's real
            GTDB MAPseq output, and
            score each sample's group profile against the ASV exact-match profile, the
            custom-database run aggregated to V4 groups by exact sequence, and (optional)
            the Phase 3 replay's fit. Records per-sample inference time and peak RSS.

    python dev/gtdb_failure_diagnosis.py fit --bundle B --asv-dir A --out O
    python dev/gtdb_failure_diagnosis.py replay --bundle B --asv-dir A --out O [D2 ...] \\
        [--space genome v4_group] [--jobs 4] [--infer-args "--alpha 0.05"]
    python dev/gtdb_failure_diagnosis.py real --bundle B --asv-dir A --out O \\
        --mseq-dir M --custom-dir C --custom-reference R --taxonomy T [--replay-dir P]

``--bundle`` is ``results/mismapping/<matrix key>/`` (``mismapping_matrix.npz`` plus
``reference/``); ``--asv-dir`` is ``results/`` from ``<batch>_results.tar.gz``
(``<sample>/asv/<sample>_asv_seqs.fasta``). For ``real``, ``--mseq-dir`` holds
``<sample>/<sample>.obs.mseq.gz`` and ``--custom-dir`` ``<sample>/<sample>.inferred_composition.csv``
(``results/mapseq`` and ``results/composition`` of the two sr-amp archives);
``--custom-reference`` is the custom bundle's ``reference/``. Needs ~6 GB RAM for GTDB r232.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "bin"))

import infer_composition as ic  # noqa: E402  (needs sys.path)
import sparse_matrix as sm  # noqa: E402  (needs sys.path)
import subspecies_infer as si  # noqa: E402  (needs sys.path)

IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "R": "[AG]", "Y": "[CT]", "S": "[CG]",
         "W": "[AT]", "K": "[GT]", "M": "[AC]", "B": "[CGT]", "D": "[AGT]", "H": "[ACT]",
         "V": "[ACG]", "N": "[ACGT]"}
RC_806R = re.compile("".join(IUPAC[b] for b in "ATTAGAWACCCBNGTAGTCC"))
SUFFIX = 120


class Bundle:
    """The archived grouped matrix, its reference set, and a V4-sequence placement index."""

    def __init__(self, path: Path) -> None:
        a = np.load(path / "mismapping_matrix.npz")
        self.S = sparse.csr_array((a["data"], a["indices"], a["indptr"]), shape=tuple(a["shape"]))
        self.group = a["group"]
        self.refs = a["refseqs"].tobytes().decode().split("\n")
        # A measured (simulate-and-map) matrix carries neither; only ``fit`` needs them.
        self.decay = float(a["decay"]) if "decay" in a.files else None
        self.distances = (a["distances"].astype(np.float64) if "distances" in a.files
                          else None)
        self.path = path
        G = self.S.shape[0]
        self.size = np.bincount(self.group, minlength=G).astype(np.float64)
        # Group -> group mass: the observation distribution of one read from each V4 group.
        self.K = sparse.csr_array((self.S.data * self.size[self.S.indices], self.S.indices,
                                   self.S.indptr), shape=(G, G))
        self.pos = {r: i for i, r in enumerate(self.refs)}
        self.seq: list[str | None] = [None] * G
        for header, sequence in si.read_fasta(path / "reference" / "amplicons.fasta"):
            g = self.group[self.pos[header]]
            if self.seq[g] is None:
                self.seq[g] = sequence
        # The stable ids infer_composition._v4_groups gives the same sequences.
        self.ids = ["v4g_" + hashlib.sha256(s.encode()).hexdigest()[:16] for s in self.seq]
        self.by_suffix = collections.defaultdict(list)
        for g, s in enumerate(self.seq):
            self.by_suffix[s[-SUFFIX:]].append(g)

    def place(self, asv: str) -> int | None:
        """Return the V4 group an ASV matches exactly (primer-trimmed, 3' anchored), or None."""
        segment = asv.split("N")[0]              # merged pairs are N-padded; keep the forward read
        primer = RC_806R.search(segment)
        if primer is None:
            return None
        body = segment[:primer.start()]
        for g in self.by_suffix.get(body[-SUFFIX:], ()):
            s = self.seq[g]
            n = min(len(s), len(body))
            if s[-n:] == body[-n:] and abs(len(s) - len(body)) <= 6:
                return g
        return None

    def distinct_kernel(self) -> sparse.csr_array:
        """Group -> group kernel weighting each distinct neighbour by ``c**d``, not by size."""
        if self.distances is None:
            raise SystemExit("this matrix has no distance strata to re-weight")
        row = np.repeat(np.arange(self.S.shape[0]), np.diff(self.S.indptr))
        w = np.where(row != self.S.indices, self.decay ** self.distances, 1.0)
        K = sparse.csr_array((w, self.S.indices, self.S.indptr), shape=self.S.shape)
        return sparse.csr_array(sparse.diags(1 / np.asarray(K.sum(axis=1)).ravel()) @ K)


def asv_samples(asv_dir: Path) -> list[str]:
    return sorted(p.name for p in asv_dir.iterdir() if (p / "asv").is_dir())


def asv_counts(asv_dir: Path, sample: str, bundle: Bundle) -> tuple[np.ndarray, int]:
    """Return one sample's exact-placed V4-group counts and its total ASV reads."""
    fasta = next((asv_dir / sample / "asv").glob("*_asv_seqs.fasta"))
    table = next((asv_dir / sample / "asv" / "16S-V4").glob("*read_counts.tsv"))
    counts = dict(line.split() for line in table.read_text().splitlines()[1:])
    y = np.zeros(bundle.S.shape[0])
    total = 0
    for name, sequence in si.read_fasta(fasta):
        n = int(counts.get(name, 0))
        total += n
        g = bundle.place(sequence)
        if g is not None:
            y[g] += n
    return y, total


def tv(y: np.ndarray, p: np.ndarray) -> float:
    return float(0.5 * np.abs(y / y.sum() - p / p.sum()).sum())


def em(A: sparse.csr_array, y: np.ndarray, iters: int) -> np.ndarray:
    """Maximum-likelihood mixture weights over the rows of ``A`` for label counts ``y``."""
    r = np.full(A.shape[0], 1 / A.shape[0])
    for _ in range(iters):
        p = r @ A
        r = r * (A @ np.divide(y, p, out=np.zeros_like(p), where=p > 0)) / y.sum()
    return r


def fit(args: argparse.Namespace) -> None:
    bundle = Bundle(args.bundle)
    K_distinct = bundle.distinct_kernel()
    T = pd.read_csv(args.bundle / "reference" / "translation_table.tsv", sep="\t")
    genome, genomes = pd.factorize(T.genome_id)
    ref_group = bundle.group[np.fromiter((bundle.pos[r] for r in T.refseq), np.int64, len(T))]
    shape = (len(genomes), bundle.S.shape[0])
    W = sparse.csr_array((T.weight.to_numpy(), (genome, ref_group)), shape=shape)   # copy weights
    Wn = sparse.csr_array((np.ones(len(T)), (genome, ref_group)), shape=shape)      # reference counts

    rows = []
    for sample in asv_samples(args.asv_dir):
        y, total = asv_counts(args.asv_dir, sample, bundle)
        N = y.sum()
        row = {"sample": sample, "asv_reads": total, "exact_fraction": N / total if total else 0.0}
        if N >= args.min_reads:
            sources = np.unique(bundle.K[:, np.flatnonzero(y)].nonzero()[0])
            active = np.unique(W[:, sources].nonzero()[0])
            A = (W[active] @ bundle.K).tocsr()
            # SVI's initial point as infer_composition builds it: MAPseq spreads a group's
            # reads evenly over its members, summed per genome, floored at 1e-4.
            theta_obs = (Wn[active] @ (y / bundle.size)) / N
            init = np.clip(theta_obs, 1e-4, None)
            init /= init.sum()
            row |= {
                "active_groups": len(sources),
                "active_genomes": len(active),
                "tv_group": tv(y, em(bundle.K[sources], y, args.iters) @ bundle.K[sources]),
                "tv_group_distinct": tv(y, em(K_distinct[sources], y, args.iters)
                                        @ K_distinct[sources]),
                "tv_genome": tv(y, em(A, y, args.iters) @ A),
                "tv_genome_init": tv(y, init @ A),
                "init_floor_mass": float(init[theta_obs < 1e-4].sum()),
                "prior_mass_share": 0.5 * len(active) / (0.5 * len(active) + N),
            }
        rows.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "fit.csv", index=False)
    print(f"exact-match coverage: mean {df.exact_fraction.mean():.4%}, "
          f"min {df.exact_fraction.min():.4%}")
    print(df.dropna().describe().round(4).to_string())


def write_replay(bundle: Bundle, y: np.ndarray, rng: np.random.Generator, path: Path) -> None:
    """Draw MAPseq labels from ``y @ K``, each on a uniformly chosen member reference."""
    p = (y / y.sum()) @ bundle.K
    labels = rng.multinomial(int(y.sum()), p / p.sum())
    order = np.argsort(bundle.group, kind="stable")
    start = np.r_[0, np.cumsum(bundle.size.astype(np.int64))]
    hits = np.concatenate([rng.choice(order[start[g]:start[g + 1]], size=labels[g])
                           for g in np.flatnonzero(labels)])
    with open(path, "w") as handle:
        handle.write("#query\tdbhit\tbitscore\tidentity\n")
        for q, r in enumerate(hits):
            handle.write(f"q{q}\t{bundle.refs[r]}\t250\t1.0\n")


def run_timed(cmd: list[str], log: Path) -> tuple[float, float]:
    """Run ``cmd`` with output to ``log``; return its wall seconds and peak RSS in GB."""
    start = time.monotonic()
    with open(log, "w") as handle:
        proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT)
        _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, f"see {log}")
    rss = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)  # bytes on macOS, KiB on Linux
    return time.monotonic() - start, rss / 1e9


def infer_and_check(args: argparse.Namespace, sample: str, mseq: Path, out: Path, space: str,
                    extra: list[str]) -> dict:
    """Run the real ``infer_composition.py`` and ``check_composition_fit.py`` on one ``.mseq``."""
    amplicon_dir, matrix = args.bundle / "reference", args.bundle / "mismapping_matrix.npz"
    out.mkdir(parents=True, exist_ok=True)
    wall, rss = run_timed([sys.executable, str(_REPO / "bin" / "infer_composition.py"),
                           "--amplicon-dir", str(amplicon_dir), "--mismapping-matrix", str(matrix),
                           "--obs-mseq", str(mseq), "--sample-id", sample, "--infer-space", space,
                           "-o", str(out), *extra], out / "infer.log")
    table = "inferred_v4_groups.csv" if space == "v4_group" else "inferred_composition.csv"
    run_timed([sys.executable, str(_REPO / "bin" / "check_composition_fit.py"),
               "--composition", str(out / table),
               "--posterior-draws", str(out / "posterior_draws.npz"),
               "--obs-mseq", str(mseq), "--amplicon-dir", str(amplicon_dir),
               "--mismapping-matrix", str(matrix), "-o", str(out / "fit.json")], out / "check.log")
    fit_json = json.loads((out / "fit.json").read_text())
    inference = pd.read_csv(out / "inference_diagnostics.csv").iloc[0]
    grouped = fit_json["unique_v4_group"] or {}
    return {
        "genomes_fitted": inference.n_genomes_fitted,
        "groups_fitted": inference.n_groups_fitted,
        "fit_status": fit_json["fit_status"],
        "group_tv": grouped.get("observed_expected_tv"),
        "group_ppc_percentile": grouped.get("posterior_predictive_percentile"),
        "draw_tv_observed_q50": grouped.get("observed_statistic", {}).get("q50"),
        "draw_tv_replicated_q50": grouped.get("replicated_statistic", {}).get("q50"),
        "infer_wall_s": wall, "infer_max_rss_gb": rss,
    }


def replay_sample(args: argparse.Namespace, bundle: Bundle, sample: str) -> list[dict]:
    """Replay one sample in every requested space; one result row per space."""
    y, _ = asv_counts(args.asv_dir, sample, bundle)
    (args.out / sample).mkdir(parents=True, exist_ok=True)
    mseq = args.out / sample / "replay.mseq"
    write_replay(bundle, y, np.random.default_rng(args.seed), mseq)
    rows = []
    for space in args.space:
        rows.append({"sample": sample, "space": space, "reads": int(y.sum())}
                    | infer_and_check(args, sample, mseq, args.out / sample / space, space,
                                      shlex.split(args.infer_args)))
        print(rows[-1], flush=True)
    return rows


def replay(args: argparse.Namespace) -> None:
    bundle = Bundle(args.bundle)
    samples = args.samples or [s for s in asv_samples(args.asv_dir)
                               if asv_counts(args.asv_dir, s, bundle)[0].sum() >= args.min_reads]
    # Threads only wait on the two subprocesses per space; that is where the time goes.
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = [row for rows in pool.map(lambda s: replay_sample(args, bundle, s), samples)
                for row in rows]
    df = pd.DataFrame(rows).sort_values(["space", "sample"])
    df.to_csv(args.out / "replay.csv", index=False)
    print(df.to_string(index=False))
    for space, g in df.groupby("space"):
        print(f"{space}: {(g.fit_status == 'ok').mean():.1%} ok of {len(g)}; median group TV "
              f"{g.group_tv.median():.4f}, per-draw observed "
              f"{g.draw_tv_observed_q50.median():.4f} / replicated "
              f"{g.draw_tv_replicated_q50.median():.4f}")


# The plan's proposed GTDB configuration, beyond --infer-space v4_group and --taxonomy.
REAL_ARGS = ["--no-presence", "--seed", "42"]


def real_args(bundle: Path) -> list[str]:
    """``REAL_ARGS`` plus the per-sample decay fit, which only an alignment build supports.

    A measured (simulate-and-map) matrix carries no distance strata: there is nothing to
    re-decay, because the leak was observed rather than derived from a decay.
    """
    with np.load(bundle / "mismapping_matrix.npz") as archive:
        strata = "distances" in archive.files
    return REAL_ARGS + (["--infer-distance-decay"] if strata else [])


def mseq_reads(path: Path) -> int:
    with gzip.open(path, "rt") as handle:
        return sum(1 for line in handle if line[0] != "#")


def tv_profiles(a: pd.Series, b: pd.Series) -> float:
    """TV between two abundance profiles indexed by V4 group id (each normalised)."""
    a, b = a.align(b, fill_value=0.0)
    return tv(a.to_numpy(), b.to_numpy())


def custom_profile(reference: Path, composition: Path, bundle: Bundle) -> tuple[pd.Series, float]:
    """A genome-space run's read mass per V4 group, and the share on sequences not in the bundle.

    The translation weights split a genome's read fraction over its copies, so a group's
    mass is the sum of ``theta_g * weight`` over references carrying its sequence.
    """
    T = pd.read_csv(reference / "translation_table.tsv", sep="\t")
    seq = dict(si.read_fasta(reference / "amplicons.fasta"))
    theta = pd.read_csv(composition).set_index("genome_id").inferred_mean
    mass = T.weight * T.genome_id.map(theta)
    ids = T.refseq.map(seq).map(dict(zip(bundle.seq, bundle.ids)))
    return mass.groupby(ids).sum(), float(mass[ids.isna()].sum() / mass.sum())


def v4_profile(path: Path) -> pd.Series:
    return pd.read_csv(path, usecols=["v4_group_id", "inferred_mean"]).set_index(
        "v4_group_id").inferred_mean


def real_sample(args: argparse.Namespace, bundle: Bundle, sample: str) -> dict:
    mseq = args.mseq_dir / sample / f"{sample}.obs.mseq.gz"
    out = args.out / sample
    row = {"sample": sample, "reads": mseq_reads(mseq)} | infer_and_check(
        args, sample, mseq, out, "v4_group",
        [*real_args(args.bundle), "--taxonomy", str(args.taxonomy),
         *shlex.split(args.infer_args)])
    fitted = v4_profile(out / "inferred_v4_groups.csv")
    y, _ = asv_counts(args.asv_dir, sample, bundle)
    asv = pd.Series(y, index=bundle.ids)[y > 0]
    custom, unmatched = custom_profile(
        args.custom_reference, args.custom_dir / sample / f"{sample}.inferred_composition.csv",
        bundle)
    draws = np.load(out / "posterior_draws.npz")
    # Raw MAPseq labels per group: how far the mapper alone already is from the ASVs.
    labels = np.zeros(len(bundle.ids))
    with gzip.open(mseq, "rt") as handle:
        for line in handle:
            if line[0] != "#":
                labels[bundle.group[bundle.pos[line.split("\t")[1]]]] += 1
    row |= {"decay_q50": float(np.median(draws["c"])) if "c" in draws else None,
            "tv_asv": tv_profiles(fitted, asv), "tv_custom": tv_profiles(fitted, custom),
            "tv_custom_asv": tv_profiles(custom, asv), "custom_unmatched": unmatched,
            "tv_mapseq_asv": tv(labels, y),
            "asv_on_unhit_groups": float(y[labels == 0].sum() / y.sum())}
    replay_table = args.replay_dir / sample / "v4_group" / "inferred_v4_groups.csv" \
        if args.replay_dir else None
    if replay_table and replay_table.exists():
        row["tv_replay"] = tv_profiles(fitted, v4_profile(replay_table))
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    return row


def real(args: argparse.Namespace) -> None:
    bundle = Bundle(args.bundle)
    samples = args.samples or [
        p.name for p in sorted(args.mseq_dir.iterdir())
        if (p / f"{p.name}.obs.mseq.gz").exists()
        and mseq_reads(p / f"{p.name}.obs.mseq.gz") >= args.min_reads]
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda s: real_sample(args, bundle, s), samples))
    df = pd.DataFrame(rows).sort_values("sample")
    df.to_csv(args.out / "real.csv", index=False)
    print(df.round(4).to_string(index=False))
    med = df.median(numeric_only=True)
    print(f"ok: {(df.fit_status == 'ok').mean():.1%} of {len(df)} (accept >= 95%)")
    print(f"median group TV to ASV {med.tv_asv:.4f}, to custom {med.tv_custom:.4f} "
          f"(accept <= 0.05 each); custom vs ASV {med.tv_custom_asv:.4f}")
    if "tv_replay" in df:
        print(f"median group TV to the Phase 3 replay fit {med.tv_replay:.4f}")
    print(f"per-sample inference: median {med.infer_wall_s:.0f} s (max {df.infer_wall_s.max():.0f}), "
          f"peak RSS median {med.infer_max_rss_gb:.2f} GB (max {df.infer_max_rss_gb.max():.2f})")


def blocks(args: argparse.Namespace) -> None:
    """What is identifiable from these labels, and how close is the fit at that resolution?

    Two source groups the mapper sends to the same labels cannot be told apart by any
    inference over those labels, however good ``M`` is. Blocks are the connected components
    of "label distributions within TV ``--block-tv``" over the sample's observed labels
    (plus a residual column for mass on labels it never saw). The fit, the raw MAPseq
    labels and the ASV profile are then compared at group level and at block level.
    """
    bundle = Bundle(args.bundle)
    matrix, group = sm.read_grouped(args.bundle / "mismapping_matrix.npz", bundle.refs)
    ids, _ = ic._v4_groups(args.bundle / "reference", bundle.refs)
    size = np.bincount(group).astype(np.float64)
    K = sparse.csr_array((matrix.data * size[matrix.indices], matrix.indices, matrix.indptr),
                         shape=matrix.shape)
    order = np.array([{i: k for k, i in enumerate(ids)}[i] for i in bundle.ids])
    rows = []
    for sample in args.samples or sorted(p.name for p in args.mseq_dir.iterdir()
                                         if (p / f"{p.name}.obs.mseq.gz").exists()):
        counts, _ = asv_counts(args.asv_dir, sample, bundle)
        asv = np.zeros(len(ids))
        asv[order] = counts
        labels = np.zeros(len(ids))
        with gzip.open(args.mseq_dir / sample / f"{sample}.obs.mseq.gz", "rt") as handle:
            for line in handle:
                if line[0] != "#":
                    labels[group[bundle.pos[line.split("\t")[1]]]] += 1
        if labels.sum() < args.min_reads:
            continue
        hit = np.flatnonzero(labels)
        sources = np.unique(K[:, hit].nonzero()[0])
        seen = K[sources][:, hit].toarray()
        seen = np.c_[seen, 1.0 - seen.sum(1)]
        distance = 1.0 - np.array([np.minimum(seen, row).sum(1) for row in seen])
        n, label = connected_components(sparse.csr_matrix(distance < args.block_tv),
                                        directed=False)
        block = np.full(len(ids), -1)
        block[sources] = label

        def pool(v: np.ndarray) -> np.ndarray:
            out = np.zeros(n + 1)
            index = np.flatnonzero(v)
            np.add.at(out, np.where(block[index] >= 0, block[index], n), v[index])
            return out

        fit = v4_profile(args.fit_dir / sample / "inferred_v4_groups.csv").reindex(
            ids).fillna(0).to_numpy()
        rows.append({"sample": sample, "sources": len(sources), "blocks": n,
                     "largest_block": int(np.bincount(label).max()),
                     "tv_group_fit_asv": tv(fit, asv),
                     "tv_block_fit_asv": tv(pool(fit), pool(asv)),
                     "tv_block_labels_asv": tv(pool(labels), pool(asv))})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "blocks.csv", index=False)
    print(df.describe().round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("fit", fit), ("replay", replay), ("real", real),
                       ("blocks", blocks)):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--asv-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--min-reads", type=int, default=1000)
    sub.choices["fit"].add_argument("--iters", type=int, default=3000)
    rp = sub.choices["replay"]
    rp.add_argument("samples", nargs="*", help="default: every sample with --min-reads")
    rp.add_argument("--space", nargs="+", choices=["genome", "v4_group"],
                    default=["genome", "v4_group"])
    rp.add_argument("--infer-args", default="", help="extra infer_composition.py flags")
    rp.add_argument("--jobs", type=int, default=1)
    rp.add_argument("--seed", type=int, default=0)
    rl = sub.choices["real"]
    rl.add_argument("samples", nargs="*", help="default: every sample with --min-reads mapped")
    rl.add_argument("--mseq-dir", type=Path, required=True)
    rl.add_argument("--custom-dir", type=Path, required=True)
    rl.add_argument("--custom-reference", type=Path, required=True)
    rl.add_argument("--taxonomy", type=Path, required=True)
    rl.add_argument("--replay-dir", type=Path, help="Phase 3 replay --out, for tv_replay")
    rl.add_argument("--infer-args", default="", help="extra infer_composition.py flags")
    rl.add_argument("--jobs", type=int, default=1)
    bl = sub.choices["blocks"]
    bl.add_argument("samples", nargs="*")
    bl.add_argument("--mseq-dir", type=Path, required=True)
    bl.add_argument("--fit-dir", type=Path, required=True, help="a `real` run's --out")
    bl.add_argument("--block-tv", type=float, default=0.10,
                    help="label-distribution TV below which two sources share a block")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
