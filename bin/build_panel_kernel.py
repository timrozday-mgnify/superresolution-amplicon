#!/usr/bin/env python
"""Measure a genome panel's confusion kernel against a generic MAPseq database.

    build_panel_kernel.py prepare --panel-amplicons DIR|FASTA --db-amplicons amplicons.fasta -o OUT
    mapseq OUT/sources.fasta amplicons.fasta amplicons.tax > home.mseq
    simulate_amplicon_reads.py --amplicons OUT/sources.fasta --n-per-ref 5000 ... -o sim.fasta
    mapseq sim.fasta amplicons.fasta amplicons.tax > sim.mseq
    build_panel_kernel.py build --prepared OUT --db-amplicons amplicons.fasta \
        --home-mseq home.mseq --sim-mseq sim.mseq -o OUT/panel_kernel.npz
    build_panel_kernel.py align --prepared OUT --db-amplicons amplicons.fasta \
        --home-mseq home.mseq --tau 1 --distance-decay 0.007 -o OUT/align_kernel.npz

``prepare`` cuts each panel genome's V4 copies and writes ``sources.fasta`` (one record per
distinct amplicon, ``v4g_<sha16>``), ``panel_translation.tsv`` (genome_id, source, weight)
and ``sources.tsv`` (source, genomes, in_db). MAPseq runs outside this script, as the same
command used for the observed reads.

``build`` writes the rectangular kernel: sources x database exact-sequence groups, ``K[s, l]``
the fraction of ``s``'s classified simulated reads MAPseq labels ``l``, and ``home[s]`` the
label of the error-free source sequence. ``panel_sources.tsv`` next to the kernel lists each
source's home, its row's top labels and its block (sources whose rows put >= 99% on one label).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import sparse_matrix as sm  # noqa: E402  (needs sys.path)
import subspecies_infer as si  # noqa: E402  (needs sys.path)

log = logging.getLogger("build_panel_kernel")

BLOCK_MASS = 0.99   # the block rule from dev/error_rate_calibration.md


def v4g(seq: str) -> str:
    return "v4g_" + hashlib.sha256(seq.encode()).hexdigest()[:16]


def panel_copies(panel: Path, fwd: str, rev: str, max_mismatch: int,
                 alias: dict[str, str]) -> tuple[dict[str, list[str]], list[str]]:
    """(genome -> its amplifiable V4 copies, genomes dropped for having none).

    Copies keep primers cut off and duplicates kept.

    A directory holds ``<genome>.amplicons.fasta`` files; a single FASTA names the genome
    before the first ``|`` of each header. ``alias`` maps a file stem/genome to a genome id.
    """
    if panel.is_dir():
        records = [(p.name.removesuffix(".amplicons.fasta"), s)
                   for p in sorted(panel.glob("*.amplicons.fasta")) for _, s in si.read_fasta(p)]
    else:
        records = [(si.genome_of_header(h), s) for h, s in si.read_fasta(panel)]
    copies: dict[str, list[str]] = defaultdict(list)
    for genome, seq in records:
        genome = alias.get(genome, genome)
        copies[genome]  # a genome with no amplifiable copy still gets an (empty) entry
        amplicon = si.extract_v4(seq, fwd, rev, max_mismatch)
        if amplicon:
            copies[genome].append(amplicon)
    if not copies:
        raise SystemExit(f"no panel genomes found in {panel}")
    # A genome whose 16S is partial or fragmented across the primer sites has nothing to
    # simulate from, so it cannot be a source. Drop it rather than failing the run: its
    # reads then land in `background`, which is what an out-of-panel genome does anyway.
    dropped = sorted(g for g, c in copies.items() if not c)
    if dropped:
        log.warning("no amplifiable V4 copy in %d/%d panel genome(s); dropped from the "
                    "panel (their reads become `background`): %s",
                    len(dropped), len(copies), ", ".join(dropped))
        for g in dropped:
            del copies[g]
    if not copies:
        raise SystemExit(f"no panel genome in {panel} has an amplifiable V4 copy; "
                         "check the primers, --max-mismatch and the panel references")
    return copies, dropped


def db_groups(db_amplicons: Path) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    """Return database headers, sorted group ids, each header's group index and each
    group's sequence."""
    records = si.read_fasta(db_amplicons)
    ids, first, label_of_ref = np.unique([v4g(s) for _, s in records], return_index=True,
                                         return_inverse=True)
    return ([h for h, _ in records], ids.tolist(), label_of_ref.astype(np.int64),
            [records[i][1] for i in first])


def _home_labels(home_mseq: Path, src: dict[str, int], label_of_hit: dict[str, int]):
    """Each source's MAPseq label (-1 when unhit) and identity, from the sources' own mapping."""
    home = np.full(len(src), -1, dtype=np.int64)
    identity = np.full(len(src), np.nan)
    for query, hit, hit_identity in _mseq_rows(home_mseq):
        if query in src and hit in label_of_hit:
            home[src[query]] = label_of_hit[hit]
            identity[src[query]] = float(hit_identity)
    return home, identity


def prepare(a) -> None:
    copies, dropped = panel_copies(a.panel_amplicons, a.fwd_primer, a.rev_primer,
                                   a.max_mismatch, dict(x.split("=", 1) for x in a.alias))
    translation = pd.DataFrame(
        [{"genome_id": g, "source": v4g(seq), "weight": n / len(seqs)}
         for g, seqs in sorted(copies.items()) for seq, n in Counter(seqs).items()])
    sequence = {v4g(s): s for seqs in copies.values() for s in seqs}
    _, label_ids, _, _ = db_groups(a.db_amplicons)
    sources = (translation.groupby("source").genome_id.agg(";".join).rename("genomes")
               .reset_index())
    sources["in_db"] = sources.source.isin(set(label_ids))
    a.out.mkdir(parents=True, exist_ok=True)
    translation.to_csv(a.out / "panel_translation.tsv", sep="\t", index=False)
    if dropped:
        (a.out / "panel_unamplifiable.txt").write_text("".join(f"{g}\n" for g in dropped))
    sources.to_csv(a.out / "sources.tsv", sep="\t", index=False)
    with open(a.out / "sources.fasta", "w") as fh:
        fh.writelines(f">{s}\n{sequence[s]}\n" for s in sources.source)
    log.info("%d genomes (%d unamplifiable, dropped), %d distinct sources, %d in the "
             "database, %d shared", len(copies), len(dropped), len(sources),
             sources.in_db.sum(), sources.genomes.str.contains(";").sum())


def _mseq_rows(path: Path):
    """Yield ``(query, hit, identity)`` for every MAPseq row, unhit rows with ``hit = ""``."""
    # The pipeline's MAPSEQ process gzips its output.
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t") + ["", "", "", ""]
            yield f[0], f[1], f[3]


def build(a) -> None:
    sources = pd.read_csv(a.prepared / "sources.tsv", sep="\t")
    source_ids = sources.source.tolist()
    src = {s: i for i, s in enumerate(source_ids)}
    headers, label_ids, label_of_ref, _ = db_groups(a.db_amplicons)
    label_of_hit = dict(zip(headers, label_of_ref.tolist()))

    home, home_identity = _home_labels(a.home_mseq, src, label_of_hit)
    if (home < 0).any():
        raise SystemExit("no home label (no database hit) for source(s): "
                         + ", ".join(np.asarray(source_ids)[home < 0]))

    n_simulated = np.zeros(len(src), dtype=np.int64)
    n_unmapped = np.zeros(len(src), dtype=np.int64)
    for query, hit, _ in _mseq_rows(a.sim_mseq):
        s = src.get(query.rsplit(":", 1)[0])
        if s is not None:
            n_simulated[s] += 1
            n_unmapped[s] += not hit
    counts, _ = si.tally_kernel([a.sim_mseq], src, label_of_hit, len(src), len(label_ids),
                                a.min_identity)
    # A source with no labelled simulated read keeps its home row: it stays a possible
    # source and passes through uncorrected, like the square form's identity fallback.
    empty = np.flatnonzero(np.diff(counts.indptr) == 0)
    if len(empty):
        log.warning("%d source(s) had no labelled simulated read; using the home row: %s",
                    len(empty), ", ".join(source_ids[i] for i in empty))
        counts = counts + sparse.csr_array((np.ones(len(empty)), (empty, home[empty])),
                                           shape=counts.shape)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    kernel = sparse.csr_array(sparse.diags_array(1.0 / totals) @ counts)

    provenance = {"prepared": str(a.prepared), "home_mseq": str(a.home_mseq),
                  "sim_mseq": str(a.sim_mseq), "min_identity": a.min_identity,
                  **dict(x.split("=", 1) for x in a.provenance)}
    sm.write_kernel(a.out, kernel, source_ids, label_ids, home, ref_headers=headers,
                    label_of_ref=label_of_ref,
                    db_amplicons_sha256=hashlib.sha256(a.db_amplicons.read_bytes()).hexdigest(),
                    n_simulated=n_simulated, n_unmapped=n_unmapped, provenance=provenance)
    source_table(sources, kernel, home, home_identity, label_ids, n_simulated, n_unmapped).to_csv(
        a.out.with_name("panel_sources.tsv"), sep="\t", index=False, float_format="%.4f")
    log.info("kernel %d sources x %d labels; %d source(s) have a home other than themselves",
             *kernel.shape, sum(label_ids[h] != s for s, h in zip(source_ids, home)))


def align(a) -> None:
    """The alignment kernel: ``K[s, l]`` proportional to ``w(l) * c ** d(s, l)`` over the
    database groups within ``--tau`` edits of source ``s``, the ``--mode align`` tie cluster
    with sources and labels from different sets. ``w`` is the ambiguity weight.

    The error-free read's entry (weight 1, distance 0) sits on the home label: MAPseq's
    label for the source when ``--home-mseq`` is given, else the source's own group. The
    source's own group is not an entry of its own, so a relabelled exact match
    (``2acb -> 81c3``) is not undone by alignment. At ``tau >= 1`` the distances are stored,
    so ``--infer-distance-decay`` can refit ``c`` per sample.
    """
    import build_mismapping_align as bma

    sources = pd.read_csv(a.prepared / "sources.tsv", sep="\t")
    source_ids = sources.source.tolist()
    src = {s: i for i, s in enumerate(source_ids)}
    sequence = dict(si.read_fasta(a.prepared / "sources.fasta"))
    headers, label_ids, label_of_ref, label_seqs = db_groups(a.db_amplicons)
    own = {label: i for i, label in enumerate(label_ids)}
    if a.home_mseq:
        home, home_identity = _home_labels(a.home_mseq, src,
                                           dict(zip(headers, label_of_ref.tolist())))
    else:
        home = np.array([own.get(s, -1) for s in source_ids], dtype=np.int64)
        home_identity = np.where(home >= 0, 1.0, np.nan)
    if (home < 0).any():
        raise SystemExit("no home label for source(s) (not in the database, or unhit in "
                         "--home-mseq): " + ", ".join(np.asarray(source_ids)[home < 0]))
    if not 0.0 < a.distance_decay <= 1.0:
        raise SystemExit("--distance-decay must be in (0, 1]")
    ambiguity = bma.ambiguity_weights(label_seqs, a.ambiguity_weight)

    rows, cols, weights, dists = [], [], [], []
    for s, source in enumerate(source_ids):
        entries = {}
        # ponytail: brute force over every group (26 x 86,557 in ~10 s); use
        # bma.pigeonhole_candidates when panels reach hundreds of sources.
        for label, target in enumerate(label_seqs):
            d = bma.bounded_iupac_distance(sequence[source], target, a.tau)
            if d <= a.tau and label != own.get(source):
                entries[label] = ((1.0 if ambiguity is None else ambiguity[label])
                                  * a.distance_decay ** d, d)
        entries[home[s]] = (1.0, 0)       # overrides the home's own neighbour entry
        total = sum(w for w, _ in entries.values())
        for label, (w, d) in entries.items():
            rows.append(s)
            cols.append(label)
            weights.append(w / total)
            dists.append(d)
    shape = (len(source_ids), len(label_ids))
    kernel = sparse.csr_array((weights, (rows, cols)), shape=shape)
    strata = None
    if a.tau >= 1:
        distances = sparse.csr_array((np.asarray(dists, dtype=np.float64) + 1.0, (rows, cols)),
                                     shape=shape)
        distances.data -= 1.0
        strata = (distances, a.distance_decay)
    provenance = {"method": "align", "prepared": str(a.prepared), "tau": a.tau,
                  "distance_decay": a.distance_decay, "ambiguity_weight": a.ambiguity_weight,
                  "home_mseq": None if a.home_mseq is None else str(a.home_mseq)}
    sm.write_kernel(a.out, kernel, source_ids, label_ids, home, ref_headers=headers,
                    label_of_ref=label_of_ref,
                    db_amplicons_sha256=hashlib.sha256(a.db_amplicons.read_bytes()).hexdigest(),
                    provenance=provenance, strata=strata)
    zeros = np.zeros(len(source_ids), dtype=np.int64)
    source_table(sources, kernel, home, home_identity, label_ids, zeros, zeros).to_csv(
        a.out.with_name("panel_sources.tsv"), sep="\t", index=False, float_format="%.4f")
    log.info("align kernel tau=%d c=%g: %d sources x %d labels, %d nonzeros", a.tau,
             a.distance_decay, *kernel.shape, kernel.nnz)


def source_table(sources: pd.DataFrame, kernel: sparse.csr_array, home: np.ndarray,
                 home_identity: np.ndarray, label_ids: list[str], n_simulated: np.ndarray,
                 n_unmapped: np.ndarray) -> pd.DataFrame:
    """One row per source: home, row summary and block."""
    top, block = [], []
    for s in range(kernel.shape[0]):
        row = kernel[[s]].tocoo()
        order = np.argsort(-row.data)[:3]
        top.append(";".join(f"{label_ids[row.col[i]]}:{row.data[i]:.3f}" for i in order))
        dominant = len(order) and row.data[order[0]] >= BLOCK_MASS
        block.append(label_ids[row.col[order[0]]] if dominant else sources.source[s])
    out = sources.assign(
        home_label=[label_ids[h] for h in home], identity_to_home=home_identity,
        home_mass=sm.home_entries(kernel, home), top_labels=top,
        unmapped_fraction=np.divide(n_unmapped, n_simulated, out=np.zeros(len(home)),
                                    where=n_simulated > 0),
        block=block)
    # Name each block by its first source so the id reads as "these sources collide".
    out["block"] = out.groupby("block").source.transform("first")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--panel-amplicons", type=Path, required=True,
                   help="directory of <genome>.amplicons.fasta, or one FASTA with genome|... headers")
    p.add_argument("--db-amplicons", type=Path, required=True)
    p.add_argument("--fwd-primer", default=si.DEFAULT_FWD_PRIMER)
    p.add_argument("--rev-primer", default=si.DEFAULT_REV_PRIMER)
    p.add_argument("--max-mismatch", type=int, default=2)
    p.add_argument("--alias", action="append", default=[],
                   help="file_stem=genome_id (repeatable)")
    p.add_argument("-o", "--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--prepared", type=Path, required=True)
    b.add_argument("--db-amplicons", type=Path, required=True)
    b.add_argument("--home-mseq", type=Path, required=True)
    b.add_argument("--sim-mseq", type=Path, required=True)
    b.add_argument("--min-identity", type=float, default=None)
    b.add_argument("--provenance", action="append", default=[],
                   help="key=value recorded in the kernel (error model, rates, seed; repeatable)")
    b.add_argument("-o", "--out", type=Path, required=True)
    al = sub.add_parser("align", help="kernel from source-to-database edit distance, no simulation")
    al.add_argument("--prepared", type=Path, required=True)
    al.add_argument("--db-amplicons", type=Path, required=True)
    al.add_argument("--home-mseq", type=Path, default=None,
                    help="MAPseq of sources.fasta; without it each source's home is its own group")
    al.add_argument("--tau", type=int, default=1)
    al.add_argument("--distance-decay", type=float, default=0.007)
    al.add_argument("--ambiguity-weight", type=float, default=0.3)
    al.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    {"prepare": prepare, "build": build, "align": align}[a.cmd](a)


if __name__ == "__main__":
    main()
