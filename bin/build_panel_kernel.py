#!/usr/bin/env python
"""Measure a genome panel's confusion kernel against a generic MAPseq database.

    build_panel_kernel.py prepare --panel-amplicons DIR|FASTA --db-amplicons amplicons.fasta -o OUT
    build_panel_kernel.py prepare --whole-database amplicon_dir -o OUT
    mapseq OUT/sources.fasta amplicons.fasta amplicons.tax > home.mseq
    simulate_amplicon_reads.py --amplicons OUT/sources.fasta --n-per-ref 5000 ... -o sim.fasta
    mapseq sim.fasta amplicons.fasta amplicons.tax > sim.mseq
    build_panel_kernel.py build --prepared OUT --db-amplicons amplicons.fasta \
        --home-mseq home.mseq --sim-mseq sim.mseq -o OUT/panel_kernel.npz
    build_panel_kernel.py align --prepared OUT --db-amplicons amplicons.fasta \
        --home-mseq home.mseq --tau 1 --distance-decay 0.007 -o OUT/align_kernel.npz

``prepare`` cuts each panel genome's V4 copies, or reuses every V4 group from an extracted
database, and writes ``sources.fasta`` (one record per distinct amplicon, ``v4g_<sha16>``),
``panel_translation.tsv`` (genome_id, source, weight) and ``sources.tsv`` (source, genomes,
in_db). A taxon entry (``--panel-taxa``) contributes each database V4 group under its lineage
as a free member ``<entry>::<v4g>`` of weight 1; inference sums members back into the entry.
MAPseq runs outside this script, as the same command used for the observed reads.

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
ENTRY_SEP = "::"    # a taxon member is "<entry>::<v4g>"; the entry is the part before it


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


def _ranks(lineage: str) -> tuple[str, ...]:
    return tuple(r.strip() for r in lineage.strip().rstrip(";").split(";"))


def _resolve_taxon(taxon: str, prefixes: set[tuple[str, ...]]) -> tuple[str, ...]:
    """A lineage prefix (``;``-separated) or a bare name that ends exactly one prefix."""
    ranks = _ranks(taxon)
    if len(ranks) > 1:
        if ranks in prefixes:
            return ranks
        candidates = sorted(p for p in prefixes if p[-1].lower() == ranks[-1].lower())
    else:
        candidates = sorted(p for p in prefixes if p[-1] == ranks[0])
        if len(candidates) == 1:
            return candidates[0]
    listed = "; ".join(";".join(c) for c in candidates[:20]) or "none"
    if len(ranks) > 1:
        raise SystemExit(f"taxon {taxon!r} is no database lineage prefix; lineages ending in "
                         f"{ranks[-1]!r}: {listed}")
    raise SystemExit(f"taxon {taxon!r} ends {len(candidates)} database lineage prefixes, "
                     f"not exactly one; give the full lineage. Candidates: {listed}")


def taxon_members(taxa: list[tuple[str, str]], headers: list[str], label_ids: list[str],
                  label_of_ref: np.ndarray, lineage_of: dict[str, str],
                  excluded: set[str], max_sources: int) -> dict[str, list[str]]:
    """Entry id -> the database V4 groups it owns.

    A group belongs to a taxon entry when any member sequence's lineage lies under it, but
    each sequence counts only for the most specific entry above it, so nested taxa split
    ("other Bacteroides"). A group in ``excluded`` (a genome entry's source, or a sequence
    MAPseq cannot place) belongs to no taxon. A group whose sequences fall under two
    unnested entries is shared by both.
    """
    lineages = {h: _ranks(lineage_of[h]) for h in headers if h in lineage_of}
    if not lineages:
        raise SystemExit("--db-taxonomy has no lineage for any database amplicon header")
    prefixes = {r[:i] for r in set(lineages.values()) for i in range(1, len(r) + 1)}
    entries = {entry: _resolve_taxon(taxon, prefixes) for entry, taxon in taxa}
    if len(set(entries.values())) < len(entries):
        raise SystemExit("two --panel-taxa entries resolve to the same lineage")
    by_prefix = {p: e for e, p in entries.items()}
    depth = sorted({len(p) for p in by_prefix}, reverse=True)
    groups: dict[str, set[str]] = defaultdict(set)
    for header, label in zip(headers, label_of_ref):
        ranks = lineages.get(header)
        group = label_ids[label]
        if ranks is None or group in excluded:
            continue
        owner = next((by_prefix[ranks[:d]] for d in depth if ranks[:d] in by_prefix), None)
        if owner is not None:
            groups[owner].add(group)
    members = {}
    for entry, prefix in entries.items():
        found = sorted(groups.get(entry, ()))
        if len(found) > max_sources:
            raise SystemExit(f"taxon entry {entry!r} ({';'.join(prefix)}) resolves to "
                             f"{len(found)} V4 groups, over --max-taxon-sources "
                             f"{max_sources}; use a lower rank")
        if not found:
            log.warning("taxon entry %r (%s) owns no database V4 group (none amplified, or "
                        "all belong to genome entries or more specific taxa); dropped",
                        entry, ";".join(prefix))
            continue
        members[entry] = found
    return members


def _home_labels(home_mseq: Path, src: dict[str, int], label_of_hit: dict[str, int]):
    """Each source's MAPseq label (-1 when unhit) and identity, from the sources' own mapping."""
    home = np.full(len(src), -1, dtype=np.int64)
    identity = np.full(len(src), np.nan)
    for query, hit, hit_identity in _mseq_rows(home_mseq):
        if query in src and hit in label_of_hit:
            home[src[query]] = label_of_hit[hit]
            identity[src[query]] = float(hit_identity)
    return home, identity


def _read_taxa(path: Path) -> list[tuple[str, str]]:
    """``id<TAB>taxon`` rows; blank lines, ``#`` comments and an ``id``/``taxon`` header skipped."""
    rows = [line.rstrip("\n").split("\t") for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    if rows and rows[0][:2] == ["id", "taxon"]:
        rows = rows[1:]
    if any(len(r) != 2 or not r[0].strip() or not r[1].strip() for r in rows):
        raise SystemExit(f"{path}: every row must be id<TAB>taxon")
    return [(i.strip(), t.strip()) for i, t in rows]


def whole_database_rows(amplicon_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return the panel translation and source sequences for an extracted database.

    The extracted directory is authoritative: its FASTA provides the byte-identical source
    sequences and its translation table provides the genome-copy weights. Multiple copies of
    the same V4 group for one genome are collapsed by summing their existing weights.
    """
    db_amplicons = amplicon_dir / "amplicons.fasta"
    db_translation = amplicon_dir / "translation_table.tsv"
    missing = [str(p) for p in (db_amplicons, db_translation) if not p.is_file()]
    if missing:
        raise SystemExit("--whole-database needs an EXTRACT_AMPLICONS directory containing "
                         "amplicons.fasta and translation_table.tsv; missing: "
                         + ", ".join(missing))

    headers, label_ids, label_of_ref, label_seqs = db_groups(db_amplicons)
    label_by_header = dict(zip(headers, (label_ids[i] for i in label_of_ref)))
    translation = pd.read_csv(db_translation, sep="\t")
    expected = {"genome_id", "refseq", "weight"}
    if not expected.issubset(translation.columns):
        raise SystemExit(f"{db_translation}: needs columns "
                         "genome_id, refseq and weight")
    unknown = sorted(set(translation.refseq) - set(label_by_header))
    if unknown:
        raise SystemExit(f"{db_translation}: refseq entries absent from {db_amplicons}: "
                         + ", ".join(unknown[:20]))

    panel = translation.loc[:, ["genome_id", "refseq", "weight"]].copy()
    panel["source"] = panel.refseq.map(label_by_header)
    panel = (panel.loc[:, ["genome_id", "source", "weight"]]
             .groupby(["genome_id", "source"], as_index=False, sort=True)
             .weight.sum())
    return panel, dict(zip(label_ids, label_seqs))


def prepare(a) -> None:
    whole_database = getattr(a, "whole_database", None)
    if whole_database and (a.panel_amplicons or a.panel_taxa):
        raise SystemExit("--whole-database cannot be combined with --panel-amplicons or "
                         "--panel-taxa")
    if not (whole_database or a.panel_amplicons or a.panel_taxa):
        raise SystemExit("prepare needs --panel-amplicons, --panel-taxa, or --whole-database")
    if a.panel_taxa and not a.db_taxonomy:
        raise SystemExit("--panel-taxa needs --db-taxonomy")

    if whole_database:
        db_amplicons = whole_database / "amplicons.fasta"
        rows, sequence = whole_database_rows(whole_database)
        rows = rows.to_dict("records")
        copies, dropped, taxa = {}, [], []
    else:
        if not a.db_amplicons:
            raise SystemExit("--db-amplicons is required unless --whole-database is used")
        db_amplicons = a.db_amplicons
        copies, dropped = ({}, []) if not a.panel_amplicons else panel_copies(
            a.panel_amplicons, a.fwd_primer, a.rev_primer, a.max_mismatch,
            dict(x.split("=", 1) for x in a.alias))
        rows = [{"genome_id": g, "source": v4g(seq), "weight": n / len(seqs)}
                for g, seqs in sorted(copies.items()) for seq, n in Counter(seqs).items()]
        sequence = {v4g(s): s for seqs in copies.values() for s in seqs}
        taxa = _read_taxa(a.panel_taxa) if a.panel_taxa else []

    ids = list(copies) + [e for e, _ in taxa]
    bad = sorted({i for i in ids if ENTRY_SEP in i or i == "background"}
                 | {i for i, n in Counter(ids).items() if n > 1})
    if bad:
        raise SystemExit("panel entry ids must be unique, not 'background' and not contain "
                         f"{ENTRY_SEP!r}: {', '.join(bad)}")
    headers, label_ids, label_of_ref, label_seqs = db_groups(db_amplicons)
    if taxa:
        import infer_composition as ic
        # A group with an ambiguous base is no source: simulated reads would carry its Ns, and
        # MAPseq may not place it at all, leaving no home label. The organism's unambiguous
        # groups stay. SILVA NR99, the 17 genera of the 20HM panel: 587 of 9,126 groups (15 unhit).
        ambiguous = {g for g, s in zip(label_ids, label_seqs) if set(s) - set("ACGT")}
        members = taxon_members(taxa, headers, label_ids, label_of_ref,
                                ic._read_taxonomy(a.db_taxonomy), set(sequence) | ambiguous,
                                a.max_taxon_sources)
        group_seq = dict(zip(label_ids, label_seqs))
        for entry, groups in members.items():
            rows += [{"genome_id": f"{entry}{ENTRY_SEP}{g}", "source": g, "weight": 1.0}
                     for g in groups]
            sequence.update((g, group_seq[g]) for g in groups)
            log.info("taxon entry %s: %d V4 group(s)", entry, len(groups))
    if not rows:
        raise SystemExit("no panel entry has a source")
    translation = pd.DataFrame(rows)
    sources = (translation.groupby("source").genome_id.agg(";".join).rename("genomes")
               .reset_index())
    sources["in_db"] = True if whole_database else sources.source.isin(set(label_ids))
    if whole_database and len(sources) > 10_000:
        log.warning("whole-database panel has %d distinct sources; this may be expensive",
                    len(sources))
    a.out.mkdir(parents=True, exist_ok=True)
    translation.to_csv(a.out / "panel_translation.tsv", sep="\t", index=False)
    if dropped:
        (a.out / "panel_unamplifiable.txt").write_text("".join(f"{g}\n" for g in dropped))
    sources.to_csv(a.out / "sources.tsv", sep="\t", index=False)
    with open(a.out / "sources.fasta", "w") as fh:
        fh.writelines(f">{s}\n{sequence[s]}\n" for s in sources.source)
    n_genomes = translation.genome_id.nunique() if whole_database else len(copies)
    log.info("%d genomes (%d unamplifiable, dropped), %d distinct sources, %d in the "
             "database, %d shared", n_genomes, len(dropped), len(sources),
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


def _align_decay(a, ka) -> float:
    """Return the requested fixed or measured alignment distance-decay value."""
    requested = str(a.distance_decay)
    if requested == "auto":
        model_pt = getattr(a, "model_pt", None)
        decay = ka.measure_error_rate(
            "trained" if model_pt else "flat",
            model_pt,
            getattr(a, "flat_sub_rate", 0.005),
            getattr(a, "flat_ins_rate", 0.0005),
            getattr(a, "flat_del_rate", 0.0005),
        )
        if decay <= 0.0:
            raise SystemExit("--distance-decay auto measured a zero error rate; set it explicitly")
        log.info("--distance-decay auto -> %.5f", decay)
    else:
        try:
            decay = float(requested)
        except ValueError as exc:
            raise SystemExit("--distance-decay must be a number or 'auto'") from exc
    if not 0.0 < decay <= 1.0:
        raise SystemExit("--distance-decay must be in (0, 1]")
    return decay


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
    import kernel_align as ka

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
    if a.tau < 0:
        raise SystemExit("--tau must be non-negative")
    a.distance_decay = _align_decay(a, ka)
    max_ambiguous_bases = getattr(a, "max_ambiguous_bases", ka.DEFAULT_MAX_AMBIGUOUS_BASES)
    max_postings = getattr(a, "max_postings", ka.DEFAULT_MAX_POSTINGS)
    if max_ambiguous_bases < 0 or max_postings < 1:
        raise SystemExit("--max-ambiguous-bases must be non-negative and --max-postings >= 1")
    ambiguity = ka.ambiguity_weights(label_seqs, a.ambiguity_weight)

    # Labels already contain one literal sequence per database group. Add only external
    # panel sequences, then use those sequence indexes as pigeonhole probes. At tau zero
    # the literal sequence lookup is the exact hash join; at tau >= 1 the candidate
    # filter avoids comparing every panel source with every database label.
    sequence_index = {seq: index for index, seq in enumerate(label_seqs)}
    unique_sequences = list(label_seqs)
    source_rows: dict[int, list[int]] = defaultdict(list)
    for row, source in enumerate(source_ids):
        source_sequence = sequence[source]
        index = sequence_index.get(source_sequence)
        if index is None:
            index = len(unique_sequences)
            sequence_index[source_sequence] = index
            unique_sequences.append(source_sequence)
        source_rows[index].append(row)

    entries_by_source = [{} for _ in source_ids]

    def add_candidate(source_index: int, label: int, distance: int) -> None:
        weight = (1.0 if ambiguity is None else ambiguity[label]) * a.distance_decay ** distance
        for row in source_rows.get(source_index, ()):
            if label != own.get(source_ids[row]):
                entries_by_source[row][label] = (weight, distance)

    for source_index in source_rows:
        label = sequence_index.get(unique_sequences[source_index])
        if label is not None and label < len(label_ids):
            add_candidate(source_index, label, 0)

    if a.tau >= 1:
        probes = np.fromiter(source_rows, dtype=np.int64)
        pairs = ka.pigeonhole_candidates(
            unique_sequences,
            a.tau,
            max_ambiguous_bases,
            max_postings,
            probes=probes,
        )
        log.info("verifying %d panel-to-database alignment candidate pair(s)", len(pairs))
        for left, right in pairs:
            distance = ka.bounded_iupac_distance(
                unique_sequences[left], unique_sequences[right], a.tau)
            if distance > a.tau:
                continue
            if left < len(label_ids):
                add_candidate(right, int(left), distance)
            if right < len(label_ids):
                add_candidate(left, int(right), distance)

    rows, cols, weights, dists = [], [], [], []
    for s, entries in enumerate(entries_by_source):
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
                  "max_ambiguous_bases": max_ambiguous_bases, "max_postings": max_postings,
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
    p.add_argument("--panel-amplicons", type=Path,
                   help="directory of <genome>.amplicons.fasta, or one FASTA with genome|... headers")
    p.add_argument("--panel-taxa", type=Path, help="TSV id<TAB>taxon (lineage prefix or bare name)")
    p.add_argument("--whole-database", type=Path,
                   help="EXTRACT_AMPLICONS directory; reuse all groups without another PCR")
    p.add_argument("--db-taxonomy", type=Path,
                   help="MAPseq .tax (header<TAB>lineage) for --db-amplicons; needed by --panel-taxa")
    # ponytail: hard cap; sub-sample or cluster groups if broad taxa turn out to be needed.
    p.add_argument("--max-taxon-sources", type=int, default=200,
                   help="fail when a taxon entry resolves to more V4 groups than this")
    p.add_argument("--db-amplicons", type=Path,
                   help="database amplicons.fasta; required unless --whole-database is used")
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
    al.add_argument("--distance-decay", default="0.007",
                    help="weight a label d edits away by c**d, or 'auto' to measure c")
    al.add_argument("--model-pt", type=Path,
                    help="pre-trained skiver model for '--distance-decay auto'")
    al.add_argument("--flat-sub-rate", type=float, default=0.005,
                    help="flat-model substitution rate for '--distance-decay auto'")
    al.add_argument("--flat-ins-rate", type=float, default=0.0005,
                    help="flat-model insertion rate for '--distance-decay auto'")
    al.add_argument("--flat-del-rate", type=float, default=0.0005,
                    help="flat-model deletion rate for '--distance-decay auto'")
    al.add_argument("--ambiguity-weight", type=float, default=0.3)
    al.add_argument("--max-ambiguous-bases", type=int,
                    default=4,
                    help="IUPAC positions tolerated across a pigeonhole candidate pair")
    al.add_argument("--max-postings", type=int, default=4096,
                    help="skip pigeonhole blocks shared by more than this many sequences")
    al.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    {"prepare": prepare, "build": build, "align": align}[a.cmd](a)


if __name__ == "__main__":
    main()
