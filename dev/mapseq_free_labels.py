"""Label observed reads without MAPseq: nearest distinct SILVA V4 amplicon by edit distance.

Question (benchmark repo, SILVA sweep): MAPseq needs a `.mscluster` for the whole reference
set, which costs tens of minutes per set. The align-mode kernels already find neighbours by
exact hash / pigeonhole + edlib. Could the observed reads be labelled the same way, and what
does it do to the composition?

    python dev/mapseq_free_labels.py --silva work/silva --samples S01 S02 ... > dev/mapseq_free_labels.csv

Per sample, a read is labelled with the distinct amplicon(s) at the smallest IUPAC edit
distance up to --tau; a tie goes to one of them at random (seeded), the way MAPseq returns
one hit. Unreached reads get no hit. Both label sets are then fitted with the benchmark's
`silva` arm point (exact-hash tau 0 kernel, v4_group space, no presence gate) and scored per
genus against the truth counted from the read names, with score_sweep.py's rules.
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))
import build_mismapping_align as bma  # noqa: E402

GENUS_RANK = 6
UNRESOLVED = "unresolved"
# Read-name genomes the benchmark example's panel does not list (its panel is the 20HM
# community minus these): their SILVA 138.2 genus lineages.
EXTRA_LINEAGE = {
    "bacteroides_uniformis_strain2": "Bacteria;Bacteroidota;Bacteroidia;Bacteroidales;Bacteroidaceae;Bacteroides",
    "salinibacter_ruber": "Bacteria;Rhodothermota;Rhodothermia;Rhodothermales;Rhodothermaceae;Salinibacter",
    "veillonella_parvula": "Bacteria;Bacillota;Negativicutes;Veillonellales-Selenomonadales;Veillonellaceae;Veillonella",
}


def genus(lineage: str) -> str | None:
    ranks = lineage.split(";")[:GENUS_RANK]
    if len(ranks) < GENUS_RANK or ranks[-1] in ("", "unclassified"):
        return None
    return ";".join(ranks)


def nearest_labels(reads: list[str], seqs: list[str], tau: int, max_postings: int):
    """{read index: (distance, [distinct amplicon indices])} for reads within ``tau``."""
    index = {s: i for i, s in enumerate(seqs)}
    out = {r: (0, [index[s]]) for r, s in enumerate(reads) if s in index}
    # Widen in steps: most reads are 1-2 edits out, and the pigeonhole pass at a small tau
    # has longer blocks, so far fewer postings to verify.
    for step in sorted({t for t in (1, 2, tau) if 1 <= t <= tau}):
        rest = [r for r in range(len(reads)) if r not in out]
        if rest:
            out.update(_within(reads, rest, seqs, step, max_postings))
    return out


def _within(reads, rest, seqs, tau, max_postings):
    pool = seqs + [reads[r] for r in rest]
    probes = np.arange(len(seqs), len(pool))
    packed = bma._block_pass(pool, probes, tau, tau + 1, max_postings)
    i, j = divmod(packed, len(pool))
    # Keep read-amplicon pairs only (i < j, so the amplicon is i).
    keep = (i < len(seqs)) & (j >= len(seqs))
    best: dict[int, tuple[int, list[int]]] = {}
    for a, q in zip(i[keep], j[keep]):
        d = bma.bounded_iupac_distance(pool[q], seqs[a], tau)
        if d > tau:
            continue
        r = rest[q - len(seqs)]
        cur = best.get(r)
        if cur is None or d < cur[0]:
            best[r] = (d, [int(a)])
        elif d == cur[0]:
            cur[1].append(int(a))
    return best


def label_sample(fasta: Path, seqs, rep_of, tau, max_postings, seed, out: Path) -> dict:
    ids, read_seqs = [], []
    for rid, s in bma.iter_fasta(fasta):
        ids.append(rid)
        read_seqs.append(s.upper())
    distinct = sorted(set(read_seqs))
    pos = {s: k for k, s in enumerate(distinct)}
    t0 = time.perf_counter()
    lab = nearest_labels(distinct, seqs, tau, max_postings)
    seconds = time.perf_counter() - t0
    rng = random.Random(seed)
    dist = Counter()
    with open(out, "w") as fh:
        fh.write("#query\tdbhit\tbitscore\tidentity\n")
        for rid, s in zip(ids, read_seqs):
            hit = lab.get(pos[s])
            if hit is None:
                dist["none"] += 1
                fh.write(f"{rid}\t\t\t\n")
                continue
            d, cands = hit
            dist[d] += 1
            a = cands[0] if len(cands) == 1 else rng.choice(cands)
            fh.write(f"{rid}\t{rep_of[a]}\t0\t{1 - d / max(len(s), 1):.4f}\n")
    return {"reads": len(ids), "distinct_reads": len(distinct), "label_seconds": seconds,
            **{f"d{k}": v / len(ids) for k, v in sorted(dist.items(), key=str)}}


def mseq_hits(path: Path) -> dict[str, str]:
    hits = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) > 1 and f[1]:
                hits[f[0]] = f[1]
    return hits


def agreement(a: Path, b: Path, group_of: dict[str, int]) -> float:
    """Share of reads both label sets hit that land in the same distinct amplicon."""
    ha, hb = mseq_hits(a), mseq_hits(b)
    both = [q for q in ha if q in hb]
    return sum(group_of[ha[q]] == group_of[hb[q]] for q in both) / max(len(both), 1)


def truth_genera(fasta: Path, lineage: dict[str, str]) -> dict[str, float]:
    counts = Counter(rid.split(":")[1] for rid, _ in bma.iter_fasta(fasta))
    out = defaultdict(float)
    for g, n in counts.items():
        out[genus(lineage[g])] += n
    return dict(out)


def predicted_genera(v4_groups: Path) -> dict[str, float]:
    out = defaultdict(float)
    with open(v4_groups, newline="") as fh:
        for row in csv.DictReader(fh):
            out[genus(row["lca"]) or UNRESOLVED] += float(row["inferred_mean"])
    return dict(out)


def tv(pred: dict, truth: dict) -> tuple[float, float, float]:
    """Genus TV (unresolved counts as error), unresolved share, and TV over resolved mass."""
    def total_variation(p_, t_):
        keys = set(p_) | set(t_)
        p, t = sum(p_.values()) or 1.0, sum(t_.values()) or 1.0
        return 0.5 * sum(abs(p_.get(k, 0) / p - t_.get(k, 0) / t) for k in keys)
    resolved = {k: v for k, v in pred.items() if k != UNRESOLVED}
    return (total_variation(pred, truth), pred.get(UNRESOLVED, 0) / (sum(pred.values()) or 1.0),
            total_variation(resolved, truth))


def infer(obs: Path, a, out: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run([sys.executable, BIN / "infer_composition.py",
                    "--amplicon-dir", a.silva / "amp", "--mismapping-matrix", a.matrix,
                    "--obs-mseq", obs, "--taxonomy", a.silva / "db" / "silva_nr99.tax",
                    "--infer-space", "v4_group", "--no-presence", "--mode", "vi",
                    "--alpha", "0.5", "--steps", "3000", "--lr", "0.02",
                    "--seed", "42", "--sample-id", out.name, "-o", out],
                   check=True, capture_output=True)
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--silva", type=Path, default=Path("work/silva"))
    p.add_argument("--reads", type=Path, default=Path("work/panel_obs"))
    p.add_argument("--matrix", type=Path, default=Path("work/silva/mapseq_free/exact.npz"))
    p.add_argument("--panel-config", type=Path, required=True,
                   help="benchmark examples/sr_amplicon_silva_sweep/config.yaml (panel lineages)")
    p.add_argument("--samples", nargs="+", required=True)
    p.add_argument("--tau", type=int, default=4)
    p.add_argument("--max-postings", type=int, default=4096)
    a = p.parse_args()

    lineage = {m["id"]: m["taxonomy"] for m in yaml.safe_load(a.panel_config.read_text())["panel"]}
    lineage.update(EXTRA_LINEAGE)
    refseqs, group, seqs = bma.dedup(a.silva / "amp" / "amplicons.fasta")
    rep_of: dict[int, str] = {}
    for r, g in zip(refseqs, group):
        rep_of.setdefault(int(g), r)
    group_of = dict(zip(refseqs, map(int, group)))
    work = a.silva / "mapseq_free"

    w = csv.writer(sys.stdout)
    w.writerow(["sample", "labels", "genus_tv", "unresolved", "resolved_tv", "infer_seconds",
                "label_seconds", "exact_share", "unlabelled_share", "agree_with_mapseq"])
    for s in a.samples:
        fasta = a.reads / f"{s}.fasta"
        mapseq = a.silva / "obs" / f"{s}.obs.mseq"
        nearest = work / f"{s}.nearest.mseq"
        stats = label_sample(fasta, seqs, rep_of, a.tau, a.max_postings, 42, nearest)
        print(s, stats, file=sys.stderr)
        truth = truth_genera(fasta, lineage)
        agree = agreement(mapseq, nearest, group_of)
        for name, obs in (("mapseq", mapseq), ("nearest", nearest)):
            out = work / "fits" / name / s
            secs = infer(obs, a, out)
            score, unres, resolved = tv(predicted_genera(out / "inferred_v4_groups.csv"), truth)
            w.writerow([s, name, f"{score:.4f}", f"{unres:.4f}", f"{resolved:.4f}", f"{secs:.0f}",
                        f"{stats['label_seconds']:.1f}" if name == "nearest" else "",
                        f"{stats.get('d0', 0):.4f}", f"{stats.get('dnone', 0):.4f}"
                        if name == "nearest" else "", f"{agree:.4f}"])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
