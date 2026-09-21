"""Per MAPseq run over probe.fasta: does the top hit / any aligned candidate carry the read's exact sequence?"""
import sys
from collections import defaultdict

seq_of, name = {}, None
for line in open("../amp/amplicons.fasta"):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:].split()[0]
    else:
        seq_of[name] = seq_of.get(name, "") + line
reads, name = {}, None
for line in open("probe.fasta"):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:]
    else:
        reads[name] = line
moved = {r: m == "1" for r, m, *_ in (l.rstrip("\n").split("\t") for l in open("probe_meta.tsv"))}


def parse(path):
    top, cands, last = {}, defaultdict(list), None
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[1].isdigit():  # -print_hits: query, rank, hit, kmers, score, identity, ...
            cands[f[0]].append((f[2], float(f[4])))
            if f[1] == "0":
                top[f[0]] = f[2]
        else:
            top[f[0]] = f[1]
    return top, cands


runs = {n: parse(f"{n}.mseq") for n in sys.argv[1:]}
for n, (top, cands) in runs.items():
    for grp in (True, False):
        rs = [r for r in reads if moved[r] == grp]
        exact_top = sum(seq_of.get(top.get(r), "") == reads[r] for r in rs)
        exact_cand = sum(any(seq_of[c] == reads[r] for c, _ in cands[r]) for r in rs)
        # exact copy aligned but a non-exact hit won anyway
        outscored = sum(any(seq_of[c] == reads[r] for c, _ in cands[r])
                        and seq_of.get(top.get(r), "") != reads[r] for r in rs)
        ncand = sum(len(cands[r]) for r in rs) / len(rs)
        print(f"{n:12s} {'moved ' if grp else 'stayed'} n={len(rs)} "
              f"top_exact={exact_top / len(rs):.3f} exact_in_candidates={exact_cand / len(rs):.3f} "
              f"exact_aligned_but_lost={outscored} mean_candidates={ncand:.1f}")
if "default" in runs and "default_rep" in runs:
    a, b = runs["default"][0], runs["default_rep"][0]
    same = sum(a[r] == b.get(r) for r in a)
    same_seq = sum(seq_of[a[r]] == seq_of.get(b.get(r), "") for r in a)
    print(f"rerun: same ref {same / len(a):.3f}, same sequence {same_seq / len(a):.3f}")
