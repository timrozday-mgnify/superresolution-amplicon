"""Does a wider MAPseq funnel / the full top-tie set make the 26 panel V4 sources more recoverable?

Per run: labels are V4 groups (sha of the hit's sequence). Half of each source's simulated
reads measure M[source, label]; random known mixtures drawn from the other half are
deconvolved by EM through M. Reported: self-label rate and median TV(estimate, truth).
With -print_hits runs, the "ties" label is the frozenset of groups tied at the top score
(an equivalence class), which keeps what random tie-breaking throws away.
"""
import hashlib
import sys
from collections import Counter, defaultdict

import numpy as np

grp = {}
name = None
for line in open("../amp/amplicons.fasta"):
    line = line.strip()
    if line.startswith(">"):
        name = line[1:].split()[0]
    else:
        grp[name] = line  # amplicons are one line each
grp = {n: "v4g_" + hashlib.sha256(s.encode()).hexdigest()[:16] for n, s in grp.items()}


def parse(path, ties):
    hits = defaultdict(list)
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[1].isdigit():
            hits[f[0]].append((float(f[4]), grp[f[2]]))
        else:
            hits[f[0]].append((float(f[2]), grp[f[1]]))
    out = {}
    for q, hs in hits.items():
        if ties:
            best = max(s for s, _ in hs)
            out[q] = frozenset(g for s, g in hs if s == best)
        else:
            out[q] = hs[0][1]
    return out


def em(counts, M, iters=500):
    labels = [l for l in counts if l in M.columns]
    n = np.array([counts[l] for l in labels], float)
    A = np.array([[M.rows[s].get(l, 0.0) for l in labels] for s in M.sources]) + 1e-9
    p = np.full(len(M.sources), 1 / len(M.sources))
    for _ in range(iters):
        mix = p @ A
        p = p * (A @ (n / mix))
        p /= p.sum()
    return p


class Kernel:
    def __init__(self, labelled, sources):
        self.sources = sources
        tally = {s: Counter() for s in sources}
        for q, l in labelled:
            tally[q.rsplit(":", 1)[0]][l] += 1
        self.rows = {s: {l: c / sum(t.values()) for l, c in t.items()} for s, t in tally.items()}
        self.columns = set().union(*self.rows.values())


def evaluate(run, ties=False, trials=40, n_reads=5000):
    lab = parse(f"{run}.mseq", ties)
    by_src = defaultdict(list)
    for q, l in lab.items():
        by_src[q.rsplit(":", 1)[0]].append((q, l))
    sources = sorted(by_src)
    train, test = [], {}
    for s in sources:
        rs = sorted(by_src[s], key=lambda x: int(x[0].rsplit(":", 1)[1]))
        train += rs[::2]
        test[s] = [l for _, l in rs[1::2]]
    M = Kernel(train, sources)
    own = np.mean([M.rows[s].get(s if not ties else None, 0) if not ties else
                   sum(v for l, v in M.rows[s].items() if l == frozenset([s])) for s in sources])
    rng = np.random.default_rng(0)
    tvs = []
    for _ in range(trials):
        truth = rng.dirichlet(np.full(len(sources), 0.5))
        k = rng.multinomial(n_reads, truth)
        counts = Counter()
        for s, c in zip(sources, k):
            for i in rng.integers(0, len(test[s]), c):
                counts[test[s][i]] += 1
        tvs.append(0.5 * np.abs(em(counts, M) - k / k.sum()).sum())
    print(f"{run:18s} {'ties' if ties else 'top ':4s} sources={len(sources)} labels={len(M.columns):5d} "
          f"self={own:.3f} TV median={np.median(tvs):.4f} p90={np.quantile(tvs, .9):.4f}")


for arg in sys.argv[1:]:
    run, _, mode = arg.partition(":")
    evaluate(run, ties=mode == "ties")
