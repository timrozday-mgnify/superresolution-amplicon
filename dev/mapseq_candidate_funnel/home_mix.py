"""Distance-0 entry as (1-e)^L on the error-free home + the rest on 1-substitution copies' labels, vs the simulated kernel."""
import sys, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "../../../bin"); import sparse_matrix as sm
g, n = {}, None
for l in open("../amp/amplicons.fasta"):
    l = l.strip()
    if l.startswith(">"): n = l[1:]
    else: g[n] = "v4g_" + hashlib.sha256(l.encode()).hexdigest()[:16]
src, n = {}, None
for l in open("../../panel_kernel/sources.fasta"):
    l = l.strip()
    if l.startswith(">"): n = l[1:]
    else: src[n] = l
k = sm.read_kernel(Path("k_sim_aap/panel_kernel.npz")); A = k.kernel.toarray(); col = {l: i for i, l in enumerate(k.label_ids)}
home = sm.read_kernel(Path("k_align0_aap/panel_kernel.npz")).home
h = defaultdict(Counter)
for l in open("homerep_sub1.mseq"):
    if l.startswith("#"): continue
    x = l.split("\t"); h[x[0].rsplit(":", 1)[0]][g[x[1]]] += 1
q = lambda x: f"median={np.median(x):.3f} p90={np.quantile(x, .9):.3f} max={max(x):.3f}"
for e in [0.0, 0.0031, 0.005]:
    tv = []
    for i, s in enumerate(k.source_ids):
        w0 = (1 - e) ** len(src[s])
        D = np.zeros(A.shape[1]); D[home[i]] += w0
        for lab, c in h[s].items(): D[col[lab]] += (1 - w0) * c / sum(h[s].values())
        tv.append(.5 * np.abs(A[i] - D).sum())
    print(f"e={e:<7} w0~{(1-e)**252:.2f}  {q(tv)}")
