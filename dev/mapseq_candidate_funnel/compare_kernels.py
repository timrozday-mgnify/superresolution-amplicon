"""Row-by-row distance between a simulate-mode kernel and an align-mode kernel."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "bin")
import sparse_matrix as sm

def rows(path):
    k = sm.read_kernel(Path(path))
    return k

a, b = rows(sys.argv[1]), rows(sys.argv[2])
assert list(a.source_ids) == list(b.source_ids) and list(a.label_ids) == list(b.label_ids)
A, B = a.kernel.toarray(), b.kernel.toarray()
tv = 0.5 * np.abs(A - B).sum(1)
outside = np.array([A[i, B[i] == 0].sum() for i in range(len(A))])
home_sim = np.array([A[i, a.home[i]] for i in range(len(A))])
home_aln = np.array([B[i, b.home[i]] for i in range(len(A))])
print(f"TV median={np.median(tv):.3f} max={tv.max():.3f} | sim mass outside align support "
      f"median={np.median(outside):.3f} max={outside.max():.3f} | home mass sim={np.median(home_sim):.3f} "
      f"align={np.median(home_aln):.3f} | homes differ={int((a.home != b.home).sum())}")
if len(sys.argv) > 3:
    for i in np.argsort(-tv)[:5]:
        top = np.argsort(-A[i])[:4]
        print(" ", a.source_ids[i], f"tv={tv[i]:.3f}", "sim:", [(a.label_ids[j][-6:], round(A[i, j], 3), round(B[i, j], 3)) for j in top])
