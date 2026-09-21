# Labelling observed reads without MAPseq (SILVA 138.2 NR99 V4)

Question from the benchmark repo's SILVA sweep: MAPseq needs a `.mscluster` for every
reference set (11.4 min for the 464,997 SILVA NR99 V4 amplicons, amd64 emulated, 8 threads).
The align-mode kernels already find neighbours by exact hash and pigeonhole + edlib. Could
the observed reads be labelled the same way, which would drop the clustering?

Script: `dev/mapseq_free_labels.py`. Raw rows: `dev/mapseq_free_labels.csv`. Six of the
Phase V samples (`work/panel_obs/S{01,04,07,10,13,16}`, about 80k merged, primer-trimmed
reads each). Each read gets the distinct amplicon at the smallest IUPAC edit distance (≤ 4,
widened 1 → 2 → 4). A tie goes to one of the tied amplicons, picked at random. Both label
sets are fitted at the benchmark's `silva` arm point: exact-hash tau-0 kernel, `v4_group`
space, no presence gate, vi with 3,000 steps. Both are scored per genus the way
`score_sweep.py` scores them, against truth counted from the read names.

## Result

| labels | genus TV (median of 6) | unresolved | TV over resolved mass | labelling time / sample |
|---|---:|---:|---:|---:|
| MAPseq | **0.318** | 0.314 | 0.307 | 18 s (12 threads, needs the `.mscluster`) |
| nearest edit distance | 0.448 | 0.444 | 0.314 | 58 s (1 thread, no index) |

**Nearest-edit labelling is worse at genus, and it is not faster per sample.** All of the
gap is mass sent to V4 groups whose LCA stops above genus. On genus-resolved mass the two
are the same.

1. **About half the reads match a SILVA amplicon byte-for-byte**: 49.6% at distance 0, 37%
   at 1, 11% at 2, and 0.13% unreached at 4.
2. **MAPseq often puts an exact-match read somewhere else.** Of the reads that match a
   SILVA amplicon exactly, MAPseq labels 41% (S01) with a *different* V4 group. In 92% of
   those cases the read's own group is shared by several SILVA references, and MAPseq picks
   a single-reference group 1–2 edits away. The labels agree on 56–57% of all reads.
3. **That preference is what wins at genus.** Exact V4 sequences shared across genera
   (*Streptococcus*, some *Bacteroides*, *Enterocloster*, *Agathobacter*) have an LCA above
   genus. MAPseq's single-reference neighbour usually carries the right genus. S01:
   *Streptococcus* 0.124 in truth, 0.125 from MAPseq, 0.058 from nearest-edit labels. It is
   not V4 evidence: the reads cannot tell those genera apart. It is MAPseq's tie-breaking
   leaning towards well-annotated singletons.
4. **Neither method recovers the Lachnospiraceae genera.** *Dorea*, *Lachnoclostridium*,
   *Coprococcus*, *[Ruminococcus] gnavus group* and *Roseburia* are about 30% of truth,
   and both methods put that mass in unresolved groups.

## Consequence

Keep MAPseq for the observed reads. Pay for the `.mscluster` once per reference set
(`--amplicon_cache`), and skip it when nothing maps (align-mode kernel with every sample's
`mseq:` supplied). The tau-0 exact-hash kernel does not model MAPseq's habit of moving
exact matches to singletons (finding 2). The mismatch sits between the kernel and the labels
it explains, so a measured (`simulate`) kernel, or an alignment kernel that encodes that
preference, is where genus accuracy would move.
