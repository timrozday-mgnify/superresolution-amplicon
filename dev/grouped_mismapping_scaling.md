# Mis-mapping matrix generation at database scale

**Question.** The alignment kernel ran out of memory somewhere past 16,000 references.
What does it cost over all of GTDB?

**Answer.** The whole GTDB SSU r232 V4 set — 1,001,241 amplifiable references out of
1,133,006 DB entries — builds in **2.3 s / 0.75 GB** at `tau = 0` and **79 s / 0.8 GB** at
`tau = 1`, writing an 11 MB matrix. Data: [`grouped_mismapping_scaling.csv`](grouped_mismapping_scaling.csv).

| references | `exact-hash` τ=0 | `kmer` τ=1 | `kmer` τ=2 | matrix |
|---:|---:|---:|---:|---:|
| 16,000 | 0.34 s | 0.70 s | 1.5 s | 0.2 MB |
| 64,000 | 0.42 s | 3.3 s | 11 s | 0.7 MB |
| 250,000 | 0.73 s | 15 s | 48 s | 2.5 MB |
| 1,001,241 | 2.3 s | 79 s | 200 s | 11 MB |

Peak RSS stays under 1.2 GB across every point (the CSV's `peak_child_rss_gb` is a
cumulative high-water mark over the sweep, so each row is an upper bound on its own run).
Amplicon extraction is excluded — it is a separate, unchanged stage, ~9 min single-threaded
over the 1.5 GB source FASTA.

## What was actually wrong

Two independent quadratics, both in the *number of references*:

1. **The output.** `M` was stored as a reference-square CSR, so an exact-duplicate group
   of size `k` cost `k²` nonzeros. GTDB SSU V4 collapses 1,001,241 references onto 86,557
   distinct amplicons, the largest group holding 99,590 of them, and `Σ k²` is
   **24.2 billion** nonzeros — about 290 GB. This, not the search, is what killed the run.
2. **The candidate search.** The k-mer index mapped every expanded 31-mer to a posting
   list, and each reference counted shared k-mers against every reference in every list it
   touched. Conserved 16S windows put thousands of references in one list, so the counting
   step was quadratic before any pair was verified.

## What replaced them

**Grouped storage.** A tie cluster is constant on exact-duplicate groups — two references
with the same amplicon are interchangeable — so `M[a, j] = S[group[a], group[j]]` for an
`S` over the *distinct* amplicons. 24.2 billion nonzeros become 264 thousand.
Row-stochasticity becomes `Σ_b S[a,b]·size[b] = 1`, and `r_true @ M` becomes a segment sum
over duplicate groups, one sparse product, and a scatter back (13 ms at full scale), so
inference never expands it either.

**Pigeonhole block filter.** Cut each amplicon into `P` disjoint blocks. A pair separated
by ≤ τ edits and `a` IUPAC positions damages at most `τ + a` blocks, so with `P > τ + a` at
least one block survives as a literally identical substring of the other sequence,
displaced by at most τ. Indexing blocks and probing the `2τ+1` displacements is therefore
an *exact* filter — no false negatives — and it never materialises a pair it does not
propose. Blocks are cut on each length's own geometry and each probe tries the `2τ+1`
lengths it could pair with, since a pair within τ edits differs in length by at most τ.

**Two passes, because ambiguity is rare and expensive.** Budgeting for IUPAC codes
globally needs `P = τ + a + 1` blocks; at `a = 4` that is 42 bp blocks on a 253 bp
amplicon, and a conserved 42-mer matches thousands of amplicons. Only 949 of the 86,557
distinct GTDB amplicons (1.1%) carry any ambiguity code at all. So: pass one uses
`P = τ + 1` long blocks over the whole set and settles every ambiguity-free pair; pass two
uses the short blocks but probes only from the ambiguous sequences, which suffices because
a pair pass one can miss has at least one ambiguous member. At τ=1 this cut postings from
88.0 M to 4.95 M and candidate pairs from 30.6 M to 2.53 M.

**Edlib for the ambiguous distance too.** The IUPAC-aware verification was a banded Python
DP whenever either sequence carried an ambiguity code. Edlib's `additionalEqualities` does
the same thing in C — verified identical over 3,000 random ambiguous pairs at k ∈ {1,2,3}
— and made verification of 2.5 M pairs take 70 s instead of minutes.

**Streaming and hashing.** References are grouped by a 128-bit digest in one pass, so
nothing holds the sequences; headers are stored as one blob rather than a fixed-width
numpy unicode array (~100× smaller for a million ids); the posting expansion is chunked;
and `read_matrix`'s O(n²) `list.index()` reorder became a dict lookup.

## What this does not fix

- **The cap is the one unsound part.** `--max_postings` (default 4096) skips a block
  shared by more than that many distinct amplicons, which can lose a pair whose *only*
  shared block is a conserved one. It is not reached on GTDB SSU at τ ∈ {1, 2} — no
  warning is emitted — but it is a real hole on a set with less variable regions.
- **`--max_ambiguous_bases` is a recall knob, not a guarantee.** The budget is counted
  across the pair *combined*; GTDB has amplicons with up to 71 ambiguity codes, and pairs
  exceeding the budget can be missed. Raising it shortens the blocks and slows the search.
- **`exact-hash` is byte identity, not IUPAC compatibility.** An `N` does not join the
  group of the sequences it is compatible with. That differs from `--backend minimap2
  --align_tau 0`, which is IUPAC-aware, on any set carrying ambiguity codes.
- **None of this changes what the kernel claims.** Every limitation in
  [the plan](../docs/alignment_mismapping_plan.md#limitations-and-biases) — the missing
  low-identity tail, `M_align` being over-confident, whole-reference queries only — is
  untouched. This is the same matrix, computed at a scale where it could not be computed
  before. The τ>0 kernel remains *unvalidated* against a measured `M`; it is now merely
  affordable.
