# Hits on database entries with no amplicon (27F-1492R against SILVA)

Long-read benchmark of AAP + sramp on simulated 20HM mocks (PacBio Sequel and Revio, ONT
R10.4.1; amplicon-analysis-pipeline `longread-test/hpc/run2`, sramp at `35af082`). Against
the panel database, `tv_genome_read` was 0.022–0.027. Against SILVA 138.1 it was
0.126–0.129, 15–16% of reads were background, Collinsella and Salinibacter had observed
abundance 0, and B. uniformis strains 2 and 3 were missed.

## Cause

Only 363,330 of SILVA's 2,098,433 entries (17%) amplify under 27F/1492R: 1492R sits at
the 3' end, where most entries are truncated. MAPseq still maps reads against every
entry, AAP's reads and sramp's simulated ones alike (`MAPSEQ_PANEL_SIM` uses the full
FASTA). The hypothesis was that the simulated reads were mapped against the amplicons
only. That part was wrong: they were mapped against the full database. The kernel then
lost the hits:

- `tally_kernel` dropped every simulated hit on an entry with no V4 group, and each row
  was renormalised over the remaining hits. So the kernel predicted none of those hits.
- `_home_labels` gave such a source its own group, even when MAPseq put its error-free
  read on the entry with no amplicon.
- The sample's hits on those entries became background (`db_refs` not staged, as in
  `run2`) or were dropped as foreign (`db_refs` present: it lists amplifiable accessions
  only).

Real reads per source genome in `syn_sequel_even` (headers name the genome; 1,650 of
9,998 reads hit an entry with no amplicon):

| genome | reads | share on entries with no amplicon | top entry |
|---|---|---|---|
| B. uniformis strain 2 / 3 | 341 / 334 | 0.79 / 0.81 | HQ776863.1.1451 (236 each) |
| Collinsella aerofaciens | 424 | 0.60 | GQ898710.1.1465 (220) |
| E. rectale | 366 | 0.42 | GQ896977.1.1492 |
| Salinibacter ruber | 79 | 0.41 | LN650024.1.1491 |
| Dorea formicigenerans | 530 | 0.36 | DQ532285.1.1494 |
| V. parvula | 340 | 0.29 | KU321270.1.1486 |

For 27F-1492R, 25 of the 20HM prep's 73 home queries hit such an entry, many exactly
(identity 1). For aap-nov2025 (V4), none of its 33 do.

## Fix

`build_panel_kernel._add_unamplified` gives each entry with no amplicon that a home or
simulated read hits a label of its own, `unamp_<accession>`, with that one accession.
`build` then measures the hits on those labels, and `_home_labels` homes a source on its
hit's label outright. The support-count tie rule can't apply here: one accession is one
reference. `align` takes the same home but never aligns to these labels, since they have
no sequence. `combine` and inference already match labels by id and accession, so they are
unchanged. `tests/test_unamplified_hits.py` covers `build`, `align` and inference.

## Result: run2, default flat rates, `--infer_presence false`

The kernel was rebuilt locally: SILVA amplicons re-extracted (sha `4547de34…`, as
`run2`), and 1,000 reads per source mapped with bioconda MAPseq 2.1.1a. The 2.1.1a build
reproduced AAP's 2.1.1b hits on 400/400 of `syn_sequel_even`'s reads given the same
cmsearch-trimmed queries.

| sample | DB | run2 | before (rebuilt) | after |
|---|---|---|---|---|
| syn_sequel_even | SILVA | 0.1264, bg 0.164 | 0.1276 | **0.0298**, bg 0.001 |
| syn_revio_even | SILVA | 0.1288, bg 0.151 | 0.1288 | **0.0288**, bg 0.001 |
| syn_r1041_even | SILVA | 0.1281, bg 0.154 | 0.1279 | **0.0294**, bg 0.003 |
| all three | panel | 0.0222 / 0.0240 / 0.0267 | same | same (identical) |

The figures are `tv_genome_read`, computed as in `bin/score_sramp.py`. After the fix:

- **Collinsella:** inferred 0.022 → 0.042–0.045 (truth 0.042); observed 0 → 0.019–0.022.
- **Salinibacter:** inferred 0.006 → 0.007–0.008 (truth 0.008); observed 0 → 0.002–0.004.
- **Missed genomes:** 2 → 0.

The "before" rebuild had `db_refs` beside the kernel, so its background is ~0: 1,645
hits were dropped as foreign instead.

## MiMiCC (V4) is unchanged

Nov2025 was run through the browser's chain (`genome_service.prepare` → `align` per
genome, `combine`, `sramp_run.py --backend numpy`), with the code before and after the
fix. All 24 aap-nov2025 parts were identical. The 50 samples' compositions differ by TV
0.0, and the largest per-genome change is 0.0.

The browser's mini-database MAPseq (M11: mini500 with `-dropkmers`) returns the same
accession as full SILVA for every 27F-1492R home hit on an entry with no amplicon. The 6
of 73 rows where they differ are ties between amplifiable entries at equal identity. So a
part built on the page gets the same `unamp_` labels as the HPC prep.
