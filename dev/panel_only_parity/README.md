# Panel-only parity baselines

Phase P.0 snapshot, captured from `da8e61ebd48cbb9b84929a9ab69ce8901358ad5b`
(`aap-merge-effects`) before removal of the square-matrix path.

The committed files are the inference result, diagnostics, fit check, and (for the
two small square runs) the published matrix bundle.  They are the outputs Phase P.7
compares; Nextflow work directories and raw simulation reads remain ignored under
`work/panel_only_parity/`.

| baseline | old path | input and scope |
| --- | --- | --- |
| `fixture/simulate` | simulate | Bundled FASTQ fixture (`assets/samplesheet.example.yml`), flat model, 500 reads/reference. |
| `fixture/align_exact_hash` | align exact-hash | Same fixture, `tau=0`. |
| `fixture/align_kmer_tau1` | align kmer | Same fixture, `tau=1`, decay `0.005`. |
| `buniformis/simulate` | simulate | The 21-genome B. uniformis reference set. Extraction produced 81 amplicons. |
| `buniformis/align_exact_hash` | align exact-hash | Same set, `tau=0`. |
| `buniformis/align_kmer_tau1` | align kmer | Same set, `tau=1`, decay `0.005`. |
| `panel_20hm/sim_trained/S01`&ndash;`S20` | existing rectangular panel | Existing 20HM genome-panel `sim_trained` profiles, diagnostics, and fit checks. |
| `silva_species_s05/sim_calibrated_flat_nogate` | existing rectangular taxon panel | Re-scored S05 from `work/silva/obs/S05.obs.mseq` against `work/silva/species/cal.npz`; no SILVA mapping was rerun. |

The B. uniformis observation is a deterministic supplied MAPseq file made from the
first 50 full-length references, repeated 25 times, and mapped against the existing
`subspecies_Buniformis_amplicon_v4/mapseq_db` database.  It supplies 1,175 mapped
reads after extraction and keeps the parity check focused on the square kernel and
inference paths rather than on a fresh observed-read mapping.

Each fixture run maps 1,500 reads and its fit check is `ok`.  The simulate
inferred-composition SHA-256 is
`f8437384ce08664754c106773f9d22355fdbfc9c207ad6ac02483f9668986544`;
the exact-hash and kmer-tau-1 profiles are identical at
`557a0e58d2b1d12368ee8818c93ac9b0b64aa3885031a9d65f1b3ee5557114d6`.
The B. uniformis inferred-composition SHA-256 values are:

| mode | SHA-256 |
| --- | --- |
| simulate | `70d1ac57744225b92e0c80e8721bd54cb9b76b6634450a4f570a8d1468ab0c7f` |
| align exact-hash | `5a36ca1744853d130923853157a4ce4d0e530e29559b88b3a7629b60c10f3fb9` |
| align kmer tau 1 | `d5a33ae8fde5896b7a49124d06ed31ae808bba024137791a6a47e5486ba66420` |

All three B. uniformis fit checks are `ok`.  The SILVA species re-score is also
`ok` with 79,472 mapped reads; its inferred-composition SHA-256 is
`8356236263d1eb50532884d7f7a5e05c4c692710b29768da9fba9c4ffcf43d6b`.

P.7 compares the future whole-database panel output with the fixture and B. uniformis
square baselines at genome TV <= 0.02.  The 20HM and SILVA species panel snapshots must
match their corresponding profile, diagnostics, and fit result exactly, because their
numerical panel code is not meant to change.
