"""How amplicon-analysis-pipeline's fastp merge treats sequencing errors (AAP plan Phase 0).

``fastp`` (0.1): builds read pairs from one real merged amplicon, places errors in them, and
runs fastp 1.0.1 with AAP's READS_QC_MERGE arguments. For each pair it records whether the
pair merged and what the merged read carries at the error. Four case families:

- ``sub_quality``: one substitution in one mate at quality ``q_err``, the other mate
  correct at ``q_other``. Which base does the merged read keep?
- ``sub_count``: ``k`` substitutions at Q38 in R1 only, or split between the mates. Where
  does the overlap mismatch limit cut in?
- ``sub_window``: ``k`` substitutions in R1, all within fragment bases 1-48 or 61-108.
- ``indel``: one insertion or deletion in one mate, at each position along the fragment.
- ``pcr``: the same substitution or indel in both mates (a polymerase error).

    python dev/aap_merge_effects.py fastp \
        --merged <aap_outdir>/<run>/qc/<run>.merged.fastq.gz --out dev/aap_merge_effects_fastp.csv

``profile`` (0.2): residual error per amplicon position in AAP's merged reads. The reference is
the V4 amplicons (primers excluded) cut from the community's genomes, not the run's ASVs, which
DADA2 derived from the same reads. Each distinct read is aligned (edlib, infix) to every
amplicon; its nearest one is the template if it is unique and within ``--max-edits``.

    python dev/aap_merge_effects.py profile --genomes <dir of genome FASTAs> \
        --runs <aap_outdir> --out dev/aap_merge_effects_profile.tsv

``primers`` (0.3): base frequencies at each degenerate primer position, read from the start
(515F) and end (rc 806R) of each merged read, per run and per source genome. A read's source
is the 20HM genome whose amplicon equals the read's primer-free interior exactly; other reads
count only towards ``all``. Also writes the lengths before the forward primer (spacer) and
after the reverse primer (overhang).

    python dev/aap_merge_effects.py primers --genomes <dir of genome FASTAs> \
        --runs <aap_outdir> --out dev/aap_merge_effects_primer_mix.tsv

``clip`` (0.4): does cutting each read to its cmsearch hit, as AAP does before MAPseq, change
MAPseq's answer? Maps one run's merged reads with AAP's MAPseq build and arguments three
times: as merged, clipped to the ``.tblout.deoverlapped`` coordinates (AAP's ``esl-sfetch
-Cf``), and as merged again (a replicate, to separate clipping from run-to-run noise).

    python dev/aap_merge_effects.py clip --merged <run>.merged.fastq.gz \
        --deoverlapped <run>.tblout.deoverlapped --db SILVA-SSU.fasta --tax SILVA-SSU-tax.txt \
        --work work/aap_clip --out dev/aap_merge_effects_clip.tsv

``builds`` (0.5): the same clipped reads through SR's MAPseq build, with AAP's arguments and
with SR's default (none), compared with ``clip``'s AAP-build run through ``iter_mseq``.

    python dev/aap_merge_effects.py builds --db SILVA-SSU.fasta --tax SILVA-SSU-tax.txt \
        --work work/aap_clip --out dev/aap_merge_effects_builds.tsv

``coverage`` (0.6): every run's reads, clipped as in ``clip``, mapped in one MAPseq pass with
AAP's build and arguments. For each run it reports the share of reads whose top hit has no V4
amplicon, and why (primer site missing). In-silico PCR runs only on references some read hit.

    python dev/aap_merge_effects.py coverage --runs <aap_outdir> --db SILVA-SSU.fasta \
        --tax SILVA-SSU-tax.txt --work work/aap_coverage --out dev/aap_merge_effects_coverage.tsv

For ``fastp``, the template is the run's most common merged read. Mates are 310 cycles, as in the Nov2025
runs, and read past the fragment into the TruSeq adapters. Base qualities are Q38 except
where a case sets them. The Nov2025 reads use three quality bins: Q12, Q24 and Q38.
"""
import argparse
import gzip
import itertools
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import edlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import subspecies_infer as si  # noqa: E402

FASTP = ("docker run --rm --platform linux/amd64 -v {dir}:{dir} -w {dir} "
         "community.wave.seqera.io/library/fastp:1.0.1--c8b87fe62dcc103c fastp")
# Verbatim from the Nov2025 runs' fastp.json "command", minus file names and --thread.
AAP_ARGS = ("-m --detect_adapter_for_pe --cut_front_window_size 1 --cut_tail_window_size 1 "
            "--cut_front 3 --cut_tail 3 --cut_right --cut_right_window_size 4 "
            "--cut_right_mean_quality 15 -l 100")
ADAPTER_R1 = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
ADAPTER_R2 = "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGTA"
CYCLES = 310
HIGH = 38


def most_common_read(path: Path) -> str:
    with gzip.open(path, "rt") as fh:
        return Counter(line.strip() for i, line in enumerate(fh) if i % 4 == 1).most_common(1)[0][0]


def other_base(base: str) -> str:
    return "ACGT"[("ACGT".index(base) + 1) % 4]


def edit(seq: str, kind: str, pos: int) -> str:
    """Substitute, insert before, or delete the base at ``pos``."""
    if kind == "sub":
        return seq[:pos] + other_base(seq[pos]) + seq[pos + 1:]
    if kind == "ins":
        return seq[:pos] + other_base(seq[pos]) + seq[pos:]
    return seq[:pos] + seq[pos + 1:]


def mate(fragment: str, adapter: str, quals: dict[int, int]) -> tuple[str, str]:
    """``CYCLES`` bases read from ``fragment`` into ``adapter``; ``quals`` maps a read
    position to its quality."""
    seq = (fragment + adapter + "A" * CYCLES)[:CYCLES]
    return seq, "".join(chr(33 + quals.get(i, HIGH)) for i in range(CYCLES))


def pair(template: str, r1_edits=(), r2_edits=(), r1_quals=None, r2_quals=None):
    """Mates of ``template``; each edit list holds ``(kind, fragment position)``, applied
    right to left so positions stay in template coordinates. R2 edits are placed on the
    template and then reverse-complemented, so both mates describe the same site."""
    f1, f2 = template, template
    for kind, pos in sorted(r1_edits, key=lambda e: -e[1]):
        f1 = edit(f1, kind, pos)
    for kind, pos in sorted(r2_edits, key=lambda e: -e[1]):
        f2 = edit(f2, kind, pos)
    return (mate(f1, ADAPTER_R1, r1_quals or {}), mate(si.revcomp(f2), ADAPTER_R2, r2_quals or {}))


def cases(template: str):
    n = len(template)
    r2pos = lambda p: n - 1 - p  # noqa: E731  fragment position -> R2 read position
    for pos, erring, q_err, q_other in itertools.product(
            (50, 150, 250), ("R1", "R2"), (2, 12, 14, 15, 24, 29, 30, 38), (12, 14, 15, 24, 30, 38)):
        q1, q2 = (q_err, q_other) if erring == "R1" else (q_other, q_err)
        edits = [("sub", pos)]
        yield (dict(family="sub_quality", mate=erring, kind="sub", pos=pos, q_err=q_err,
                    q_other=q_other),
               pair(template, edits if erring == "R1" else (), edits if erring == "R2" else (),
                    {pos: q1}, {r2pos(pos): q2}))
    for k, split in itertools.product(range(11), (False, True)):
        sites = [round(20 + i * (n - 40) / max(k - 1, 1)) for i in range(k)]
        r1 = [("sub", p) for i, p in enumerate(sites) if not split or i % 2 == 0]
        r2 = [("sub", p) for i, p in enumerate(sites) if split and i % 2 == 1]
        yield (dict(family="sub_count", mate="split" if split else "R1", kind="sub", k=k),
               pair(template, r1, r2))
    for k, start in itertools.product(range(3, 11), (0, 60)):
        sites = [start + 1 + round(i * 47 / (k - 1)) for i in range(k)]
        yield (dict(family="sub_window", mate="R1", kind="sub", k=k, pos=start),
               pair(template, [("sub", p) for p in sites]))
    for kind, erring, pos in itertools.product(("ins", "del"), ("R1", "R2"), range(0, n, 4)):
        edits = [(kind, pos)]
        yield (dict(family="indel", mate=erring, kind=kind, pos=pos),
               pair(template, edits if erring == "R1" else (), edits if erring == "R2" else ()))
    for kind, pos in itertools.product(("sub", "ins", "del"), (50, 150, 250)):
        yield (dict(family="pcr", mate="both", kind=kind, pos=pos),
               pair(template, [(kind, pos)], [(kind, pos)]))


def run_fastp(pairs: list, fastp: str) -> dict[str, str]:
    """Merged sequence by pair name."""
    # Under dev/, not the system temp dir: Docker Desktop does not share /private/var.
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        tmp = Path(tmp).resolve()
        for i, name in enumerate(("r1", "r2")):
            with open(tmp / f"{name}.fastq", "w") as fh:
                for j, mates in enumerate(pairs):
                    seq, qual = mates[i]
                    fh.write(f"@c{j:05d} {i + 1}:N:0:1\n{seq}\n+\n{qual}\n")
        cmd = (f"{fastp.format(dir=tmp)} --in1 r1.fastq --in2 r2.fastq --out1 o1.fastq "
               f"--out2 o2.fastq --merged_out merged.fastq --json fastp.json --html fastp.html "
               f"{AAP_ARGS}")
        done = subprocess.run(shlex.split(cmd), cwd=tmp, capture_output=True, text=True)
        if done.returncode:
            sys.exit(f"fastp failed ({done.returncode}):\n{done.stderr}")
        lines = (tmp / "merged.fastq").read_text().splitlines()
    return {lines[i][1:].split()[0]: lines[i + 1] for i in range(0, len(lines), 4)}


def describe(row: dict, template: str, merged: str | None) -> dict:
    row["merged"] = merged is not None
    if merged is None:
        return row
    found = edlib.align(merged, template, task="path")
    row.update(merged_len=len(merged), merged_edits=found["editDistance"], cigar=found["cigar"])
    if row["family"] == "sub_quality":
        row["merged_base"] = ("template" if merged[row["pos"]] == template[row["pos"]]
                              else "error" if merged[row["pos"]] == other_base(template[row["pos"]])
                              else merged[row["pos"]])
    return row


def fastp_cmd(a) -> None:
    template = a.template or most_common_read(a.merged)
    rows, pairs = zip(*cases(template))
    merged = run_fastp(list(pairs), a.fastp)
    table = pd.DataFrame([describe(dict(r), template, merged.get(f"c{j:05d}"))
                          for j, r in enumerate(rows)])
    table.to_csv(a.out, index=False)
    print(f"template: {len(template)} bp  {template}")

    quality = table[table.family == "sub_quality"]
    print("\nsub_quality: base the merged read keeps (pooled over positions)")
    print(pd.crosstab([quality.mate, quality.q_err], quality.q_other,
                      values=quality.merged_base.fillna("unmerged"),
                      aggfunc=lambda s: "/".join(sorted(set(s)))))
    count = table[table.family == "sub_count"]
    print("\nsub_count: merged? (merged read edits)")
    print(count.assign(v=count.merged.map({True: "yes", False: "no"})
                       + count.merged_edits.map(lambda m: "" if pd.isna(m) else f" ({m:.0f})"))
          .pivot(index="k", columns="mate", values="v"))
    window = table[table.family == "sub_window"]
    print("\nsub_window: merged? by errors in R1 and window start")
    print(window.pivot(index="k", columns="pos", values="merged"))
    indel = table[table.family == "indel"]
    print("\nindel: unmerged fragment positions; merged-read CIGARs against the template")
    for (kind, erring), group in indel.groupby(["kind", "mate"]):
        print(f"  {kind} {erring}: unmerged {group[~group.merged].pos.astype(int).tolist()}; "
              f"CIGARs {group.cigar.str.replace(r'\d+=', '=', regex=True).value_counts().to_dict()}")
    print("\npcr:")
    print(table[table.family == "pcr"][["kind", "pos", "merged", "cigar"]].to_string(index=False))


MAX_EDITS = 10  # tallies are kept per edit count up to this; --max-edits picks a cut below it
LONGEST = 320


def amplicons(seq: str, fwd: str, rev: str, max_mismatch: int = 3) -> list[str]:
    """Every primer-free V4 amplicon in ``seq``, both strands. ``si.extract_v4`` returns only
    the first per record, and a closed genome carries several rRNA copies.

    ponytail: anchored on the forward primer's non-degenerate 3' 10-mer, exact; a copy with
    a mismatch there would not amplify well anyway.
    """
    anchor, rc_rev, found = fwd[-10:], si.revcomp(rev), []
    for strand in (seq, si.revcomp(seq)):
        for hit in re.finditer(anchor, strand):
            start = hit.end()
            if start < len(fwd) or si._find_primer(strand[start - len(fwd):start], fwd,
                                                   max_mismatch) != 0:
                continue
            end = si._find_primer(strand[start + 200:start + LONGEST], rc_rev, max_mismatch)
            if end is not None:
                found.append(strand[start:start + 200 + end])
    return found


def tally_run(merged: Path, refs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per read-weighted edit count ``d``: coverage, sub, ins and del by template position,
    shape ``(MAX_EDITS + 1, 4, LONGEST)``, plus reads by outcome
    ``[d = 0..MAX_EDITS, beyond, tied]``."""
    with gzip.open(merged, "rt") as fh:
        reads = Counter(line.strip() for i, line in enumerate(fh) if i % 4 == 1)
    counts = np.zeros((MAX_EDITS + 1, 4, LONGEST))
    outcome = np.zeros(MAX_EDITS + 3)
    for read, n in reads.items():
        best, best_d, tied = None, MAX_EDITS + 1, False
        for ref in refs:
            d = 0 if ref in read else edlib.align(ref, read, mode="HW", task="distance",
                                                  k=best_d)["editDistance"]
            if d == -1:
                continue
            if d < best_d:
                best, best_d, tied = ref, d, False
            elif d == best_d:
                tied = True
        if best is None:
            outcome[MAX_EDITS + 1] += n
            continue
        if tied:
            outcome[MAX_EDITS + 2] += n
            continue
        outcome[best_d] += n
        row = counts[best_d]
        row[0, :len(best)] += n
        if best_d == 0:
            continue
        pos = 0
        for length, op in re.findall(r"(\d+)([=XID])",
                                     edlib.align(best, read, mode="HW", task="path")["cigar"]):
            length = int(length)
            if op == "D":  # read base absent from the template: an insertion before ``pos``
                row[2, min(pos, len(best) - 1)] += n * length
                continue
            if op != "=":
                row[1 if op == "X" else 3, pos:pos + length] += n
            pos += length
    return counts, outcome


INTERIOR = slice(25, -10)  # template columns: past fastp's filtered 5' window (fragment ~44) and
# the ragged 3' end next to the reverse primer
OUTLIER = 5  # a column whose substitution rate exceeds this many times the median is a variant


def interior_rates(kept: np.ndarray) -> dict:
    """Rates over interior template columns of ``kept`` (``(4, LONGEST)``): the median column,
    and the mean without outlier columns (strain variants, systematic sites)."""
    cols = kept[:, kept[0] > 0][:, INTERIOR]
    pos = np.flatnonzero(kept[0] > 0)[INTERIOR] + len(si.DEFAULT_FWD_PRIMER)
    rates = cols[1:] / cols[0]
    typical = rates[0] <= OUTLIER * np.median(rates[0])
    kept_cols = cols[:, typical]
    # 20-column windows over typical columns: the Phase 3.3 trigger is max/min > 3
    windows = [kept_cols[1, i:i + 20].sum() / kept_cols[0, i:i + 20].sum()
               for i in range(0, kept_cols.shape[1] - 19, 20)]
    return {**{f"{op}_median": np.median(rates[i]) for i, op in enumerate(("sub", "ins", "del"))},
            **{op: kept_cols[i].sum() / kept_cols[0].sum() for i, op in enumerate(("sub", "ins", "del"), 1)},
            "outlier_columns": " ".join(map(str, pos[~typical])),
            "sub_window_max_min": max(windows) / max(min(windows), 1e-12)}


def profile_cmd(a) -> None:
    fwd, rev = si.DEFAULT_FWD_PRIMER, si.DEFAULT_REV_PRIMER
    refs = sorted({amp for path in sorted(a.genomes.iterdir()) if ".f" in path.name
                   for _, seq in si.read_fasta(path) for amp in amplicons(seq, fwd, rev)})
    runs = sorted(a.runs.glob("*/qc/*.merged.fastq.gz"))
    names = [path.name.removesuffix(".merged.fastq.gz") for path in runs]
    print(f"{len(refs)} distinct amplicons; {len(runs)} runs", flush=True)
    with ProcessPoolExecutor(a.threads) as pool:
        results = list(pool.map(tally_run, runs, [refs] * len(runs)))
    results.append(tuple(map(sum, zip(*results))))

    frames, summary = [], []
    for name, (counts, outcome) in zip(names + ["pooled"], results):
        kept = counts[:a.max_edits + 1].sum(axis=0)
        covered = kept[0] > 0
        rates = kept[1:, covered] / kept[0, covered]
        frames.append(pd.DataFrame({"run": name, "pos": np.flatnonzero(covered) + len(fwd),
                                    "coverage": kept[0, covered].astype(int), "sub": rates[0],
                                    "ins": rates[1], "del": rates[2]}))
        total = outcome.sum()
        summary.append({"run": name, "reads": int(total),
                        "kept": outcome[:a.max_edits + 1].sum() / total,
                        "exact": outcome[0] / total,
                        "beyond": outcome[a.max_edits + 1:MAX_EDITS + 2].sum() / total,
                        "tied": outcome[MAX_EDITS + 2] / total, **interior_rates(kept)})
    pd.concat(frames).to_csv(a.out, sep="\t", index=False, float_format="%.3g")
    table = pd.DataFrame(summary)
    table.to_csv(a.out.with_suffix(".summary.tsv"), sep="\t", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(table.assign(run=table.run.str[:26], outlier_columns=table.outlier_columns.str.count(" ") + 1)
          .to_string(index=False, float_format="%.3g"))
    print("\npooled interior rates by edit cap:")
    pooled = results[-1][0]
    print(pd.DataFrame([{"cap": cap, **interior_rates(pooled[:cap + 1].sum(axis=0))}
                        for cap in (1, 2, 3, 4, 6, 8, 10)]).to_string(index=False, float_format="%.3g",
                                                                       max_colwidth=60))


def primer_pattern(primer: str) -> re.Pattern:
    """Degenerate positions match any base, so off-code bases are counted too; the fixed
    positions must match exactly."""
    return re.compile("".join("." if len(si._IUPAC[c]) > 1 else c for c in primer))


def tally_primers(merged: Path, amp_genome: dict[str, str]) -> tuple[Counter, Counter, Counter]:
    """Reads by ``(genome, primer, position, base)``, by ``(spacer, overhang, forward
    degenerate bases, reverse degenerate bases)``, and by outcome. Reverse-primer bases are reported on the primer's own strand."""
    fwd, rc_rev = si.DEFAULT_FWD_PRIMER, si.revcomp(si.DEFAULT_REV_PRIMER)
    fwd_re, rev_re = primer_pattern(fwd), primer_pattern(rc_rev)
    fwd_sites = [i for i, c in enumerate(fwd) if len(si._IUPAC[c]) > 1]
    rev_sites = [i for i, c in enumerate(rc_rev) if len(si._IUPAC[c]) > 1]
    with gzip.open(merged, "rt") as fh:
        reads = Counter(line.strip() for i, line in enumerate(fh) if i % 4 == 1)
    bases, ends, outcome = Counter(), Counter(), Counter()
    for read, n in reads.items():
        start = fwd_re.search(read, 0, 40)
        # ponytail: last exact-fixed-base hit in the final 60 bases
        tail = len(read) - 60
        stops = list(rev_re.finditer(read, max(tail, 0)))
        if start is None or not stops:
            outcome["no_primer" if start is None else "no_rev_primer"] += n
            continue
        stop = stops[-1]
        genome = amp_genome.get(read[start.end():stop.start()], None)
        outcome["assigned" if genome else "unassigned"] += n
        ends[(start.start(), len(read) - stop.end(),
              "".join(read[start.start() + i] for i in fwd_sites),
              si.revcomp("".join(read[stop.start() + i] for i in rev_sites)))] += n
        for label in ("all", genome) if genome else ("all",):
            for i in fwd_sites:
                bases[(label, "515F", i, read[start.start() + i])] += n
            for i in rev_sites:
                primer_pos = len(rc_rev) - 1 - i
                bases[(label, "806R", primer_pos, si.revcomp(read[stop.start() + i]))] += n
    return bases, ends, outcome


def primers_cmd(a) -> None:
    fwd, rev = si.DEFAULT_FWD_PRIMER, si.DEFAULT_REV_PRIMER
    amp_genome: dict[str, set] = {}
    for path in sorted(a.genomes.iterdir()):
        if ".f" in path.name:
            for _, seq in si.read_fasta(path):
                for amp in amplicons(seq, fwd, rev):
                    amp_genome.setdefault(amp, set()).add(path.name.split(".f")[0])
    amp_genome = {amp: "+".join(sorted(g)) for amp, g in amp_genome.items()}
    runs = sorted(a.runs.glob("*/qc/*.merged.fastq.gz"))
    names = [path.name.removesuffix(".merged.fastq.gz") for path in runs]
    with ProcessPoolExecutor(a.threads) as pool:
        results = list(pool.map(tally_primers, runs, [amp_genome] * len(runs)))
    results.append(tuple(sum((r[i] for r in results), Counter()) for i in range(3)))

    rows, end_rows, outcomes = [], [], []
    for name, (bases, ends, outcome) in zip(names + ["pooled"], results):
        table = (pd.Series(bases).rename_axis(["genome", "primer", "pos", "base"]).unstack("base")
                 .reindex(columns=list("ACGT")).fillna(0))
        reads = table.sum(axis=1)
        table = table.div(reads, axis=0).assign(reads=reads.astype(int)).reset_index()
        primer_seq = {"515F": fwd, "806R": rev}
        table.insert(3, "code", [primer_seq[p][i] for p, i in zip(table.primer, table.pos)])
        rows.append(table.assign(run=name))
        end_rows.append(pd.Series(ends).rename_axis(["spacer", "overhang", "fwd_bases", "rev_bases"]).rename("reads")
                        .reset_index().assign(run=name))
        outcomes.append({"run": name, **outcome})
    mix = pd.concat(rows)[["run", "genome", "primer", "pos", "code", "A", "C", "G", "T", "reads"]]
    mix.to_csv(a.out, sep="\t", index=False, float_format="%.4f")
    pd.concat(end_rows)[["run", "spacer", "overhang", "fwd_bases", "rev_bases", "reads"]].to_csv(
        a.out.with_suffix(".ends.tsv"), sep="\t", index=False)

    pd.set_option("display.width", 250)
    out = pd.DataFrame(outcomes).fillna(0).set_index("run")
    print((out.div(out.sum(axis=1), axis=0)).assign(reads=out.sum(axis=1).astype(int))
          .rename(index=lambda r: r[:26]).to_string(float_format="%.3f"))
    pooled = mix[mix.run == "pooled"]
    print("\npooled, all reads:")
    print(pooled[pooled.genome == "all"].to_string(index=False, float_format="%.3f"))
    print("\npooled, per genome (frequency of each code's options):")
    per = pooled[pooled.genome != "all"].copy()
    per["mix"] = [" ".join(f"{b}{r[b]:.2f}" for b in sorted(si._IUPAC[r.code]))
                  + (f" off{1 - sum(r[b] for b in si._IUPAC[r.code]):.3f}") for _, r in per.iterrows()]
    print(per.pivot_table(index=["genome"], columns=["primer", "pos", "code"], values="mix",
                          aggfunc="first").assign(reads=per.groupby("genome").reads.max())
          .sort_values("reads", ascending=False).to_string())
    print("\nper-run spread, all reads (min-max of the major option):")
    allrows = mix[(mix.genome == "all") & (mix.run != "pooled")]
    for (primer, pos), g in allrows.groupby(["primer", "pos"]):
        major = g[list("ACGT")].mean().idxmax()
        print(f"  {primer} {pos} {major}: {g[major].min():.3f}-{g[major].max():.3f}")
    ends = pd.concat(end_rows)
    ends = ends[ends.run == "pooled"]
    for col in ("spacer", "overhang"):
        dist = ends.groupby(col).reads.sum()
        print(f"\n{col}: " + ", ".join(f"{k}:{v / dist.sum():.3f}" for k, v in dist.items()
                                        if v / dist.sum() >= 0.001))
    rev_sites = sorted(i for i, c in enumerate(rev) if len(si._IUPAC[c]) > 1)
    for col in ("spacer", "overhang"):
        print(f"\npooled mix by {col} length (806R N shows no G, so G is left out there):")
        for length, g in ends.groupby(col):
            if g.reads.sum() / ends.reads.sum() < 0.001:
                continue
            w = g.reads / g.reads.sum()
            cells = [f"515F8 A{(w * (g.fwd_bases.str[1] == 'A')).sum():.2f}"]
            cells += [f"806R{p} " + " ".join(f"{b}{(w * (g.rev_bases.str[k] == b)).sum():.2f}"
                                             for b in sorted(si._IUPAC[rev[p]]) if b != "G" or p != 7)
                      for k, p in enumerate(rev_sites)]
            print(f"  {length}: " + "  ".join(cells))
    print("\nsites independent? largest |joint - product of marginals| over base pairs, pooled"
          " and within each overhang length:")
    total = ends.reads.sum()
    bases = pd.DataFrame({"f": ends.fwd_bases.str[1], **{f"r{k}": ends.rev_bases.str[k] for k in range(3)},
                          "reads": ends.reads, "spacer": ends.spacer.astype(str),
                          "overhang": ends.overhang.astype(str)})
    cols = ["f", "r0", "r1", "r2", "spacer", "overhang"]
    for x, y in itertools.combinations(cols, 2):
        joint = bases.groupby([x, y]).reads.sum() / total
        px, py = bases.groupby(x).reads.sum() / total, bases.groupby(y).reads.sum() / total
        dev = max(abs(v - px[i] * py[j]) for (i, j), v in joint.items())
        strata = []
        for _, g in bases.groupby("overhang"):
            n = g.reads.sum()
            if n < 0.001 * total:
                continue
            gj, gx, gy = (g.groupby([x, y]).reads.sum() / n, g.groupby(x).reads.sum() / n,
                          g.groupby(y).reads.sum() / n)
            strata.append(max(abs(v - gx[i] * gy[j]) for (i, j), v in gj.items()))
        print(f"  {x}-{y}: {dev:.4f}  within overhang: {max(strata):.4f}")


MAPSEQ = ("docker run --rm --platform linux/amd64 -v {work}:{work} -v {db_dir}:{db_dir} "
          "quay.io/biocontainers/mapseq:{tag} mapseq")
AAP_MAPSEQ_TAG = "2.1.1b--h3ab3c3b_0"  # AAP modules/ebi-metagenomics/mapseq
SR_MAPSEQ_TAG = "2.1.1b--hc47f52e_1"  # nextflow.config mapseq_tag
# AAP's conf/modules.config MAPSEQ ext.args; SR's default params.mapseq_args is empty
MAPSEQ_ARGS = "-seed 12 -tophits 80 -topotus 40 -outfmt simple"


def run_mapseq(out: Path, fasta: Path, db: Path, tax: Path, args: str, mapseq: str, tag: str,
               threads: int) -> None:
    """Map ``fasta`` unless ``out`` already holds a finished run."""
    if out.exists() and out.stat().st_size:
        return
    cmd = (f"{mapseq.format(work=out.parent, db_dir=db.parent, tag=tag)} {fasta} {db} {tax} "
           f"-nthreads {threads} {args}")
    print(f"mapping {out.name}", flush=True)
    with open(out.with_suffix(".tmp"), "w") as fh:
        done = subprocess.run(shlex.split(cmd), stdout=fh, stderr=subprocess.PIPE, text=True)
    if done.returncode:
        sys.exit(f"mapseq failed ({done.returncode}):\n{done.stderr[-2000:]}")
    out.with_suffix(".tmp").rename(out)


def v4_changes(pairs, db: Path) -> Counter:
    """For ``(hit, other hit)`` pairs that differ: same V4 amplicon, different, or a hit
    without one. SR's labels are V4 groups, so only ``different_v4`` changes its input."""
    pairs = [(x, y) for x, y in pairs if x != y]
    need, amplicon, name = {h for pair in pairs for h in pair}, {}, None
    with open(db) as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
            elif name in need:
                amplicon[name] = si.extract_v4(line.strip().upper().replace("U", "T"),
                                               si.DEFAULT_FWD_PRIMER, si.DEFAULT_REV_PRIMER, 3)
    return Counter("no_amplicon" if amplicon[x] is None or amplicon[y] is None
                   else "same_v4" if amplicon[x] == amplicon[y] else "different_v4"
                   for x, y in pairs)


def read_mseq(path: Path) -> pd.DataFrame:
    """Query, top hit and final classification of a ``-outfmt simple`` file."""
    rows = [line.rstrip("\n").split("\t") for line in open(path) if not line.startswith("#")]
    return pd.DataFrame({"query": [r[0] for r in rows], "dbhit": [r[1] for r in rows],
                         "tax": [r[13] if len(r) > 13 else "" for r in rows]}).set_index("query")


def clipped_reads(merged: Path, deoverlapped: Path):
    """Yield ``(name, read, clipped read)`` for reads with a cmsearch hit, cut to the hit's
    columns 8-9 as AAP's ``esl-sfetch -Cf`` does (reverse-complemented when start > end)."""
    hits = {}
    for line in open(deoverlapped):
        f = line.split()
        hits[f[0]] = (int(f[7]), int(f[8]))
    with gzip.open(merged, "rt") as fh:
        for i, line in enumerate(fh):
            if i % 4 == 0:
                name = line[1:].split()[0]
            elif i % 4 == 1 and name in hits:
                read, (start, end) = line.strip(), hits[name]
                yield name, read, (read[start - 1:end] if start <= end
                                   else si.revcomp(read[end - 1:start]))


def clip_cmd(a) -> None:
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "merged.fasta", "w") as whole, open(work / "clipped.fasta", "w") as clipped:
        for name, read, cut in clipped_reads(a.merged, a.deoverlapped):
            whole.write(f">{name}\n{read}\n")
            clipped.write(f">{name}\n{cut}\n")
    db, tax = a.db.resolve(), a.tax.resolve()
    for label, fasta in (("merged", "merged"), ("clipped", "clipped"), ("replicate", "merged")):
        run_mapseq(work / f"{label}.mseq", work / f"{fasta}.fasta", db, tax, MAPSEQ_ARGS,
                   a.mapseq, AAP_MAPSEQ_TAG, a.threads)

    merged, clipped, replicate = (read_mseq(work / f"{x}.mseq") for x in ("merged", "clipped", "replicate"))
    ranks = ["sk", "k", "p", "c", "o", "f", "g", "s"]

    def first_difference(x: str, y: str) -> str:
        """Rank at which two classifications part, or ``same``."""
        xs, ys = x.split(";"), y.split(";")
        for i in range(max(len(xs), len(ys))):
            if i >= len(xs) or i >= len(ys) or xs[i] != ys[i]:
                return ranks[i] if i < len(ranks) else str(i)
        return "same"

    rows = []
    for label, other in (("replicate", replicate), ("clipped", clipped)):
        both = merged.join(other, rsuffix="_other", how="inner")
        classified = (both.tax != "") | (both.tax_other != "")
        rows.append({"comparison": f"merged vs {label}", "reads": len(both),
                     "same_dbhit": (both.dbhit == both.dbhit_other).mean(),
                     "same_tax": (both.tax == both.tax_other).mean(),
                     "classified_only_one": ((both.tax == "") != (both.tax_other == "")).mean(),
                     **{f"parts_at_{r}": n / len(both) for r, n in Counter(
                         first_difference(x, y) for x, y in zip(both.tax[classified],
                                                                both.tax_other[classified])).items()
                        if r != "same"}})
    # A changed top hit matters to SR only if it changes the V4 group: are the two hits'
    # amplicons identical?
    both = merged.join(clipped, rsuffix="_clipped")
    groups = v4_changes(zip(both.dbhit, both.dbhit_clipped), db)
    rows[1].update({f"dbhit_changed_{k}": v / len(both) for k, v in groups.items()})
    table = pd.DataFrame(rows).fillna(0)
    table.to_csv(a.out, sep="\t", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(table.set_index("comparison").T.round(5).to_string())
    changed = both[both.tax != both.tax_clipped]
    print("\nmost common classification changes (merged -> clipped):")
    print(changed.groupby(["tax", "tax_clipped"]).size().sort_values(ascending=False).head(10)
          .to_string())


def builds_cmd(a) -> None:
    """0.5: AAP's MAPseq build against SR's on the clipped reads from ``clip``."""
    work, db, tax = a.work.resolve(), a.db.resolve(), a.tax.resolve()
    reference = work / "clipped.mseq"
    if not reference.exists():
        sys.exit(f"{reference} missing: run the clip subcommand first")
    runs = {"sr_build_aap_args": MAPSEQ_ARGS, "sr_build_sr_default_args": ""}
    for label, args in runs.items():
        run_mapseq(work / f"{label}.mseq", work / "clipped.fasta", db, tax, args, a.mapseq,
                   SR_MAPSEQ_TAG, a.threads)

    def header(path: Path) -> list[str]:
        with open(path) as fh:
            return next(line for line in fh if line.startswith("#query")).rstrip("\n").split("\t")

    # iter_mseq reads query, hit and identity at 0, 1 and 3: check those names in each format
    wanted = {si._MSEQ_QUERY: "#query", si._MSEQ_HIT: "dbhit", si._MSEQ_IDENTITY: "identity"}
    aap = dict(si.iter_mseq(reference))
    rows = []
    for label in ["aap_build_aap_args", *runs]:
        path = reference if label == "aap_build_aap_args" else work / f"{label}.mseq"
        cols = header(path)
        hits = dict(si.iter_mseq(path))
        with open(path) as fh:
            first = next(line for line in fh if not line.startswith("#")).rstrip("\n").split("\t")
        groups = v4_changes(((aap[q], hits.get(q, "")) for q in aap), db)
        rows.append({"run": label, "columns_ok": all(cols[i] == n for i, n in wanted.items()),
                     "columns": len(cols), "rows_read": len(hits),
                     "tax_column": first[13] if len(first) > 13 else "",
                     "same_dbhit_as_aap": sum(hits.get(q) == h for q, h in aap.items()) / len(aap),
                     **{f"changed_{k}": v / len(aap) for k, v in groups.items()}})
    table = pd.DataFrame(rows).fillna(0)
    table.drop(columns="tax_column").to_csv(a.out, sep="\t", index=False, float_format="%.5f")
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(table.set_index("run").T.to_string())


STATUSES = ("amplicon", "fwd_site_indel", "rev_site_indel", "no_fwd", "no_rev", "neither")


def amplifiable(seq: str) -> str:
    """``amplicon`` if ``si.extract_v4`` cuts one (3 mismatches, as EXTRACT_AMPLICONS).
    Otherwise, for the site the ungapped scan misses: ``*_site_indel`` if it is within 3 edits
    with gaps allowed, ``no_*`` if it is not there at all, ``neither`` if both sites are
    absent. Checked on the strand where the other primer sits."""
    fwd, rc_rev = si.DEFAULT_FWD_PRIMER, si.revcomp(si.DEFAULT_REV_PRIMER)
    seq = seq.upper().replace("U", "T")
    if si.extract_v4(seq, fwd, si.DEFAULT_REV_PRIMER, 3):
        return "amplicon"
    iupac = [(c, "".join(sorted(v))) for c, v in si._IUPAC.items() if len(v) > 1]

    def site(strand: str, primer: str) -> str:
        if si._find_primer(strand, primer, 3) is not None:
            return "ok"
        found = edlib.align(primer, strand, mode="HW", task="distance", k=3,
                            additionalEqualities=[(c, b) for c, bases in iupac for b in bases])
        return "indel" if found["editDistance"] != -1 else "absent"

    best = None
    for strand in (seq, si.revcomp(seq)):
        f, r = site(strand, fwd), site(strand, rc_rev)
        rank = (f != "absent") + (r != "absent")
        if best is None or rank > best[0]:
            best = (rank, f, r)
    _, f, r = best
    if f == "absent" and r == "absent":
        return "neither"
    if f != "ok":
        return "fwd_site_indel" if f == "indel" else "no_fwd"
    return "rev_site_indel" if r == "indel" else "no_rev"


def coverage_cmd(a) -> None:
    """0.6: share of each run's reads whose top hit has no V4 amplicon."""
    work, db, tax = a.work.resolve(), a.db.resolve(), a.tax.resolve()
    work.mkdir(parents=True, exist_ok=True)
    runs = sorted(a.runs.glob("*/qc/*.merged.fastq.gz"))
    names = [path.name.removesuffix(".merged.fastq.gz") for path in runs]
    mseq = work / "all_runs.mseq"
    if not (mseq.exists() and mseq.stat().st_size):
        with open(work / "all_runs.fasta", "w") as fh:
            for i, (path, name) in enumerate(zip(runs, names)):
                (deoverlapped,) = path.parents[1].glob("sequence-categorisation/*.deoverlapped")
                for read_name, _, cut in clipped_reads(path, deoverlapped):
                    fh.write(f">{i}|{read_name}\n{cut}\n")
        run_mapseq(mseq, work / "all_runs.fasta", db, tax, MAPSEQ_ARGS, a.mapseq, AAP_MAPSEQ_TAG,
                   a.threads)
        (work / "all_runs.fasta").unlink()  # ponytail: ~3 GB, rebuilt from the runs if needed

    hits_by_run = [Counter() for _ in runs]
    unhit, total = Counter(), Counter()
    with open(mseq) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 2)
            run = int(f[0].split("|", 1)[0])
            total[run] += 1
            if f[1]:
                hits_by_run[run][f[1]] += 1
            else:
                unhit[run] += 1
    need = set().union(*hits_by_run)
    status, name = {}, None
    with open(db) as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
            elif name in need:
                status[name] = amplifiable(line.strip())
    tax_of = {}
    with open(tax) as fh:
        for line in fh:
            ref, _, lineage = line.rstrip("\n").partition("\t")
            if ref in need and status[ref] != "amplicon":
                tax_of[ref] = lineage

    rows, missing = [], Counter()
    for i, name in enumerate(names):
        by_status = Counter()
        for ref, n in hits_by_run[i].items():
            by_status[status[ref]] += n
            if status[ref] != "amplicon":
                missing[(status[ref], ref)] += n
        rows.append({"run": name, "reads": total[i], "unhit": unhit[i] / total[i],
                     **{k: by_status[k] / total[i] for k in
                        STATUSES}})
    table = pd.DataFrame(rows)
    pooled = (table.drop(columns="run").mul(table.reads, axis=0).sum() / table.reads.sum())
    table = pd.concat([table, pd.DataFrame([{"run": "pooled", **pooled,
                                             "reads": int(table.reads.sum())}])])
    table["non_amplifiable"] = 1 - table.amplicon - table.unhit
    table.to_csv(a.out, sep="\t", index=False, float_format="%.5f")
    pd.DataFrame([{"status": st, "ref": ref, "reads": n, "lineage": tax_of.get(ref, "")}
                  for (st, ref), n in missing.most_common()]).to_csv(
        a.out.with_suffix(".refs.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(f"{len(need)} distinct top hits; "
          f"{sum(v != 'amplicon' for v in status.values())} without an amplicon")
    print(table.assign(run=table.run.str[:26]).to_string(index=False, float_format="%.4f"))
    print("\nmost-hit references without an amplicon:")
    for (st, ref), n in missing.most_common(12):
        print(f"  {n:7d} {st:9s} {ref:22s} {tax_of.get(ref, '')[-70:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    fp = sub.add_parser("fastp", help="placed errors through AAP's fastp merge (0.1)")
    fp.add_argument("--merged", type=Path, help="an AAP run's .merged.fastq.gz; the template source")
    fp.add_argument("--template", help="use this fragment instead of the most common merged read")
    fp.add_argument("--fastp", default=FASTP, help="fastp command; {dir} is the work directory")
    fp.add_argument("--out", type=Path, required=True)
    fp.set_defaults(func=fastp_cmd)
    pr = sub.add_parser("profile", help="per-position residual error in merged reads (0.2)")
    pr.add_argument("--genomes", type=Path, required=True,
                    help="directory of the community's genome FASTAs (.gz ok)")
    pr.add_argument("--runs", type=Path, required=True, help="AAP outdir; reads <run>/qc/*.merged.fastq.gz")
    pr.add_argument("--max-edits", type=int, default=4, choices=range(MAX_EDITS + 1))
    pr.add_argument("--threads", type=int, default=8)
    pr.add_argument("--out", type=Path, required=True)
    pr.set_defaults(func=profile_cmd)
    pm = sub.add_parser("primers", help="base mix at degenerate primer positions (0.3)")
    pm.add_argument("--genomes", type=Path, required=True)
    pm.add_argument("--runs", type=Path, required=True)
    pm.add_argument("--threads", type=int, default=8)
    pm.add_argument("--out", type=Path, required=True)
    pm.set_defaults(func=primers_cmd)
    cl = sub.add_parser("clip", help="MAPseq on merged against cmsearch-clipped reads (0.4)")
    cl.add_argument("--merged", type=Path, required=True)
    cl.add_argument("--deoverlapped", type=Path, required=True)
    cl.add_argument("--db", type=Path, required=True, help="MAPseq FASTA; .mscluster beside it")
    cl.add_argument("--tax", type=Path, required=True)
    cl.add_argument("--mapseq", default=MAPSEQ, help="{work}, {db_dir} and {tag} are filled in")
    cl.add_argument("--threads", type=int, default=10)
    cl.add_argument("--work", type=Path, required=True, help="keeps FASTAs and .mseq; reruns reuse them")
    cl.add_argument("--out", type=Path, required=True)
    cl.set_defaults(func=clip_cmd)
    bd = sub.add_parser("builds", help="AAP's MAPseq build against SR's, on clip's reads (0.5)")
    bd.add_argument("--db", type=Path, required=True)
    bd.add_argument("--tax", type=Path, required=True)
    bd.add_argument("--mapseq", default=MAPSEQ, help="{work}, {db_dir} and {tag} are filled in")
    bd.add_argument("--threads", type=int, default=10)
    bd.add_argument("--work", type=Path, required=True, help="the clip subcommand's work directory")
    bd.add_argument("--out", type=Path, required=True)
    bd.set_defaults(func=builds_cmd)
    cv = sub.add_parser("coverage", help="reads whose top hit has no V4 amplicon, per run (0.6)")
    cv.add_argument("--runs", type=Path, required=True, help="AAP outdir")
    cv.add_argument("--db", type=Path, required=True)
    cv.add_argument("--tax", type=Path, required=True)
    cv.add_argument("--mapseq", default=MAPSEQ, help="{work}, {db_dir} and {tag} are filled in")
    cv.add_argument("--threads", type=int, default=10)
    cv.add_argument("--work", type=Path, required=True, help="keeps the combined .mseq")
    cv.add_argument("--out", type=Path, required=True)
    cv.set_defaults(func=coverage_cmd)
    a = ap.parse_args()
    if a.func is fastp_cmd and not (a.merged or a.template):
        ap.error("fastp needs --merged or --template")
    a.func(a)


if __name__ == "__main__":
    main()
