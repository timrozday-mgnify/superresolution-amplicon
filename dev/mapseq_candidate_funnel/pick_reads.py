"""Pick S01 reads that match a SILVA amplicon exactly but MAPseq labelled elsewhere."""
import random
from collections import defaultdict
amp = defaultdict(list); seq_of = {}
name = None
for line in open("../amp/amplicons.fasta"):
    line = line.strip()
    if line.startswith(">"): name = line[1:].split()[0]
    else: seq_of[name] = seq_of.get(name, "") + line
for n, s in seq_of.items(): amp[s].append(n)
reads = {}; name = None
for line in open("../../panel_obs/S01.fasta"):
    line = line.strip()
    if line.startswith(">"): name = line[1:]
    else: reads[name] = reads.get(name, "") + line
hit = {}
for line in open("../obs/S01.obs.mseq"):
    if line.startswith("#"): continue
    f = line.split("\t"); hit[f[0]] = f[1]
exact = [r for r, s in reads.items() if s in amp]
moved = [r for r in exact if r in hit and seq_of[hit[r]] != reads[r]]
stay = [r for r in exact if r in hit and seq_of[hit[r]] == reads[r]]
print(f"reads {len(reads)} exact {len(exact)} moved {len(moved)} stayed {len(stay)}")
random.seed(1)
pick = random.sample(moved, 1000) + random.sample(stay, 1000)
with open("probe.fasta", "w") as fh:
    for r in pick: fh.write(f">{r}\n{reads[r]}\n")
with open("probe_meta.tsv", "w") as fh:
    for r in pick:
        s = reads[r]
        fh.write(f"{r}\t{int(r in set(moved))}\t{len(amp[s])}\t{hit[r]}\t{len(amp[seq_of[hit[r]]])}\n")
