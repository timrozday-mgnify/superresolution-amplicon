#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root; MAPseq outputs land in work/silva/mapseq_probe
S=work/silva; P=$S/mapseq_probe
IMG=quay.io/biocontainers/mapseq:2.1.1b--hc47f52e_1
mapseq() { docker run --rm --platform linux/amd64 -v "$PWD:$PWD" -w "$PWD" "$IMG" mapseq "$@"; }
run() { n=$1; shift; [ -s $P/$n.mseq ] || mapseq $P/sim600.fasta $S/amp/amplicons.fasta $S/amp/amplicons.tax -nthreads 12 "$@" > $P/$n.mseq 2> $P/$n.log; echo "$n done"; }
run sim_default
run sim_default_rep
run sim_t50 -otulim 0 -topotus 50
run sim_t200 -otulim 0 -topotus 200 -tophits 100
run sim_default_hits -print_hits
run sim_t50_hits -otulim 0 -topotus 50 -print_hits
