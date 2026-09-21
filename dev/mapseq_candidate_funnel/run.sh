#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root; MAPseq outputs land in work/silva/mapseq_probe
S=work/silva; P=$S/mapseq_probe
IMG=quay.io/biocontainers/mapseq:2.1.1b--hc47f52e_1
mapseq() { docker run --rm --platform linux/amd64 -v "$PWD:$PWD" -w "$PWD" "$IMG" mapseq "$@"; }
run() { n=$1; shift; [ -s $P/$n.mseq ] || mapseq $P/probe.fasta $S/amp/amplicons.fasta $S/amp/amplicons.tax -nthreads 12 -print_hits "$@" > $P/$n.mseq 2> $P/$n.log; echo "$n done"; }
run default
run default_rep
run otulim0 -otulim 0
run wide -otulim 0 -topotus 50 -tophits 200
