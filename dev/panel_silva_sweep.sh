#!/usr/bin/env bash
# Phase V: panel reinterpretation against SILVA 138.2 NR99 V4 (see dev/panel_silva_sweep.py).
# Idempotent: every step is skipped when its output exists. Run from the repo root.
#   SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz must be in work/silva/.
set -euo pipefail
S=work/silva
IMG=quay.io/biocontainers/mapseq:2.1.1b--hc47f52e_1
mapseq() { docker run --rm --platform linux/amd64 -v "$PWD:$PWD" -w "$PWD" "$IMG" mapseq "$@"; }
map() {  # map QUERY OUT
    [ -s "$2" ] && return
    mapseq "$1" $S/amp/amplicons.fasta $S/amp/amplicons.tax -nthreads 12 > "$2.tmp" 2> "$2.log"
    mv "$2.tmp" "$2"
}
CAL=(--error-model flat --sub-rate 0.0031 --ins-rate 0.000024 --del-rate 0.000024)

# 1. Database: SILVA -> references + real taxonomy -> V4 amplicons -> MAPseq clusters.
[ -s $S/db/silva_nr99.fasta ] || python bin/build_mapseq_database.py \
    --silva-fasta $S/SILVA_138.2_SSURef_NR99_tax_silva.fasta.gz --output-prefix $S/db/silva_nr99
[ -s $S/amp/amplicons.fasta ] || python bin/subspecies_infer.py amplicons \
    --db-fasta $S/db/silva_nr99.fasta --primer-mismatches 3 -o $S/amp
[ -s $S/amp/amplicons.fasta.mscluster ] || mapseq $S/amp/amplicons.fasta \
    $S/amp/amplicons.fasta $S/amp/amplicons.tax -nthreads 12 > /dev/null 2> $S/amp/cluster.log

# 2. The GTDB sweep's reads (20 samples, and the panel sources' simulated reads) on SILVA.
mkdir -p $S/obs $S/sim
for f in work/panel_obs/S*.fasta; do map "$f" $S/obs/$(basename "$f" .fasta).obs.mseq; done
for m in trained_s0 cal; do map work/panel_kernel/sim_$m.fasta $S/sim/sim_$m.mseq; done

# 3. Panels, their home labels, and simulated reads for the taxon sources (flat, calibrated;
#    1,000 per source rather than 5,000: 9,100 sources).
[ -s $S/taxa/sim_sources.fasta ] || python dev/panel_silva_sweep.py prepare --silva $S
for p in genome taxa species; do map $S/$p/sources.fasta $S/$p/home.mseq; done
if [ ! -s $S/sim/sim_taxa_cal.mseq ]; then
    python - "$S" <<'PY'
import sys
from pathlib import Path
s = Path(sys.argv[1])
records = (s / "taxa" / "sim_sources.fasta").read_text().split(">")[1:]
for k in range(8):
    (s / "sim" / f"taxa_shard{k}.fasta").write_text("".join(">" + r for r in records[k::8]))
PY
    for k in $(seq 0 7); do
        [ -s $S/sim/taxa_shard$k.sim.fasta ] || python bin/simulate_amplicon_reads.py \
            --amplicons $S/sim/taxa_shard$k.fasta "${CAL[@]}" --n-per-ref 1000 --seed $k \
            -o $S/sim/taxa_shard$k.sim.fasta 2> $S/sim/taxa_shard$k.sim.log &
    done
    wait
    cat $S/sim/taxa_shard?.sim.fasta > $S/sim/sim_taxa_cal.fasta
    map $S/sim/sim_taxa_cal.fasta $S/sim/sim_taxa_cal.mseq
    rm $S/sim/sim_taxa_cal.fasta $S/sim/taxa_shard*
fi

# 4. Kernels.
kernel() {  # kernel PANEL NAME SIM_MSEQ
    [ -s $S/$1/$2.npz ] || python bin/build_panel_kernel.py build --prepared $S/$1 \
        --db-amplicons $S/amp/amplicons.fasta --home-mseq $S/$1/home.mseq --sim-mseq "$3" \
        -o $S/$1/$2.npz
}
kernel genome trained_s0 $S/sim/sim_trained_s0.mseq
kernel genome cal $S/sim/sim_cal.mseq
[ -s $S/sim/sim_taxa_all_cal.mseq ] || cat $S/sim/sim_cal.mseq $S/sim/sim_taxa_cal.mseq > $S/sim/sim_taxa_all_cal.mseq
kernel taxa cal $S/sim/sim_taxa_all_cal.mseq
# Species sources are a subset of the genus panel's, so they reuse its simulated reads.
kernel species cal $S/sim/sim_taxa_all_cal.mseq
echo "kernels built"
