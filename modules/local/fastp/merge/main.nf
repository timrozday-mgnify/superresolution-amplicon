// Merge simulated read pairs with amplicon-analysis-pipeline's own fastp build and
// arguments (READS_QC_MERGE, conf/modules.config), so the simulated reads pass through
// the same merge as the observed ones (--sim_read_structure pairs). Unmerged pairs are
// dropped, as AAP drops them. The merged reads go to MAPseq as FASTA named by their first
// token, which strips fastp's " merged_x_y" suffix and keeps the source-carrying name.
// yield.tsv is each source's merged over simulated pairs (plan effect 5).
// ponytail: no cmsearch clip. The pairs carry no spacer or overhang, so a merged read
// already spans primer to primer, which is what the clip leaves (dev/aap_merge_effects.md 0.4).
process FASTP_MERGE {
    tag "$meta.id"
    label 'process_medium'

    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container
        ? 'https://community-cr-prod.seqera.io/docker/registry/v2/blobs/sha256/52/527b18847a97451091dba07a886b24f17f742a861f9f6c9a6bfb79d4f1f3bf9d/data'
        : 'community.wave.seqera.io/library/fastp:1.0.1--c8b87fe62dcc103c'}"

    input:
    tuple val(meta), path(pairs)

    output:
    tuple val(meta), path("${meta.id}.sim.fasta"), emit: reads
    tuple val(meta), path("${meta.id}.yield.tsv"), emit: yield
    tuple val(meta), path("${meta.id}.fastp.json"), emit: json
    path "versions.yml",                            emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def (r1, r2) = pairs.sort { it.name }
    """
    fastp \\
        --in1 ${r1} \\
        --in2 ${r2} \\
        --merged_out merged.fastq.gz \\
        --json ${prefix}.fastp.json \\
        --html ${prefix}.fastp.html \\
        --thread ${task.cpus} \\
        $args

    zcat < merged.fastq.gz | awk 'NR%4==1 { print ">" substr(\$1, 2) } NR%4==2' > ${prefix}.sim.fasta

    printf 'source\\tsimulated\\tmerged\\tyield\\n' > ${prefix}.yield.tsv
    { zcat < ${r1} | awk 'NR%4==1 { print "s", substr(\$1, 2) }'
      awk '/^>/ { print "m", substr(\$1, 2) }' ${prefix}.sim.fasta; } \\
        | awk '{ sub(/:[0-9]+\$/, "", \$2); n[\$2] += (\$1 == "s"); m[\$2] += (\$1 == "m") }
               END { for (k in n) printf "%s\\t%d\\t%d\\t%.4f\\n", k, n[k], m[k], m[k] / n[k] }' \\
        | sort >> ${prefix}.yield.tsv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastp: \$(fastp --version 2>&1 | sed -e "s/fastp //g")
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    printf '>ref|0|x:0\\nACGTACGTACGT\\n' > ${prefix}.sim.fasta
    printf 'source\\tsimulated\\tmerged\\tyield\\nref|0|x\\t1\\t1\\t1.0000\\n' > ${prefix}.yield.tsv
    echo '{}' > ${prefix}.fastp.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        fastp: stub
    END_VERSIONS
    """
}
