// Convert the sample's fastq reads to the FASTA mapseq takes, optionally trimming the
// primers so the reads sit in the same coordinate space as the reference amplicons and
// the simulated reads.
process READS_TO_FASTA {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path("${meta.id}.obs.fasta"), emit: reads
    path "versions.yml",                           emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args   ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    reads_to_fasta.py ${reads} \\
        --seed ${params.seed} \\
        -o ${prefix}.obs.fasta \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    printf '>r_0\\nACGTACGTACGT\\n' > ${prefix}.obs.fasta

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
