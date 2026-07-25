process SKIVER_DUMP {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path("${meta.id}.base_observations.tsv"), emit: base
    path "versions.yml",                                       emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args   ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    // skiver dump reads a single sequence file; concatenate paired/multiple
    // inputs (gzip streams concatenate cleanly) so PE training data is used too.
    """
    cat ${reads} > reads_concat.fastq.gz

    skiver dump reads_concat.fastq.gz \\
        -o ${prefix} \\
        --base \\
        $args

    rm -f reads_concat.fastq.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        skiver: \$(skiver --version 2>&1 | sed 's/skiver //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.base_observations.tsv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        skiver: stub
    END_VERSIONS
    """
}
