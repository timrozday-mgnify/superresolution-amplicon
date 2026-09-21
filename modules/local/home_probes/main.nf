// Each distinct amplicon once error-free and ext.args' --home-probes times one base off,
// for MAPSEQ_HOME to label with the same database and flags the reads go through. Where
// those land becomes each group's distance-0 mass in GROUPED_MISMAPPING.
process HOME_PROBES {
    tag "$meta.id"
    label 'process_single'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicons)

    output:
    tuple val(meta), path("${meta.id}.home_probes.fasta"), emit: probes
    path "versions.yml",                                   emit: versions

    script:
    def args = task.ext.args ?: ''
    """
    build_mismapping_align.py \\
        --amplicons ${amplicons} \\
        --write-home-probes ${meta.id}.home_probes.fasta \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    """
    printf '>stub:e\\nACGT\\n' > ${meta.id}.home_probes.fasta

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
