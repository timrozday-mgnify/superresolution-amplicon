// Cut a genome panel's V4 copies into distinct sources (v4g_<sha16>) and the genome ->
// source translation. The sources are then mapped (home labels) and simulated from against
// the sample's generic database, the same way the real reads are.
process PANEL_PREPARE {
    tag "$meta.id"
    label 'process_low'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(panel), path(db_amplicons)

    output:
    tuple val(meta), path("${meta.id}_prepared"),               emit: prepared
    tuple val(meta), path("${meta.id}_prepared/sources.fasta"), emit: sources
    path "versions.yml",                                        emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    build_panel_kernel.py prepare \\
        --panel-amplicons ${panel} \\
        --db-amplicons ${db_amplicons} \\
        -o ${meta.id}_prepared \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${meta.id}_prepared
    printf '>v4g_0000000000000000\\nACGTACGTACGT\\n' > ${meta.id}_prepared/sources.fasta
    printf 'genome_id\\tsource\\tweight\\nref\\tv4g_0000000000000000\\t1.0\\n' > ${meta.id}_prepared/panel_translation.tsv
    printf 'source\\tgenomes\\tin_db\\nv4g_0000000000000000\\tref\\tTrue\\n' > ${meta.id}_prepared/sources.tsv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
