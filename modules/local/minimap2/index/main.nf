// Build the minimap2 minimizer index for the reference amplicons once, so the all-vs-all
// pass (and any re-run of it) does not re-index. Mirrors MAPSEQ_CLUSTER, which does the
// same for mapseq's .mscluster.
//
// The index bakes in -k/-w: minimap2 ignores those (and any preset) on the command line
// when the target is a prebuilt .mmi. params.minimap2_index_args must therefore carry the
// same -k/-w as params.minimap2_args, which is checked in the workflow.
process MINIMAP2_INDEX {
    tag "$meta.id"
    label 'process_medium'

    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container
        ? "https://depot.galaxyproject.org/singularity/minimap2:${params.minimap2_tag}"
        : "quay.io/biocontainers/minimap2:${params.minimap2_tag}"}"

    input:
    tuple val(meta), path(amplicons)

    output:
    tuple val(meta), path("${meta.id}.mmi"), emit: index
    path "versions.yml",                     emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    minimap2 -t ${task.cpus} $args -d ${prefix}.mmi ${amplicons}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        minimap2: \$(minimap2 --version)
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    : > ${prefix}.mmi

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        minimap2: stub
    END_VERSIONS
    """
}
