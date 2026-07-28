// Classify reads against the amplicon reference set with mapseq. One process for both
// the simulated reads (-> the mis-mapping matrix) and the real reads (-> the observed
// per-reference counts), aliased MAPSEQ_SIM / MAPSEQ_OBS in the workflow and told apart
// only by ext.prefix: the whole point is that the confusion matrix is measured with the
// same mapper and settings the real reads go through.
//
// The pre-built <fasta>.mscluster is staged alongside the fasta, so mapseq reuses it
// instead of re-clustering per task.
process MAPSEQ {
    tag "$meta.id"
    label 'process_medium'

    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container
        ? "https://depot.galaxyproject.org/singularity/mapseq:${params.mapseq_tag}"
        : "quay.io/biocontainers/mapseq:${params.mapseq_tag}"}"

    input:
    tuple val(meta), path(query), path(fasta), path(tax), path(mscluster)

    output:
    tuple val(meta), path("*.mseq.gz"), emit: mseq
    path "versions.yml",                emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args   ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    mapseq ${query} ${fasta} ${tax} -nthreads ${task.cpus} $args > ${prefix}.mseq
    gzip ${prefix}.mseq

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mapseq: ${params.mapseq_tag}
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    printf '#stub\\n' | gzip > ${prefix}.mseq.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mapseq: stub
    END_VERSIONS
    """
}
