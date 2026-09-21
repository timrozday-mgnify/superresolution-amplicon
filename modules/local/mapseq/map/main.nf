// Classify reads against the reference FASTA with mapseq. One process for both
// simulated panel reads (-> the panel kernel) and real reads (-> observed per-reference
// counts), aliased MAPSEQ_PANEL_SIM / MAPSEQ_OBS in the workflow and told apart only by
// ext.prefix. Simulation uses the same mapper and settings as the real-read path.
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
