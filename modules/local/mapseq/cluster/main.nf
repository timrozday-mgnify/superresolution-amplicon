// Cluster the reference FASTA against itself so mapseq builds & caches its clustering
// file (<fasta>.mscluster). We only keep that side-effect; the search results printed to
// stdout are discarded. Doing it once means MAPSEQ_SIM and MAPSEQ_OBS don't each rebuild
// it. Ported from synthetic-metagenomic-benchmark-pipeline's MAPSEQ_CLUSTER.
process MAPSEQ_CLUSTER {
    tag "$meta.id"
    label 'process_medium'

    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container
        ? "https://depot.galaxyproject.org/singularity/mapseq:${params.mapseq_tag}"
        : "quay.io/biocontainers/mapseq:${params.mapseq_tag}"}"

    input:
    tuple val(meta), path(fasta), path(tax)

    output:
    tuple val(meta), path("${fasta}.mscluster"), emit: mscluster
    path "versions.yml",                         emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    mapseq ${fasta} ${fasta} ${tax} -nthreads ${task.cpus} $args > /dev/null

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mapseq: ${params.mapseq_tag}
    END_VERSIONS
    """

    stub:
    """
    printf '0 0\\n' > ${fasta}.mscluster

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mapseq: stub
    END_VERSIONS
    """
}
