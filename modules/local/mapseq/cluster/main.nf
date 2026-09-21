// Turn an extracted reference set into a complete MAPseq database directory: the same
// files plus mapseq's clustering (<fasta>.mscluster), which is the shape of a prebuilt
// database. Every MAPseq task stages that .mscluster alongside the fasta, so none of them
// re-clusters, and the published directory can be a later run's `references`.
// Ported from synthetic-metagenomic-benchmark-pipeline's MAPSEQ_CLUSTER.
process MAPSEQ_CLUSTER {
    tag "$meta.id"
    label 'process_medium'

    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container
        ? "https://depot.galaxyproject.org/singularity/mapseq:${params.mapseq_tag}"
        : "quay.io/biocontainers/mapseq:${params.mapseq_tag}"}"

    input:
    // Staged under another name: the extracted directory is also "<set id>_amplicons".
    tuple val(meta), path(extracted, stageAs: 'extracted')

    output:
    tuple val(meta), path("${meta.id}_amplicons"), emit: dir
    path "versions.yml",                           emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    # Real copies, not links: storeDir moves the directory and publishDir copies it.
    mkdir ${meta.id}_amplicons
    cp -L extracted/* ${meta.id}_amplicons/
    cd ${meta.id}_amplicons
    # mapseq clusters the whole reference set whatever the query, so a one-record query
    # builds the same .mscluster without the self-search (~15% of the run on SILVA NR99).
    awk '/^>/ { n++ } n <= 1' amplicons.fasta > ../probe.fasta
    mapseq ../probe.fasta amplicons.fasta amplicons.tax -nthreads ${task.cpus} $args > /dev/null
    cd ..

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mapseq: ${params.mapseq_tag}
    END_VERSIONS
    """

    stub:
    """
    mkdir ${meta.id}_amplicons
    cp -L extracted/* ${meta.id}_amplicons/
    printf '0 0\\n' > ${meta.id}_amplicons/amplicons.fasta.mscluster

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        mapseq: stub
    END_VERSIONS
    """
}
