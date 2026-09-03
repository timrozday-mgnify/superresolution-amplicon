// All-vs-all alignment of the reference amplicons with minimap2, as an alternative source
// of distances for ALIGN_MISMAPPING (params.align_backend = 'minimap2').
//
// Run directly rather than through the `mappy` Python bindings: mappy exposes -N but not
// -p, and the preset thresholds matter even more than either. The defaults
// (`-k 11 -w 5 -p 0 -N 1000 --secondary=yes -c`) are the settings that returned the most
// complete set of pairs in dev/alignment_backend_benchmark.md — no preset, small
// minimizers, no score-ratio filter, and -c so NM is present in the PAF.
//
// The target is MINIMAP2_INDEX's prebuilt .mmi, so a large reference DB is indexed once
// rather than on every invocation. -k/-w live in the index and are ignored here.
process MINIMAP2_ALLVSALL {
    tag "$meta.id"
    label 'process_medium'

    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container
        ? "https://depot.galaxyproject.org/singularity/minimap2:${params.minimap2_tag}"
        : "quay.io/biocontainers/minimap2:${params.minimap2_tag}"}"

    input:
    tuple val(meta), path(amplicons), path(index)

    output:
    tuple val(meta), path("${meta.id}.allvsall.paf"), emit: paf
    path "versions.yml",                              emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    minimap2 -t ${task.cpus} $args ${index} ${amplicons} > ${prefix}.allvsall.paf

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        minimap2: \$(minimap2 --version)
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    : > ${prefix}.allvsall.paf

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        minimap2: stub
    END_VERSIONS
    """
}
