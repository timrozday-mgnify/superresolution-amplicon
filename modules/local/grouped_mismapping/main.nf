// Build M by collapsing exact-duplicate amplicons, optionally widened to tau >= 1 by a
// pigeonhole block filter and a bounded IUPAC edit distance. Writes the grouped matrix,
// which stores one entry per *distinct* amplicon pair instead of per reference pair.
process GROUPED_MISMAPPING {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicons)

    output:
    tuple val(meta), path("${meta.id}.mismapping_matrix.npz"), emit: mismapping
    path "versions.yml",                                       emit: versions

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    build_mismapping_align.py \\
        --backend ${params.align_backend} \\
        --amplicons ${amplicons} \\
        --tau ${params.align_tau} \\
        --distance-decay ${params.align_distance_decay} \\
        --max-ambiguous-bases ${params.max_ambiguous_bases} \\
        --max-postings ${params.max_postings} \\
        --ambiguity-weight ${params.align_ambiguity_weight} \\
        -o ${prefix}.mismapping_matrix.npz \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    # An empty file, not a real .npz: a stub run never reads the matrix (the
    # publish stub copies it, the inference stub writes a fixed CSV), and building
    # one needs numpy, which a bare CI runner has no reason to carry.
    touch ${prefix}.mismapping_matrix.npz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
