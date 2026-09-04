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
    python -c "import numpy as np; np.savez_compressed('${prefix}.mismapping_matrix.npz', format='grouped', data=[1.0], indices=[0], indptr=[0, 1], shape=[1, 1], group=[0], refseqs=np.frombuffer(b'ref|0|x', dtype=np.uint8))"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
