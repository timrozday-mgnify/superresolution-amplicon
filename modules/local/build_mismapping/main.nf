process BUILD_MISMAPPING {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicon_dir), path(sim_mseq)

    output:
    tuple val(meta), path("${meta.id}.mismapping_matrix.npz"), emit: mismapping
    path "versions.yml",                                  emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    infer_composition.py \\
        --amplicon-dir ${amplicon_dir} \\
        --sim-mseq ${sim_mseq} \\
        --build-mismapping \\
        -o out \\
        $args

    cp out/mismapping_matrix.npz ${prefix}.mismapping_matrix.npz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    python -c "import numpy as np; np.savez_compressed('${prefix}.mismapping_matrix.npz', data=[1.0], indices=[0], indptr=[0, 1], shape=[1, 1], refseqs=['ref|0|x'])"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
