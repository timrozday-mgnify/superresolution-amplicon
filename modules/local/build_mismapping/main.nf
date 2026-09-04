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
