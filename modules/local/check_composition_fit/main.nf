process CHECK_COMPOSITION_FIT {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicon_dir), path(mismapping_matrix), path(obs_mseq),
          path(composition), path(posterior_draws)

    output:
    tuple val(meta), path("${meta.id}.fit_diagnostics.json"), emit: diagnostics
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    check_composition_fit.py \\
        --composition ${composition} \\
        --posterior-draws ${posterior_draws} \\
        --obs-mseq ${obs_mseq} \\
        --amplicon-dir ${amplicon_dir} \\
        --mismapping-matrix ${mismapping_matrix} \\
        --seed ${params.seed} \\
        -o ${prefix}.fit_diagnostics.json \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    echo '{"fit_status": "ok"}' > ${prefix}.fit_diagnostics.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
