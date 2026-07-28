process INFER_COMPOSITION {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicon_dir), path(sim_mseq), path(obs_mseq)

    output:
    tuple val(meta), path("${meta.id}.inferred_composition.csv"),    emit: composition
    tuple val(meta), path("${meta.id}.mismapping_matrix.csv"),       emit: mismapping
    tuple val(meta), path("${meta.id}.inference_diagnostics.csv"),   emit: diagnostics
    tuple val(meta), path("${meta.id}.loss_trace.csv"), optional: true, emit: loss
    path "versions.yml",                                             emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args   ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    infer_composition.py \\
        --amplicon-dir ${amplicon_dir} \\
        --sim-mseq ${sim_mseq} \\
        --obs-mseq ${obs_mseq} \\
        --sample-id ${prefix} \\
        --seed ${params.seed} \\
        -o out \\
        $args

    cp out/inferred_composition.csv  ${prefix}.inferred_composition.csv
    cp out/mismapping_matrix.csv     ${prefix}.mismapping_matrix.csv
    cp out/inference_diagnostics.csv ${prefix}.inference_diagnostics.csv
    [ -f out/loss_trace.csv ] && cp out/loss_trace.csv ${prefix}.loss_trace.csv || true

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    echo 'sample,genome_id,observed_rel_abundance,inferred_mean,inferred_lo,inferred_hi' > ${prefix}.inferred_composition.csv
    echo ',ref|0|x' > ${prefix}.mismapping_matrix.csv
    echo 'sample,mode,likelihood,n_reads' > ${prefix}.inference_diagnostics.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
