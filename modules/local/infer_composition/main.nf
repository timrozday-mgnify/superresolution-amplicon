process INFER_COMPOSITION {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicon_dir), path(mismapping_matrix), path(obs_mseq), val(matrix_key)
    path taxonomy   // MAPseq .tax for the v4-group lca column; [] when not supplied

    output:
    tuple val(meta), path("${meta.id}.inferred_composition.csv"),    emit: composition
    tuple val(meta), path("${meta.id}.inferred_v4_groups.csv"), optional: true, emit: v4_groups
    tuple val(meta), path("${meta.id}.inference_diagnostics.csv"),   emit: diagnostics
    tuple val(meta), path("${meta.id}.posterior_draws.npz"),          emit: posterior
    tuple val(meta), path("${meta.id}.loss_trace.csv"), optional: true, emit: loss
    tuple val(meta), path("${meta.id}.ambiguity_pairs.csv"), optional: true, emit: ambiguity_pairs
    tuple val(meta), path("${meta.id}.ambiguity_sets.csv"),  optional: true, emit: ambiguity_sets
    tuple val(meta), path("${meta.id}.lca_composition.csv"), optional: true, emit: lca_composition
    path "versions.yml",                                             emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args   ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def taxonomy_arg = taxonomy ? "--taxonomy ${taxonomy}" : ''
    """
    infer_composition.py \\
        --amplicon-dir ${amplicon_dir} \\
        --mismapping-matrix ${mismapping_matrix} \\
        --obs-mseq ${obs_mseq} \\
        --mismapping-group-id ${matrix_key} \\
        --mismapping-matrix-path mismapping/${matrix_key}/mismapping_matrix.npz \\
        --sample-id ${prefix} \\
        --seed ${params.seed} \\
        ${taxonomy_arg} \\
        -o out \\
        $args

    cp out/inferred_composition.csv  ${prefix}.inferred_composition.csv
    cp out/inference_diagnostics.csv ${prefix}.inference_diagnostics.csv
    cp out/posterior_draws.npz       ${prefix}.posterior_draws.npz
    [ -f out/inferred_v4_groups.csv ] && cp out/inferred_v4_groups.csv ${prefix}.inferred_v4_groups.csv || true
    [ -f out/loss_trace.csv ] && cp out/loss_trace.csv ${prefix}.loss_trace.csv || true
    [ -f out/ambiguity_pairs.csv ] && cp out/ambiguity_pairs.csv ${prefix}.ambiguity_pairs.csv || true
    [ -f out/ambiguity_sets.csv ] && cp out/ambiguity_sets.csv ${prefix}.ambiguity_sets.csv || true
    [ -f out/lca_composition.csv ] && cp out/lca_composition.csv ${prefix}.lca_composition.csv || true

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    def v4_groups = params.infer_space == 'v4_group'
        ? "echo 'sample,v4_group_id,inferred_mean' > ${prefix}.inferred_v4_groups.csv" : ''
    """
    echo 'sample,genome_id,observed_rel_abundance,inferred_mean,inferred_lo,inferred_hi,presence_prob' > ${prefix}.inferred_composition.csv
    echo 'sample,mode,likelihood,n_reads,mismapping_group_id,mismapping_matrix_path' > ${prefix}.inference_diagnostics.csv
    touch ${prefix}.posterior_draws.npz
    ${v4_groups}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
