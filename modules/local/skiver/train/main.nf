process SKIVER_TRAIN {
    tag "$meta.id"
    label 'process_high'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(base_tsv)

    output:
    tuple val(meta), path("${meta.id}.model.pt"),               emit: model
    tuple val(meta), path("${meta.id}.phred_calibration.json"), emit: calibration
    tuple val(meta), path("${meta.id}.context_model_aic.csv"),  emit: aic
    path "versions.yml",                                        emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args     = task.ext.args ?: ''
    def prefix   = task.ext.prefix ?: "${meta.id}"
    def platform = meta.platform
    // Candidate component strings: a single override wins; otherwise the
    // comma-separated candidate list is fitted and the min-AIC model kept.
    // (HEAD skiver is model-config driven; this is AIC selection over a fixed set.)
    def components = params.error_model_components ?: params.error_model_candidates
    """
    export SKIVER_SCRIPTS=\${SKIVER_SCRIPTS:-/opt/skiver/scripts}

    # skiver train discovers <data-root>/<platform>/*.base_observations.tsv
    mkdir -p data/${platform}
    cp ${base_tsv} data/${platform}/${prefix}.base_observations.tsv

    # Build a model-config JSON from the candidate component strings.
    build_model_config.py "${components}" > model_config.json

    python \$SKIVER_SCRIPTS/train_context_error_models.py \\
        --model-config model_config.json \\
        --data-root data \\
        --platform ${platform} \\
        --no-split \\
        --output-dir out \\
        --seed ${params.seed} \\
        $args

    # Pick the min-AIC (MLE) model and expose it under a stable name.
    WINNER=\$(pick_best_model.py out/context_model_aic.csv ${platform})
    cp out/\${WINNER}_${platform}.pt ${prefix}.model.pt
    cp out/context_model_aic.csv ${prefix}.context_model_aic.csv

    # Fit P(Q | error_type) calibration from the same dump TSV.
    python \$SKIVER_SCRIPTS/fit_phred_calibration.py \\
        --base-tsv ${base_tsv} \\
        --output ${prefix}.phred_calibration.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        skiver: \$(skiver --version 2>&1 | sed 's/skiver //')
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.model.pt
    echo '{}' > ${prefix}.phred_calibration.json
    echo 'model_id,inference,aic' > ${prefix}.context_model_aic.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        skiver: stub
        python: stub
    END_VERSIONS
    """
}
