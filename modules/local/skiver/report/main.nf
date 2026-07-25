process SKIVER_REPORT {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(model_pt), path(aic_csv)

    output:
    tuple val(meta), path("${meta.id}.error_model_report.html"), emit: report
    path "versions.yml",                                         emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def prefix   = task.ext.prefix ?: "${meta.id}"
    def platform = meta.platform
    // The vendored skiver notebooks are named with underscores (context_error_model_hq_illumina.qmd)
    // while platform ids use hyphens (hq-illumina); the wrapper itself takes the hyphenated form as a param.
    def qmd = "context_error_model_${platform.replace('-', '_')}.qmd"
    """
    export SKIVER_NOTEBOOKS=\${SKIVER_NOTEBOOKS:-/opt/skiver/notebooks}

    # Render locally (the qmd's `execute-dir: file` resolves `results_dir` relative
    # to the qmd's own directory, so copy it in rather than pointing at /opt).
    cp \$SKIVER_NOTEBOOKS/${qmd} .
    cp \$SKIVER_NOTEBOOKS/_context_error_model_body.qmd .

    mkdir -p report_input
    cp ${aic_csv} report_input/context_model_aic.csv
    cp ${model_pt} report_input/

    quarto render ${qmd} \\
        -P results_dir:report_input \\
        -P platform:${platform} \\
        --output ${prefix}.error_model_report.html

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        quarto: \$(quarto --version)
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    touch ${prefix}.error_model_report.html

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        quarto: stub
    END_VERSIONS
    """
}
