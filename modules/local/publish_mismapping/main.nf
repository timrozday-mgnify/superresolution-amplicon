process PUBLISH_MISMAPPING {
    tag "$meta.matrix_key"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicon_dir), path(matrix)

    output:
    tuple val(meta), path("${meta.matrix_key}"), emit: bundle
    path("${meta.matrix_key}.group.tsv"), emit: record
    path "versions.yml", emit: versions

    script:
    def provenance = groovy.json.JsonOutput.toJson(meta.provenance).bytes.encodeBase64().toString()
    def members = groovy.json.JsonOutput.toJson(meta.members).bytes.encodeBase64().toString()
    """
    write_mismapping_bundle.py \\
        --matrix ${matrix} \\
        --amplicon-dir ${amplicon_dir} \\
        --output-dir ${meta.matrix_key} \\
        --provenance-base64 '${provenance}' \\
        --members-base64 '${members}' \\
        --group-record ${meta.matrix_key}.group.tsv
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """
}
