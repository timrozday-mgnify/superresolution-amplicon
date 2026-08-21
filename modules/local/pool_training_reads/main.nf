process POOL_TRAINING_READS {
    tag "$platform"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(platform), path(reads, stageAs: 'inputs??/*')

    output:
    tuple val([id: "pooled_${platform}", platform: platform]), path("pooled_${platform}.fastq.gz"), emit: reads
    path "versions.yml", emit: versions

    script:
    """
    pool_training_reads.py --input ${reads} \\
        --max-reads ${params.trained_error_model_max_reads} \\
        --seed ${params.seed} \\
        --output pooled_${platform}.fastq.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """
}
