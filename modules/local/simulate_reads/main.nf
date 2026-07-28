// Sample errored reads from each reference amplicon. Read names carry their source
// reference, so mapping these (MAPSEQ_SIM) and tallying where they land measures the
// mis-mapping matrix M directly. The error model is either the trained skiver model or
// the naive flat per-mutation-type one (params.sim_error_model); under 'flat' the model
// slot is the assets/NO_MODEL placeholder and no training runs at all.
process SIMULATE_READS {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicons), path(model_pt)

    output:
    tuple val(meta), path("${meta.id}.sim.fasta"), emit: reads
    path "versions.yml",                           emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args      = task.ext.args ?: ''
    def prefix    = task.ext.prefix ?: "${meta.id}"
    def model_arg = model_pt.name == 'NO_MODEL' ? '' : "--model-pt ${model_pt}"
    """
    export SKIVER_SCRIPTS=\${SKIVER_SCRIPTS:-/opt/skiver/scripts}

    simulate_amplicon_reads.py \\
        --amplicons ${amplicons} \\
        ${model_arg} \\
        --seed ${params.seed} \\
        -o ${prefix}.sim.fasta \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    printf '>ref|0|x:0\\nACGTACGTACGT\\n' > ${prefix}.sim.fasta

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
