// Sample errored reads from each reference amplicon. Read names carry their source
// reference, so mapping these (MAPSEQ_PANEL_SIM) and tallying where they land measures the
// panel kernel directly. The error model is either the trained skiver model or
// the naive flat per-mutation-type one (params.sim_error_model); under 'flat' the model
// slot is the assets/NO_MODEL placeholder and no training runs at all.
process SIMULATE_READS {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicons), path(model_pt)
    // params.primer_mix, or [] when unset (see simulate_amplicon_reads.py --primer-mix)
    path primer_mix

    output:
    // One of the two, by params.sim_read_structure: reads (merged) or mate pairs (pairs).
    tuple val(meta), path("${meta.id}.sim.fasta"),           emit: reads, optional: true
    tuple val(meta), path("${meta.id}.sim_{1,2}.fastq.gz"),  emit: pairs, optional: true
    path "versions.yml",                           emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args      = task.ext.args ?: ''
    def prefix    = task.ext.prefix ?: "${meta.id}"
    def model_arg = model_pt.name == 'NO_MODEL' ? '' : "--model-pt ${model_pt}"
    def mix_arg   = primer_mix ? "--primer-mix ${primer_mix}" : ''
    def output    = params.sim_read_structure == 'pairs'
        ? "${prefix}.sim --mate-len ${params.sim_mate_len}" : "${prefix}.sim.fasta"
    """
    export SKIVER_SCRIPTS=\${SKIVER_SCRIPTS:-/opt/skiver/scripts}

    simulate_amplicon_reads.py \\
        --amplicons ${amplicons} \\
        ${model_arg} \\
        ${mix_arg} \\
        --seed ${params.seed} \\
        -o ${output} \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    if [ "${params.sim_read_structure}" = pairs ]; then
        for m in 1 2; do printf '@ref|0|x:0\\nACGT\\n+\\nIIII\\n' | gzip > ${prefix}.sim_\$m.fastq.gz; done
    else
        printf '>ref|0|x:0\\nACGTACGTACGT\\n' > ${prefix}.sim.fasta
    fi

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
