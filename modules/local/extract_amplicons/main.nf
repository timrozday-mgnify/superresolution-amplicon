// In-silico PCR over the reference DB: the primer-trimmed amplicons become the mapseq
// reference set (fasta + the .tax sidecar mapseq requires), alongside the genome ->
// reference translation table T. No error model involved — mis-mapping is measured later
// by simulating reads and mapping them (SIMULATE_READS -> MAPSEQ_SIM).
process EXTRACT_AMPLICONS {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(references)

    output:
    tuple val(meta), path("${meta.id}_amplicons"), emit: dir
    tuple val(meta), path("${meta.id}_amplicons/amplicons.fasta"),
                     path("${meta.id}_amplicons/amplicons.tax"),         emit: refs
    tuple val(meta), path("${meta.id}_amplicons/translation_table.csv"), emit: translation
    path "versions.yml",                                                 emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args   ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    subspecies_infer.py amplicons \\
        --db-fasta ${references} \\
        -o ${prefix}_amplicons \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    mkdir -p ${prefix}_amplicons
    cd ${prefix}_amplicons
    printf '>ref|0|x\\nACGTACGTACGT\\n' > amplicons.fasta
    printf '#cutoff: 0.00:0.08\\n#name: refdb\\n#levels: Kingdom Genome Copy\\nref|0|x\\tBacteria;ref;ref|0|x\\n' > amplicons.tax
    printf ',ref|0|x\\nref,1.0\\n' > translation_table.csv
    printf 'refseq,genome,amplicon_len,amplifiable\\nref|0|x,ref,12,True\\n' > refseq_index.csv
    cd ..

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
