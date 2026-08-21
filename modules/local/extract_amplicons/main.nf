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
    REF_ID="ref|0|\$(sha256sum ${references} | awk '{print \$1}' | cut -c1-12)"
    mkdir -p ${prefix}_amplicons
    cd ${prefix}_amplicons
    printf '>%s\\nACGTACGTACGT\\n' "\$REF_ID" > amplicons.fasta
    printf '#cutoff: 0.00:0.08\\n#name: refdb\\n#levels: Kingdom Genome Copy\\n%s\\tBacteria;ref;%s\\n' "\$REF_ID" "\$REF_ID" > amplicons.tax
    printf ',%s\\nref,1.0\\n' "\$REF_ID" > translation_table.csv
    printf 'refseq,genome,amplicon_len,amplifiable\\n%s,ref,12,True\\n' "\$REF_ID" > refseq_index.csv
    cd ..

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
