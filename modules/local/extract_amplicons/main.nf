// In-silico PCR over the reference DB: the primer-trimmed amplicons (the database's V4
// groups, and the panel workflow's sources and alignment input), the genome -> reference
// translation table T, and references.tax, the MAPseq tax for mapping against the DB
// FASTA itself.
process EXTRACT_AMPLICONS {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(references)

    output:
    // The files, not their directory: storeDir (--amplicon_cache) moves each output with
    // `mv -f`, which replaces a file atomically but fails on a directory another run has
    // already moved there. So concurrent cold runs on one set each overwrite the same
    // (identical) files instead of all but one failing. The workflow takes their parent.
    tuple val(meta), path("${meta.id}_amplicons/amplicons.fasta"), path("${meta.id}_amplicons/references.tax"),
        path("${meta.id}_amplicons/translation_table.tsv"), path("${meta.id}_amplicons/refseq_index.csv"), emit: dir
    path "versions.yml",                           emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args   ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    subspecies_infer.py amplicons \\
        --db-fasta ${references} \\
        -o ${prefix}_amplicons \\
        --threads ${task.cpus} \\
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
    printf '#cutoff: 0.00:0.08\\n#name: refdb\\n#levels: Kingdom Genome Copy\\n%s\\tBacteria;ref;%s\\n' "\$REF_ID" "\$REF_ID" > references.tax
    printf 'genome_id\\trefseq\\tweight\\nref\\t%s\\t1.0\\n' "\$REF_ID" > translation_table.tsv
    printf 'refseq,genome,amplicon_len,amplifiable\\n%s,ref,12,True\\n' "\$REF_ID" > refseq_index.csv
    cd ..

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
