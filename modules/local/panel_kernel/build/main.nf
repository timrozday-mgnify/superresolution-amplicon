// The bundle's provenance.json: what --panel_kernel checks a reused kernel against.
def provenanceCmd(meta) {
    def json = groovy.json.JsonOutput.toJson(meta.provenance).replace("'", "'\\''")
    "printf '%s\\n' '${json}' > ${meta.id}/provenance.json"
}

// Tally the rectangular panel kernel K (panel sources x database V4 groups) from the
// simulated reads' MAPseq labels, with each source's home label from its own mapping.
// The output directory is what INFER_COMPOSITION and CHECK_COMPOSITION_FIT take as
// --amplicon-dir; the kernel is named mismapping_matrix.npz like every other matrix.
process PANEL_KERNEL {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(prepared), path(db_amplicons), path(home_mseq), path(sim_mseq)

    output:
    tuple val(meta), path("${meta.id}"), emit: kernel
    path "versions.yml",                 emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    mkdir -p ${meta.id}
    cp ${prepared}/panel_translation.tsv ${prepared}/sources.tsv ${meta.id}/
    ${provenanceCmd(meta)}
    build_panel_kernel.py build \\
        --prepared ${prepared} \\
        --db-amplicons ${db_amplicons} \\
        --home-mseq ${home_mseq} \\
        --sim-mseq ${sim_mseq} \\
        -o ${meta.id}/mismapping_matrix.npz \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${meta.id}
    cp ${prepared}/panel_translation.tsv ${prepared}/sources.tsv ${meta.id}/
    ${provenanceCmd(meta)}
    touch ${meta.id}/mismapping_matrix.npz
    printf 'source\\thome_label\\n' > ${meta.id}/panel_sources.tsv
    # Which file the labels were grouped from: the extracted amplicons, never the MAPseq FASTA.
    printf '%s\\n' ${db_amplicons} > ${meta.id}/stub_db_amplicons.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}

// Build the rectangular panel kernel from sequence distance rather than simulated reads.
// MAPseq still establishes every source's home label before this process runs.
process PANEL_ALIGN {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(prepared), path(db_amplicons), path(home_mseq), path(decay_model)

    output:
    tuple val(meta), path("${meta.id}"), emit: kernel
    path "versions.yml",                 emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def model_arg = params.align_decay_model ? "--model-pt ${decay_model}" : ''
    def all_args = [model_arg, args].findAll { it }.join(' ')
    """
    mkdir -p ${meta.id}
    cp ${prepared}/panel_translation.tsv ${prepared}/sources.tsv ${meta.id}/
    ${provenanceCmd(meta)}
    build_panel_kernel.py align \\
        --prepared ${prepared} \\
        --db-amplicons ${db_amplicons} \\
        --home-mseq ${home_mseq} \\
        -o ${meta.id}/mismapping_matrix.npz \\
        $all_args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${meta.id}
    cp ${prepared}/panel_translation.tsv ${prepared}/sources.tsv ${meta.id}/
    ${provenanceCmd(meta)}
    touch ${meta.id}/mismapping_matrix.npz
    printf 'source\\thome_label\\n' > ${meta.id}/panel_sources.tsv
    # Which file the labels were grouped from: the extracted amplicons, never the MAPseq FASTA.
    printf '%s\\n' ${db_amplicons} > ${meta.id}/stub_db_amplicons.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
