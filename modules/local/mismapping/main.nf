process MISMAPPING {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(references), path(model_pt)

    output:
    tuple val(meta), path("${meta.id}_mismapping"), emit: dir
    tuple val(meta), path("${meta.id}_mismapping/mismapping_matrix.csv"),   emit: matrix
    tuple val(meta), path("${meta.id}_mismapping/translation_table.csv"),   emit: translation
    path "versions.yml",                                                    emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args   ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    // --score-components triggers the sampling path (edlib argmax-assign) so we get the
    // simulated matrix + comparison alongside the analytic one, and the D tensor the
    // score-hist inference consumes.
    """
    export SKIVER_SCRIPTS=\${SKIVER_SCRIPTS:-/opt/skiver/scripts}

    subspecies_infer.py mismapping \\
        --db-fasta ${references} \\
        --model-pt ${model_pt} \\
        --method likelihood \\
        --score-components \\
        --seed ${params.seed} \\
        -o ${prefix}_mismapping \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    mkdir -p ${prefix}_mismapping
    cd ${prefix}_mismapping
    echo ',ref' > mismapping_matrix.csv
    echo ',ref' > translation_table.csv
    echo 'refseq,genome,v4_len,amplifiable' > refseq_index.csv
    touch mismapping_matrix_sim.csv mismapping_compare.csv v4_amplicons.fasta
    python -c "import numpy as np; np.savez_compressed('score_components.npz', D=np.zeros((1,1,1)), bin_edges=np.zeros(2), refseqs=np.array(['ref'],dtype=object))" 2>/dev/null || touch score_components.npz
    cd ..

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
