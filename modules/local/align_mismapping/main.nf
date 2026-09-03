// Build the mis-mapping matrix M by aligning the reference amplicons to each other,
// instead of simulating errored reads and mapping them (SIMULATE_READS + MAPSEQ_SIM +
// BUILD_MISMAPPING, which this replaces wholesale). A read from reference `a` is taken to
// be assigned uniformly over the references within `align_tau` edit operations of `a`.
//
// On the 21-genome B. uniformis V4 set this lands inside the simulate+map measurement's
// own seed-to-seed noise and is ~460x cheaper — see dev/alignment_mismapping.md. It also
// needs no mapper, no error model and therefore no skiver training at all.
process ALIGN_MISMAPPING {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicons)

    output:
    tuple val(meta), path("${meta.id}.mismapping_matrix.csv"), emit: mismapping
    path "versions.yml",                                       emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    build_mismapping_align.py \\
        --amplicons ${amplicons} \\
        -o ${prefix}.mismapping_matrix.csv \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
        edlib: \$(python -c 'from importlib.metadata import version; print(version("edlib"))')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    echo ',ref|0|x' > ${prefix}.mismapping_matrix.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
        edlib: stub
    END_VERSIONS
    """
}
