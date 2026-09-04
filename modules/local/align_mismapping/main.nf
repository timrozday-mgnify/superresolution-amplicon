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
    tuple val(meta), path(amplicons), path(paf)

    output:
    tuple val(meta), path("${meta.id}.mismapping_matrix.npz"), emit: mismapping
    path "versions.yml",                                       emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    build_mismapping_align.py \\
        --amplicons ${amplicons} \\
        --paf ${paf} \\
        -o ${prefix}.mismapping_matrix.npz \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version 2>&1 | sed 's/Python //')
    END_VERSIONS
    """

    stub:
    def prefix = task.ext.prefix ?: "${meta.id}"
    """
    # An empty file, not a real .npz: a stub run never reads the matrix (the
    # publish stub copies it, the inference stub writes a fixed CSV), and building
    # one needs numpy, which a bare CI runner has no reason to carry.
    touch ${prefix}.mismapping_matrix.npz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: stub
    END_VERSIONS
    """
}
