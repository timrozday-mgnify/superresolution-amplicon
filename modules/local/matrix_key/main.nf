process MATRIX_KEY {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    tuple val(meta), path(amplicon_dir), path(identity_file), val(identity)

    output:
    tuple val(meta), path(amplicon_dir), path(identity_file), val(identity), path("matrix_key.txt"), path("reference_sha256.txt"), emit: key
    path "versions.yml", emit: versions

    script:
    def settings = [
        mismapping_method: params.mismapping_method, align_backend: params.align_backend,
        align_tau: params.align_tau,
        align_ambiguity_weight: params.align_ambiguity_weight,
        max_ambiguous_bases: params.max_ambiguous_bases,
        max_postings: params.max_postings,
        minimap2_args: params.minimap2_args,
        minimap2_index_args: params.minimap2_index_args,
        minimap2_tag: params.minimap2_tag,
        sim_error_model: params.sim_error_model, sim_n_per_ref: params.sim_n_per_ref,
        sim_read_len: params.sim_read_len, flat_sub_rate: params.flat_sub_rate,
        flat_ins_rate: params.flat_ins_rate, flat_del_rate: params.flat_del_rate,
        mapseq_args: params.mapseq_args, mapseq_min_identity: params.mapseq_min_identity,
        mapseq_tag: params.mapseq_tag, seed: params.seed, identity: identity
    ].collect { k, v -> "${k}=${v}" }.sort().join('\n')
    """
    sha256sum ${amplicon_dir}/amplicons.fasta | awk '{print \$1}' > reference_sha256.txt
    { cat reference_sha256.txt; printf '%s\\n' '${settings}'; sha256sum ${identity_file} | awk '{print \$1}'; } | sha256sum | awk '{print \$1}' > matrix_key.txt
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        sha256sum: system
    END_VERSIONS
    """
}
