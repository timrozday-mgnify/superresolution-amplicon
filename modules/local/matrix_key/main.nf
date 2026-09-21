process MATRIX_KEY {
    tag "$meta.id"
    label 'process_medium'

    container "ghcr.io/timrozday-mgnify/sra-skiver:${params.sra_skiver_tag}"

    input:
    // mapseq_db: the MAPseq database's identity (the workflow's ch_sets). Every label in the
    // kernel comes from mapping against it, so it is part of the key.
    tuple val(meta), path(amplicon_dir), path(identity_file), val(identity), val(mapseq_db)

    output:
    tuple val(meta), path(amplicon_dir), path(identity_file), val(identity), path("matrix_key.txt"), path("reference_sha256.txt"), val(mapseq_db), emit: key
    path "versions.yml", emit: versions

    script:
    def settings = [
        // Must equal sparse_matrix.KERNEL_VERSION (tests/test_measured_mismapping.py checks).
        kernel_version: 2,
        panel: meta.panel ?: 'database', panel_taxa: meta.panel_taxa ?: '',
        panel_taxon_max_sources: params.panel_taxon_max_sources,
        mismapping_method: params.mismapping_method,
        align_tau: params.align_tau,
        align_distance_decay: params.align_distance_decay,
        align_ambiguity_weight: params.align_ambiguity_weight,
        max_ambiguous_bases: params.max_ambiguous_bases, max_postings: params.max_postings,
        sim_error_model: params.sim_error_model, sim_n_per_ref: params.sim_n_per_ref,
        sim_read_len: params.sim_read_len, flat_sub_rate: params.flat_sub_rate,
        flat_ins_rate: params.flat_ins_rate, flat_del_rate: params.flat_del_rate,
        fwd_primer: params.fwd_primer, rev_primer: params.rev_primer,
        primer_mismatches: params.primer_mismatches, trim_primers: params.trim_primers,
        // ponytail: keyed by path, like the reference set; hash it if mixes get edited in place.
        primer_mix: params.primer_mix ?: '',
        sim_read_structure: params.sim_read_structure, sim_mate_len: params.sim_mate_len,
        mapseq_args: params.mapseq_args, mapseq_min_identity: params.mapseq_min_identity,
        mapseq_tag: params.mapseq_tag, mapseq_db: mapseq_db, seed: params.seed, identity: identity
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
