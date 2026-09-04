//
// superresolution-amplicon: extract the reference amplicons, establish reference-to-
// reference mis-mapping — by simulating reads and mapping them with the same mapper the
// real reads go through (mapseq), or by aligning the reference amplicons to each other
// (params.mismapping_method) — and infer the true genome composition.
//
include { TRAIN_ERROR_MODEL    } from '../subworkflows/local/train_error_model/main'
include { EXTRACT_AMPLICONS    } from '../modules/local/extract_amplicons/main'
include { MAPSEQ_CLUSTER       } from '../modules/local/mapseq/cluster/main'
include { MAPSEQ_CLUSTER as MAPSEQ_CLUSTER_MATRIX } from '../modules/local/mapseq/cluster/main'
include { MAPSEQ as MAPSEQ_SIM } from '../modules/local/mapseq/map/main'
include { MAPSEQ as MAPSEQ_OBS } from '../modules/local/mapseq/map/main'
include { SIMULATE_READS       } from '../modules/local/simulate_reads/main'
include { BUILD_MISMAPPING     } from '../modules/local/build_mismapping/main'
include { ALIGN_MISMAPPING     } from '../modules/local/align_mismapping/main'
include { GROUPED_MISMAPPING   } from '../modules/local/grouped_mismapping/main'
include { MINIMAP2_INDEX       } from '../modules/local/minimap2/index/main'
include { MINIMAP2_ALLVSALL    } from '../modules/local/minimap2/allvsall/main'
include { MATRIX_KEY            } from '../modules/local/matrix_key/main'
include { PUBLISH_MISMAPPING    } from '../modules/local/publish_mismapping/main'
include { POOL_TRAINING_READS   } from '../modules/local/pool_training_reads/main'
include { READS_TO_FASTA       } from '../modules/local/reads_to_fasta/main'
include { INFER_COMPOSITION    } from '../modules/local/infer_composition/main'

workflow SUPERRESOLUTION_AMPLICON {
    take:
    ch_reads      // [ meta, [ reads ] ]                (meta.id, meta.platform)
    ch_refs       // [ meta, references_fasta ]
    ch_pretrained // retained for the public workflow signature

    main:
    ch_versions = Channel.empty()

    // Error model: only the read simulator uses it, so the whole skiver training
    // subworkflow is skipped under the flat model. [ id, model_pt ] either way.
    if (!(params.mismapping_method in ['simulate', 'align'])) {
        error "--mismapping_method must be 'simulate' or 'align'"
    }
    if (!(params.align_backend in ['minimap2', 'exact-hash', 'kmer'])) {
        error "--align_backend must be 'minimap2', 'exact-hash' or 'kmer'"
    }
    // `as int`: a --align_tau on the command line arrives as a String, and comparing that
    // to a number silently misjudges every backend check below.
    if (params.align_backend == 'kmer' && (params.align_tau as int) < 1) {
        error "--align_backend kmer requires --align_tau >= 1; use exact-hash for tau=0"
    }
    if (params.align_backend == 'exact-hash' && (params.align_tau as int) != 0) {
        error "--align_backend exact-hash requires --align_tau 0; use kmer for tau >= 1"
    }
    if (params.minimap2_args =~ /(^|\s)-[kw]\b/) {
        // Silently ignored: with a prebuilt .mmi target, minimap2 takes -k/-w from the
        // index. Putting them here would look like they applied when they did not.
        error "-k/-w belong in --minimap2_index_args, not --minimap2_args"
    }
    if (params.mismapping_method == 'align' && params.sim_read_len) {
        // Whole-reference distances cannot see what a short read cannot see, and quietly
        // understating confusion is worse than refusing.
        error "--mismapping_method align builds M from whole-reference alignments and " +
              "cannot model --sim_read_len windows. Use --mismapping_method simulate for " +
              "reads shorter than the amplicon, or merge pairs so queries span it."
    }

    if (params.mismapping_matrix) {
        ch_model = ch_reads.map { meta, reads -> [ meta.id, file(params.mismapping_matrix, checkIfExists: true), 'supplied' ] }
    }
    else if (params.mismapping_method == 'align') {
        // M comes from reference-to-reference alignment: no reads are simulated, so no
        // error model is needed and the skiver training subworkflow never runs.
        ch_model = ch_reads.map { meta, reads -> [ meta.id, file("${projectDir}/assets/NO_MODEL"), 'align' ] }
    }
    else if (params.sim_error_model == 'flat') {
        ch_model = ch_reads.map { meta, reads -> [ meta.id, file("${projectDir}/assets/NO_MODEL"), 'flat' ] }
    }
    else {
        if (!(params.trained_error_model_scope in ['per-sample', 'pooled'])) {
            error "--trained_error_model_scope must be 'per-sample' or 'pooled'"
        }
        ch_reads
            .branch { meta, reads ->
                pretrained: meta.error_model != null
                train:      true
            }
            .set { ch_split }

        ch_supplied_model = ch_split.pretrained.map { meta, reads -> [ meta.id, meta.error_model, 'supplied' ] }
        if (params.trained_error_model_scope == 'pooled') {
            ch_pool_input = ch_split.train
                .flatMap { meta, reads -> reads.collect { read -> [ meta.platform, read ] } }
                .groupTuple()
            POOL_TRAINING_READS(ch_pool_input)
            ch_pooled_reads = POOL_TRAINING_READS.out.reads
                .map { platform, reads -> [ [id: "pooled_${platform}", platform: platform], [reads] ] }
            TRAIN_ERROR_MODEL(ch_pooled_reads)
            ch_versions = ch_versions.mix(POOL_TRAINING_READS.out.versions).mix(TRAIN_ERROR_MODEL.out.versions)
            ch_pooled_model = ch_split.train
                .map { meta, reads -> [ meta.platform, meta.id ] }
                .combine(TRAIN_ERROR_MODEL.out.model.map { meta, model -> [ meta.platform, model ] }, by: 0)
                .map { platform, id, model -> [ id, model, "pooled:${platform}" ] }
            ch_model = ch_pooled_model.mix(ch_supplied_model)
        }
        else {
            TRAIN_ERROR_MODEL(ch_split.train)
            ch_versions = ch_versions.mix(TRAIN_ERROR_MODEL.out.versions)
            ch_model = TRAIN_ERROR_MODEL.out.model
                .map { meta, model -> [ meta.id, model, "per-sample:${meta.id}" ] }
                .mix(ch_supplied_model)
        }
    }

    // In-silico PCR -> the mapseq reference set (+ translation table T).
    EXTRACT_AMPLICONS(ch_refs)
    ch_versions = ch_versions.mix(EXTRACT_AMPLICONS.out.versions)

    // Build the per-sample mapseq clustering for observed-read mapping.
    MAPSEQ_CLUSTER(EXTRACT_AMPLICONS.out.refs)
    ch_versions = ch_versions.mix(MAPSEQ_CLUSTER.out.versions)

    // [ id, fasta, tax, mscluster ] — the mapseq DB slots, shared by both mappings.
    ch_db = EXTRACT_AMPLICONS.out.refs
        .map { meta, fasta, tax -> [ meta.id, fasta, tax ] }
        .join(MAPSEQ_CLUSTER.out.mscluster.map { meta, mscluster -> [ meta.id, mscluster ] })

    // Fingerprint extracted amplicons and group all samples that experience the same
    // simulation + mapper configuration. The group metadata is also bundle provenance.
    MATRIX_KEY(EXTRACT_AMPLICONS.out.dir
        .map { meta, d -> [ meta.id, meta, d ] }
        .join(ch_model)
        .map { id, meta, d, model, identity -> [ meta, d, model, identity ] })
    ch_matrix_groups = MATRIX_KEY.out.key
        .map { meta, d, identity_file, identity, key_file, ref_file ->
            [ key_file.text.trim(), [meta, d, identity_file, identity, ref_file.text.trim()] ]
        }
        .groupTuple()
        .map { key, entries ->
            def representative = entries[0]
            def members = entries.collect { [id: it[0].id, platform: it[0].platform] }
            def scope = representative[3].tokenize(':')[0]
            def source = params.mismapping_matrix ? 'supplied' : 'generated'
            def provenance = [
                matrix_key: key, reference_sha256: representative[4], model_scope: scope,
                source: source, mismapping_method: params.mismapping_method,
                align_backend: params.align_backend, align_tau: params.align_tau,
                align_ambiguity_weight: params.align_ambiguity_weight,
                max_ambiguous_bases: params.max_ambiguous_bases,
                max_postings: params.max_postings,
                minimap2_args: params.minimap2_args,
                minimap2_index_args: params.minimap2_index_args,
                sim_error_model: params.sim_error_model,
                sim_n_per_ref: params.sim_n_per_ref, sim_read_len: params.sim_read_len,
                flat_sub_rate: params.flat_sub_rate, flat_ins_rate: params.flat_ins_rate,
                flat_del_rate: params.flat_del_rate, mapseq_args: params.mapseq_args,
                mapseq_min_identity: params.mapseq_min_identity, mapseq_tag: params.mapseq_tag,
                seed: params.seed, samples: members
            ]
            [[id: "matrix_${key}", matrix_key: key, reference_sha256: representative[4],
              model_scope: scope, source: source, members: members, provenance: provenance],
             representative[1], representative[2]]
        }

    if (params.mismapping_matrix) {
        ch_bundle_in = ch_matrix_groups.map { meta, d, supplied_matrix -> [ meta, d, supplied_matrix ] }
    }
    else if (params.mismapping_method == 'align') {
        // One alignment of the reference amplicons against themselves replaces the whole
        // simulate -> cluster -> map -> tally chain.
        ch_align_refs = ch_matrix_groups.map { meta, d, model -> [ meta, d.resolve('amplicons.fasta') ] }
        // Index once, then align against it — the reference set is the target of its own
        // all-vs-all, so without this every run re-indexes the whole DB.
        if (params.align_backend in ['kmer', 'exact-hash']) {
            GROUPED_MISMAPPING(ch_align_refs)
            ch_versions = ch_versions.mix(GROUPED_MISMAPPING.out.versions)
            ch_bundle_in = GROUPED_MISMAPPING.out.mismapping
                .map { meta, matrix -> [ meta.id, meta, matrix ] }
                .join(ch_matrix_groups.map { meta, d, model -> [ meta.id, d ] })
                .map { id, meta, matrix, d -> [ meta, d, matrix ] }
        }
        else {
        MINIMAP2_INDEX(ch_align_refs)
        MINIMAP2_ALLVSALL(ch_align_refs
            .map { meta, fasta -> [ meta.id, meta, fasta ] }
            .join(MINIMAP2_INDEX.out.index.map { meta, mmi -> [ meta.id, mmi ] })
            .map { id, meta, fasta, mmi -> [ meta, fasta, mmi ] })
        ch_versions = ch_versions.mix(MINIMAP2_INDEX.out.versions)
                                 .mix(MINIMAP2_ALLVSALL.out.versions)
        ch_align_in = ch_align_refs
            .map { meta, fasta -> [ meta.id, meta, fasta ] }
            .join(MINIMAP2_ALLVSALL.out.paf.map { meta, paf -> [ meta.id, paf ] })
            .map { id, meta, fasta, paf -> [ meta, fasta, paf ] }
        ALIGN_MISMAPPING(ch_align_in)
        ch_versions = ch_versions.mix(ALIGN_MISMAPPING.out.versions)
        ch_bundle_in = ALIGN_MISMAPPING.out.mismapping
            .map { meta, matrix -> [ meta.id, meta, matrix ] }
            .join(ch_matrix_groups.map { meta, d, model -> [ meta.id, d ] })
            .map { id, meta, matrix, d -> [ meta, d, matrix ] }
        }
    }
    else {
        // The representative's amplicon directory is sufficient for the common matrix.
        ch_group_refs = ch_matrix_groups.map { meta, d, model ->
            [ meta, d, d.resolve('amplicons.fasta'), d.resolve('amplicons.tax'), model ]
        }
        MAPSEQ_CLUSTER_MATRIX(ch_group_refs.map { meta, d, fasta, tax, model -> [ meta, fasta, tax ] })
        SIMULATE_READS(ch_group_refs.map { meta, d, fasta, tax, model -> [ meta, fasta, model ] })
        ch_versions = ch_versions.mix(MAPSEQ_CLUSTER_MATRIX.out.versions).mix(SIMULATE_READS.out.versions)
        MAPSEQ_SIM(SIMULATE_READS.out.reads
            .map { meta, reads -> [ meta.id, meta, reads ] }
            .join(ch_group_refs.map { meta, d, fasta, tax, model -> [ meta.id, d, fasta, tax ] })
            .join(MAPSEQ_CLUSTER_MATRIX.out.mscluster.map { meta, cluster -> [ meta.id, cluster ] })
            .map { id, meta, reads, d, fasta, tax, cluster -> [ meta, reads, fasta, tax, cluster ] })
        ch_versions = ch_versions.mix(MAPSEQ_SIM.out.versions)
        BUILD_MISMAPPING(ch_group_refs
            .map { meta, d, fasta, tax, model -> [ meta.id, meta, d ] }
            .join(MAPSEQ_SIM.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
            .map { id, meta, d, mseq -> [ meta, d, mseq ] })
        ch_versions = ch_versions.mix(BUILD_MISMAPPING.out.versions)
        ch_bundle_in = BUILD_MISMAPPING.out.mismapping
            .map { meta, matrix -> [ meta.id, meta, matrix ] }
            .join(ch_matrix_groups.map { meta, d, model -> [ meta.id, d ] })
            .map { id, meta, matrix, d -> [ meta, d, matrix ] }
    }
    PUBLISH_MISMAPPING(ch_bundle_in)
    ch_versions = ch_versions.mix(PUBLISH_MISMAPPING.out.versions)
    PUBLISH_MISMAPPING.out.record.collectFile(
        name: 'groups.tsv', storeDir: "${params.outdir}/mismapping", keepHeader: true, skip: 1
    )
    ch_mismapping = PUBLISH_MISMAPPING.out.bundle
        .flatMap { meta, bundle -> meta.members.collect { member ->
            [ member.id, bundle.resolve('mismapping_matrix.npz'), meta.matrix_key ]
        } }

    // Real reads -> fasta -> mapseq -> the observed per-reference counts.
    READS_TO_FASTA(ch_reads)
    ch_versions = ch_versions.mix(READS_TO_FASTA.out.versions)

    MAPSEQ_OBS(READS_TO_FASTA.out.reads
        .map { meta, reads -> [ meta.id, meta, reads ] }
        .join(ch_db)
        .map { id, meta, reads, fasta, tax, mscluster -> [ meta, reads, fasta, tax, mscluster ] })
    ch_versions = ch_versions.mix(MAPSEQ_OBS.out.versions)

    // INFER_COMPOSITION: amplicon dir + canonical matrix + observed mseq, joined by id.
    ch_infer_in = EXTRACT_AMPLICONS.out.dir
        .map { meta, d -> [ meta.id, meta, d ] }
        .join(ch_mismapping)
        .join(MAPSEQ_OBS.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
        .map { id, meta, d, matrix, matrix_key, obs -> [ meta, d, matrix, obs, matrix_key ] }
    INFER_COMPOSITION(ch_infer_in)
    ch_versions = ch_versions.mix(INFER_COMPOSITION.out.versions)

    // Collate the per-process versions into one file.
    ch_versions
        .collectFile(name: 'software_versions.yml', storeDir: "${params.outdir}/pipeline_info")

    emit:
    amplicons   = EXTRACT_AMPLICONS.out.dir
    mismapping  = PUBLISH_MISMAPPING.out.bundle
    composition = INFER_COMPOSITION.out.composition
    versions    = ch_versions
}
