//
// superresolution-amplicon: extract the reference amplicons, measure reference-to-
// reference mis-mapping by simulating reads and mapping them with the same mapper the
// real reads go through (mapseq), and infer the true genome composition.
//
include { TRAIN_ERROR_MODEL    } from '../subworkflows/local/train_error_model/main'
include { EXTRACT_AMPLICONS    } from '../modules/local/extract_amplicons/main'
include { MAPSEQ_CLUSTER       } from '../modules/local/mapseq/cluster/main'
include { MAPSEQ_CLUSTER as MAPSEQ_CLUSTER_MATRIX } from '../modules/local/mapseq/cluster/main'
include { MAPSEQ as MAPSEQ_SIM } from '../modules/local/mapseq/map/main'
include { MAPSEQ as MAPSEQ_OBS } from '../modules/local/mapseq/map/main'
include { SIMULATE_READS       } from '../modules/local/simulate_reads/main'
include { BUILD_MISMAPPING     } from '../modules/local/build_mismapping/main'
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
    if (params.mismapping_matrix) {
        ch_model = ch_reads.map { meta, reads -> [ meta.id, file(params.mismapping_matrix, checkIfExists: true), 'supplied' ] }
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
            TRAIN_ERROR_MODEL(POOL_TRAINING_READS.out.reads.map { meta, reads -> [ meta, [ reads ] ] })
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
                source: source, sim_error_model: params.sim_error_model,
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
            [ member.id, bundle.resolve('mismapping_matrix.csv'), meta.matrix_key ]
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
