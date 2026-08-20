//
// superresolution-amplicon: extract the reference amplicons, measure reference-to-
// reference mis-mapping by simulating reads and mapping them with the same mapper the
// real reads go through (mapseq), and infer the true genome composition.
//
include { TRAIN_ERROR_MODEL    } from '../subworkflows/local/train_error_model/main'
include { EXTRACT_AMPLICONS    } from '../modules/local/extract_amplicons/main'
include { MAPSEQ_CLUSTER       } from '../modules/local/mapseq/cluster/main'
include { MAPSEQ as MAPSEQ_SIM } from '../modules/local/mapseq/map/main'
include { MAPSEQ as MAPSEQ_OBS } from '../modules/local/mapseq/map/main'
include { SIMULATE_READS       } from '../modules/local/simulate_reads/main'
include { BUILD_MISMAPPING     } from '../modules/local/build_mismapping/main'
include { READS_TO_FASTA       } from '../modules/local/reads_to_fasta/main'
include { INFER_COMPOSITION    } from '../modules/local/infer_composition/main'

workflow SUPERRESOLUTION_AMPLICON {
    take:
    ch_reads      // [ meta, [ reads ] ]                (meta.id, meta.platform)
    ch_refs       // [ meta, references_fasta ]
    ch_pretrained // [ id, model_pt ]  (samples that supply their own error model)

    main:
    ch_versions = Channel.empty()

    // Error model: only the read simulator uses it, so the whole skiver training
    // subworkflow is skipped under the flat model. [ id, model_pt ] either way.
    if (params.mismapping_matrix || params.sim_error_model == 'flat') {
        ch_model = ch_reads.map { meta, reads -> [ meta.id, file("${projectDir}/assets/NO_MODEL") ] }
    }
    else {
        // Split samples that bring a pre-trained model from those that need training.
        ch_reads
            .branch { meta, reads ->
                pretrained: meta.error_model != null
                train:      true
            }
            .set { ch_split }

        TRAIN_ERROR_MODEL(ch_split.train)
        ch_versions = ch_versions.mix(TRAIN_ERROR_MODEL.out.versions)

        ch_model = TRAIN_ERROR_MODEL.out.model
            .map { meta, model -> [ meta.id, model ] }
            .mix(ch_pretrained)
    }

    // In-silico PCR -> the mapseq reference set (+ translation table T).
    EXTRACT_AMPLICONS(ch_refs)
    ch_versions = ch_versions.mix(EXTRACT_AMPLICONS.out.versions)

    // Build the mapseq clustering once; both mappings reuse it.
    MAPSEQ_CLUSTER(EXTRACT_AMPLICONS.out.refs)
    ch_versions = ch_versions.mix(MAPSEQ_CLUSTER.out.versions)

    // [ id, fasta, tax, mscluster ] — the mapseq DB slots, shared by both mappings.
    ch_db = EXTRACT_AMPLICONS.out.refs
        .map { meta, fasta, tax -> [ meta.id, fasta, tax ] }
        .join(MAPSEQ_CLUSTER.out.mscluster.map { meta, mscluster -> [ meta.id, mscluster ] })

    // A supplied matrix skips the simulation and mapping work. Otherwise build and
    // materialise it before inference so its CSV can be reused in a later run.
    if (params.mismapping_matrix) {
        ch_mismapping = EXTRACT_AMPLICONS.out.dir
            .map { meta, d -> [ meta, file(params.mismapping_matrix, checkIfExists: true) ] }
    }
    else {
        ch_sim_in = EXTRACT_AMPLICONS.out.refs
            .map { meta, fasta, tax -> [ meta.id, meta, fasta ] }
            .join(ch_model)
            .map { id, meta, fasta, model -> [ meta, fasta, model ] }
        SIMULATE_READS(ch_sim_in)
        ch_versions = ch_versions.mix(SIMULATE_READS.out.versions)

        MAPSEQ_SIM(SIMULATE_READS.out.reads
            .map { meta, reads -> [ meta.id, meta, reads ] }
            .join(ch_db)
            .map { id, meta, reads, fasta, tax, mscluster -> [ meta, reads, fasta, tax, mscluster ] })
        ch_versions = ch_versions.mix(MAPSEQ_SIM.out.versions)

        BUILD_MISMAPPING(EXTRACT_AMPLICONS.out.dir
            .map { meta, d -> [ meta.id, meta, d ] }
            .join(MAPSEQ_SIM.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
            .map { id, meta, d, mseq -> [ meta, d, mseq ] })
        ch_versions = ch_versions.mix(BUILD_MISMAPPING.out.versions)
        ch_mismapping = BUILD_MISMAPPING.out.mismapping
    }

    // Real reads -> fasta -> mapseq -> the observed per-reference counts.
    READS_TO_FASTA(ch_reads)
    ch_versions = ch_versions.mix(READS_TO_FASTA.out.versions)

    MAPSEQ_OBS(READS_TO_FASTA.out.reads
        .map { meta, reads -> [ meta.id, meta, reads ] }
        .join(ch_db)
        .map { id, meta, reads, fasta, tax, mscluster -> [ meta, reads, fasta, tax, mscluster ] })
    ch_versions = ch_versions.mix(MAPSEQ_OBS.out.versions)

    // INFER_COMPOSITION: amplicon dir + matrix + observed mseq, joined by id.
    ch_infer_in = EXTRACT_AMPLICONS.out.dir
        .map { meta, d -> [ meta.id, meta, d ] }
        .join(ch_mismapping.map { meta, matrix -> [ meta.id, matrix ] })
        .join(MAPSEQ_OBS.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
        .map { id, meta, d, matrix, obs -> [ meta, d, matrix, obs ] }
    INFER_COMPOSITION(ch_infer_in)
    ch_versions = ch_versions.mix(INFER_COMPOSITION.out.versions)

    // Collate the per-process versions into one file.
    ch_versions
        .collectFile(name: 'software_versions.yml', storeDir: "${params.outdir}/pipeline_info")

    emit:
    amplicons   = EXTRACT_AMPLICONS.out.dir
    mismapping  = INFER_COMPOSITION.out.mismapping
    composition = INFER_COMPOSITION.out.composition
    versions    = ch_versions
}
