//
// superresolution-amplicon: train an error model from amplicon reads, quantify
// reference-to-reference mis-mapping, and infer true genome composition.
//
include { TRAIN_ERROR_MODEL } from '../subworkflows/local/train_error_model/main'
include { MISMAPPING        } from '../modules/local/mismapping/main'
include { INFER_COMPOSITION } from '../modules/local/infer_composition/main'

workflow SUPERRESOLUTION_AMPLICON {
    take:
    ch_reads      // [ meta, [ reads ] ]                (meta.id, meta.platform)
    ch_refs       // [ meta, references_fasta ]
    ch_pretrained // [ id, model_pt ]  (samples that supply their own error model)

    main:
    ch_versions = Channel.empty()

    // Split samples that bring a pre-trained model from those that need training.
    ch_reads
        .branch { meta, reads ->
            pretrained: meta.error_model != null
            train:      true
        }
        .set { ch_split }

    TRAIN_ERROR_MODEL(ch_split.train)
    ch_versions = ch_versions.mix(TRAIN_ERROR_MODEL.out.versions)

    // Unified model channel keyed by id: [ id, model_pt ].
    ch_model = TRAIN_ERROR_MODEL.out.model
        .map { meta, model -> [ meta.id, model ] }
        .mix(ch_pretrained)

    // MISMAPPING: join references with the model by id -> [ meta, refs, model ].
    ch_mismap_in = ch_refs
        .map { meta, refs -> [ meta.id, meta, refs ] }
        .join(ch_model)
        .map { id, meta, refs, model -> [ meta, refs, model ] }
    MISMAPPING(ch_mismap_in)
    ch_versions = ch_versions.mix(MISMAPPING.out.versions)

    // INFER_COMPOSITION: reads + mismap dir + model, joined by id.
    ch_infer_in = ch_reads
        .map { meta, reads -> [ meta.id, meta, reads ] }
        .join(MISMAPPING.out.dir.map { meta, d -> [ meta.id, d ] })
        .join(ch_model)
        .map { id, meta, reads, d, model -> [ meta, reads, d, model ] }
    INFER_COMPOSITION(ch_infer_in)
    ch_versions = ch_versions.mix(INFER_COMPOSITION.out.versions)

    // Collate the per-process versions into one file.
    ch_versions
        .collectFile(name: 'software_versions.yml', storeDir: "${params.outdir}/pipeline_info")

    emit:
    model       = ch_model
    mismapping  = MISMAPPING.out.dir
    composition = INFER_COMPOSITION.out.composition
    versions    = ch_versions
}
