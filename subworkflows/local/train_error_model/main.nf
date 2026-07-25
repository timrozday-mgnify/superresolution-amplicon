//
// Train a skiver context error model + Phred calibration from a natural
// (non-synthetic) metagenome. Reference-free: skiver dump builds a consensus.
//

include { SKIVER_DUMP   } from '../../../modules/local/skiver/dump/main'
include { SKIVER_TRAIN  } from '../../../modules/local/skiver/train/main'
include { SKIVER_REPORT } from '../../../modules/local/skiver/report/main'

workflow TRAIN_ERROR_MODEL {
    take:
    ch_reads // channel: [ val(meta), [ reads ] ]  (meta.id = train_id, meta.platform)

    main:
    ch_versions = Channel.empty()

    SKIVER_DUMP(ch_reads)
    ch_versions = ch_versions.mix(SKIVER_DUMP.out.versions.first())

    SKIVER_TRAIN(SKIVER_DUMP.out.base)
    ch_versions = ch_versions.mix(SKIVER_TRAIN.out.versions.first())

    SKIVER_REPORT(SKIVER_TRAIN.out.model.join(SKIVER_TRAIN.out.aic))
    ch_versions = ch_versions.mix(SKIVER_REPORT.out.versions.first())

    emit:
    model       = SKIVER_TRAIN.out.model        // [ meta, model.pt ]
    calibration = SKIVER_TRAIN.out.calibration  // [ meta, phred_calibration.json ]
    aic         = SKIVER_TRAIN.out.aic          // [ meta, context_model_aic.csv ]
    report      = SKIVER_REPORT.out.report      // [ meta, error_model_report.html ]
    versions    = ch_versions
}
