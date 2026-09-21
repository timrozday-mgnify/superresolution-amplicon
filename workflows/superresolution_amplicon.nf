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
include { MAPSEQ as MAPSEQ_HOME } from '../modules/local/mapseq/map/main'
include { HOME_PROBES          } from '../modules/local/home_probes/main'
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
include { CHECK_COMPOSITION_FIT } from '../modules/local/check_composition_fit/main'
include { PANEL_PREPARE        } from '../modules/local/panel_kernel/prepare/main'
include { PANEL_KERNEL         } from '../modules/local/panel_kernel/build/main'
include { SIMULATE_READS as SIMULATE_PANEL_READS } from '../modules/local/simulate_reads/main'
include { MAPSEQ as MAPSEQ_PANEL_HOME } from '../modules/local/mapseq/map/main'
include { MAPSEQ as MAPSEQ_PANEL_SIM  } from '../modules/local/mapseq/map/main'

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
    if ((params.align_home_probes as int) > 0 && params.align_backend == 'minimap2') {
        error "--align_home_probes needs a grouped backend (--align_backend exact-hash or kmer)"
    }
    if (params.align_backend == 'exact-hash' && (params.align_tau as int) != 0) {
        error "--align_backend exact-hash requires --align_tau 0; use kmer for tau >= 1"
    }
    def auto_decay = params.align_distance_decay.toString() == 'auto'
    if (!auto_decay && !((params.align_distance_decay as double) >= 0.0
                         && (params.align_distance_decay as double) <= 1.0)) {
        error "--align_distance_decay must be in [0, 1], or 'auto'"
    }
    if (params.align_decay_model && !auto_decay) {
        error "--align_decay_model is only read by --align_distance_decay auto"
    }
    // toString(): a command-line `--flag false` arrives as the truthy String "false".
    if (params.infer_distance_decay.toString() == 'true') {
        // The latent needs the distances behind the matrix's nonzeros, and needs them to
        // differ within a row. Only a tau >= 1 alignment build has either.
        if (params.mismapping_method != 'align' || (params.align_tau as int) < 1) {
            error "--infer_distance_decay needs --mismapping_method align at " +
                  "--align_tau >= 1: no other matrix records the distance behind each " +
                  "nonzero, and at tau 0 every distance is 0 and c cancels."
        }
        if (params.mismapping_matrix) {
            log.warn "--infer_distance_decay with a supplied --mismapping_matrix: it will " +
                     "be refused unless that matrix was built with distance strata."
        }
    }
    if ((params.align_tau as int) >= 1 && !auto_decay && params.infer_distance_decay.toString() != 'true'
        && (params.align_distance_decay as double) == 1.0) {
        // Not an error: it is the behaviour every matrix built before the knob existed
        // has, so a rerun of one must still be possible.
        log.warn "--align_tau ${params.align_tau} with --align_distance_decay 1 treats a " +
                 "reference within tau as an exact duplicate, which overstates confusion " +
                 "between references a base or two apart. Set --align_distance_decay to " +
                 "about the per-base error rate, or 'auto' to measure it, or set " +
                 "--infer_distance_decay to fit it per sample."
    }
    if (!(params.infer_space in ['genome', 'v4_group'])) {
        error "--infer_space must be 'genome' or 'v4_group'"
    }
    if (params.infer_horseshoe.toString() == 'true'
        && (params.infer_presence.toString() == 'true' || params.infer_mode != 'vi')) {
        error "--infer_horseshoe replaces the presence gate and needs --infer_mode vi " +
              "--infer_presence false"
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

    // In-silico PCR -> the mapseq reference set (+ translation table T), then its mapseq
    // clustering: once per distinct reference FASTA, however many samples name it. At GTDB
    // scale each is hours. The set's meta holds only its id, so adding or removing a
    // sample leaves both tasks cached.
    // The id is also the --amplicon_cache key (conf/modules.config), shared across runs,
    // so it names everything that shapes the amplicons: the FASTA, the primers and the
    // extractor's code.
    // ponytail: the FASTA by location, size and mtime, not content. The same bytes at two
    // paths are extracted twice. Key by a content digest if that happens, which costs a
    // hash of the whole FASTA per run.
    def extractor = file("${projectDir}/bin/subspecies_infer.py").text.md5()
    ch_refs_by_set = ch_refs.map { meta, refs ->
        def key = [refs.toUriString(), refs.size(), refs.lastModified(), params.fwd_primer,
                   params.rev_primer, params.primer_mismatches, extractor].join('|')
        [ "refs_" + key.md5().take(12), meta, refs ]
    }
    EXTRACT_AMPLICONS(ch_refs_by_set
        .unique { it[0] }
        .map { set, meta, refs -> [ [ id: set ], refs ] })
    ch_versions = ch_versions.mix(EXTRACT_AMPLICONS.out.versions)
    // [ set meta, amplicons.fasta, amplicons.tax ]: the mapseq reference set.
    ch_set_refs = EXTRACT_AMPLICONS.out.dir.map { meta, d ->
        [ meta, d.resolve('amplicons.fasta'), d.resolve('amplicons.tax') ] }

    // Cluster only a set something maps against with mapseq: a sample's own reads (no
    // `mseq:`) or a panel's sources. An align-mode matrix never reads the .mscluster, so a
    // sweep over supplied classifications skips it; ch_db then holds only the samples that
    // use it.
    // Home probes map the square matrix's amplicons against their own set.
    def home_probes = params.mismapping_method == 'align' && !params.mismapping_matrix &&
        (params.align_home_probes as int) > 0
    ch_cluster_sets = ch_refs_by_set
        .map { set, meta, refs -> [ meta.id, set, meta ] }
        .join(ch_reads.map { meta, reads -> [ meta.id, !meta.mseq ] })
        .filter { id, set, meta, maps_reads ->
            maps_reads || meta.panel || meta.panel_taxa || home_probes }
        .map { id, set, meta, maps_reads -> [ set ] }
        .unique()
    MAPSEQ_CLUSTER(ch_set_refs
        .map { meta, fasta, tax -> [ meta.id, meta, fasta, tax ] }
        .join(ch_cluster_sets)
        .map { set, meta, fasta, tax -> [ meta, fasta, tax ] })
    ch_versions = ch_versions.mix(MAPSEQ_CLUSTER.out.versions)

    // Fan the shared results back out to samples by set id. combine, not join: join is
    // 1:1 and would keep one sample per set.
    ch_sample_sets = ch_refs_by_set.map { set, meta, refs -> [ set, meta ] }
    // [ sample meta, amplicon_dir ]
    ch_amplicons = ch_sample_sets
        .combine(EXTRACT_AMPLICONS.out.dir.map { meta, d -> [ meta.id, d ] }, by: 0)
        .map { set, meta, d -> [ meta, d ] }
    // [ sample id, fasta, tax, mscluster ] — the mapseq DB slots, shared by both mappings.
    ch_db = ch_sample_sets
        .combine(ch_set_refs
            .map { meta, fasta, tax -> [ meta.id, fasta, tax ] }
            .join(MAPSEQ_CLUSTER.out.mscluster.map { meta, mscluster -> [ meta.id, mscluster ] }), by: 0)
        .map { set, meta, fasta, tax, mscluster -> [ meta.id, fasta, tax, mscluster ] }

    // Fingerprint extracted amplicons and group all samples that experience the same
    // simulation + mapper configuration. The group metadata is also bundle provenance.
    MATRIX_KEY(ch_amplicons
        .map { meta, d -> [ meta.id, meta, d ] }
        .join(ch_model)
        .map { id, meta, d, model, identity -> [ meta, d, model, identity ] })
    // Panel samples get a kernel of their own (below); only the rest share a square matrix.
    MATRIX_KEY.out.key
        .map { meta, d, identity_file, identity, key_file, ref_file ->
            [ key_file.text.trim(), [meta, d, identity_file, identity, ref_file.text.trim()] ]
        }
        .branch { key, entry ->
            panel:  entry[0].panel || entry[0].panel_taxa
            square: true
        }
        .set { ch_keyed }
    ch_matrix_groups = ch_keyed.square
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
                align_distance_decay: params.align_distance_decay,
                align_decay_model: params.align_decay_model,
                align_ambiguity_weight: params.align_ambiguity_weight,
                align_home_probes: params.align_home_probes,
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
        // Only read by '--align_distance_decay auto'; the placeholder keeps the input
        // slot filled and means "measure the flat rates instead". This is a *pre-trained*
        // model — align mode still never runs the skiver training subworkflow.
        ch_decay_model = file(params.align_decay_model ?: "${projectDir}/assets/NO_MODEL",
                              checkIfExists: true)
        ch_align_refs = ch_matrix_groups.map { meta, d, model ->
            [ meta, d.resolve('amplicons.fasta'), ch_decay_model ] }
        // Index once, then align against it — the reference set is the target of its own
        // all-vs-all, so without this every run re-indexes the whole DB.
        if (params.align_backend in ['kmer', 'exact-hash']) {
            // [ meta, home mseq | NO_HOME ]. The probes map against the representative
            // member's database, the same one its reads map against.
            if (home_probes) {
                HOME_PROBES(ch_matrix_groups.map { meta, d, model ->
                    [ meta, d.resolve('amplicons.fasta') ] })
                MAPSEQ_HOME(HOME_PROBES.out.probes
                    .map { meta, probes -> [ meta.members[0].id, meta, probes ] }
                    .combine(ch_db, by: 0)
                    .map { id, meta, probes, fasta, tax, mscluster ->
                        [ meta, probes, fasta, tax, mscluster ] })
                ch_versions = ch_versions.mix(HOME_PROBES.out.versions)
                                         .mix(MAPSEQ_HOME.out.versions)
                ch_home = MAPSEQ_HOME.out.mseq
            }
            else {
                ch_home = ch_matrix_groups.map { meta, d, model ->
                    [ meta, file("${projectDir}/assets/NO_HOME", checkIfExists: true) ] }
            }
            GROUPED_MISMAPPING(ch_align_refs
                .map { meta, fasta, model -> [ meta.id, meta, fasta, model ] }
                .join(ch_home.map { meta, home -> [ meta.id, home ] })
                .map { id, meta, fasta, model, home -> [ meta, fasta, model, home ] })
            ch_versions = ch_versions.mix(GROUPED_MISMAPPING.out.versions)
            ch_bundle_in = GROUPED_MISMAPPING.out.mismapping
                .map { meta, matrix -> [ meta.id, meta, matrix ] }
                .join(ch_matrix_groups.map { meta, d, model -> [ meta.id, d ] })
                .map { id, meta, matrix, d -> [ meta, d, matrix ] }
        }
        else {
        ch_minimap2_refs = ch_align_refs.map { meta, fasta, model -> [ meta, fasta ] }
        MINIMAP2_INDEX(ch_minimap2_refs)
        MINIMAP2_ALLVSALL(ch_minimap2_refs
            .map { meta, fasta -> [ meta.id, meta, fasta ] }
            .join(MINIMAP2_INDEX.out.index.map { meta, mmi -> [ meta.id, mmi ] })
            .map { id, meta, fasta, mmi -> [ meta, fasta, mmi ] })
        ch_versions = ch_versions.mix(MINIMAP2_INDEX.out.versions)
                                 .mix(MINIMAP2_ALLVSALL.out.versions)
        ch_align_in = ch_align_refs
            .map { meta, fasta, model -> [ meta.id, meta, fasta, model ] }
            .join(MINIMAP2_ALLVSALL.out.paf.map { meta, paf -> [ meta.id, paf ] })
            .map { id, meta, fasta, model, paf -> [ meta, fasta, paf, model ] }
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
    // [ id, amplicon_dir, matrix, matrix_key ]: a square-matrix sample keeps its own
    // extracted amplicons.
    ch_mismapping = PUBLISH_MISMAPPING.out.bundle
        .flatMap { meta, bundle -> meta.members.collect { member ->
            [ member.id, bundle.resolve('mismapping_matrix.npz'), meta.matrix_key ]
        } }
        .join(ch_amplicons.map { meta, d -> [ meta.id, d ] })
        .map { id, matrix, key, d -> [ id, d, matrix, key ] }

    // Panel reinterpretation: one rectangular kernel per (database + model + settings, panel).
    // Its sources are the panel's distinct V4 amplicons and its labels the database's V4
    // groups, measured with the same MAPseq database the sample's reads are mapped against.
    ch_panel_groups = ch_keyed.panel
        .map { key, entry -> [ [key, entry[0].panel, entry[0].panel_taxa], entry ] }
        .groupTuple()
        .map { group, entries ->
            def (key, panel, panel_taxa) = group
            def rep = entries[0]
            // Genome-only panels keep their pre-taxa id (and so their published kernel dir).
            def taxa_key = panel_taxa ? "|${panel_taxa}|${params.panel_taxon_max_sources}" : ''
            def id = "panel_" + "${key}|${panel}|${params.panel_sim_n_per_ref}${taxa_key}".toString().md5().take(16)
            [[id: id, matrix_key: id, panel: panel, panel_taxa: panel_taxa, model_identity: rep[3],
              members: entries.collect { it[0].id }, db_id: rep[0].id],
             panel, rep[1], rep[2]]
        }
    PANEL_PREPARE(
        ch_panel_groups.map { meta, panel, d, model ->
            [ meta, panel ?: [], meta.panel_taxa ?: [], d.resolve('amplicons.fasta') ] },
        params.taxonomy ? file(params.taxonomy, checkIfExists: true) : [])
    ch_versions = ch_versions.mix(PANEL_PREPARE.out.versions)
    // [ meta, sources, fasta, tax, mscluster ] against the representative's database.
    ch_panel_db = PANEL_PREPARE.out.sources
        .map { meta, sources -> [ meta.db_id, meta, sources ] }
        .combine(ch_db, by: 0)
        .map { db_id, meta, sources, fasta, tax, mscluster -> [ meta, sources, fasta, tax, mscluster ] }
    MAPSEQ_PANEL_HOME(ch_panel_db)
    SIMULATE_PANEL_READS(PANEL_PREPARE.out.sources
        .map { meta, sources -> [ meta.id, meta, sources ] }
        .join(ch_panel_groups.map { meta, panel, d, model -> [ meta.id, model ] })
        .map { id, meta, sources, model -> [ meta, sources, model ] })
    MAPSEQ_PANEL_SIM(SIMULATE_PANEL_READS.out.reads
        .map { meta, reads -> [ meta.id, reads ] }
        .join(ch_panel_db.map { meta, sources, fasta, tax, mscluster -> [ meta.id, meta, fasta, tax, mscluster ] })
        .map { id, reads, meta, fasta, tax, mscluster -> [ meta, reads, fasta, tax, mscluster ] })
    ch_versions = ch_versions.mix(MAPSEQ_PANEL_HOME.out.versions)
        .mix(SIMULATE_PANEL_READS.out.versions).mix(MAPSEQ_PANEL_SIM.out.versions)
    PANEL_KERNEL(PANEL_PREPARE.out.prepared
        .map { meta, prepared -> [ meta.id, meta, prepared ] }
        .join(ch_panel_db.map { meta, sources, fasta, tax, mscluster -> [ meta.id, fasta ] })
        .join(MAPSEQ_PANEL_HOME.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
        .join(MAPSEQ_PANEL_SIM.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
        .map { id, meta, prepared, fasta, home, sim -> [ meta, prepared, fasta, home, sim ] })
    ch_versions = ch_versions.mix(PANEL_KERNEL.out.versions)
    // The kernel directory is the panel sample's amplicon dir (panel_translation.tsv).
    ch_mismapping = ch_mismapping.mix(PANEL_KERNEL.out.kernel
        .flatMap { meta, kdir -> meta.members.collect { member ->
            [ member, kdir, kdir.resolve('mismapping_matrix.npz'), meta.matrix_key ]
        } })

    // Real reads -> fasta -> mapseq -> the observed per-reference counts.
    // A sample carrying `mseq:` in the samplesheet supplies that classification instead
    // and skips both steps — mapping the same reads against the same reference set is by
    // far the most expensive part of a run, so a parameter sweep over the mis-mapping or
    // inference knobs should do it once. The supplied file must have been produced
    // against THIS reference set (its reference ids are matched to the extracted
    // amplicons) and bakes in the read-preparation settings it was made with
    // (--obs_max_reads, --trim_primers, --min_pair_overlap): those params no longer
    // apply to that sample.
    ch_reads
        .branch { meta, reads ->
            supplied: meta.mseq
            map:      true
        }
        .set { ch_obs }

    READS_TO_FASTA(ch_obs.map)
    ch_versions = ch_versions.mix(READS_TO_FASTA.out.versions)

    MAPSEQ_OBS(READS_TO_FASTA.out.reads
        .map { meta, reads -> [ meta.id, meta, reads ] }
        .join(ch_db)
        .map { id, meta, reads, fasta, tax, mscluster -> [ meta, reads, fasta, tax, mscluster ] })
    ch_versions = ch_versions.mix(MAPSEQ_OBS.out.versions)

    ch_obs_mseq = MAPSEQ_OBS.out.mseq
        .map { meta, mseq -> [ meta.id, mseq ] }
        .mix(ch_obs.supplied.map { meta, reads -> [ meta.id, meta.mseq ] })

    // INFER_COMPOSITION: amplicon dir + canonical matrix + observed mseq, joined by id.
    ch_infer_in = ch_amplicons
        .map { meta, d -> [ meta.id, meta ] }
        .join(ch_mismapping)
        .join(ch_obs_mseq)
        .map { id, meta, d, matrix, matrix_key, obs -> [ meta, d, matrix, obs, matrix_key ] }
    INFER_COMPOSITION(ch_infer_in,
                      params.taxonomy ? file(params.taxonomy, checkIfExists: true) : [])
    ch_versions = ch_versions.mix(INFER_COMPOSITION.out.versions)

    // Reconstruct the observation-space forward fit for every sample. This remains
    // downstream of the depth gate so low-depth samples receive an explicit diagnostic
    // rather than a silently interpretable composition. In v4_group space the group table
    // is the fitted result; the genome table is only a labelled split of it.
    ch_fit_composition = params.infer_space == 'v4_group'
        ? INFER_COMPOSITION.out.v4_groups : INFER_COMPOSITION.out.composition
    ch_fit_in = ch_infer_in
        .map { meta, d, matrix, obs, matrix_key -> [ meta.id, meta, d, matrix, obs ] }
        .join(ch_fit_composition.map { meta, composition -> [ meta.id, composition ] })
        .join(INFER_COMPOSITION.out.posterior.map { meta, posterior -> [ meta.id, posterior ] })
        .map { id, meta, d, matrix, obs, composition, posterior ->
            [ meta, d, matrix, obs, composition, posterior ] }
    CHECK_COMPOSITION_FIT(ch_fit_in)
    ch_versions = ch_versions.mix(CHECK_COMPOSITION_FIT.out.versions)

    // Collate the per-process versions into one file.
    ch_versions
        .collectFile(name: 'software_versions.yml', storeDir: "${params.outdir}/pipeline_info")

    emit:
    amplicons   = ch_amplicons
    mismapping  = PUBLISH_MISMAPPING.out.bundle
    composition = INFER_COMPOSITION.out.composition
    fit_diagnostics = CHECK_COMPOSITION_FIT.out.diagnostics
    versions    = ch_versions
}
