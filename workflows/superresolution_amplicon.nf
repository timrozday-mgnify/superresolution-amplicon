//
// superresolution-amplicon: extract the reference amplicons, build one rectangular panel
// kernel per panel/database/settings combination, then infer the composition from MAPseq's
// observed database labels.
//
include { TRAIN_ERROR_MODEL    } from '../subworkflows/local/train_error_model/main'
include { EXTRACT_AMPLICONS    } from '../modules/local/extract_amplicons/main'
include { MAPSEQ_CLUSTER       } from '../modules/local/mapseq/cluster/main'
include { MAPSEQ as MAPSEQ_OBS } from '../modules/local/mapseq/map/main'
include { MATRIX_KEY            } from '../modules/local/matrix_key/main'
include { POOL_TRAINING_READS   } from '../modules/local/pool_training_reads/main'
include { READS_TO_FASTA       } from '../modules/local/reads_to_fasta/main'
include { INFER_COMPOSITION    } from '../modules/local/infer_composition/main'
include { CHECK_COMPOSITION_FIT } from '../modules/local/check_composition_fit/main'
include { PANEL_PREPARE        } from '../modules/local/panel_kernel/prepare/main'
include { PANEL_KERNEL; PANEL_ALIGN } from '../modules/local/panel_kernel/build/main'
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
    // `as int`: a --align_tau on the command line arrives as a String, and comparing that
    // to a number silently misjudges the alignment validation below.
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
    if (params.mismapping_method == 'align' && params.sim_read_len) {
        // Whole-reference distances cannot see what a short read cannot see, and quietly
        // understating confusion is worse than refusing.
        error "--mismapping_method align builds M from whole-reference alignments and " +
              "cannot model --sim_read_len windows. Use --mismapping_method simulate for " +
              "reads shorter than the amplicon, or merge pairs so queries span it."
    }

    if (params.panel_kernel) {
        // A supplied kernel: nothing is simulated or aligned, so nothing needs a model.
        ch_model = ch_reads.map { meta, reads -> [ meta.id, file("${projectDir}/assets/NO_MODEL"), 'supplied-kernel' ] }
    }
    else if (params.mismapping_method == 'align') {
        // Alignment never trains an error model. `auto` can, however, measure a supplied
        // pre-trained model's error rate, so its content fingerprint enters the panel key.
        def decay_model = params.align_decay_model
            ? file(params.align_decay_model, checkIfExists: true)
            : file("${projectDir}/assets/NO_MODEL")
        def decay_identity = params.align_decay_model ? 'align-decay-model' : 'align-flat'
        ch_model = ch_reads.map { meta, reads -> [ meta.id, decay_model, decay_identity ] }
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

    // In-silico PCR -> the V4 amplicons (+ translation table T), then the FASTA's mapseq
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

    // The MAPseq database is the reference FASTA itself, not its amplicons: every mapping
    // (observed reads, panel home and simulated reads) goes against it, so a database
    // AAP ships (FASTA + .mscluster) labels reads exactly as AAP does. Its amplicon hits
    // carry the FASTA's own headers; hits on an entry with no amplicon are off-target.
    // MAPseq gets the generated references.tax: the tax file changes only the taxonomy
    // columns, never the hit (400/400 SILVA queries identical under AAP's tax file).
    // A `<fasta>.mscluster` beside the FASTA is used as is; otherwise it is built once.
    // [ set, refs, tax, prebuilt mscluster | null, amplicon dir ]
    ch_set_src = ch_refs_by_set
        .map { set, meta, refs -> [ set, refs ] }
        .unique { it[0] }
        .join(EXTRACT_AMPLICONS.out.dir.map { meta, d -> [ meta.id, d ] })
        .map { set, refs, d ->
            if (refs.name.endsWith('.gz')) {
                error "MAPseq maps against the reference FASTA itself and cannot read a " +
                      "gzipped one: gunzip ${refs}"
            }
            def mscluster = refs.resolveSibling(refs.name + '.mscluster')
            if (!mscluster.exists() && refs.size() > 100_000_000) {
                log.warn "${refs} has no ${mscluster.name} beside it: clustering it here, " +
                         "once per --amplicon_cache, can take hours"
            }
            [ set, refs, d.resolve('references.tax'), mscluster.exists() ? mscluster : null, d ]
        }
        .branch { set, refs, tax, mscluster, d ->
            prebuilt: mscluster
            cluster:  true
        }
    MAPSEQ_CLUSTER(ch_set_src.cluster.map { set, refs, tax, mscluster, d -> [ [ id: set ], refs, tax ] })
    ch_versions = ch_versions.mix(MAPSEQ_CLUSTER.out.versions)
    // [ set, amplicon dir, fasta, tax, mscluster, database identity ]. The identity names
    // the FASTA and where its clustering came from, not the clustering's work path, so the
    // panel key is stable across runs.
    ch_sets = ch_set_src.prebuilt
        .map { set, refs, tax, mscluster, d ->
            [ set, d, refs, tax, mscluster,
              [refs.toUriString(), refs.size(), refs.lastModified(), mscluster.size(),
               mscluster.lastModified()].join('|') ] }
        .mix(ch_set_src.cluster
            .map { set, refs, tax, mscluster, d -> [ set, refs, tax, d ] }
            .join(MAPSEQ_CLUSTER.out.mscluster.map { meta, mscluster -> [ meta.id, mscluster ] })
            .map { set, refs, tax, d, mscluster ->
                [ set, d, refs, tax, mscluster,
                  [refs.toUriString(), refs.size(), refs.lastModified(), 'clustered'].join('|') ] })

    // Fan the shared results back out to samples by set id. combine, not join: join is
    // 1:1 and would keep one sample per set.
    ch_sample_sets = ch_refs_by_set.map { set, meta, refs -> [ set, meta ] }
    ch_sample_db = ch_sample_sets.combine(ch_sets, by: 0)
    // [ sample meta, amplicon_dir ]
    ch_amplicons = ch_sample_db.map { set, meta, d, fasta, tax, mscluster, db -> [ meta, d ] }
    // [ sample id, fasta, tax, mscluster ] — the mapseq DB slots, shared by every mapping.
    ch_db = ch_sample_db.map { set, meta, d, fasta, tax, mscluster, db ->
        [ meta.id, fasta, tax, mscluster ] }

    // Fingerprint every panel kernel. The hash includes the extracted database, the MAPseq
    // database, panel identity, model identity, and every method setting that can affect
    // the kernel.
    MATRIX_KEY(ch_sample_db
        .map { set, meta, d, fasta, tax, mscluster, db -> [ meta.id, meta, d, db ] }
        .join(ch_model)
        .map { id, meta, d, db, model, identity -> [ meta, d, model, identity, db ] })
    ch_panel_groups = MATRIX_KEY.out.key
        .map { meta, d, identity_file, identity, key_file, ref_file, mapseq_db ->
            [ key_file.text.trim(), [meta, d, identity_file, identity, ref_file.text.trim(), mapseq_db] ]
        }
        .groupTuple()
        .map { key, entries ->
            def rep = entries[0]
            def panel = rep[0].panel ?: 'database'
            def panel_taxa = rep[0].panel_taxa
            def id = "panel_${key.take(16)}"
            def members = entries.collect { it[0].id }
            // Strings, not paths: this is written to the bundle's provenance.json, and
            // --panel_kernel compares it field by field.
            def provenance = [
                matrix_key: id, reference_sha256: rep[4], mapseq_db: rep[5],
                panel: panel.toString(), panel_taxa: panel_taxa?.toString(), model_identity: rep[3],
                mismapping_method: params.mismapping_method, align_tau: params.align_tau,
                align_distance_decay: params.align_distance_decay,
                align_ambiguity_weight: params.align_ambiguity_weight,
                max_ambiguous_bases: params.max_ambiguous_bases, max_postings: params.max_postings,
                sim_error_model: params.sim_error_model,
                sim_n_per_ref: params.panel_sim_n_per_ref, sim_read_len: params.sim_read_len,
                flat_sub_rate: params.flat_sub_rate, flat_ins_rate: params.flat_ins_rate,
                flat_del_rate: params.flat_del_rate, mapseq_args: params.mapseq_args,
                mapseq_min_identity: params.mapseq_min_identity, mapseq_tag: params.mapseq_tag,
                seed: params.seed, samples: members
            ]
            [[id: id, matrix_key: id, panel: panel, panel_taxa: panel_taxa,
              model_identity: rep[3], members: members, db_id: rep[0].id,
              provenance: provenance], panel, rep[1], rep[2]]
        }

    if (params.panel_kernel) {
        // Reuse a published bundle instead of building one. Its kernel is only valid for
        // the label space (extracted amplicons), MAPseq database and panel it was built
        // against, so each of those must match every sample's.
        def kdir = file(params.panel_kernel, checkIfExists: true)
        ['mismapping_matrix.npz', 'panel_translation.tsv', 'sources.tsv', 'provenance.json'].each {
            if (!kdir.resolve(it).exists()) {
                error "--panel_kernel ${kdir} is not a mismapping/panel_<key>/ bundle: no ${it}"
            }
        }
        def supplied = new groovy.json.JsonSlurper().parseText(kdir.resolve('provenance.json').text)
        ch_panel_kernel = ch_panel_groups.map { meta, panel, d, model ->
            def diff = ['reference_sha256', 'mapseq_db', 'panel', 'panel_taxa'].findAll { k ->
                supplied[k]?.toString() != meta.provenance[k]?.toString() }
            if (diff) {
                error "--panel_kernel ${kdir} was built for a different " +
                      diff.collect { k -> "${k} (${supplied[k]}, not ${meta.provenance[k]})" }.join(', ') +
                      " than sample(s) ${meta.members.join(', ')}"
            }
            [ meta + [matrix_key: supplied.matrix_key], kdir ]
        }
    }
    else {
        PANEL_PREPARE(
            ch_panel_groups.map { meta, panel, d, model ->
                [ meta, panel == 'database' ? [] : panel ?: [], meta.panel_taxa ?: [],
                  d.resolve('amplicons.fasta'), d ] },
            params.taxonomy ? file(params.taxonomy, checkIfExists: true) : [])
        ch_versions = ch_versions.mix(PANEL_PREPARE.out.versions)
        // [ meta, sources, fasta, tax, mscluster ] against the representative's database.
        ch_panel_db = PANEL_PREPARE.out.sources
            .map { meta, sources -> [ meta.db_id, meta, sources ] }
            .combine(ch_db, by: 0)
            .map { db_id, meta, sources, fasta, tax, mscluster -> [ meta, sources, fasta, tax, mscluster ] }
        MAPSEQ_PANEL_HOME(ch_panel_db)
        // [ panel id, database amplicons.fasta ]
        ch_db_amplicons = ch_panel_groups.map { meta, panel, d, model -> [ meta.id, d.resolve('amplicons.fasta') ] }
        ch_versions = ch_versions.mix(MAPSEQ_PANEL_HOME.out.versions)
        if (params.mismapping_method == 'simulate') {
            SIMULATE_PANEL_READS(PANEL_PREPARE.out.sources
                .map { meta, sources -> [ meta.id, meta, sources ] }
                .join(ch_panel_groups.map { meta, panel, d, model -> [ meta.id, model ] })
                .map { id, meta, sources, model -> [ meta, sources, model ] })
            MAPSEQ_PANEL_SIM(SIMULATE_PANEL_READS.out.reads
                .map { meta, reads -> [ meta.id, reads ] }
                .join(ch_panel_db.map { meta, sources, fasta, tax, mscluster -> [ meta.id, meta, fasta, tax, mscluster ] })
                .map { id, reads, meta, fasta, tax, mscluster -> [ meta, reads, fasta, tax, mscluster ] })
            ch_versions = ch_versions.mix(SIMULATE_PANEL_READS.out.versions)
                .mix(MAPSEQ_PANEL_SIM.out.versions)
            // The kernel's labels are the database's V4 groups, so it takes the extracted
            // amplicons, never the MAPseq FASTA (whole SSU sequences would group nothing).
            PANEL_KERNEL(PANEL_PREPARE.out.prepared
                .map { meta, prepared -> [ meta.id, meta, prepared ] }
                .join(ch_db_amplicons)
                .join(MAPSEQ_PANEL_HOME.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
                .join(MAPSEQ_PANEL_SIM.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
                .map { id, meta, prepared, fasta, home, sim -> [ meta, prepared, fasta, home, sim ] })
            ch_versions = ch_versions.mix(PANEL_KERNEL.out.versions)
            ch_panel_kernel = PANEL_KERNEL.out.kernel
        }
        else {
            PANEL_ALIGN(PANEL_PREPARE.out.prepared
                .map { meta, prepared -> [ meta.id, meta, prepared ] }
                .join(ch_db_amplicons)
                .join(MAPSEQ_PANEL_HOME.out.mseq.map { meta, mseq -> [ meta.id, mseq ] })
                .join(ch_panel_groups.map { meta, panel, d, model -> [ meta.id, model ] })
                .map { id, meta, prepared, fasta, home, model -> [ meta, prepared, fasta, home, model ] })
            ch_versions = ch_versions.mix(PANEL_ALIGN.out.versions)
            ch_panel_kernel = PANEL_ALIGN.out.kernel
        }
    }
    // The kernel directory carries panel_translation.tsv and sources.tsv for inference.
    ch_mismapping = ch_panel_kernel
        .flatMap { meta, kdir -> meta.members.collect { member ->
            [ member, kdir, kdir.resolve('mismapping_matrix.npz'), meta.matrix_key ]
        } }

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
    mismapping  = ch_panel_kernel
    composition = INFER_COMPOSITION.out.composition
    fit_diagnostics = CHECK_COMPOSITION_FIT.out.diagnostics
    versions    = ch_versions
}
