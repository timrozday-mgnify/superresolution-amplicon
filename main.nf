#!/usr/bin/env nextflow

include { SUPERRESOLUTION_AMPLICON } from './workflows/superresolution_amplicon'

// Resolve a samplesheet-relative path: absolute / URL passes through, otherwise
// resolve against the pipeline projectDir.
def resolveFile(String p) {
    (p.startsWith('/') || p =~ /^[a-z]+:\/\//)
        ? file(p, checkIfExists: true)
        : file("${workflow.projectDir}/${p}", checkIfExists: true)
}

workflow {
    main:
    if (!params.input) {
        error "Provide a samplesheet with --input"
    }

    // YAML samplesheet: a list of sample entries, each:
    //   id, reads (or fastq_1[/fastq_2]), platform, references (optional),
    //   error_model (optional), mseq (optional), merged (optional),
    //   panel_references (optional), panel_taxa (optional)
    def loaded = new org.yaml.snakeyaml.Yaml().load(file(params.input, checkIfExists: true).text)
    def rows = (loaded instanceof Map) ? loaded.samples : loaded
    if (!(rows instanceof List)) {
        error "Samplesheet ${params.input} must be a YAML list of samples (or a map with 'samples:')"
    }

    if (params.panel_kernel_only && params.panel_kernel) {
        error "--panel_kernel_only builds a kernel; --panel_kernel reuses one. Pick one."
    }
    if (!(params.sim_read_structure in ['merged', 'pairs'])) {
        error "--sim_read_structure must be 'merged' or 'pairs'"
    }
    if (params.sim_read_structure == 'pairs' && params.trim_primers.toString() == 'true') {
        error "--sim_read_structure pairs simulates AAP merged reads, which keep their primers: set --trim_primers false"
    }

    ch_rows = Channel.fromList(rows)

    // Whether this run has to train an error model from reads. Only the `simulate`
    // kernel's `trained` model reads a FASTQ at all; a supplied kernel, `align` and the
    // flat model each take the reads channel for its meta and drop the files.
    def trains_a_model = !params.panel_kernel && !params.panel_kernel_only &&
                         params.mismapping_method != 'align' &&
                         params.sim_error_model != 'flat'

    // [ meta, [ reads ] ]
    ch_reads = ch_rows.map { row ->
        if (!row.id) error "Each sample needs an 'id'"
        def reads = []
        // Paired only when the sample says so: `fastq_1`+`fastq_2`, or an explicit
        // `paired: true`. A two-entry `reads` list is ambiguous (two single-end runs look
        // identical to one pair), so it is never guessed at.
        def paired = false
        if (row.reads) {
            reads = (row.reads instanceof List ? row.reads : [row.reads]).collect { resolveFile(it.toString()) }
            paired = (row.paired?.toString() == 'true')
            if (paired && reads.size() != 2) {
                error "Sample ${row.id}: 'paired: true' needs exactly two files in 'reads'"
            }
        }
        else if (row.fastq_1) {
            reads << resolveFile(row.fastq_1.toString())
            if (row.fastq_2?.toString()?.trim()) {
                reads << resolveFile(row.fastq_2.toString())
                paired = true
            }
        }
        else if (row.mseq) {
            // Reads are optional on an `mseq:` row: the classification already stands in
            // for them, and nothing downstream opens the files. This is the OTU
            // reinterpretation route -- an amplicon-analysis-pipeline run's own MAPseq
            // labels re-read over a panel, with no FASTQ fetched at all. Keep `merged:
            // true` on such a row anyway: it still declares how the reads behind that
            // classification were prepared, which --trim_primers has to match.
            if (trains_a_model && !(row.error_model || params.error_model)) {
                error "Sample ${row.id}: 'mseq' without 'reads' has nothing to train an " +
                      "error model on. Give the row an 'error_model' (or --error_model), " +
                      "or pick a kernel that needs none: --sim_error_model flat, " +
                      "--mismapping_method align, or --panel_kernel."
            }
        }
        else if (!params.panel_kernel_only) {
            error "Sample ${row.id} needs 'reads' (list) or 'fastq_1'[/'fastq_2'], or 'mseq' on its own"
        }
        // `merged: true` marks reads that are already merged pairs, e.g. AAP's
        // qc/<id>.merged.fastq.gz (bin/aap_samplesheet.py). They are never merged again,
        // and never primer-trimmed: AAP's reads (and its .mseq) keep their primers, and
        // the simulated reads must be prepared the same way, which --trim_primers sets
        // for the whole run.
        if (row.merged?.toString() == 'true') {
            if (paired) error "Sample ${row.id}: 'merged: true' reads are one file, not a pair"
            // A low merge rate means pairs were lost to the merge unevenly across sources
            // (long amplicons first), which edit distances cannot describe.
            if (params.mismapping_method == 'align' && row.merge_rate != null
                    && (row.merge_rate as double) < 0.8) {
                error "Sample ${row.id}: merge_rate ${row.merge_rate} < 0.8, so merging lost reads; " +
                      "--mismapping_method align cannot account for that. Use simulate with --sim_read_structure pairs"
            }
            if (params.trim_primers.toString() == 'true') {
                error "Sample ${row.id}: 'merged: true' needs --trim_primers false (merged reads keep their primers)"
            }
        }
        if (params.sim_read_structure == 'pairs' && params.sim_error_model == 'trained'
                && !(row.error_model || params.error_model)) {
            error "Sample ${row.id}: --sim_read_structure pairs simulates mates, so a trained model must be a mate model: set error_model"
        }
        def meta = [
            id:          row.id,
            platform:    (row.platform ?: params.platform),
            paired:      paired,
            error_model: (row.error_model ? resolveFile(row.error_model.toString()) :
                          (params.error_model ? resolveFile(params.error_model.toString()) : null)),
        ]
        // A precomputed mapseq classification of THIS sample's reads against THIS
        // reference set: supplying it skips READS_TO_FASTA + MAPSEQ_OBS. Added only when
        // present so the mapping tasks of runs that don't use it keep their cached hash.
        if (row.mseq) meta.mseq = resolveFile(row.mseq.toString())
        [ meta, reads ]
    }

    // A supplied kernel with no database named anywhere: its bundle already maps every
    // database header to a V4 label, so each row's mseq is reinterpreted without the
    // database. Naming one keeps the full check of the bundle against it.
    def bundle_only = params.panel_kernel && !rows.any { it.references ?: params.references }

    // [ meta, references_fasta ] — per-sample 'references' overrides the global param.
    ch_refs = ch_rows.map { row ->
        def ref = row.references ?: params.references
        if (!ref && !bundle_only) error "Sample ${row.id}: no references (set samplesheet 'references' or --references)"
        def meta = [ id: row.id, platform: (row.platform ?: params.platform) ]
        // A supplied genome/taxon panel can reinterpret this sample's database labels. A
        // normal sample instead uses every extracted database V4 group as its panel.
        def panel = row.panel_references ?: params.panel_references
        def panel_taxa = row.panel_taxa ?: params.panel_taxa
        if (panel || panel_taxa) {
            // Taxon entries resolve against the database's own lineages.
            if (panel_taxa && !params.taxonomy) {
                error "Sample ${row.id}: panel_taxa needs --taxonomy (the database's MAPseq .tax)"
            }
            if (panel) meta.panel = resolveFile(panel.toString())
            if (panel_taxa) meta.panel_taxa = resolveFile(panel_taxa.toString())
        }
        else {
            meta.panel = 'database'
        }
        [ meta, ref ? resolveFile(ref.toString()) : [] ]
    }

    // [ id, model_pt ] for samples supplying a pre-trained error model.
    ch_pretrained = ch_rows
        .filter { row -> row.error_model }
        .map { row -> [ row.id, resolveFile(row.error_model.toString()) ] }

    // AAP batches (any `merged: true` row) default to one pooled model: every run went
    // through the same read preparation, and pooling trains on all their reads at once.
    def trained_scope = params.trained_error_model_scope
        ?: (rows.any { it.merged?.toString() == 'true' } ? 'pooled' : 'per-sample')

    SUPERRESOLUTION_AMPLICON(ch_reads, ch_refs, ch_pretrained, trained_scope, bundle_only)
}
