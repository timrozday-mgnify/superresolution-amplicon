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
    //   error_model (optional), mseq (optional), panel_references (optional),
    //   panel_taxa (optional)
    def loaded = new org.yaml.snakeyaml.Yaml().load(file(params.input, checkIfExists: true).text)
    def rows = (loaded instanceof Map) ? loaded.samples : loaded
    if (!(rows instanceof List)) {
        error "Samplesheet ${params.input} must be a YAML list of samples (or a map with 'samples:')"
    }

    ch_rows = Channel.fromList(rows)

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
        else {
            error "Sample ${row.id} needs 'reads' (list) or 'fastq_1'[/'fastq_2']"
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

    // [ meta, references_fasta ] — per-sample 'references' overrides the global param.
    ch_refs = ch_rows.map { row ->
        def ref = row.references ?: params.references
        if (!ref) error "Sample ${row.id}: no references (set samplesheet 'references' or --references)"
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
        [ meta, resolveFile(ref.toString()) ]
    }

    // [ id, model_pt ] for samples supplying a pre-trained error model.
    ch_pretrained = ch_rows
        .filter { row -> row.error_model }
        .map { row -> [ row.id, resolveFile(row.error_model.toString()) ] }

    SUPERRESOLUTION_AMPLICON(ch_reads, ch_refs, ch_pretrained)
}
