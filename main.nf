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
    //   id, reads (or fastq_1[/fastq_2]), platform, references (optional), error_model (optional)
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
        if (row.reads) {
            reads = (row.reads instanceof List ? row.reads : [row.reads]).collect { resolveFile(it.toString()) }
        }
        else if (row.fastq_1) {
            reads << resolveFile(row.fastq_1.toString())
            if (row.fastq_2?.toString()?.trim()) reads << resolveFile(row.fastq_2.toString())
        }
        else {
            error "Sample ${row.id} needs 'reads' (list) or 'fastq_1'[/'fastq_2']"
        }
        def meta = [
            id:          row.id,
            platform:    (row.platform ?: params.platform),
            error_model: (row.error_model ? resolveFile(row.error_model.toString()) : null),
        ]
        [ meta, reads ]
    }

    // [ meta, references_fasta ] — per-sample 'references' overrides the global param.
    ch_refs = ch_rows.map { row ->
        def ref = row.references ?: params.references
        if (!ref) error "Sample ${row.id}: no references (set samplesheet 'references' or --references)"
        def meta = [ id: row.id, platform: (row.platform ?: params.platform) ]
        [ meta, resolveFile(ref.toString()) ]
    }

    // [ id, model_pt ] for samples supplying a pre-trained error model.
    ch_pretrained = ch_rows
        .filter { row -> row.error_model }
        .map { row -> [ row.id, resolveFile(row.error_model.toString()) ] }

    SUPERRESOLUTION_AMPLICON(ch_reads, ch_refs, ch_pretrained)
}
