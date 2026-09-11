# Repository instructions

## GTDB-scale amplicon inference

Before changing GTDB reference preparation, mis-mapping, composition inference, or report
interpretation, read [the GTDB inference recovery plan](docs/gtdb_inference_recovery_plan.md).
It is the approved implementation specification for this work.

Key constraints to preserve:

- Do not run or interpret raw genome/strain calls from the full GTDB V4 reference set.
- Use a batch-level, recall-oriented candidate database of unique V4 groups; retain
  ambiguous groups and report identifiable blocks/LCA rather than a MAPseq top-hit genome.
- The production candidate matrix may use `align/kmer`, `tau=1`, only with calibrated
  distance decay, candidate-DB remapping of the observed reads, calibration checks, and
  posterior-predictive release gates.
- Implement Phase 0 diagnostics before treating a completed inference as healthy.

