## ADDED Requirements

### Requirement: Sink-issued diagnostic checkpoints
`DiagnosticSink` SHALL issue an opaque `DiagnosticCheckpoint` from its current length. Code that examines new diagnostics SHALL call `iter_since` with this checkpoint instead of a raw `usize` offset.

The checkpoint constructor SHALL remain private to the diagnostic-session module. The sink SHALL only append diagnostics while a checkpoint is active.

#### Scenario: Iteration starts at a checkpoint
- **WHEN** inference records a checkpoint and then appends diagnostics
- **THEN** `iter_since` returns the diagnostics that follow the checkpoint in insertion order

#### Scenario: Earlier diagnostics exist
- **WHEN** the sink contains diagnostics before it issues a checkpoint
- **THEN** `iter_since` excludes those earlier diagnostics

#### Scenario: A caller supplies a raw offset
- **WHEN** code passes a `usize` to `iter_since`
- **THEN** Rust rejects the call because `iter_since` requires `DiagnosticCheckpoint`
