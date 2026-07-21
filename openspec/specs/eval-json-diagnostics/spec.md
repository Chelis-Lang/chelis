# Eval JSON Diagnostics Specification

## Purpose

Define the stdout, stderr, and exit-status contract for empty and non-empty
`chelis eval --json` results across direct-file and Reef-context dispatch.

## Requirements

### Requirement: Empty JSON evaluations emit an operator diagnostic

When `chelis eval --json` completes successfully with an `EvalResult` whose
`roots` and `transcript` are both empty, the CLI SHALL emit
`warning: input contains only def declarations; nothing to evaluate\n` exactly
once on stderr.

#### Scenario: Direct-file empty result

- **WHEN** a direct file outside a Reef package contains only function declarations and is evaluated with `chelis eval --json --file`
- **THEN** the CLI emits the no-evaluable-roots warning exactly once on stderr

#### Scenario: Reef-context empty result

- **WHEN** a file routed through Reef-context evaluation contains only function declarations and is evaluated with `chelis eval --json --file`
- **THEN** the CLI emits the same no-evaluable-roots warning exactly once on stderr

### Requirement: Empty JSON output remains wire-compatible

For a successful result with empty `roots` and empty `transcript`, the CLI MUST
preserve exit status `0` and MUST write exactly `{"roots":[]}\n` to stdout as
one parseable JSON document.

#### Scenario: Direct-file wire compatibility

- **WHEN** direct-file JSON evaluation produces empty roots and an empty transcript
- **THEN** the process exits `0` and stdout bytes equal `{"roots":[]}\n`

#### Scenario: Reef-context wire compatibility

- **WHEN** Reef-context JSON evaluation produces empty roots and an empty transcript
- **THEN** the process exits `0` and stdout bytes equal `{"roots":[]}\n`

### Requirement: The empty-result warning is selective

The CLI MUST NOT emit the no-evaluable-roots warning when an `EvalResult` has
at least one root or at least one transcript entry, and MUST NOT substitute
that warning for an evaluation or serialization error.

#### Scenario: Non-empty direct-file result

- **WHEN** direct-file JSON evaluation succeeds with at least one root
- **THEN** stdout contains the result document and stderr does not contain the no-evaluable-roots warning

#### Scenario: Non-empty Reef-context result

- **WHEN** Reef-context JSON evaluation succeeds with at least one root
- **THEN** stdout contains the result document and stderr does not contain the no-evaluable-roots warning

#### Scenario: Transcript-only result

- **WHEN** JSON evaluation succeeds with no roots and at least one transcript entry
- **THEN** the CLI does not emit the no-evaluable-roots warning

#### Scenario: Evaluation failure

- **WHEN** JSON evaluation fails before producing an `EvalResult`
- **THEN** the process exits nonzero, stdout remains empty, stderr contains the evaluation error, and stderr does not contain the no-evaluable-roots warning
