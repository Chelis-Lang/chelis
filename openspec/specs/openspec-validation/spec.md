# OpenSpec Validation

## Purpose

Initial, non-blocking adoption of an OpenSpec planning tree for Chelis: structural validation of `openspec/` in CI and locally, with no governed-change enforcement and no coupling to `spec/**`.

## Requirements

### Requirement: Canonical OpenSpec planning tree
Chelis SHALL maintain a canonical `openspec/` planning tree using the built-in spec-driven schema, and its specifications and changes SHALL pass `openspec validate --all --strict`.

#### Scenario: Well-formed tree validates
- **WHEN** the `openspec/` tree contains only schema-valid specifications and changes
- **THEN** `openspec validate --all --strict` SHALL succeed

#### Scenario: Malformed artifact is rejected
- **WHEN** a specification or change under `openspec/` violates the spec-driven schema
- **THEN** `openspec validate --all --strict` SHALL fail and the CI validation job SHALL report the failure

### Requirement: OpenSpec validation is advisory
Chelis SHALL run OpenSpec structural validation in a dedicated CI workflow with only `contents: read`, triggered by changes under `openspec/`. The workflow SHALL NOT be a required status check, SHALL NOT gate merges, and SHALL NOT govern changes under `spec/**`. The governed-change enforcement described in `spec/design/spec_provenance.md` Phase 0 SHALL remain inactive.

#### Scenario: Non-openspec change is unaffected
- **WHEN** a pull request changes no path under `openspec/`
- **THEN** OpenSpec validation SHALL NOT block or fail that pull request

#### Scenario: Spec edit is not governed by validation
- **WHEN** a change edits a path under `spec/**` without any OpenSpec lifecycle
- **THEN** OpenSpec validation SHALL NOT fail or block the change on that basis

### Requirement: OpenSpec stays planning evidence
OpenSpec artifacts SHALL remain planning and authoring evidence only. They SHALL NOT become Chelis runtime, specification, coverage, or implementation-correctness authority, and SHALL NOT override `spec/**` or an executable acceptance oracle.

#### Scenario: Validation authorizes only structural validity
- **WHEN** `openspec validate` succeeds for a change
- **THEN** the result SHALL authorize only that the artifacts are schema-valid and SHALL NOT imply implementation correctness

#### Scenario: OpenSpec conflicts with owning authority
- **WHEN** an OpenSpec artifact contradicts `spec/**` or an executable oracle
- **THEN** the owning authority SHALL remain controlling and the OpenSpec artifact SHALL be corrected
