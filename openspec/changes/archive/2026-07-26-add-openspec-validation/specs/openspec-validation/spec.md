## ADDED Requirements

### Requirement: Canonical OpenSpec planning tree
Chelis SHALL maintain a canonical `openspec/` planning tree with the built-in spec-driven schema. Specifications and active changes SHALL pass `openspec validate --all --strict`.

#### Scenario: Well-formed tree validates
- **WHEN** the `openspec/` tree contains only schema-valid specifications and active changes
- **THEN** `openspec validate --all --strict` SHALL succeed

#### Scenario: Malformed active artifact produces an advisory finding
- **WHEN** a specification or active change violates the spec-driven schema
- **THEN** the consumer checker SHALL exit `1`
- **THEN** the central action SHALL emit a warning and the advisory job SHALL succeed

#### Scenario: Archived changes are outside validation scope
- **WHEN** `openspec validate --all --strict` runs
- **THEN** it SHALL enumerate only specifications and active changes
- **THEN** it SHALL NOT enumerate artifacts under `openspec/changes/archive/`

### Requirement: OpenSpec validation is advisory
Chelis SHALL run structural validation with the pinned central action in advisory mode. Schema findings SHALL produce warnings. Operational failures SHALL remain nonzero.

The consumer checker SHALL perform structural validation only. It SHALL NOT enforce lifecycle, citation, planning-order, branch-scope, or `spec/**` policy.

#### Scenario: Operational failure stays nonzero
- **WHEN** the OpenSpec process cannot start, times out, or returns an operational exit code
- **THEN** the consumer checker SHALL exit `2`
- **THEN** the central action SHALL fail in advisory mode

#### Scenario: Non-OpenSpec change is unaffected
- **WHEN** a pull request changes no OpenSpec, validation workflow, or consumer checker path
- **THEN** the OpenSpec validation workflow SHALL NOT run

#### Scenario: Specification edit is not governed by validation
- **WHEN** a change edits `spec/**` without an OpenSpec lifecycle
- **THEN** OpenSpec validation SHALL NOT fail or block the change on that basis

### Requirement: OpenSpec stays planning evidence
OpenSpec artifacts SHALL remain planning and authoring evidence inside the authority boundary in `spec/design/spec_provenance.md` § OpenSpec boundary. That section remains controlling.

#### Scenario: Validation authorizes only structural validity
- **WHEN** `openspec validate` succeeds for a change
- **THEN** the result SHALL authorize only schema validity
- **THEN** the result SHALL NOT imply implementation correctness

#### Scenario: OpenSpec conflicts with owning authority
- **WHEN** an OpenSpec artifact contradicts `spec/**` or an executable oracle
- **THEN** the owning authority SHALL remain controlling and the OpenSpec artifact SHALL be corrected
