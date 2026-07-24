# Spec-Driven Change Governance

## Purpose

Classification, lifecycle, artifact, review-readiness, implementation-citation, completion, synchronization, archival, and maintenance-exemption requirements for Chelis changes.

## Requirements

### Requirement: Phase 0 activation is evidence-gated
Chelis SHALL treat OpenSpec governance as inactive until the repository configuration, contributor and agent instructions, review routing, validation suite, and pull-request gate are merged from a branch whose named `openspec_adoption` and PR-gate oracles are green. Activation SHALL govern planning and review only and SHALL NOT claim compiler, runtime, language, numerical, provenance, coverage, or implementation correctness.

#### Scenario: Adoption surface is incomplete
- **WHEN** any Phase 0 component or named oracle is absent or failing
- **THEN** repository documentation and automation SHALL describe OpenSpec as pending rather than enforced

#### Scenario: Adoption branch is accepted
- **WHEN** the complete adoption branch is archived, its deterministic checks pass, and its exact hosted PR gate is green
- **THEN** merging that branch SHALL activate the OpenSpec planning workflow for subsequent governed changes

### Requirement: Governed changes are classified before implementation
Every significant change subject to Phase 0 SHALL use OpenSpec, including every agent-authored feature or behavior change and every change to `spec/**`. Significant scope includes externally observable behavior, language or compiler architecture, public or machine-facing contracts, algorithms or supported domains, dependency strategy, data or interchange formats, assurance or security posture, and operational behavior. A nonnormative documentation, formatting, typo, or behavior-preserving maintenance change outside `spec/**` MAY use the repository-owned maintenance exemption path instead.

#### Scenario: Significant work is proposed
- **WHEN** work changes a governed category
- **THEN** the contributor SHALL create or update one OpenSpec change before implementation begins

#### Scenario: Normative Markdown changes
- **WHEN** a branch changes any path under `spec/**`
- **THEN** the branch SHALL use an OpenSpec lifecycle even if every changed file is Markdown
- **THEN** documentation-only detection SHALL NOT exempt the branch

#### Scenario: Nonnormative maintenance remains behavior-preserving
- **WHEN** every changed non-governance path is outside `spec/**` and the work is editorial or behavior-preserving
- **THEN** the branch MAY add one `openspec/exemptions/YYYY-MM-DD-<kebab-case-id>.toml` manifest instead of a full lifecycle

### Requirement: Maintenance exemptions are exact and repository-owned
A maintenance exemption SHALL be a newly added regular non-symlink TOML file containing exactly `kind = "maintenance"`, a non-empty `reason`, and a `paths` array that equals the complete set of changed non-governance repository paths. Exemptions SHALL NOT cover `spec/**`, another governance path, behavior changes, or independently reviewable work. A label, review, or other provider state SHALL NOT replace the committed exemption.

#### Scenario: Exemption exactly classifies maintenance
- **WHEN** one valid manifest names every changed non-governance path and none are normative or behavior-changing
- **THEN** merge-bound governance SHALL accept the maintenance classification

#### Scenario: Exempt scope expands
- **WHEN** a branch adds an undeclared path, touches `spec/**`, or becomes behavior-changing
- **THEN** the exemption SHALL fail
- **THEN** significant implementation SHALL stop until the branch carries an apply-ready OpenSpec lifecycle

### Requirement: Planning precedes implementation and enters review visibly
A governed branch SHALL use one unique lowercase kebab-case change identifier. Before its first production-code commit, the branch SHALL contain a planning-only ancestor commit that adds the lifecycle marker, proposal, and requirement deltas. Those artifacts SHALL identify affected owning specifications and positive and negative scenarios. The implementation pull request SHALL cite the identifier using the repository-defined OpenSpec change field so reviewers can enter the plan into the human review queue before evaluating implementation.

#### Scenario: Planning commit precedes production work
- **WHEN** the first production path is changed
- **THEN** an ancestor branch commit SHALL already contain the new lifecycle marker, proposal, and required delta specifications

#### Scenario: Planning and implementation are collapsed
- **WHEN** the lifecycle evidence first appears in the same commit as production changes or after them
- **THEN** the governance gate SHALL fail the ordering requirement

#### Scenario: Pull request omits its change citation
- **WHEN** a governed pull request does not cite its exact OpenSpec change identifier
- **THEN** the PR gate SHALL fail even if local artifact validation succeeds

### Requirement: Planning artifacts stay coherent and addressable
Each governed change SHALL identify every affected capability, include one delta specification per capability, record at least one positive and one negative WHEN/THEN scenario for each changed behavior, explain design and authority boundaries, and maintain an evidence-ordered task list. Proposal, delta specifications, design, and tasks SHALL remain mutually consistent. An implementation agent SHALL name the active change and the specific requirement or design section being implemented.

#### Scenario: Apply-ready plan is reviewed
- **WHEN** proposal capabilities match delta specs, design resolves their implementation constraints, tasks derive from them, and strict validation succeeds
- **THEN** the plan MAY proceed through human review toward implementation

#### Scenario: Artifact or citation drifts
- **WHEN** a capability, requirement, scenario, design decision, task, or implementation citation is missing or contradicts another planning artifact
- **THEN** governance validation SHALL fail or implementation SHALL pause until the artifacts are reconciled

### Requirement: Change branches remain isolated
A governed non-empty branch SHALL add exactly one lifecycle or exactly one maintenance exemption. It SHALL NOT combine multiple lifecycle identifiers, combine a lifecycle with an exemption, modify a lifecycle inherited from the comparison base, or hide lifecycle evidence in malformed or dot-prefixed paths. Independently reviewable work SHALL move to another branch and change.

#### Scenario: One lifecycle owns branch scope
- **WHEN** all significant branch changes implement one newly added lifecycle
- **THEN** branch-scope validation SHALL accept the lifecycle count independently of branch naming

#### Scenario: Branch mixes governance records
- **WHEN** a diff includes multiple lifecycle identifiers, an inherited lifecycle mutation, or both a lifecycle and exemption
- **THEN** branch-scope validation SHALL fail

### Requirement: Completion synchronizes and archives the lifecycle
A governed change SHALL be complete only after every required artifact is done, every task is checked from observed evidence, strict OpenSpec validation passes, the named Chelis acceptance oracle is green, delta specifications are synchronized into baseline `openspec/specs/`, and the complete lifecycle is moved to `openspec/changes/archive/YYYY-MM-DD-<change-id>`. Merge-bound validation SHALL reject active lifecycles, unchecked or undescribed tasks, malformed archive dates, and baseline requirements that diverge from replaying the archived delta against the comparison base.

#### Scenario: Completed lifecycle reaches merge validation
- **WHEN** implementation evidence is accepted and the synchronized archive exactly represents the change
- **THEN** merge-bound validation SHALL accept the lifecycle state

#### Scenario: Active or unsynchronized lifecycle reaches merge validation
- **WHEN** any lifecycle remains active, any task is incomplete, or the archived delta does not reproduce the baseline specification
- **THEN** merge-bound validation SHALL fail

### Requirement: OpenSpec remains planning evidence
OpenSpec artifacts and provider metadata SHALL NOT become Chelis runtime authority, atom lifecycle or identity, freshness or coverage evidence, canonical graph input, implementation-correctness proof, or a substitute for repository-owned specifications, executable acceptance oracles, and approval records. Existing owning specifications SHALL be linked rather than duplicated or weakened in OpenSpec.

#### Scenario: Planning validation succeeds
- **WHEN** a change passes every OpenSpec governance check
- **THEN** the result SHALL authorize only the planned scope and SHALL NOT imply that implementation behavior is correct

#### Scenario: OpenSpec conflicts with owning authority
- **WHEN** an OpenSpec artifact contradicts `spec/**`, an executable oracle, or a future pinned Buoy authority
- **THEN** the owning authority SHALL remain controlling and the OpenSpec artifacts SHALL be corrected before acceptance
