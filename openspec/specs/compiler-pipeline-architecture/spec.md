# Compiler Pipeline Architecture Specification

## Purpose

This specification defines ownership and state boundaries for the shared compiler pipeline.

It records architecture only. The language, type, serialization, and backend specifications remain authoritative for behavior.

## Requirements

This capability defines implementation architecture only. The pass order remains under [`spec/04-type-system.md`](../../../spec/04-type-system.md), and wire compatibility remains under [`spec/10-serialization.md`](../../../spec/10-serialization.md).

Backend separation remains under [`spec/08-backends.md`](../../../spec/08-backends.md). These requirements do not change language or backend semantics.

### Requirement: Canonical production pipeline owner

`chelis-compiler-api` SHALL own production orchestration for source preparation, type analysis, effect checks, linearity checks, and optional DAG lowering.

Production consumers that need two or more semantic stages SHALL delegate to this owner. Lower-layer crates SHALL retain their individual stage implementations.

#### Scenario: Production consumer delegates

- **WHEN** a CLI, edit, compiler-API, or E2E path needs a full semantic check
- **THEN** the path requests a compiler-API pipeline goal instead of calling the stages in sequence

#### Scenario: Production consumer duplicates the sequence

- **WHEN** a production file outside the owner orchestrates two or more canonical semantic stages
- **THEN** the source architecture guard rejects the file and identifies the duplicated stage calls

### Requirement: Closed pipeline goals and fixed pass order

The pipeline SHALL provide closed goals for type analysis, full semantic checks, and lowering. Each goal SHALL run only its required prefix.

The pipeline SHALL preserve the pass order from `spec/04-type-system.md`. A consumer SHALL NOT omit, repeat, or reorder a required stage.

#### Scenario: Full check runs the required order

- **WHEN** a consumer requests a full semantic check for a valid source
- **THEN** the pipeline runs type analysis, effect checks, and linearity checks in that order

#### Scenario: Type rejection stops later stages

- **WHEN** type analysis rejects a source
- **THEN** the pipeline returns the type rejection without an effect, linearity, or lowering result

### Requirement: Legal phase states by construction

The pipeline SHALL represent type rejection, accepted type analysis, complete semantic success, and lowered success with distinct typed variants.

`CheckedCompilation` SHALL exist only after type, effect, and linearity success. `LoweredCompilation` SHALL contain a valid `CheckedCompilation` and a DAG.

A rejected variant SHALL NOT expose a checked program or DAG. The implementation SHALL NOT model these states with unrelated optional fields.

#### Scenario: Valid source reaches lowered state

- **WHEN** a valid source requests lowering and every required stage accepts it
- **THEN** the result contains a `LoweredCompilation` with its checked compilation and DAG

#### Scenario: Rejected source cannot expose a success product

- **WHEN** a compile-fail fixture tries to obtain a checked program or DAG from a rejection
- **THEN** the fixture fails to compile because that rejection variant has no success product

### Requirement: Single type-inference product per selected path

Each selected monolithic semantic path SHALL run one full type-inference session. Fitness data and the accepted checked program SHALL derive from that session.

A contextual cache load and its documented monolithic error fallback SHALL count as separate selected paths. The fallback SHALL remain explicit.

#### Scenario: Accepted analysis runs once

- **WHEN** an instrumented valid program requests full semantic checks
- **THEN** the type-inference session counter records one session and both products report the same inference statistics

#### Scenario: Duplicate inference is introduced

- **WHEN** an instrumented adapter calls both the fitness and typed-check compatibility functions for one selected path
- **THEN** the parity fixture records two sessions and fails the single-product requirement

### Requirement: Compatibility at consumer boundaries

Each migrated consumer SHALL preserve its current public behavior. This parity includes diagnostics, fitness, inferred signatures, root metadata, JSON bytes, and exit codes.

The internal pipeline types SHALL NOT implement Serde wire traits. Existing compiler-API schema types SHALL remain the machine-facing models.

#### Scenario: Accepted consumers retain output

- **WHEN** baseline accepted fixtures run through compiler API, CLI, edit, and E2E adapters
- **THEN** each adapter returns the same public value, bytes, and exit status as its frozen baseline

#### Scenario: Rejected consumers retain diagnostics

- **WHEN** baseline type, effect, linearity, or lowering failures run through each applicable adapter
- **THEN** each adapter preserves diagnostic stage, kind, text, order, severity, span, bytes, and exit status

### Requirement: Context and cache parity

Monolithic and contextual modes SHALL use the same semantic state transitions. The layered Reef path SHALL preserve its clean cache path and monolithic error fallback.

Fitness formulas SHALL have one implementation owner. Layered code SHALL NOT copy fitness weights or clean-report formulas.

#### Scenario: Warm contextual check matches cold check

- **WHEN** a valid Reef program runs through cold, warm, and cache-disabled checks
- **THEN** all three modes produce byte-identical public output

#### Scenario: Contextual check rejects new code

- **WHEN** non-stdlib code fails a semantic stage against a valid cached library context
- **THEN** the documented fallback produces the same rejection as the monolithic path

### Requirement: Canonical root metadata

The pipeline SHALL derive root names, tuple root suffixes, tensor roots, and DAG node mappings once from the checked program.

Compiler API and E2E consumers SHALL use this metadata instead of independent Surf-tree reconstruction.

#### Scenario: Tuple roots retain canonical names

- **WHEN** a valid program lowers a tuple-valued root
- **THEN** compiler API and E2E results expose equal ordered names and node mappings for every tuple element

#### Scenario: Independent root reconstruction diverges

- **WHEN** a negative fixture supplies root metadata that differs from the checked-program derivation
- **THEN** the parity fixture rejects the independent metadata before it becomes a consumer result

### Requirement: Policy and backend boundaries remain separate

Reef preparation, style policy, report presentation, exit-code selection, and backend target selection SHALL remain outside the semantic pipeline.

Backend emitters SHALL remain final target-specific correctness boundaries. The pipeline SHALL propagate lowering failures and SHALL NOT fabricate backend success.

#### Scenario: Backend accepts a lowered program

- **WHEN** the semantic pipeline returns a lowered program and the selected backend supports it
- **THEN** the backend emits its target-specific artifacts through its existing interface

#### Scenario: Backend rejects a target-specific case

- **WHEN** the semantic pipeline returns a lowered program that the selected backend does not support
- **THEN** the backend returns its existing target-specific diagnostic and the pipeline does not replace it with a placeholder

### Requirement: Authoritative pipeline oracle

The change SHALL provide `.venv/bin/python scripts/compiler_pipeline_oracle.py` as its authoritative acceptance oracle.

The oracle SHALL run positive and negative parity suites for compiler API, CLI, edit, E2E, cache, root metadata, and the source guard.

#### Scenario: Shared pipeline satisfies all contracts

- **WHEN** the authoritative oracle runs after every production consumer delegates
- **THEN** every parity suite and the actual-workspace source guard pass

#### Scenario: A duplicate production pipeline is planted

- **WHEN** the source-guard negative fixture adds a second semantic sequence to an in-memory production inventory
- **THEN** the authoritative oracle fails and names the duplicated stages and source path
