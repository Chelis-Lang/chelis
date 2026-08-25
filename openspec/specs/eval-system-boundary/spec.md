# Evaluator System Boundary Specification

## Purpose

Define one evaluator-only boundary for filesystem and subprocess access without changing Chelis language effects or compiled-runtime behavior.

## Requirements

### Requirement: One evaluator system boundary

Every compiler API evaluator filesystem and subprocess operation SHALL pass through one injected system boundary.

The boundary SHALL cover `read_file`, `write_file`, `read_lines`, `read_bytes`, `file_exists`, `list_dir`, `mmap_file`, and `process_run`.

`print` and `debug` SHALL remain evaluator transcript operations. They SHALL NOT become operating-system console operations under this capability.

#### Scenario: Filesystem operation uses the boundary

- **WHEN** an evaluator program calls any covered filesystem builtin
- **THEN** the evaluator sends that operation to the injected system boundary exactly once

#### Scenario: Subprocess operation uses the boundary

- **WHEN** an evaluator program calls `process_run`
- **THEN** the evaluator sends the program and argument vector to the injected system boundary exactly once

#### Scenario: Transcript operation remains internal

- **WHEN** an evaluator program calls `print` or `debug`
- **THEN** the evaluator records the rendered value without an operating-system console call

### Requirement: Typed evaluator capability policy

The evaluator boundary SHALL classify each covered operation as `Filesystem` or `Process`.

A policy SHALL decide permission from this typed capability before the system adapter receives the operation.

A refused operation SHALL return an error that names the operation and capability. The evaluator SHALL NOT execute or silently replace that operation.

Program evaluation entry points SHALL use a policy that permits both capabilities.

Invariant predicate evaluation SHALL refuse both capabilities. This rule enforces the effect prohibition in `spec/04-type-system.md` §2.5.1.

#### Scenario: Permissive policy executes an operation

- **WHEN** a program evaluation entry point executes a covered operation
- **THEN** the default adapter executes that operation under the contract below

#### Scenario: Filesystem capability is refused

- **WHEN** a policy refuses `Filesystem` and the program calls `read_file`
- **THEN** evaluation returns the capability error before the adapter reads the path

#### Scenario: Process capability is refused

- **WHEN** a policy refuses `Process` and the program calls `process_run`
- **THEN** evaluation returns the capability error before the adapter starts a process

### Requirement: Default adapter contract

`read_file` SHALL read the complete file as valid UTF-8 text. `write_file` SHALL create or truncate the file and write the string bytes.

`read_lines` SHALL split valid UTF-8 text at LF and CRLF endings. It SHALL remove each ending from the returned item.

A final LF or CRLF SHALL NOT create an extra empty item.

`read_bytes` SHALL return each file byte as an integer from 0 through 255. `mmap_file` SHALL load all bytes into a mapped-file runtime value.

`file_exists` SHALL return false for an absent path or a metadata access error.

`list_dir` SHALL return lossy Unicode basenames in operating-system enumeration order. It SHALL NOT sort them.

`list_dir` SHALL stop at the first directory-open or directory-entry error. Both error paths SHALL use the same template below.

`process_run` SHALL pass the program and arguments directly to the operating system without a shell. It SHALL capture both output streams.

A missing process exit status SHALL become `-1`. Both output streams SHALL use lossy UTF-8 decoding.

The adapter SHALL use these exact system-error templates:

| Operation | Error template |
|---|---|
| `read_file` | ``read_file failed for `{path}`: {source}`` |
| `write_file` | ``write_file failed for `{path}`: {source}`` |
| `read_lines` | ``read_lines failed for `{path}`: {source}`` |
| `read_bytes` | ``read_bytes failed for `{path}`: {source}`` |
| `list_dir` | ``list_dir failed for `{path}`: {source}`` |
| `mmap_file` | ``mmap_file failed for `{path}`: {source}`` |
| `process_run` | ``process_run failed to spawn `{program}`: {source}`` |

`{source}` SHALL preserve the underlying system error display without replacement or reclassification.

#### Scenario: Default filesystem adapter succeeds

- **WHEN** a covered filesystem operation succeeds through the default adapter
- **THEN** the evaluator returns the value defined by this requirement

#### Scenario: Default filesystem adapter fails

- **WHEN** a covered filesystem operation fails through the default adapter
- **THEN** the evaluator returns the applicable exact error template and underlying system error display

#### Scenario: Default process adapter completes

- **WHEN** `process_run` completes through the default adapter
- **THEN** the evaluator applies the exit-status and output-decoding rules from this requirement

### Requirement: Evaluator-only assurance scope

This boundary SHALL govern only system operations from the compiler API evaluator.

It SHALL NOT claim mediation for generated C, HIP, or Metal artifacts. It SHALL NOT claim mediation for the `chelis-runtime` C ABI.

The language-visible `IO` semantics remain under `spec/04-type-system.md` §7. The `process_run` contract remains under `spec/05-risc-primitives.md` §2.6.

#### Scenario: Compiled filesystem builtin remains separate

- **WHEN** a supported filesystem builtin reaches a compiled backend
- **THEN** its existing backend and runtime path remains unchanged by this capability

### Requirement: System access remains isolated

Evaluator runtime code outside the selected system adapter SHALL NOT invoke filesystem, path-existence, or process APIs directly.

Every evaluator context SHALL receive a policy boundary before it can evaluate an expression.

#### Scenario: Evaluator dispatch needs file access

- **WHEN** evaluator dispatch needs filesystem data or metadata
- **THEN** it calls the injected boundary instead of a filesystem or path API

#### Scenario: Evaluator dispatch needs process access

- **WHEN** evaluator dispatch needs to start a process
- **THEN** it calls the injected boundary instead of a process API

#### Scenario: Invariant evaluator context is constructed

- **WHEN** the decoder constructs a context for invariant predicate evaluation
- **THEN** that context receives the deny-all policy boundary before expression evaluation
