## ADDED Requirements

### Requirement: Tide denies external evaluator builtins by default
Every unconfigured Tide evaluation SHALL supply explicit deny-external mode. File reads/writes/existence/listing, mapped-file open, and subprocess execution MUST return policy denial before any filesystem or process API is invoked.

#### Scenario: Tide denies a file read
- **WHEN** unconfigured Tide evaluation reaches `read_file`
- **THEN** it returns the structured policy-denied diagnostic and performs no filesystem read

#### Scenario: Tide denies a subprocess
- **WHEN** unconfigured Tide evaluation reaches `process_run`
- **THEN** it returns policy denial and launches no child process

### Requirement: Nested evaluator paths cannot bypass denial
Deny-external mode SHALL propagate through nested functions, callbacks, transforms, imported definitions, and test bodies, and SHALL be checked at the common external-builtin dispatch boundary. One typed host-capable builtin/effect/policy registry SHALL own denial classification; adding or reclassifying an external builtin without a denial entry MUST fail the consistency gate.

#### Scenario: Nested callback remains denied
- **WHEN** a callback invoked from a collection transform reaches a file builtin
- **THEN** the same policy denial occurs before any host action

#### Scenario: Missing propagation fails the acceptance gate
- **WHEN** a fixture invokes an external builtin through a supported call form that does not carry deny-external mode
- **THEN** the denial suite detects the attempted host action or missing policy-denied result and fails

### Requirement: Pure and captured-event evaluation remains available
Deny-external mode SHALL NOT reject pure expressions or deterministic evaluator-local `print` and `debug` capture. Those builtins SHALL return captured events without terminal output or an external host request.

#### Scenario: Pure Tide evaluation succeeds
- **WHEN** an unconfigured Tide evaluation uses no external builtin
- **THEN** it preserves the established value result

#### Scenario: Captured output performs no terminal action
- **WHEN** Tide evaluation reaches only `print` or `debug`
- **THEN** it returns deterministic captured events without terminal, filesystem, or subprocess action
