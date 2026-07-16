## ADDED Requirements

### Requirement: [FCIS-EVIDENCE-001] Active FCIS changes have versioned machine manifests
Every active FCIS change SHALL be registered in a versioned root registry and SHALL own a typed per-change manifest containing stable change/capability IDs, change kind, prerequisites, exact core and adapter boundaries, forbidden capabilities, enforcement layers and blind spots, Dylint diagnostic registrations with detector classes/polarity fixtures and `NoFix` or machine-applicable fix policies/parity fixtures, protocol/identity classification, surface-matrix references, slices, final oracle, rollback boundary, and requirement/scenario/fixture traceability. Unknown fields, unknown enum tags, duplicate IDs, and unsupported schema versions MUST fail closed.

#### Scenario: [FCIS-EVIDENCE-001-P1] Complete manifest is accepted
- **WHEN** an active change supplies every required field with known typed values and valid references
- **THEN** the contract checker normalizes and accepts the manifest

#### Scenario: [FCIS-EVIDENCE-001-N1] Unknown manifest data fails closed
- **WHEN** a manifest contains an unknown capability class, identity classification, field, or schema version
- **THEN** the checker returns a structured manifest error and the owning oracle cannot pass

### Requirement: [FCIS-EVIDENCE-002] Prerequisite and slice graphs are deterministic and acyclic
The checker SHALL construct prerequisite and slice graphs from stable IDs, order independent nodes by bytewise UTF-8 ID, and reject self-edges, unknown nodes, cycles, duplicate edges, and a final oracle that omits an active slice. A focused slice MUST NOT be represented as final completion evidence.

#### Scenario: [FCIS-EVIDENCE-002-P1] Valid prerequisite order is canonical
- **WHEN** equal acyclic prerequisite graphs are supplied in different TOML declaration orders
- **THEN** the checker emits the same canonical execution order

#### Scenario: [FCIS-EVIDENCE-002-N1] Cyclic evidence graph is rejected
- **WHEN** two changes or slices depend on each other directly or transitively
- **THEN** the checker reports the stable cycle path and no affected oracle is runnable as completion evidence

### Requirement: [FCIS-EVIDENCE-003] Requirements and scenarios have complete traceability
Every active requirement SHALL have a repository-unique stable requirement ID and at least one positive and one negative scenario. Every scenario SHALL have a stable ID, explicit polarity, and one or more fixture mappings. Every fixture SHALL name an existing executable test, compile-fail case, architecture fixture, golden vector, or documented manual prerequisite and SHALL belong to exactly one focused slice included by the final oracle. Scenario polarity MUST NOT be inferred from prose titles.

#### Scenario: [FCIS-EVIDENCE-003-P1] Scenario is reachable from the final oracle
- **WHEN** a scenario maps to a fixture, the fixture maps to a focused slice, and the final oracle includes that slice
- **THEN** the checker records a complete scenario-to-oracle trace

#### Scenario: [FCIS-EVIDENCE-003-N1] Missing negative parity fails
- **WHEN** an active requirement has positive coverage but no explicitly negative scenario or mapped negative fixture
- **THEN** the checker reports a coverage-parity failure and the final oracle cannot pass

#### Scenario: [FCIS-EVIDENCE-003-N2] Orphan fixture fails
- **WHEN** a fixture has no scenario owner, names a nonexistent test target/path, or belongs to an unknown slice
- **THEN** the checker reports the orphan and does not count it as evidence

### Requirement: [FCIS-EVIDENCE-004] FCIS commands come from one argv-only registry
`scripts/fcis_gate.py` SHALL accept only oracle and slice IDs from a validated typed command plan. Every command SHALL be represented as an argv array plus a workspace-relative working directory, declared environment policy, and mutation policy of `ReadOnlyCheckout` or `DisposableFixtureCopy`. The runner MUST invoke commands without a shell and MUST reject shell strings, redirections, substitutions, unknown IDs, absolute working directories outside the checkout, and undeclared environment inheritance. A command containing `cargo dylint --fix` MUST use `DisposableFixtureCopy`, name its source fixture and disposable destination, and verify tracked-checkout state before and after; `--fix` in a read-only plan MUST fail validation.

#### Scenario: [FCIS-EVIDENCE-004-P1] Registered argv plan executes
- **WHEN** a caller selects a registered oracle/slice whose command plan contains valid argv arrays
- **THEN** the runner executes those argv arrays in canonical order with `shell=False`

#### Scenario: [FCIS-EVIDENCE-004-N1] Shell command text is rejected
- **WHEN** a manifest or plan contains a pipeline, redirection, command substitution, or free-form shell command string
- **THEN** plan validation fails before any evidence command executes

#### Scenario: [FCIS-EVIDENCE-004-P2] Registered fix runs in a disposable fixture copy
- **WHEN** a validated plan selects a registered machine-applicable Dylint fix test with a declared source fixture, disposable destination, and checkout-state guard
- **THEN** the runner applies `cargo dylint --fix` only inside that disposable copy and verifies the tracked checkout remains unchanged

#### Scenario: [FCIS-EVIDENCE-004-N2] Read-only fix plan is rejected
- **WHEN** a read-only command plan contains `cargo dylint --fix`, omits its disposable destination, or lacks tracked-checkout pre/post verification
- **THEN** plan validation fails before Dylint executes

### Requirement: [FCIS-EVIDENCE-005] Missing or planned evidence fails explicitly
Every active oracle and slice SHALL be registered before domain implementation. A registered command whose test, fixture, checker, or implementation does not exist SHALL return a structured `evidence_not_implemented`, `fixture_missing`, or `command_missing` failure. Missing evidence MUST NOT be silently skipped, treated as not applicable, or reported green.

#### Scenario: [FCIS-EVIDENCE-005-P1] Implemented evidence can pass
- **WHEN** every declared fixture and command exists and returns its expected successful structured result
- **THEN** the owning slice may pass

#### Scenario: [FCIS-EVIDENCE-005-N1] Planned slice remains non-green
- **WHEN** a domain slice is registered but its implementation or fixture is still planned
- **THEN** the runner reports it blocked or failed and never counts it as successful evidence

### Requirement: [FCIS-EVIDENCE-006] Gate reports make contradictory states unrepresentable
The typed report SHALL distinguish `Passed`, `Failed`, and `Blocked`. `Passed` SHALL carry an empty error list; `Failed` and `Blocked` SHALL carry a nonempty error collection. Check records and errors SHALL be canonically ordered by oracle, slice, check ID, error code, and stable detail key. Absolute checkout path, duration, terminal mode, process identity, and localized OS text SHALL be non-semantic report metadata.

#### Scenario: [FCIS-EVIDENCE-006-P1] Equal evidence produces equal reports
- **WHEN** equal normalized contract inputs and check results are assembled under different absolute roots or terminal configurations
- **THEN** the semantic report fields and canonical JSON are byte-equal

#### Scenario: [FCIS-EVIDENCE-006-N1] Passed report with errors is rejected
- **WHEN** an adapter attempts to assemble `Passed` with a nonempty error list
- **THEN** typed construction or report validation rejects the contradictory state

### Requirement: [FCIS-EVIDENCE-007] Architecture evidence is layered and threat-modelled
Every designated FCIS core SHALL declare its resolved dependency allowlist, exact mixed-module manifest where needed, forbidden capability classes, public-interface capability policy, proven negative fixture classes, behavioral evidence classes, and known blind spots. Compiled Rust boundary evidence SHALL consume the independently accepted `establish-dylint-tooling` prerequisite and its pinned `chelis-fcis-boundaries` library for resolved forbidden API/type/macro references, function-item escape, core-to-adapter references, ambient mutable state, and capability-bearing public interfaces. The contract oracle SHALL execute `.venv/bin/python scripts/dylint_gate.py` against the same revision and SHALL verify its `Passed`/empty-error report, schema version, diagnostic/detector/fix registry, exact pins, and root-stable isolation before domain evidence can run. This change MUST NOT create, duplicate, or repin the Dylint library. Domain Dylint policy SHALL be projected from validated FCIS manifests into the prerequisite's strict schema through the command plan's declared `DYLINT_TOML` value rather than maintained as a second authoritative capability list. Every invocation SHALL reference only accepted diagnostic/classes/fix IDs, declare package, target, feature, and configuration lanes, emit machine-readable diagnostics, and fail on an unresolved configured entry, an unmatched exception, a missing declared lane, or zero matched production-core items. The harness MAY additionally use dependency checks, `forbid(unsafe_code)`, compile-fail/UI fixtures, and source checks, but MUST NOT describe lexical scanning, Dylint, or bounded fixtures as a complete transitive purity proof. Active-cfg coverage, procedural-macro/build-script implementation effects, arbitrary dynamic dispatch, and precompiled dependency behavior SHALL remain explicit blind spots unless independently covered.

#### Scenario: [FCIS-EVIDENCE-007-P1] Layered architecture claim is accepted
- **WHEN** the current-revision Dylint prerequisite passes and a change declares its actual dependency/module layers, accepted Dylint compilation matrix, resolved entries, matched production-core evidence, fixtures, threat model, and blind spots without overstating them
- **THEN** the checker accepts the architecture evidence declaration for execution

#### Scenario: [FCIS-EVIDENCE-007-P2] Resolved Dylint fixture detects a forbidden reference
- **WHEN** a declared compilation lane contains a fixture-registered direct, aliased, re-exported, qualified, function-item, callback, trait, declarative-macro-expanded, or adapter reference to a forbidden capability
- **THEN** the accepted Dylint prerequisite library emits the registered stable diagnostic class and the architecture check is non-green

#### Scenario: [FCIS-EVIDENCE-007-N1] False completeness claim fails
- **WHEN** a change claims complete detection of arbitrary macro or build-script implementation effects, dynamic dispatch, transitive side effects, unexecuted cfg/target/feature code, precompiled dependency behavior, or future Rust syntax from lexical or Dylint fixtures
- **THEN** the checker rejects the threat-model declaration

#### Scenario: [FCIS-EVIDENCE-007-N2] Vacuous or stale Dylint configuration fails
- **WHEN** a Dylint plan cannot resolve a configured forbidden entry, matches no production-core item, leaves an exception unmatched, omits a declared compilation lane, references an unknown diagnostic/class/fix ID, or consumes policy that differs from the validated FCIS manifest projection
- **THEN** the architecture check returns a structured non-green contract error before the result can count as boundary evidence

#### Scenario: [FCIS-EVIDENCE-007-N3] Dylint prerequisite mismatch fails
- **WHEN** the standalone prerequisite is absent, stale, failed, blocked, or reports a schema, registry, pin, or root-isolation state different from the accepted contract
- **THEN** `fcis-contract` fails before projecting or executing domain Dylint policy

### Requirement: [FCIS-EVIDENCE-008] Machine-facing tables have one typed owner
Every FCIS table that controls behavior or acceptance coverage—including evaluator host-builtin classification, lint rule registration, compiler surface/target support, proof engine authorization, Reef action/lock/report vocabularies, and shipped identity domains—SHALL have one typed owner. Markdown and secondary code projections SHALL be generated from that owner or tripwire-checked against it. Independently maintained string lists MUST NOT jointly claim authority.

#### Scenario: [FCIS-EVIDENCE-008-P1] Projection matches typed owner
- **WHEN** a documented capability matrix and every consuming projection match the typed registry
- **THEN** the tripwire passes and the acceptance harness derives its matrix from that owner

#### Scenario: [FCIS-EVIDENCE-008-N1] Registry drift is detected
- **WHEN** a builtin, rule, target pair, engine, action, lock rank, report state, or identity domain changes in only one projection
- **THEN** the tripwire fails and identifies the missing or conflicting projection

### Requirement: [FCIS-EVIDENCE-009] Prerequisite evidence is fresh and independently owned
A child FCIS final oracle SHALL execute or transitively include its prerequisite's regression suite against the same working revision. This includes `fcis-contract` executing the independently owned `dylint-tooling` oracle before domain registration/evidence and the domain final oracles executing their security prerequisites. A stale handwritten completion record MUST NOT satisfy a prerequisite. The prerequisite retains its own independent final oracle and completion claim; a child oracle MUST NOT re-claim that prerequisite as its own completion.

#### Scenario: [FCIS-EVIDENCE-009-P1] Child runs current prerequisite regression
- **WHEN** `fcis-contract`, `proof-execution`, `evaluator-effects`, or `reef-workflows` runs against a revision whose prerequisite suite passes
- **THEN** the child report records current prerequisite regression evidence and continues to its own checks

#### Scenario: [FCIS-EVIDENCE-009-N1] Stale prerequisite record is insufficient
- **WHEN** a prior revision has a recorded green prerequisite but the current revision's prerequisite regression fails or is absent
- **THEN** the child oracle is blocked or failed and cannot rely on the stale record

### Requirement: [FCIS-EVIDENCE-010] Shared evidence does not become a shared production effect framework
The FCIS evidence crate and runner MAY share manifest parsing, graph validation, canonical identity test helpers, architecture fixture execution, and report types. Compiler, evaluator, proof, lint, Reef, and conformance production code SHALL retain domain-owned decision, request, observation, policy, plan, and failure types and MUST NOT depend on a generic production workflow/effect abstraction introduced by this change.

#### Scenario: [FCIS-EVIDENCE-010-P1] Domains use shared test mechanics only
- **WHEN** two domain changes use the same fixture runner or canonical-encoding golden harness
- **THEN** their production crates still expose separate domain-owned algebras

#### Scenario: [FCIS-EVIDENCE-010-N1] Generic production protocol dependency is rejected
- **WHEN** a domain core imports a shared generic FCIS request, observation, continuation, decision, or workflow runtime from the evidence crate
- **THEN** the dependency/architecture gate fails

### Requirement: [FCIS-EVIDENCE-011] FCIS Dylint uses bind accepted detector and fix evidence
Every Dylint diagnostic used by an FCIS oracle SHALL reference a production-loadable stable diagnostic ID and supported syntax classes from the accepted `establish-dylint-tooling` registry. Its domain registration SHALL name at least one allowed-positive fixture, at least one violating-negative fixture, and the exact expected diagnostic class for every claimed domain use. Each claimed alias, re-export, qualified, function-item, callback, trait, macro, adapter-reference, ambient-state, public-interface, and test-classification form SHALL have its own positive/negative evidence before that claim can count. Missing, stale, unsupported, or one-polarity evidence MUST remain non-green. Detector fixtures prove only their registered forms and MUST NOT be represented as arbitrary Rust completeness.

Each domain registration SHALL declare `FixPolicy::NoFix` or reference a prerequisite-supported stable `FixPolicy::MachineApplicable` fix ID plus disposable positive fix fixtures, negative no-machine-fix fixtures, and a domain parity plan. Because the initial prerequisite registry is entirely `NoFix`, no machine-applicable domain registration is valid until a separately accepted Dylint-tooling change adds that fix and proves exact expected rewrite, successful canonical formatting and compilation, a clean rerun of the same lint, required behavioral/parity evidence, and second-run idempotence. Negative fix fixtures SHALL prove that ambiguous or unsafe cases emit no machine-applicable suggestion. An FCIS fix plan MUST NOT add lint suppression, edit FCIS manifests/configuration, widen a boundary or exception, move code outside the checked boundary, or delete tests/evidence. Acceptance oracles SHALL run supported fixes only in disposable fixture copies and SHALL fail if tracked checkout state changes; explicitly invoked developer fix commands MAY mutate the working tree.

#### Scenario: [FCIS-EVIDENCE-011-P1] Detector contract passes both polarities
- **WHEN** every claimed syntax class has an allowed fixture that remains clean and a violating fixture that emits the exact registered diagnostic under the declared Dylint lane
- **THEN** the detector contract may count as architecture evidence for only those fixture-proven forms

#### Scenario: [FCIS-EVIDENCE-011-N1] One-polarity detector evidence is rejected
- **WHEN** a Dylint diagnostic has only violating fixtures, only allowed fixtures, a stale expected diagnostic, or a claimed syntax class without both polarities
- **THEN** the architecture slice remains non-green and the lint cannot contribute acceptance evidence

#### Scenario: [FCIS-EVIDENCE-011-P2] Registered machine fix is safe and idempotent
- **WHEN** a detector-green prerequisite-supported fix's disposable positive fixture is run with `cargo dylint --fix`
- **THEN** the result equals the registered expected rewrite, formats and compiles, reruns lint-clean, passes its domain parity plan, remains unchanged on a second fix run, and leaves the tracked checkout unchanged

#### Scenario: [FCIS-EVIDENCE-011-N2] Unsafe or unregistered fix is not applied
- **WHEN** a lint has `NoFix`, references an unknown or prerequisite-unsupported fix ID, lacks required fix evidence, encounters an ambiguous negative fix fixture, or proposes suppression, policy/manifest mutation, boundary escape, or evidence deletion
- **THEN** it emits no machine-applicable suggestion and any FCIS command attempting to apply that fix fails non-green

#### Scenario: [FCIS-EVIDENCE-011-N3] Acceptance fix execution cannot mutate the checkout
- **WHEN** an acceptance command exercises registered fixes
- **THEN** it runs them against disposable fixture copies and fails if any tracked checkout file differs afterward
