## ADDED Requirements

### Requirement: [DYLINT-TOOLING-001] Dylint tooling is isolated and exactly pinned
The repository SHALL contain a nested Cargo workspace at `tools/dylint` that is not a member or dependency of the stable root workspace. It SHALL pin `cargo-dylint = 6.0.1`, `dylint-link = 6.0.1`, `dylint_linting = 6.0.1`, `dylint_testing = 6.0.1`, and `nightly-2026-04-16` with `rustc-dev` and `llvm-tools-preview`, and SHALL lock all remaining tooling dependencies. Acceptance MUST NOT download, upgrade, or silently substitute tooling. The root toolchain and ordinary workspace gate SHALL remain stable-only.

#### Scenario: [DYLINT-TOOLING-001-P1] Exact isolated toolchain is accepted
- **WHEN** the nested manifests, lockfile, pin registry, selected `cargo dylint` executable, dated nightly, and components all match and the root workspace has no dependency edge to the lint library
- **THEN** the tooling-isolation check passes and the nested workspace may compile with its pinned nightly

#### Scenario: [DYLINT-TOOLING-001-N1] Pin drift or root contamination fails
- **WHEN** a pin floats or disagrees, a required component is absent, the selected executable has another version, the root toolchain changes from stable, or a root production crate depends on the Dylint workspace
- **THEN** the oracle returns `Failed` or `Blocked` with a structured reason and no lint result counts as acceptance evidence

### Requirement: [DYLINT-TOOLING-002] Category and diagnostic ownership is typed and unique
The initial `tools/dylint/fcis-boundaries` category library, package `chelis-fcis-boundaries`, SHALL own one typed registry for configuration schema version, stable diagnostic IDs, rustc lint names, supported detector classes, production loadability, and fix policy. FCIS capability, adapter-reference, ambient-state, and capability-bearing-interface diagnostics with the same policy owner and compilation/evidence lifecycle SHALL remain passes in that library. A sibling category SHALL name a distinct owner or lifecycle and MUST NOT duplicate or fork FCIS boundary policy.

#### Scenario: [DYLINT-TOOLING-002-P1] Same-owner passes share the category library
- **WHEN** the initial registered boundary diagnostics have unique IDs and lint names, one schema owner, and the same compilation/evidence lifecycle
- **THEN** registry validation accepts them as passes in `chelis-fcis-boundaries`

#### Scenario: [DYLINT-TOOLING-002-N1] Duplicate or unjustified category ownership is rejected
- **WHEN** a diagnostic ID or lint name is duplicated, a pass lacks registry metadata, or another library copies FCIS boundary policy without a distinct declared owner or lifecycle
- **THEN** registry validation fails before any category can be loaded as accepted evidence

### Requirement: [DYLINT-TOOLING-003] Boundary configuration is strict, versioned, and policy-neutral
The Dylint library SHALL accept only its strict `schema_version = 1` configuration from the declared `DYLINT_TOML` path. The schema SHALL represent stable IDs for protected crate/module scopes, adapter identities, forbidden resolved definition identities and capability classes, capability-bearing types or traits, exact item-scoped exceptions, and minimum matched production-item counts. Unknown fields or tags, unsupported versions, duplicate IDs, invalid paths, unmatched exceptions, and contradictory ownership MUST fail closed. This prerequisite SHALL check in fixture policy only and MUST NOT become an independent authority for any domain's core, adapter, exception, or compilation-lane policy.

#### Scenario: [DYLINT-TOOLING-003-P1] Valid fixture configuration is consumed exactly
- **WHEN** a v1 fixture configuration has unique IDs, valid protected/adapter scopes, registered capability classes, exact exceptions, and a nonzero expected production scope
- **THEN** the library consumes those values without fallback and reports the schema and configuration-entry IDs used by each check

#### Scenario: [DYLINT-TOOLING-003-N1] Malformed or permissive fallback configuration is rejected
- **WHEN** configuration is absent, has an unsupported version, contains an unknown field or enum, duplicates an ID, names an invalid scope, has an unmatched exception, or would require a default allow policy
- **THEN** lint execution terminates non-green with a structured configuration error before architecture diagnostics can count

#### Scenario: [DYLINT-TOOLING-003-N2] Domain policy cannot be handwritten into the tool prerequisite
- **WHEN** the tooling workspace or root Dylint metadata attempts to define compiler, evaluator, proof, lint, Reef, or conformance production boundaries independently of a later validated FCIS manifest
- **THEN** policy-owner tripwire validation fails and identifies the unauthorized second authority

### Requirement: [DYLINT-TOOLING-004] Every detector claim has allowed-positive and violating-negative evidence
Every production-loadable diagnostic SHALL name its claimed detector classes in the typed registry and SHALL map each class to at least one allowed-positive fixture and at least one violating-negative fixture in the canonical gate registry. Initial claimed classes SHALL cover the registered subsets of direct and qualified references, aliases, re-exports, associated or UFCS references where applicable, escaped function items, callback/function-pointer positions, configured trait/type references, visible declarative-macro expansions, adapter-module references, mutable static/thread-local state, configured capability-bearing public signatures, production items compiled in test builds, and true `cfg(test)` owners. UI goldens SHALL lock the expected diagnostic ID, class, primary span, configuration-entry ID, and normalized message class.

#### Scenario: [DYLINT-TOOLING-004-P1] Violating fixture emits the exact registered diagnostic
- **WHEN** a violating-negative fixture exercises one claimed detector class in a protected fixture scope under its declared lane
- **THEN** the exact registered diagnostic, detector class, configuration-entry ID, owner, and primary span match the golden evidence

#### Scenario: [DYLINT-TOOLING-004-P2] Allowed fixture remains clean
- **WHEN** an allowed-positive fixture exercises same-spelling local code, invocation-local mutation, an allowed outer adapter, an allowed test owner, or a non-capability interface paired with the claimed detector class
- **THEN** the registered architecture diagnostic is absent while the fixture still compiles and executes its intended compile-time assertion

#### Scenario: [DYLINT-TOOLING-004-N1] Missing detector polarity is non-green
- **WHEN** a diagnostic or claimed syntax class has only an allowed fixture, only a violating fixture, a stale golden, or no fixture mapping
- **THEN** registry validation fails and that diagnostic cannot be production-loadable or counted as evidence

#### Scenario: [DYLINT-TOOLING-004-N2] Wrong or overbroad detection fails
- **WHEN** a violating fixture emits another class/span/configuration ID or an allowed fixture emits the architecture diagnostic
- **THEN** the UI/live-probe suite fails rather than accepting approximate detection

### Requirement: [DYLINT-TOOLING-005] Live lanes prove resolution and reject vacuous success
The tooling oracle SHALL run live `cargo dylint` probes for every declared fixture package, library/test target, feature, and configuration lane. Each configured forbidden identity SHALL have a probe that resolves and references it under the pinned compiler. Production items compiled in test lanes SHALL remain protected unless their owner is actually classified as test-only. Every protected scope SHALL match at least its configured minimum production-item count. Missing lanes, unresolved configured identities, unmatched exceptions, zero matched production items, zero executed production diagnostics, and a probe corpus that executes no detector class MUST fail closed.

#### Scenario: [DYLINT-TOOLING-005-P1] Declared live lanes resolve and classify owners
- **WHEN** all registered library/test fixture lanes run, every configured identity is observed by its probe, production owners remain protected in test builds, and true allowed `cfg(test)` owners are classified as such
- **THEN** the report records the lane, resolution set, owner classes, production-item count, and detector classes as successful evidence

#### Scenario: [DYLINT-TOOLING-005-N1] Omitted lane or unresolved identity fails
- **WHEN** a declared lane is skipped, a configured identity cannot be resolved by its probe, an exception matches nothing, or a protected scope matches fewer production items than required
- **THEN** the oracle is non-green even if every executed lint invocation emitted no error

#### Scenario: [DYLINT-TOOLING-005-N2] Test compilation cannot exempt production code
- **WHEN** a production owner is compiled by a test target and references a registered forbidden capability
- **THEN** the same architecture diagnostic is required; treating the whole test lane as exempt fails its classification fixture

### Requirement: [DYLINT-TOOLING-006] Machine diagnostics are deterministic and semantically complete
Live lint invocations SHALL use rustc JSON diagnostics. The gate SHALL normalize only declared non-semantic fields such as disposable absolute roots and compiler progress noise, then canonically order records by lane, diagnostic ID, detector class, configuration-entry ID, owner, span, and message key. It MUST NOT normalize away diagnostic kind, applicability, owner, primary span, configuration ID, count, or failure. Equal fixture inputs and pins SHALL produce byte-equal semantic reports.

#### Scenario: [DYLINT-TOOLING-006-P1] Equal runs produce equal normalized evidence
- **WHEN** the same registered fixtures run from different disposable absolute directories under the exact pinned toolchain
- **THEN** their semantic JSON diagnostics and canonically ordered check records are byte-equal

#### Scenario: [DYLINT-TOOLING-006-N1] Normalization cannot hide a diagnostic change
- **WHEN** diagnostic applicability, class, owner, span, configuration ID, count, or outcome differs from its golden
- **THEN** the deterministic-evidence check fails and reports the semantic field that changed

### Requirement: [DYLINT-TOOLING-007] Fixes are denied until detector and rewrite safety are proven
Every registered diagnostic SHALL declare `FixPolicy::NoFix` or `FixPolicy::MachineApplicable` with a stable fix ID, disposable positive fix fixtures, negative no-machine-fix fixtures, and a domain parity plan. All initial `chelis-fcis-boundaries` diagnostics SHALL be `NoFix`, SHALL emit no `Applicability::MachineApplicable` suggestion, and SHALL leave disposable violating sources unchanged when `cargo dylint --fix` is invoked.

A future machine-applicable suggestion MUST NOT be registered until its detector polarities pass and its fixtures prove exact rewritten output, canonical formatting, successful compilation, a clean rerun of the same lint, required domain parity, no machine-applicable suggestion for ambiguous or unsafe inputs, byte-identical output after a second fix run, and no mutation outside the disposable copy. A fix MUST NOT add lint suppression, edit FCIS manifests or Dylint configuration, widen a boundary or exception, move code outside protected scope, or delete tests/evidence.

#### Scenario: [DYLINT-TOOLING-007-P1] Initial NoFix diagnostics cannot mutate source
- **WHEN** `cargo dylint --fix` runs against a disposable violating fixture for an initial boundary diagnostic
- **THEN** no machine-applicable suggestion is emitted, source bytes remain unchanged, and the same diagnostic remains reproducible

#### Scenario: [DYLINT-TOOLING-007-P2] Future registered fix meets the complete contract
- **WHEN** a later detector-green lint supplies every registered positive, negative, compile/lint, domain-parity, idempotence, and isolation fixture
- **THEN** its stable fix ID may be registered as machine-applicable and exercised only in disposable copies during acceptance

#### Scenario: [DYLINT-TOOLING-007-N1] Missing or unsafe fix evidence is rejected
- **WHEN** a lint is `NoFix`, lacks either fix-fixture polarity, changes behavior, fails formatting or compilation, remains lint-dirty, is non-idempotent, or proposes suppression, policy mutation, boundary escape, or evidence deletion
- **THEN** it emits no machine-applicable suggestion and any acceptance plan attempting to apply it fails non-green

#### Scenario: [DYLINT-TOOLING-007-N2] Fix execution cannot mutate the tracked checkout
- **WHEN** an acceptance fix check changes any tracked file or writes outside its declared disposable destination
- **THEN** the oracle fails, reports the escaped path class, and cannot restore a green result merely by reverting after detection

### Requirement: [DYLINT-TOOLING-008] Architecture claims remain layered and scope-bounded
The Dylint layer SHALL supplement Cargo dependency boundaries and domain behavioral evidence; it SHALL NOT replace them or be described as a proof of referential transparency or transitive purity. Reports and documentation SHALL enumerate the exact fixture-proven detector classes and SHALL retain inactive cfg/target/feature code, procedural-macro and build-script implementation effects, arbitrary dynamic dispatch, precompiled dependency behavior, and future Rust syntax as blind spots unless separate evidence covers them.

#### Scenario: [DYLINT-TOOLING-008-P1] Honest scoped claim is accepted
- **WHEN** a report names exact pins, lanes, configuration entries, matched counts, fixture-proven forms, detector/fix states, and known blind spots without overstating them
- **THEN** the tooling oracle accepts the report as scoped architecture evidence

#### Scenario: [DYLINT-TOOLING-008-N1] Completeness claim is rejected
- **WHEN** documentation or machine metadata claims arbitrary transitive, macro/build-script, dynamic-dispatch, inactive-lane, dependency, future-syntax, or behavioral purity from the Dylint fixtures
- **THEN** threat-model tripwire validation fails and identifies the unsupported claim class

### Requirement: [DYLINT-TOOLING-009] One standalone oracle completes the prerequisite
The sole completion oracle SHALL be `.venv/bin/python scripts/dylint_gate.py`. Its canonically ordered `tools/dylint/gate.toml` registry SHALL map every requirement scenario through positive/negative fixtures and checks to the `dylint-tooling` oracle. The runner SHALL execute argv arrays with `shell=False`, a declared environment allowlist, disposable fixture roots, and an isolated `CARGO_TARGET_DIR`. Its typed result SHALL be `Passed` with an empty error list, `Failed` with a nonempty error list, or `Blocked` with a nonempty error list. Success requires every registered detector class in both polarities, every live lane and configuration probe, all NoFix checks, exact pins, stable-root isolation, deterministic output, and an unchanged tracked checkout.

#### Scenario: [DYLINT-TOOLING-009-P1] Complete standalone oracle passes
- **WHEN** the exact acceptance command runs every registered check against one revision and all requirements hold
- **THEN** it exits 0 and emits `Passed` with an empty error list and complete scenario/fixture/check reachability

#### Scenario: [DYLINT-TOOLING-009-N1] Missing evidence cannot be skipped green
- **WHEN** a fixture, lane, executable, nightly component, golden, detector polarity, registry mapping, or expected production match is absent or not executed
- **THEN** the oracle exits nonzero with `Failed` or `Blocked` and a nonempty structured error list

#### Scenario: [DYLINT-TOOLING-009-N2] Focused checks are not completion
- **WHEN** only a UI subset, one diagnostic, one lane, or a developer-focused command passes
- **THEN** it is supporting evidence and MUST NOT produce the prerequisite completion state

### Requirement: [DYLINT-TOOLING-010] Dylint tooling precedes FCIS contract and domain work
`establish-dylint-tooling` SHALL have no dependency on `establish-fcis-contract-mechanics`. Contract mechanics SHALL treat the `dylint-tooling` oracle, schema version, and diagnostic registry as current-revision prerequisite inputs, then own FCIS-manifest projection and domain command plans. Every domain FCIS change SHALL depend transitively on this prerequisite through contract mechanics and MUST NOT create a competing Dylint implementation.

#### Scenario: [DYLINT-TOOLING-010-P1] Downstream mechanics consumes the accepted prerequisite
- **WHEN** contract mechanics begins after the standalone oracle passes on the same revision
- **THEN** it validates and consumes the registered Dylint schema/diagnostics and adds only manifest projection, domain lanes, and report integration

#### Scenario: [DYLINT-TOOLING-010-N1] Circular or bypassed prerequisite is rejected
- **WHEN** the Dylint oracle requires `scripts/fcis_gate.py`, contract mechanics proceeds without current Dylint evidence, or a domain change introduces another boundary-lint library
- **THEN** prerequisite or typed-owner validation fails before downstream acceptance can be green
