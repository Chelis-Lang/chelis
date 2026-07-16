## Context

The active FCIS designs use Cargo boundaries as their primary capability boundary and require a resolved Rust layer for references that crate topology alone cannot express. They currently describe Dylint implementation inside `establish-fcis-contract-mechanics`, even though that change also has to create manifests, traceability, reporting, and orchestration. The Dylint tool must be executable and adversarially validated before those later contracts can safely depend on it.

Chelis' root workspace is stable-only. Dylint 6.0.1 uses rustc-private APIs and therefore requires a dated nightly, `rustc-dev`, and `llvm-tools-preview`. The tooling must not become a production dependency or make ordinary Cargo commands select nightly. It also cannot own domain boundary policy: later validated FCIS manifests remain authoritative for domain core scopes, adapter scopes, forbidden identities, exceptions, and compilation lanes.

This is evidence tooling rather than a production FCIS core. The lint passes consume explicit compiler state plus strict configuration and return diagnostics through rustc; `scripts/dylint_gate.py` is an imperative test orchestrator. No cache, persistent replay record, query key, evidence digest, or semantic result identity is introduced. Stable diagnostic, fixture, and configuration IDs exist only to make the tool contract reviewable and testable.

## Goals / Non-Goals

**Goals:**

- Establish a root-stable, nested-nightly Dylint workspace before FCIS contract mechanics or domain migrations.
- Provide one initial `chelis-fcis-boundaries` category library with a strict versioned configuration surface and stable diagnostic registry.
- Prove every claimed detector form with allowed-positive and violating-negative fixtures, live resolved-path probes, exact diagnostics, and fail-closed malformed/vacuous cases.
- Establish `NoFix` as the initial policy and the test/evidence threshold that any later machine-applicable suggestion must meet.
- Provide one independently runnable `dylint-tooling` acceptance oracle that later FCIS gates can execute as a current-revision prerequisite.

**Non-Goals:**

- Replacing Cargo dependency boundaries, behavioral denial/determinism/replay/parity tests, or domain acceptance suites.
- Defining compiler, evaluator, proof, lint, Reef, or conformance core/adapter policy before their FCIS manifests exist.
- Claiming arbitrary macro, build-script, dynamic-dispatch, transitive-dependency, inactive-cfg, or future-Rust completeness.
- Moving the root workspace to nightly, adding a production dependency, or shipping Dylint in Chelis runtime/toolchain artifacts.
- Shipping automatic architectural rewrites in the initial library. Every initial diagnostic is `NoFix`.
- Creating one crate per diagnostic or speculative category libraries without distinct ownership and evidence lifecycles.

## Decisions

### 1. Use one isolated `tools/dylint` workspace

The repository adds:

```text
tools/dylint/
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── pins.toml
├── gate.toml
└── fcis-boundaries/
    ├── Cargo.toml
    ├── build.rs
    ├── src/
    ├── ui/
    └── fixtures/
```

`tools/dylint` is a nested Cargo workspace and is not a root workspace member. Its toolchain is exactly `nightly-2026-04-16` with `rustc-dev` and `llvm-tools-preview`. `pins.toml` records `cargo-dylint = 6.0.1`, `dylint-link = 6.0.1`, `dylint_linting = 6.0.1`, and `dylint_testing = 6.0.1`; package manifests and the nested lockfile must agree. The gate fails with installation guidance if the selected `cargo dylint` executable is absent or has a different version; it never downloads or opportunistically upgrades tooling during acceptance.

The root `rust-toolchain.toml`, root workspace members, root dependency graph, and normal stable gate remain unchanged. Root metadata may contain only inert path discovery metadata if Dylint requires it; no domain policy may be placed there.

Alternative considered: adding the library to `chelis-lint` or `chelis-fcis-contract`. Rejected because those are stable product/evidence crates, while Dylint links rustc-private nightly internals. An external repository was rejected because lint code, fixtures, pinned compiler internals, and consuming Chelis changes need atomic review.

### 2. Group diagnostics by policy and evidence ownership

The initial category is `tools/dylint/fcis-boundaries`, package `chelis-fcis-boundaries`, built as a Dylint `cdylib`. One typed static registry in this library owns:

- configuration schema version;
- stable diagnostic ID and rustc lint name;
- supported detector classes;
- whether the diagnostic is production-loadable; and
- `FixPolicy`, initially `NoFix` for every entry.

The initial diagnostics cover resolved forbidden API/type/macro references and function-item escape, core-to-adapter references, mutable/thread-local ambient state, and capability-bearing public interfaces. Multiple passes may share traversal/indexing internally, but each emitted diagnostic retains a stable registered ID.

A sibling category is allowed only when it has a distinct typed policy owner or compilation/evidence lifecycle. It cannot duplicate FCIS boundary diagnostics or maintain an independent copy of FCIS domain policy. Per-diagnostic crates were rejected because they multiply nightly builds, lockstep pins, and fixture harnesses without providing an ownership boundary.

### 3. Keep the library generic and configuration strict

The Dylint library owns a strict `schema_version = 1` input shape read from the declared `DYLINT_TOML` path. Configuration names protected crate/module scopes, adapter module identities, forbidden resolved definition identities grouped by capability class, capability-bearing types/traits, exact item-scoped exceptions, and expected minimum protected production-item counts. Every entry has a stable ID. Unknown fields/tags, duplicate IDs, unsupported versions, invalid paths, unresolved live-probe identities, unmatched exceptions, and a protected scope with zero matched production items are contract errors.

The initial change checks only fixture policy from `tools/dylint/fixtures`; it does not check in domain core/adapter policy. `establish-fcis-contract-mechanics` later projects validated FCIS manifests into this schema and owns domain package/target/feature/configuration lane selection. The projection is tested against the Dylint parser so neither side can silently reinterpret fields.

Matching uses rustc-resolved identities and HIR/type information, not lexical spelling. Configured forbidden identities have live probe fixtures that intentionally resolve and reference the identity. A configured entry that cannot be proven by its probe cannot count as active evidence. This avoids treating an unused or misspelled string as successfully resolved.

Alternative considered: a handwritten repository `dylint.toml` containing all domain boundaries. Rejected because it would become a second authority beside FCIS manifests.

### 4. Claim only fixture-proven detector forms

Each registry entry has a matrix in `tools/dylint/gate.toml` mapping stable detector-class IDs to at least one allowed-positive fixture and one violating-negative fixture. The initial claimed forms are direct and qualified references, aliases, re-exports, associated/UFCS references where applicable, escaped function items, callback/function-pointer positions, configured trait/type references, declarative-macro expansions visible in the selected compilation lane, adapter-module references, mutable static/thread-local state, configured capability-bearing public signatures, production items compiled in test builds, and true `cfg(test)` owners.

Allowed-positive fixtures include same-spelling local APIs, references outside the protected scope, invocation-owned local mutation, immutable constants/statics without configured interior-mutable capability, sealed/non-capability interfaces, and true test-owned code where policy allows it. Production code compiled by a test target remains protected unless its owner is actually test-classified.

`dylint_testing` UI tests lock diagnostic code, primary span, stable configuration-entry ID, and normalized message class in `.stderr` goldens. Live `cargo dylint` probes run library and test targets and emit rustc JSON. The gate normalizes only fixture-relative paths and explicitly non-semantic compiler noise; it does not erase diagnostic kind, configuration ID, owner, span, or count.

These tests prove only registered forms. Procedural-macro/build-script implementation effects, inactive cfg/target/feature branches, arbitrary dynamic dispatch, behavior inside precompiled dependencies, and unknown future syntax remain declared blind spots.

### 5. Make fix eligibility explicit and initially deny all fixes

`FixPolicy` is `NoFix` or `MachineApplicable { fix_id, positive_fix_fixtures, negative_fix_fixtures, parity_plan }`. All initial production diagnostics are `NoFix` and UI/live tests assert they emit no `Applicability::MachineApplicable` suggestion. Running `cargo dylint --fix` against their disposable violating fixtures must not change source and must leave the diagnostic reproducible.

A later change may register `MachineApplicable` only after detector polarity is green and before suggestion implementation it adds failing fixtures that prove:

1. exact expected rewritten output;
2. canonical formatting and successful compilation;
3. a clean rerun of the same lint;
4. the registered domain behavioral/parity command;
5. no machine-applicable suggestion for ambiguous or unsafe negative cases;
6. byte-identical output after a second `cargo dylint --fix`; and
7. no mutation outside the declared disposable fixture copy.

Fixes may not add `allow`/`expect`, edit FCIS manifests or Dylint configuration, widen exceptions, move code outside protected scope, or delete tests/evidence. The foundational gate supports and unit-tests disposable-copy and checkout-state guards, but it does not claim a positive production fix until a real registered fix supplies all evidence.

### 6. Give the prerequisite an independent bootstrap oracle

`.venv/bin/python scripts/dylint_gate.py` is the sole `dylint-tooling` completion oracle. `tools/dylint/gate.toml` is its typed, canonically ordered self-test registry. The runner uses argv arrays with `shell=False`, a declared environment allowlist, and an isolated `CARGO_TARGET_DIR`. It performs:

1. pin/toolchain/lockfile consistency checks;
2. root stable-workspace isolation checks;
3. nested workspace formatting, compilation, and tests;
4. all registered UI detector matrices;
5. live library/test lane probes with JSON diagnostics;
6. malformed, unresolved, omitted-lane, and zero-match failures;
7. `NoFix` disposable-copy checks and runner tests for future fix isolation; and
8. tracked-checkout pre/post state comparison.

Its result is `Passed { checks, errors: [] }`, `Failed { checks, errors: NonEmpty }`, or `Blocked { checks, errors: NonEmpty }`, serialized in canonical check/error order. Absolute checkout paths, temporary directories, durations, and process IDs are non-semantic metadata. Missing nightly components, missing exact `cargo-dylint`, absent fixtures, stale goldens, zero executed detector classes, or checkout mutation cannot be reported as success.

Focused developer commands may run subsets, but only the exact command

```text
.venv/bin/python scripts/dylint_gate.py
```

with exit status 0, `Passed`, an empty error list, all registered detector classes executed in both polarities, and an unchanged tracked checkout completes this change.

Alternative considered: making this first change depend on the future `scripts/fcis_gate.py`. Rejected because that would create a prerequisite cycle. Contract mechanics will later invoke this standalone oracle and map its report into the FCIS report schema.

### 7. Make Dylint tooling the root prerequisite

The delivery DAG is:

```text
establish-dylint-tooling
    -> establish-fcis-contract-mechanics
        -> security corrections and domain FCIS migrations
```

`establish-fcis-contract-mechanics` consumes the completed Dylint input schema and diagnostic registry, projects domain manifests into `DYLINT_TOML`, registers domain compilation lanes and fix policies, and executes `scripts/dylint_gate.py` against the current revision as prerequisite evidence. Domain changes depend transitively on this change through contract mechanics and must not reimplement the Dylint library.

Rollback removes the nested tooling workspace, bootstrap runner, and its documentation before any downstream change is implemented. After downstream manifests consume the schema, rollback requires first removing those consumers; there is no production data or API migration.

## Risks / Trade-offs

- **[Nightly/rustc-private breakage]** → Exact dated pins, a nested lockfile, live UI probes, and explicit reviewed upgrades; no floating nightly.
- **[False confidence from green fixtures]** → Claims are enumerated by detector class, both polarities are required, live probes prevent vacuous strings, and blind spots remain explicit.
- **[Duplicate policy authority]** → The tool owns only schema and diagnostic vocabulary; later FCIS manifests own domain policy and generate Dylint configuration.
- **[Category proliferation]** → A sibling library requires distinct typed ownership or lifecycle and a registry tripwire; same-owner diagnostics remain passes in `fcis-boundaries`.
- **[Acceptance mutates source through `--fix`]** → Initial diagnostics are `NoFix`; all fix checks use disposable copies and compare tracked state before/after.
- **[Bootstrap runner diverges from later FCIS runner]** → The bootstrap report and command remain narrow; contract mechanics invokes it rather than duplicating its Dylint semantics.
- **[Local tool prerequisite unavailable]** → The oracle reports `Blocked` with exact pin/install guidance, never skipped or green; CI provisions the exact version explicitly.

## Migration Plan

1. Land the new OpenSpec and make it the first documented prerequisite.
2. Add failing schema, registry, detector-polarity, live-probe, `NoFix`, runner, and root-isolation tests.
3. Add the nested workspace and exact pins, then implement the registered diagnostics until the prerequisite oracle passes.
4. Update developer/CI documentation and provision the exact Dylint executable/toolchain without changing the root stable gate.
5. Only after the `dylint-tooling` oracle is green, implement `establish-fcis-contract-mechanics` and replace its Dylint-creation tasks with schema/projection/integration tasks.
6. Register domain policies and compilation lanes only through the contract-mechanics manifests.

## Open Questions

None. Diagnostic expansion and machine-applicable production fixes require later explicit registrations and evidence; they are not implicit follow-up work in this prerequisite.
