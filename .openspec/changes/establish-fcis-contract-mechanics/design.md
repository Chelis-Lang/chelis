## Context

`.openspec/config.yaml` and `.openspec/FCIS_ARCHITECTURE.md` require explicit boundaries, positive/negative coverage, deterministic ordering, one acceptance oracle, architecture threat models, and honest focused-slice evidence. The active changes satisfy those requirements narratively, but prerequisite edges, slice membership, scenario polarity, test ownership, exact module boundaries, and oracle commands are repeated across Markdown. OpenSpec strict validation cannot evaluate those cross-artifact invariants, and every active oracle currently points at a runner that has not been implemented.

Chelis already uses a successful machine-form-plus-tripwire pattern in `chelis-conformance`: typed `MANIFEST`/`REGISTRY` values are checked against normative prose and CI configuration. `scripts/gate.py` similarly owns one argv registry rather than allowing docs and CI to hand-inline divergent commands. The independent `establish-dylint-tooling` prerequisite owns and accepts the isolated lint implementation, schema, registry, fixture corpus, and standalone oracle before this change begins. FCIS evidence should consume that prerequisite and reuse the existing machine-form patterns without creating a second lint implementation or generic production effect framework.

## Goals / Non-Goals

**Goals:**

- Make FCIS change structure, dependencies, coverage, boundaries, slices, and oracles mechanically checkable before domain implementation.
- Bind the accepted `establish-dylint-tooling` schema and diagnostics to manifest-derived domain policy and execute its standalone oracle as current-revision prerequisite evidence.
- Give requirements, scenarios, fixtures, and gates stable identities with complete positive/negative traceability.
- Make the checker a deterministic one-shot functional core over parsed manifest and extracted artifact data.
- Make missing manifests, fixtures, commands, prerequisites, or evidence fail closed with stable diagnostics.
- Share architecture-test mechanics while keeping production protocol and decision types domain-owned.
- Preserve the existing OpenSpec Markdown as the reviewable design surface and lock it to the machine form with tripwires.

**Non-Goals:**

- Implementing any compiler, evaluator, proof, lint, Reef, or conformance migration.
- Creating, repinning, or duplicating the Dylint workspace, generic diagnostics, or foundational detector/fix fixture corpus owned by `establish-dylint-tooling`.
- Defining a generic production `Decision`, `Step`, effect, request, observation, or workflow framework.
- Treating source scanning as a proof of transitive purity.
- Executing arbitrary shell strings from a manifest.
- Giving focused slices authority to claim final change completion.
- Adding persistent semantic identities to domains that explicitly defer them.

## Decisions

### 1. Versioned manifests are the canonical machine form

`.openspec/fcis/registry.toml` declares `schema_version = 1` and the canonically ordered active change IDs. Each active change owns `.openspec/changes/<change>/fcis.toml`. The per-change manifest contains:

- stable change ID, capability IDs, and `change_kind`;
- prerequisite change IDs;
- exact designated core crates/modules and adapter crates/modules;
- forbidden capability classes;
- architecture enforcement mechanism, Dylint diagnostic registrations, claimed detector classes and positive/negative fixtures, `NoFix` or machine-applicable fix policies and their disposable fix/parity fixtures, proven fixture classes, and known blind spots;
- one-shot versus request/observation protocol classification;
- persistent identity classification: `none`, `structural-only`, or named versioned identity domains;
- public surface/capability matrix references where applicable;
- focused slice IDs, one final oracle ID, and command-plan IDs;
- stable requirement/scenario IDs, explicit scenario polarity, test/fixture IDs, and owning slice;
- compatibility/rollback boundary and deferred contracts.

Manifests use only typed TOML values; commands are references to a separately validated argv plan and are never shell strings. Unknown fields, duplicate IDs, unknown enum tags, unknown references, and schema-version mismatch fail closed. Registry and manifest order is canonical bytewise UTF-8 order unless a domain manifest declares a separately checked semantic order such as engine registration or lock rank.

The Markdown remains the human review surface. Tripwire tests parse exact machine-relevant blocks or stable annotations from proposals, designs, specs, and tasks and require agreement with the manifests. Neither form may silently override the other: drift is a failing contract diagnostic.

### 2. The checker is itself FCIS

A new dependency-minimal `crates/chelis-fcis-contract` library owns parsed data types, normalization, graph checks, traceability checks, tripwire comparisons, command-plan validation, and structured reports. Its core entry point is conceptually:

```text
check_contract(bundle: ContractBundle) -> ContractReport
```

`ContractBundle` contains already-read manifest bytes, extracted OpenSpec artifact text, Cargo metadata projected into deterministic values, public-API/architecture fixture results, and declared filesystem existence observations. The checker core does not read files, run Cargo, inspect Git, execute tests, inspect environment, or print. A small binary/loading adapter may acquire those inputs and serialize the report.

The designated core modules are `model`, `normalize`, `graph`, `traceability`, `tripwire`, `plan`, and `report`. Binary loading, Cargo metadata acquisition, Git revision reporting, subprocess execution, and terminal rendering are adapters. The core uses ordered collections or explicit final sorting; equal bundles produce byte-equal canonical JSON reports.

### 3. Stable IDs make coverage explicit

Requirement IDs are unique repository-wide and use an uppercase domain prefix plus number, for example `EVAL-PROTOCOL-001`. Scenario IDs extend the requirement ID with polarity and ordinal, for example `EVAL-PROTOCOL-001-P1` and `EVAL-PROTOCOL-001-N1`. Every scenario declares exactly one of `positive` or `negative` in the manifest; prose titles are not used to infer polarity.

Every active requirement has at least one positive and one negative scenario. Every active scenario maps to one or more fixture IDs. Every fixture declares an executable Rust test, Python test, compile-fail case, architecture fixture, golden-vector case, or explicit manual prerequisite. FCIS completion oracles may not depend on an undocumented manual fixture. Every fixture belongs to exactly one focused slice while the final oracle includes the transitive union of all slices.

The checker rejects duplicate IDs, orphan fixtures, unmapped scenarios, missing polarity parity, fixture paths or test targets that do not exist, unknown slices, prerequisite cycles, slice cycles, and final oracles that omit active slices. A requirement may explicitly mark a scenario `not-applicable` only through a separately reviewed requirement-state change; absence is not treated as not applicable.

### 4. One typed command registry drives all FCIS gates

`scripts/fcis_gate.py` is a thin Python 3.11+ orchestrator. It accepts only oracle and slice IDs emitted by the Rust checker's validated command plan. The plan contains argv arrays, workspace-relative working directories, prerequisite edges, environment policy, mutation policy, expected evidence-report schema, and deterministic order. Mutation policy is `ReadOnlyCheckout` or `DisposableFixtureCopy`. A plan containing `cargo dylint --fix` must use `DisposableFixtureCopy`, name its source fixture and generated disposable destination, and declare tracked-checkout pre/post state verification; `--fix` is invalid in a read-only plan. The plan cannot contain shell pipelines, redirections, command substitution, or free-form shell text. The Python runner invokes commands with `shell=False`, captures structured results, stops or continues according to the declared fail-closed policy, and never reimplements domain semantics.

The initial final oracle IDs are:

- `dylint-tooling` (an independently owned prerequisite invoked through `scripts/dylint_gate.py`);
- `fcis-contract`;
- `tide-evaluator-denial`;
- `lint-snapshots`;
- `compiler-core`;
- `evaluator-effects`;
- `proof-forced-result-removal`;
- `proof-execution`;
- `reef-explicit-credentials`; and
- `reef-workflows`.

Every ID is registered before domain implementation. A planned but unimplemented command returns `evidence_not_implemented`; it is not skipped and cannot produce success. Child final oracles execute or include their prerequisite regression suites against the same working revision rather than trusting a stale handwritten evidence claim.

### 5. Reports use an algebraic, deterministic schema

The versioned runner report is:

```text
GateReport = Passed { checks, errors: [] }
           | Failed { checks, errors: NonEmpty<GateError> }
           | Blocked { checks, errors: NonEmpty<GateError> }
```

Each check records stable check ID, oracle/slice, outcome class, normalized command-plan ID, and structured evidence references. Wall-clock duration, absolute checkout path, terminal mode, process ID, and raw localized OS text are optional report metadata and do not participate in semantic report equality. Errors are ordered by oracle, slice, check ID, error code, and stable detail key. A passed report with a nonempty error list is structurally unrepresentable in Rust and rejected during Python report assembly.

A focused slice can pass while its parent final oracle remains blocked or failed. Only the manifest's final oracle may produce the evidence state used for change completion.

### 6. Architecture evidence is layered and honest

The shared architecture harness supports:

1. resolved Cargo dependency allowlists from `cargo metadata`;
2. exact production-module manifests for temporary mixed-crate boundaries;
3. `#![forbid(unsafe_code)]` for designated core crates and modules where the mixed boundary can enforce it;
4. a shared manifest-configured Dylint library for resolved forbidden API/type/macro references, forbidden function-item escape, core-to-adapter references, thread-local or mutable-static ambient state, and capability-bearing public interfaces;
5. negative compile-fail/Dylint-UI fixtures for the specifically claimed direct, alias, re-export, qualified, callback, function-pointer, trait, declarative-macro expansion, adapter-import, production-in-test-build, and classified `cfg(test)` classes; and
6. domain behavioral determinism, denial, replay, parity, and host-action instrumentation.

The completed `establish-dylint-tooling` prerequisite owns the isolated `tools/dylint` workspace, `chelis-fcis-boundaries` passes, strict configuration schema, typed diagnostic/detector/fix registry, exact pins, foundational UI/live fixtures, and `.venv/bin/python scripts/dylint_gate.py` oracle. This change executes that oracle against the same revision and verifies the accepted schema version, diagnostic IDs/classes, fix capabilities, pins, and root-stable isolation before any domain architecture command. A missing/stale prerequisite report, schema drift, unknown diagnostic, or registry mismatch is non-green; this change does not rebuild or repin the lint library.

The validated FCIS manifests remain the only owner of domain core/module boundaries, forbidden capability classes, adapter sets, exceptions, and required package/target/feature/configuration lanes. The checker projects that policy deterministically into the prerequisite's strict Dylint configuration through the command plan's declared `DYLINT_TOML` value; a handwritten domain `dylint.toml` capability list cannot become a second authority. Static workspace metadata may locate the accepted lint package but contains no independent boundary policy. Projection fixtures round-trip each manifest field through the prerequisite parser and reject unknown fields, unsupported schema versions, and semantic loss.

Each domain `DylintRegistration` references a production-loadable prerequisite diagnostic ID, only detector classes that the prerequisite registry supports, domain allowed-positive and violating-negative fixture/probe IDs for every claimed use, and `FixPolicy`. A domain detector use cannot enter an acceptance command until the standalone prerequisite is green, each claimed class has both polarities, and the violating fixture emits the exact diagnostic while the allowed fixture remains clean. Live domain plans additionally fail when a forbidden identity cannot resolve, a declared lane is omitted, an exception is unmatched, or a configured core matches zero production items. This establishes only fixture-bounded evidence and cannot be promoted to arbitrary Rust completeness.

A domain registration declares `NoFix` or references a prerequisite-supported `MachineApplicable { fix_id, positive_fix_fixtures, negative_fix_fixtures, parity_plan }`. Because every initial prerequisite diagnostic is `NoFix`, no FCIS manifest may register a machine-applicable fix until a separately accepted Dylint-tooling change adds that fix and its complete evidence. The checker rejects unknown or unsupported fix IDs and requires disposable exact-output/no-fix, formatting/compilation, clean-rerun, domain-parity, idempotence, and checkout-immutability plans. Fix commands cannot insert suppression, edit policy, widen exceptions, escape boundaries, or delete evidence; acceptance runs them only in declared disposable copies. A separate explicitly invoked developer command may apply a proven prerequisite-supported fix to a working tree.

Every command plan declares the package, target, feature, and configuration lanes it compiles and requests Cargo JSON diagnostics. The FCIS runner normalizes Dylint diagnostic IDs and evidence references into the architecture report together with exact toolchain/library/dependency pins, resolved configured entries, and matched production-core counts. `cfg(test)` coverage compiles the declared test lane and classifies test-only owners without exempting production functions compiled in that lane. Optional and platform-specific code counts only when its declared lane is executed.

Each change manifest states which layers it uses, the threat model, and blind spots. Resolved Dylint checks commonly see declarative-macro-expanded forbidden calls, but do not prove side-effect freedom of procedural-macro or build-script implementations. They also do not prove code removed by an unexecuted `cfg`, target, or feature lane, arbitrary dynamic-dispatch targets, or transitive behavior hidden in precompiled dependencies. Dependency, Dylint, compile-fail, and source checks are guardrails. The checker rejects a claim of arbitrary macro/build-script, dynamic-dispatch, transitive-side-effect, uncompiled-code, or future-syntax completeness from those layers.

The harness may share manifest parsing, Dylint configuration projection, canonical identity test helpers, fixture execution, and report types. Production domains retain separate enums, policies, protocols, and result algebras.

### 7. Mechanical tables have one owner

Machine-relevant registries and matrices have one typed owner and are generated into docs or protected by tripwires. Initial required owners include:

- evaluator host-capable builtin/effect/policy/target classification;
- lint rule order, severity, blocking status, and requirements;
- compiler target/surface/preflight capability matrix;
- proof engine descriptor and trusted authorization entries;
- Reef request/action, lock-rank, transaction-state, and report-state vocabularies; and
- identity-domain strings, schema versions, field tags, and golden vectors for domains that actually ship identities.

String lists independently repeated across frontends are not accepted as the authoritative registry. Public closed vocabularies use exhaustive enums or generated tables with duplicate and completeness checks.

## Risks / Trade-offs

- **Manifest duplication:** machine forms can drift from Markdown. Tripwire tests deliberately make that drift blocking rather than relying on review memory.
- **Large initial registration cost:** dozens of requirements and hundreds of scenarios need stable IDs and fixture mappings. Registration is performed before domain implementation and may land in bounded change-by-change slices.
- **Checker overreach:** static checks can create false purity confidence. Reports expose enforcement layers, compilation lanes, matched-item counts, and blind spots, and behavioral gates remain authoritative.
- **Nightly compiler coupling:** Dylint uses unstable compiler internals and its lint library must track one dated nightly. The nested workspace, exact pins, lockfile, fixture corpus, and reviewed upgrade procedure isolate that churn from Chelis's stable production workspace.
- **Vacuous or partial lint coverage:** a renamed module, unresolved API identity, disabled feature, or cfg-elided item could otherwise disappear from evidence. Resolved-entry checks, nonzero matched production counts, detector polarity parity, and an explicit package/target/feature/configuration matrix fail closed.
- **Unsafe automatic remediation:** a fix can make a lint disappear while preserving or hiding the capability. Machine-applicable fixes therefore require a typed policy, disposable exact-output and no-fix fixtures, post-fix compile/lint/parity checks, idempotence, and rejection of suppression, boundary escape, policy edits, or evidence deletion.
- **Command registry rigidity:** new slices require manifest and checker updates. This is intentional so an undocumented command cannot become completion evidence.
- **Evidence infrastructure in the workspace:** the checker adds compile cost. It remains dependency-minimal, has no production runtime dependency edge, and can be built/tested independently.
- **OpenSpec schema evolution:** schema changes require a new manifest schema version and migration tests; unknown future fields fail closed rather than being ignored.

## Migration Plan

1. Add failing manifest-schema, graph, traceability, report-algebra, tripwire, and command-plan tests.
2. Create the pure checker model/validation/report core and loader adapter; pass `--slice schema`.
3. Register stable IDs and traceability for the three narrow security prerequisite changes first; pass `--slice traceability-security`.
4. Register lint and compiler, then evaluator and proof, then Reef, preserving the delivery order in `FCIS_ARCHITECTURE.md`; pass each focused registration slice.
5. Execute the accepted standalone `dylint-tooling` oracle on the current revision, bind its schema/diagnostic/fix registry into the contract model, implement manifest-to-`DYLINT_TOML` round-trip projection, and register each domain detector use, fixture polarity, `NoFix` or supported fix reference, live resolution/count probes, and compilation lane before passing `--slice architecture`.
6. Add the Python orchestrator and tests for argv-only execution, unknown IDs, missing evidence, report invariants, prerequisite execution, and deterministic ordering; pass `--slice runner`.
7. Add Markdown/manifests/CI tripwires and update every active change to depend on the mechanics oracle before implementation.
8. Run the final oracle. Domain changes remain blocked until their manifest and failing contract fixtures are registered.

Rollback removes only evidence tooling and cannot be used to claim an FCIS domain complete. Once a domain implementation begins under the manifest contract, weakening or removing its registered evidence requires a separately reviewed OpenSpec change.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py fcis-contract
```

The runner must validate the current-revision `dylint-tooling` prerequisite report and accepted schema/diagnostic/fix registry; schema rejection and normalization; complete active-change registration; acyclic prerequisite/slice graphs; stable requirement/scenario IDs; positive/negative parity; scenario-to-fixture-to-slice-to-final-oracle reachability; exact oracle command tripwires; argv-only command plans; deterministic report ordering; success/error algebra; layered architecture threat-model declarations; exact accepted Dylint/toolchain/dependency pins; manifest-derived lint policy; declared package/target/feature/configuration lanes; resolved configured entries; nonzero matched production-core counts; normalized Dylint diagnostics; detector positive/negative parity; typed supported fix policies; disposable exact-output/no-fix/parity/idempotence fix evidence; tracked-checkout immutability; and negative fixtures for duplicate, unknown, missing, stale, cyclic, shell-string, contradictory-report, prerequisite/schema/registry mismatch, unresolved-lint-entry, zero-match, omitted-lane, one-polarity detector evidence, unsupported or unsafe fix, checkout mutation, false-purity, and false-completion cases. Success means exit status 0, a `Passed` report with an empty error list, every active FCIS change registered, every active scenario reachable from its final oracle, no unimplemented evidence counted as skipped or green, and no focused slice represented as final completion evidence.
