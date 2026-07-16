## Context

`openspec/config.yaml` and `openspec/FCIS_ARCHITECTURE.md` require explicit boundaries, positive/negative coverage, deterministic ordering, one acceptance oracle, architecture threat models, and honest focused-slice evidence. The active changes satisfy those requirements narratively, but prerequisite edges, slice membership, scenario polarity, test ownership, exact module boundaries, and oracle commands are repeated across Markdown. OpenSpec strict validation cannot evaluate those cross-artifact invariants, and every active oracle currently points at a runner that has not been implemented.

Chelis already uses a successful machine-form-plus-tripwire pattern in `chelis-conformance`: typed `MANIFEST`/`REGISTRY` values are checked against normative prose and CI configuration. `scripts/gate.py` similarly owns one argv registry rather than allowing docs and CI to hand-inline divergent commands. FCIS evidence should reuse those patterns without creating a generic production effect framework.

## Goals / Non-Goals

**Goals:**

- Make FCIS change structure, dependencies, coverage, boundaries, slices, and oracles mechanically checkable before domain implementation.
- Give requirements, scenarios, fixtures, and gates stable identities with complete positive/negative traceability.
- Make the checker a deterministic one-shot functional core over parsed manifest and extracted artifact data.
- Make missing manifests, fixtures, commands, prerequisites, or evidence fail closed with stable diagnostics.
- Share architecture-test mechanics while keeping production protocol and decision types domain-owned.
- Preserve the existing OpenSpec Markdown as the reviewable design surface and lock it to the machine form with tripwires.

**Non-Goals:**

- Implementing any compiler, evaluator, proof, lint, Reef, or conformance migration.
- Defining a generic production `Decision`, `Step`, effect, request, observation, or workflow framework.
- Treating source scanning as a proof of transitive purity.
- Executing arbitrary shell strings from a manifest.
- Giving focused slices authority to claim final change completion.
- Adding persistent semantic identities to domains that explicitly defer them.

## Decisions

### 1. Versioned manifests are the canonical machine form

`openspec/fcis/registry.toml` declares `schema_version = 1` and the canonically ordered active change IDs. Each active change owns `openspec/changes/<change>/fcis.toml`. The per-change manifest contains:

- stable change ID, capability IDs, and `change_kind`;
- prerequisite change IDs;
- exact designated core crates/modules and adapter crates/modules;
- forbidden capability classes;
- architecture enforcement mechanism, proven fixture classes, and known blind spots;
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

`scripts/fcis_gate.py` is a thin Python 3.11+ orchestrator. It accepts only oracle and slice IDs emitted by the Rust checker's validated command plan. The plan contains argv arrays, workspace-relative working directories, prerequisite edges, environment policy, expected evidence-report schema, and deterministic order. It cannot contain shell pipelines, redirections, command substitution, or free-form shell text. The Python runner invokes commands with `shell=False`, captures structured results, stops or continues according to the declared fail-closed policy, and never reimplements domain semantics.

The initial final oracle IDs are:

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
3. `forbid(unsafe_code)` and resolved Clippy/custom-lint forbidden API or macro checks where configured;
4. public-interface checks against unsealed capability-bearing callbacks or traits;
5. negative compile/source fixtures for the specifically claimed direct, alias, re-export, qualified, callback, function-pointer, trait, macro, adapter-import, and `cfg(test)` classes; and
6. domain behavioral determinism, denial, replay, parity, and host-action instrumentation.

Each change manifest states which layers it uses, the threat model, and blind spots. Dependency and source checks are guardrails. The checker rejects a claim of arbitrary macro expansion, dynamic-dispatch, transitive side-effect, or future-syntax completeness.

The harness may share manifest parsing, canonical identity test helpers, fixture execution, and report types. Production domains retain separate enums, policies, protocols, and result algebras.

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
- **Large initial registration cost:** 79 requirements and 200 scenarios need stable IDs and fixture mappings. Registration is performed before domain implementation and may land in bounded change-by-change slices.
- **Checker overreach:** static checks can create false purity confidence. Reports expose enforcement layers and blind spots, and behavioral gates remain authoritative.
- **Command registry rigidity:** new slices require manifest and checker updates. This is intentional so an undocumented command cannot become completion evidence.
- **Evidence infrastructure in the workspace:** the checker adds compile cost. It remains dependency-minimal, has no production runtime dependency edge, and can be built/tested independently.
- **OpenSpec schema evolution:** schema changes require a new manifest schema version and migration tests; unknown future fields fail closed rather than being ignored.

## Migration Plan

1. Add failing manifest-schema, graph, traceability, report-algebra, tripwire, and command-plan tests.
2. Create the pure checker model/validation/report core and loader adapter; pass `--slice schema`.
3. Register stable IDs and traceability for the three narrow security prerequisite changes first; pass `--slice traceability-security`.
4. Register lint and compiler, then evaluator and proof, then Reef, preserving the delivery order in `FCIS_ARCHITECTURE.md`; pass each focused registration slice.
5. Add layered architecture fixtures and threat-model validation; pass `--slice architecture`.
6. Add the Python orchestrator and tests for argv-only execution, unknown IDs, missing evidence, report invariants, prerequisite execution, and deterministic ordering; pass `--slice runner`.
7. Add Markdown/manifests/CI tripwires and update every active change to depend on the mechanics oracle before implementation.
8. Run the final oracle. Domain changes remain blocked until their manifest and failing contract fixtures are registered.

Rollback removes only evidence tooling and cannot be used to claim an FCIS domain complete. Once a domain implementation begins under the manifest contract, weakening or removing its registered evidence requires a separately reviewed OpenSpec change.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py fcis-contract
```

The runner must validate schema rejection and normalization, complete active-change registration, acyclic prerequisite/slice graphs, stable requirement/scenario IDs, positive/negative parity, scenario-to-fixture-to-slice-to-final-oracle reachability, exact oracle command tripwires, argv-only command plans, deterministic report ordering, success/error algebra, layered architecture threat-model declarations, and negative fixtures for duplicate, unknown, missing, stale, cyclic, shell-string, contradictory-report, and false-completion cases. Success means exit status 0, a `Passed` report with an empty error list, every active FCIS change registered, every active scenario reachable from its final oracle, no unimplemented evidence counted as skipped or green, and no focused slice represented as final completion evidence.
