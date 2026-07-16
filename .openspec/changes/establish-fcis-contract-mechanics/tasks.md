## 1. Lock The Evidence Contract With Failing Tests

- [ ] 1.1 Add positive and negative schema fixtures for every required manifest field, Dylint diagnostic/detector/fix registration, enum tag, unknown field, unsupported version, duplicate ID, and unknown reference
- [ ] 1.2 Add graph fixtures for canonical independent-node ordering, valid prerequisites, self-edges, unknown nodes, duplicate edges, prerequisite cycles, slice cycles, and final-oracle slice omission
- [ ] 1.3 Add stable requirement/scenario ID, explicit polarity, positive/negative parity, orphan fixture, missing target/path, duplicate ownership, and complete scenario-to-final-oracle reachability fixtures
- [ ] 1.4 Add command-plan fixtures for argv arrays, canonical order, workspace-relative directories, declared environment and read-only/disposable-fixture mutation policy, tracked-checkout pre/post verification, unknown oracle/slice IDs, shell strings, pipelines, redirections, substitutions, checkout escape, read-only `--fix`, and undeclared disposable destinations
- [ ] 1.5 Add report-algebra tests proving `Passed` has no errors, failed/blocked reports have nonempty errors, and contradictory adapter input is rejected
- [ ] 1.6 Add deterministic report fixtures varying TOML declaration order, absolute checkout root, locale-dependent raw error text, duration, and terminal mode while preserving byte-equal semantic JSON
- [ ] 1.7 Add planned/missing evidence fixtures proving unimplemented, absent, skipped, stale, and not-applicable-without-review cases remain non-green
- [ ] 1.8 Add prerequisite freshness fixtures proving a child final oracle uses the current revision's prerequisite regression and cannot trust a stale evidence record
- [ ] 1.9 Add architecture declaration fixtures for dependency allowlists, module manifests, the accepted `dylint-tooling` prerequisite report/schema/diagnostic/fix registry and exact pins, package/target/feature/configuration lanes, manifest-derived lint policy, public callback/trait policy, resolved-entry and matched-core counts, typed detector contracts with positive/negative polarity, supported typed fix policies, proven fixture classes, blind spots, and false lexical/resolved-lint/transitive completeness claims
- [ ] 1.10 Add table-tripwire fixtures for evaluator builtins, lint rules, compiler surface/targets, proof authorization, Reef actions/locks/reports, identity domains, and deliberate one-sided drift
- [ ] 1.11 Add dependency fixtures proving domain production crates cannot import generic request/observation/workflow runtime types from the evidence crate
- [ ] 1.12 Commit the test stubs and verify every intended duplicate, missing, cyclic, shell-string, stale, contradictory, false-completion, missing/stale Dylint prerequisite or schema/registry mismatch, unresolved-Dylint-entry, zero-matched-core, omitted-compilation-lane, missing detector polarity, unsupported/unsafe/non-idempotent fix, fix-caused checkout mutation, and false-purity case fails before implementation
- [ ] 1.13 Register `.venv/bin/python scripts/fcis_gate.py fcis-contract --slice contracts` around the failing contract fixtures before implementing the checker

## 2. Build The Deterministic Contract Checker

- [ ] 2.1 Create dependency-minimal `chelis-fcis-contract` with separate pure `model`, `normalize`, `graph`, `traceability`, `tripwire`, `plan`, and `report` modules plus outer loading/binary adapters
- [ ] 2.2 Define strict schema-v1 registry and per-change manifest types with unknown-field denial and typed enums for change kind, capability class, enforcement layer, Dylint detector class, fix policy/applicability, command mutation policy, protocol class, identity class, fixture kind, polarity, slice, and oracle
- [ ] 2.3 Implement pure manifest normalization with canonical bytewise UTF-8 ordering and explicit preservation of separately declared semantic orders
- [ ] 2.4 Implement prerequisite/slice graph validation and canonical topological ordering with stable cycle diagnostics
- [ ] 2.5 Implement requirement-to-scenario-to-fixture-to-slice-to-final-oracle traceability and positive/negative parity validation
- [ ] 2.6 Implement existence-observation validation without reading the filesystem from checker-core modules
- [ ] 2.7 Implement typed `GateReport` constructors that make passed-with-errors and failed/blocked-without-errors unrepresentable
- [ ] 2.8 Implement canonical structured JSON rendering with report metadata excluded from semantic equality
- [ ] 2.9 Add the loader adapter for manifest/artifact bytes, projected Cargo metadata, fixture observations, and optional report metadata
- [ ] 2.10 Run `.venv/bin/python scripts/fcis_gate.py fcis-contract --slice schema` and require green before active-change registration

## 3. Register Active Changes And Traceability

- [ ] 3.1 Add `.openspec/fcis/registry.toml` and schema-v1 per-change manifest templates for every active FCIS change, including the independently accepted Dylint prerequisite
- [ ] 3.2 Register `dylint-tooling` as the root prerequisite with its standalone command, accepted schema/diagnostic/fix registry, current-revision freshness rule, and no dependency on `fcis-contract`; pass `--slice traceability-dylint`
- [ ] 3.3 Assign stable requirement/scenario IDs and explicit polarity to the three narrow security changes: proof forced-result removal, Tide evaluator denial, and explicit Reef credentials
- [ ] 3.4 Map every security scenario to positive/negative fixture IDs and the independently owned final oracle; pass `--slice traceability-security`
- [ ] 3.5 Assign IDs and fixture/slice traces for lint snapshots and compiler core; preserve the compiler surface/target semantic matrix and lint collection-correction corpus
- [ ] 3.6 Pass `--slice traceability-lint-compiler` with no orphan scenario, fixture, slice, or final-oracle edge
- [ ] 3.7 Assign IDs and fixture/slice traces for evaluator effects and proof execution, including prerequisite regression ownership and no-persistent-cache negative surfaces
- [ ] 3.8 Pass `--slice traceability-evaluator-proof` with complete protocol, denial, trust, replay, and public-surface matrices
- [ ] 3.9 Assign IDs and fixture/slice traces for Reef workflows, including read-only, protocol, local-install, conformance, source-wiring, remote-publish, credential regression, transaction, and report-state coverage
- [ ] 3.10 Pass `--slice traceability-reef` with every command family reachable from the final Reef oracle
- [ ] 3.11 Add tripwires ensuring `openspec validate --all --strict` capability names and manifest capability IDs agree

## 4. Integrate Layered Architecture Evidence

- [ ] 4.1 Implement deterministic `cargo metadata` projection and per-core dependency allowlist validation
- [ ] 4.2 Implement exact production-module manifest checking for mixed crates without treating the manifest as a crate boundary
- [ ] 4.3 Execute `.venv/bin/python scripts/dylint_gate.py` against the current revision and validate its `Passed`/empty-error result, exact accepted pins, root-stable isolation, configuration schema version, diagnostic/detector/fix registry, and complete foundational fixture state; missing, stale, failed, blocked, or mismatched prerequisite evidence fails closed
- [ ] 4.4 Add positive and negative round-trip fixtures from typed FCIS manifest boundary policy into the prerequisite's strict `DYLINT_TOML` schema, rejecting unknown fields/diagnostics/classes/fix IDs, unsupported versions, semantic loss, handwritten domain configuration, and policy divergence
- [ ] 4.5 Implement manifest-derived Dylint command plans for declared domain package/target/feature/configuration lanes and domain allowed-positive/violating-negative fixture/live-probe registrations; unresolved identities, unmatched exceptions, zero matched production items, missing polarity, and omitted lanes fail closed
- [ ] 4.6 Validate every domain registration as `NoFix` or a prerequisite-supported machine-applicable fix ID with disposable exact-rewrite/no-fix, formatting/compilation, clean-rerun, domain-parity, idempotence, and checkout-immutability plans; do not implement or repin lint suggestions in this change
- [ ] 4.7 Ensure every architecture report includes prerequisite oracle/schema/registry state, dependency/module enforcement, accepted Dylint/toolchain/dependency pins, package/target/feature/configuration lanes, resolved configured entries, matched production-core counts, normalized diagnostic IDs, detector-contract state, fix policy and fix-fixture/parity state, threat model, proven fixture classes, and known blind spots
- [ ] 4.8 Add negative tests rejecting stale prerequisite reports, schema/registry drift, competing Dylint libraries, handwritten domain policy, and lexical scans, Dylint, or bounded fixtures described as complete transitive purity proof; record blind spots for unexecuted cfg/target/feature code, procedural-macro/build-script implementation effects, arbitrary dynamic dispatch, and precompiled dependency behavior
- [ ] 4.9 Run `.venv/bin/python scripts/fcis_gate.py fcis-contract --slice architecture`

## 5. Implement The Canonical FCIS Runner

- [ ] 5.1 Implement `scripts/fcis_gate.py` with the uv-managed Python, argv-only subprocess execution, no shell, repository-root resolution, and oracle/slice choices obtained from the validated Rust command plan
- [ ] 5.2 Register every active final oracle and focused slice before domain implementation; missing implementations return structured non-green evidence
- [ ] 5.3 Make `fcis-contract` execute/include the current-revision standalone `dylint-tooling` oracle and make child final oracles execute/include their current prerequisite regression suites without transferring completion ownership
- [ ] 5.4 Implement deterministic command/check aggregation and typed report assembly through the Rust checker rather than duplicating success semantics in Python
- [ ] 5.5 Add Python tests for unknown IDs, malformed plans, command failure, missing executable/fixture, environment policy, directory escape, read-only `--fix`, undeclared disposable fixture mutation, tracked-checkout drift, interruption, report contradictions, and stable ordering
- [ ] 5.6 Add `--list` and `--list --json` views generated from the same registry and mark focused versus final oracles explicitly
- [ ] 5.7 Run `.venv/bin/python scripts/fcis_gate.py fcis-contract --slice runner`

## 6. Single-Source Mechanical Registries And Tripwires

- [ ] 6.1 Define the owner/tripwire contract for evaluator host-capable builtin effect, evaluator policy, request/event mapping, and target support
- [ ] 6.2 Define one typed lint rule registry carrying dispatch order, severity, blocking status, surface requirements, data requirements, and index requirements
- [ ] 6.3 Define one typed compiler surface/operation/target/preflight capability matrix and tripwire the OpenSpec table and acceptance corpus
- [ ] 6.4 Define the proof engine descriptor/authorization registry contract without allowing adapter descriptors to self-authorize
- [ ] 6.5 Define the Reef request/action, lock-rank, transaction-state, remote-state, and report-state closed-vocabulary owners
- [ ] 6.6 Define one registry of shipped identity domain strings, versions, field tags, canonical encoders, and golden vectors while recording explicit `none` for deferred identities
- [ ] 6.7 Add negative projection-drift fixtures and run `.venv/bin/python scripts/fcis_gate.py fcis-contract --slice registries`

## 7. Synchronize OpenSpec And Evidence Claims

- [ ] 7.1 Update `.openspec/config.yaml` and `.openspec/FCIS_ARCHITECTURE.md` with the Dylint-first prerequisite order, manifest, stable-ID, traceability, command-registry, report-algebra, prerequisite-freshness, typed-registry, accepted pinned Dylint architecture evidence, and exact-bound requirements
- [ ] 7.2 Update every active change proposal/design/spec/tasks to depend transitively on `dylint-tooling` through `fcis-contract`, name its manifest and registry owners, and distinguish planned, focused, prerequisite, and final evidence
- [ ] 7.3 Add exact decision/failure projection tables to compiler, evaluator, proof, and Reef manifests and tripwire their design/spec prose
- [ ] 7.4 Require exact v1 payload, entry-count, path/argument, transcript, and total-memory bounds in every domain manifest before its protocol or collector implementation can begin
- [ ] 7.5 Require canonical total orders for every returned or identity-bearing collection, including lint rejected-entry locators
- [ ] 7.6 Require every shipped identity input to be core-computed or validated against the value it identifies; reject caller-asserted prepared-program/blob/descriptor identities
- [ ] 7.7 Run `openspec validate --all --strict` and the complete FCIS contract traceability report

## 8. Documentation And Acceptance

- [ ] 8.1 Document manifest schema evolution, ID conventions, fixture kinds, consumption of the standalone Dylint prerequisite, domain detector trust registrations, `NoFix`/supported-machine-fix binding and disposable `--fix` workflow, configuration-projection and compilation-matrix procedures, architecture threat-model declarations, runner output, prerequisite freshness, and focused-versus-final evidence
- [ ] 8.2 Document how a new FCIS change, requirement, scenario, fixture, slice, capability table, and identity domain is registered
- [ ] 8.3 Add current-state documentation stating that registered-but-unimplemented domain evidence remains blocked and is not part of the normal green repository gate
- [ ] 8.4 Consolidate all focused mechanics slices into `.venv/bin/python scripts/fcis_gate.py fcis-contract`
- [ ] 8.5 Run the final oracle and require exit 0, a `Passed` report with an empty error list, a current-revision green `dylint-tooling` prerequisite with matching schema/registry/pins, complete active-change registration and traceability, no shell-string or read-only-`--fix` plans, complete domain detector polarity and supported fix evidence, no tracked-checkout mutation from fix tests, no stale prerequisite acceptance, no unimplemented evidence counted green, and no focused slice counted as final completion
- [ ] 8.6 Run the repository local gate and record any CI-owned workspace evidence required for completion
