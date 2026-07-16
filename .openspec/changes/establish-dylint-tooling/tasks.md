## 1. Bootstrap contract and failing tests

- [ ] 1.1 Add failing `.venv/bin/python` unit tests for `scripts/dylint_gate.py` covering `Passed`/`Failed`/`Blocked` invariants, canonical ordering, missing tools/components/fixtures, pin drift, root-workspace contamination, focused-run non-completion, interruption, and tracked-checkout mutation
- [ ] 1.2 Add strict positive and negative fixtures for `tools/dylint/pins.toml` and `tools/dylint/gate.toml`, including unknown fields/tags, unsupported versions, duplicate IDs, noncanonical order, missing scenario reachability, shell command text, undeclared environment, and zero registered checks
- [ ] 1.3 Before lint implementation, add allowed-positive and violating-negative `dylint_testing` UI fixture stubs and expected diagnostic goldens for every claimed direct, qualified, alias, re-export, associated/UFCS, function-item, callback/function-pointer, configured trait/type, visible declarative-macro, adapter-reference, ambient-state, public-interface, production-in-test-build, and true-`cfg(test)` detector class
- [ ] 1.4 Add failing live-probe fixtures for every configured forbidden identity, all declared library/test/feature/configuration lanes, unresolved identities, omitted lanes, unmatched exceptions, zero protected production items, and zero executed detector classes
- [ ] 1.5 Add failing `NoFix` fixtures asserting every initial diagnostic emits no machine-applicable suggestion, `cargo dylint --fix` leaves disposable source unchanged, and the diagnostic remains reproducible
- [ ] 1.6 Add failing registry/tripwire fixtures for duplicate diagnostic/lint IDs, missing detector polarity, stale goldens, unregistered production-loadable passes, per-lint crate proliferation, unjustified category splits, duplicated domain policy, and unsupported completeness claims

## 2. Isolated workspace and exact pins

- [ ] 2.1 Create the nested `tools/dylint` workspace, `fcis-boundaries` `cdylib` package, dated `rust-toolchain.toml`, `pins.toml`, `gate.toml`, fixture directories, and workspace-local documentation without adding a root workspace member or dependency
- [ ] 2.2 Pin `cargo-dylint = 6.0.1`, `dylint-link = 6.0.1`, `dylint_linting = 6.0.1`, `dylint_testing = 6.0.1`, `nightly-2026-04-16`, `rustc-dev`, and `llvm-tools-preview`; generate and commit the nested lockfile with all remaining exact dependency resolution
- [ ] 2.3 Configure Dylint library linking and test support under the nested nightly while keeping root `rust-toolchain.toml`, root dependency resolution, and ordinary stable Cargo commands unchanged
- [ ] 2.4 Add pin/toolchain/root-isolation validation and explicit blocked diagnostics with exact installation guidance; never install, download, or upgrade tooling from the acceptance oracle
- [ ] 2.5 If static Dylint discovery metadata is required, add only the library path and prove it contains no domain policy or dependency edge; otherwise run all commands with an explicit library path

## 3. Typed registry and strict configuration

- [ ] 3.1 Implement the single typed `chelis-fcis-boundaries` registry for schema version, stable diagnostic IDs, rustc lint names, detector classes, production loadability, and `FixPolicy`
- [ ] 3.2 Implement strict `schema_version = 1` `DYLINT_TOML` parsing for protected scopes, adapters, forbidden resolved identities/capability classes, capability-bearing types/traits, exact item exceptions, and minimum production-item counts with unknown-field denial
- [ ] 3.3 Reject absent/unsupported/malformed configuration, duplicate IDs, contradictory ownership, invalid scopes, unmatched exceptions, default-allow fallback, and any checked-in compiler/evaluator/proof/lint/Reef/conformance domain policy
- [ ] 3.4 Implement deterministic owner/scope indexing and true owner-based `cfg(test)` classification so production items compiled in test targets remain protected
- [ ] 3.5 Implement live identity-probe and scope-count accounting that reports every configured entry, matched owner, exception, and production-item count and fails unresolved or vacuous configuration
- [ ] 3.6 Emit stable diagnostic metadata for configuration-entry ID, detector class, protected owner, and semantic message key without using source spelling as identity

## 4. FCIS boundary detector implementation

- [ ] 4.1 Implement resolved forbidden API/type/macro reference detection for the registered direct, qualified, alias, re-export, associated/UFCS, and visible declarative-macro fixture forms
- [ ] 4.2 Implement forbidden function-item escape detection for assignment, return, argument, callback, and function-pointer fixture forms without flagging same-spelling allowed local functions
- [ ] 4.3 Implement protected-core-to-adapter reference detection for registered import, path, type, trait, and function-item fixture forms
- [ ] 4.4 Implement mutable-static, thread-local, and configured interior-mutable ambient-state detection while allowing invocation-owned local mutation and unconfigured immutable constants/statics
- [ ] 4.5 Implement configured capability-bearing public-interface detection for callback, function-pointer, trait/type, and capability-handle signatures while preserving registered non-capability/sealed cases
- [ ] 4.6 Make every allowed-positive and violating-negative UI fixture pass with exact diagnostic ID, class, configuration entry, owner, and primary-span goldens; do not broaden claims beyond passing fixtures

## 5. Live lanes and deterministic evidence

- [ ] 5.1 Run every canonical fixture package/library/test/feature/configuration lane through the exact pinned `cargo dylint` library and prove production-in-test-build versus true-test-owner behavior
- [ ] 5.2 Make unresolved identities, missing lanes, unmatched exceptions, under-minimum production scopes, zero production diagnostics, and zero executed detector classes return structured non-green results
- [ ] 5.3 Capture rustc JSON diagnostics and implement narrow normalization plus canonical ordering by lane, diagnostic ID, detector class, configuration-entry ID, owner, span, and message key
- [ ] 5.4 Add equal-input/different-disposable-root goldens and negative tests proving applicability, kind, owner, span, configuration ID, count, and failures cannot be normalized away
- [ ] 5.5 Generate the scoped architecture evidence summary with exact pins, lanes, resolved entries, matched counts, detector/fix states, proven fixture forms, and explicit inactive-cfg, proc-macro/build-script, dynamic-dispatch, precompiled-dependency, and future-syntax blind spots

## 6. Fix-denial and future fix-safety harness

- [ ] 6.1 Register every initial production diagnostic as `FixPolicy::NoFix` and make UI and live JSON assertions reject any `Applicability::MachineApplicable` suggestion
- [ ] 6.2 Exercise `cargo dylint --fix` only against disposable copies of initial violating fixtures and prove source bytes remain unchanged, the diagnostic remains reproducible, and no writes escape the disposable root
- [ ] 6.3 Implement and unit-test the registration validator for a future `MachineApplicable` fix ID, requiring detector-green positive exact-rewrite and negative no-machine-fix fixtures, formatting/compilation, clean same-lint rerun, domain parity, second-run idempotence, and checkout immutability before acceptance
- [ ] 6.4 Add negative registration/runner fixtures for missing polarity, ambiguous rewrites, lint-dirty output, behavior change, non-idempotence, lint suppression, FCIS/Dylint policy edits, boundary escape, exception widening, and test/evidence deletion

## 7. Standalone prerequisite oracle

- [ ] 7.1 Implement strict parsing and canonical validation of `pins.toml` and `gate.toml`, including complete requirement/scenario/fixture/check reachability to the sole `dylint-tooling` oracle
- [ ] 7.2 Implement `scripts/dylint_gate.py` with argv-only `shell=False` execution, an explicit environment allowlist, isolated `CARGO_TARGET_DIR`, disposable fixture roots, fail-closed command handling, and no auto-provisioning
- [ ] 7.3 Implement the typed deterministic `Passed { errors: [] }`, `Failed { errors: NonEmpty }`, and `Blocked { errors: NonEmpty }` report and exclude absolute roots, durations, temporary paths, and process IDs from semantic equality
- [ ] 7.4 Implement tracked-file pre/post state capture that detects content, mode, index, rename, deletion, and untracked writes outside declared disposable/target roots and cannot convert detected mutation to success by reverting afterward
- [ ] 7.5 Implement focused developer selectors as supporting commands that cannot emit final completion, while the no-selector invocation executes every registered check
- [ ] 7.6 Make all Python runner positive/negative tests pass, including missing executable/component, malformed JSON, stale golden, command failure, interruption, escaped write, report contradiction, and stable ordering cases

## 8. Documentation, CI provisioning, and downstream handoff

- [ ] 8.1 Document workspace layout, exact provisioning commands, pin upgrades, diagnostic/category registration, strict configuration, detector fixture authoring, `NoFix` defaults, future disposable fix workflow, blind spots, and troubleshooting
- [ ] 8.2 Add CI provisioning for the exact `cargo-dylint` and nightly components plus the standalone oracle without adding Dylint to the normal stable gate or allowing floating upgrades
- [ ] 8.3 Add tripwires proving all current domain OpenSpecs name `establish-fcis-contract-mechanics` as their boundary-evidence owner and do not instantiate another Dylint library or handwritten boundary configuration
- [ ] 8.4 Document the downstream handoff: contract mechanics must run this oracle on the same revision, consume the registered schema/diagnostics, project FCIS manifests into `DYLINT_TOML`, and own domain lanes/fix registrations without changing this prerequisite's independent completion claim
- [ ] 8.5 Verify root stable formatting, clippy/build/test behavior and shipped package contents are unchanged by the nested tooling workspace

## 9. Acceptance and adversarial validation

- [ ] 9.1 Run every focused tooling check and confirm each is supporting evidence only, with no partial command able to emit final `Passed`
- [ ] 9.2 Run a fresh local red-team agent against the spec, registry, lint code, allowed/violating fixtures, malformed/vacuous configurations, JSON normalization, NoFix behavior, checkout isolation, and root stable-workspace boundary; add and execute adversarial fixtures for every valid finding
- [ ] 9.3 Run the sole final oracle `.venv/bin/python scripts/dylint_gate.py` and require exit 0, `Passed` with an empty error list, exact pins, all registered detector classes in both polarities, every declared lane and live probe, zero unresolved/vacuous entries, deterministic JSON, all initial diagnostics `NoFix`, and an unchanged tracked checkout
- [ ] 9.4 Re-run `openspec validate --all --strict`, documentation checks, and repository diff hygiene, and require no downstream FCIS change to be marked implementation-ready before the `dylint-tooling` oracle is green
