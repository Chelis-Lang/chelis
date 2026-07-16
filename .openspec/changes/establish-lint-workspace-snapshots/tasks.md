## 0. Register Mechanical Evidence

- [ ] 0.1 Register the change, exact pure/collector/legacy boundaries, stable requirement/scenario IDs, explicit polarity, and complete fixture-to-slice-to-final-oracle traces in the FCIS contract manifest
- [ ] 0.2 Pin exact v1 per-entry, entry-count, path/locator, and total-snapshot bounds plus the total representable/unrepresentable locator order; absent bounds block collector implementation
- [ ] 0.3 Register one typed rule owner for order, severity, blocking status, surfaces, data/index requirements, and legacy status, with tripwires for CLI/style-gate projections
- [ ] 0.4 Register the exact collection/analysis failure projection table and architecture threat-model layers
- [ ] 0.5 Make `.venv/bin/python scripts/fcis_gate.py lint-snapshots --slice contracts` exist and fail on current hidden-I/O, silent-skip, missing-input, and tombstone fixtures
- [ ] 0.6 Pass the owning FCIS contract registration/traceability slice before snapshot implementation

## 1. Lock Collection And Analysis Contracts With Tests

- [ ] 1.1 Add positive and negative snapshot fixtures for path ordering, ignore policy, handle-relative symlink containment, non-UTF-8 diagnostic locators, entry replacement during collection, decoding, size bounds, unreadable entries, invalid text, and logical paths
- [ ] 1.2 Add equal-snapshot and host-change-after-collection determinism tests, including structural inequality for unequal rejected-entry tombstones and equality despite absolute-root or locale-dependent OS-report differences
- [ ] 1.3 Add rule-descriptor completeness tests, including a named failure when required snapshot data is absent
- [ ] 1.4 Add shared opaque-type and Cargo-package index completeness/conflict-order tests
- [ ] 1.5 Add lint parity fixtures for codes, severities, paths, ordering, exceptions, clean examples, and negative cross-file cases
- [ ] 1.6 Add CLI tests pinning blocking collection-diagnostic exit and machine-output behavior
- [ ] 1.7 Add legacy-rule positive compatibility and negative not-counted-as-pure evidence tests
- [ ] 1.8 Before lint-core migration, add domain allowed-positive and violating-negative fixtures/live probes for each claimed prerequisite-supported direct, aliased, re-exported, qualified, function-item, callback, declarative-macro-expanded, trait-hidden, collector/legacy-adapter-reference, entropy, scheduling, unsafe-FFI, mutable-global, production-in-test-build, and classified `cfg(test)` form plus allowed invocation-local mutation; register every domain use as `NoFix` and reject unknown diagnostics/classes or unsupported fix IDs
- [ ] 1.9 Commit all stubs and verify intended hidden-I/O, missing-input, escape, unreadable, invalid, and legacy-boundary cases fail before implementation
- [ ] 1.10 Add the `--slice contracts` runner around the failing contract fixtures before snapshot implementation

## 2. Introduce Immutable Workspace Snapshots

- [ ] 2.1 Define immutable logical path, diagnostic-only raw-component `HostEntryLocator`, normalized snapshot entry, rejected-entry tombstone, source buffer, required metadata, policy, and canonical collection diagnostic types
- [ ] 2.2 Add serializable rule requirement descriptors for selected file classes, decoded text, directory entries, Cargo metadata, and shared indexes
- [ ] 2.3 Implement one path-sorted collector with explicit ignore, handle-relative symlink, stable-open, non-UTF-8 path, UTF-8 decoding, size, classification, and containment policies
- [ ] 2.4 Return unreadable, invalid, escaped, oversized, and rejected entries as structured diagnostics with pinned blocking/advisory classification
- [ ] 2.5 Define structural snapshot equality over successful entries, canonical diagnostics, and rejected-entry tombstones while excluding absolute roots, locale-dependent OS text, and renderer metadata; expose no persistent cache/hash contract
- [ ] 2.6 Run `.venv/bin/python scripts/fcis_gate.py lint-snapshots --slice contracts` and require green before pure-rule migration

## 3. Migrate Pure Lint Analysis

- [ ] 3.1 Create dependency-minimal `chelis-lint-core` and add versioned `SnapshotRule` and `SnapshotRuleContext` without a reopenable root or host-capability callback
- [ ] 3.2 Add `lint_snapshot` and adapt normal path-based entry points to collect once and delegate
- [ ] 3.3 Build opaque-type, Cargo-package, and other cross-file indexes once from the complete snapshot
- [ ] 3.4 Migrate every built-in rule to supplied entries/indexes and remove built-in traversal and reads
- [ ] 3.5 Preserve canonical violation ordering and inline/registry exception behavior
- [ ] 3.6 Implement/extend and run `.venv/bin/python scripts/fcis_gate.py lint-snapshots --slice analysis`

## 4. Preserve A Bounded Compatibility Window

- [ ] 4.1 Keep the existing `Rule`/`Context` API for one documented minor version behind clearly named `lint_legacy_path_rules`
- [ ] 4.2 Deprecate the legacy API and document that arbitrary legacy rules are outside FCIS guarantees and acceptance evidence
- [ ] 4.3 Ensure no built-in rule or normal CLI lint path uses the legacy adapter

## 5. Lock The Architecture

- [ ] 5.1 Enforce the `chelis-lint-core` dependency allowlist and check in the temporary compatibility-facade collector/legacy module manifest
- [ ] 5.2 Run the manifest-configured accepted-prerequisite Dylint layer for the declared `chelis-lint-core` library/test lanes and require both detector polarities before rejecting the specifically documented direct, alias, re-export, qualified, function-item, callback, declarative-macro-expanded, forbidden-trait, and collector/legacy-adapter references for filesystem, traversal, environment, process, network, clock, terminal, entropy, scheduling, unsafe-FFI, and mutable-global capabilities
- [ ] 5.3 Fail on unresolved configured entries, zero matched production-core items, a missing declared lane, missing detector polarity, or unknown/prerequisite-unsupported/unsafe/non-idempotent fix evidence; verify test-only I/O is classified by `cfg(test)` owner without exempting production code, exercise any prerequisite-supported fix only in disposable copies with compile/lint/parity/idempotence and checkout-immutability checks, and document active-cfg, proc-macro/build-script, arbitrary-dynamic-dispatch, and transitive-dependency blind spots

## 6. Documentation And Acceptance

- [ ] 6.1 Update lint architecture, rule-authoring, public migration, collection-diagnostic, structural-equality/no-cache, and current-state docs
- [ ] 6.2 Update CLI machine-output fixtures and executable examples affected by collection diagnostics
- [ ] 6.3 Consolidate the already landed focused commands and implement the final `.venv/bin/python scripts/fcis_gate.py lint-snapshots` oracle
- [ ] 6.4 Run each focused command before landing its slice; do not claim change completion until the final `.venv/bin/python scripts/fcis_gate.py lint-snapshots` exits 0 with an empty error list, one snapshot collection phase per invocation, zero built-in-rule host actions, and tombstone-sensitive structural equality
- [ ] 6.5 Run the repository local gate when implementation begins and record any CI-owned workspace evidence required for completion
