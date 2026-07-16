## 1. Lock Collection And Analysis Contracts With Tests

- [ ] 1.1 Add positive and negative snapshot fixtures for path ordering, ignore policy, symlink containment, decoding, size bounds, unreadable entries, invalid text, and logical paths
- [ ] 1.2 Add equal-snapshot and host-change-after-collection determinism tests
- [ ] 1.3 Add rule-descriptor completeness tests, including a named failure when required snapshot data is absent
- [ ] 1.4 Add shared opaque-type and Cargo-package index completeness/conflict-order tests
- [ ] 1.5 Add lint parity fixtures for codes, severities, paths, ordering, exceptions, clean examples, and negative cross-file cases
- [ ] 1.6 Add CLI tests pinning blocking collection-diagnostic exit and machine-output behavior
- [ ] 1.7 Add legacy-rule positive compatibility and negative not-counted-as-pure evidence tests
- [ ] 1.8 Add architecture fixtures for direct, aliased, re-exported, qualified, callback-hidden, trait-hidden, and `cfg(test)` host capabilities
- [ ] 1.9 Commit all stubs and verify intended hidden-I/O, missing-input, escape, unreadable, invalid, and legacy-boundary cases fail before implementation

## 2. Introduce Immutable Workspace Snapshots

- [ ] 2.1 Define immutable logical path, normalized snapshot entry, source buffer, required metadata, policy, and collection diagnostic types
- [ ] 2.2 Add serializable rule requirement descriptors for selected file classes, decoded text, directory entries, Cargo metadata, and shared indexes
- [ ] 2.3 Implement one path-sorted collector with explicit ignore, symlink, UTF-8 decoding, size, classification, and containment policies
- [ ] 2.4 Return unreadable, invalid, escaped, oversized, and rejected entries as structured diagnostics with pinned blocking/advisory classification
- [ ] 2.5 Define `LintQueryKey` and `LintOutcomeDigest` without absolute host root or renderer metadata

## 3. Migrate Pure Lint Analysis

- [ ] 3.1 Add versioned `SnapshotRule` and `SnapshotRuleContext` without a reopenable root
- [ ] 3.2 Add `lint_snapshot` and adapt normal path-based entry points to collect once and delegate
- [ ] 3.3 Build opaque-type, Cargo-package, and other cross-file indexes once from the complete snapshot
- [ ] 3.4 Migrate every built-in rule to supplied entries/indexes and remove built-in traversal and reads
- [ ] 3.5 Preserve canonical violation ordering and inline/registry exception behavior

## 4. Preserve A Bounded Compatibility Window

- [ ] 4.1 Keep the existing `Rule`/`Context` API for one documented minor version behind clearly named `lint_legacy_path_rules`
- [ ] 4.2 Deprecate the legacy API and document that arbitrary legacy rules are outside FCIS guarantees and acceptance evidence
- [ ] 4.3 Ensure no built-in rule or normal CLI lint path uses the legacy adapter

## 5. Lock The Architecture

- [ ] 5.1 Check in the exact core/collector/legacy module manifest
- [ ] 5.2 Reject filesystem, traversal, environment, process, network, clock, terminal, callback, and forbidden-trait capabilities from designated analysis modules
- [ ] 5.3 Verify test-only I/O is classified by `cfg(test)` body without exempting production code

## 6. Documentation And Acceptance

- [ ] 6.1 Update lint architecture, rule-authoring, public migration, collection-diagnostic, query-key, and current-state docs
- [ ] 6.2 Update CLI machine-output fixtures and executable examples affected by collection diagnostics
- [ ] 6.3 Implement `.venv/bin/python scripts/fcis_gate.py lint-snapshots`
- [ ] 6.4 Run `.venv/bin/python scripts/fcis_gate.py lint-snapshots` and require exit 0, an empty error list, one collection per invocation, and zero built-in-rule host actions
- [ ] 6.5 Run the repository local gate when implementation begins and record any CI-owned workspace evidence required for completion
