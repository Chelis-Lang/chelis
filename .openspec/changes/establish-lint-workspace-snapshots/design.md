## Context

`chelis-lint::Context` currently exposes `root`, `path`, optional source text, and surface. The main driver walks and reads each entry, while cross-file rules such as opaque-domain and documentation naming rules perform additional directory walks and reads behind that context. Unreadable text is silently skipped. This prevents one lint run from having one authoritative content input.

This change is a one-shot FCIS migration: collect once in an imperative adapter, then analyze immutable data. It follows `.openspec/FCIS_ARCHITECTURE.md`.

## Goals / Non-Goals

**Goals:**

- Make built-in lint analysis a deterministic snapshot-to-diagnostics function.
- Ensure built-in rules and indexes cannot reopen or traverse the workspace.
- Make collection, decoding, path, symlink, and ignore policy explicit.
- Compute shared indexes once from the complete snapshot.
- Preserve public lint behavior except for explicitly specified collection diagnostics.
- Give existing external rule authors a documented migration window without claiming their legacy code is pure.

**Non-Goals:**

- Changing rule meaning or severity intentionally.
- Building a general virtual filesystem framework.
- Sharing lint snapshot types with Reef before concrete common requirements exist.
- Proving arbitrary out-of-tree legacy `Rule` implementations are side-effect free.

## Decisions

### 1. Collection and analysis are separate

`collect_workspace(request, policy)` is an adapter operation. It resolves one logical root and returns a `WorkspaceSnapshot` containing successful `SnapshotEntry` values and rejected-entry tombstones that carry canonical `CollectionDiagnostic` data. Entries are stable and path-sorted, with normalized logical paths, immutable bytes, optional decoded text, surface inputs, and declared metadata. Keeping failures inside structural snapshot equality prevents analysis from silently treating a partial collection as a complete collection.

`lint_snapshot(snapshot, rule_set, exceptions)` is the pure core. Host changes after collection are invisible to analysis.

### 2. Collection policy and compatibility corrections are explicit

`SnapshotPolicy` contains ignore rules, symlink/containment behavior, path normalization, decoding policy, selected file classes, and size limits. V1 uses the repository's current skip-directory policy. Logical paths are relative UTF-8 with `/` separators, preserved case, no `.`/`..` components, and bytewise UTF-8 ordering. A host path that cannot be represented losslessly in that form produces a blocking `non_utf8_path` tombstone rather than a lossy path. Because that entry has no logical UTF-8 path, its `HostEntryLocator` contains the normalized UTF-8 parent prefix, platform tag, and length-prefixed raw bytes of the first unrepresentable component; human rendering uses escaped hexadecimal bytes. This locator is diagnostic data, never a valid analysis path.

V1 does not follow directory symlinks. A selected file symlink may be read only through a containment-safe handle-relative open; its resolved target must remain under the logical root. If the platform cannot provide that guarantee, the symlink is rejected rather than canonicalized and reopened by path. The collector verifies entry identity and relevant metadata around each read; replacement or drift during collection produces a blocking `entry_changed_during_collection` tombstone and contributes no bytes. Required source text is decoded as UTF-8.

Current silently skipped unreadable or invalid required text becomes a structured blocking collection diagnostic. Non-UTF-8 paths, unstable entries, oversized required entries, and escaping or unsupported symlinks are also blocking. Ignored and unselected entries remain absent without a diagnostic. Canonical diagnostic equality contains only operation, normalized failure class, stable entry locator, and policy-relevant fields; locale-dependent `std::io::Error` text and raw absolute host paths are optional report metadata and do not affect analysis equality. CLI `lint --check` exits nonzero for a blocking collection diagnostic; advisory rule behavior is unchanged. These corrections receive dedicated CLI and machine-output fixtures.

### 3. Rule requirements and indexes are values

Each pure rule declares a serializable `RuleRequirements` descriptor naming file classes, decoded text, directory entries, Cargo package metadata, and shared indexes it needs. The collector unions descriptors before reading content. The analysis driver builds opaque-type, Cargo-package, and other catalogs once from the complete snapshot before the first rule executes.

Rule ordering is the explicit registry order; diagnostics are finally ordered by normalized path, line, rule id, and stable detail key. Equal snapshots, rule configurations, and exceptions produce equal output.

### 4. Public compatibility does not weaken the boundary

A new versioned `SnapshotRule` interface receives only `SnapshotRuleContext` and immutable indexes. The existing root-exposing `Rule`/`Context` API remains for one documented minor-version window behind `lint_legacy_path_rules`; it may collect or reopen host state and is explicitly outside the pure-core guarantee and acceptance evidence. Built-in rules migrate completely and never use the legacy adapter. The old interface is deprecated rather than silently redefined.

### 5. Exact boundary

The designated core is a planned dependency-minimal `crates/chelis-lint-core` crate containing snapshot values, analysis, indexes, the versioned pure rule interface, and every built-in rule after migration. The existing `chelis-lint` crate remains the compatibility facade and imperative shell for collection, the legacy root-based API, and fix execution; CLI style/lint orchestration and rendering remain outer adapters. `chelis-lint-core` has no dependency on `walkdir` or an adapter crate and exposes no callback or trait capable of obtaining host state.

The architecture gate uses Cargo dependency allowlists as its primary boundary and, through contract mechanics, the accepted `establish-dylint-tooling` prerequisite's pinned `chelis-fcis-boundaries` library as its resolved Rust layer. Manifest-derived Dylint policy checks every compiled core path/method/type reference for filesystem, traversal, environment, process, network, clock, terminal, entropy, scheduling, unsafe-FFI, and mutable-global capabilities; it also rejects forbidden function-item escape, references into collector/legacy adapter modules, and public callback/function-pointer/unsealed-trait inputs. The declared library and test lanes must resolve every configured entry and match production `chelis-lint-core` items or fail non-green.

Each lint registration must first pass allowed-positive and violating-negative Dylint UI detector fixtures for every claimed direct, aliased, re-exported, qualified, callback, function-pointer, trait, declarative-macro-expanded, adapter-reference, production-in-test-build, and classified `cfg(test)` form. Test-only bodies are classified without exempting production functions compiled in the same test lane. Each domain registration declares `NoFix`; it may reference a machine-applicable fix only after a separately accepted Dylint-tooling change supports that fix ID and `FCIS-EVIDENCE-011` validates its disposable positive rewrite, negative no-machine-fix, compile/lint/parity, idempotence, and checkout-immutability plans. The threat model does not claim unexecuted cfg/feature/target code, procedural-macro or build-script implementation effects, arbitrary dynamic dispatch, precompiled dependency behavior, or future Rust syntax. Behavioral snapshot determinism and host-change tests remain authoritative.

### 6. Snapshot equality, not persistent identity

This change introduces no persistent lint-result cache, replay record, query-key API, or stable hash encoding. `WorkspaceSnapshot` structural equality includes successful entries, canonical collection diagnostics, and rejected-entry tombstones. Equal snapshots, rule configurations, indexes, and exceptions must produce equal outcomes; unequal tombstones must make snapshots unequal. A future cache must define its identity and compatibility contract in a separate proposal after this boundary is accepted.

### 7. Rule registration and snapshot access are mechanically complete

One typed `RuleRegistration` registry owns rule ID, dispatch order, severity, blocking/advisory status, applicable surfaces, declared `RuleRequirements`, shared indexes, and compatibility status. The CLI, style gate, collector union, rule selection, docs, and acceptance corpus consume or tripwire that registry; separate blocking/advisory string or constructor lists are not authoritative.

`SnapshotRuleContext` does not expose optional data with empty/default fallback. Access to decoded text, directory entries, package metadata, or an index checks the active rule's declared requirements and returns a named `MissingDeclaredInput` contract failure if the snapshot cannot supply it. Snapshot entry variants distinguish required text, bytes, and directory data so a rule cannot silently interpret absent required source as an empty file.

Canonical locator order is total: representable logical paths sort first by their UTF-8 bytes and entry-kind rank; diagnostic-only unrepresentable locators sort afterward by normalized representable parent prefix, platform tag, raw-component length, and raw component bytes. Canonical diagnostic fields then break ties. Absolute root spelling and localized report text never participate.

Before collector implementation, the FCIS manifest pins exact v1 per-entry, entry-count, and total-snapshot bounds, registers stable requirement/scenario/fixture IDs, records the pure/legacy boundary, and makes the failing `lint-snapshots --slice contracts` runner executable. An absent bound or fixture blocks implementation rather than selecting an adapter default.

## Risks / Trade-offs

- **Snapshot memory:** collect only descriptor-required classes and share immutable buffers.
- **External rule compatibility:** the legacy adapter is clearly outside purity claims and expires after one minor version.
- **Collection diagnostics change exit behavior:** pin the correction with positive/negative CLI fixtures and active-doc updates.
- **Descriptor omissions can hide required data:** tests remove each declared requirement and require a named missing-input failure rather than a silent empty result.
- **Non-UTF-8 diagnostics lack a logical path:** use `HostEntryLocator` only for collection reporting and never admit it into rule analysis as a fabricated path.

## Migration Plan

1. Add snapshot policy, host-change, descriptor-completeness, collection-diagnostic, legacy-boundary, and parity tests.
2. Add snapshot and logical-path data types plus one path-sorted collector.
3. Add `SnapshotRule`, shared indexes, structural snapshot equality, and explicit no-persistent-cache scope.
4. Migrate built-in rules and remove all built-in hidden I/O.
5. Route path-based CLI entry points through collect-once analysis; retain only the named legacy adapter for external rules.
6. Add architecture gates, docs, examples, and the focused acceptance runner.

Rollback preserves the legacy adapter and established public output; it must not reclassify legacy rules as pure evidence.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py lint-snapshots
```

The runner must execute collection policy, unreadable/invalid/non-UTF-8/unstable/escape diagnostics, canonical diagnostic versus report-metadata behavior, equal-snapshot and host-change determinism, descriptor completeness, shared-index ordering, the manifest-configured pinned Dylint dependency/API/adapter/interface/ambient-state plans for every declared lint-core lane, positive/negative detector fixtures, registered disposable fix/no-fix/parity/idempotence fixtures, legacy-boundary tests, clean/negative corpus parity, and CLI/style-gate integration. Success means exit status 0, an empty reported error list, one snapshot collection phase per invocation, complete nonvacuous Dylint lane evidence, no rule-initiated traversal or open from a built-in rule, and structurally unequal snapshots whenever rejected-entry tombstones differ. Stable-open metadata checks or handle-relative reads inside that single collector phase do not count as hidden second collections.

## Deferred Follow-ups

Serialization, stable snapshot hashing, result caching, or sharing snapshots with Reef requires a separate proposal.
