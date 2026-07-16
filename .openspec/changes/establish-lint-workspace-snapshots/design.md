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

`collect_workspace(request, policy)` is an adapter operation. It resolves one logical root and returns a `WorkspaceSnapshot` plus `CollectionDiagnostic` values. The snapshot contains stable path-sorted `SnapshotEntry` values with normalized logical paths, immutable bytes, optional decoded text, surface inputs, and declared metadata.

`lint_snapshot(snapshot, rule_set, exceptions)` is the pure core. Host changes after collection are invisible to analysis.

### 2. Collection policy and compatibility corrections are explicit

`SnapshotPolicy` contains ignore rules, symlink/containment behavior, path normalization, decoding policy, selected file classes, and size limits. V1 uses the repository's current skip-directory policy, rejects symlink escape, sorts normalized paths bytewise, and decodes required text as UTF-8.

Current silently skipped unreadable or invalid required text becomes a structured blocking collection diagnostic. Ignored and unselected entries remain absent without a diagnostic. CLI `lint --check` exits nonzero for a blocking collection diagnostic; advisory rule behavior is unchanged. These corrections receive dedicated CLI and machine-output fixtures.

### 3. Rule requirements and indexes are values

Each pure rule declares a serializable `RuleRequirements` descriptor naming file classes, decoded text, directory entries, Cargo package metadata, and shared indexes it needs. The collector unions descriptors before reading content. The analysis driver builds opaque-type, Cargo-package, and other catalogs once from the complete snapshot before the first rule executes.

Rule ordering is the explicit registry order; diagnostics are finally ordered by normalized path, line, rule id, and stable detail key. Equal snapshots, rule configurations, and exceptions produce equal output.

### 4. Public compatibility does not weaken the boundary

A new versioned `SnapshotRule` interface receives only `SnapshotRuleContext` and immutable indexes. The existing root-exposing `Rule`/`Context` API remains for one documented minor-version window behind `lint_legacy_path_rules`; it may collect or reopen host state and is explicitly outside the pure-core guarantee and acceptance evidence. Built-in rules migrate completely and never use the legacy adapter. The old interface is deprecated rather than silently redefined.

### 5. Exact boundary

The designated core is planned `crates/chelis-lint/src/{snapshot,analysis,indexes,rule_v2}.rs` plus built-in `rules/**` after migration. Adapter modules are planned `collector.rs` and `legacy.rs`; CLI style/lint orchestration and fix application remain outer adapters. Designated modules cannot import the collector or legacy adapter.

The architecture manifest records the exact transitive production module set. Checks reject direct, aliased, re-exported, qualified, callback-hidden, and trait-hidden filesystem, environment, process, network, clock, or terminal capabilities. `cfg(test)` bodies are classified separately without exempting an entire production file. Behavioral snapshot determinism and host-change tests remain authoritative.

### 6. Query and outcome identity

`LintQueryKey` covers snapshot content/metadata digests, normalized snapshot policy version, pure rule ids/configuration/versions, indexes version, and exceptions. `LintOutcomeDigest` covers canonically ordered collection diagnostics and violations. Absolute host root spelling, collection time, terminal mode, and renderer configuration are excluded.

## Risks / Trade-offs

- **Snapshot memory:** collect only descriptor-required classes and share immutable buffers.
- **External rule compatibility:** the legacy adapter is clearly outside purity claims and expires after one minor version.
- **Collection diagnostics change exit behavior:** pin the correction with positive/negative CLI fixtures and active-doc updates.
- **Descriptor omissions can hide required data:** tests remove each declared requirement and require a named missing-input failure rather than a silent empty result.

## Migration Plan

1. Add snapshot policy, host-change, descriptor-completeness, collection-diagnostic, legacy-boundary, and parity tests.
2. Add snapshot and logical-path data types plus one path-sorted collector.
3. Add `SnapshotRule`, shared indexes, query keys, and outcome digests.
4. Migrate built-in rules and remove all built-in hidden I/O.
5. Route path-based CLI entry points through collect-once analysis; retain only the named legacy adapter for external rules.
6. Add architecture gates, docs, examples, and the focused acceptance runner.

Rollback preserves the legacy adapter and established public output; it must not reclassify legacy rules as pure evidence.

## Acceptance Oracle

The authoritative completion oracle is:

```text
.venv/bin/python scripts/fcis_gate.py lint-snapshots
```

The runner must execute collection policy, unreadable/invalid/escape diagnostics, equal-snapshot and host-change determinism, descriptor completeness, shared-index ordering, built-in no-I/O architecture fixtures, legacy-boundary tests, clean/negative corpus parity, and CLI/style-gate integration. Success means exit status 0, an empty reported error list, one content collection per invocation, and no host access from a built-in rule.

## Deferred Follow-ups

Serialization or sharing of snapshots with Reef requires a separate proposal.
