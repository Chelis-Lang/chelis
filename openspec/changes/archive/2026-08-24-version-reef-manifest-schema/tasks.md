## 1. Add test stubs and fixtures

- [x] 1.1 Add manifest and lock fixtures for absent, current, malformed, and future schema values.
- [x] 1.2 Add strict-key fixtures for each fixed manifest and lock table.
- [x] 1.3 Add upgrade fixtures for check, in-place, supported target, unsupported target, no-op, and invalid mode combinations.
- [x] 1.4 Add comment and key-order fixtures for legacy manifest upgrades.
- [x] 1.5 Add exact-value snapshots for lock schema-0 to schema-1 upgrades.
- [x] 1.6 Add failure fixtures for serialization, temporary writes, rename errors, and parent-directory sync errors.
- [x] 1.7 Add target fixtures for symbolic links and multiply-hard-linked documents.
- [x] 1.8 Add interrupted-sequence fixtures with a current manifest and legacy lock.
- [x] 1.9 Add schema drift fixtures for manifest and lock wire models.
- [x] 1.10 Add a pre-schema toolchain routing fixture for the exact compiler pin.

## 2. Add schema boundaries

- [x] 2.1 Add `ManifestSchemaVersion` and `LockSchemaVersion` invariant types.
- [x] 2.2 Parse the minimal schema header before any complete document DTO.
- [x] 2.3 Preserve a schema-0 reader for legacy documents and emit one upgrade warning.
- [x] 2.4 Add strict schema-1 wire DTOs with table-aware unknown-key errors.
- [x] 2.5 Dispatch each supported schema to its exact conversion boundary.
- [x] 2.6 Reject malformed and future schemas before package or lock parsing.
- [x] 2.7 Make `reef init` write manifest schema 1.
- [x] 2.8 Make every new lock write use lock schema 1.
- [x] 2.9 Bump the prepared-graph cache version and prove fail-closed rebuilds.

## 3. Add the upgrade framework

- [x] 3.1 Add an ordered manifest migration registry and an ordered lock migration registry.
- [x] 3.2 Add strict known-key preflight for legacy schema-0 manifests.
- [x] 3.3 Add the manifest schema-0 to schema-1 migration.
- [x] 3.4 Add the lock schema-0 to schema-1 migration.
- [x] 3.5 Add `chelis reef upgrade --check|--inplace [--path <path>] [--manifest-to <n>] [--lock-to <n>]` dispatch.
- [x] 3.6 Preserve manifest comments and key order with `toml_edit`.
- [x] 3.7 Report every planned step in check mode without a write.
- [x] 3.8 Make current documents an unchanged success.
- [x] 3.9 Return typed errors for document, schema, step, path, and failed operation.

## 4. Make document replacement safe

- [x] 4.1 Add one persistent package-root project lock and generated ignore rule.
- [x] 4.2 Acquire the project lock before manifest, lock, or package-archive replacement.
- [x] 4.3 Serialize and parse the complete result before any final replacement.
- [x] 4.4 Write through a unique sibling file and flush its bytes.
- [x] 4.5 Preserve an existing target through a unique command-owned recovery hard link.
- [x] 4.6 Rename the sibling over the target and sync the parent where supported.
- [x] 4.7 Restore the prior target when parent-directory sync fails.
- [x] 4.8 Remove only siblings that the current command created.
- [x] 4.9 Replace the manifest before the lock.
- [x] 4.10 Accept a current manifest with every supported older lock schema.
- [x] 4.11 Make interrupted and repeated upgrades idempotent.
- [x] 4.12 Ignore unknown sibling files without changing or removing them.
- [x] 4.13 Prove that temporary and recovery sibling collisions preserve unknown files.
- [x] 4.14 Prove that a failed lock replacement preserves the prior readable lock.
- [x] 4.15 Prove that concurrent project writers serialize without sibling deletion.

## 5. Ship checked editor schemas

- [x] 5.1 Add the manifest schema-1 JSON Schema artifact.
- [x] 5.2 Add the lock schema-1 JSON Schema artifact.
- [x] 5.3 Generate both artifacts from the versioned wire models or exact descriptors.
- [x] 5.4 Add byte-for-byte drift tests for committed schema artifacts.
- [x] 5.5 Document the syntax-only assurance boundary for editor validation.
- [x] 5.6 Add editor configuration examples without making editor validation a build gate.

## 6. Migrate controlled documents and documentation

- [x] 6.1 Run the upgrade command on controlled package manifests and lockfiles.
- [x] 6.2 Update `docs/book/src/reef.md` with schema and upgrade behavior.
- [x] 6.3 Amend `spec/design/reef_distribution.md` with document schema ownership.
- [x] 6.4 Amend `spec/design/chelis_packaging_and_install.md` with recoverable two-file upgrades.
- [x] 6.5 Update examples and `CHANGELOG.md` with the legacy upgrade command.
- [x] 6.6 Record resolver-2 semantics as owned by `adopt-semver-for-reef`.
- [x] 6.7 Record manifest schema 2 as the next allocation for `add-bounded-reef-remote-discovery`.
- [x] 6.8 Record manifest schema 3 as the next allocation for `add-reef-package-metadata`.

## 7. Run strict validation

- [x] 7.1 Verify OpenSpec 1.6.0, then run `openspec validate --all --strict --no-interactive`.
- [x] 7.2 Run `cargo fmt --all -- --check` and correct all format failures.
- [x] 7.3 Run targeted Clippy with warnings denied for `chelis-reef` and `chelis-cli`.
- [x] 7.4 Run the authoritative oracle: `cargo nextest run -p chelis-cli -p chelis-reef -E 'binary(reef_document_schema) | test(/document_schema::tests/)' --no-fail-fast`.
- [x] 7.5 Verify that every requirement has positive and negative executable evidence.
- [x] 7.6 Keep schema-1 writes inactive until the authoritative oracle passes.

## 8. Run adversarial validation

- [x] 8.1 Start a fresh local red-team agent after the authoritative oracle passes.
- [x] 8.2 Mutate schema types, values, table keys, migration order, temp files, and rename outcomes.
- [x] 8.3 Stop the upgrade between document replacements and prove successful recovery.
- [x] 8.4 Mutate one DTO without its JSON Schema and require a drift failure.
- [x] 8.5 Correct all material findings and rerun the authoritative oracle.
- [x] 8.6 Record residual risks and prove that OpenSpec validation remains structural evidence only.

## 9. Record the hosted acceptance handoff

- [x] 9.1 Record that the implementation review starts only after local strict and adversarial validation passes.
- [x] 9.2 Record the required Linux integration, documentation, and applicable release checks.
- [x] 9.3 Verify that hosted fixtures expose no private repository credentials or local file content.
- [x] 9.4 Record that hosted results support but do not replace the authoritative oracle.
