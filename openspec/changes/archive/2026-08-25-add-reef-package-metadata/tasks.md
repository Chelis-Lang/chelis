## 1. Verify prerequisites and add test stubs

- [x] 1.1 Record accepted resolver-2 activation and manifest schema-2 revisions.
- [x] 1.2 Recheck that manifest schema 3 is the next free allocation on current `main`.
- [x] 1.3 Add description fixtures for valid, empty, multiline, control, and size-limit values.
- [x] 1.4 Add license fixtures for valid SPDX, invalid SPDX, license files, and duplicate license forms.
- [x] 1.5 Add URL fixtures for valid HTTPS, missing host, HTTP, relative, username, and password values.
- [x] 1.6 Add path fixtures for separators, NFC, lengths, controls, reserved characters, trailing forms, device names, and Reef documents.
- [x] 1.7 Add filesystem fixtures for regular files, missing files, directories, final links, and parent links.
- [x] 1.8 Add stable-snapshot fixtures for per-file size, byte changes, and post-snapshot path changes.
- [x] 1.9 Add archive snapshots for declared files, undeclared files, duplicate paths, source collisions, order, and normalized headers.
- [x] 1.10 Add schema-2 to schema-3 upgrade and JSON Schema drift fixtures.
- [x] 1.11 Add compatibility snapshots for unchanged `reef.lock`, `index.json`, `.chb`, and lock schema 1.
- [x] 1.12 Add failures that preserve the prior archive and registry outputs.

## 2. Add typed schema-3 metadata boundaries

- [x] 2.1 Add pinned workspace dependencies for SPDX, URL, and Unicode NFC parsing.
- [x] 2.2 Extend document dispatch with the manifest schema-3 wire DTO.
- [x] 2.3 Add invariant types for descriptions, SPDX expressions, HTTPS URLs, and portable package paths.
- [x] 2.4 Convert every wire metadata field through `TryFrom` before a consumer receives it.
- [x] 2.5 Reject both license forms and return field-specific typed errors.
- [x] 2.6 Keep descriptive URLs outside every provider and source-selection path.
- [x] 2.7 Add serializable invariant metadata types to the prepared manifest model.
- [x] 2.8 Bump the prepared-graph cache version and prove fail-closed rebuilds.

## 3. Implement portable package paths

- [x] 3.1 Enforce the 1024-byte path and 255-byte segment limits.
- [x] 3.2 Require Unicode NFC and `/` as the only separator.
- [x] 3.3 Reject backslashes, controls, empty segments, `.`, `..`, and reserved characters.
- [x] 3.4 Reject trailing spaces, trailing periods, Windows device names, `reef.toml`, and `reef.lock`.
- [x] 3.5 Store parsed segments and one canonical portable archive string.
- [x] 3.6 Use the portable type for declared archive-member keys without widening the source-root grammar.
- [x] 3.7 Return errors that identify the manifest field, segment, and failed rule.

## 4. Open declared files without link traversal

- [x] 4.1 Open the package root as a directory handle.
- [x] 4.2 Walk each supported Unix segment through relative no-follow opens.
- [x] 4.3 Require the final handle to identify one regular file.
- [x] 4.4 Reject unsupported Unix enforcement without a check-then-open fallback.
- [x] 4.5 Prove rejection for final links, parent links, and replacement races.
- [x] 4.6 Keep all filesystem authority inside the package-root handle.

## 5. Capture bounded stable snapshots

- [x] 5.1 Enforce the exact per-file limit from the capability specification and compute its derived combined maximum.
- [x] 5.2 Read each final handle twice from offset zero.
- [x] 5.3 Compare both byte sequences and retain the identity fixed by the open handle.
- [x] 5.4 Return a typed unstable-file error after any mismatch.
- [x] 5.5 Keep captured bytes in the verified archive-member model.
- [x] 5.6 Complete every metadata snapshot before final archive replacement.
- [x] 5.7 Prove that archive serialization never reopens a declared path.

## 6. Extend deterministic archive construction

- [x] 6.1 Insert verified README and license snapshots into the canonical member map.
- [x] 6.2 Deduplicate two declarations that identify one portable path.
- [x] 6.3 Deduplicate a metadata path already selected by a source root.
- [x] 6.4 Preserve distinct declared paths for two hard links.
- [x] 6.5 Preserve canonical order and normalized archive headers.
- [x] 6.6 Keep undeclared README and license files outside the archive.
- [x] 6.7 Acquire the package-root project lock before final archive work.
- [x] 6.8 Build and flush a unique sibling archive before final rename.
- [x] 6.9 Remove only the temporary sibling that the current command created.
- [x] 6.10 Preserve the prior archive after snapshot, compression, flush, or rename failure.
- [x] 6.11 Prove that concurrent project writers serialize without sibling deletion.
- [x] 6.12 Prove that metadata edits change archive hashes without changing package matching.
- [x] 6.13 Prove that `.chb`, `reef.lock`, and `index.json` serializers remain unchanged.

## 7. Add schema-3 migration and documentation

- [x] 7.1 Register the manifest schema-2 to schema-3 upgrade step.
- [x] 7.2 Preserve comments, key order, and package semantics without invented metadata.
- [x] 7.3 Add `docs/schemas/reef/manifest-v3.schema.json` and its drift test.
- [x] 7.4 Make `reef init` select schema 3 through the newest-supported-schema rule.
- [x] 7.5 Update `docs/book/src/reef.md` with fields, licenses, URLs, paths, limits, and stable snapshots.
- [x] 7.6 Amend `spec/design/reef_distribution.md` with metadata ownership and archive snapshots.
- [x] 7.7 Amend `spec/design/chelis_packaging_and_install.md` with schema-3 ownership.
- [x] 7.8 Add metadata only to packages with authoritative values and declared files.
- [x] 7.9 Update examples and `CHANGELOG.md` without implying a remote publication capability.
- [x] 7.10 Record deferred workspaces, features, development dependencies, publication controls, and archive patterns.

## 8. Run strict validation

- [x] 8.1 Verify OpenSpec 1.6.0, then run `openspec validate --all --strict --no-interactive`.
- [x] 8.2 Run `cargo fmt --all -- --check` and correct all format failures.
- [x] 8.3 Run targeted Clippy with warnings denied for `chelis-reef` and `chelis-cli`.
- [x] 8.4 Run the authoritative oracle: `cargo nextest run -p chelis-cli --test reef_package_metadata --no-fail-fast`.
- [x] 8.5 Verify that every requirement has positive and negative executable evidence.
- [x] 8.6 Keep schema-3 writes inactive until the authoritative oracle passes.

## 9. Run adversarial validation

- [x] 9.1 Start a fresh local red-team agent after the authoritative oracle passes.
- [x] 9.2 Mutate Unicode forms, separators, segments, links, file contents, sizes, collisions, and archive order.
- [x] 9.3 Verify that every rejected path or snapshot preserves final artifacts and registry state.
- [x] 9.4 Verify deterministic archive bytes across repeated builds and path-discovery order changes.
- [x] 9.5 Correct all material findings and rerun the authoritative oracle.
- [x] 9.6 Record residual risks and prove that OpenSpec validation remains structural evidence only.

## 10. Obtain hosted acceptance

- [x] 10.1 Open the implementation review only after local strict and adversarial validation passes.
- [x] 10.2 Require green Linux, macOS, documentation, and applicable release checks on the final revision.
- [x] 10.3 Verify that hosted fixtures expose no private repository credentials or local file content.
- [x] 10.4 Record hosted results as supporting evidence without replacing the authoritative oracle.
