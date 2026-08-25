## Context

Reef parses `[package]` into a typed manifest model. The model currently contains build identity and source roots only.

The source archive contains `reef.toml`, `reef.lock`, `src/`, and declared source roots. It omits README and license files.

`version-reef-manifest-schema` adds strict versioned DTOs and one upgrade command. Resolver-2 activation allocates manifest schema 2.

This change follows that activation and allocates manifest schema 3. The lock remains at schema 1.

Revision `1299452528e3735bf42edd584f6b90878b028906` activated resolver 2. Revision `91cfff09824e5dd505fae5d7b09c9a42494e3a94` allocated manifest schema 2.

Manifest schema 3 is the next free allocation at the accepted implementation base.

A lexical path check and later `fs::read` leave a symbolic-link race. Cross-platform artifact claims also require one host-independent path grammar.

## Goals / Non-Goals

**Goals:**

- Add common package metadata without a Cargo target or build-script model.
- Parse metadata into types that exclude invalid states.
- Define one portable package-file path grammar.
- Open declared files without symbolic-link traversal on supported Unix platforms.
- Archive stable captured file bytes.
- Include declared metadata files in deterministic source archives.
- Keep descriptive metadata outside resolution and package identity.
- Preserve existing package commands and compatibility formats.

**Non-Goals:**

- Add remote publication, registry search, or publisher selection.
- Add general archive file patterns.
- Add workspaces, features, development dependencies, or dependency patches.
- Add arbitrary tool metadata.
- Change `.chb`, `reef.lock`, `index.json`, or lock schema 1.
- Add Windows runtime support or Windows reparse-point handling.
- Transfer specification authority from `spec/**` to OpenSpec.

## Decisions

### 1. Add a small Cargo-style metadata set in manifest schema 3

Add these optional keys under `[package]`:

- `description`
- `license`
- `license-file`
- `repository`
- `documentation`
- `homepage`
- `readme`

A schema-3 wire DTO contains optional strings. `TryFrom` converts each value into a private invariant type.

A description contains 1 through 512 UTF-8 bytes after trim. It contains no line break, control character, or Unicode format character.

Existing schema-2 manifests upgrade to schema 3 without invented metadata. `chelis reef init` does not invent metadata values.

Alternative required metadata breaks local and private packages before Reef has a publication boundary.

### 2. Parse licenses as SPDX expressions

Use the `spdx` crate as the sole parser for `package.license`. Store the validated declared expression in a private `SpdxLicense` type.

Reef preserves the declared SPDX text. Parser-version changes cannot rewrite a manifest during an unrelated operation.

Use `package.license-file` for a license without an SPDX expression. A manifest cannot declare both fields.

An expression contains at most 1024 UTF-8 bytes. An empty or invalid expression fails at manifest load.

Reef does not infer a license from repository files.

Alternative free-form license strings weaken legal inspection. Alternative automatic file discovery makes archive membership undeclared.

### 3. Parse metadata URLs as absolute HTTPS URLs

Use the `url` crate to parse `repository`, `documentation`, and `homepage`. Each raw and normalized URL contains at most 2048 UTF-8 bytes.

Each URL uses HTTPS and contains a host.

A URL contains no username or password. Reef stores its canonical form in a private `PackageUrl` type.

These URLs are descriptive. Reef does not use them for provider selection, dependency fetches, release discovery, authentication, mirrors, or source trust.

Alternative source selection from `repository` creates a second publisher model.

### 4. Define one portable package-file grammar

Convert `readme` and `license-file` into `PortablePackagePath` at manifest load.

The path uses these lexical rules:

- It contains 1 through 1024 UTF-8 bytes.
- It is Unicode NFC.
- It uses `/` as its only separator.
- It contains no backslash, control character, Unicode format character, or line separator.
- It contains no empty, `.`, or `..` segment.
- Each segment contains 1 through 255 UTF-8 bytes.
- A segment contains none of `<`, `>`, `:`, `"`, `|`, `?`, or `*`.
- A segment does not end in a space or period.
- A segment is not a Windows device name, with case ignored and any extension removed.
- Its complete portable form is not `reef.toml` or `reef.lock`, with ASCII case ignored.

Windows device names are `con`, `prn`, `aux`, `nul`, `com1` through `com9`, and `lpt1` through `lpt9`.

Store segments and the portable archive string. Do not convert raw manifest text into a host `PathBuf` before this parse succeeds.

Use the same lexical type for declared metadata paths and their archive-member keys. Keep the existing source-root path grammar unchanged.

Reject a source and metadata spelling collision after Unicode NFC and ASCII case normalization. Do not emit duplicate platform-equivalent members.

Alternative host `Path::components` gives backslashes and prefixes different meanings on different platforms.

### 5. Open declared files relative to a package-root handle

Open the canonical package root as a directory handle. Resolve each parsed segment relative to the preceding directory handle.

Linux and macOS are the supported Unix platforms. Use no-follow relative opens for each segment on those platforms.

Reject a supported Unix platform when its APIs cannot enforce the no-follow contract. Do not fall back to check-then-open.

Windows runtime support remains outside this change. The lexical grammar still rejects Windows-sensitive portable names.

The final handle must identify one regular file. The open handle fixes its object identity through snapshot capture.

This boundary never follows a symbolic link outside or inside the package root.

Alternative canonicalization follows links before Reef can reject them. Alternative `symlink_metadata` followed by `fs::read` leaves a replacement race.

### 6. Capture bounded stable file snapshots

The `reef-package-metadata` capability specification owns the exact per-file limit. Two declared files have a derived combined maximum.

Read the final handle into memory with checked byte accounting. Seek to the start and read it again.

Require both reads to have identical bytes. The open handle fixes the file identity before both reads.

Archive the captured bytes, not a later path read. A concurrent content change causes a typed unstable-file error.

Complete all declared-file snapshots before any final archive replacement.

Alternative direct streaming from a path can mix path replacement with archive output.

### 7. Add snapshots to the canonical archive member map

Archive construction starts with its current member set. It then inserts declared README and license snapshots under their portable paths.

The existing bytewise `BTreeMap` order remains the archive order. The map deduplicates one path selected by a source root and metadata field.

If a source root and metadata declaration select one path, the captured metadata snapshot is authoritative for that archive member.

If two metadata fields identify one path, open and snapshot it once. If two different paths are hard links, preserve both declared path entries.

All members retain the current normalized mode, owner, group, and modification time. Reef discovers no undeclared metadata files.

Acquire the package-root project lock before final archive work.

Serialize and compress the complete archive into a unique sibling file. Rename it over the final archive only after successful finish and flush.

Remove only the sibling that the current command created. Do not remove an unknown sibling from another writer.

Alternative Cargo-style `include` and `exclude` patterns enlarge package-content policy without a current requirement.

### 8. Keep metadata outside package identity and source policy

Dependency resolution identifies a package by parsed name, exact version, selected source, and verified bytes.

A metadata edit changes the source archive hash because `reef.toml` and declared files are archive members. It does not change SemVer precedence or requirement matching.

The lock, registry index, and `.chb` continue to use their current fields. A verified source archive remains the metadata source of truth.

Do not copy metadata into three compatibility formats. Multiple copies can disagree.

### 9. Extend the document schema and prepared cache

Add the manifest schema-2 to schema-3 migration to `chelis reef upgrade`.

The migration adds schema 3 without adding metadata values. It preserves comments and key order.

Ship `docs/schemas/reef/manifest-v3.schema.json` and its drift test. Lock schema 1 remains unchanged.

Add serializable invariant metadata types to `PreparedReefGraph`. Bump the prepared-graph cache version.

A stale cache follows the current fail-closed rebuild path. Strict unknown-key behavior remains owned by `version-reef-manifest-schema`.

### 10. Use one executable completion oracle

The authoritative completion oracle is:

```sh
cargo nextest run -p chelis-cli --test reef_package_metadata --no-fail-fast
```

The suite covers every field, path segment rule, filesystem race, size limit, archive member, schema migration, and compatibility format.

A fresh local red-team agent mutates Unicode forms, separators, device names, links, file contents, sizes, collisions, and archive order.

OpenSpec validation and documentation builds are supporting evidence. They do not replace the oracle.

## Validation Record

The implementation used OpenSpec 1.6.0. Strict validation passed for all active items.

Thirteen metadata oracle tests passed before schema-3 writes became active. The final authoritative oracle passed 22 tests.

The complete `chelis-reef` suite passed 145 tests, with 2 skipped. The Reef CLI set passed 146 tests, with 10 skipped.

The prerequisite document, versioning, remote-discovery, conformance, and metadata unit set passed 99 tests.

Targeted Clippy passed with warnings denied. Formatting, the documentation build, schema generation, and three Python target-directory tests passed.

Both regenerated `chelis-std` artifact pairs passed complete verification. Their archive and shell hashes match every controlled lock fixture.

Three fresh adversarial audits examined Unicode, bounds, schema bypasses, handle walks, special files, races, archives, caches, and compatibility formats.

The implementation corrected all material findings. Corrections include nonblocking file opens, stable manifest bytes, typed link errors, and source-grammar compatibility.

Corrections also include normalized URL bounds, collision checks, private test hooks, cache revalidation, and manifest-level field diagnostics.

OpenSpec validation proves the artifact structure only. It does not replace executable oracles or adversarial audits.

Windows runtime file snapshots remain unsupported. Linux and macOS enforce the no-follow handle contract.

## Hosted Acceptance Handoff

An implementation review can start only after the local oracles, strict validation, formatting, Clippy, and documentation build pass.

The hosted gate requires green Linux, macOS, documentation, and applicable release checks on the final revision.

Hosted fixtures use fixed test values and temporary files. They contain no private credentials or local file content.

Hosted results provide supporting evidence. They do not replace the authoritative local oracle.

## Risks / Trade-offs

- **[Risk] Portable rules reject an existing unusual file name.** → Metadata fields remain optional and errors identify the exact segment rule.
- **[Risk] Double reads add package-build work.** → Each file is bounded, and metadata builds normally contain two small files.
- **[Risk] Supported Unix file APIs differ.** → Fail closed when a supported platform cannot enforce relative no-follow opens.
- **[Risk] Metadata changes alter source archive hashes.** → Use the normal version and release process for affected packages.
- **[Risk] SPDX parser updates change accepted expressions.** → Pin one major version and lock positive and negative fixtures.
- **[Risk] Schema allocation changes before implementation.** → Recheck current `main` and rebase the schema number before code starts.

## Migration Plan

1. Complete manifest schema 1 and resolver-2 schema-2 activation.
2. Recheck that manifest schema 3 is the next free allocation.
3. Add positive and negative metadata, path, handle, snapshot, and archive fixtures.
4. Add metadata invariant types and parser dependencies.
5. Add portable paths and race-safe file handles.
6. Add bounded stable snapshots and canonical archive membership.
7. Add schema-3 upgrade support, editor schema, and cache invalidation.
8. Add metadata only to packages with authoritative values and files.
9. Run strict validation, the oracle, and fresh adversarial validation.
10. Obtain hosted acceptance before package release.

Rollback first removes metadata declarations from affected manifests. Then it migrates controlled manifests back to schema 2 before it restores an older toolchain.

Archives use the existing tar format. An older reader can extract additional regular files after schema-compatible package selection.

## Open Questions

None.
