## Why

Reef package manifests contain build identity but cannot record legal, source, or documentation metadata. Source archives also omit declared license and README files.

This omission weakens package inspection and future publication. Reef needs typed metadata before it adds a remote registry or publisher workflow.

Declared metadata files also need one portable and race-safe path boundary. A check followed by a separate path open does not prevent symbolic-link replacement.

## What Changes

- Follow manifest schema 2 and allocate manifest schema 3.
- Add optional Cargo-style `description`, `license`, `license-file`, `repository`, `documentation`, `homepage`, and `readme` fields under `[package]`.
- Parse SPDX license expressions and absolute HTTPS URLs at the manifest boundary.
- Treat `license` and `license-file` as exclusive alternatives.
- Parse declared files as bounded NFC paths with `/` separators and portable segments.
- Reject backslashes, unsafe components, platform-reserved segments, and reserved Reef document paths.
- Open declared files relative to a package-root directory handle without symbolic-link traversal on supported Unix platforms.
- Read stable bounded file snapshots and archive those captured bytes.
- Include a declared README and license file in the deterministic source archive.
- Keep package metadata out of dependency matching, package identity, `.chb`, `reef.lock`, and `index.json`.
- Add manifest schema-2 to schema-3 upgrade support and a schema-3 editor artifact.
- Preserve current build, local publication, and remote source-selection behavior.

Out of scope:

- A public registry, remote publication command, or `publish = false` behavior.
- Keywords, categories, authors, badges, or automatic README discovery.
- General archive `include` or `exclude` patterns.
- Development dependencies, features, workspaces, target-specific dependencies, build scripts, or dependency patches.
- A change to `.chb`, `reef.lock`, `index.json`, or lock schema 1.
- Compiler, runtime, backend, language, or generated-code semantics.

## Capabilities

### New Capabilities

- `reef-package-metadata`: Defines typed package metadata, portable declared files, race-safe snapshots, and deterministic archive membership.

### Modified Capabilities

None.

## Impact

- Manifest parsing and archive construction change in `crates/chelis-reef`.
- Schema-3 upgrade and newest-schema initialization change in `crates/chelis-cli`.
- The workspace adds direct SPDX, URL, Unicode-normalization, and Unicode-category parser dependencies.
- No-follow relative file opening can require existing low-level filesystem dependencies.
- `docs/book/src/reef.md` and the Reef package design documents require updates.
- Existing schema-2 manifests remain valid after a schema-3 upgrade because every metadata field is optional.
- Schema-3 migration changes archived `reef.toml` bytes and source hashes for every upgraded package.
- Declared README or license bytes add further source-hash changes.
- The change must follow resolver-2 activation because that change owns manifest schema 2.
