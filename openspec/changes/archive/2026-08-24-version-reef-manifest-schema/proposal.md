## Why

Reef has no document-level schema discriminator for `reef.toml` or `reef.lock`. Strict unknown-key rejection alone cannot distinguish a typo from a newer supported format.

The active SemVer and package-metadata changes both alter the manifest shape. Reef needs one migration framework before either change adds fields.

## What Changes

- Add `schema = "1"` at the top level of each `reef.toml` and `reef.lock` document.
- Treat a missing schema as legacy schema 0 for existing documents.
- Parse the schema discriminator before any strict version-specific wire DTO.
- Reject a newer schema with an error that names `chelis reef upgrade`.
- Add `chelis reef upgrade --check|--inplace [--path <path>] [--manifest-to <n>] [--lock-to <n>]` as the only document-upgrade command.
- Keep manifest and lock schemas independent, even when both use schema 1 initially.
- Register ordered migration steps for later Reef changes.
- Serialize project-file writes under one project-local lock.
- Write each document through an atomic single-file replacement.
- Make mixed-version recovery explicit because two files cannot share one atomic replacement.
- Never remove a temporary sibling that belongs to another command.
- Ship checked JSON Schema files for each supported manifest and lock schema.
- Make later changes allocate their next schema on current `main` before implementation.

Out of scope:

- Cargo-style dependency requirements or package metadata fields.
- Remote candidate discovery, mirrors, source patches, or publication.
- A change to package identity, compiler pins, archive fields, `.chb`, `index.json`, or generated code.
- A joint filesystem transaction across the project and the Reef registry.
- Compiler, runtime, backend, or language semantics.

## Capabilities

### New Capabilities

- `reef-document-schema`: Defines schema discrimination, strict version dispatch, document upgrades, recovery, and editor schema artifacts.

### Modified Capabilities

None.

## Impact

- Manifest and lock parsing change in `crates/chelis-reef`.
- Reef command dispatch changes in `crates/chelis-cli`.
- Controlled `reef.toml` and `reef.lock` files gain one top-level field.
- The repository gains versioned JSON Schema artifacts and drift tests.
- `adopt-semver-for-reef` must follow this change and define resolver-2 semantics.
- `add-bounded-reef-remote-discovery` must then allocate manifest schema 2 and expose those semantics.
- `add-reef-package-metadata` must follow schema-2 activation and allocate manifest schema 3.
- Existing documents remain readable through the legacy schema-0 parser.
