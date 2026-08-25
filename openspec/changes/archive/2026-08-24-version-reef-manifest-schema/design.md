## Context

Reef deserializes `reef.toml` and `reef.lock` directly into one current Rust shape. Neither document states which shape it uses.

Unknown keys are currently ignored. The active package changes need strict versioned DTOs for new resolver and metadata fields.

A dependency resolver version cannot identify the complete document shape. Manifest and lock formats also evolve for different reasons.

The exact compiler pin routes normal project commands to one toolchain. It does not give editors, migration tools, or direct old-tool invocations a format discriminator.

## Goals / Non-Goals

**Goals:**

- Give each Reef document an independent schema identity.
- Dispatch to one strict DTO after schema selection.
- Preserve legacy documents until an explicit upgrade.
- Give every future format change one ordered migration step.
- Preserve a valid document after each failed write.
- Make a mixed manifest and lock version recoverable.
- Ship editor schemas that match the accepted DTOs.

**Non-Goals:**

- Define Cargo-style requirements or package metadata.
- Change package resolution, artifact identity, or package publication.
- Make one atomic transaction across two files.
- Make a pre-schema Reef binary understand future schema values.
- Transfer specification authority from `spec/**` to OpenSpec.

## Decisions

### 1. Give each document one schema field

Add this top-level field to both document types:

```toml
schema = "1"
```

The value is a decimal ASCII integer without signs, whitespace, or leading zeroes. Schema values start at 1.

A missing field identifies legacy schema 0. Reef does not write schema 0.

Manifest and lock schema numbers are independent. A future manifest change does not force a lock schema change.

Alternative product-version inference couples file syntax to releases. Alternative resolver inference covers only dependency semantics.

### 2. Parse the schema before the strict document DTO

Parse a minimal header first. The header accepts only the optional schema field and ignores the remaining keys during this first pass.

Convert the header value into `ManifestSchemaVersion` or `LockSchemaVersion`. Then dispatch to the exact private wire DTO for that version.

A supported schema uses `deny_unknown_fields` on each fixed-shape table. Dynamic package, source, and artifact keys still pass their own boundary parsers.

An unsupported newer schema fails before the strict DTO sees a new key. The error names the found schema, the newest supported schema, and `chelis reef upgrade`.

A malformed schema fails as a schema error. Reef does not report it as an unknown key or package error.

Alternative direct deserialization cannot distinguish a typo from a newer format.

### 3. Keep a legacy reader and make new writes strict

Schema 0 uses the current permissive reader for command compatibility. It emits one warning that names the project and the upgrade command.

`chelis reef upgrade` applies a strict known-key census before it upgrades schema 0. It rejects unknown legacy keys instead of discarding them.

`chelis reef init` writes manifest schema 1. Every new `reef.lock` write uses lock schema 1.

Controlled repository manifests and locks move to schema 1 before later schema changes start.

Alternative automatic silent upgrade can change reviewed project files during an ordinary build.

### 4. Use one general upgrade command

Add this command:

```text
chelis reef upgrade --check|--inplace [--path <path>] [--manifest-to <n>] [--lock-to <n>]
```

Exactly one mode is required. `--path` selects a package root and defaults to the current package.

Each target option selects one supported schema for its document. A target cannot be less than the current document schema.

An omitted target selects the newest supported schema for that document. The command stops after the selected target step.

`--check` parses and preflights the complete selected migration. It writes nothing and reports each required document step.

`--inplace` runs the same preflight. It then applies every selected step in ascending order.

The initial migrations are:

- manifest schema 0 to 1: add `schema = "1"` and preserve all accepted fields
- lock schema 0 to 1: add `schema = "1"` and preserve exact lock data

Later changes extend the registry. They do not add another migration command.

Bounded remote discovery registers the SemVer schema-1 to schema-2 transformation. Package metadata later allocates schema 3.

Alternative feature-specific migration commands divide format ownership and create conflicting rewrite logic.

### 5. Lock project writes and make each replacement recoverable

Use one persistent `.reef-write.lock` file under the package root. Add this file to generated project ignore rules.

Acquire this project lock before any final manifest, lock, or package-archive replacement. Concurrent project writers serialize through this lock.

Serialize and validate a complete document before any final path opens for replacement.

Reject symbolic-link and multiply-hard-linked document targets before any document replacement. Do not detach another path from the updated document.

Write each result to a unique sibling file. Flush the file before replacement.

If a target exists, create one unique command-owned recovery hard link before replacement. A sibling-name collision fails without changing either sibling.

Rename the temporary sibling over the target, then sync the parent directory where the platform supports it.

If the parent sync fails, restore the prior target through the recovery link. Remove only siblings that the current command created.

Never remove or overwrite an unknown sibling from another command.

The manifest replacement occurs first. The lock replacement occurs second because Reef can reconstruct the lock from the manifest and verified artifacts.

A process can stop between these replacements. Every supported reader accepts a current manifest with a legacy lock without a write.

A later upgrade invocation reruns the remaining lock step. The command is idempotent and preserves an accepted current document.

Do not claim joint atomicity. The two paths can use different filesystems in future layouts.

Alternative rollback of an earlier document after a later document failure can overwrite a valid newer document. Reef does not perform that two-document rollback.

### 6. Ship checked editor schemas

Commit these versioned JSON Schema files:

```text
docs/schemas/reef/manifest-v1.schema.json
docs/schemas/reef/lock-v1.schema.json
```

Generate each file from the same versioned wire model or an exact model descriptor. A drift test compares committed bytes with generated bytes.

The manifest schema identifies fixed keys and dynamic table shapes. It does not claim package values pass filesystem or remote checks.

The docs explain this validation boundary. Editor validation is advisory and does not replace Reef parsing.

### 7. Allocate later schema numbers in one sequence

Before a change adds or removes a document key, it reads the highest schema on current `main`.

The change allocates the next integer and adds one migration step. It updates the DTO, JSON Schema, fixtures, and user documentation together.

Two active changes cannot claim the same next schema. A dependent change waits or rebases its allocation before implementation.

The planned sequence is:

1. this change: manifest 1 and lock 1
2. `adopt-semver-for-reef`: resolver-2 typed semantics
3. `add-bounded-reef-remote-discovery`: manifest 2
4. `add-reef-package-metadata`: manifest 3

The lock remains at schema 1 while its fields remain unchanged.

### 8. Preserve cache and artifact boundaries

The schema field enters manifest and lock bytes. It therefore changes the prepared-graph determinant and package source archive through `reef.toml`.

Bump the prepared-graph cache version when the typed manifest payload changes. A stale cache follows the current fail-closed rebuild path.

Do not add the schema field to `.chb`, `index.json`, or artifact filenames.

### 9. Use one executable completion oracle

The authoritative completion oracle is:

```sh
cargo nextest run -p chelis-cli -p chelis-reef -E 'binary(reef_document_schema) | test(/document_schema::tests/)' --no-fail-fast
```

The suite covers legacy reads, strict current reads, targets, links, unsupported schemas, project locks, replacement failures, interrupted sequences, byte preservation, and schema-file drift.

A fresh local red-team agent mutates schema values, unknown keys, temp files, rename failures, and mixed document versions.

OpenSpec validation and documentation builds are supporting evidence. They do not replace the oracle.

Hosted acceptance starts after the local change archive is ready for review. The final revision requires green Linux integration, documentation, and applicable release checks. Hosted fixtures must not contain credentials or local paths. Hosted results remain supporting evidence and do not replace the local oracle.

## Risks / Trade-offs

- **[Risk] Legacy schema 0 still permits ignored keys during ordinary reads.** → Emit a warning and require strict preflight before upgrade.
- **[Risk] The new field changes archive hashes.** → Migrate controlled packages through their normal version and release process.
- **[Risk] An old pre-schema binary ignores the discriminator.** → Keep exact compiler routing and document the minimum compatible toolchain.
- **[Risk] A process stops between manifest and lock replacement.** → Accept mixed versions and make the upgrade idempotent.
- **[Risk] JSON Schema suggests stronger validation than it provides.** → Document that Reef performs typed and filesystem checks after schema validation.
- **[Risk] A process can ignore the advisory project lock.** → Reject unsafe link targets and keep every document replacement independently recoverable.
- **[Risk] Non-Unix targets do not expose portable parent-directory sync or link-count checks.** → Keep those checks explicit where the platform supports them.
- **[Risk] A stopped process can leave its unique sibling file.** → Ignore unknown siblings and never remove a sibling from another command.

## Migration Plan

1. Add legacy and current document fixtures.
2. Add schema header types and version dispatch.
3. Add schema-1 DTOs and strict unknown-key errors.
4. Add the upgrade registry and atomic single-file writer.
5. Add JSON Schema artifacts and drift tests.
6. Migrate controlled documents before later schema allocation.
7. Run the oracle and fresh adversarial validation.
8. Publish the compatible compiler before any schema-2 manifest lands.

Rollback removes schema-1 fields from controlled documents before it restores a pre-schema compiler.

A rollback after later schemas exist must use their registered reverse release procedure. It cannot silently remove unknown fields.

## Open Questions

None.
