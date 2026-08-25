# Reef Package Metadata Specification

## Purpose

Define typed package metadata, portable declared files, race-safe snapshots, and deterministic metadata archive membership.

## Requirements

### Requirement: Manifest schema 3 accepts typed descriptive metadata

Manifest schema 3 SHALL accept optional `description`, `repository`, `documentation`, `homepage`, and `readme` fields under `[package]`.

A description SHALL contain 1 through 512 UTF-8 bytes after trim. It SHALL NOT contain a line break, control character, or Unicode format character.

A schema-3 manifest can omit every metadata field without a package behavior change.

Schema 2 SHALL reject metadata keys as unknown fields.

#### Scenario: Package declares a valid description

- **WHEN** a schema-3 manifest declares `description = "Tensor primitives for Chelis"`
- **THEN** Reef accepts the description as package metadata

#### Scenario: Package description is empty

- **WHEN** a schema-3 description is empty after trim
- **THEN** Reef rejects the manifest and identifies `package.description`

#### Scenario: Package description contains two lines

- **WHEN** a schema-3 description contains a line break
- **THEN** Reef rejects the manifest and identifies `package.description`

#### Scenario: Existing package omits metadata

- **WHEN** a valid schema-3 manifest contains none of the optional metadata fields
- **THEN** Reef accepts it with unchanged package behavior

#### Scenario: Schema-2 package declares metadata

- **WHEN** a schema-2 manifest declares `description`
- **THEN** Reef rejects the key and identifies the schema-2 `[package]` table

### Requirement: Reef uses SPDX license declarations

Manifest schema 3 SHALL accept one optional `license` field or one optional `license-file` field. It SHALL reject both fields together.

The `license` value SHALL contain at most 1024 UTF-8 bytes and SHALL be a valid SPDX license expression. Reef SHALL parse it before graph construction.

The `license-file` value SHALL use the portable package-file rules. Reef SHALL NOT infer a license from an undeclared file.

#### Scenario: Package declares a valid SPDX expression

- **WHEN** a manifest declares `license = "MIT OR Apache-2.0"`
- **THEN** Reef accepts the parsed SPDX expression

#### Scenario: Package declares an invalid SPDX expression

- **WHEN** a manifest declares `license = "some permissive license"`
- **THEN** Reef rejects the manifest and identifies `package.license`

#### Scenario: Package declares a custom license file

- **WHEN** a manifest declares `license-file = "LICENSE.custom"` and that safe regular file exists
- **THEN** Reef accepts the license-file declaration

#### Scenario: Package declares two license forms

- **WHEN** a manifest declares both `license` and `license-file`
- **THEN** Reef rejects the manifest before graph construction

### Requirement: Package URLs are absolute safe metadata

The `repository`, `documentation`, and `homepage` fields SHALL contain at most 2048 UTF-8 bytes before and after URL normalization. They SHALL contain absolute HTTPS URLs with a host.

A package URL SHALL NOT contain a username or password. Reef SHALL parse each URL at manifest load.

Package URLs SHALL remain descriptive. Reef SHALL NOT use them for source providers, authentication, mirrors, or release discovery.

#### Scenario: Package declares valid HTTPS URLs

- **WHEN** a manifest declares absolute HTTPS repository and documentation URLs without credentials
- **THEN** Reef accepts the parsed URLs

#### Scenario: Package URL uses HTTP

- **WHEN** a manifest declares `repository = "http://example.com/package"`
- **THEN** Reef rejects the manifest and identifies `package.repository`

#### Scenario: Package URL contains credentials

- **WHEN** a manifest declares an HTTPS URL with a username or password
- **THEN** Reef rejects the URL without using the credentials

#### Scenario: Repository URL differs from the fetch origin

- **WHEN** a verified package declares a repository URL that differs from its locked origin
- **THEN** Reef keeps the locked origin as the source and treats the URL as metadata

### Requirement: Declared package files use one portable path grammar

A `readme` or `license-file` value SHALL contain 1 through 1024 UTF-8 bytes in Unicode NFC.

The path SHALL use `/` as its only separator. It SHALL contain no backslash, control character, Unicode format character, line separator, empty segment, `.`, or `..` segment.

Each segment SHALL contain 1 through 255 UTF-8 bytes. It SHALL contain none of `<`, `>`, `:`, `"`, `|`, `?`, or `*`.

A segment SHALL NOT end in a space or period. It SHALL NOT equal a Windows device name when case and any extension are ignored.

The complete portable path SHALL NOT equal `reef.toml` or `reef.lock`, with ASCII case ignored.

#### Scenario: README uses a safe nested path

- **WHEN** a manifest declares `readme = "docs/README.md"`
- **THEN** Reef accepts its parsed portable segments

#### Scenario: README escapes through a parent segment

- **WHEN** a manifest declares `readme = "../README.md"`
- **THEN** Reef rejects the manifest before filesystem access

#### Scenario: README uses a backslash separator

- **WHEN** a manifest declares `readme = "docs\\README.md"`
- **THEN** Reef rejects the backslash before host path conversion

#### Scenario: README is not NFC

- **WHEN** a manifest path contains a decomposed Unicode spelling
- **THEN** Reef rejects the path and identifies the NFC rule

#### Scenario: Segment uses a reserved device name

- **WHEN** a manifest declares `readme = "docs/CON.txt"`
- **THEN** Reef rejects the device-name segment on every platform

#### Scenario: Segment ends in a period

- **WHEN** a manifest declares `readme = "docs/readme."`
- **THEN** Reef rejects the trailing-period segment

#### Scenario: Declared path names a Reef document

- **WHEN** a manifest declares `readme = "reef.toml"`
- **THEN** Reef rejects the reserved document path

### Requirement: Reef opens declared files without symbolic-link traversal

Reef SHALL open the package root as a directory handle. It SHALL resolve each declared segment relative to directory handles.

On supported Unix platforms, Reef SHALL reject every symbolic-link component. The final handle SHALL identify one regular file.

Reef SHALL NOT fall back to a path check followed by a separate path open.

If a supported Unix platform cannot enforce this contract, Reef SHALL fail before it reads the declared file.

#### Scenario: Final metadata path is a symbolic link

- **WHEN** a declared README is a symbolic link
- **THEN** Reef rejects the file without reading its target

#### Scenario: Parent component is a symbolic link

- **WHEN** one parent segment resolves to a symbolic link
- **THEN** Reef rejects the file before it opens a child segment

#### Scenario: Path changes after lexical parsing

- **WHEN** an attacker replaces one path component before the handle walk reaches it
- **THEN** the no-follow open rejects the replacement without reading its target

#### Scenario: Supported Unix platform lacks a safe relative open

- **WHEN** Reef cannot enforce no-follow component opens on a supported Unix platform
- **THEN** archive construction fails without a check-then-open fallback

### Requirement: Reef captures bounded stable metadata-file bytes

Each declared metadata file SHALL contain at most 4 MiB. Two declared files SHALL therefore contain at most 8 MiB.

Reef SHALL read each final file handle twice from the start. Both byte sequences SHALL match.

Reef SHALL archive the captured bytes. It SHALL NOT reopen the declared path during archive serialization.

A concurrent content or identity change SHALL fail before final archive replacement.

#### Scenario: Metadata file is stable

- **WHEN** both bounded reads return identical bytes from one regular-file identity
- **THEN** Reef accepts those captured bytes as one archive member snapshot

#### Scenario: README exceeds its limit

- **WHEN** a declared README contains more than 4 MiB
- **THEN** Reef stops the read and identifies the file-size limit

#### Scenario: File changes between reads

- **WHEN** the two reads return different bytes
- **THEN** Reef reports an unstable declared file and preserves the prior archive

#### Scenario: Path changes after snapshot capture

- **WHEN** a path changes after Reef captures stable bytes
- **THEN** Reef archives only the captured bytes without another path read

### Requirement: Source archives contain declared metadata deterministically

A source archive SHALL contain each declared README and license snapshot at its portable path.

Archive construction SHALL use the existing canonical member order, mode, owner, group, and modification time.

If two metadata fields identify one path, the archive SHALL contain one member. A path already selected by a source root SHALL also appear once.

The captured metadata snapshot SHALL supply the bytes when a source root selects the same path.

Reef SHALL reject different source and metadata spellings that collide after Unicode NFC and ASCII case normalization.

Reef SHALL NOT discover undeclared README or license files.

Reef SHALL acquire the package-root project lock before final archive work.

Reef SHALL build the complete archive in a unique sibling file and replace the final archive only after successful finish and flush.

Reef SHALL remove only the sibling that the current command created. It SHALL NOT remove an unknown sibling from another writer.

#### Scenario: Archive contains declared metadata files

- **WHEN** a package declares a README and license file and both snapshots pass
- **THEN** its source archive contains both captured files at their portable paths

#### Scenario: Two fields identify one file

- **WHEN** two metadata fields identify the same portable path
- **THEN** Reef opens it once and writes one archive member

#### Scenario: Source root already selects a declared path

- **WHEN** a declared metadata path is also below a declared source root
- **THEN** the canonical member map writes one member for that path

#### Scenario: Source and metadata spellings collide portably

- **WHEN** a source member and declared metadata path differ only by Unicode normalization or ASCII case
- **THEN** Reef rejects the collision instead of writing duplicate platform-equivalent members

#### Scenario: Package contains an undeclared README

- **WHEN** a package root contains `README.md` without a `readme` declaration
- **THEN** Reef does not add that file to the archive

#### Scenario: Archive construction fails

- **WHEN** one snapshot or archive operation fails
- **THEN** Reef preserves the previous final archive and removes only its own incomplete sibling

#### Scenario: Concurrent build holds the project lock

- **WHEN** one package build holds the package-root project lock
- **THEN** another project writer waits without changing the archive or any temporary sibling

### Requirement: Package metadata does not change identity or compatibility formats

Reef SHALL exclude descriptive metadata from package identity, SemVer precedence, dependency matching, and source selection.

`reef.lock`, `index.json`, and `.chb` SHALL retain their current field locations and types. Lock schema 1 SHALL remain unchanged.

A metadata edit SHALL affect the source archive hash through the archived manifest or declared file bytes.

#### Scenario: Two candidates differ only in descriptive metadata

- **WHEN** two candidate records have the same identity, source, and verified bytes
- **THEN** Reef does not treat metadata as another version selector

#### Scenario: Package adds metadata without a compatibility-format migration

- **WHEN** a package adds valid metadata and builds successfully
- **THEN** its lock, index, and `.chb` retain their current field locations and types

#### Scenario: Declared README content changes

- **WHEN** a package changes the captured README bytes
- **THEN** the next source archive has a different content hash

### Requirement: The schema-2 to schema-3 transformation preserves package semantics

The schema-2 to schema-3 transformation SHALL preserve comments, key order, and all existing package semantics.

It SHALL change the manifest schema to 3. It SHALL NOT invent metadata values.

Reef SHALL ship a schema-3 manifest JSON Schema that matches the schema-3 wire DTO.

#### Scenario: Schema-2 manifest transforms

- **WHEN** Reef applies the schema-2 to schema-3 transformation to a valid manifest
- **THEN** the result declares schema 3 without a metadata field

#### Scenario: Schema artifact drifts

- **WHEN** a metadata wire field changes without the schema-3 JSON Schema change
- **THEN** the document-schema drift test fails
