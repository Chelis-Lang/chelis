# Reef Document Schema Specification

## Purpose

Define versioned Reef document formats, strict version dispatch, safe upgrades, and checked editor schemas.

## Requirements

### Requirement: Every Reef document declares one supported schema

Every nonlegacy `reef.toml` and `reef.lock` SHALL declare a top-level schema field.

The initial supported manifest and lock schema SHALL be `schema = "1"`. A new document SHALL use the newest supported schema for its document type.

A schema value SHALL be a decimal ASCII integer without a sign, whitespace, or leading zeroes. A missing schema SHALL identify legacy schema 0.

Manifest and lock schema values SHALL evolve independently.

#### Scenario: Initial manifest declares schema 1

- **WHEN** Reef reads a valid manifest with `schema = "1"`
- **THEN** Reef dispatches it to the manifest schema-1 parser

#### Scenario: New package uses the newest initial schema

- **WHEN** `reef init` runs with manifest schema 1 supported
- **THEN** the new manifest declares schema 1

#### Scenario: Legacy manifest omits the schema

- **WHEN** Reef reads a valid legacy manifest without `schema`
- **THEN** Reef accepts it through schema 0 and reports that an upgrade is available

#### Scenario: Schema value has a leading zero

- **WHEN** a document declares `schema = "01"`
- **THEN** Reef rejects the schema before it parses package or lock fields

#### Scenario: Manifest and lock use different supported schemas

- **WHEN** a package contains a current manifest and a supported legacy lock
- **THEN** Reef reads each document through its own schema parser

### Requirement: Reef selects the schema before strict field parsing

Reef SHALL parse the schema header before it selects a complete wire DTO.

Each current fixed-shape table SHALL reject unknown keys. An error SHALL identify the key and its containing table.

A newer unsupported schema SHALL produce an unsupported-schema error. Reef SHALL NOT misreport its new fields as unknown keys.

The error SHALL identify the document, found schema, newest supported schema, and `chelis reef upgrade` command.

#### Scenario: Current package field contains a typo

- **WHEN** a schema-1 manifest declares unknown package key `additional_source`
- **THEN** Reef rejects that key and identifies the `[package]` table

#### Scenario: Manifest uses a future schema

- **WHEN** a manifest declares a schema newer than the newest supported manifest schema
- **THEN** Reef reports the unsupported schema before it reports any key from that schema

#### Scenario: Lock uses a future schema

- **WHEN** a lock declares a schema newer than the newest supported lock schema
- **THEN** Reef rejects the lock without package resolution or registry mutation

#### Scenario: Schema has the wrong TOML type

- **WHEN** a document declares `schema = 1`
- **THEN** Reef rejects the schema and identifies the required string form

### Requirement: Reef upgrades documents through one command

`chelis reef upgrade` SHALL require exactly one of `--check` or `--inplace`. It SHALL accept an optional package path.

The command SHALL accept optional `--manifest-to <n>` and `--lock-to <n>` targets. An omitted target SHALL select the newest supported schema.

A target SHALL identify a supported schema that is not less than the current document schema. The command SHALL stop at each selected target.

The command SHALL preflight the complete selected manifest and lock migrations before it writes either document.

`--check` SHALL write nothing. It SHALL report every selected migration step.

`--inplace` SHALL apply selected registered steps in ascending order. It SHALL preserve accepted manifest comments and key order.

The schema-0 to schema-1 manifest step SHALL add only the schema field. The lock step SHALL preserve every exact package, source, compiler, and hash value.

#### Scenario: Check mode finds legacy documents

- **WHEN** `reef upgrade --check` reads valid schema-0 manifest and lock documents
- **THEN** it reports both schema-1 steps and leaves both files byte-identical

#### Scenario: In-place mode upgrades legacy documents

- **WHEN** `reef upgrade --inplace` reads valid schema-0 manifest and lock documents
- **THEN** both results declare schema 1 and preserve all prior semantic values

#### Scenario: Current documents need no upgrade

- **WHEN** `reef upgrade --inplace` reads current manifest and lock schemas
- **THEN** it succeeds without changing either file

#### Scenario: User selects the current manifest schema

- **WHEN** a schema-1 manifest uses `--manifest-to 1`
- **THEN** Reef succeeds without changing the manifest

#### Scenario: User selects an unsupported manifest schema

- **WHEN** only schema 1 is supported and a manifest uses `--manifest-to 2`
- **THEN** Reef rejects the target and leaves every document unchanged

#### Scenario: Legacy manifest contains an unknown key

- **WHEN** strict upgrade preflight finds an unknown legacy key
- **THEN** Reef identifies the key and leaves the manifest and lock unchanged

#### Scenario: Mode selection is invalid

- **WHEN** a user supplies both modes or neither mode
- **THEN** the command rejects its arguments before it reads or writes a package file

### Requirement: Each document replacement is atomic and recoverable

Reef SHALL acquire one persistent package-root project lock before it replaces a manifest, lock, or package archive.

A project writer SHALL add ignore rules for the lock and command-owned siblings. It SHALL preserve existing ignore content.

Reef SHALL serialize and parse each complete upgraded document before it replaces that document.

Reef SHALL reject a document target that is a symbolic link or has more than one hard link. This check SHALL occur before any document replacement.

Reef SHALL write a unique sibling file and flush it. If the target exists, Reef SHALL create a unique command-owned recovery hard link.

Reef SHALL rename the temporary sibling over the target and sync the parent where supported. A failed sync SHALL restore the prior target.

A sibling-name collision SHALL fail without changing that sibling. A command SHALL remove only siblings that it created.

A command SHALL NOT remove or overwrite an unknown sibling from another command.

Reef SHALL replace the manifest before the lock. It SHALL accept and recover a current manifest with a supported legacy lock.

The upgrade operation SHALL be idempotent. Reef SHALL NOT claim joint atomicity for the two documents.

#### Scenario: Manifest serialization fails

- **WHEN** Reef cannot serialize the upgraded manifest
- **THEN** it leaves the existing manifest and lock unchanged

#### Scenario: Manifest target is a symbolic link

- **WHEN** an in-place upgrade targets a symbolic-link `reef.toml`
- **THEN** Reef rejects the target and preserves the link and linked file

#### Scenario: Manifest target has another hard link

- **WHEN** an in-place upgrade targets a multiply-hard-linked `reef.toml`
- **THEN** Reef rejects the target and preserves both names and their bytes

#### Scenario: Lock replacement fails after manifest success

- **WHEN** the manifest replacement succeeds and the lock replacement fails
- **THEN** the package contains a valid current manifest and the prior supported lock

#### Scenario: Upgrade resumes after interruption

- **WHEN** a later upgrade reads a current manifest and legacy lock
- **THEN** it preserves the manifest and completes the remaining lock step

#### Scenario: Concurrent upgrade holds the project lock

- **WHEN** one upgrade holds the package-root project lock
- **THEN** another project writer waits without changing a target or temporary sibling

#### Scenario: Existing package lacks project-writer ignore rules

- **WHEN** a project writer acquires the package-root lock
- **THEN** Reef preserves existing ignore content and adds the required lock and sibling rules

#### Scenario: Temporary file remains after a crash

- **WHEN** Reef finds an uncommitted sibling that it did not create
- **THEN** it ignores that sibling and does not remove or treat it as the current document

#### Scenario: Recovery sibling name collides

- **WHEN** an unknown recovery sibling has the next command-owned candidate name
- **THEN** Reef fails without changing the target or that sibling

#### Scenario: Parent sync fails

- **WHEN** parent-directory sync fails after target replacement
- **THEN** Reef restores the prior target through its command-owned recovery link

### Requirement: Reef ships exact editor schemas

Reef SHALL ship one versioned JSON Schema file for each supported manifest and lock schema.

A drift test SHALL compare each committed schema with the current versioned wire model.

The manifest JSON Schema SHALL describe syntax and table shape. It SHALL NOT claim to prove filesystem, network, or artifact safety.

#### Scenario: Editor schema matches the manifest DTO

- **WHEN** the schema generator runs for manifest schema 1
- **THEN** its bytes equal `docs/schemas/reef/manifest-v1.schema.json`

#### Scenario: A DTO field changes without a schema update

- **WHEN** a mutation adds a manifest wire field without the corresponding schema artifact change
- **THEN** the drift test fails and identifies manifest schema 1

#### Scenario: Schema-valid metadata refers to an unsafe path

- **WHEN** a future manifest passes JSON Schema but its declared file path fails Reef safety rules
- **THEN** Reef rejects the manifest through its typed or filesystem boundary
