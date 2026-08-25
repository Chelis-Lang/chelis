# Reef Remote Discovery Specification

## Purpose

Define bounded remote candidate discovery, explicit refresh behavior, safe candidate inspection, and recoverable cache publication.

## Requirements

### Requirement: Every resolver operation selects one discovery mode

Reef SHALL select one closed discovery mode before it creates a candidate provider.

A valid lock SHALL use locked mode and perform no version listing. Normal unlocked resolution SHALL prefer a complete local graph.

`chelis reef update` SHALL use refresh mode and query remote providers even when a local graph is complete.

`chelis reef outdated` SHALL use inspect mode and write no final registry, index, or lock state.

When network access is disabled, Reef SHALL NOT create an HTTP client or call a remote provider.

#### Scenario: Valid lock prevents version discovery

- **WHEN** a valid lock supplies one complete graph during a normal build
- **THEN** Reef uses exact locked versions without a release-list request

#### Scenario: Local graph satisfies normal unlocked resolution

- **WHEN** no valid lock exists and local candidates form one complete graph
- **THEN** normal resolution uses that graph without remote discovery

#### Scenario: Full update has a compatible local graph

- **WHEN** a newer compatible remote graph exists and the local registry also contains a compatible graph
- **THEN** `reef update` queries the remote provider and selects the newest complete compatible graph

#### Scenario: Outdated inspects remote versions

- **WHEN** a user runs `reef outdated --json`
- **THEN** Reef reports remote compatibility without a final registry, index, or lock write

#### Scenario: Network access is disabled

- **WHEN** any resolver mode runs with network access disabled
- **THEN** Reef uses no remote provider and makes no HTTP request

### Requirement: Remote providers use one typed source-neutral boundary

Reef SHALL obtain remote candidates through a typed provider interface.

A provider SHALL return parsed package identities, typed source locators, and bounded manifest data. The resolver SHALL NOT contain provider-specific HTTP logic.

The first supported provider SHALL use GitHub Releases. Reef SHALL reject an unsupported source scheme.

Current and future descriptive manifest URLs SHALL NOT select a provider, source, mirror, release, or credential.

#### Scenario: GitHub provider returns a candidate

- **WHEN** the GitHub provider accepts one release
- **THEN** it returns a parsed package identity and typed GitHub source locator

#### Scenario: Manifest contains an unrelated URL field in a later schema

- **WHEN** a later supported schema adds a descriptive URL field
- **THEN** provider selection remains based only on typed source configuration and locked origins

#### Scenario: Provider returns malformed identity text

- **WHEN** a provider response contains an invalid package name or version
- **THEN** Reef rejects that response before it enters the candidate set

#### Scenario: Lock contains an unsupported origin scheme

- **WHEN** Reef reads a locked origin with an unsupported scheme
- **THEN** it rejects the origin without a fallback request

### Requirement: Remote discovery obeys finite production limits

Reef SHALL enforce the production limits in the following table.

| Dimension | Limit |
|---|---:|
| GitHub release pages per package | 10 |
| HTTP requests per command | 2048 |
| accepted release tags per package | 256 |
| fetched candidate manifests per package | 64 |
| compressed candidate archive | 64 MiB |
| scanned decompressed bytes per candidate | 256 MiB |
| parsed `reef.toml` bytes | 1 MiB |
| total candidate download bytes | 1 GiB |
| elapsed time per HTTP request | 60 seconds |

Remote candidates SHALL also count against the local candidate limit from `reef-package-versioning`.

Remote resolution SHALL inherit the local package, dependency, depth, and resolver-state limits from `reef-package-versioning`.

A production command SHALL NOT disable a limit. Budget counters SHALL use checked arithmetic.

A budget error SHALL identify the dimension, limit, observed value, package, and provider operation.

#### Scenario: Release pagination reaches its limit

- **WHEN** a package requires an eleventh GitHub release page
- **THEN** Reef stops before that request and reports the page limit

#### Scenario: Counter arithmetic overflows

- **WHEN** budget accounting cannot represent the next observed value
- **THEN** Reef reports a budget failure instead of wrapping the counter

### Requirement: Candidate manifest inspection does not extract the archive

Reef SHALL download a candidate archive into command-owned temporary storage. It SHALL enforce compressed and total byte limits during download.

Reef SHALL scan the compressed tar stream without unpacking it. It SHALL accept exactly one root regular-file entry named `reef.toml`.

Reef SHALL reject duplicate manifests, links, devices, absolute paths, parent paths, malformed records, excessive scan bytes, and excessive manifest bytes.

Reef SHALL parse the manifest schema and strict DTO before the candidate enters resolution.

#### Scenario: Candidate has one valid root manifest

- **WHEN** an archive contains one bounded regular `reef.toml` at its root
- **THEN** Reef parses that manifest without extracting another archive member

#### Scenario: Candidate has two root manifests

- **WHEN** an archive contains duplicate root `reef.toml` entries
- **THEN** Reef rejects the candidate as ambiguous

#### Scenario: Candidate manifest is a symbolic link

- **WHEN** the `reef.toml` tar entry is a symbolic link
- **THEN** Reef rejects the candidate without reading the link target

#### Scenario: Decompression scan exceeds its limit

- **WHEN** Reef scans more than 256 MiB before candidate inspection completes
- **THEN** it stops the stream and reports the decompressed-byte limit

#### Scenario: Candidate uses a future manifest schema

- **WHEN** candidate `reef.toml` declares an unsupported schema
- **THEN** Reef rejects the candidate before dependency resolution

### Requirement: Remote resolution remains deterministic

Reef SHALL sort package names by byte value. Resolve mode SHALL prefer a compatible lock candidate before descending SemVer precedence.

Refresh and inspect modes SHALL remove the selected package's lock preference and use descending SemVer precedence.

Reef SHALL cache one provider query result for each package and provider during one command.

Reef SHALL memoize failed resolver states and count each new state against the state budget.

API order, response time, and source metadata text SHALL NOT change the selected graph.

#### Scenario: Provider response order changes

- **WHEN** two runs return the same releases in different API orders
- **THEN** Reef selects the same graph and reports the same ordered conflict data

#### Scenario: One candidate creates a dead end

- **WHEN** the highest candidate creates an unsatisfied transitive requirement
- **THEN** Reef tries the next candidate within the resolution budget

#### Scenario: A release tag contains build metadata

- **WHEN** a provider returns a version with Semantic Version build metadata
- **THEN** the package-version boundary rejects that result before candidate insertion

### Requirement: Update and outdated expose compatible version state

`chelis reef update` SHALL resolve without the selected lock preferences. A full update SHALL refresh all remote package candidates.

A targeted update SHALL refresh the named package. It SHALL keep unrelated locked packages fixed unless the selected target requires a transitive change.

`chelis reef outdated` SHALL report the current version, newest compatible version, newest incompatible version, and blocked reason where applicable.

An exact requirement SHALL remain exact. A failed command SHALL preserve the prior lock.

#### Scenario: Full update finds a newer remote package

- **WHEN** a newer remote package satisfies all requirements
- **THEN** `reef update` selects it and writes its exact version after complete verification

#### Scenario: Targeted update preserves an unrelated package

- **WHEN** one target updates without a transitive change to an unrelated package
- **THEN** Reef keeps the unrelated locked version

#### Scenario: Exact requirement blocks an update

- **WHEN** a package requirement is `=1.2.3`
- **THEN** update and outdated report `1.2.3` as the only compatible version

#### Scenario: JSON outdated result has no changes

- **WHEN** every locked version is the newest compatible version
- **THEN** `reef outdated --json` reports an empty update set and writes no final state

### Requirement: Reef publishes selected packages before the new lock

Reef SHALL verify every selected package in temporary storage before final publication.

Reef SHALL acquire the package-root project lock before the Reef registry lock. It SHALL hold both locks through package, index, and project `reef.lock` replacement.

Under both locks, Reef SHALL publish each absent complete package directory through a same-parent rename.

An existing same-name and same-version directory SHALL match the selected bytes. Different bytes SHALL cause a source conflict.

Reef SHALL update the registry index only after complete package directories exist. It SHALL replace the project lock last through atomic replacement.

A failure after cache publication can leave unused complete verified entries. Reef SHALL NOT write a lock that names an incomplete entry.

#### Scenario: Candidate verification fails

- **WHEN** one selected artifact pair fails complete verification
- **THEN** Reef preserves the prior lock and publishes none of the staged candidates

#### Scenario: Concurrent update holds the project lock

- **WHEN** one update holds the package-root project lock
- **THEN** another project writer waits without changing a project document or temporary sibling

#### Scenario: Final package already contains identical bytes

- **WHEN** a concurrent command published the selected exact bytes first
- **THEN** Reef accepts that cache entry and continues

#### Scenario: Final package contains different bytes

- **WHEN** a final name and version contain bytes that differ from the selected pair
- **THEN** Reef reports a source conflict and preserves the prior lock

#### Scenario: Lock replacement fails after cache publication

- **WHEN** complete cache entries and the index succeed but the project lock replacement fails
- **THEN** Reef preserves the prior lock and permits the unused complete cache entries to remain

#### Scenario: Command succeeds

- **WHEN** all selected cache entries and the final lock replacement succeed
- **THEN** the new lock names only complete verified cache entries

### Requirement: Locked integrity failures never start discovery

A valid lock SHALL use exact versions and exact recorded origins. Reef SHALL NOT list releases for a locked package.

A missing local locked package can use one exact-origin fetch. A hash failure, unavailable release, or source mismatch SHALL stop the operation.

Reef SHALL NOT replace a failed locked artifact with another compatible version during a normal command.

#### Scenario: Locked bytes fail their hash

- **WHEN** exact locked-origin bytes differ from the recorded hashes
- **THEN** Reef fails without release discovery or lock replacement

#### Scenario: Locked release is unavailable

- **WHEN** the exact locked origin no longer supplies its release
- **THEN** Reef fails without selecting another release

#### Scenario: User requests an explicit update after lock failure

- **WHEN** a user corrects the integrity condition and then runs `reef update`
- **THEN** the separate refresh command can perform bounded discovery

### Requirement: Manifest schema 2 exposes resolver-2 semantics

Manifest schema 2 SHALL require `package.resolver = "2"`. It SHALL reject a missing or unsupported resolver before dependency conversion.

The schema-2 DTO SHALL accept resolver-2 string and inline dependency forms. It SHALL convert them through `reef-package-versioning`.

The schema-1 to schema-2 transformation SHALL rewrite each exact registry dependency `X.Y.Z` as `=X.Y.Z`.

It SHALL preserve path-only dependencies. It SHALL rewrite a version on a path dependency through the same exact rule.

The transformation SHALL preserve comments, key order, and exact package selection.

Reef SHALL ship a schema-2 manifest JSON Schema that matches the schema-2 DTO.

#### Scenario: Schema-2 manifest selects resolver 2

- **WHEN** a schema-2 manifest declares `resolver = "2"` and dependency version `1.2.3`
- **THEN** Reef converts that dependency through resolver-2 caret semantics

#### Scenario: Schema-2 manifest omits the resolver

- **WHEN** a schema-2 manifest omits `package.resolver`
- **THEN** Reef rejects the manifest before dependency conversion

#### Scenario: Exact dependency transforms to schema 2

- **WHEN** a schema-1 manifest contains dependency version `1.2.3`
- **THEN** its schema-2 form contains `=1.2.3` and preserves exact selection

#### Scenario: Path-only dependency transforms to schema 2

- **WHEN** a schema-1 manifest contains a dependency with only `path`
- **THEN** its schema-2 form preserves that path without an invented version

#### Scenario: Invalid exact dependency blocks transformation

- **WHEN** a schema-1 dependency is not one complete Semantic Version
- **THEN** Reef identifies that dependency and leaves the manifest unchanged

#### Scenario: Schema-2 editor artifact drifts

- **WHEN** the schema-2 DTO changes without the schema-2 JSON Schema change
- **THEN** the document-schema drift test fails
