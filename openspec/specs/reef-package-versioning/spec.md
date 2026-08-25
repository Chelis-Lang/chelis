# Reef Package Versioning Specification

## Purpose

Define typed package identities, Cargo-style requirements, bounded local resolution, and exact lock stability.

## Requirements

### Requirement: Reef accepts only canonical package identities

A package name SHALL contain 1 through 64 ASCII bytes. Its first byte SHALL be an ASCII lowercase letter.

All other bytes SHALL be ASCII lowercase letters, digits, or hyphens. Each hyphen SHALL occur between two alphanumeric bytes.

A package name SHALL NOT be `con`, `prn`, `aux`, `nul`, `com1` through `com9`, or `lpt1` through `lpt9`.

Reef SHALL apply this rule to manifests, dependency keys, locks, registry entries, shell metadata, and release repository names. Reef SHALL NOT normalize an invalid name.

A package version SHALL be a complete Semantic Version. It can contain prerelease identifiers. It SHALL NOT contain build metadata.

Reef SHALL apply the version rule at every package identity boundary. A release tag can add one leading `v`.

#### Scenario: Canonical package identity is valid

- **WHEN** a manifest declares `name = "nautilus-core"` and `version = "1.2.3-alpha.1"`
- **THEN** Reef accepts the parsed package identity

#### Scenario: Package name contains a path separator

- **WHEN** a manifest declares `name = "../nautilus"`
- **THEN** Reef rejects `package.name` before it constructs a registry path

#### Scenario: Package name uses a noncanonical spelling

- **WHEN** a dependency key is `Nautilus_Core`
- **THEN** Reef rejects the dependency key without normalization

#### Scenario: Package name is reserved on a supported platform

- **WHEN** a manifest declares `name = "con"`
- **THEN** Reef rejects `package.name` before graph construction

#### Scenario: Partial package version is invalid

- **WHEN** a manifest declares `version = "1.2"`
- **THEN** Reef rejects the manifest and identifies `package.version`

#### Scenario: Package version contains build metadata

- **WHEN** a manifest declares `version = "1.2.3+portable"`
- **THEN** Reef rejects build metadata because dependency requirements cannot select it

#### Scenario: Direct release tag is invalid

- **WHEN** a user requests `Chelis-Lang/nautilus@latest`
- **THEN** Reef rejects the release specification before it derives an asset name

### Requirement: Each resolver version has one dependency meaning

Resolver 1 SHALL accept only complete exact package versions. Resolver 2 SHALL parse dependency versions as Cargo-style requirements.

An unsupported resolver SHALL fail before dependency conversion.

#### Scenario: Resolver-1 dependency is exact

- **WHEN** resolver 1 receives dependency version `1.2.3`
- **THEN** Reef matches only package version `1.2.3`

#### Scenario: Resolver 2 uses caret semantics

- **WHEN** resolver 2 receives dependency version `1.2.3`
- **THEN** Reef interprets the dependency as `^1.2.3`

#### Scenario: Resolver value is unsupported

- **WHEN** dependency conversion receives resolver 3
- **THEN** Reef rejects the resolver before it parses the dependency

### Requirement: Resolver-2 dependencies use Cargo-style forms

A resolver-2 dependency SHALL use a version string or an inline table. A version string SHALL equal an inline table that contains only `version`.

Reef SHALL evaluate all requirement comparators as one intersection. A bare version SHALL use caret semantics.

An `=` operator SHALL require one exact version. A stable requirement SHALL exclude prerelease versions under Cargo prerelease rules.

Resolver 1 SHALL reject a requirement operator because its dependency value is one complete exact package version.

#### Scenario: Shorthand and inline forms are equivalent

- **WHEN** resolver 2 receives `nautilus = "1.2.3"` and `nautilus = { version = "1.2.3" }`
- **THEN** Reef parses both forms as the same requirement

#### Scenario: Bare stable requirement permits a compatible release

- **WHEN** a resolver-2 dependency declares `version = "1.2.3"`
- **THEN** `1.9.0` matches and `2.0.0` does not match

#### Scenario: Bare zero-major requirement stays in one minor line

- **WHEN** a resolver-2 dependency declares `version = "0.2.3"`
- **THEN** `0.2.9` matches and `0.3.0` does not match

#### Scenario: Exact requirement preserves exact behavior

- **WHEN** a resolver-2 dependency declares `version = "=1.2.3"`
- **THEN** `1.2.3` matches and `1.2.4` does not match

#### Scenario: Stable requirement excludes a prerelease

- **WHEN** a resolver-2 dependency declares `version = "1.2.3"`
- **THEN** `1.3.0-alpha.1` does not match

#### Scenario: Invalid requirement fails at the resolver boundary

- **WHEN** a resolver-2 dependency declares `version = ">=1.0 <2.0"`
- **THEN** Reef rejects the value and identifies that dependency requirement

#### Scenario: Resolver 1 rejects a requirement operator

- **WHEN** a resolver-1 dependency declares `version = "=1.2.3"`
- **THEN** Reef rejects it as an invalid exact package version

### Requirement: Path dependencies can retain version requirements

A path dependency SHALL declare `path`. It can also declare `version`.

Reef SHALL verify that the loaded path package name equals the dependency key. If `version` exists, Reef SHALL verify the loaded package version against it.

Resolver 1 SHALL treat the path version as exact. Resolver 2 SHALL treat it as a Cargo-style requirement.

One active path source for a package name SHALL override registry candidates for that name.

The path package version SHALL satisfy every active registry requirement. Two different active paths for one name SHALL cause a path-source conflict.

Reef SHALL NOT compare an active path override with bytes from an unused registry candidate.

A path-only dependency SHALL remain valid for local work. `chelis reef publish` SHALL continue to reject every path dependency.

#### Scenario: Path package satisfies its requirement

- **WHEN** resolver 2 receives `{ path = "../octant", version = "0.4" }` and the path package is version `0.4.7`
- **THEN** Reef accepts the path package for local resolution

#### Scenario: Path package violates its requirement

- **WHEN** resolver 2 requires `0.4` and the path package version is `0.5.0`
- **THEN** Reef rejects the dependency and identifies the required and actual versions

#### Scenario: Path package name differs from the dependency key

- **WHEN** dependency key `octant` loads a path package named `shoals`
- **THEN** Reef rejects the dependency before graph construction

#### Scenario: Path-only dependency remains valid locally

- **WHEN** a dependency declares only a safe path to a valid package
- **THEN** Reef resolves that package without an invented requirement

#### Scenario: Path override shares a version with the registry

- **WHEN** a valid path package and unused registry package share one name and version with different bytes
- **THEN** Reef selects the path package without a registry-byte comparison

#### Scenario: Two path overrides disagree

- **WHEN** active requirements select two different paths for one package name
- **THEN** Reef rejects a path-source conflict and identifies both paths

#### Scenario: Publication contains a path override

- **WHEN** a package contains a path dependency and a user runs `chelis reef publish`
- **THEN** Reef rejects publication through the existing path-dependency rule

### Requirement: Toolchain and runtime versions remain fixed

The `package.compiler` field SHALL contain one exact complete stable release version with an `=` operator.

It SHALL NOT contain a range, prerelease, or build metadata.

Reef SHALL treat the compiler-bundled `chelis-std` version as the only runtime candidate. An explicit runtime requirement SHALL match that candidate.

#### Scenario: Exact compiler pin matches the active toolchain

- **WHEN** a package declares the exact active compiler version
- **THEN** Reef accepts the compiler pin

#### Scenario: Compiler range is invalid

- **WHEN** a package declares `compiler = "^0.18.4"`
- **THEN** Reef rejects the manifest before dependency resolution

#### Scenario: Runtime requirement matches the bundled runtime

- **WHEN** an explicit `chelis-std` requirement matches the bundled runtime
- **THEN** Reef resolves that runtime without registry or network access

#### Scenario: Runtime requirement excludes the bundled runtime

- **WHEN** an explicit `chelis-std` requirement excludes the bundled runtime
- **THEN** Reef rejects the graph and does not search for another runtime

### Requirement: Reef bounds deterministic local resolution

Reef SHALL resolve at most one version for each package name. Every selected version SHALL satisfy all active requirements for that name.

Reef SHALL sort unresolved package names by byte value. A compatible locked candidate SHALL sort before unlocked local candidates. Other candidates SHALL use descending SemVer precedence.

Local resolution SHALL enforce these limits:

| Dimension | Limit |
|---|---:|
| candidates per package name | 256 |
| resolved package names | 256 |
| dependencies per manifest | 256 |
| dependency depth | 128 |
| explored resolver states | 100000 |

Reef SHALL memoize failed states and use checked counters. It SHALL report a limit before it exceeds that limit.

A non-path candidate with one exact identity from two local sources SHALL have identical verified bytes and source identity.

An active path override SHALL exclude unused registry candidates from this source comparison.

#### Scenario: Numeric order selects the highest compatible version

- **WHEN** compatible local candidates are `1.5.0` and `1.19.0`
- **THEN** Reef resolves `1.19.0`

#### Scenario: Resolver tries a lower candidate after a dead end

- **WHEN** the highest candidate creates an unsatisfied transitive requirement
- **THEN** Reef tries the next candidate while the state budget remains

#### Scenario: Two parents require incompatible versions

- **WHEN** two packages require disjoint version sets for one dependency
- **THEN** Reef rejects the graph and identifies both requesters and requirements

#### Scenario: Candidate count exceeds its limit

- **WHEN** a package has 257 local candidate versions
- **THEN** Reef rejects local resolution and identifies the 256-candidate limit

#### Scenario: Resolver state reaches its limit

- **WHEN** the next branch requires resolver state 100001
- **THEN** Reef stops before that branch and reports the state limit

#### Scenario: Same non-path identity has different bytes

- **WHEN** two non-path sources claim one name and version with different verified bytes
- **THEN** Reef rejects the source conflict without a priority tie-break

### Requirement: A valid lock keeps the exact resolved graph

`reef.lock` SHALL record exact package versions, source origins, compiler pins, and content hashes.

Before reuse, Reef SHALL verify the root identity and every locked version against current requirements.

Reef SHALL verify that each locked source kind matches the current dependency declaration. A locked path SHALL equal the current canonical path declaration.

A registry-to-path, path-to-registry, or path-value change SHALL invalidate that locked source preference.

Reef SHALL preserve a valid lock without a version search or rewrite.

If a manifest requirement excludes a locked version, Reef SHALL invalidate that preference and run bounded local resolution.

If no local graph completes, the local resolver SHALL return the unresolved requirement set to its caller.

An integrity failure SHALL remain a hard failure. Reef SHALL NOT replace damaged or unavailable locked bytes with another version.

#### Scenario: Valid lock prevents an implicit upgrade

- **WHEN** a lock pins `1.2.3`, its requirement still matches, and local `1.9.0` exists
- **THEN** a normal command uses `1.2.3` and leaves the lock unchanged

#### Scenario: Manifest change invalidates a locked version

- **WHEN** a requirement excludes the locked version and one local graph completes
- **THEN** Reef selects that local graph after complete bounded resolution

#### Scenario: Registry dependency becomes a path override

- **WHEN** the manifest changes one locked registry dependency to a valid path dependency
- **THEN** Reef invalidates the registry preference and resolves the path package

#### Scenario: Locked path differs from the manifest path

- **WHEN** a path dependency now names a different canonical path
- **THEN** Reef invalidates the locked path before graph reconstruction

#### Scenario: Locked artifact fails its hash check

- **WHEN** locked package bytes do not match the recorded hashes
- **THEN** Reef fails without version search or lock replacement

#### Scenario: Install from lock uses exact versions

- **WHEN** a user runs `chelis reef install --from-lockfile`
- **THEN** Reef fetches only exact versions and origins from the lock

#### Scenario: No local candidate completes an invalidated lock

- **WHEN** a stale requirement has no complete local graph
- **THEN** the local resolver returns the unresolved package and active requirements to its caller

### Requirement: Version fields retain their external locations

Reef SHALL preserve string fields for names and versions in `reef.toml`, `reef.lock`, `index.json`, `.chb`, and archive names.

Reef SHALL sort local registry versions through SemVer precedence. It SHALL sort lock dependencies by canonical package name.

A stale prepared-graph cache SHALL rebuild through current manifest and lock validation.

#### Scenario: Lock version remains an exact string

- **WHEN** Reef serializes a locally resolved lock
- **THEN** each package version remains one exact string in its current TOML location

#### Scenario: Registry index uses numeric order

- **WHEN** an index contains `1.5.0` and `1.19.0`
- **THEN** Reef orders `1.5.0` before `1.19.0`

#### Scenario: Graph traversal order changes

- **WHEN** two equivalent graphs use different internal traversal orders
- **THEN** Reef writes byte-identical ordered lock dependencies

#### Scenario: Internal cache uses the prior typed manifest

- **WHEN** Reef reads a prepared cache from the prior model
- **THEN** it rejects the stale cache and rebuilds through current boundaries
