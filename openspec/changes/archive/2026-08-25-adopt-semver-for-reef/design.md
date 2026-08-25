## Context

Reef stores package names and versions as `String` values across manifests, indexes, locks, release specifications, and shell IDs.

The current resolver requires exact text equality. It sorts registry versions as text and resolves the manifest again during each package build.

`version-reef-manifest-schema` adds schema 1, strict current DTOs, and one general upgrade command.

Remote range discovery needs pagination, archive inspection, update policy, and network budgets. `add-bounded-reef-remote-discovery` owns those mechanisms.

The `.chb`, archive, registry-index, lock, and prepared-graph formats are compatibility surfaces. This change preserves their version field locations.

## Accepted prerequisites and implementation order

The schema prerequisite is commit `d6479aaa2ad5fea22ac7ac773e88491e22ffbbd0`. Its final document-schema oracle passed 32 tests.

This change lands before the Reef slice of `migrate-rust-errors-to-snafu`. The resolver owns one self-contained typed error until that later migration.

Do not implement both error changes in parallel. The SNAFU change must adopt the accepted resolver error after this change lands.

## Goals / Non-Goals

**Goals:**

- Parse untrusted package identities and versions at the first Reef boundary.
- Add Cargo-compatible requirement semantics without silent legacy reinterpretation.
- Support concise dependencies and checked local path overrides.
- Produce one deterministic package version for each package name.
- Bound local graph search and return actionable conflicts.
- Keep a valid exact lock stable.
- Preserve exact toolchain and bundled-runtime selection.
- Prepare a provider boundary without remote requests.

**Non-Goals:**

- Enumerate remote releases or add update and outdated commands.
- Activate resolver 2 for generated or migrated manifests.
- Add a public registry, mirrors, multiple publishers, or dependency patches.
- Implement Cargo features or permit two versions of one package name.
- Change artifact formats, hashes, compiler semantics, or backend output.
- Transfer specification authority from `spec/**` to OpenSpec.

## Decisions

### 1. Use `semver` version 1 as the package policy engine

Add `semver = { version = "1", features = ["serde"] }` to workspace dependencies. `chelis-reef` consumes the workspace entry.

Use `semver::Version` for complete package versions. Use `semver::VersionReq` for schema-2 dependency requirements.

A package version can contain a SemVer prerelease. It cannot contain build metadata.

Cargo requirements ignore build metadata. Accepting it creates two package identities that no requirement can select separately.

`chelis-version` retains toolchain path and installability policy. This change does not broaden toolchain references.

Alternative custom parsing repeats logic that already drifted across Reef and version tooling.

### 2. Parse one canonical package name

Introduce a private `PackageName` type. Parse it at manifest, dependency, lock, index, shell, and release boundaries.

A package name must meet these rules:

- It contains 1 through 64 ASCII bytes.
- Its first byte is an ASCII lowercase letter.
- Its other bytes are ASCII lowercase letters, digits, or hyphens.
- Each hyphen occurs between two alphanumeric bytes.
- It is not `con`, `prn`, `aux`, `nul`, `com1` through `com9`, or `lpt1` through `lpt9`.

Do not normalize an invalid name. Reject it before it becomes a map key, path component, archive name, or source selector.

Alternative warning-only checks permit unsafe strings to reach filesystem code.

### 3. Separate wire text from typed resolver models

Use the schema dispatch from `version-reef-manifest-schema`.

Manifest schema 1 preserves current exact dependency semantics. This change defines the typed resolver-2 model for a later schema-2 DTO.

The typed model contains these forms:

- `PackageName`
- `PackageVersion(Version)`
- `ResolverVersion` with versions 1 and 2
- `PackageRequirement(VersionReq)`
- `ExactCompilerVersion(Version)`
- `DependencySource::Registry`
- `DependencySource::Path`
- `ResolvedPackageId`

Convert the schema-1 wire DTO and resolver-2 test inputs through `TryFrom`. Resolver code receives no raw package identity or dependency option pair.

Compiler-pin conversion requires one exact comparator with major, minor, and patch values. It rejects prerelease and build metadata.

Serialized models use canonical strings in their current field locations.

`chelis-shell::PackageId` remains unchanged because it is part of `.chb`. Reef parses its fields after shell decode and before verification.

### 4. Keep resolver semantics explicit

Schema 1 permits no resolver field and accepts only complete exact dependency versions.

The resolver-2 model parses each registry dependency through `VersionReq`. A later schema-2 DTO requires `resolver = "2"`.

A bare resolver-2 version uses caret semantics. An `=` operator requires one exact version.

Resolver 2 accepts these equivalent forms:

```toml
[dependencies]
nautilus = "0.7"
coral = { version = "=0.6.2" }
```

This change defines the exact schema-1 to schema-2 dependency transformation. Bounded remote discovery later adds the DTO and activates that transformation.

Alternative a missing resolver in schema 2 creates another implicit semantic branch.

### 5. Permit checked path requirements and overrides

A path dependency can declare only `path`, or it can declare `path` and `version`.

Reef loads the path package and verifies that its package name equals the dependency key.

Schema 1 treats a path version as exact. Resolver 2 treats it as a Cargo-style requirement.

One active path source for a package name overrides registry candidates for that name. Its parsed version must satisfy every active registry requirement.

Two different active paths for one package name cause a path-source conflict. Reef does not compare override bytes with an unused registry candidate.

A path-only dependency remains valid for local work. `chelis reef publish` continues to reject every path dependency.

Alternative byte comparison with registry candidates defeats the purpose of a local override.

### 6. Resolve bounded local candidate sets

Local candidate sources are:

1. the current exact lock
2. the local registry index
3. the compiler-bundled `chelis-std`
4. path dependencies

No source in this change performs a network request.

The `reef-package-versioning` capability specification owns the exact limits for candidates, packages, dependencies, depth, and resolver states.

Sort unresolved package names by byte value. Sort candidates by descending SemVer precedence.

Use deterministic depth-first search. A failed candidate branch returns to the next candidate while the state budget remains.

Memoize failed states. Use checked counters and return a typed limit error before an operation exceeds a limit.

The first complete graph wins. The single-version rule remains explicit because module and linker identities use the package name.

Non-path candidates with one exact version from multiple sources must have identical verified bytes and source identity. Otherwise, resolution fails as a source conflict.

An active path override follows the path-source rules and excludes unused registry candidates from this comparison.

Alternative greedy selection can fail when a high candidate adds an incompatible transitive requirement.

### 7. Make the lock the preferred exact solution

A normal command reads `reef.lock` first when it exists. Reef parses every identity and verifies these facts:

- the root package identity equals the current manifest
- every locked version satisfies current requirements
- every locked source kind matches the current dependency declaration
- every locked path equals the current canonical path declaration
- every locked source contains required metadata
- every consumed artifact passes its hash checks

If all facts hold, graph reconstruction uses the lock. It performs no version search and does not rewrite the file.

A registry-to-path or path-to-registry declaration change invalidates that locked source preference.

If a requirement excludes a locked version, the local resolver uses valid locked entries as preferences where permitted.

A hash failure, unavailable locked origin, or source mismatch remains a hard integrity failure. It does not start local or remote replacement.

`chelis reef install --from-lockfile` stays exact. It never interprets lock versions as requirements.

Alternative resolution on every build creates implicit compatible upgrades.

### 8. Keep the bundled runtime and compiler exact

`package.compiler` remains one exact complete stable version with an `=` operator.

The compiler-bundled `chelis-std` version is the only runtime candidate. An explicit requirement must match that candidate.

Reef does not search the local registry or a provider for another runtime.

This rule preserves the toolchain contract in `spec/design/chelis_packaging_and_install.md`.

### 9. Preserve external formats and invalidate internal caches

Keep package-name and version fields as strings in their current locations.

Keep release assets and registry paths based on canonical exact values. Sort local index versions through parsed SemVer precedence.

Reject an invalid stored package name or version as index or lock corruption.

Keep lock dependencies in canonical package-name order. Do not let graph traversal order change lock bytes.

Bump `PREPARED_GRAPH_CACHE_VERSION` for the typed manifest payload. A stale cache follows the current fail-closed rebuild path.

Do not change `.chb`, source archive, `reef.lock`, or `index.json` field locations.

### 10. Return typed local resolution errors

Add typed errors for these classes:

- invalid package name
- invalid package version
- invalid dependency requirement
- invalid exact compiler pin
- unsupported resolver version
- path package mismatch
- local candidate limit
- graph limit
- incompatible requirement set
- source conflict
- invalid lock
- unavailable locked origin

Each conflict carries the package, requester names, requirement text, considered candidates, and reached limit where applicable.

Presentation converts the typed error at the CLI boundary.

Choose the Reef error owner before implementation. If the SNAFU Reef phase lands first, use that owner.

Do not implement the SNAFU Reef phase and this resolver in parallel branches.

### 11. Hand activation to bounded remote discovery

This change adds the local resolver and typed resolver-2 model. It does not add a supported schema-2 document parser.

`add-bounded-reef-remote-discovery` adds the schema-2 DTO, migration registration, editor schema, remote providers, update policy, and activation.

Resolver-2 output activates only after the schema, local resolver, and remote discovery oracles pass on one revision.

This ordering prevents a fresh resolver-2 project from requiring an unbounded or absent remote path.

### 12. Use one executable completion oracle

The authoritative completion oracle is:

```sh
cargo nextest run -p chelis-reef --test reef_package_versioning --no-fail-fast
```

The suite covers identities, both resolver parsers, path requirements, local search, limits, lock reuse, integrity failures, and format stability.

A fresh local red-team agent mutates names, versions, requirements, graph order, graph size, local sources, indexes, and locks.

OpenSpec validation and documentation builds are supporting evidence. They do not replace the oracle.

Hosted acceptance starts after the local archive is ready. The final revision requires green Linux integration, documentation, and applicable release checks.

Hosted fixtures use private canonical repositories and perform no remote version enumeration. Hosted results support but do not replace the local oracle.

## Risks / Trade-offs

- **[Risk] Two resolver parsers increase the test matrix.** → Keep two closed manifest schemas and paired fixtures.
- **[Risk] A local limit rejects a large valid graph.** → Report the exact capability limit and change that requirement through review.
- **[Risk] Malformed local indexes now fail where text equality skipped rows.** → Report the path and invalid value, then require registry repair.
- **[Risk] Resolver 2 appears available before remote support.** → Keep generation and migration inactive until bounded discovery passes.
- **[Risk] The SNAFU migration touches the same errors.** → Select one implementation order before either branch starts.
- **[Risk] Schema 1 cannot exercise compatible candidate selection.** → Run its exact graph through the same bounded resolver core and keep resolver-2 activation separate.
- **[Risk] Transitive lock verification reads local manifests before graph reconstruction.** → Prefer correctness and keep the exact lock stable after verification.
- **[Risk] Existing public Reef APIs return text errors.** → Preserve those APIs and keep resolver failures typed until the legacy library boundary. The later SNAFU change owns broader propagation.

## Migration Plan

1. Land `version-reef-manifest-schema` and migrate controlled documents to schema 1.
2. Select the Reef error-owner order with `migrate-rust-errors-to-snafu`.
3. Add positive and negative identity, parser, resolver, limit, and lock fixtures.
4. Add typed models while external serializers remain unchanged.
5. Add the typed resolver-2 model and bounded local resolver.
6. Add lock verification and deterministic local reconstruction.
7. Run strict validation, the oracle, and fresh adversarial validation.
8. Land `add-bounded-reef-remote-discovery` before resolver-2 activation.
9. Run all prerequisite oracles on one activation revision.

Rollback removes the resolver-2 model and local resolver while schema-1 exact behavior remains active.

No manifest or lock conversion is necessary before activation because controlled documents remain on schema 1.

## Open Questions

None.
