## Why

Reef treats package versions as opaque strings and compares them for exact text equality. This blocks compatible requirements and numeric ordering.

A direct switch to Cargo caret semantics can silently reinterpret every existing dependency. Package names also become paths without one canonical identity type.

Remote range discovery adds separate network and resource risks. The `add-bounded-reef-remote-discovery` change owns that surface and resolver-2 activation.

## What Changes

- Depend on `version-reef-manifest-schema` and its strict schema-1 boundary.
- Add the Rust `semver` crate as the sole parser and matcher for Reef package versions and dependency requirements.
- Add private types for package names, package versions, requirements, compiler pins, resolver versions, and resolved package IDs.
- Reject package-version build metadata because Cargo requirements cannot select it.
- Define the typed resolver-2 semantics that manifest schema 2 will expose.
- Keep manifest schema 1 on legacy exact dependency semantics.
- Accept dependency shorthand such as `nautilus = "0.7"` in schema 2.
- Permit a path dependency to retain a version requirement and verify its package identity.
- Keep `package.compiler` and the compiler-bundled `chelis-std` version exact.
- Resolve at most one version for each package name.
- Use deterministic bounded depth-first search across lock, local registry, bundled runtime, and path candidates.
- Reuse an exact valid lock without version search or file rewrite.
- Reject invalid identities, incompatible requirements, source conflicts, and integrity failures before graph construction.
- Preserve string locations in `reef.toml`, `reef.lock`, `index.json`, `.chb`, and artifact names.
- Let `add-bounded-reef-remote-discovery` own the schema-2 DTO, migration registration, editor schema, generation, and remote commands.

Out of scope:

- Remote release enumeration, `reef update`, `reef outdated`, mirrors, or multiple publishers.
- Resolver-2 activation in generated, controlled, or downstream manifests.
- Multiple resolved versions of one package name.
- SemVer ranges for `chelisup` or toolchain selection.
- Descriptive package metadata, development dependencies, features, workspaces, build scripts, or dependency patches.
- A change to `.chb`, archive, registry-index, or lock field locations.
- Compiler, runtime, backend, language, or generated-code semantics.

## Capabilities

### New Capabilities

- `reef-package-versioning`: Defines typed package identities, versioned requirement syntax, bounded local resolution, and exact lock stability.

### Modified Capabilities

None.

## Impact

- Package behavior changes in `crates/chelis-reef`.
- The workspace adds `semver` with Serde support.
- Manifest schema 2 remains unsupported until bounded remote discovery adds its complete document boundary.
- Existing schema-1 manifests retain exact dependency behavior.
- Lockfiles continue to record exact versions, sources, compiler pins, and hashes.
- The Reef error owner must be stable before implementation starts. Do not implement its SNAFU slice in parallel.
- `docs/book/src/reef.md` and Reef design documents require local-resolver updates.
