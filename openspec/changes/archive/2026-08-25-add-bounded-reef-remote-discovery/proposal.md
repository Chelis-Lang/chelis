## Why

Cargo-style requirements need remote version discovery when no exact lock supplies a complete graph. The current SemVer proposal combines that discovery with local parsing and resolution.

Unbounded release pagination, archive inspection, and recursive search create network and resource risks. Local candidates can also suppress remote checks during an explicit update.

## What Changes

- Add explicit locked, local, refresh, and read-only discovery modes.
- Make normal commands reuse a valid lock without a version query.
- Make `chelis reef update` query remote providers even when a local graph is complete.
- Add `chelis reef outdated [<package>] [--json]` without final registry or lock writes.
- Add a provider-neutral candidate interface with one GitHub Releases implementation.
- Keep current and future descriptive manifest URLs outside source selection.
- Inherit local graph limits and add finite limits for pagination, requests, candidate inspection, request time, and downloaded bytes.
- Inspect remote `reef.toml` entries without unrestricted archive extraction.
- Keep candidate bytes in command-owned temporary storage until selection completes.
- Publish only complete verified package entries into the append-only registry cache.
- Acquire the project lock before the registry lock for every final update commit.
- Replace `reef.lock` atomically after selected cache entries are complete.
- Permit unused complete cache entries after a crash or failed final lock replacement.
- Preserve exact locked-origin behavior and fail closed on integrity errors.
- Activate resolver 2 generation and its schema-1 to schema-2 migration only after this change passes its oracle.

Out of scope:

- A public registry service, mirrors, multiple publishers, or source patches.
- User-provided freshness commands or package-manifest command execution.
- Multiple resolved versions of one package name.
- A change to GitHub authentication or direct exact-tag installation.
- A joint atomic transaction across the project lock and the home registry.
- Compiler, runtime, backend, language, or generated-code semantics.

## Capabilities

### New Capabilities

- `reef-remote-discovery`: Defines bounded provider discovery, explicit refresh modes, safe candidate inspection, and recoverable cache publication.

### Modified Capabilities

None.

## Impact

- Remote package code changes in `crates/chelis-reef`.
- Reef update and inspection commands change in `crates/chelis-cli`.
- The change depends on `version-reef-manifest-schema` and the local resolver from `adopt-semver-for-reef`.
- Resolver 2 remains inactive for generated and migrated manifests until this change passes its oracle.
- The local registry can retain unused verified entries. A later garbage-collection change can remove them safely.
- `docs/book/src/reef.md` and the Reef distribution designs require updates.
