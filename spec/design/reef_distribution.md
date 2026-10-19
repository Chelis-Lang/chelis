# Reef Distribution

**Status:** Current Phase A distribution contract, with historical implementation-plan
notes preserved for context. Phase A is complete in `spec/12-roadmap.md`: the shipped
surface includes `chelis reef install --from-github`, `--bootstrap`, default-on
auto-fetch during `chelis reef build`, lockfile `remote_origin`, bounded resolver-2 discovery,
`chelis reef update`, `chelis reef outdated`, and `chelis reef install --from-lockfile`.
The public registry server remains deferred.

Companion to `chelis_trust_stack.md` and `effect_taxonomy_expansion.md` (which adds
install-time effect manifests on top of the install path described here).

---

## Context

Reef is the Chelis package manager. It has a content-addressed local registry,
validated install paths, lockfiles with `(name, version, source-kind, compiler-pin,
archive_sha256, shell_sha256)` tuples, SHA256 verification on install, and remote
fetch through canonical GitHub release assets.

This document records the contract that closed the pre-launch distribution gap:
canonical shell artifacts are fetched from the canonical hosting org's release tags,
installed through the same validation path as local artifacts, and recorded with source
provenance so fresh checkouts can reproduce dependency state. The eventual public
registry server is post-launch and is recorded here for forward-compatibility but is
not designed in this round.

---

## Background

### Reef document schema ownership

`reef.toml` and `reef.lock` each carry an independent top-level `schema` string. Schema 1 is the first written version.

A missing schema identifies legacy schema 0. Ordinary reads keep the legacy parser and report the general upgrade command.

Reef parses the schema header before a complete document. Each current fixed table rejects unknown keys through its version-specific wire type.

`chelis reef upgrade --check|--inplace` owns all document migration. Target options select supported manifest and lock schemas independently.

The command preflights both selected migrations before replacement. It preserves accepted manifest comments and key order.

Each replacement uses a unique sibling, flushes the bytes, renames the sibling over the target, and syncs the parent where supported.

One persistent `.reef-write.lock` serializes project manifest, lock, and package archive replacement. Registry mutation keeps its separate registry lock.

The manifest is replaced before the lock. Every reader accepts a current manifest with a supported older lock.

A repeated upgrade completes an interrupted lock step without rewriting the current manifest. The two-file operation does not claim joint atomicity.

Versioned editor schemas under `docs/schemas/reef/` describe syntax only. Executable Reef parsing remains the document validation boundary.

The schema allocation sequence is manifest 1, resolver-2 manifest 2, and package-metadata manifest 3. The lock remains schema 1 until its fields change.

### Typed package identities and local candidates

Reef parses each package name, package version, compiler pin, lock identity, index identity, release identity, and shell identity before use.

Canonical package names use lowercase ASCII letters, digits, and internal hyphens. Complete package versions use Semantic Versioning without build metadata.

Manifest schema 1 retains complete exact dependency versions. Manifest schema 2 requires resolver 2 and uses Cargo-style requirements.

Candidates can come from an exact lock, the local registry, GitHub Releases, the bundled runtime, or a path.

Resolver 2 uses bytewise package order, lock preference, descending version precedence, bounded depth-first search, and failed-state memoization.

All modes limit candidates and resolved packages to 256. They limit dependencies to 256, depth to 128, and explored states to 100000.

One active path source overrides registry candidates for its package name. Reef rejects two active paths or conflicting verified non-path sources.

Discovery traversal rejects a cyclic path dependency graph with an ordered cycle report. A shared path package loads once for each canonical root.

For project resolution, a valid lock is the preferred exact graph. Reef reuses it without release listing after complete identity and integrity checks.

A requirement or source declaration change invalidates the preference. A locked hash failure or unavailable origin remains a hard error. Refresh and inspect check registry index entries against matching existing lock pins even if cached directories are missing, including packages outside a targeted refresh. They check fetched archives and shells against matching lock pins when index entries are absent, before selecting or reporting versions.

An uncached registry lock pin reserves one of the bounded candidate-manifest slots, ahead of release candidates ranked by requirements and version. A pin omitted from the bounded release listing, or one that cannot be verified, fails closed before refresh or inspect can select another version.

Remote discovery uses one private provider interface. `GitHubReleaseProvider` is the only implementation.

The production budget limits pages, requests, tags, manifests, request time, and downloaded bytes. Local resolver limits remain authoritative for graph search.

Candidate archives stay in command-owned temporary storage. Reef scans one root `reef.toml` without complete archive extraction.

A project refresh stages and verifies each selected pair. Then it gets the project lock before the registry lock.

Reef publishes complete package directories, replaces `index.json`, and replaces the project lock last. A late failure can leave complete unused cache entries.

### Package metadata and declared files

Manifest schema 3 owns optional descriptive package metadata. It accepts descriptions, SPDX licenses, HTTPS URLs, READMEs, and custom license files.

Descriptive metadata is not part of the logical `(name, version, source)` package identity or dependency selection. The fields are not serialized as fields into `reef.lock`, `index.json`, or `.chb`; byte-integrity hashes still reflect the complete source archive, including declared metadata and `reef.toml`.

Declared files use a bounded Unicode NFC path grammar with `/` separators. The grammar rejects unsafe components and platform-sensitive names.

On supported Unix platforms, Reef opens each component relative to the package-root handle. Each open rejects symbolic links.

Each declared file contains at most 4 MiB. Reef reads one open handle twice and rejects different byte sequences.

The source archive contains stable captured bytes for declared files. A bytewise member map preserves canonical order and deduplicates identical declared paths.

A metadata snapshot supplies member bytes when a source root selects the same path. Two distinct hard-link paths remain distinct archive members.

Reef completes all snapshots before final archive replacement. A snapshot failure preserves the prior archive.

### Source checking and library modules

`chelis check` accepts a named Surf module with an empty member list, both
inside and outside a Reef package. Module identity and imports pass
through the same package preparation and validation as other library modules.
Linking removes namespace forms from the executable entry declarations; that
empty linked list does not make the original source declaration-free.
Source emptiness is established before selecting the layered or monolithic
checker; disabling the standard-library cache preserves the checking result.

Source with no declarations, including whitespace and comments, remains an
input error. `chelis build` retains its rejection of an empty linked entry
declaration list. Property discovery on an empty library emits no property records.

### Shipped surface

- `chelis reef init` scaffolds a new package with the newest supported manifest schema.
- `chelis reef upgrade` checks or applies ordered manifest and lock schema migrations.
- `chelis reef update [<package>]` performs a full or targeted remote refresh.
- `chelis reef outdated [<package>] [--json]` reports remote state without final writes.
- `chelis reef build` resolves dependencies from the local registry, and auto-fetches
  missing dependencies from recorded or canonical GitHub release origins unless
  `--no-auto-fetch` is passed. It type-checks, lowers, and emits
  `dist/<name>-<version>.{chb,tar.zst}` plus a `reef.lock` recording the resolved
  dependency tuples.
- `chelis check <package-source>` and `chelis build <package-source>` use the same
  resolved package graph and repair a missing `reef.lock`, including when the prepared
  graph came from a warm cache. A malformed identity or integrity failure remains hard.
- `chelis check`, `build`, `cost`, and `prove` repair a missing lock only when the
  process can write the package root. A source in a read-only store is an example of a
  root that the process cannot write. For such a root, these commands resolve the graph
  in memory and write nothing into the root. `chelis eval --file` and `chelis test`
  always resolve a missing lock in memory. A resolution error stays hard in each case.
- `chelis reef publish` runs build, then copies artifacts into
  `$CHELIS_REEF_HOME/packages/<name>/<version>/` and updates
  `$CHELIS_REEF_HOME/index.json`.
- `chelis reef install --from-monorepo <path>` discovers packages under
  `<path>/packages/`, validates prebuilt artifacts (archive SHA256 match,
  shell readability, name/version agreement, embedded archive hash check),
  copies into the local registry, and updates the index.
- `chelis reef install --from-github <org>/<repo>@<tag>` fetches release assets from
  the GitHub API, validates the same artifact pair, records `remote_origin`, and
  updates the index.
- `chelis reef install --bootstrap [<org>/<repo>@<tag>...]` installs canonical shells
  in topological dependency order, rejecting `chelis-std` because it is the bundled
  runtime rather than a bootstrap target.
- `chelis reef install --from-lockfile` re-fetches lockfile dependencies from their
  recorded `remote_origin` and verifies bytes against the lockfile hashes.
- `chelis reef fetch-plan [--package-root <path>] [--platform <slug>]` prints the selected artifacts as JSON.
  It gives each public download URL and pinned SHA-256. It reads only `reef.lock` and makes no network request.
- `chelis reef install --from-lockfile --artifact-dir <dir>` imports the files that
  `fetch-plan` names from `<dir>`. It makes no network request. It verifies each file
  against the `reef.lock` hashes, and it records the locked `remote_origin`.
- `chelis reef verify-artifact --shell <path> --archive <path>` performs the
  install path's complete, read-only CHB/archive validation for release gates.

The validation step at install time is the same regardless of source. It
runs entirely on bytes already on disk; no code from the artifact executes.
CHB decoding consumes the complete input and re-encodes to the same canonical
bytes. CHB format 5 begins with the `CHELCHB\0` magic and an explicit little-
endian format version; predecessor and unknown-version layouts are rejected
rather than interpreted through bincode field coincidence. Each exported
function carries a canonical `type_variable_restrictions` ledger keyed to the
alpha-canonical type-variable identity in its printed type. CHB format 5 adds
a `collection_obligations` ledger carrying the checked relation of a
first-class `len`, `index`, `append`, `concat`, `key_from_seed`, `split_key`,
`split_keys`, or `fold_in` value as the rule name plus
its operand and result types in canonical Deep form under those same
identities. A relation variable absent from the printed callable type is
rejected rather than published as a hidden predicate. `reef schema` JSON uses
schema `format_version: 4` and the same two ledgers, so the machine-readable
authoring ABI and the installable CHB describe the same published checked
scheme identity. These publication surfaces are not compiler checker-reuse
inputs in this slice; serialized TypeEnv owns that path.
The ledger is derived from the checker Scheme by structural traversal across
separate type, dimension, and rank namespaces.
Its operation-value domains retain scalar-or-tensor operand restrictions
separately from the primitive-only authored dtype bounds (spec/04 §3.1).
The format bump rejects predecessor readers/writers rather than dropping
those checked requirements.

Decoding rejects appended bytes, truncation, malformed field encodings,
noncanonical metadata ordering, duplicate module/export/dependency entries,
invalid exact compiler pins, and invalid embedded SHA-256 text before checking
the paired archive digest. These checks cover fields the installer does not
otherwise consume (chelis#972).

### Locked artifact downloads and offline import

A tool that fetches files outside Reef, such as a Nix fixed-output derivation, uses
these two commands. Reef then owns the registry layout and `index.json`.

`fetch-plan` output is one JSON object with `format_version` 1, the root `package`
(`name`, `version`), and an `artifacts` array in lock order. Each element has these
fields:

- `name` and `version`: the locked package.
- `kind`: `archive`, `shell`, or `binary`.
- `file`: the release asset file name. The import reads `<dir>/<file>`.
- `url`: `https://github.com/<org>/<repo>/releases/download/<tag>/<file>`, from the
  `github://<org>/<repo>@<tag>` origin.
- `sha256`: the lowercase hex SHA-256 that `reef.lock` pins.
- `remote_origin`: the locked origin.
- `platform`: present only for `binary`, the selected build-platform slug.

A `local_registry` entry gives two elements. The `archive` file is
`<repo>-<version>.tar.zst` and the `shell` file is `<repo>-<version>.chb`, where
`<version>` is the tag without one leading `v`. A selected `binary` entry gives one element
whose `file` is its locked `asset`. `path` and `bundled` entries, and a legacy
`local_registry` entry for `chelis-std`, give no element. A `local_registry` entry
with no `remote_origin`, or with an origin that does not parse, makes the command
fail. A `binary` entry also makes the command fail when its `name` is not a bare file
name, or when its `asset` is not a bare file name of ASCII letters, digits, `.`, `-`,
`_`, or `~` (chelis#3891). The diagnostic names the lock entry. The public URL serves
public releases only.

A binary entry appears only when its platform matches the selected build platform.
The default platform is the compiler host.
`--platform <slug>` selects another build platform without a download or execution.

The import walks the entries of `reef.lock` as `--from-lockfile` does. For each
element it reads `<dir>/<file>` and compares the file SHA-256 with the lock before any
registry write. A missing file or a different hash fails that entry and changes
nothing for it.

A verified `local_registry` pair then passes the install validation and enters the
registry and `index.json` with the locked `remote_origin`. A verified binary for the
host platform goes to `<chelis_home>/bin/<name>`. A binary for another platform is
skipped, as with `--from-lockfile`. The command fails when any entry fails.

### Reproducible package artifact contract

For identical package inputs, compiler version, and `SOURCE_DATE_EPOCH`,
repeated `chelis reef build` invocations produce byte-identical
`<name>-<version>.tar.zst` and `<name>-<version>.chb` artifacts. Source archive
members are emitted in bytewise lexical order by their UTF-8 package-relative
paths. Every regular-file tar header has mode `0644`, uid `0`, gid `0`, and an
mtime equal to `SOURCE_DATE_EPOCH`; when the variable is absent, Reef uses the
fixed Unix epoch (`0`). A present value must be a non-negative integer number
of seconds or the build fails. Filesystem mtimes, ownership, permissions, and
directory enumeration order never enter the artifact.

Declared README and license snapshots join the same canonical member map. Reef does not discover undeclared metadata files.

The CHB continues to embed the SHA-256 of the resulting canonical source
archive. Changing `SOURCE_DATE_EPOCH` can therefore intentionally change both
artifacts; keeping it fixed (or absent) makes repeated builds reproducible.
The executable contract oracle is:

```sh
cargo test -p chelis-cli --test reef_build_reproducible
```

### Historical gap this doc closed

Before Phase A, Reef only installed from monorepo-built artifacts. New developers had
to clone shell repos and publish them into the local registry manually in dependency
order, builds could not recover from an empty registry, and lockfiles did not record
where installed bytes came from. Items 6-9 below are the shipped answer to that gap.

### Pre-launch backend choice

The pre-launch artifact backend is GitHub Releases on the canonical hosting
org's repositories. Each shell repository maintains its own release tags
(`v0.5.0` for Nautilus, `v0.5.0` for Coral, `v0.2.0` for Shoals,
`v0.4.0` for Octant — each shell's own semver trajectory; the version is
not synchronized across shells; default bootstrap resolves compatible versions
from the canonical repositories named by `DEFAULT_BOOTSTRAP_ROOTS` in
`crates/chelis-reef/src/lib.rs`). Each release
attaches the prebuilt `<name>-<version>.chb` and
`<name>-<version>.tar.zst` files (filenames omit the leading `v` even
though the tag carries it).

Public releases are fetched anonymously when no token is available. For private
repositories, `GITHUB_TOKEN` or `gh auth token` supplies a token with read
access. Authenticated requests also have a higher GitHub API rate limit.

The post-launch endgame is a registry server (Item 10). That is recorded
here as future work but is not designed in this round.

---

## Design

### Item 6 — `chelis reef install --from-github`

**Surface.**

```
chelis reef install --from-github <org>/<repo>@<tag>
```

`<tag>` is the shell's release tag, of the form `v<shell-version>` (for example,
`v0.5.0` for the current Nautilus release). This is the **shell's own version**, not
the chelis-compiler version; the two are independent. The shell's
`compiler = "=X.Y.Z"` pin in its `reef.toml` is the constraint that links
it to a chelis version.

Multiple `--from-github` flags are independent installs.

**Behavior.**

- Parse `<org>/<repo>@<tag>` into components.
- Resolve auth: read `GITHUB_TOKEN` from the environment; if unset, shell
  out to `gh auth token`. If neither yields a token, fetch anonymously.
  A private release that rejects an anonymous request reports the token
  needed to access it.
- Fetch the two release assets via the GitHub REST API in two steps:
  1. `GET https://api.github.com/repos/<org>/<repo>/releases/tags/<tag>`
     with `Accept: application/vnd.github+json` to look up the
     release's asset list. Find the entries whose `name` matches
     `<repo>-<ver>.tar.zst` and `<repo>-<ver>.chb`; capture each
     asset's numeric `id`.
  2. `GET https://api.github.com/repos/<org>/<repo>/releases/assets/<asset_id>`
     with `Accept: application/octet-stream` to stream the bytes to a
     temp directory. Send the auth header only when a token is available.

  GitHub's public-facing `/releases/download/<tag>/<asset>` URL form
  does not serve private-repo asset bytes even with a valid
  `Authorization: token …` header — it returns 404. The canonical
  API path also serves public releases anonymously. Test fixtures inject a localhost
  base URL via `CHELIS_REEF_GITHUB_BASE_API` (default
  `https://api.github.com`).
- Call the existing validation and placement logic, refactored out of
  `install_from_monorepo` into a shared
  `install_validated_artifact_pair(archive_path, shell_path, name, version,
  registry_root)` helper. Same SHA256 verification, same
  name/version-agreement check, same registry placement.
- Before placement, call the read-only `verify_artifact_pair` boundary used by
  `chelis reef verify-artifact`. A successful decode must consume and
  canonically validate the complete CHB envelope, not merely the package ID and
  embedded archive digest needed by installation (chelis#972).
- Lockfile records the fetched artifacts' content hashes. First fetch is
  trust-on-first-use against the bytes the canonical repo serves;
  subsequent fetches verify against the pinned hashes from the lockfile
  (same semantics as `--from-monorepo`).
- Clear error categories: auth failure (`AuthMissing` for no token,
  `AuthRejected` for 401/403 on either API endpoint), metadata-step 404
  (`ReleaseTagNotFoundOrUnauthorized` naming the tag URL; this 404 is
  ambiguous because GitHub returns 404 both for a missing tag and for a
  private repo the token cannot read, so the message names both
  possibilities and points at `gh release view <tag> --repo
  <org>/<repo>` for verification, per issue #147), asset-step 404
  (`ReleaseAssetNotFound`, which fires only after metadata fetch
  succeeded, so the repo is reachable and the tag exists; it names the
  expected asset name plus the assets actually present on the release),
  rate-limit (`RateLimited` carrying `Retry-After`), 5xx
  (`ServerError`), network/DNS (`Network`), hash mismatch on subsequent
  fetch (`Validation` from the shared helper; do not auto-overwrite),
  I/O error (`Io`).

**Acceptance oracle.**

- `chelis reef install --from-github <org>/nautilus@v0.5.0` with
  `GITHUB_TOKEN` set in env successfully fetches and installs Nautilus.
  The local registry afterward has `packages/nautilus/0.5.0/` populated
  identically to the `--from-monorepo` outcome.
- Without a token, a public release installs and a private release reports
  the token needed to access it.
- With a tampered local cache (alter the archive's hash), subsequent
  install fails with hash-mismatch error.
- The MCP and HTTP surfaces of Tide are unaffected.

**Scope.** Approximately 180 lines of new code across two files:
- `crates/chelis-reef/src/lib.rs`: `install_validated_artifact_pair`
  is the shared validation+placement helper (currently around line
  892); it is invoked by both `install_from_monorepo` (line ~1000) and
  `install_from_github` (line ~1528). Line numbers may drift; the
  helper's name is the stable reference.
- `crates/chelis-cli/src/main.rs`: `--from-github` handler that parses
  the repo+tag and calls `install_from_github`. The CLI handler stays
  thin — the HTTP layer lives in `chelis-reef` per the locked layering
  decision in this document. ~100 lines.
- `crates/chelis-reef/Cargo.toml` adds
  `reqwest = { version = "0.11", default-features = false, features = ["blocking", "rustls-tls", "rustls-tls-native-roots"] }`.

---

### Item 7 — Bootstrap: `chelis reef install --bootstrap`

**Surface.**

```
chelis reef install --bootstrap <list>
```

With explicit `org/repo@tag` entries, bootstrap installs only that closed
input graph through Item 6's per-shell install path. Without entries, it
resolves the canonical default roots and their complete dependency closure.
Both forms report shells in deterministic dependency order.

**Behavior.**

- For explicit selectors, parse entries into a graph keyed by `(name, version)` with edges
  derived from the shell's published `reef.toml` dependencies. The
  dependency edges are read from each shell's `<name>-<version>.tar.zst`
  archive after fetch but before placement. Each dependency requirement uses
  the manifest's typed resolver semantics and must match the selected version
  of its named peer in the explicit input set. A missing peer or incompatible
  selected version fails before installation; bootstrap does not discover
  additional peers.
- Topologically sort. If a cycle is detected, fail with a clear error
  naming the cycle.
- Publish explicit shells in order through Item 6's verified pair installer.
  Default discovery publishes the selected verified remote material through
  the shared registry publisher.
- The versionless default roots are `nautilus`, `coral`, and `shoals`.
  Default mode uses the existing bounded typed remote resolver over canonical
  GitHub release observations, including transitive dependencies. Every
  selected shell pins the running compiler exactly; runtime requirements must
  match the compiler-bundled runtime. Default mode does not substitute a
  compiler or runtime, including when dependency compiler drift is allowed
  for project development.
- Default discovery uses the existing provider, resolver ranking, search and
  download limits, and artifact-pair verification. Registry state is verified
  cache/publication state; it does not choose candidate versions or origins.
  Selected entries record the observed canonical origins even when identical
  bytes already exist in the cache under a different recorded origin.
  Fixed observed remote inputs produce fixed selected identities, hashes and
  dependency order. Public releases work anonymously; restricted releases
  produce actionable authentication errors.
- Default resolution starts from a typed in-memory seed. It creates no project
  manifest, project lockfile, or other working-directory state. The complete
  selected graph and pairs are validated before registry publication, which
  reuses the existing registry lock and hash/conflict checks. A late failure
  may leave complete unused cache entries; the operation does not promise a
  joint filesystem transaction.
- A compatible published graph need not exist immediately after a compiler
  release. Absence within the observed bounded inputs produces a typed error
  naming the compiler and default roots. The same compiler can resolve a
  cohort published later without an embedded tag-table update. Historical
  releases outside the shared discovery bounds are not promised.
- Explicit selectors never enter default discovery or expand their input set.

**chelis-std is the language runtime, not a shell.** The runtime is
distributed bundled with the compiler — it version-marches with the
toolchain and cannot be substituted independently. Programs depend on
it the same way Rust programs depend on `core`/`std`. Concretely:

- `chelis-std` is **never** in the bootstrap input list. An explicit
  `chelis-std` entry in `--bootstrap` arguments is rejected with a
  typed `BootstrapError::RuntimeNotABootstrapTarget` error naming
  both the requested and the bundled version.
- A shell's `reef.toml` may declare `chelis-std = { version = "X" }`.
  The bootstrap installer checks `X` against the compiler's bundled
  runtime version using the manifest's typed dependency requirement:
  schema 1 uses an exact package version; schemas 2 and 3 use resolver-2
  requirements, including `=X.Y.Z`. Invalid requirements fail manifest
  validation. On match the dep is filtered from the
  bootstrap graph (it is implicit, not an edge in the install loop);
  on mismatch the bootstrap aborts with a typed validation error
  naming both versions.
- Lockfile entries for chelis-std use `LockSource::Bundled
  { compiler_version }`, recording the version of the compiler that
  supplied the bytes for auditability. There is no archive to fetch
  and integrity comes from the compiler binary itself.

**Bundling and lockfile synthesis (Phase A correction).** The runtime
bytes, both the source archive (`.tar.zst`) and the shell (`.chb`), are
baked into each chelis binary. The `chelis-std-bundle` crate packs them
while the binary builds and exposes them as an `EmbeddedRuntime`, which
every reef graph entry point takes as an explicit parameter. The reef
loader serves chelis-std from that runtime; everything else falls through
to the local-registry path. Three consequences:

1. **Project-driven blanket synthesis.** Every reef.toml has a
   `compiler =` pin, and that pin IS the runtime declaration.
   `build_lockfile` therefore unconditionally records a
   `LockSource::Bundled` chelis-std entry for every project,
   regardless of whether the project listed chelis-std in
   `[dependencies]`. The synthesis is idempotent with the explicit-
   listing path: when chelis-std is in the dep graph, the same
   `Bundled` entry is produced; when absent, the lockfile-build step
   appends it after iterating the graph. The `archive_sha256` and
   `shell_sha256` fields come from the embedded runtime for both paths
   so the recorded entry is byte-identical. The graph-loading commands
   (`chelis check`, `build`, `eval`, `test`, and `reef build`) reuse a
   lock's chelis-std entry only when its version, its `Bundled` source
   with this compiler's version, and both hashes name the embedded
   runtime. For them any difference, a std rebuild that changes only the
   hashes included, makes the lock stale, and reef resolves the graph
   again as it does after a compiler upgrade. `reef export-bundle` and
   `reef install --from-lockfile` read the lock without that assessment
   (chelis#3025).
2. **Build-time packing, nothing committed.** The bundle crate's build
   script stages `packages/chelis-std/reef.toml`, the `.ch` files under
   its declared source roots, and its declared metadata files, then packs
   them with `pack_runtime_package`, the packing step `chelis reef build`
   runs. Every archive member's mtime is 0 whatever `SOURCE_DATE_EPOCH`
   says, so the embedded pair is a function of those inputs and the
   packing code, and no other file in the source tree enters it.
   `chelis reef build packages/chelis-std` writes the same pair from a
   copy of the package holding only those inputs, with `SOURCE_DATE_EPOCH`
   unset; it archives every file under `src/` and honours
   `SOURCE_DATE_EPOCH`, so a stray file or a set epoch changes its output.
   The std version is the one `packages/chelis-std/reef.toml` names. `chelis-reef`
   does not depend on the bundle: the bundle build-depends on reef, and
   only binaries and test harnesses depend on the bundle, so a library
   that forgets to pass the runtime fails to compile. No dist pair or lock
   recording the bundled runtime is committed:
   `scripts/check_std_bundle_untracked.py` refuses, in the gate and in CI,
   a tracked file under either `dist/` directory and a tracked lock
   anywhere that records the bundled runtime, and the pre-commit hook
   runs it on a commit that stages a file under either `dist/` directory
   or `packages/chelis-std/reef.lock`.
   `scripts/check_std_bundle_reproducible.py` requires two runs of the
   bundle's build script, in separate processes and output directories,
   to pack the same bytes.
3. **No registry seeding required.** `chelis reef build` against a
   project that depends on chelis-std (implicitly or explicitly)
   succeeds against an empty `$CHELIS_REEF_HOME`. The bundled bytes
   are reachable from the chelis binary, not from the filesystem.
   Manual gates that previously pre-installed chelis-std via
   `--from-monorepo` no longer need that step.

**Acceptance oracle.**

- `chelis reef install --bootstrap` resolves the three default roots and
  their complete compatible published closure against a fresh registry.
  Verify the exact compiler, runtime authority, origins, hashes and dependency
  order. Public localhost fixtures require neither a token nor `gh`.
- With one compiler binary, an observed snapshot lacking a compatible cohort
  fails without installation; a later snapshot containing that cohort succeeds.
  Joint root constraints and release-enumeration permutations retain the
  shared resolver's deterministic graph selection.
- Cache versions and recorded fork origins do not change default selection.
  Incompatible, cyclic, corrupt or out-of-budget inputs retain typed failures.
- The real-network manual gate requires an available compatible published
  cohort; it is not a pre-publication compiler release prerequisite.
- Explicit list form `chelis reef install --bootstrap
  <org>/nautilus@v0.5.0 <org>/coral@v0.5.0` installs only the two
  named shells in topological order.

**Scope.** ~50 lines on top of Item 6. The graph build is small (each
shell has a handful of dependencies); the topological sort is a textbook
post-order traversal; the install step is a loop over Item 6's helper.

---

### Item 8 — Auto-fetch during `chelis reef build`

**Surface.** Default-on; opt out with `--no-auto-fetch`.

**Behavior.**

- The missing-from-registry error site at
  `crates/chelis-reef/src/lib.rs:1243-1245` currently bails with
  "missing from local registry index — run `chelis reef build` first to
  populate the cache". The message is a workaround for the absence of
  remote fetch.
- When auto-fetch is enabled (the default), insert a fetch attempt
  before bailing. The fetch source is derived from the lockfile's
  `remote_origin` field (Item 9) if present; otherwise from the
  canonical hosting-org default for the named package (e.g.
  `<canonical-org>/<name>@v<version>`).
- On fetch success, retry the registry lookup. On fetch failure, bail
  with a clearer error than the current one: it now names the URL that
  was tried and the failure category (auth, 404, hash mismatch).
- `--no-auto-fetch` disables this fallback for users who want explicit
  control of when network access happens during build.

**Acceptance oracle.**

- A clean dev environment with `GITHUB_TOKEN` set runs `chelis reef
  build` on a project that depends on Nautilus (no manual install
  step), and the build succeeds with an auto-fetch in the middle.
- Same project under `--no-auto-fetch` fails with the existing error
  shape but improved wording.
- A project whose lockfile pins Nautilus to a specific
  `remote_origin` and whose registry is empty fetches from that
  origin under auto-fetch.

**Scope.** Roughly one day of work. The fetch path itself is Item 6's
helper; the work is wiring it into the build error site, propagating
the `--no-auto-fetch` flag through the build command, and refining the
error catalog.

**Canary/dev escape hatch — `CHELIS_REEF_ALLOW_DEP_COMPILER_DRIFT`.** When
this env is set to a non-empty, non-`0` value, `validate_manifest` waives
*only* the `package.compiler` equality assertion — every other manifest
check stays enforced — and emits a loud per-package warning naming the
package and both pins. Its sole purpose is to let the ecosystem drift
canary (and local dev) build a shell's *code* against chelis HEAD while its
released dependencies still pin the compiler they were cut against: the
canary asks "does this code still compile against HEAD", whereas a
dependency's pin freshness is that shell's own release-cadence concern, not
a code-drift signal. It is **not** for a shell's own CI gate, where the pin
equality is the whole point of the check; leave it unset there.

---

### Item 9 — Lockfile records remote origin

**Surface.** Lockfile gains an optional field on each entry:

```toml
[[dependencies]]
name = "nautilus"
version = "0.5.0"
compiler = "=<compiler-version>"
archive_sha256 = "..."
shell_sha256 = "..."

[dependencies.source]
kind = "local_registry"
remote_origin = "github://<canonical-org>/nautilus@v0.5.0"
```

The `remote_origin` field is optional for backward compatibility;
older lockfiles without it continue to deserialize. New lockfiles
populate it whenever the install came from a remote source.

**New command:** `chelis reef install --from-lockfile`. Reads the
project's lockfile, re-fetches every dependency from its recorded
remote origin, and validates against the lockfile's pinned hashes.
This gives a one-command "match the lockfile" path for fresh
checkouts.

**Behavior.**

- Lockfile serialization gains the optional field. Deserialization is
  forward-compatible (old lockfiles deserialize fine; new ones round-
  trip through cargo-style serde with the field present or absent).
- `chelis reef install --from-lockfile` walks the lockfile, calls
  Item 6's helper for each entry that has a `remote_origin`, and
  errors clearly on entries that lack one (suggesting a manual
  `--bootstrap` to populate them).

**Acceptance oracle.**

- One developer fetches a set of dependencies via `--from-github`
  and commits the resulting `reef.lock`.
- A second developer clones the repo and runs `chelis reef install
  --from-lockfile`; ends with the same local registry state, all
  hashes verifying against the lockfile pins.
- The same operation from a stale registry (different bytes already
  installed under the same name/version) fails with hash-mismatch.

**Scope.** Roughly one day. Field addition is small; the install-from-
lockfile command is a thin wrapper over Item 6's helper.

---

### Item 10 — Public registry server (deferred)

Post-launch endgame. A dedicated registry service can replace GitHub
Releases as the artifact backend. It adds multiple publishers and registry discovery around the typed resolver. Substantial product
work; do not start without a specific driver pulling for it. Recorded
here for forward-compatibility so the design of Items 6-9 stays
compatible with a future migration (the lockfile's `remote_origin`
already records source provenance in a string form that can route to
either a GitHub release or a registry URL).

---

### Item 11 — First-class binary distribution (chelis#468)

**Status:** planned. Tracked by chelis#468; framed as Layer 1 of
[`chelis_packaging_and_install.md`](chelis_packaging_and_install.md).

Items 6-9 deliver and verify *source* packages (a `.tar.zst` archive plus a
`.chb` shell). They cannot carry a **compiled binary**. Every binary-bearing
component in the ecosystem — the toolchain, nautilus, shoals, and octant's
translator — therefore distributes its binary out of band through a GitHub
release tarball, and every consumer hand-rolls download-by-tag plus SHA
verification (octant's `docs/src/consuming.md`; downstream verify-in-Dockerfile
steps). reef does the source half; an informal channel does the binary half, and
the integrity work reef would own is reimplemented per consumer. Item 11 folds
that verb into the resolver and the lockfile.

**Manifest.** A dedicated `[artifacts]` section — binaries are neither reef
packages nor Cargo crates, so they get their own artifact-class section, the
same shape chelis#571 established with `[chelis-src]`:

```toml
[artifacts.octant-translator]
repo = "Chelis-Lang/octant"          # publisher; sensible default
tag  = "v0.4.2"
platforms.linux-x86_64 = { asset = "octant-translator-linux-x86_64.tar.gz", sha256 = "…" }
platforms.darwin-arm64 = { asset = "octant-translator-darwin-arm64.tar.gz",  sha256 = "…" }
```

`linux-x86_64` is the minimum; additional platforms are additive. The
publisher's release SHA is the source of truth — the consumer pins and verifies
against it rather than transcribing a copy.

**Install flow.** `chelis reef install` (and `reef setup`) resolves the **host
platform** entry, downloads the asset through the existing GitHub REST path
(Item 6's `install_from_github`), **SHA-256-verifies fail-closed** — a mismatch
aborts before any placement — extracts, and places a runnable binary at
`~/.chelis/bin/<name>`. A new `chelis reef which <artifact>` prints the resolved
path so consumers point at it with no out-of-band knowledge.

**Lockfile.** Schema 1 retains the `[[dependencies]]` array and all current fields.

Each binary identity uses `(name, source.platform)`.
A writer records every declared platform, independent of the writer host.

Each entry retains the version, compiler pin, origin, asset, and SHA-256.

The `source.sha256` and `archive_sha256` pins must agree.
`shell_sha256` must be empty.

Reef rejects duplicate binary identities before a fetch or an install.
Identical manifest declarations share one entry.
Conflicting declarations for the same identity fail.

The normal install and the offline import select only the compiler host entry.
Other platform entries remain in the lock.

Reef maps macOS arm64 to `darwin-arm64` and macOS x86_64 to `darwin-x86_64`.
It maps Linux x86_64 to `linux-x86_64` and Linux arm64 to `linux-arm64`.
These artifact platforms do not depend on compiler-release availability.
The lock fields do not change, so the lock schema remains 1.

**Reuse.** `install_validated_artifact_pair` (validation + placement); the
shipped SHA-256 helpers; `install_from_github` (REST asset fetch);
`try_github_token` (private-repo auth, added in chelis#571); the Item 9
`remote_origin` lockfile pattern; the `cmd_reef_*` dispatch. The one genuinely
new mechanism is host-platform / target-triple selection.

**Acceptance:**

- `reef.toml` expresses a per-platform, SHA-pinned binary artifact.
- `reef install` on a supported platform downloads, SHA-verifies fail-closed,
  and installs a runnable binary, with no consumer-side hand-rolled download or
  SHA literal.
- The binary dependency and its SHA are recorded in the lockfile.
- A hybrid package (octant) declares both its source shell and its binary, and a
  consumer obtains both through one dependency set.
- octant's `consuming.md` workaround and downstream download-and-verify
  Dockerfile steps retire in favor of the reef path.

**Toolchain note.** The chelis toolchain itself is binary-bearing, but its
installer of record is `chelisup` (chelis#164), not `reef install` — see
[`chelis_packaging_and_install.md`](chelis_packaging_and_install.md) §5.6. Item
11 may *describe* the toolchain as an `[artifacts]` entry for reproducibility,
but does not install it.

---

## Acceptance summary

Per item:

- Item 6: install single shell from canonical org's release; lockfile
  records hashes; subsequent fetches verify.
- Item 7: default mode installs the three canonical roots and their compatible
  closure; explicit mode installs its closed selected graph in dependency order.
- Item 8: a fresh `chelis reef build` succeeds end-to-end without
  manual install steps.
- Item 9: a developer can re-create another developer's local
  registry state from the committed lockfile alone.
- Item 11 (planned): a per-platform SHA-pinned binary is declared once,
  downloaded + verified fail-closed, placed runnable, and lock-recorded;
  a consumer obtains a hybrid package's source shell and binary through one
  dependency set, retiring the out-of-band download-and-verify channel.

With Items 6-9 shipped, a fresh dev environment's onboarding is
`git clone <project> && chelis reef build` for public dependencies. Private
dependencies require a token. Item 11 extends this to binary
dependencies; the full cross-class onboarding becomes `chelis reef setup`
([`chelis_packaging_and_install.md`](chelis_packaging_and_install.md) §7).

---

## Historical Sequencing

Item 6 had to land first; Items 7, 8, 9 each depended on Item 6's helper.
This section is retained as implementation history, not a remaining schedule.

Item 10 is demand-driven and not part of this design's
implementation scope.

The original Item 6-9 estimate was 3-4 days of focused work. It is retained only to
explain the scope of the shipped Phase A change set.

---

## Reused machinery

- `install_validated_artifact_pair` is the shared validation+placement
  helper (currently in `crates/chelis-reef/src/lib.rs` around line
  892; line numbers drift, the function name is the stable
  reference). Item 6 extracted it from `install_from_monorepo` so
  both `--from-monorepo` and `--from-github` call into it. Both
  produce byte-identical local registry state for the same
  `(name, version)` — locked as a contract invariant by the byte-
  equality sub-case of the named oracle.
- `verify_artifact_pair` is the read-only complete-envelope and archive-digest
  boundary. The CLI exposes it as `chelis reef verify-artifact`; installation
  calls it before creating package directories or updating `index.json`
  (chelis#972).
- SHA256 verification logic already shipped (computes
  `archive_sha256` and `shell_sha256` per
  `LockedDependency` at `lib.rs:64-71`).
- Lockfile schema (cargo-style toml serde). Item 9 added an optional field that
  round-trips via existing serde paths.
- `crates/chelis-cli/src/main.rs::cmd_reef_*` handlers as the
  command-dispatch entry points.
- The error site at `lib.rs:1243-1245` for Item 8's insertion
  point.

---

## Out of scope

- **Semver / version-range resolution.** Today's reef is exact-pin
  only. Version ranges are a real feature for a public ecosystem
  but not needed for a small private team using exact pins. Defer
  to public-launch alongside Item 10.
- **Mirror configuration / private registries.** Users may want to
  host their own mirrors. Post-launch concern.
- **Multi-publisher support.** Today's hardcoded canonical-org
  default works for the team. Multi-publisher comes with Item 10.
- **Authenticity beyond content hashing.** Cryptographic signing
  of artifacts (publisher key, signature verification on install)
  is tracked separately under the trust-stack expansion's Item 5
  and is demand-driven; not in this round.
- **Cross-machine reproducibility certification.** The build contract above
  canonicalizes package inputs, archive metadata, and the epoch, and its
  executable oracle compares repeated builds on one runner. A release matrix
  that compares independent rebuilds across operating systems, filesystems,
  and compression-library implementations is not yet present. Until that
  matrix exists, claim the deterministic build contract and same-runner byte
  oracle, not empirical cross-machine certification.

---

## Cross-references

- `effect_taxonomy_expansion.md` Item 4 (install-time effect
  manifests) is the natural composition point: the install path
  designed here gains a `--print-effects` and `--refuse` flag once
  the effect aggregation is available. The two designs share the
  install boundary.
- `chelis_trust_stack.md` references this doc as the Phase A distribution surface.
- `spec/12-roadmap.md` Phase A (distribution unblock) covers
  Items 6-9 of this doc.
