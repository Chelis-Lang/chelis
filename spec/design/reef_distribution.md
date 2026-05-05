# Reef Distribution

**Status:** Planning. Source-of-truth design for making reef end-to-end usable
for the dev team during the pre-launch era. Companion to `chelis_trust_stack.md`
and `effect_taxonomy_expansion.md` (which adds install-time effect manifests
on top of the install path designed here).

---

## Context

Reef is the Chelis package manager. It already has the foundational pieces:
content-addressed local registry, validated install path, lockfile with
`(name, version, source-kind, compiler-pin, archive_sha256, shell_sha256)`
tuples, and SHA256 verification on install. What it does not have is a
remote-fetch path. Today the only install source is `--from-monorepo`, which
walks `<chelis-monorepo>/packages/` and discovers prebuilt artifacts there.
Shells live in separate repositories outside that directory; consequently a
new dev needs to clone four shell repos and manually publish them into the
local registry in dependency order before `chelis reef build` will succeed
in any downstream project.

This design closes that gap by adding remote fetch from the canonical
hosting org's release tags as the pre-launch artifact backend. The eventual
public registry server is post-launch and is recorded here for
forward-compatibility but explicitly not designed in this round.

---

## Background

### What works today

- `chelis reef init` scaffolds a new package.
- `chelis reef build` resolves dependencies (from the local registry only),
  type-checks, lowers, and emits `dist/<name>-<version>.{chb,tar.zst}` plus
  a `reef.lock` recording the resolved dependency tuples.
- `chelis reef publish` runs build, then copies artifacts into
  `$CHELIS_REEF_HOME/packages/<name>/<version>/` and updates
  `$CHELIS_REEF_HOME/index.json`.
- `chelis reef install --from-monorepo <path>` discovers packages under
  `<path>/packages/`, validates prebuilt artifacts (archive SHA256 match,
  shell readability, name/version agreement, embedded archive hash check),
  copies into the local registry, and updates the index.

The validation step at install time is the same regardless of source. It
runs entirely on bytes already on disk; no code from the artifact executes.

### What is missing

- Remote fetch. There is no path that downloads bytes from a URL and feeds
  them into the validation+placement step.
- Topological dependency-ordered install. Today a user must know to install
  nautilus first, then coral (which depends on nautilus), etc., manually.
  (chelis-std is the bundled runtime and is never installed via reef.)
- Auto-fetch during `chelis reef build`. The build errors with "missing
  from local registry index — run `chelis reef build` first to populate
  the cache" at `crates/chelis-reef/src/lib.rs:1243-1245` if a dependency
  is not pre-installed.
- Lockfile origin records. Today the lockfile records `kind:
  LocalRegistry` for any installed dependency, which is correct as a
  description of where the bytes currently sit but loses the information
  about where they came from. A second developer cloning the repo cannot
  reproduce the install state without out-of-band knowledge of which
  remote source to fetch from.

### Pre-launch backend choice

The pre-launch artifact backend is GitHub Releases on the canonical hosting
org's repositories. Each shell repository maintains its own release tags
(`v0.4.0` for Nautilus, `v0.4.0` for Coral, `v0.1.0` for Shoals,
`v0.1.0` for Octant — each shell's own semver trajectory; the version is
not synchronized across shells). Each release attaches the prebuilt
`<name>-<version>.chb` and `<name>-<version>.tar.zst` files.

The dev team has authenticated access to the canonical org. Authentication
is via standard GitHub PATs supplied through the `GITHUB_TOKEN` environment
variable, with an unauthenticated-fallback path that does not exist today
(the canonical repos are private; unauthenticated fetch would simply 404).

The post-launch endgame is a registry server (Item 10). That is recorded
here as future work but is not designed in this round.

---

## Design

### Item 6 — `chelis reef install --from-github`

**Surface.**

```
chelis reef install --from-github <org>/<repo>@<tag>
```

`<tag>` is the shell's release tag, of the form `v<shell-version>` (e.g.
`v0.4.0`, `v0.1.0`). This is the **shell's own version**, not the
chelis-compiler version; the two are independent. The shell's
`compiler = "=X.Y.Z"` pin in its `reef.toml` is the constraint that links
it to a chelis version.

Multiple `--from-github` flags are independent installs.

**Behavior.**

- Parse `<org>/<repo>@<tag>` into components.
- Resolve auth: read `GITHUB_TOKEN` from the environment; if unset, shell
  out to `gh auth token` to obtain one. Hard-fail with a clear error if
  neither yields a token, suggesting `export GITHUB_TOKEN=$(gh auth token)`
  as the fix. Authentication is mandatory because the canonical repos are
  private.
- Fetch the two release assets via the GitHub REST API in two steps:
  1. `GET https://api.github.com/repos/<org>/<repo>/releases/tags/<tag>`
     with `Accept: application/vnd.github+json` to look up the
     release's asset list. Find the entries whose `name` matches
     `<repo>-<ver>.tar.zst` and `<repo>-<ver>.chb`; capture each
     asset's numeric `id`.
  2. `GET https://api.github.com/repos/<org>/<repo>/releases/assets/<asset_id>`
     with `Accept: application/octet-stream` and the auth header to
     stream the bytes to a temp directory.

  GitHub's public-facing `/releases/download/<tag>/<asset>` URL form
  does not serve private-repo asset bytes even with a valid
  `Authorization: token …` header — it returns 404. The canonical
  chelis-lang shells are private during the pre-launch era; the API
  path is required, not optional. Test fixtures inject a localhost
  base URL via `CHELIS_REEF_GITHUB_BASE_API` (default
  `https://api.github.com`).
- Call the existing validation and placement logic, refactored out of
  `install_from_monorepo` into a shared
  `install_validated_artifact_pair(archive_path, shell_path, name, version,
  registry_root)` helper. Same SHA256 verification, same
  name/version-agreement check, same registry placement.
- Lockfile records the fetched artifacts' content hashes. First fetch is
  trust-on-first-use against the bytes the canonical repo serves;
  subsequent fetches verify against the pinned hashes from the lockfile
  (same semantics as `--from-monorepo`).
- Clear error categories: auth failure (`AuthMissing` for no token,
  `AuthRejected` for 401/403 on either API endpoint), asset 404
  (`ReleaseAssetNotFound` — split into "metadata 404" naming the tag
  URL and "asset-list mismatch" naming the expected asset name plus
  the assets actually present on the release), rate-limit
  (`RateLimited` carrying `Retry-After`), 5xx (`ServerError`),
  network/DNS (`Network`), hash mismatch on subsequent fetch
  (`Validation` from the shared helper; do not auto-overwrite), I/O
  error (`Io`).

**Acceptance oracle.**

- `chelis reef install --from-github <org>/nautilus@v0.4.0` with
  `GITHUB_TOKEN` set in env successfully fetches and installs Nautilus.
  The local registry afterward has `packages/nautilus/0.4.0/` populated
  identically to the `--from-monorepo` outcome.
- Without the token, fails with a clear error pointing at the env var.
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
  thin — the HTTP layer lives in `chelis-reef` per the locked
  decision in `phaseA_reef_distribution.md`. ~100 lines.
- `crates/chelis-reef/Cargo.toml` adds
  `reqwest = { version = "0.11", default-features = false, features = ["blocking", "rustls-tls"] }`.

---

### Item 7 — Bootstrap: `chelis reef install --bootstrap`

**Surface.**

```
chelis reef install --bootstrap <list>
```

Where `<list>` is a sequence of `org/repo@tag` entries (or a config-file
reference, or the built-in default list of canonical shells). Installs
them in topological dependency order, using Item 6's per-shell install
path under the hood.

**Behavior.**

- Parse the entries into a graph keyed by `(name, version)` with edges
  derived from the shell's published `reef.toml` dependencies. The
  dependency edges are read from each shell's `<name>-<version>.tar.zst`
  archive after fetch but before placement.
- Topologically sort. If a cycle is detected, fail with a clear error
  naming the cycle.
- Install each shell in order via Item 6's `install_validated_artifact_pair`
  + lockfile update.
- The built-in default list covers the canonical shells: `nautilus`,
  `coral`, `shoals`, `octant`. This list is hard-coded for the
  pre-launch dev team; multi-publisher generalization is post-launch.

**chelis-std is the language runtime, not a shell.** The runtime is
distributed bundled with the compiler — it version-marches with the
toolchain and cannot be substituted independently. Programs depend on
it the same way Rust programs depend on `core`/`std`. Concretely:

- `chelis-std` is **never** in the bootstrap input list. An explicit
  `chelis-std` entry in `--bootstrap` arguments is rejected with a
  typed `BootstrapError::RuntimeNotABootstrapTarget` error naming
  both the requested and the bundled version.
- A shell's `reef.toml` may declare `chelis-std = { version = "X" }`.
  The bootstrap installer soft-verifies `X` against the compiler's
  bundled runtime version: on match the dep is filtered from the
  bootstrap graph (it is implicit, not an edge in the install loop);
  on mismatch the bootstrap aborts with a typed validation error
  naming both versions.
- Lockfile entries for chelis-std use `LockSource::Bundled
  { compiler_version }`, recording the version of the compiler that
  supplied the bytes for auditability. There is no archive to fetch
  and integrity comes from the compiler binary itself.

**Acceptance oracle.**

- A clean dev environment with `GITHUB_TOKEN` set runs `chelis reef
  install --bootstrap` (no arguments — uses the default list) and ends
  with all four canonical shells (`nautilus`, `coral`, `shoals`,
  `octant`) installed in the local registry, in the correct order so
  each shell's dependencies were already present when it was
  installed.
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

---

### Item 9 — Lockfile records remote origin

**Surface.** Lockfile gains an optional field on each entry:

```toml
[[dependencies]]
name = "nautilus"
version = "0.4.0"
compiler = "=0.4.0"
archive_sha256 = "..."
shell_sha256 = "..."

[dependencies.source]
kind = "local_registry"
remote_origin = "github://<canonical-org>/nautilus@v0.4.0"
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

Post-launch endgame. A dedicated registry service replaces GitHub
Releases as the artifact backend. Adds version search, semver
resolution, multiple publishers, discoverability. Substantial product
work; do not start without a specific driver pulling for it. Recorded
here for forward-compatibility so the design of Items 6-9 stays
compatible with a future migration (the lockfile's `remote_origin`
already records source provenance in a string form that can route to
either a GitHub release or a registry URL).

---

## Acceptance summary

Per item:

- Item 6: install single shell from canonical org's release; lockfile
  records hashes; subsequent fetches verify.
- Item 7: install all canonical shells in dependency order from one
  command.
- Item 8: a fresh `chelis reef build` succeeds end-to-end without
  manual install steps.
- Item 9: a developer can re-create another developer's local
  registry state from the committed lockfile alone.

After all four items land, a fresh dev environment's onboarding is
`git clone <project> && export GITHUB_TOKEN=$(gh auth token) &&
chelis reef build`. That is the experience.

---

## Sequencing

Item 6 must land first; Items 7, 8, 9 each depend on Item 6's
helper. Items 7-9 can land in any order after Item 6 and are
roughly equal-effort (~half a day to one day each).

Item 10 is demand-driven and not part of this design's
implementation scope.

The combined Item 6-9 work is approximately 3-4 days of focused
work. The upper-bound estimate is for time spent on auth-error
catalogs, hash-mismatch error wording, and the corpus of
integration tests against the canonical org.

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
- SHA256 verification logic already shipped (computes
  `archive_sha256` and `shell_sha256` per
  `LockedDependency` at `lib.rs:64-71`).
- Lockfile schema (cargo-style toml serde). Item 9 adds an
  optional field; round-trips via existing serde paths.
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
- **Bit-reproducible artifact comparison across machines.** The
  install path verifies bytes received against bytes pinned in the
  lockfile, but does not verify that two independent rebuilds from
  source would produce bit-identical bytes. That is a separate
  workstream contingent on auditing the C emitter for non-
  determinism.

---

## Cross-references

- `effect_taxonomy_expansion.md` Item 4 (install-time effect
  manifests) is the natural composition point: the install path
  designed here gains a `--print-effects` and `--refuse` flag once
  the effect aggregation is available. The two designs share the
  install boundary.
- `chelis_trust_stack.md` "Limits of the current trust stack"
  references this doc as the planned distribution surface.
- `spec/12-roadmap.md` Phase A (distribution unblock) covers
  Items 6-9 of this doc.
